"""
Module 3: Order Manager — Core execution layer.
Replaces tracker.open_trade() and tracker.close_trade().
Dynamic OTM strike is passed in from on_bar_complete() via pick_best_otm_strike().

v2 — execution-safety revision
  * A fill timeout / API error NEVER blindly returns "failed". The order is
    reconciled against the broker (status, filled qty, net position) first.
  * Entry: partial fills are kept (quantity trimmed to what actually filled);
    a timed-out order is cancelled so it cannot fill later unseen.
  * Exit: refuses to SELL unless the broker shows a LONG net position
    (impossible to open a naked short); sells only the quantity actually held.
  * Exit: the exchange-side SL order is settled (cancelled, or recognised as
    already triggered) BEFORE selling, so SL + manual sell can never double-sell.
  * Exit failure re-arms the exchange-side SL if place_sl_order is enabled.
  * Partial exit fills are tracked; remainder is retried next bar.
  * `busy` flag lets PositionSync's thread stay out of the way mid-order.
  * Returned exit summary carries an explicit "success": True.
"""
import asyncio
import logging
from datetime import datetime
from kiteconnect import KiteConnect

from order_tracker import (
    wait_for_fill, cancel_order, get_order_status,
    get_live_position, OrderFillError, OrderTimeoutError,
)
from risk_gate import RiskGate
from instrument_mapper import InstrumentMapper

log = logging.getLogger("order_manager")

# ── product name → KiteConnect constant ──────────────────────────────
_PRODUCT_MAP = {
    "MIS":  KiteConnect.PRODUCT_MIS,
    "NRML": KiteConnect.PRODUCT_NRML,
    "CNC":  KiteConnect.PRODUCT_CNC,
}

_TERMINAL_STATUSES = ("COMPLETE", "REJECTED", "CANCELLED")
_TICK = 0.05


def _to_tick(x: float) -> float:
    return round(round(x / _TICK) * _TICK, 2)


class OrderManager:
    def __init__(self, kite: KiteConnect, mapper: InstrumentMapper,
                 risk_gate: RiskGate, config: dict):
        self.kite      = kite
        self.mapper    = mapper
        self.risk_gate = risk_gate
        self.cfg       = config
        self.active_trade: dict | None = None
        # True while an entry/exit order is in flight. PositionSync checks this
        # so its background thread never "fixes" state in the middle of an order.
        self.busy = False

    # ══════════════════════════════════════════════════════════════════
    # BROKER LOOKUP HELPERS
    # ══════════════════════════════════════════════════════════════════
    def _net_qty_sync(self, tradingsymbol: str, product: str):
        """Net broker quantity for the symbol/product, or None if lookup failed."""
        product_const = _PRODUCT_MAP.get(product, product)
        try:
            positions = self.kite.positions().get("net", [])
        except Exception as e:
            log.error(f"order_manager: positions lookup failed: {e}")
            return None
        total = 0
        for p in positions:
            if p.get("tradingsymbol") != tradingsymbol:
                continue
            p_prod = p.get("product")
            if p_prod and p_prod != product_const:
                continue
            total += int(p.get("quantity", 0) or 0)
        return total

    async def _net_qty(self, tradingsymbol: str, product: str, loop):
        return await loop.run_in_executor(
            None, self._net_qty_sync, tradingsymbol, product
        )

    def _last_sell_fill_sync(self, tradingsymbol: str):
        """Average price of the most recent COMPLETE SELL order for the symbol."""
        try:
            orders = self.kite.orders()
        except Exception as e:
            log.warning(f"order_manager: orders lookup failed: {e}")
            return None
        best, best_ts = None, None
        for o in orders:
            if (o.get("tradingsymbol") != tradingsymbol
                    or o.get("transaction_type") != "SELL"
                    or o.get("status") != "COMPLETE"):
                continue
            avg = float(o.get("average_price", 0) or 0)
            if avg <= 0:
                continue
            ts = o.get("exchange_timestamp") or o.get("order_timestamp")
            if best is None or (ts is not None and (best_ts is None or ts >= best_ts)):
                best, best_ts = avg, ts
        return best

    async def _order_status(self, order_id: str, loop):
        return await loop.run_in_executor(
            None, lambda: get_order_status(self.kite, order_id)
        )

    async def _settle_order(self, order_id: str, tradingsymbol: str, loop):
        """
        Drive an order to a terminal state and return its status dict
        (None if the broker can't be queried). If it is still working it is
        cancelled, so it cannot fill later without the bot knowing.
        """
        st = await self._order_status(order_id, loop)
        if st is None or st["status"] in _TERMINAL_STATUSES:
            return st
        log.warning(
            f"order_manager: order {order_id} [{tradingsymbol}] still "
            f"'{st['status']}' — cancelling to settle."
        )
        await loop.run_in_executor(
            None,
            lambda: cancel_order(
                self.kite, order_id,
                variety       = KiteConnect.VARIETY_REGULAR,
                tradingsymbol = tradingsymbol,
            ),
        )
        await asyncio.sleep(1.0)
        st2 = await self._order_status(order_id, loop)
        return st2 if st2 is not None else st

    # ══════════════════════════════════════════════════════════════════
    # ENTRY
    # ══════════════════════════════════════════════════════════════════
    async def place_entry(self,
                    signal:             str,
                    snap_price:         float,
                    sl_price:           float,
                    conf:               float,
                    agree:              float,
                    method:             str,
                    bar_dt:             datetime,
                    bar_count:          int,
                    cooldown_remaining: int   = 0,
                    tradingsymbol:      str   = "",
                    inst_token:         int   = 0,
                    lot_size:           int   = 65,
                    lots:               int   = 1,
                    total_cost:         float = 0.0,
                    ) -> dict | None:
        self.busy = True
        try:
            return await self._place_entry_inner(
                signal, snap_price, sl_price, conf, agree, method, bar_dt,
                bar_count, cooldown_remaining, tradingsymbol, inst_token,
                lot_size, lots, total_cost,
            )
        finally:
            self.busy = False

    async def _place_entry_inner(self, signal, snap_price, sl_price, conf, agree,
                                 method, bar_dt, bar_count, cooldown_remaining,
                                 tradingsymbol, inst_token, lot_size, lots,
                                 total_cost):
        dry_run = self.cfg.get("dry_run", True)
        loop    = asyncio.get_running_loop()

        # ── no new entries after cutoff time ────────────────────────
        # Gated on bar_dt (market/bar time), not datetime.now(). Only blocks
        # NEW entries — an existing active_trade still exits via place_exit().
        cutoff_str  = self.cfg.get("no_new_trades_after", "14:00")
        cutoff_time = datetime.strptime(cutoff_str, "%H:%M").time()
        if self.active_trade is None and bar_dt.time() >= cutoff_time:
            log.info(
                f"order_manager: BLOCKED [{signal}] "
                f"new entries disabled after {cutoff_str} "
                f"(bar_dt={bar_dt.time()})"
            )
            return None

        # ── risk gate check ───────────────────────────────────────────
        allowed, reason = self.risk_gate.check_entry(
            signal             = signal,
            entry_price        = snap_price,
            active_trade       = self.active_trade,
            cooldown_remaining = cooldown_remaining,
            lot_size           = lot_size,
            lots               = lots,
            total_cost         = total_cost,
        )
        if not allowed:
            log.info(f"order_manager: BLOCKED [{signal}] {reason}")
            return None

        # ── validate dynamic strike fields ────────────────────────────
        if not tradingsymbol:
            log.error("order_manager: ENTRY BLOCKED — tradingsymbol is empty!")
            return None
        if lot_size <= 0 or lots <= 0:
            log.error(
                f"order_manager: ENTRY BLOCKED — "
                f"invalid lot_size={lot_size} or lots={lots}"
            )
            return None

        quantity = lot_size * lots
        product  = self.cfg.get("product", "MIS")

        log.info(
            f"order_manager: {'[DRY] ' if dry_run else ''}ENTRY "
            f"{signal} {tradingsymbol} "
            f"qty={quantity} (lot={lot_size}×{lots}) "
            f"snap=₹{snap_price:.2f} sl=₹{sl_price:.2f} "
            f"cost=₹{total_cost:.0f} conf={conf:.3f}"
        )

        # ── place order ───────────────────────────────────────────────
        order_id   = None
        fill_price = snap_price   # default for dry run

        if not dry_run:
            order_id, fill_price, filled_qty = await self._place_buy(
                tradingsymbol, quantity, product, loop
            )
            if order_id is None:
                return None

            if not isinstance(fill_price, (int, float)) or fill_price <= 0:
                log.critical(
                    f"order_manager: ⚠️  order {order_id} [{tradingsymbol}] "
                    f"has invalid fill_price={fill_price!r}. A POSITION MAY "
                    f"EXIST — CHECK ZERODHA IMMEDIATELY. Entry aborted."
                )
                return None

            if filled_qty != quantity:
                log.warning(
                    f"order_manager: PARTIAL ENTRY {tradingsymbol} "
                    f"filled {filled_qty}/{quantity} — managing {filled_qty}."
                )
                quantity   = filled_qty
                lots       = max(1, filled_qty // lot_size)
                total_cost = fill_price * filled_qty

        # ── build active trade dict ───────────────────────────────────
        self.active_trade = {
            "signal":           signal,
            "entry":            fill_price,
            "snap":             snap_price,
            "sl":               sl_price,
            "sl_method":        method,
            "method":           method,
            "bar_held":         0,
            "bar_open":         bar_count,
            "bar_dt":           bar_dt,
            "conf":             conf,
            "agree":            agree,
            "tradingsymbol":    tradingsymbol,
            "symbol":           tradingsymbol,
            "inst_token":       inst_token,
            "lot_size":         lot_size,
            "lots":             lots,
            "quantity":         quantity,
            "qty":              quantity,
            "total_cost":       total_cost,
            "order_id":         order_id,
            "sl_order_id":      None,
            "product":          product,
            "exchange":         "NSE",
            "dry_run":          dry_run,
            "fill_time":        datetime.now(),
            "prob_long":        0.5,
            "prob_short":       0.5,
            "realized_partial": 0.0,
        }

        log.info(
            f"order_manager: {'[DRY] ' if dry_run else ''}✅ ENTRY CONFIRMED "
            f"{tradingsymbol} fill=₹{fill_price:.2f} qty={quantity} "
            f"cost=₹{total_cost:.0f}"
        )

        # ── optional Zerodha-side SL order ────────────────────────────
        if not dry_run and self.cfg.get("place_sl_order", False):
            sl_oid = await self._place_slm(
                tradingsymbol, quantity, sl_price, product, loop
            )
            self.active_trade["sl_order_id"] = sl_oid

        self.risk_gate.on_entry_filled()
        return self.active_trade

    # ══════════════════════════════════════════════════════════════════
    # EXIT
    # ══════════════════════════════════════════════════════════════════
    async def place_exit(self, reason: str, exit_price_hint: float,
                         bar_dt: datetime) -> dict | None:
        """
        Returns an exit summary dict (with "success": True) once the position
        is confirmed closed, or None if it is NOT closed (caller retries next
        bar / verifies the broker). Never raises for order problems.
        """
        if self.active_trade is None:
            log.warning("order_manager: place_exit() — no active trade!")
            return None

        self.busy = True
        try:
            return await self._place_exit_inner(reason, exit_price_hint, bar_dt)
        finally:
            self.busy = False

    async def _place_exit_inner(self, reason, exit_price_hint, bar_dt):
        loop = asyncio.get_running_loop()
        at   = self.active_trade

        dry_run       = at.get("dry_run", True)
        tradingsymbol = at.get("tradingsymbol") or at.get("symbol", "")
        quantity      = at.get("quantity") or at.get("qty") or 0
        product       = at.get("product", "MIS")
        entry_price   = at.get("entry", 0.0)

        # ── validate tradingsymbol ────────────────────────────────────
        if not tradingsymbol:
            log.error(
                "order_manager: ⚠️  EXIT BLOCKED — tradingsymbol missing from "
                "active_trade. CHECK ZERODHA POSITIONS IMMEDIATELY."
            )
            self.active_trade = None
            return None

        if not isinstance(entry_price, (int, float)):
            log.error(
                f"order_manager: ⚠️  CORRUPTED active_trade — entry is "
                f"{type(entry_price).__name__} not float. Clearing state. "
                f"CHECK ZERODHA POSITIONS IMMEDIATELY."
            )
            self.active_trade = None
            return None

        log.info(
            f"order_manager: {'[DRY] ' if dry_run else ''}EXIT "
            f"{reason} {tradingsymbol} qty={quantity} "
            f"hint=₹{exit_price_hint:.2f}"
        )

        if dry_run:
            return self._close_out(reason, exit_price_hint, bar_dt)

        # ── 1. settle the exchange-side SL order BEFORE selling ───────
        # If it already triggered, the position is closed — do NOT sell again.
        # If it is still live it must be cancelled, otherwise it could fire
        # after our sell and open a naked short.
        sl_oid = at.get("sl_order_id")
        if sl_oid:
            st = await self._settle_order(sl_oid, tradingsymbol, loop)
            if st is not None and st["status"] == "COMPLETE" and st["average_price"] > 0:
                log.warning(
                    f"order_manager: exchange SL {sl_oid} already TRIGGERED at "
                    f"₹{st['average_price']:.2f} — position closed, NOT selling."
                )
                return self._close_out("SL_EXCHANGE", st["average_price"], bar_dt)
            if st is None or st["status"] not in _TERMINAL_STATUSES:
                log.error(
                    f"order_manager: ⚠️  could not settle SL order {sl_oid} "
                    f"(status={st['status'] if st else 'UNKNOWN'}). Refusing to "
                    f"sell while an SL order may still be live — will retry next bar."
                )
                return None
            at["sl_order_id"] = None   # cancelled / rejected → safe to proceed

        # ── 2. sell (position-verified) ───────────────────────────────
        status, fill, sold_qty = await self._place_sell(
            tradingsymbol, quantity, product, reason, loop
        )

        if status == "FLAT":
            # Broker shows no position. Our own sell may have filled despite a
            # timeout/exception, or it was closed externally. Try to recover the fill.
            recovered = await loop.run_in_executor(
                None, self._last_sell_fill_sync, tradingsymbol
            )
            if recovered:
                log.warning(
                    f"order_manager: broker flat for {tradingsymbol}; using last "
                    f"SELL fill ₹{recovered:.2f} as exit price."
                )
                return self._close_out(reason, recovered, bar_dt)
            log.warning(
                f"order_manager: broker flat for {tradingsymbol} and no SELL fill "
                f"found — position closed externally. Returning None so the caller "
                f"records EXTERNAL_CLOSE."
            )
            return None

        if status == "PARTIAL":
            fill = fill or exit_price_hint
            remaining = quantity - sold_qty
            pnl_unit  = fill - entry_price
            at["realized_partial"] = at.get("realized_partial", 0.0) + pnl_unit * sold_qty
            at["quantity"] = at["qty"] = max(0, remaining)
            lot_size = at.get("lot_size", self.cfg.get("lot_size", 65))
            if remaining > 0:
                at["lots"] = max(1, remaining // lot_size)
            self.risk_gate.on_exit_filled(pnl_unit, lot_size=sold_qty, lots=1)
            log.warning(
                f"order_manager: PARTIAL EXIT {tradingsymbol} sold {sold_qty} "
                f"@₹{fill:.2f}, remaining {remaining}. Will retry next bar."
            )
            if remaining <= 0:
                return self._close_out(reason, fill, bar_dt)
            await self._rearm_sl(loop)
            return None

        if status != "FILLED":   # FAILED
            log.error(
                "order_manager: ⚠️  SELL FAILED — keeping active_trade alive. "
                "Will retry next bar."
            )
            await self._rearm_sl(loop)
            return None

        return self._close_out(reason, fill, bar_dt)

    async def _rearm_sl(self, loop):
        """After a failed/partial exit, restore the exchange-side SL if enabled."""
        at = self.active_trade
        if (at is None or at.get("dry_run", True)
                or not self.cfg.get("place_sl_order", False)
                or at.get("sl_order_id")):
            return
        qty = at.get("quantity") or at.get("qty") or 0
        if qty <= 0:
            return
        at["sl_order_id"] = await self._place_slm(
            at.get("tradingsymbol", ""), qty, at.get("sl", 0.0),
            at.get("product", "MIS"), loop,
        )

    def _close_out(self, reason: str, fill_price: float, bar_dt: datetime) -> dict:
        """Finalise accounting, notify risk gate, clear state, return summary."""
        at         = self.active_trade
        entry      = at.get("entry", 0.0)
        quantity   = at.get("quantity") or at.get("qty") or 0
        dry_run    = at.get("dry_run", True)

        # Bot ALWAYS buys options first (never naked sells), so profit is
        # fill - entry regardless of LONG/SHORT market direction.
        pnl_per_unit = fill_price - entry
        final_pnl    = pnl_per_unit * quantity
        total_pnl    = final_pnl + at.get("realized_partial", 0.0)
        brokerage    = self.cfg.get("brokerage_per_trade", 40.0)
        net_pnl      = total_pnl - brokerage

        log.info(
            f"order_manager: {'[DRY] ' if dry_run else ''}✅ EXIT CONFIRMED "
            f"{reason} {at.get('tradingsymbol', '?')} "
            f"fill=₹{fill_price:.2f} "
            f"pnl=₹{total_pnl:+.2f} "
            f"net=₹{net_pnl:+.2f} "
            f"(brokerage=₹{brokerage:.0f})"
        )

        summary = {
            **at,
            "success":      True,
            "exit_price":   fill_price,
            "exit_reason":  reason,
            "exit_time":    datetime.now(),
            "exit_bar_dt":  bar_dt,
            "pnl_per_unit": pnl_per_unit,
            "total_pnl":    total_pnl,
            "brokerage":    brokerage,
            "net_pnl":      net_pnl,
        }

        # quantity-based accounting (correct even after partial fills)
        self.risk_gate.on_exit_filled(pnl_per_unit, lot_size=quantity, lots=1)
        self.active_trade = None
        return summary

    # ══════════════════════════════════════════════════════════════════
    # _place_buy  →  (order_id | None, fill_price, filled_qty)
    # ══════════════════════════════════════════════════════════════════
    async def _place_buy(self, tradingsymbol: str, quantity: int,
                         product: str, loop: asyncio.AbstractEventLoop):
        market_protection = self.cfg.get("market_protection", 5.0)
        product_const     = _PRODUCT_MAP.get(product, KiteConnect.PRODUCT_MIS)
        fill_timeout      = self.cfg.get("fill_timeout_secs", 15)

        try:
            oid = await loop.run_in_executor(
                None,
                lambda: self.kite.place_order(
                    variety           = KiteConnect.VARIETY_REGULAR,
                    exchange          = KiteConnect.EXCHANGE_NFO,
                    tradingsymbol     = tradingsymbol,
                    transaction_type  = KiteConnect.TRANSACTION_TYPE_BUY,
                    quantity          = quantity,
                    order_type        = KiteConnect.ORDER_TYPE_MARKET,
                    product           = product_const,
                    validity          = KiteConnect.VALIDITY_DAY,
                    market_protection = market_protection,
                    tag               = "v14bot",
                )
            )
        except Exception as e:
            # No order id → nothing was placed.
            log.error(f"order_manager: BUY place_order failed: {e}")
            return None, 0.0, 0

        oid = str(oid)

        try:
            fill = await wait_for_fill(
                self.kite, oid, fill_timeout,
                tradingsymbol = tradingsymbol,
                inst_token    = 0,
            )
            log.info(
                f"order_manager: ✅ BUY FILLED order={oid} "
                f"symbol={tradingsymbol} avg=₹{fill:.2f} qty={quantity}"
            )
            return oid, fill, quantity
        except Exception as e:
            log.error(
                f"order_manager: BUY wait failed ({type(e).__name__}): {e} "
                f"— reconciling order {oid} against broker"
            )

        # ── reconcile: the order WAS placed, so it may have filled ────
        st = await self._settle_order(oid, tradingsymbol, loop)
        if st is not None:
            filled = st["filled_quantity"]
            if filled > 0 and st["average_price"] > 0:
                log.warning(
                    f"order_manager: BUY {oid} reconciled — filled "
                    f"{filled}/{quantity} @₹{st['average_price']:.2f} "
                    f"(status={st['status']})"
                )
                return oid, st["average_price"], filled
            log.error(
                f"order_manager: BUY {oid} reconciled — NOT filled "
                f"(status={st['status']}). No position opened."
            )
            return None, 0.0, 0

        # Order status unavailable → fall back to the live position.
        net = await self._net_qty(tradingsymbol, product, loop)
        if net and net > 0:
            pos = await loop.run_in_executor(
                None, lambda: get_live_position(self.kite, tradingsymbol)
            )
            if pos and pos["average_price"] > 0:
                log.warning(
                    f"order_manager: BUY {oid} status unknown but broker shows "
                    f"qty={net} @₹{pos['average_price']:.2f} — adopting position."
                )
                return oid, pos["average_price"], min(net, quantity)
        log.critical(
            f"order_manager: ⚠️  BUY {oid} [{tradingsymbol}] state UNKNOWN "
            f"(order status and positions both unavailable). CHECK ZERODHA."
        )
        return None, 0.0, 0

    # ══════════════════════════════════════════════════════════════════
    # _place_sell  →  (status, fill_price, sold_qty)
    #   status ∈ FILLED | PARTIAL | FLAT | FAILED
    # ══════════════════════════════════════════════════════════════════
    async def _place_sell(self, tradingsymbol: str, quantity: int,
                          product: str, reason: str,
                          loop: asyncio.AbstractEventLoop):
        market_protection = self.cfg.get("market_protection", 5.0)
        product_const     = _PRODUCT_MAP.get(product, KiteConnect.PRODUCT_MIS)
        fill_timeout      = self.cfg.get("fill_timeout_secs", 15)

        # ── never sell blind, never sell what we don't hold ───────────
        live = await self._net_qty(tradingsymbol, product, loop)
        if live is None:
            log.error(
                f"order_manager: SELL [{reason}] aborted — cannot verify broker "
                f"position for {tradingsymbol}. Will retry."
            )
            return "FAILED", 0.0, 0
        if live == 0:
            log.warning(f"order_manager: SELL [{reason}] — broker already FLAT on {tradingsymbol}.")
            return "FLAT", 0.0, 0
        if live < 0:
            log.critical(
                f"order_manager: ⚠️  broker NET SHORT {live} on {tradingsymbol} — "
                f"refusing to sell more. FIX MANUALLY."
            )
            return "FAILED", 0.0, 0

        sell_qty = min(live, quantity) if quantity > 0 else live
        if live != quantity:
            log.warning(
                f"order_manager: broker qty={live} differs from tracked "
                f"qty={quantity} on {tradingsymbol}; selling {sell_qty}."
            )

        try:
            oid = await loop.run_in_executor(
                None,
                lambda: self.kite.place_order(
                    variety           = KiteConnect.VARIETY_REGULAR,
                    exchange          = KiteConnect.EXCHANGE_NFO,
                    tradingsymbol     = tradingsymbol,
                    transaction_type  = KiteConnect.TRANSACTION_TYPE_SELL,
                    quantity          = sell_qty,
                    order_type        = KiteConnect.ORDER_TYPE_MARKET,
                    product           = product_const,
                    validity          = KiteConnect.VALIDITY_DAY,
                    market_protection = market_protection,
                    tag               = "v14exit",
                )
            )
        except Exception as e:
            log.error(f"order_manager: SELL place_order failed [{reason}]: {e}")
            # A network error can hide an order that actually went through.
            await asyncio.sleep(1.0)
            live2 = await self._net_qty(tradingsymbol, product, loop)
            if live2 is not None and live2 < live:
                log.warning("order_manager: SELL appears to have executed despite the error.")
                return ("FLAT", 0.0, 0) if live2 <= 0 else ("PARTIAL", 0.0, live - live2)
            return "FAILED", 0.0, 0

        oid = str(oid)

        try:
            fill = await wait_for_fill(
                self.kite, oid, fill_timeout,
                tradingsymbol = tradingsymbol,
                inst_token    = 0,
            )
            log.info(
                f"order_manager: ✅ SELL FILLED order={oid} "
                f"symbol={tradingsymbol} avg=₹{fill:.2f} qty={sell_qty}"
            )
            return "FILLED", fill, sell_qty
        except (OrderFillError, OrderTimeoutError, Exception) as e:
            log.error(
                f"order_manager: SELL wait failed ({type(e).__name__}) [{reason}]: {e} "
                f"— reconciling order {oid} against broker"
            )

        # ── reconcile instead of blindly retrying ─────────────────────
        st = await self._settle_order(oid, tradingsymbol, loop)
        if st is not None:
            filled = st["filled_quantity"]
            avg    = st["average_price"]
            if filled >= sell_qty and avg > 0:
                return "FILLED", avg, filled
            if filled > 0:
                return "PARTIAL", avg, filled
            return "FAILED", 0.0, 0

        live2 = await self._net_qty(tradingsymbol, product, loop)
        if live2 is not None and live2 <= 0:
            return "FLAT", 0.0, 0
        return "FAILED", 0.0, 0

    # ══════════════════════════════════════════════════════════════════
    # _place_slm  (SL-limit order, exchange-side protection)
    # ══════════════════════════════════════════════════════════════════
    async def _place_slm(self, tradingsymbol: str, quantity: int,
                         sl_price: float, product: str,
                         loop: asyncio.AbstractEventLoop):
        min_price = self.cfg.get("sl_min_price_for_order", 1.0)
        if sl_price < min_price:
            log.warning(
                f"order_manager: exchange SL skipped — sl_price ₹{sl_price:.2f} "
                f"< ₹{min_price:.2f} (premium too low for a meaningful stop)."
            )
            return None

        trigger       = _to_tick(sl_price)
        limit_price   = _to_tick(max(_TICK, sl_price - self.cfg.get("sl_limit_buffer", 1.0)))
        product_const = _PRODUCT_MAP.get(product, KiteConnect.PRODUCT_MIS)

        try:
            oid = await loop.run_in_executor(
                None,
                lambda: self.kite.place_order(
                    variety          = KiteConnect.VARIETY_REGULAR,
                    exchange         = KiteConnect.EXCHANGE_NFO,
                    tradingsymbol    = tradingsymbol,
                    transaction_type = KiteConnect.TRANSACTION_TYPE_SELL,
                    quantity         = quantity,
                    order_type       = KiteConnect.ORDER_TYPE_SL,
                    trigger_price    = trigger,
                    price            = limit_price,
                    product          = product_const,
                    validity         = KiteConnect.VALIDITY_DAY,
                    tag              = "v14sl",
                )
            )
            log.info(
                f"order_manager: SL-L placed "
                f"{tradingsymbol} trigger=₹{trigger:.2f} limit=₹{limit_price:.2f}"
            )
            return str(oid)
        except Exception as e:
            log.warning(f"order_manager: SL-L failed: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════
    # STATE ACCESSORS
    # ══════════════════════════════════════════════════════════════════
    def has_active_trade(self) -> bool:
        return self.active_trade is not None

    def get_active_trade(self) -> dict | None:
        return self.active_trade

    def get_entry_price(self) -> float:
        return self.active_trade["entry"] if self.active_trade else 0.0