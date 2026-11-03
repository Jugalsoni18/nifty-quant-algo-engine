"""
position_sync.py — Background reconciliation every 5 mins.
Handles crashes, ghost trades, orphan positions, auto-squareoff.

v2 — safety revision
  * Stays out of the way while OrderManager has an order in flight (`busy`).
  * Ignores dry-run trades (no real position exists to reconcile against).
  * Ignores a trade for GHOST_GRACE_SECS after its fill (Kite's positions
    endpoint can lag a few seconds behind a fill).
  * A positions-API failure SKIPS the cycle; it is never read as "flat".
  * A ghost must be CONFIRMED by a second lookup before state is cleared, and
    the trade must still be the same object when it is cleared.
  * Ghost handler recovers the real exit fill (SL-order fill, or latest SELL
    fill) and syncs realised PnL into RiskGate, so the daily-loss limit stays
    honest after exchange-SL / auto-squareoff / manual closes.
  * Quantity corrections also fix `lots`.
  * First reconcile happens after one interval (not at t=0, which raced with
    restore() and produced a false ORPHAN warning on every start).
  * restore() uses the broker's real product, adopts a live exchange-side SL
    order (and its trigger price) if one exists, and otherwise marks the SL
    for recalculation (live bot replaces the 1.5% placeholder on the first bar).
"""
import time
import logging
import threading
from datetime import datetime, time as dtime
from kiteconnect import KiteConnect
from order_tracker import get_order_status
from order_manager import OrderManager
from instrument_mapper import InstrumentMapper


log = logging.getLogger("position_sync")

GHOST_GRACE_SECS   = 90      # skip trades filled more recently than this
GHOST_CONFIRM_WAIT = 3.0     # seconds between the two ghost lookups
_OPEN_SL_STATUSES  = ("TRIGGER PENDING", "OPEN")


def _is_nifty50(sym: str) -> bool:
    return (
        "NIFTY" in sym
        and "BANK"  not in sym
        and "MIDCP" not in sym
        and "FIN"   not in sym
    )


class PositionSync:
    def __init__(self, kite: KiteConnect,
                 order_manager: OrderManager,
                 mapper: InstrumentMapper,
                 config: dict,
                 interval_secs: int = 300):
        self.kite          = kite
        self.order_manager = order_manager
        self.mapper        = mapper
        self.cfg           = config
        self.interval      = interval_secs
        self._stop_evt     = threading.Event()
        self._thread       = None
        self._sync_count   = 0

    @property
    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_evt.is_set()

    def start(self):
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="PositionSync"
        )
        self._thread.start()
        log.info(f"position_sync: Started (every {self.interval}s, first run after one interval)")

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=10)
        log.info("position_sync: Stopped.")

    def _loop(self):
        # wait FIRST so we never race with restore() at startup
        while not self._stop_evt.wait(self.interval):
            try:
                self.reconcile()
            except Exception as e:
                log.error(f"position_sync: Reconcile error: {e}", exc_info=True)

    # ─── BROKER HELPERS ───────────────────────────────────────────────

    def _fetch_net_positions(self):
        """List of net positions, or None if the API call failed."""
        try:
            return self.kite.positions().get("net", [])
        except Exception as e:
            log.warning(f"position_sync: positions fetch failed: {e}")
            return None

    @staticmethod
    def _find_position(positions, sym: str):
        for pos in positions:
            if pos.get("tradingsymbol") == sym:
                return pos
        return None

    def _find_exit_fill(self, sym: str):
        """Average price of the most recent COMPLETE SELL order for `sym`."""
        try:
            orders = self.kite.orders()
        except Exception as e:
            log.warning(f"position_sync: orders fetch failed: {e}")
            return None
        best, best_ts = None, None
        for o in orders:
            if (o.get("tradingsymbol") != sym
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

    def _find_open_sl_order(self, sym: str):
        """(order_id, trigger_price) of a live SELL SL order on `sym`, else (None, None)."""
        try:
            for o in self.kite.orders():
                if (o.get("tradingsymbol") == sym
                        and o.get("transaction_type") == "SELL"
                        and o.get("order_type") in ("SL", "SL-M")
                        and o.get("status") in _OPEN_SL_STATUSES):
                    return str(o.get("order_id")), float(o.get("trigger_price", 0) or 0)
        except Exception as e:
            log.warning(f"position_sync: SL-order scan failed: {e}")
        return None, None

    # ─── MAIN RECONCILE ───────────────────────────────────────────────

    def reconcile(self):
        self._sync_count += 1

        if self.order_manager.busy:
            log.debug("position_sync: order in flight — skipping this cycle.")
            return

        active = self.order_manager.get_active_trade()

        if active is None:
            self._check_orphans()
            return

        # dry-run trades have no broker position — nothing to reconcile
        if active.get("dry_run", False):
            return

        # Kite positions can lag a fill by a few seconds
        fill_time = active.get("fill_time")
        if isinstance(fill_time, datetime):
            age = (datetime.now() - fill_time).total_seconds()
            if age < GHOST_GRACE_SECS:
                log.debug(f"position_sync: trade is only {age:.0f}s old — skipping.")
                return

        sym = active.get("tradingsymbol") or active.get("symbol", "")

        positions = self._fetch_net_positions()
        if positions is None:
            return                      # API failure ≠ flat

        pos = self._find_position(positions, sym)
        qty = int(pos.get("quantity", 0)) if pos else 0

        if qty == 0:
            # ── confirm before declaring a ghost ──────────────────────
            log.warning(
                f"position_sync: ⚠️  possible GHOST — bot has [{sym}] but "
                f"Zerodha shows qty=0. Re-checking in {GHOST_CONFIRM_WAIT:.0f}s..."
            )
            time.sleep(GHOST_CONFIRM_WAIT)

            if self.order_manager.busy:
                return
            if self.order_manager.get_active_trade() is not active:
                return                  # bot changed state meanwhile

            positions = self._fetch_net_positions()
            if positions is None:
                return
            pos = self._find_position(positions, sym)
            qty = int(pos.get("quantity", 0)) if pos else 0
            if qty == 0:
                self._handle_ghost(active)
                return

        # ── position exists: verify quantity ──────────────────────────
        exp_qty = active.get("quantity") or active.get("qty", 0)
        if qty != exp_qty:
            lot_size = active.get("lot_size") or self.cfg.get("lot_size", 65)
            log.warning(
                f"position_sync: Qty mismatch for {sym}: "
                f"expected={exp_qty} actual={qty}. Correcting."
            )
            active["quantity"] = qty
            active["qty"]      = qty
            active["lots"]     = max(1, round(qty / lot_size)) if lot_size else active.get("lots", 1)

        log.debug(
            f"position_sync: ✅ {sym} qty={qty} "
            f"ltp=₹{float(pos.get('last_price', 0)):.2f} "
            f"unrealised=₹{float(pos.get('unrealised', 0)):.2f}"
        )

    # ─── RESTORE ON RESTART ───────────────────────────────────────────

    def restore(self) -> bool:
        """
        Called ONCE on bot startup before the feed connects.
        If Zerodha has an open NIFTY50 position but the bot has no active_trade
        (crash/restart), rebuilds active_trade from broker data.

        SL: if a live exchange-side SL order exists it is adopted (id + trigger).
        Otherwise a 1.5% placeholder is used and sl_needs_recalc=True is set —
        the live bot replaces it with its real static SL on the first bar.
        """
        if self.order_manager.get_active_trade() is not None:
            return False

        try:
            positions = self.kite.positions().get("net", [])
        except Exception as e:
            log.error(f"position_sync: restore() — could not fetch positions: {e}")
            return False

        cfg_product = self.cfg.get("product", "MIS")
        dry_run     = self.cfg.get("dry_run", False)
        lot_size    = self.cfg.get("lot_size", 65)

        candidates = [
            p for p in positions
            if int(p.get("quantity", 0)) != 0
            and p.get("exchange", "") == "NFO"
            and _is_nifty50(p.get("tradingsymbol", ""))
        ]
        if not candidates:
            return False
        if len(candidates) > 1:
            log.error(
                f"position_sync: {len(candidates)} open NIFTY option positions "
                f"found ({[c.get('tradingsymbol') for c in candidates]}). The bot "
                f"manages ONE trade — restoring the first; HANDLE THE REST MANUALLY."
            )

        pos = candidates[0]
        qty = int(pos.get("quantity", 0))
        sym = pos.get("tradingsymbol", "")

        if qty < 0:
            log.error(
                f"position_sync: {sym} is NET SHORT {qty}. The bot only trades long "
                f"options — NOT restoring. CLOSE/FIX MANUALLY."
            )
            return False

        avg_price  = float(pos.get("average_price", 0))
        last_price = float(pos.get("last_price",   0))
        product    = pos.get("product") or cfg_product

        signal = "LONG" if sym.endswith("CE") else "SHORT"
        lots   = max(1, round(qty / lot_size))

        sl_oid, sl_trigger = self._find_open_sl_order(sym)
        if sl_oid and sl_trigger and sl_trigger > 0:
            sl_price, sl_method, needs_recalc = sl_trigger, "RESTORED_FROM_EXCHANGE_SL", False
        else:
            sl_oid = None
            sl_price = round(avg_price * (1 - 0.015), 2)
            sl_method, needs_recalc = "RESTORED_FIXED_1.5PCT", True

        restored_trade = {
            # ── identity ─────────────────────────────────────────────
            "signal"         : signal,
            "tradingsymbol"  : sym,
            "symbol"         : sym,
            "inst_token"     : pos.get("instrument_token", 0),
            "exchange"       : "NSE",

            # ── sizing ────────────────────────────────────────────────
            "quantity"       : qty,
            "qty"            : qty,
            "lot_size"       : lot_size,
            "lots"           : lots,
            "total_cost"     : avg_price * qty,

            # product must match the original entry product
            "product"        : product,

            # ── pricing ───────────────────────────────────────────────
            "entry"          : avg_price,
            "snap"           : avg_price,
            "sl"             : sl_price,
            "sl_method"      : sl_method,
            "sl_needs_recalc": needs_recalc,

            # ── state ─────────────────────────────────────────────────
            "bar_held"       : 0,
            "bar_open"       : 0,
            "prob_long"      : 0.5,
            "prob_short"     : 0.5,
            "conf"           : 0.5,
            "agree"          : 0.5,
            "realized_partial": 0.0,

            # ── order refs ────────────────────────────────────────────
            "order_id"       : None,
            "sl_order_id"    : sl_oid,

            # ── flags ─────────────────────────────────────────────────
            "dry_run"        : dry_run,
            "fill_time"      : datetime.now(),
        }

        self.order_manager.active_trade = restored_trade

        if dry_run:
            log.warning(
                "position_sync: dry_run=True but a REAL broker position exists — "
                "the restored trade will NOT send real sell orders. Set dry_run=False "
                "or close it manually."
            )
        log.warning(
            f"position_sync: ⚠️  RESTORED position from Zerodha — "
            f"symbol={sym} signal={signal} qty={qty} product={product} "
            f"avg=₹{avg_price:.2f} ltp=₹{last_price:.2f} "
            f"sl=₹{sl_price:.2f} ({sl_method}). "
            + ("Exchange SL order adopted." if sl_oid
               else "No exchange-side SL exists — bot-side SL only.")
        )
        return True

    # ─── GHOST TRADE HANDLER ──────────────────────────────────────────

    def _clear(self, active) -> None:
        """Clear state only if the trade is still the same object."""
        if self.order_manager.get_active_trade() is active:
            self.order_manager.active_trade = None
        else:
            log.warning("position_sync: active_trade changed during ghost handling — not clearing.")

    def _sync_pnl(self, active: dict, fill: float, label: str) -> None:
        entry = active.get("entry", fill)
        qty   = active.get("quantity") or active.get("qty") or 0
        pnl_per_unit = fill - entry
        if qty > 0:
            self.order_manager.risk_gate.on_exit_filled(
                pnl_per_unit=pnl_per_unit, lot_size=qty, lots=1,
            )
        log.info(
            f"position_sync: {label} PnL synced — entry=₹{entry:.2f} "
            f"exit=₹{fill:.2f} pnl_unit=₹{pnl_per_unit:+.2f} qty={qty}"
        )

    def _handle_ghost(self, active: dict):
        """
        Bot has active_trade but Zerodha shows flat (confirmed twice).
        Never calls place_exit() (that would create a naked short). Syncs PnL
        into RiskGate where the real exit fill can be found, then clears state.
        """
        sym = active.get("tradingsymbol", "?")

        # 1. exchange-side SL order triggered?
        sl_oid = active.get("sl_order_id")
        if sl_oid:
            sl_status = get_order_status(self.kite, sl_oid)
            if sl_status and sl_status["status"] == "COMPLETE":
                fill = sl_status["average_price"]
                log.warning(
                    f"position_sync: exchange SL triggered at ₹{fill:.2f}. "
                    "Syncing PnL — NOT placing a sell order."
                )
                self._sync_pnl(active, fill, "SL")
                self._clear(active)
                return

        # 2. entry order was rejected / cancelled → nothing ever opened
        entry_oid = active.get("order_id")
        if entry_oid:
            entry_status = get_order_status(self.kite, entry_oid)
            if entry_status and entry_status["status"] in ("REJECTED", "CANCELLED") \
                    and entry_status.get("filled_quantity", 0) == 0:
                log.warning(
                    f"position_sync: Entry order {entry_oid} was "
                    f"{entry_status['status']}. Clearing ghost state (no PnL)."
                )
                self._clear(active)
                return

        # 3. look for the real exit fill (auto-squareoff / manual close / etc.)
        fill = self._find_exit_fill(sym)
        in_squareoff_window = datetime.now().time() >= dtime(15, 20)

        if fill:
            cause = "Zerodha 15:20 auto-squareoff" if in_squareoff_window else "external close"
            log.warning(
                f"position_sync: {sym} closed outside the bot ({cause}); "
                f"last SELL fill ₹{fill:.2f}."
            )
            self._sync_pnl(active, fill, cause)
            self._clear(active)
            return

        if in_squareoff_window:
            log.warning(
                f"position_sync: Zerodha 15:20 auto-squareoff assumed for {sym} "
                f"(no SELL fill found — PnL NOT synced). Clearing state."
            )
            self._clear(active)
            return

        log.error(
            f"position_sync: 🚨 UNKNOWN GHOST — {sym} vanished from Zerodha with "
            f"no known cause and no SELL fill found. Clearing state — "
            f"CHECK ZERODHA TRADEBOOK IMMEDIATELY!"
        )
        self._clear(active)

    # ─── ORPHAN POSITION CHECKER ──────────────────────────────────────

    def _check_orphans(self):
        positions = self._fetch_net_positions()
        if positions is None:
            return
        for pos in positions:
            qty  = int(pos.get("quantity", 0))
            sym  = pos.get("tradingsymbol", "")
            exch = pos.get("exchange", "")
            if qty == 0 or exch != "NFO" or not _is_nifty50(sym):
                continue
            log.warning(
                f"position_sync: 🚨 ORPHAN POSITION — "
                f"{sym} qty={qty} "
                f"avg=₹{float(pos.get('average_price', 0)):.2f} "
                f"ltp=₹{float(pos.get('last_price', 0)):.2f}. "
                "Bot has no active_trade. Close manually or restart to restore()."
            )

    # ─── STATE ────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "running":    self._running,
            "sync_count": self._sync_count,
        }