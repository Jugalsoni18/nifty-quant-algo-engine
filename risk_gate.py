"""
risk_gate.py — Pre-trade guardrails.
Uses dynamic lot_size / lots / total_cost from pick_best_otm_strike()
via order_manager, instead of hardcoded config values.
"""
import logging
from datetime import time, datetime
from kiteconnect import KiteConnect

log = logging.getLogger("risk_gate")


class RiskGate:
    def __init__(self, kite: KiteConnect, config: dict):
        self.kite                = kite
        self.cfg                 = config
        self._trades_today       = 0
        self._realized_pnl_today = 0.0
        self._market_open        = config.get("market_open",  time(9, 20))
        self._market_close       = config.get("market_close", time(15, 10))


    # ─── ENTRY CHECK ──────────────────────────────────────────────────────────

    def check_entry(self,
                    signal:             str,
                    entry_price:        float,
                    active_trade:       dict | None,
                    cooldown_remaining: int   = 0,
                    lot_size:           int   = 0,
                    lots:               int   = 1,
                    total_cost:         float = 0.0,
                    ) -> tuple[bool, str]:
        """
        Runs all pre-trade safety checks in priority order.
        lot_size / lots / total_cost come from pick_best_otm_strike()
        so checks use the ACTUAL option premium cost, not config defaults.
        """

        # ── 1. Market hours ───────────────────────────────────────────────────
        now_t = datetime.now().time()
        if not (self._market_open <= now_t <= self._market_close):
            return False, f"OUTSIDE_HOURS {now_t.strftime('%H:%M')}"

        # ── 2. Already in a trade ─────────────────────────────────────────────
        if active_trade is not None:
            return False, f"ALREADY_IN_TRADE [{active_trade.get('signal')}]"

        # ── 3. Cooldown ───────────────────────────────────────────────────────
        if cooldown_remaining > 0:
            return False, f"COOLDOWN {cooldown_remaining} bars remaining"

        # ── 4. Max trades per day ─────────────────────────────────────────────
        max_trades = self.cfg.get("max_trades_per_day", 5)
        if self._trades_today >= max_trades:
            return False, f"MAX_TRADES {self._trades_today}/{max_trades}"

        # ── 5. Daily loss limit ───────────────────────────────────────────────
        loss_limit = self.cfg.get("daily_loss_limit", -10000.0)
        if self._realized_pnl_today <= loss_limit:
            return False, f"DAILY_LOSS_LIMIT ₹{self._realized_pnl_today:.0f}"

        # ── 6. Margin / capital check (uses dynamic values) ───────────────────
        effective_lot_size = lot_size   if lot_size    > 0   else self.cfg.get("lot_size", 65)
        effective_lots     = lots       if lots        > 0   else self.cfg.get("max_lots", 1)
        effective_cost     = total_cost if total_cost  > 0.0 else (
            entry_price * effective_lot_size * effective_lots
        )

        ok, reason = self._check_margin(
            entry_price = entry_price,
            lot_size    = effective_lot_size,
            lots        = effective_lots,
            total_cost  = effective_cost,
        )
        if not ok:
            return False, reason

        return True, ""


    # ─── MARGIN CHECK ─────────────────────────────────────────────────────────

    def _check_margin(self,
                      entry_price: float,
                      lot_size:    int   = 0,
                      lots:        int   = 1,
                      total_cost:  float = 0.0,
                      ) -> tuple[bool, str]:
        """
        Validates:
          1. total_cost ≤ max_capital_per_trade config cap
          2. Available Zerodha margin covers total_cost + 10% buffer

        BUG FIX: was margins.get("net", 0) which returns inflated net balance
        including unrealised positions. Now uses live_balance → cash → net
        fallback chain, matching get_available_capital() in instrument_mapper.
        """
        if lot_size <= 0:
            lot_size = self.cfg.get("lot_size", 65)
        if lots <= 0:
            lots = self.cfg.get("max_lots", 1)

        required = total_cost if total_cost > 0.0 else (entry_price * lot_size * lots)
        max_cap  = self.cfg.get("max_capital_per_trade", 50000.0)

        # ── cap check ─────────────────────────────────────────────────────────
        if required > max_cap:
            return False, (
                f"PREMIUM_TOO_HIGH ₹{required:,.0f} > "
                f"cap ₹{max_cap:,.0f} "
                f"(lot={lot_size}×{lots})"
            )

        # ── live margin check ─────────────────────────────────────────────────
        try:
            margins = self.kite.margins(segment="equity")

            # FIX: 'net' alone is inflated by unrealised P&L — use the
            # live_balance → cash → net fallback chain (same as get_available_capital)
            available = float(
                margins.get("available", {}).get("live_balance", 0)
                or margins.get("available", {}).get("cash", 0)
                or margins.get("net", 0)
            )

            if available < required * 1.10:
                return False, (
                    f"INSUFFICIENT_MARGIN "
                    f"available=₹{available:,.0f} "
                    f"need=₹{required * 1.10:,.0f} "
                    f"(cost=₹{required:,.0f} + 10% buffer)"
                )

        except Exception as e:
            log.warning(f"risk_gate: Margin API failed: {e} — allowing trade.")

        return True, ""


    # ─── POST-FILL CALLBACKS ──────────────────────────────────────────────────

    def on_entry_filled(self) -> None:
        self._trades_today += 1
        log.info(f"risk_gate: Trade #{self._trades_today} today.")

    def on_exit_filled(self,
                       pnl_per_unit: float,
                       lot_size:     int = 0,
                       lots:         int = 0,
                       ) -> None:
        """
        Updates realized PnL using actual lot_size × lots from the trade.
        Falls back to config values only if caller doesn't pass them.

        NOTE: order_manager.place_exit() MUST pass lot_size and lots
        explicitly — otherwise falls back to config which may differ from
        the actual trade, causing silent PnL tracking errors.
        """
        if lot_size <= 0:
            lot_size = self.cfg.get("lot_size", 65)
        if lots <= 0:
            lots = self.cfg.get("max_lots", 1)

        realized = pnl_per_unit * lot_size * lots
        self._realized_pnl_today += realized

        log.info(
            f"risk_gate: Exit PnL ₹{realized:+.2f} "
            f"(₹{pnl_per_unit:+.2f}/unit × {lot_size}×{lots}) | "
            f"Day total ₹{self._realized_pnl_today:+.2f}"
        )

        if self._realized_pnl_today <= self.cfg.get("daily_loss_limit", -10000.0):
            log.warning(
                f"risk_gate: ⚠️  DAILY LOSS LIMIT HIT — "
                f"₹{self._realized_pnl_today:.0f} ≤ "
                f"₹{self.cfg.get('daily_loss_limit', -10000.0):.0f} "
                "— no more trades today."
            )


    # ─── STATE ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "trades_today":       self._trades_today,
            "realized_pnl_today": self._realized_pnl_today,
            "trading_allowed": (
                self._realized_pnl_today > self.cfg.get("daily_loss_limit", -10000.0)
                and self._trades_today   < self.cfg.get("max_trades_per_day", 5)
            ),
        }

    def reset_day(self) -> None:
        self._trades_today       = 0
        self._realized_pnl_today = 0.0
        log.info("risk_gate: Day counters reset.")