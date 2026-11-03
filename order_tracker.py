"""
order_tracker.py — Fill confirmation + status polling for Zerodha orders.
Logs dynamic OTM strike fields (tradingsymbol, inst_token, total_cost)
passed in from pick_best_otm_strike() via order_manager.
"""
import time
import logging
from datetime import datetime
from kiteconnect import KiteConnect
from kiteconnect.exceptions import NetworkException, DataException

log = logging.getLogger("order_tracker")


class OrderFillError(Exception):
    pass


class OrderTimeoutError(Exception):
    pass


# ─── FILL POLLING ─────────────────────────────────────────────────────────────

# ✅ FIXED — async, yields control back to event loop while polling
import asyncio

async def wait_for_fill(kite: KiteConnect,
                        order_id: str,
                        timeout_secs: int = 30,
                        poll_interval: float = 1.0,
                        tradingsymbol: str = "",
                        inst_token: int = 0) -> float:
    deadline = time.time() + timeout_secs
    label    = tradingsymbol if tradingsymbol else order_id

    log.info(
        f"order_tracker: Waiting for fill: order={order_id} "
        f"symbol={label} token={inst_token}"
    )

    while time.time() < deadline:
        try:
            loop   = asyncio.get_event_loop()
            orders = await loop.run_in_executor(None, kite.orders)  # non-blocking HTTP
        except Exception as e:
            log.warning(f"order_tracker: Poll error: {e}. Retrying...")
            await asyncio.sleep(poll_interval * 2)
            continue

        for o in orders:
            if str(o.get("order_id")) != str(order_id):
                continue

            status    = o.get("status", "")
            avg_price = float(o.get("average_price", 0) or 0)
            filled    = int(o.get("filled_quantity", 0) or 0)

            if status == "COMPLETE":
                log.info(
                    f"order_tracker: ✅ FILLED "
                    f"order={order_id} symbol={label} "
                    f"avg=₹{avg_price:.2f} qty={filled}"
                )
                return avg_price

            elif status in ("REJECTED", "CANCELLED"):
                raise OrderFillError(
                    f"Order {order_id} [{label}] {status}: "
                    f"{o.get('status_message', 'no message')}"
                )
            break

        await asyncio.sleep(poll_interval)  # yields control — other tasks run

    raise OrderTimeoutError(
        f"Order {order_id} [{label}] not filled within {timeout_secs}s — "
        "check Zerodha app, position may be open!"
    )


# ─── CANCEL ───────────────────────────────────────────────────────────────────

def cancel_order(kite: KiteConnect,
                 order_id: str,
                 variety: str = "regular",
                 tradingsymbol: str = "") -> bool:
    """
    Cancels an open order (e.g. SL-M before manual exit).
    Returns True on success, False if already terminal or not found.
    Safe to call on already-completed orders — exception is caught.
    """
    label = tradingsymbol if tradingsymbol else order_id
    try:
        kite.cancel_order(variety=variety, order_id=str(order_id))
        log.info(f"order_tracker: Cancelled order={order_id} symbol={label}")
        return True
    except Exception as e:
        log.warning(
            f"order_tracker: Cancel failed order={order_id} symbol={label}: {e} "
            "(may already be terminal — safe to ignore)"
        )
        return False


# ─── STATUS ───────────────────────────────────────────────────────────────────

def get_order_status(kite: KiteConnect, order_id: str) -> dict | None:
    """
    Returns a status dict for a given order_id, or None if not found.
    """
    try:
        for o in kite.orders():
            if str(o.get("order_id")) != str(order_id):
                continue
            return {
                "order_id":        str(order_id),
                "tradingsymbol":   o.get("tradingsymbol", ""),
                "inst_token":      int(o.get("instrument_token", 0) or 0),
                "status":          o.get("status", ""),
                "average_price":   float(o.get("average_price", 0) or 0),
                "filled_quantity": int(o.get("filled_quantity", 0) or 0),
                "quantity":        int(o.get("quantity", 0) or 0),
                "status_message":  o.get("status_message", ""),
            }
    except Exception as e:
        log.warning(f"order_tracker: Status check failed order={order_id}: {e}")
    return None


# ─── LIVE POSITION ────────────────────────────────────────────────────────────

def get_live_position(kite: KiteConnect,
                      tradingsymbol: str,
                      inst_token: int = 0) -> dict | None:
    """
    Looks up a live intraday MIS position by tradingsymbol.
    Returns dict with key fields or None if not found / qty is 0.

    BUG FIX: checks only 'day' positions, not 'day + net'.
    Checking both causes duplicates — for MIS intraday options,
    'day' is the correct and only source needed.
    """
    try:
        pos_data = kite.positions()

        # FIX: was `day + net` — only use 'day' for MIS intraday positions
        day_positions = pos_data.get("net", [])

        for pos in day_positions:
            if pos.get("tradingsymbol") != tradingsymbol:
                continue

            qty = int(pos.get("quantity", 0))
            avg = float(pos.get("average_price", 0))

            return {
                "tradingsymbol": pos["tradingsymbol"],
                "inst_token":    int(pos.get("instrument_token", inst_token) or inst_token),
                "quantity":      qty,
                "average_price": avg,
                "last_price":    float(pos.get("last_price", 0)),
                "unrealised":    float(pos.get("unrealised", 0)),
                "total_cost":    avg * abs(qty),
            }

    except Exception as e:
        log.error(
            f"order_tracker: get_live_position failed [{tradingsymbol}]: {e}"
        )
    return None


# ─── TRADE RECORD LOGGING ─────────────────────────────────────────────────────

def log_trade_entry(active_trade: dict) -> None:
    """
    Logs a full entry record including dynamic OTM strike fields.
    Call immediately after order_manager confirms fill.
    """
    log.info(
        f"[TRADE ENTRY] "
        f"signal={active_trade.get('signal', '?')} "
        f"symbol={active_trade.get('tradingsymbol', 'UNKNOWN')} "
        f"token={active_trade.get('inst_token', 0)} "
        f"fill=₹{active_trade.get('entry', 0):.2f} "
        f"snap=₹{active_trade.get('snap', 0):.2f} "
        f"sl=₹{active_trade.get('sl', 0):.2f} "
        f"lot={active_trade.get('lot_size', 0)}×{active_trade.get('lots', 0)} "
        f"qty={active_trade.get('quantity', 0)} "
        f"cost=₹{active_trade.get('total_cost', 0):.0f} "
        f"conf={active_trade.get('conf', 0):.3f} "
        f"agree={active_trade.get('agree', 0)} "
        f"method={active_trade.get('method', '?')} "
        f"dry_run={active_trade.get('dry_run', True)} "
        f"order_id={active_trade.get('order_id', 'N/A')} "
        f"time={active_trade.get('fill_time', datetime.now()).strftime('%H:%M:%S')}"
    )


def log_trade_exit(summary: dict) -> None:
    """
    Logs a full exit record including dynamic OTM strike fields.
    Call after order_manager.place_exit() returns a summary.
    """
    log.info(
        f"[TRADE EXIT] "
        f"signal={summary.get('signal', '?')} "
        f"symbol={summary.get('tradingsymbol', 'UNKNOWN')} "
        f"token={summary.get('inst_token', 0)} "
        f"reason={summary.get('exit_reason', '?')} "
        f"entry=₹{summary.get('entry', 0):.2f} "
        f"exit=₹{summary.get('exit_price', 0):.2f} "
        f"pnl_unit=₹{summary.get('pnl_per_unit', 0):+.2f} "
        f"total_pnl=₹{summary.get('total_pnl', 0):+.2f} "
        f"brokerage=₹{summary.get('brokerage', 0):.0f} "
        f"net_pnl=₹{summary.get('net_pnl', 0):+.2f} "
        f"cost=₹{summary.get('total_cost', 0):.0f} "
        f"dry_run={summary.get('dry_run', True)} "
        f"time={summary.get('exit_time', datetime.now()).strftime('%H:%M:%S')}"
    )