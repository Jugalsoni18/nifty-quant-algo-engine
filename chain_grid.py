"""
chain_grid.py  --  DYNAMIC-ATM version
======================================
Replaces the week-pinned grid. The model was trained on the rolling series:
every bar, ATM = round(spot/50)*50 and the nifty_WEEK_ATM{plus,minus}N_* columns
are the strikes ATM+N*50. This class reproduces that live.

How
---
* Subscribes a WIDE band of strikes (center +/- half_width strikes) once.
* Every leg keeps its own per-bar OHLC / volume / IV / OI history.
* At each bar close, ATM is recomputed from the bar's closing spot and the
  model's raw columns are read from the legs at ATM+offset*step.
  -> the 9 "ATM" columns are the true dynamic ATM leg (not a fixed strike).
* `atm` (dict CALL/PUT -> bar) is what the bot pushes into its buffer.
* leg_series(strike, side) gives one strike's own history; used for the
  traded option's ATR / supertrend / IV z-score / OI change, which must be a
  single-strike series (an ATM series that hops strikes would fake OI/IV moves).
* maybe_recenter() shifts the subscribed band when ATM drifts; call it OUTSIDE
  the asyncio loop (see patches) because feed.subscribe_symbols is sync.
"""

import re
import logging
from collections import deque
import requests

log = logging.getLogger("v14_l5")

_COL_RE = re.compile(
    r"^nifty_WEEK_ATM(?:(plus|minus)(\d+))?_(CALL|PUT)_1m__(open|volume|iv|oi|spot)$"
)
HIST_LEN = 240


def finite_positive(value):
    """True only for a finite positive numeric value."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return value > 0 and value < float("inf") and value == value


def np_is_finite_positive(value):
    # Kept as a descriptive alias in flush_bar; this module does not require
    # NumPy just to validate a tick price.
    return finite_positive(value)


def parse_chain_columns(feature_names):
    """-> list of (column_name, offset, side, field)"""
    out = []
    for f in feature_names:
        m = _COL_RE.match(f)
        if not m:
            continue
        sign, n, side, field = m.groups()
        off = 0 if sign is None else (int(n) if sign == "plus" else -int(n))
        out.append((f, off, side, field))
    return out


def _tag(off):
    return "ATM" if off == 0 else (f"ATMplus{off}" if off > 0 else f"ATMminus{-off}")


def _new_leg(sid, strike, side):
    return dict(
        sid=sid, strike=strike, side=side,
        last_px=None, last_oi=0.0, last_iv=None,
        cum=None, last_cum=None,
        bar_open=None, bar_high=None, bar_low=None, bar=None,
        hist={k: deque(maxlen=HIST_LEN) for k in ("high", "low", "close", "iv", "oi")},
    )


class ChainGrid:
    def __init__(self, feature_names, weekly_df, expiry_date, iv_fn,
                 center_strike, step=50, half_width=14, recenter_at=4):
        """
        feature_names : model.features
        weekly_df     : scrip rows of the target weekly expiry
        iv_fn         : compute_live_iv(price, spot, strike, expiry, 'c'|'p')
        center_strike : startup ATM
        half_width    : strikes either side of center to subscribe (14 -> 29 strikes
                        x 2 sides = 58 legs, safely under Dhan's 100/request limit)
        recenter_at   : re-center the band once ATM is this many strikes from center
        """
        self.cols = parse_chain_columns(feature_names)
        self.offsets = sorted({off for _, off, _, _ in self.cols})
        max_off = max((abs(o) for o in self.offsets), default=10)
        self.step = step
        self.expiry = expiry_date
        self.iv_fn = iv_fn
        self.recenter_at = recenter_at
        # band must still cover ATM+-max_off when ATM sits recenter_at strikes off-center
        self.half_width = max(half_width, max_off + recenter_at)
        self.center = int(center_strike)

        self.legs = {}         # (strike, side) -> leg
        self.id_to_leg = {}    # security_id   -> leg
        self.latest = None     # window snapshot {column_name: value}
        self.atm = None        # {"CALL": bar, "PUT": bar} for current ATM
        self.atm_strike = None
        self.last_spot = None

        self._sid = {}
        for s, o, i in zip(weekly_df["SEM_STRIKE_PRICE"],
                           weekly_df["SEM_OPTION_TYPE"],
                           weekly_df["SEM_SMST_SECURITY_ID"]):
            try:
                opt = str(o).strip().upper()
                strike = int(round(float(s)))
                sid = str(int(i))
                if opt in ("CE", "CALL"):
                    self._sid[(strike, "CE")] = sid
                elif opt in ("PE", "PUT"):
                    self._sid[(strike, "PE")] = sid
            except (TypeError, ValueError):
                continue

        self._build_band(self.center)
        log.info(f"ChainGrid[DYNAMIC]: center={self.center} half_width={self.half_width} "
                 f"legs={len(self.legs)} window_offsets={self.offsets[0]}..{self.offsets[-1]} "
                 f"chain columns={len(self.cols)}")

    # ── band management ────────────────────────────────────────────────────
    def _build_band(self, center):
        added, missing = [], []
        for k in range(-self.half_width, self.half_width + 1):
            strike = center + k * self.step
            for side, opt in (("CALL", "CE"), ("PUT", "PE")):
                if (strike, side) in self.legs:
                    continue
                sid = self._sid.get((strike, opt))
                if sid is None:
                    missing.append((strike, side))
                    continue
                leg = _new_leg(sid, strike, side)
                self.legs[(strike, side)] = leg
                self.id_to_leg[sid] = leg
                added.append(leg)
        if missing:
            log.warning(f"ChainGrid: no security_id for {len(missing)} legs "
                        f"{missing[:6]}{'...' if len(missing) > 6 else ''}")
        return added

    def ensure_leg(self, strike, side):
        """Ensure a contract exists even outside the dynamic-ATM band.

        The caller should protect this (strike, side) in maybe_recenter().
        Returns the Dhan security ID, or None if the contract is unavailable.
        """
        strike = int(round(float(strike)))
        side = str(side).upper()
        if side not in ("CALL", "PUT"):
            raise ValueError(f"Unsupported option side: {side}")

        key = (strike, side)
        existing = self.legs.get(key)
        if existing is not None:
            return existing["sid"]

        opt_type = "CE" if side == "CALL" else "PE"
        sid = self._sid.get((strike, opt_type))
        if sid is None:
            log.error(
                f"ChainGrid.ensure_leg: no security ID for {strike} {side}"
            )
            return None

        leg = _new_leg(sid, strike, side)
        self.legs[key] = leg
        self.id_to_leg[str(sid)] = leg
        log.info(
            f"ChainGrid: ensured fixed rule leg strike={strike} "
            f"side={side} sid={sid}"
        )
        return sid

    @property
    def ids(self):
        return set(self.id_to_leg.keys())

    def maybe_recenter(self, protect=()):
        """
        Re-center the band on the current ATM once it has drifted recenter_at
        strikes. `protect` = {(strike, side)} legs to keep (open trade).
        Returns (added_ids, dropped_ids) -- caller (un)subscribes them.
        """
        if self.atm_strike is None:
            return [], []
        if abs(self.atm_strike - self.center) < self.recenter_at * self.step:
            return [], []
        new_c = self.atm_strike
        lo = new_c - self.half_width * self.step
        hi = new_c + self.half_width * self.step
        added = [l["sid"] for l in self._build_band(new_c)]
        dropped = []
        protect = {(int(s), sd) for s, sd in protect}
        for key, leg in list(self.legs.items()):
            if key in protect:
                continue
            if not (lo <= key[0] <= hi):
                dropped.append(leg["sid"])
                del self.legs[key]
                del self.id_to_leg[leg["sid"]]
        log.info(f"ChainGrid: re-centered {self.center} -> {new_c}  "
                 f"+{len(added)} legs -{len(dropped)} legs")
        self.center = new_c
        return added, dropped

    # ── cold-start seed (REST). NOT verified against a live response. ──────
    def seed_from_rest(self, client_id, access_token, ids=None):
        ids = sorted(self.ids if ids is None else {str(i) for i in ids})
        if not ids:
            return 0
        try:
            r = requests.post(
                "https://api.dhan.co/v2/marketfeed/quote",
                json={"NSE_FNO": [int(s) for s in ids]}, timeout=15,
                headers={"Content-Type": "application/json", "Accept": "application/json",
                         "access-token": access_token, "client-id": client_id})
            if r.status_code != 200:
                log.warning(f"ChainGrid.seed: HTTP {r.status_code}: {r.text[:200]}")
                return 0
            data = r.json().get("data", {}).get("NSE_FNO", {})
            n = 0
            for sid, q in data.items():
                leg = self.id_to_leg.get(str(sid))
                if not leg or not isinstance(q, dict):
                    continue
                px = float(q.get("last_price") or 0)
                if px > 0:
                    leg["last_px"] = px
                    n += 1
                leg["last_oi"] = float(q.get("oi") or 0)
                vol = q.get("volume")
                if vol:
                    leg["cum"] = leg["last_cum"] = float(vol)
            log.info(f"ChainGrid.seed: seeded {n}/{len(ids)} legs from REST quote")
            return n
        except Exception as e:
            log.warning(f"ChainGrid.seed failed ({e}) - legs fill from live ticks only")
            return 0

    # ── tick path: cheap, no IV maths ──────────────────────────────────────
    def on_tick(self, sec_id, ltp, cum_volume, oi):
        # Dhan/feed versions may deliver the same security ID as int or str;
        # IDs are normalized to strings when the grid is built.
        leg = self.id_to_leg.get(str(sec_id))
        if leg is None:
            return False
        if ltp and ltp > 0:
            if leg["bar_open"] is None:
                leg["bar_open"] = leg["bar_high"] = leg["bar_low"] = ltp
            else:
                leg["bar_high"] = max(leg["bar_high"], ltp)
                leg["bar_low"] = min(leg["bar_low"], ltp)
            leg["last_px"] = ltp
        if cum_volume and cum_volume > 0:
            leg["cum"] = float(cum_volume)
        if oi and oi > 0:
            leg["last_oi"] = float(oi)
        return True

    # ── bar boundary ───────────────────────────────────────────────────────
    def flush_bar(self, spot):
        """
        Close the bar for EVERY leg (so each has continuous history), then
        resolve the dynamic ATM from this bar's closing spot.
        Returns {"CALL": bar, "PUT": bar} for the ATM, or None if the ATM
        legs have no price (caller should skip the bar).
        bar = dict(open, high, low, close, volume, oi, iv)
        """
        try:
            spot = float(spot)
        except (TypeError, ValueError):
            return None
        if not finite_positive(spot):
            return None
        self.last_spot = spot
        atm = int(round(spot / self.step) * self.step)
        if self.atm_strike is not None and atm != self.atm_strike:
            log.info(f"ATM roll {self.atm_strike} -> {atm}  spot={spot:.2f}")
        self.atm_strike = atm

        for leg in self.legs.values():
            px = leg["last_px"]
            if px is None or not np_is_finite_positive(px):
                # Do not carry a stale partial OHLC accumulator across an
                # empty minute. Otherwise a later tick would inherit the
                # previous bar's open/high/low and corrupt indicators.
                leg["bar"] = None
                leg["bar_open"] = leg["bar_high"] = leg["bar_low"] = None
                continue
            opn = leg["bar_open"] if leg["bar_open"] is not None else px
            hi = leg["bar_high"] if leg["bar_high"] is not None else px
            lo = leg["bar_low"] if leg["bar_low"] is not None else px

            # per-BAR volume from cumulative day volume; first observation -> 0
            if leg["cum"] is None or leg["last_cum"] is None:
                vol = 0.0
            else:
                vol = leg["cum"] - leg["last_cum"]
                if vol < 0:                      # session reset
                    vol = leg["cum"]
            if leg["cum"] is not None:
                leg["last_cum"] = leg["cum"]

            try:
                iv = self.iv_fn(
                    px, spot, leg["strike"], self.expiry,
                    "c" if leg["side"] == "CALL" else "p",
                )
                if iv is not None and finite_positive(iv):
                    leg["last_iv"] = iv_use = float(iv)
                else:
                    iv_use = leg["last_iv"] if leg["last_iv"] is not None else \
                             (18.0 if leg["side"] == "CALL" else 19.0)
            except Exception as exc:
                log.debug(
                    f"ChainGrid IV calc failed for {leg['strike']} {leg['side']}: {exc}"
                )
                iv_use = leg["last_iv"] if leg["last_iv"] is not None else \
                         (18.0 if leg["side"] == "CALL" else 19.0)

            leg["bar"] = dict(open=float(opn), high=float(hi), low=float(lo),
                              close=float(px), volume=float(max(vol, 0.0)),
                              oi=float(leg["last_oi"]), iv=float(iv_use))
            h = leg["hist"]
            h["high"].append(hi); h["low"].append(lo); h["close"].append(px)
            h["iv"].append(iv_use); h["oi"].append(leg["last_oi"])
            leg["bar_open"] = leg["bar_high"] = leg["bar_low"] = None

        # window snapshot relative to the CURRENT atm
        snap = {}
        for off in self.offsets:
            for side in ("CALL", "PUT"):
                leg = self.legs.get((atm + off * self.step, side))
                if leg is None or leg["bar"] is None:
                    continue
                b = leg["bar"]
                pre = f"nifty_WEEK_{_tag(off)}_{side}_1m__"
                snap[pre + "open"] = b["open"]
                snap[pre + "volume"] = b["volume"]
                snap[pre + "iv"] = b["iv"]
                snap[pre + "oi"] = b["oi"]
                snap[pre + "spot"] = float(spot)
        self.latest = snap

        c, p = self.legs.get((atm, "CALL")), self.legs.get((atm, "PUT"))
        if not c or not p or c["bar"] is None or p["bar"] is None:
            self.atm = None
            return None
        self.atm = {"CALL": c["bar"], "PUT": p["bar"]}
        return self.atm

    def merge_into(self, base_row):
        """base_row + every grid column the model wants (all relative to dynamic ATM)."""
        row = dict(base_row)
        if self.latest:
            for name, _off, _side, _field in self.cols:
                if name in self.latest:
                    row[name] = self.latest[name]
        return row

    # ── single-strike accessors (traded option) ────────────────────────────
    def leg_series(self, strike, side):
        key = (int(round(float(strike))), str(side).upper())
        leg = self.legs.get(key)
        if leg is None or len(leg["hist"]["close"]) == 0:
            return None
        return leg["hist"]

    def leg_close(self, strike, side):
        key = (int(round(float(strike))), str(side).upper())
        leg = self.legs.get(key)
        return leg["bar"]["close"] if leg and leg["bar"] else None

    def subscription_tuples(self, market_feed_cls, exclude_ids=()):
        ex = {str(x) for x in exclude_ids}
        return [(market_feed_cls.NSE_FNO, sid, market_feed_cls.Full)
                for sid in sorted(self.ids) if sid not in ex]

    def coverage(self):
        """(# of ATM+-N window legs with a live price, window size)"""
        tot = live = 0
        atm = self.atm_strike if self.atm_strike is not None else self.center
        for off in self.offsets:
            for side in ("CALL", "PUT"):
                tot += 1
                leg = self.legs.get((atm + off * self.step, side))
                if leg and leg["last_px"]:
                    live += 1
        return live, tot