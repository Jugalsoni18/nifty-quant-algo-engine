from __future__ import annotations
import numpy as np
from datetime import time as dtime
from collections import deque
from typing import Dict, Optional

"""
build_features_v12.py  (v12_l3, CORRECTED)
===========================================
Standalone feature-builder module for nifty_v12_l3, matching the shape of
the build_features_v12.py reference file you originally pasted in, with the
SAME fix set applied to this file that v12_inference.py (l3) now has.

CONFIRMED FIXES (via l3's own trained feature-name list, extracted from the
model's pickle -- extracted_features.json / the 245-name dump you pulled).
A feature surviving as its own distinct, named, non-constant entry in a
trained model's feature list cannot actually be a hardcoded constant or an
always-identical duplicate in training data, so these are treated as
confirmed, not just suspected -- the same standard of evidence that already
settled the identical questions for l5 directly against its training script:

  1. adx_1m_rank / price_position_rank / adx_5m_rank / adx_15m_rank /
     adx_30m_rank  -- WAS hardcoded to 0.5. All five are real, separately-
                       named entries in l3's trained feature list. Fixed:
                       RvolHistoryV12 now tracks adx_1m/price self-rank
                       history; HTFStateV12 tracks a per-timeframe ADX-rank
                       history, appended once per newly-COMPLETED HTF bar
                       (ported verbatim from the l5 fix -- same formula
                       family, long-horizon self-rank).
  2. di_cross_15m / di_cross_30m
                    -- were never emitted at all (only di_cross_5m was),
                       despite HTFStateV12.get() already computing di_cross
                       generically for every timeframe. Both names are in
                       l3's trained feature list. Fixed.
  3. near_s1        -- get_features() computed spot_near_support (a near-
                       duplicate check against s1) but never emitted a bare
                       "near_s1" key. near_s1 is confirmed present in l3's
                       trained feature list. Fixed.
  4. stoch_k / stoch_d
                    -- _stoch() returned (K, K), identical every bar. Both
                       names are SEPARATELY present in l3's trained feature
                       list, which is only possible if they differ in
                       training data. Fixed with a real double-SMA tracker
                       (raw %K -> SMA(3) = stoch_k -> SMA(3) of THAT =
                       stoch_d), ported verbatim from the l5 fix.

UNVERIFIED / BEST-EVIDENCE-ONLY (a feature-name list proves a feature is
real, not a constant -- it cannot prove which WINDOW or ORDERING convention
produced it. These three are applied here because l3's own previously-
pasted build_features_v12.py reference file already used the l5-confirmed
convention, and because matching l5's training-script-proven formula is the
best evidence available without l3's own training script. Treat these as
PROVISIONAL, not proven, until l3's training script or a computed-value
dump is available to arbitrate):

  5. rvol_pct_rank  -- switched from rank-BEFORE-update (current bar
                       excluded) to update-THEN-rank (self-rank, current
                       bar included).
  6. iv_rank / atm_call_iv_rank / atm_put_iv_rank
                    -- switched from a [-21:-1] (current-bar-excluded)
                       window to [-20:] (current-bar-included).
  7. rvol_roll3_max -- switched from a shifted/misaligned-denominator
                       formula to reading max() of the last 3 already-
                       computed rvol values straight from RvolHistoryV12's
                       history deque (matches training's
                       df["rvol"].rolling(3).max() convention, as confirmed
                       for l5). rvol_roll3_max IS a confirmed real trained
                       l3 feature; only the exact formula is inferred here.

Also present, unchanged from the original file: atr_5m_x / atr_5m_y /
atr_pct_5m_x / atr_pct_5m_y -- duplicate keys not in the trained feature
list, harmless (dropped by X_row[self.features] selection at inference
time), left in place for parity with the deployed file rather than removed
here.

If you obtain l3's own training script or a bar-by-bar computed-value dump,
re-verify items 5-7 against it the same rigorous way items 1-4 were already
settled, and update this file again if they disagree.
"""

PERSIST_WINDOW = 5


# ══════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ══════════════════════════════════════════════════════════════════════

def _ema(arr: np.ndarray, span: int) -> float:
    if len(arr) == 0: return 0.0
    alpha = 2.0 / (span + 1)
    v = float(arr[0])
    for x in arr[1:]:
        v = alpha * float(x) + (1 - alpha) * v
    return v

def _rsi(arr: np.ndarray, p: int = 14) -> float:
    if len(arr) < p + 1: return 50.0
    d = np.diff(arr[-(p+1):].astype(float))
    g = np.where(d > 0, d, 0.0).mean()
    l = np.where(d < 0, -d, 0.0).mean()
    return float(100 - 100 / (1 + g / (l + 1e-9)))

def _macdh(arr: np.ndarray, f=12, sl=26, sg=9) -> float:
    if len(arr) < sl: return 0.0
    a   = arr.astype(float)
    ml  = _ema(a, f) - _ema(a, sl)
    mls = np.array([_ema(a[:i+1], f) - _ema(a[:i+1], sl)
                    for i in range(max(0, len(a)-sg-1), len(a))])
    return float(ml - (_ema(mls, sg) if len(mls) > 0 else ml))

def _adx(arr: np.ndarray, p: int = 14):
    if len(arr) < p + 1: return 20.0, 20.0, 20.0
    a   = arr.astype(float)
    pdm = np.maximum(np.diff(a), 0)
    ndm = np.maximum(-np.diff(a), 0)
    pdm = np.where(pdm > ndm, pdm, 0.0)
    ndm = np.where(ndm > pdm, ndm, 0.0)
    atr_= np.abs(np.diff(a))
    def _sm(x): return _ema(x[-(p+1):], p)
    pdi = 100 * _sm(pdm) / (_sm(atr_) + 1e-9)
    ndi = 100 * _sm(ndm) / (_sm(atr_) + 1e-9)
    dx  = 100 * abs(pdi - ndi) / (pdi + ndi + 1e-9)
    return float(dx), float(pdi), float(ndi)

def _bb(arr: np.ndarray, p: int = 20):
    if len(arr) < p: return 0.5, 0.1
    w = arr[-p:].astype(float)
    m = w.mean(); s = w.std(ddof=0) + 1e-9
    u = m + 2*s; l = m - 2*s
    return float((arr[-1] - l) / (u - l + 1e-9)), float((u - l) / (m + 1e-9))

def _atr(arr: np.ndarray, p: int = 14) -> float:
    if len(arr) < p + 1: return float(arr[-1] * 0.01) if len(arr) else 1.0
    return float(np.abs(np.diff(arr[-(p+1):].astype(float))).mean())

def _cmf(hi, lo, cl, vol, p=14) -> float:
    if len(cl) < p: return 0.0
    h=hi[-p:]; l=lo[-p:]; c=cl[-p:]; v=vol[-p:]
    mfv = ((c-l)-(h-c))/(h-l+1e-9)*v
    return float(mfv.sum()/(v.sum()+1e-9))

def _supertrend_dir(hi, lo, cl, p=10, mult=3.0) -> int:
    if len(cl) < p+1: return 1
    atr = _atr(np.array(cl), p)
    lb  = float(hi[-1]+lo[-1])/2 - mult*atr
    return 1 if cl[-1] > lb else -1

def _raw_stoch_k(hi, lo, cl, k=14) -> float:
    """Single-period raw %K -- feed this into RvolHistoryV12.stoch_update() to get
    training's actual double-smoothed stoch_k/stoch_d (smooth_k=3, d=3), not a
    one-shot value. [CONFIRMED FIX #4]"""
    if len(cl) < k: return 50.0
    h = hi[-k:].max(); l = lo[-k:].min()
    return float((cl[-1]-l)/(h-l+1e-9)*100)


# ══════════════════════════════════════════════════════════════════════
# HTF STATE  (leak-proof — completed bars only)
# ══════════════════════════════════════════════════════════════════════

class HTFStateV12:
    WINDOWS = {5: 300, 15: 200, 30: 100}
    # adx_{tf}_rank: confirmed present as REAL trained features (l3's
    # extracted feature list lists adx_5m_rank/adx_15m_rank/adx_30m_rank
    # distinctly from adx_5m/15m/30m). Was hardcoded to 0.5 -- ported the l5
    # fix verbatim: one rank value per newly-completed HTF bar (not per 1m
    # tick), long-horizon self-rank of ADX on that timeframe's own history.
    # [CONFIRMED FIX #1]
    RANK_WINDOWS = {5: 1500, 15: 500, 30: 250}
    RANK_MIN_PERIODS = 20

    def __init__(self):
        self._bars          = {tf: deque(maxlen=w) for tf, w in self.WINDOWS.items()}
        self._open_bar       = {tf: None            for tf in self.WINDOWS}
        self._adx_rank_hist  = {tf: deque(maxlen=w) for tf, w in self.RANK_WINDOWS.items()}
        self._last_adx_rank  = {tf: 0.5 for tf in self.WINDOWS}

    def update(self, bar_time, o, h, l, c, vol, sp):
        for tf in self.WINDOWS:
            bucket = (bar_time.hour*60 + bar_time.minute) // tf * tf
            ob = self._open_bar[tf]
            if ob is None or ob["bucket"] != bucket:
                if ob is not None:
                    self._bars[tf].append(ob)
                    self._update_adx_rank(tf)   # a bar for this tf just completed
                self._open_bar[tf] = {"bucket":bucket,"o":o,"h":h,"l":l,"c":c,"vol":vol,"sp":sp}
            else:
                ob["h"]   = max(ob["h"], h)
                ob["l"]   = min(ob["l"], l)
                ob["c"]   = c
                ob["vol"] += vol
                ob["sp"]  = sp

    def _update_adx_rank(self, tf):
        bars = list(self._bars[tf])
        if len(bars) < 15:
            return
        cl = np.array([b["c"] for b in bars], dtype=float)
        adx_v, _, _ = _adx(cl, 14)
        hist = self._adx_rank_hist[tf]
        hist.append(adx_v)
        if len(hist) < self.RANK_MIN_PERIODS:
            self._last_adx_rank[tf] = 0.5
        else:
            a = np.array(hist)
            self._last_adx_rank[tf] = float((a <= a[-1]).mean())

    def get_rank(self, tf):
        return self._last_adx_rank.get(tf, 0.5)

    def get(self, tf):
        bars = list(self._bars[tf])
        if len(bars) < 2:
            return {"rsi":50.0,"adx":20.0,"pdi":20.0,"ndi":20.0,
                    "bb_pctb":0.5,"bb_bw":0.1,"macdh":0.0,
                    "atr":1.0,"atr_pct":1.0,"cmf":0.0,
                    "st_dir":1,"trend":0,"ema200_gap_pct":0.0,
                    "above_ema200":0,"adx_regime":0,"di_cross":0}
        cl  = np.array([b["c"]   for b in bars], dtype=float)
        hi  = np.array([b["h"]   for b in bars], dtype=float)
        lo  = np.array([b["l"]   for b in bars], dtype=float)
        vol = np.array([b["vol"] for b in bars], dtype=float)
        rsi_v             = _rsi(cl, 14)
        adx_v, pdi_v, ndi_v = _adx(cl, 14)
        pctb, bw          = _bb(cl, 20)
        macdh_v           = _macdh(cl)
        atr_v             = _atr(cl, 14)
        ema200_v          = _ema(cl, min(200, len(cl)))
        cmf_v             = _cmf(hi, lo, cl, vol, 14)
        st_v              = _supertrend_dir(hi, lo, cl, 10, 3.0)
        ema9 = _ema(cl,9); ema21 = _ema(cl,21)
        trend_v = 1 if (cl[-1]>ema9 and ema9>ema21) else -1 if (cl[-1]<ema9 and ema9<ema21) else 0
        ema200_gap = float((cl[-1]-ema200_v)/(cl[-1]+1e-9)*100)
        return {
            "rsi":rsi_v,"adx":adx_v,"pdi":pdi_v,"ndi":ndi_v,
            "bb_pctb":pctb,"bb_bw":bw,"macdh":macdh_v,
            "atr":atr_v,"atr_pct":atr_v/(cl[-1]+1e-9)*100,"cmf":cmf_v,
            "st_dir":st_v,"trend":trend_v,"ema200_gap_pct":ema200_gap,
            "above_ema200":int(cl[-1]>ema200_v),
            "adx_regime":int(adx_v>=25),
            "di_cross":1 if pdi_v>ndi_v else -1,
        }


# ══════════════════════════════════════════════════════════════════════
# PIVOT STATE  (previous day — no lookahead)
# ══════════════════════════════════════════════════════════════════════

class PivotStateV12:
    def __init__(self):
        self._date = None
        self._h = self._l = self._c = None
        self.pivot = self.s1 = self.r1 = None

    def update(self, bar_date, spot_high, spot_low, spot_close):
        if self._date != bar_date:
            if self._date is not None and self._h is not None:
                p = (self._h + self._l + self._c) / 3
                self.pivot = p
                self.s1    = 2*p - self._h
                self.r1    = 2*p - self._l
            self._date = bar_date
            self._h = spot_high; self._l = spot_low; self._c = spot_close
        else:
            self._h = max(self._h or spot_high, spot_high)
            self._l = min(self._l or spot_low,  spot_low)
            self._c = spot_close

    def get_features(self, spot):
        tol = 0.003
        # near_s1: confirmed present as its OWN real trained feature in l3's
        # extracted feature list, distinct from the near-duplicate
        # spot_near_support. Previously never emitted at all -- live
        # inference fed 0.0 for a real feature on every bar. [CONFIRMED FIX #3]
        if self.pivot is None:
            return {"near_pivot":0,"near_r1":0,"near_s1":0,"spot_near_support":0,
                    "spot_near_resistance":0,"dist_to_pivot_pct":0.0}
        return {
            "near_pivot"           : int(abs(spot-self.pivot)/(spot+1e-9)<tol),
            "near_r1"              : int(abs(spot-self.r1)   /(spot+1e-9)<tol),
            "near_s1"              : int(abs(spot-self.s1)   /(spot+1e-9)<tol),
            "spot_near_support"    : int(abs(spot-self.s1)   /(spot+1e-9)<tol),
            "spot_near_resistance" : int(abs(spot-self.r1)   /(spot+1e-9)<tol),
            "dist_to_pivot_pct"    : float((spot-self.pivot) /(spot+1e-9)),
        }


# ══════════════════════════════════════════════════════════════════════
# RVOL HISTORY  (long-horizon percentile rank)
# ══════════════════════════════════════════════════════════════════════

class RvolHistoryV12:
    W = 20 * 375
    def __init__(self):
        self._rv  = deque(maxlen=self.W)
        self._vix = deque(maxlen=self.W)
        # adx_1m_rank / price_position_rank: confirmed present as REAL trained
        # features. Was hardcoded to 0.5 -- ported the l5 fix verbatim (same
        # formula family as rvol_pct_rank: long-horizon self-rank, tracked
        # incrementally rather than re-derived from the raw buffer every bar).
        # [CONFIRMED FIX #1]
        self._adx1m  = deque(maxlen=self.W)
        self._price  = deque(maxlen=self.W)
        # stoch_k / stoch_d: confirmed present as two SEPARATE real trained
        # features. A one-shot _stoch() returning (K, K) fed the model
        # stoch_k==stoch_d on every bar -- impossible for two independently-
        # varying trained columns. Fixed via genuine double-SMA smoothing.
        # [CONFIRMED FIX #4]
        self._raw_k  = deque(maxlen=5)
        self._k3     = deque(maxlen=5)

    def update(self, rvol, vix):
        self._rv.append(rvol); self._vix.append(vix)

    def stoch_update(self, raw_k):
        self._raw_k.append(raw_k)
        arr = np.array(self._raw_k)
        k3 = float(arr[-min(3, len(arr)):].mean())
        self._k3.append(k3)

    def stoch_kd(self):
        if not self._k3: return 50.0, 50.0
        k3 = self._k3[-1]
        arr = np.array(self._k3)
        d3 = float(arr[-min(3, len(arr)):].mean())
        return k3, d3

    def update_extra(self, adx1m, price):
        self._adx1m.append(adx1m); self._price.append(price)

    def pct_rank(self):
        if len(self._rv) < 100: return 0.5
        a = np.array(self._rv)
        return float((a <= a[-1]).mean())

    def rvol_roll3_max(self):
        """UNVERIFIED CONVENTION [#7]: assumes training does
        df["rvol_roll3_max"] = df["rvol"].rolling(3).max() on already-computed
        rvol values (the l5-confirmed formula), read straight from history.
        rvol_roll3_max IS a confirmed real trained l3 feature; this exact
        formula is inferred by analogy, not confirmed against l3's own
        training script."""
        if not self._rv: return 0.0
        return float(max(list(self._rv)[-3:]))

    def adx1m_rank(self):
        if len(self._adx1m) < 100: return 0.5
        a = np.array(self._adx1m)
        return float((a <= a[-1]).mean())

    def price_rank(self):
        if len(self._price) < 100: return 0.5
        a = np.array(self._price)
        return float((a <= a[-1]).mean())

    def vix_ma20(self):
        return float(np.mean(list(self._vix))) if len(self._vix) >= 100 else 16.0

    def vix_med20(self):
        return float(np.median(list(self._vix))) if len(self._vix) >= 100 else 16.0


# ══════════════════════════════════════════════════════════════════════
# FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_features(
    raw_row: dict,           # full CSV row dict (all ATM/ATMminus/ATMplus cols)
    buf: dict,               # rolling buffer: call_close, put_close, call_iv,
                             #   put_iv, call_oi, put_oi, call_vol, put_vol, spot
                             #   call_open, call_high, call_low, datetime
    htf: HTFStateV12,        # HTF state object — updated by caller before this call
    pivot: PivotStateV12,    # Pivot level state
    rvol_hist: RvolHistoryV12, # RVOL percentile history (+ adx1m/price/stoch state)
    india_vix: float = 16.0,
    vix_chg: float   = 0.0,
    feat_cols: list  = None, # list of features model expects (top_features_v12.pkl)
) -> Optional[dict]:
    """
    Build the full v12_l3 feature row for inference.

    Usage:
        feat_cols = joblib.load(MODEL_DIR / 'top_features_v12.pkl')
        htf       = HTFStateV12()
        pivot     = PivotStateV12()
        rvol_hist = RvolHistoryV12()

        # In the bar loop:
        htf.update(bar_time, open_, high, low, close, volume, spot)
        pivot.update(bar_date, spot_high, spot_low, spot_close)
        feats = build_features(raw_row, buf, htf, pivot, rvol_hist,
                                india_vix, vix_chg, feat_cols)
    """
    # Warmup guard
    if len(buf.get("call_close", [])) < 60:
        return None

    def _arr(key, default=0.0):
        v = buf.get(key, [])
        return np.array(list(v), dtype=float) if len(v) > 0 else np.array([default])

    c   = _arr("call_close")
    p   = _arr("put_close")
    ch  = _arr("call_high",  c[-1] if len(c) else 100.0)
    cl_ = _arr("call_low",   c[-1] if len(c) else 100.0)
    co  = _arr("call_open",  c[-1] if len(c) else 100.0)
    cv  = _arr("call_vol",   1e4).clip(0)
    pv  = _arr("put_vol",    1e4).clip(0)
    coi = _arr("call_oi",    1e7).clip(0)
    poi = _arr("put_oi",     1e7).clip(0)
    civ = _arr("call_iv",    20.0)
    piv = _arr("put_iv",     20.0)
    sp  = _arr("spot",       22000.0)

    now = list(buf["datetime"])[-1] if "datetime" in buf else None
    if now is None: return None
    t   = now.time()
    dow = now.weekday()
    iv_mid = (civ + piv) / 2

    # ── Volume / RVOL  [UNVERIFIED CONVENTION #5]
    # update-THEN-rank (self-rank, current bar included) -- matches l5's
    # training-script-confirmed convention and l3's own build_features_v12.py
    # reference file. NOT confirmed against an l3 training script.
    cv_c = cv.clip(0); pv_c = pv.clip(0)
    tv   = cv_c + pv_c
    tv20 = float(np.mean(tv[-20:]) + 1e-9)
    rvol = float(tv[-1]) / tv20
    rvol_hist.update(rvol, india_vix)   # append current bar FIRST
    rvol_pct_rank = rvol_hist.pct_rank() # then rank (includes current bar)
    vix_ma20_v    = rvol_hist.vix_ma20()
    vix_med20_v   = rvol_hist.vix_med20()

    # ── 1m call indicators
    rsi_1m            = _rsi(c, 14)
    adx_1m, pdi_, ndi_= _adx(c, 14)
    bb_pctb, bb_bw    = _bb(c, 20)
    macdh_1m          = _macdh(c)
    atr14             = _atr(c, 14)
    ema200_c          = _ema(c, min(200, len(c)))
    rvol_hist.stoch_update(_raw_stoch_k(ch, cl_, c, 14))
    stoch_k, stoch_d  = rvol_hist.stoch_kd()
    cmf_1m            = _cmf(ch, cl_, c, cv, 14)

    # adx_1m_rank / price_position_rank: self-rank against long history.
    # [CONFIRMED FIX #1]
    rvol_hist.update_extra(adx_1m, float(c[-1]))
    adx1m_pr = rvol_hist.adx1m_rank()
    price_pr = rvol_hist.price_rank()

    # ── HTF
    f5  = htf.get(5); f15 = htf.get(15); f30 = htf.get(30)

    # ── Regime
    vol_regime   = 0 if india_vix<12 else 1 if india_vix<16 else 2 if india_vix<22 else 3
    sp_ema50d    = _ema(sp, min(len(sp), 18750))
    trend_regime = 1 if sp[-1] > sp_ema50d else -1

    # ── MTF
    mtf_bull = int(f5["trend"]==1  and f15["st_dir"]==1  and f15["ema200_gap_pct"]>0)
    mtf_bear = int(f5["trend"]==-1 and f15["st_dir"]==-1 and f15["ema200_gap_pct"]<0)
    mtf_soft = int(f5["trend"]==1) + int(f15["st_dir"]==1) + int(f15["rsi"]>50)

    # ── Pivot
    piv_f = pivot.get_features(float(sp[-1]))

    # ── Time
    def sess(t_):
        if t_<dtime(9,30):  return 0
        if t_<dtime(11,30): return 1
        if t_<dtime(13,30): return 2
        return 3
    delta = (1-dow)%7
    if delta==0 and t>=dtime(15,30): delta=7

    # ── Candle
    _rng  = float(ch[-1]-cl_[-1]+1e-9)
    _body = abs(float(c[-1]-co[-1]))
    _cp   = (c[-1]-cl_[-1])/_rng

    row: Dict[str, float] = {}

    # ── 1. Raw chain columns (pass-through from raw_row)
    for col, val in raw_row.items():
        if "nifty_WEEK_ATM" in col:
            row[col] = float(val) if val is not None else 0.0

    # ── 2. Computed features
    row.update({
        "rvol"                  : rvol,
        "rvol_capped"           : min(rvol, 5),
        "U02_RVOL"              : 1 if rvol>=1.2 else -1 if rvol<0.8 else 0,
        # rvol_roll3_max [UNVERIFIED CONVENTION #7]
        "rvol_roll3_max"        : rvol_hist.rvol_roll3_max(),
        "rvol_pct_rank"         : rvol_pct_rank,
        "rvol_bucket"           : 0 if rvol_pct_rank<0.30 else 1 if rvol_pct_rank<0.70 else 2 if rvol_pct_rank<0.90 else 3,
        "rvol_x_pcr"            : rvol*(float(poi[-2]/(coi[-2]+1e-9)) if len(poi)>1 else 1.0),

        "returns_1"             : float((sp[-1]-sp[-2])/(sp[-2]+1e-9)) if len(sp)>1 else 0.0,
        "returns_5"             : float((sp[-1]-sp[-6])/(sp[-6]+1e-9)) if len(sp)>5 else 0.0,
        "returns_15"            : float((sp[-1]-sp[-16])/(sp[-16]+1e-9)) if len(sp)>15 else 0.0,
        "ret_1"                 : float((c[-1]-c[-2])/(c[-2]+1e-9)) if len(c)>1 else 0.0,
        "ret_5"                 : float((c[-1]-c[-6])/(c[-6]+1e-9)) if len(c)>5 else 0.0,
        "ret_10"                : float((c[-1]-c[-11])/(c[-11]+1e-9)) if len(c)>10 else 0.0,
        "momentum_5"            : float(c[-1]-c[-6]) if len(c)>5 else 0.0,
        "momentum_10"           : float(c[-1]-c[-11]) if len(c)>10 else 0.0,

        "rsi"                   : rsi_1m,
        "rsi_1m"                : rsi_1m,
        "rsi_slope"             : float(rsi_1m-_rsi(c[:-2],14)) if len(c)>15 else 0.0,
        "macdh"                 : macdh_1m,
        "macdh_1m"              : macdh_1m,
        "macd_slope"            : float(macdh_1m-_macdh(c[:-2])) if len(c)>28 else 0.0,
        "adx_1m"                : adx_1m,
        "pdi_1m"                : pdi_,
        "ndi_1m"                : ndi_,
        "adx_regime"            : int(adx_1m>=20),
        # adx_1m_rank [CONFIRMED FIX #1]
        "adx_1m_rank"           : adx1m_pr,
        "atr_pct_1m"            : atr14/(c[-1]+1e-9)*100,
        "atr_5m_x"              : f5["atr"],
        "atr_pct_5m_x"          : f5["atr_pct"],
        "atr_5m_y"              : f5["atr"],
        "atr_pct_5m_y"          : f5["atr_pct"],
        "bb_pctb"               : bb_pctb,
        "bb_bw"                 : bb_bw,
        "bb_squeeze"            : int(bb_bw<0.10),
        "stoch_k"               : stoch_k,
        "stoch_d"               : stoch_d,
        "above_ema200_1m"       : int(c[-1]>ema200_c),
        # price_position_rank [CONFIRMED FIX #1]
        "price_position_rank"   : price_pr,
        "cmf_1m"                : cmf_1m,
        "U05_CMF_Direction"     : 1 if cmf_1m>0.05 else -1 if cmf_1m<-0.05 else 0,

        "iv_skew"               : float(piv[-1]-civ[-1]),
        "iv_skew_slope"         : float((piv[-1]-civ[-1])-(piv[-4]-civ[-4])) if len(civ)>3 else 0.0,
        "iv_momentum_5bar"      : float(iv_mid[-1]-iv_mid[-6]) if len(iv_mid)>5 else 0.0,
        "iv_skew_ratio"         : float(piv[-1]/(civ[-1]+1e-9)),
        # iv_rank / atm_call_iv_rank / atm_put_iv_rank [UNVERIFIED CONVENTION #6]
        "iv_rank"               : float((iv_mid[-1]-iv_mid[-20:].min())/(iv_mid[-20:].max()-iv_mid[-20:].min()+1e-9)) if len(iv_mid)>=20 else 0.5,
        "atm_call_iv_rank"      : float((civ[-1]-civ[-20:].min())/(civ[-20:].max()-civ[-20:].min()+1e-9)) if len(civ)>=20 else 0.5,
        "atm_put_iv_rank"       : float((piv[-1]-piv[-20:].min())/(piv[-20:].max()-piv[-20:].min()+1e-9)) if len(piv)>=20 else 0.5,

        "pcr_vol"               : float(pv_c[-1]/(cv_c[-1]+1e-9)),
        "pcr_oi"                : float(poi[-1]/(coi[-1]+1e-9)),
        "pcr_oi_lag1"           : float(poi[-2]/(coi[-2]+1e-9)) if len(poi)>1 else 1.0,
        "pcr_oi_lag2"           : float(poi[-3]/(coi[-3]+1e-9)) if len(poi)>2 else 1.0,
        "pcr_oi_slope"          : float(poi[-1]/(coi[-1]+1e-9)-poi[-4]/(coi[-4]+1e-9)) if len(poi)>3 else 0.0,
        "call_oi_chg_norm"      : float(np.clip(coi[-1]-coi[-2],-1e7,1e7)/((coi[-1]+poi[-1])/2+1e-9)) if len(coi)>1 else 0.0,
        "put_oi_chg_norm"       : float(np.clip(poi[-1]-poi[-2],-1e7,1e7)/((coi[-1]+poi[-1])/2+1e-9)) if len(poi)>1 else 0.0,
        "net_oi_chg_norm"       : float((np.clip(poi[-1]-poi[-2],-1e7,1e7)-np.clip(coi[-1]-coi[-2],-1e7,1e7))/((coi[-1]+poi[-1])/2+1e-9)) if len(poi)>1 else 0.0,
        "oi_buildup"            : float((np.clip(poi[-1]-poi[-2],-1e7,1e7)+1e-9)/(abs(np.clip(coi[-1]-coi[-2],-1e7,1e7))+1e-9)) if len(poi)>1 else 1.0,
        "atm_oi_concentration"  : float(coi[-1]/(coi[-1]+poi[-1]+1e-9)),

        "max_pain_pct"          : float((c[-1]-round(sp[-1]/50)*50)/(c[-1]+1e-9)),
        "candle_body_ratio"     : float((c[-1]-co[-1])/_rng),
        "P01_Strong_Candle"     : 1 if (c[-1]>co[-1] and _body/_rng>0.5) else -1 if (c[-1]<co[-1] and _body/_rng>0.5) else 0,
        "P04_HH_HL_Pattern"     : 1 if (len(c)>1 and ch[-1]>ch[-2] and cl_[-1]>cl_[-2]) else -1 if (len(c)>1 and ch[-1]<ch[-2] and cl_[-1]<cl_[-2]) else 0,
        "P02_Close_Position"    : 1 if _cp>0.6 else -1 if _cp<0.4 else 0,
        "spot_mod_100"          : float(sp[-1]%100)/100,
        "spot_mod_50"           : float(sp[-1]%50)/50,

        "vix_rank"                  : min(india_vix/30.0, 1.0),
        "vix_ma20"                  : vix_ma20_v,
        "vix_above_ma"              : int(india_vix>vix_ma20_v),
        "vix_above_rolling_median"  : int(india_vix>vix_med20_v),
        "is_high_vol_regime_v2"     : int(vol_regime>=2),
        "is_crisis_regime"          : int(vol_regime==3),
        "vol_regime"                : float(vol_regime),
        "trend_regime"              : trend_regime,
        "U08_Volume_Surge_Dir"      : 1 if (rvol>1.3 and c[-1]>co[-1]) else -1 if (rvol>1.3 and c[-1]<co[-1]) else 0,

        **piv_f,

        "hour"              : now.hour,
        "minute"            : now.minute,
        "dayofweek"         : dow,
        "day_of_week"       : dow,
        "session"           : sess(t),
        "is_expiry"         : int(dow==1),
        "expiry_today"      : int(dow==1),
        "is_pre_close"      : int(t>=dtime(14,45)),
        "is_open_rush"      : int(dtime(9,15)<=t<=dtime(9,44)),
        "days_to_expiry"    : int(delta),

        "rsi_5m"            : f5["rsi"],    "macdh_5m"          : f5["macdh"],
        "adx_5m"            : f5["adx"],    "pdi_5m"            : f5["pdi"],
        "ndi_5m"            : f5["ndi"],    "atr_5m"            : f5["atr"],
        "atr_pct_5m"        : f5["atr_pct"],"bb_pctb_5m"        : f5["bb_pctb"],
        "bb_bw_5m"          : f5["bb_bw"], "cmf_5m"             : f5["cmf"],
        "ema200_gap_pct_5m" : f5["ema200_gap_pct"],
        "st_dir_5m"         : f5["st_dir"],"trend_5m"           : f5["trend"],
        # adx_5m_rank [CONFIRMED FIX #1]
        "adx_regime_5m"     : f5["adx_regime"],"adx_5m_rank"    : htf.get_rank(5),
        "di_cross_5m"       : f5["di_cross"],

        "rsi_15m"           : f15["rsi"],   "macdh_15m"         : f15["macdh"],
        "adx_15m"           : f15["adx"],   "pdi_15m"           : f15["pdi"],
        "ndi_15m"           : f15["ndi"],   "atr_15m"           : f15["atr"],
        "atr_pct_15m"       : f15["atr_pct"],"bb_pctb_15m"      : f15["bb_pctb"],
        "bb_bw_15m"         : f15["bb_bw"],"cmf_15m"            : f15["cmf"],
        "ema200_gap_pct_15m": f15["ema200_gap_pct"],
        "st_dir_15m"        : f15["st_dir"],"trend_15m"          : f15["trend"],
        # adx_15m_rank [CONFIRMED FIX #1]; di_cross_15m [CONFIRMED FIX #2]
        "adx_regime_15m"    : f15["adx_regime"],"adx_15m_rank"   : htf.get_rank(15),
        "di_cross_15m"      : f15["di_cross"],

        "rsi_30m"           : f30["rsi"],   "macdh_30m"         : f30["macdh"],
        "adx_30m"           : f30["adx"],   "pdi_30m"           : f30["pdi"],
        "ndi_30m"           : f30["ndi"],   "atr_30m"           : f30["atr"],
        "atr_pct_30m"       : f30["atr_pct"],"bb_pctb_30m"      : f30["bb_pctb"],
        "bb_bw_30m"         : f30["bb_bw"],"cmf_30m"            : f30["cmf"],
        "ema200_gap_pct_30m": f30["ema200_gap_pct"],
        "st_dir_30m"        : f30["st_dir"],"trend_30m"          : f30["trend"],
        # adx_30m_rank [CONFIRMED FIX #1]; di_cross_30m [CONFIRMED FIX #2]
        "adx_regime_30m"    : f30["adx_regime"],"adx_30m_rank"   : htf.get_rank(30),
        "di_cross_30m"      : f30["di_cross"],

        "mtf_aligned_bull"  : mtf_bull,
        "mtf_aligned_bear"  : mtf_bear,
        "mtf_bull_soft"     : mtf_soft,
    })

    if feat_cols is not None:
        for f in feat_cols:
            if f not in row:
                row[f] = 0.0
    return row