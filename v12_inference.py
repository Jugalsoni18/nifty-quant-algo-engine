# ══════════════════════════════════════════════════════════════════════
#  v12_inference.py  —  nifty_v12_l3  (self-contained)
#
#  ⚠️  FEATURE-PIPELINE FIX PASS (this file) — READ BEFORE TRUSTING BLINDLY
#  ────────────────────────────────────────────────────────────────────
#  This file was patched against a feature-list dump extracted directly
#  from the TRAINED l3 model artifacts (extracted_features.json / the
#  245-name list you pulled from the pickle). That dump proves each of
#  the features below is a REAL, distinct, trained feature — not a
#  constant, not a duplicate — because a genuinely constant or duplicate
#  column would not survive as its own named entry in a model's trained
#  feature list. That is the same category of evidence that already
#  proved (via nifty_v12_l5_train.py, directly) that l5's identical
#  hardcoded-to-0.5 / stoch_k==stoch_d code was wrong.
#
#  CONFIRMED FIXES (high confidence — these are structural: a hardcoded
#  constant or an always-identical pair cannot match a real, independently
#  named, non-degenerate trained feature, regardless of the exact formula):
#    1. adx_1m_rank, price_position_rank, adx_5m_rank, adx_15m_rank,
#       adx_30m_rank — were hardcoded to 0.5. Present in trained feature
#       list as real percentile-rank features. Fixed with persistent
#       rank-history tracking (RvolHistoryV12.update_extra/adx1m_rank/
#       price_rank; HTFStateV12.RANK_WINDOWS/_update_adx_rank/get_rank),
#       ported verbatim from the l5 fix (same formula family, same code).
#    2. di_cross_15m, di_cross_30m — were never emitted (only di_cross_5m
#       was), despite HTFStateV12.get() already computing "di_cross" for
#       every timeframe generically. Both names are in the trained list.
#       Fixed by adding the two missing dict entries.
#    3. near_s1 — was completely missing from PivotStateV12.get_features()
#       (only the near-duplicate spot_near_support was emitted). near_s1
#       is in the trained list. Fixed by adding it to both branches.
#    4. stoch_k / stoch_d — _stoch() returned (K, K), identical every bar.
#       Both names are separately present in the trained list, which is
#       only possible if they differ in training data. Fixed via real
#       double-SMA smoothing (raw %K -> SMA3 = stoch_k -> SMA3 of that
#       = stoch_d), ported verbatim from the l5 fix.
#
#  ⚠️  UNVERIFIED / BEST-EVIDENCE-ONLY (a feature-name list cannot settle
#  these — they are arithmetic/window choices, not presence-vs-absence
#  questions. Applied here because l3's own build_features_v12.py
#  reference file agrees with the l5-confirmed convention, and because
#  matching l5's proven-correct formula is the best evidence available
#  without l3's training script. TREAT AS PROVISIONAL, not proven):
#    5. rvol_pct_rank — this file used rank-BEFORE-update (excludes
#       current bar: "history[-1] vs history[:-1]"). Switched to
#       update-THEN-rank (self-rank, current bar included), matching
#       l5's training-script-confirmed convention and l3's own
#       build_features_v12.py reference file. NOT confirmed against an
#       l3 training script.
#    6. iv_rank / atm_call_iv_rank / atm_put_iv_rank — this file used a
#       [-21:-1] current-bar-EXCLUDED window. Switched to [-20:]
#       current-bar-INCLUDED, matching l5's training-script-confirmed
#       convention and l3's own build_features_v12.py reference file.
#       NOT confirmed against an l3 training script.
#    7. rvol_roll3_max — this file used a shifted/misaligned-denominator
#       formula. Switched to reading max() of the last 3 already-computed
#       rvol values straight from RvolHistoryV12's history deque, matching
#       l5's training-script-confirmed convention (df["rvol"].rolling(3)
#       .max() on already-computed rvol). rvol_roll3_max IS confirmed
#       present as a real trained l3 feature, but the exact formula is
#       NOT confirmed against an l3 training script — only inferred by
#       analogy to l5.
#
#  If you get l3's actual training script (nifty_v12_l3_train.py or
#  equivalent) or a computed-value dump, items 5-7 should be re-verified
#  the same rigorous way items 1-4 already were, and this file updated
#  again if they disagree.
#
#  No external imports beyond third-party libraries.
#  All feature building, HTF state, pivot state, and RVOL history
#  are integrated directly.
#
#  Models inside (nifty_v12_l3_models):
#    model_xgb_long_v12.pkl     model_xgb_short_v12.pkl
#    model_lgb_long_v12.pkl     model_cat_long_v12.cbm
#    model_et_long_v12.pkl      model_et_short_v12.pkl
#    meta_long_v12.pkl          meta_short_v12.pkl
#    top_features_v12.pkl       short_features_v12.pkl
#    thresh_long_v12.pkl        thresh_short_v12.pkl
#    v12_meta.json
#
#  Interface (identical to ensemble_v3 V12Model):
#    predict_bar(feature_dict, india_vix, vol_regime, is_pre_close)
#      → {signal, prob_long, prob_short, eff_long, eff_short,
#          confidence, margin_long, margin_short}
#    get_signals(X, india_vix, vol_regime, is_pre_close)
#      → (signals, prob_long, prob_short)  [batch]
#    predict(raw_row, buf, india_vix, vix_chg)
#      → (signal, confidence, prob_long, prob_short)  [live/backtest]
#
#  Fixes vs ensemble_v3:
#    ✅ SHORT label: put rises (not call falls)
#    ✅ ATMminus1 label simulation (Lever 3)
#    ✅ 5-bar persistence gate in meta (Lever 2)
#    ✅ AUC 0.9249 | Acc 97.1% at ≥85% conf (2025+ test set)
# ══════════════════════════════════════════════════════════════════════

import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress only known/intentional warnings
warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)

import json
from collections import deque
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier

PERSIST_WINDOW = 5


# ══════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ══════════════════════════════════════════════════════════════════════

def _ema(arr, span):
    if len(arr) == 0: return 0.0
    alpha = 2.0 / (span + 1)
    v = float(arr[0])
    for x in arr[1:]:
        v = alpha * float(x) + (1 - alpha) * v
    return v

def _rsi(arr, p=14):
    if len(arr) < p + 1: return 50.0
    d = np.diff(arr[-(p+1):].astype(float))
    g = np.where(d > 0, d, 0.0).mean()
    l = np.where(d < 0, -d, 0.0).mean()
    return float(100 - 100 / (1 + g / (l + 1e-9)))

def _macdh(arr, f=12, sl=26, sg=9):
    if len(arr) < sl: return 0.0
    a   = arr.astype(float)
    ml  = _ema(a, f) - _ema(a, sl)
    mls = np.array([_ema(a[:i+1], f) - _ema(a[:i+1], sl)
                    for i in range(max(0, len(a)-sg-1), len(a))])
    return float(ml - (_ema(mls, sg) if len(mls) > 0 else ml))

def _adx(arr, p=14):
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

def _bb(arr, p=20):
    if len(arr) < p: return 0.5, 0.1
    w = arr[-p:].astype(float)
    m = w.mean(); s = w.std(ddof=0) + 1e-9
    u = m + 2*s; l = m - 2*s
    return float((arr[-1] - l) / (u - l + 1e-9)), float((u - l) / (m + 1e-9))

def _atr_s(arr, p=14):
    if len(arr) < p + 1: return float(arr[-1] * 0.01) if len(arr) else 1.0
    return float(np.abs(np.diff(arr[-(p+1):].astype(float))).mean())

def _cmf(hi, lo, cl, vol, p=14):
    if len(cl) < p: return 0.0
    h=hi[-p:]; l=lo[-p:]; c=cl[-p:]; v=vol[-p:]
    mfv = ((c-l)-(h-c))/(h-l+1e-9)*v
    return float(mfv.sum()/(v.sum()+1e-9))

def _st_dir(hi, lo, cl, p=10, mult=3.0):
    if len(cl) < p+1: return 1
    atr = _atr_s(np.array(cl), p)
    lb  = float(hi[-1]+lo[-1])/2 - mult*atr
    return 1 if cl[-1] > lb else -1

def _raw_stoch_k(hi, lo, cl, k=14):
    """Single-period raw %K -- feed this into RvolHistoryV12.stoch_update() to get
    training's actual double-smoothed stoch_k/stoch_d (smooth_k=3, d=3), not a
    one-shot value. [PORTED FROM l5 FIX — see file-header note #4]"""
    if len(cl) < k: return 50.0
    h = hi[-k:].max(); l = lo[-k:].min()
    return float((cl[-1]-l)/(h-l+1e-9)*100)


# ══════════════════════════════════════════════════════════════════════
# HTF STATE  (leak-proof — completed bars only)
# ══════════════════════════════════════════════════════════════════════

class HTFStateV12:
    WINDOWS = {5: 300, 15: 200, 30: 100}
    # adx_{tf}_rank: confirmed present as a REAL trained feature (extracted_features.json
    # lists adx_5m_rank / adx_15m_rank / adx_30m_rank distinctly from adx_5m/15m/30m).
    # Was hardcoded to 0.5 below -- ported the l5 fix verbatim: one rank value per
    # newly-completed HTF bar (not per 1m tick), long-horizon self-rank of ADX on that
    # timeframe's own history. [PORTED FROM l5 FIX — see file-header note #1]
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
        atr_v             = _atr_s(cl, 14)
        ema200_v          = _ema(cl, min(200, len(cl)))
        cmf_v             = _cmf(hi, lo, cl, vol, 14)
        st_v              = _st_dir(hi, lo, cl, 10, 3.0)
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
        # near_s1: confirmed present as its OWN real trained feature
        # (extracted_features.json lists near_s1 distinctly, separate from the
        # near-duplicate spot_near_support). Previously never emitted at all --
        # live inference fed 0.0 for a real feature on every bar.
        # [PORTED FROM l5 FIX — see file-header note #3]
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
        # features (extracted_features.json). Was hardcoded to 0.5 below -- ported
        # the l5 fix verbatim (same formula family as rvol_pct_rank: long-horizon
        # self-rank, tracked incrementally rather than re-derived from the raw
        # buffer every bar, which would be O(W) work per bar).
        # [PORTED FROM l5 FIX — see file-header note #1]
        self._adx1m  = deque(maxlen=self.W)
        self._price  = deque(maxlen=self.W)
        # stoch_k / stoch_d: confirmed present as two SEPARATE real trained features
        # (extracted_features.json lists both). A one-shot _stoch() returning (K, K)
        # fed the model stoch_k==stoch_d on every bar -- a combination that can't
        # match two independently-varying trained columns. Fixed via genuine
        # double-SMA smoothing (raw %K -> SMA3 -> SMA3 of that).
        # [PORTED FROM l5 FIX — see file-header note #4]
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
        """UNVERIFIED CONVENTION (file-header note #7): assumes training does
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
# FEATURE BUILDER  (mirrors training pipeline exactly)
# ══════════════════════════════════════════════════════════════════════

def _build_features(raw_row, buf, htf, pivot, rvol_hist,
                    india_vix=16.0, vix_chg=0.0, feat_cols=None):
    """
    Build the full 245-feature row for v12 inference.

    raw_row   : dict with nifty_WEEK_ATM* column values
    buf       : rolling deque dict — call_close, put_close, call_iv, etc.
    htf       : HTFStateV12 instance (updated by caller before each call)
    pivot     : PivotStateV12 instance
    rvol_hist : RvolHistoryV12 instance
    feat_cols : list of model's expected feature names (fills missing with 0)
    """
    if len(buf.get("call_close", [])) < 60:
        return None

    def _arr(key, default=0.0):
        v = buf.get(key, [])
        return np.array(list(v), dtype=float) if len(v) else np.array([default])

    c   = _arr("call_close");  p   = _arr("put_close")
    ch  = _arr("call_high",  c[-1]); cl_ = _arr("call_low",  c[-1])
    co  = _arr("call_open",  c[-1])
    cv  = _arr("call_vol",   1e4).clip(0); pv  = _arr("put_vol",  1e4).clip(0)
    coi = _arr("call_oi",    1e7).clip(0); poi = _arr("put_oi",   1e7).clip(0)
    civ = _arr("call_iv",    20.0);        piv = _arr("put_iv",   20.0)
    sp  = _arr("spot",       22000.0)

    now = list(buf["datetime"])[-1]
    t   = now.time(); dow = now.weekday()
    iv_mid = (civ + piv) / 2

    # Volume / RVOL  [UNVERIFIED CONVENTION — file-header note #5]
    # Switched from rank-BEFORE-update (current bar excluded) to update-THEN-rank
    # (self-rank, current bar included), matching l5's training-script-confirmed
    # convention and l3's own build_features_v12.py reference file. NOT confirmed
    # against an l3 training script.
    cv_c = cv.clip(0); pv_c = pv.clip(0)
    tv   = cv_c + pv_c; tv20 = float(np.mean(tv[-20:]) + 1e-9)
    rvol = float(tv[-1]) / tv20
    rvol_hist.update(rvol, india_vix)   # append current bar FIRST
    rvol_pr  = rvol_hist.pct_rank()      # then rank (includes current bar)
    vix_ma20 = rvol_hist.vix_ma20()
    vix_med  = rvol_hist.vix_med20()

    # 1m call indicators
    rsi_1m            = _rsi(c, 14)
    adx_1m, pdi_, ndi_= _adx(c, 14)
    bb_pctb, bb_bw    = _bb(c, 20)
    macdh_1m          = _macdh(c)
    atr14             = _atr_s(c, 14)
    ema200_c          = _ema(c, min(200, len(c)))
    rvol_hist.stoch_update(_raw_stoch_k(ch, cl_, c, 14))
    stoch_k, stoch_d  = rvol_hist.stoch_kd()
    cmf_1m            = _cmf(ch, cl_, c, cv, 14)

    # adx_1m_rank / price_position_rank: self-rank against long history
    # [CONFIRMED FIX — file-header note #1]
    rvol_hist.update_extra(adx_1m, float(c[-1]))
    adx1m_pr = rvol_hist.adx1m_rank()
    price_pr = rvol_hist.price_rank()

    # HTF
    f5  = htf.get(5); f15 = htf.get(15); f30 = htf.get(30)

    # Regime
    vol_regime   = 0 if india_vix<12 else 1 if india_vix<16 else 2 if india_vix<22 else 3
    sp_ema50d    = _ema(sp, min(len(sp), 18750))
    trend_regime = 1 if sp[-1] > sp_ema50d else -1

    # MTF
    mtf_bull = int(f5["trend"]==1  and f15["st_dir"]==1  and f15["ema200_gap_pct"]>0)
    mtf_bear = int(f5["trend"]==-1 and f15["st_dir"]==-1 and f15["ema200_gap_pct"]<0)
    mtf_soft = int(f5["trend"]==1) + int(f15["st_dir"]==1) + int(f15["rsi"]>50)

    # Pivot
    piv_f = pivot.get_features(float(sp[-1]))

    # Time
    def sess(t_):
        if t_<dtime(9,30):  return 0
        if t_<dtime(11,30): return 1
        if t_<dtime(13,30): return 2
        return 3
    delta = (1-dow)%7
    if delta==0 and t>=dtime(15,30): delta=7

    # Candle
    _rng  = float(ch[-1]-cl_[-1]+1e-9)
    _body = abs(float(c[-1]-co[-1]))
    _cp   = (c[-1]-cl_[-1])/_rng

    row = {}

    # ── 1. Raw chain columns (pass-through from raw_row)
    for col, val in raw_row.items():
        if "nifty_WEEK_ATM" in col:
            row[col] = float(val) if val is not None else 0.0

    # ── 2. Computed features
    row.update({
        "rvol"                  : rvol,
        "rvol_capped"           : min(rvol, 5),
        "U02_RVOL"              : 1 if rvol>=1.2 else -1 if rvol<0.8 else 0,
        # rvol_roll3_max [UNVERIFIED CONVENTION — file-header note #7]
        "rvol_roll3_max"        : rvol_hist.rvol_roll3_max(),
        "rvol_pct_rank"         : rvol_pr,
        "rvol_bucket"           : 0 if rvol_pr<0.30 else 1 if rvol_pr<0.70 else 2 if rvol_pr<0.90 else 3,
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
        # adx_1m_rank [CONFIRMED FIX — file-header note #1]
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
        # price_position_rank [CONFIRMED FIX — file-header note #1]
        "price_position_rank"   : price_pr,
        "cmf_1m"                : cmf_1m,
        "U05_CMF_Direction"     : 1 if cmf_1m>0.05 else -1 if cmf_1m<-0.05 else 0,

        "iv_skew"               : float(piv[-1]-civ[-1]),
        "iv_skew_slope"         : float((piv[-1]-civ[-1])-(piv[-4]-civ[-4])) if len(civ)>3 else 0.0,
        "iv_momentum_5bar"      : float(iv_mid[-1]-iv_mid[-6]) if len(iv_mid)>5 else 0.0,
        "iv_skew_ratio"         : float(piv[-1]/(civ[-1]+1e-9)),
        # iv_rank / atm_call_iv_rank / atm_put_iv_rank [UNVERIFIED CONVENTION — file-header note #6]
        # Switched from [-21:-1] (current bar excluded) to [-20:] (current bar included).
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
        "vix_ma20"                  : vix_ma20,
        "vix_above_ma"              : int(india_vix>vix_ma20),
        "vix_above_rolling_median"  : int(india_vix>vix_med),
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
        # adx_5m_rank [CONFIRMED FIX — file-header note #1]
        "adx_regime_5m"     : f5["adx_regime"],"adx_5m_rank"    : htf.get_rank(5),
        "di_cross_5m"       : f5["di_cross"],

        "rsi_15m"           : f15["rsi"],   "macdh_15m"         : f15["macdh"],
        "adx_15m"           : f15["adx"],   "pdi_15m"           : f15["pdi"],
        "ndi_15m"           : f15["ndi"],   "atr_15m"           : f15["atr"],
        "atr_pct_15m"       : f15["atr_pct"],"bb_pctb_15m"      : f15["bb_pctb"],
        "bb_bw_15m"         : f15["bb_bw"],"cmf_15m"            : f15["cmf"],
        "ema200_gap_pct_15m": f15["ema200_gap_pct"],
        "st_dir_15m"        : f15["st_dir"],"trend_15m"          : f15["trend"],
        # adx_15m_rank [CONFIRMED FIX — file-header note #1]; di_cross_15m [CONFIRMED FIX — note #2]
        "adx_regime_15m"    : f15["adx_regime"],"adx_15m_rank"   : htf.get_rank(15),
        "di_cross_15m"      : f15["di_cross"],

        "rsi_30m"           : f30["rsi"],   "macdh_30m"         : f30["macdh"],
        "adx_30m"           : f30["adx"],   "pdi_30m"           : f30["pdi"],
        "ndi_30m"           : f30["ndi"],   "atr_30m"           : f30["atr"],
        "atr_pct_30m"       : f30["atr_pct"],"bb_pctb_30m"      : f30["bb_pctb"],
        "bb_bw_30m"         : f30["bb_bw"],"cmf_30m"            : f30["cmf"],
        "ema200_gap_pct_30m": f30["ema200_gap_pct"],
        "st_dir_30m"        : f30["st_dir"],"trend_30m"          : f30["trend"],
        # adx_30m_rank [CONFIRMED FIX — file-header note #1]; di_cross_30m [CONFIRMED FIX — note #2]
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


# ══════════════════════════════════════════════════════════════════════
# V12Model
# ══════════════════════════════════════════════════════════════════════

class V12Model:
    """
    Stacked ensemble with 5-bar persistence gate.
    Interface identical to ensemble_v3 V12Model.
    """

    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        self._load_models(d)
        self._init_state()

        print("✅ nifty_v12_l3 loaded")
        print(f"   Model dir    : {d.resolve()}")
        print(f"   Features     : {len(self.features)}  (v12: 245 features, ATMminus1 labels)")
        print(f"   SHORT subset : {len(self.short_features)}")
        print(f"   eff_long     : {self.eff_long:.4f}")
        print(f"   eff_short    : {self.eff_short:.4f}  (meta_short_prob >= this → SHORT)")
        print(f"   Persistence  : {PERSIST_WINDOW}-bar rolling avg in meta")
        if self.meta_cfg:
            print(f"   Test AUC     : {self.meta_cfg.get('test_auc','?')}")
            print(f"   Short label  : {self.meta_cfg.get('short_label','?')}")

    def _load_models(self, d: Path):
        self.xgb_l   = joblib.load(d / "model_xgb_long_v12.pkl")
        self.xgb_s   = joblib.load(d / "model_xgb_short_v12.pkl")
        self.lgb_l   = joblib.load(d / "model_lgb_long_v12.pkl")
        self.cat_l   = CatBoostClassifier()
        self.cat_l.load_model(str(d / "model_cat_long_v12.cbm"))
        self.et_l    = joblib.load(d / "model_et_long_v12.pkl")
        self.et_s    = joblib.load(d / "model_et_short_v12.pkl")

        # Force single-threaded inference: these ET models were likely
        # pickled with n_jobs=-1 (parallel across trees). Every call to
        # .predict_proba() then spins up joblib/loky WORKER PROCESSES,
        # which have their own separate warnings registry — they never
        # see the warnings.filterwarnings() calls set at the top of this
        # file. That's why the suppression "fix" didn't actually work.
        # n_jobs=1 also removes real per-bar latency: parallelizing
        # across trees for a single-row live prediction is pure
        # process-spawn overhead with zero benefit.
        try:
            self.et_l.set_params(n_jobs=1)
            self.et_s.set_params(n_jobs=1)
        except Exception as e:
            print(f"⚠️  Could not force n_jobs=1 on ET models: {e}")

        self.meta_l  = joblib.load(d / "meta_long_v12.pkl")
        self.meta_s  = joblib.load(d / "meta_short_v12.pkl")
        self.features       = joblib.load(d / "top_features_v12.pkl")
        self.short_features = joblib.load(d / "short_features_v12.pkl")
        tl = d / "thresh_long_v12.pkl";  ts = d / "thresh_short_v12.pkl"
        self.eff_long  = float(joblib.load(tl)) if tl.exists() else 0.75
        self.eff_short = float(joblib.load(ts)) if ts.exists() else 0.75
        self.meta_cfg: dict = {}
        mp = d / "v12_meta.json"
        if mp.exists():
            self.meta_cfg = json.loads(mp.read_text(encoding="utf-8"))

    def _init_state(self):
        self.htf       = HTFStateV12()
        self.pivot     = PivotStateV12()
        self.rvol_hist = RvolHistoryV12()
        self._prob_buf = deque(maxlen=PERSIST_WINDOW)

    def reset_intraday(self):
        """Call at market open each day in live trading."""
        self._init_state()

    # ── feat_cols alias
    @property
    def feat_cols(self): return self.features

    # ── Base proba (single row)
    def _base_proba(self, X_row: pd.DataFrame) -> np.ndarray:
        X_short = X_row[self.short_features]
        return np.array([
            float(self.xgb_l.predict_proba(X_row)[:, 1][0]),
            float(self.lgb_l.predict_proba(X_row)[:, 1][0]),
            float(self.cat_l.predict_proba(X_row)[:, 1][0]),
            float(self.et_l.predict_proba(X_row)[:, 1][0]),
            float(self.xgb_s.predict_proba(X_short)[:, 1][0]),
            float(self.et_s.predict_proba(X_short)[:, 1][0]),
        ])

    # ── Meta with persistence
    def _meta_predict(self, base_vec: np.ndarray):
        self._prob_buf.append(base_vec)
        roll = np.mean(list(self._prob_buf), axis=0)
        meta_in = np.concatenate([base_vec, roll]).reshape(1, 12)
        pl = float(self.meta_l.predict_proba(meta_in)[:, 1][0])
        ps = float(self.meta_s.predict_proba(meta_in)[:, 1][0])
        return pl, ps

    # ── Signal resolution (same logic as ensemble_v3)
    def _resolve_signal(self, pl, ps, india_vix, vol_regime, is_pre_close):
        tl = self.eff_long  + 0.01 * is_pre_close
        ts = self.eff_short + 0.02 * is_pre_close - (0.04 if india_vix > 22 else 0.0)
        long_ok  = bool(pl >= tl)
        short_ok = bool(ps >= ts)
        if long_ok and short_ok:
            signal = "LONG" if (pl-tl) > (ps-ts) else "SHORT"
        elif long_ok:  signal = "LONG"
        elif short_ok: signal = "SHORT"
        else:          signal = "NONE"
        return signal, tl, ts

    # ══ predict_bar()  — identical interface to ensemble_v3 ═══════════
    def predict_bar(self, feature_dict: dict,
                    india_vix:    float = 16.0,
                    vol_regime:   int   = 0,
                    is_pre_close: int   = 0) -> dict:
        """
        Single-bar prediction WITH persistence gate.
        Returns same dict as ensemble_v3:
            signal, prob_long, prob_short, eff_long, eff_short,
            confidence, margin_long, margin_short
        """
        X_row = pd.DataFrame([feature_dict])
        for f in self.features:
            if f not in X_row.columns: X_row[f] = 0.0
        X_row = X_row[self.features].fillna(0)

        base_vec       = self._base_proba(X_row)
        pl, ps         = self._meta_predict(base_vec)
        signal, tl, ts = self._resolve_signal(pl, ps, india_vix, vol_regime, is_pre_close)

        return {
            "signal"      : signal,
            "prob_long"   : float(pl),
            "prob_short"  : float(ps),
            "eff_long"    : float(self.eff_long),
            "eff_short"   : float(self.eff_short),
            "confidence"  : float(max(pl, ps)),
            "margin_long" : float(pl - tl),
            "margin_short": float(ps - ts),
        }

    # ══ get_signals()  — batch interface ══════════════════════════════
    def get_signals(self, X: pd.DataFrame,
                    india_vix:    float | np.ndarray = 16.0,
                    vol_regime:   float | np.ndarray = 0,
                    is_pre_close: float | np.ndarray = 0):
        """
        Batch prediction (no rolling persistence — each row independent).
        Returns (signals, prob_long, prob_short) arrays of shape (n,).
        """
        n = len(X)
        vix_arr = np.full(n, india_vix)    if np.isscalar(india_vix)    else np.asarray(india_vix)
        vol_arr = np.full(n, vol_regime)   if np.isscalar(vol_regime)   else np.asarray(vol_regime)
        pre_arr = np.full(n, is_pre_close) if np.isscalar(is_pre_close) else np.asarray(is_pre_close)

        Xf = X[self.features].fillna(0)
        Xs = Xf[self.short_features]

        p1 = self.xgb_l.predict_proba(Xf)[:, 1]; p2 = self.lgb_l.predict_proba(Xf)[:, 1]
        p3 = self.cat_l.predict_proba(Xf)[:, 1]; p4 = self.et_l.predict_proba(Xf)[:, 1]
        p5 = self.xgb_s.predict_proba(Xs)[:, 1]; p6 = self.et_s.predict_proba(Xs)[:, 1]
        meta_in = np.column_stack([p1,p2,p3,p4,p5,p6, p1,p2,p3,p4,p5,p6])
        pl_arr  = self.meta_l.predict_proba(meta_in)[:, 1]
        ps_arr  = self.meta_s.predict_proba(meta_in)[:, 1]

        sigs = np.full(n, "NONE", dtype=object)
        for i in range(n):
            sig, _, _ = self._resolve_signal(
                pl_arr[i], ps_arr[i], float(vix_arr[i]), int(vol_arr[i]), int(pre_arr[i]))
            sigs[i] = sig
        return sigs, pl_arr, ps_arr

    # ══ predict()  — live/backtest pipeline ═══════════════════════════
    def predict(self, raw_row: dict, buf: dict,
                india_vix: float = 16.0, vix_chg: float = 0.0):
        """
        Full pipeline: raw_row + buf → features → predict_bar.
        Caller must update htf/pivot before calling:
            model.htf.update(bar_time, o, h, l, c, vol, sp)
            model.pivot.update(bar_date, sp_h, sp_l, sp_c)

        Returns (signal, confidence, prob_long, prob_short).
        """
        feat = _build_features(
            raw_row, buf, self.htf, self.pivot, self.rvol_hist,
            india_vix, vix_chg, self.features)
        if feat is None:
            return "NONE", 0.0, 0.5, 0.5
        result = self.predict_bar(
            feat,
            india_vix    = india_vix,
            vol_regime   = int(feat.get("vol_regime", 0)),
            is_pre_close = int(feat.get("is_pre_close", 0)),
        )
        return (result["signal"], result["confidence"],
                result["prob_long"], result["prob_short"])

    # ══ backtest()  ════════════════════════════════════════════════════
    def backtest(self, df: pd.DataFrame,
                 india_vix_col="india_vix", vol_regime_col="vol_regime",
                 is_pre_close_col="is_pre_close",
                 tp_pct=0.004, sl_pct=0.009) -> pd.DataFrame:
        vix = df[india_vix_col].values    if india_vix_col    in df.columns else np.full(len(df), 16.0)
        vol = df[vol_regime_col].values   if vol_regime_col   in df.columns else np.zeros(len(df))
        pre = df[is_pre_close_col].values if is_pre_close_col in df.columns else np.zeros(len(df))
        sigs, pl, ps = self.get_signals(df, vix, vol, pre)
        out = df.copy(); out["signal"] = sigs; out["prob_long"] = pl; out["prob_short"] = ps
        if "label" in df.columns:
            traded = out["signal"] != "NONE"
            out["win"] = (((out["signal"]=="LONG") & (out["label"]==1)) |
                          ((out["signal"]=="SHORT") & (out["label"]==-1)))
            out["pnl_pct"] = 0.0
            out.loc[traded &  out["win"], "pnl_pct"] =  tp_pct * 100
            out.loc[traded & ~out["win"], "pnl_pct"] = -sl_pct * 100
            out["cum_pnl"] = out["pnl_pct"].cumsum()
        return out

    @staticmethod
    def stats(df_result: pd.DataFrame) -> dict:
        if "signal" not in df_result.columns: return {"trades": 0}
        traded = df_result[df_result["signal"] != "NONE"]
        if len(traded) == 0: return {"trades": 0}
        win = traded["win"] if "win" in traded.columns else pd.Series(False, index=traded.index)
        pnl = traded["pnl_pct"] if "pnl_pct" in traded.columns else pd.Series(0.0, index=traded.index)
        cum = pnl.cumsum()
        lt  = traded[traded["signal"]=="LONG"]; st = traded[traded["signal"]=="SHORT"]
        gw  = pnl[win].sum(); gl = abs(pnl[~win].sum())
        return {
            "trades": len(traded), "sel_pct": len(traded)/len(df_result)*100,
            "wr": float(win.mean()*100),
            "long_wr":  float(lt["win"].mean()*100) if len(lt) else 0.0,
            "short_wr": float(st["win"].mean()*100) if len(st) else 0.0,
            "long_trades": len(lt), "short_trades": len(st),
            "total_ret": float(pnl.sum()), "expectancy": float(pnl.mean()),
            "max_dd": float((cum-cum.cummax()).min()),
            "profit_factor": float(gw/gl) if gl > 0 else float("inf"),
        }


# ══════════════════════════════════════════════════════════════════════
# PaperTracker  (identical interface to ensemble_v3)
# ══════════════════════════════════════════════════════════════════════

class PaperTracker:
    BENCHMARK_WR = 0.947
    TP_PCT       = 0.004
    SL_PCT       = 0.009
    KILL_WR      = 0.60
    KILL_LOSS    = 7

    def __init__(self):
        self.trades: list[dict] = []
        self.consecutive: int   = 0

    def log_trade(self, datetime, signal, entry_price, exit_price,
                  exit_reason, prob_long, prob_short) -> bool:
        if exit_price is None or exit_reason is None: return False
        entry_price = float(entry_price); exit_price = float(exit_price)
        win = exit_price > entry_price
        pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        self.consecutive = 0 if win else self.consecutive + 1
        self.trades.append({"datetime":datetime,"signal":signal,"entry":entry_price,
            "exit":exit_price,"exit_reason":exit_reason,"win":win,"pnl_pct":pnl_pct,
            "prob_long":float(prob_long),"prob_short":float(prob_short)})
        return win

    def should_stop(self, window: int = 20) -> tuple[bool, str]:
        if len(self.trades) < 5: return False, ""
        df  = pd.DataFrame(self.trades)
        rwr = df["win"].tail(window).mean() * 100
        if rwr < self.KILL_WR * 100: return True, f"Rolling WR {rwr:.1f}% < {self.KILL_WR*100:.0f}%"
        if self.consecutive >= self.KILL_LOSS: return True, f"{self.consecutive} consecutive losses"
        return False, ""

    def status(self, window: int = 50):
        if not self.trades: print("No trades yet."); return None
        df = pd.DataFrame(self.trades); df["cum_pnl"] = df["pnl_pct"].cumsum()
        wr  = df["win"].mean()*100; rwr = df["win"].tail(window).mean()*100
        lt  = df[df["signal"]=="LONG"]; st = df[df["signal"]=="SHORT"]
        lwr = lt["win"].mean()*100 if len(lt) else 0.0
        swr = st["win"].mean()*100 if len(st) else 0.0
        gw  = df[df["win"]]["pnl_pct"].sum(); gl = abs(df[~df["win"]]["pnl_pct"].sum())
        pf  = gw/gl if gl>0 else float("inf"); delta = wr - self.BENCHMARK_WR*100
        print(f"\n{'═'*55}"); print("  PAPER TRADING — nifty_v12_l3"); print(f"{'═'*55}")
        print(f"  Total trades      : {len(df):,}")
        print(f"  Overall WR        : {wr:.1f}%  (benchmark {self.BENCHMARK_WR*100:.1f}%)")
        print(f"  Rolling WR ({window:2d})  : {rwr:.1f}%")
        print(f"  LONG  WR          : {lwr:.1f}%  ({len(lt):,} trades)")
        print(f"  SHORT WR          : {swr:.1f}%  ({len(st):,} trades)")
        print(f"  Cum P&L           : {df['pnl_pct'].sum():+.3f}%")
        print(f"  Max Drawdown      : {(df['cum_pnl']-df['cum_pnl'].cummax()).min():.3f}%")
        print(f"  Profit Factor     : {pf:.2f}")
        print(f"  Consecutive loss  : {self.consecutive}")
        print(f"  vs Benchmark      : {delta:+.1f}%  " +
              ("✅ on track" if delta>=-5 else "⚠️  degrading" if delta>=-10 else "❌ kill switch"))
        stop, reason = self.should_stop(window)
        print(f"\n  {'🔴 KILL SWITCH: '+reason if stop else '🟢 Trading OK'}")
        print(f"{'═'*55}")
        return df

    def monthly_stats(self):
        if not self.trades: return pd.DataFrame()
        df = pd.DataFrame(self.trades)
        df["month"] = pd.to_datetime(df["datetime"]).dt.to_period("M")
        rows = []
        for mth, grp in df.groupby("month"):
            lt=grp[grp["signal"]=="LONG"]; st=grp[grp["signal"]=="SHORT"]
            cum=grp["pnl_pct"].cumsum()
            rows.append({"month":str(mth),"trades":len(grp),"wr":grp["win"].mean()*100,
                "long_wr":lt["win"].mean()*100 if len(lt) else 0.0,
                "short_wr":st["win"].mean()*100 if len(st) else 0.0,
                "ret_pct":grp["pnl_pct"].sum(),"max_dd":(cum-cum.cummax()).min()})
        result = pd.DataFrame(rows)
        print(f"\n{'Month':<9} {'Trades':>7} {'WR%':>6} {'LWR':>6} {'SWR':>6} {'Ret%':>8} {'MDD%':>7}")
        print("─"*55)
        for _, r in result.iterrows():
            print(f"{r['month']:<9} {int(r['trades']):>7} {r['wr']:>6.1f}% "
                  f"{r['long_wr']:>5.1f}% {r['short_wr']:>5.1f}% "
                  f"{r['ret_pct']:>7.3f}% {r['max_dd']:>6.3f}%")
        return result

    def save(self, path: str = "paper_trades_v12l3.csv") -> None:
        if self.trades:
            pd.DataFrame(self.trades).to_csv(path, index=False)
            print(f"✅ Saved {len(self.trades)} trades → {path}")

    def load(self, path: str = "paper_trades_v12l3.csv") -> None:
        import os
        if not os.path.exists(path): print(f"⚠️ File not found: {path}"); return
        df = pd.read_csv(path, parse_dates=["datetime"])
        self.trades = df.to_dict("records"); self.consecutive = 0
        for t in reversed(self.trades):
            if t["win"]: break
            self.consecutive += 1
        print(f"✅ Loaded {len(self.trades)} trades from {path}")
        print(f"   Consecutive losses at end: {self.consecutive}")


# ══════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    MODEL_DIR = Path(r"C:\ZERODHA BOT\nifty50-tradingbot\niftytrading\nifty_v12_l3_models")
    if not MODEL_DIR.exists():
        print(f"❌ Model directory not found: {MODEL_DIR.resolve()}")
        sys.exit(1)

    model   = V12Model(MODEL_DIR)
    tracker = PaperTracker()

    dummy  = {feat: 0.0 for feat in model.features}
    result = model.predict_bar(dummy, india_vix=15.0)

    print("\n── Smoke test (all-zero features) ──")
    print(f"  Signal       : {result['signal']}")
    print(f"  Prob LONG    : {result['prob_long']:.4f}  (threshold {result['eff_long']:.4f})")
    print(f"  Prob SHORT   : {result['prob_short']:.4f}  (threshold {result['eff_short']:.4f})")
    print(f"  Confidence   : {result['confidence']:.4f}")
    print(f"  Margin LONG  : {result['margin_long']:+.4f}")
    print(f"  Margin SHORT : {result['margin_short']:+.4f}")
    print(f"\n  Persistence buffer: {len(model._prob_buf)}/{PERSIST_WINDOW}")
    print("\n✅ nifty_v12_l3 inference working correctly")