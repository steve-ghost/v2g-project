#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# -----------------------------
# 상수
# -----------------------------
EMISSION_FACTOR_KG_PER_KWH = 0.495   # 국내 전력 1kWh 생산시 약 0.495 kgCO2e
MJ_PER_M2_TO_KWH_PER_M2     = 0.27778 # MJ/m² → kWh/m²

# =========================
# 1) 한글 폰트
# =========================
def set_korean_font():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, "NanumGothic.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# =========================
# 2) 재무 유틸
# =========================
def price_with_cagr(base_price, base_year, year, cagr):
    return base_price * (1 + cagr) ** (year - base_year)

def npv(rate: float, cashflows: list[float]) -> float:
    if rate <= -0.999999:
        return float("nan")
    return float(sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows)))

def irr_bisection(cashflows: list[float], lo=-0.99, hi=3.0, tol=1e-7, max_iter=200):
    def f(r):
        try: return npv(r, cashflows)
        except: return np.nan
    f_lo, f_hi = f(lo), f(hi)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if np.isnan(f_mid): return None
        if abs(f_mid) < tol: return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return mid

def discounted_payback(cashflows: list[float], rate: float):
    disc = []
    cum = 0.0
    for t, cf in enumerate(cashflows):
        val = cf / ((1 + rate) ** t)
        cum += val
        disc.append(cum)
    for k, v in enumerate(disc):
        if v >= 0:
            if k == 0: return 0.0
            prev = disc[k - 1]
            frac = 0.0 if v == prev else (-prev) / (v - prev)
            return (k - 1) + max(0.0, min(1.0, frac))
    return None

def won_formatter(x, pos):
    return f"{int(x):,}"

# =========================
# 3) CSV → 연도별 PV kWh 계산 (일사합)
# =========================
def load_irradiance_csv(csv_path: str) -> pd.DataFrame:
    """
    연도별 '일사합(MJ/m²)' CSV를 읽어 표준 컬럼으로 정리:
      - year        : ['연도','year','일시'] 중 하나
      - ghi_mj_m2   : ['일사합(mj/m2)','일사합(mj/m²)','ghi_mj','ghi(mj/m2)' ...]
    반환 DF: ['year','ghi_mj_m2']
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV가 없음: {csv_path}")
    tried = []
    for enc in ["utf-8-sig", "cp949"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            tried.append((enc, str(e)))
            df = None
    if df is None:
        raise ValueError(f"CSV 읽기 실패: {tried}")

    df.columns = [str(c).strip().lower() for c in df.columns]

    year_candidates = ["연도", "year", "일시"]
    year_col = next((c for c in year_candidates if c in df.columns), None)
    if year_col is None:
        raise ValueError(f"[연도 컬럼 없음] 실제 열: {list(df.columns)}")

    ghi_candidates = [
        "일사합(mj/m2)", "일사합(mj/m²)", "ghi_mj", "ghi(mj/m2)", "ghi(mj/m²)",
        "solar_mj", "solar(mj/m2)"
    ]
    ghi_col = next((c for c in ghi_candidates if c in df.columns), None)
    if ghi_col is None:
        maybe = [c for c in df.columns if ("mj" in c and "m" in c)]
        raise ValueError(f"[일사합 컬럼 없음] 실제 열: {list(df.columns)} / 유사: {maybe}")

    out = pd.DataFrame({
        "year": pd.to_numeric(df[year_col], errors="coerce"),
        "ghi_mj_m2": pd.to_numeric(df[ghi_col], errors="coerce")
    }).dropna()
    out["year"] = out["year"].astype(int)
    return out.sort_values("year").reset_index(drop=True)

def compute_pv_kwh_by_year(irr_df: pd.DataFrame,
                           panel_width_m=1.46,
                           panel_height_m=0.98,
                           n_modules=250,
                           pr_base=0.82, availability=0.98, soiling=0.02, inv_eff=0.97,
                           pr_manual=None):
    """
    PV 연간 발전량(kWh) = (일사합 MJ/m² × 0.27778) × 총면적(m²) × PR
    pr_manual 지정 시 그 값 사용, 아니면 고정 손실계수로 PR 산정.
    반환: (dict{year:kWh}, area_m2, PR)
    """
    area_m2 = float(panel_width_m) * float(panel_height_m) * int(n_modules)
    if pr_manual is None:
        PR = pr_base * availability * (1.0 - soiling) * inv_eff
    else:
        PR = float(pr_manual)

    ghi_kwh_m2 = irr_df["ghi_mj_m2"].astype(float) * MJ_PER_M2_TO_KWH_PER_M2
    pv_kwh = ghi_kwh_m2 * area_m2 * PR
    out = dict(zip(irr_df["year"].astype(int).tolist(), pv_kwh.astype(float).tolist()))
    return out, area_m2, PR

# =========================
# 4) SMP(시간대별 단가) 처리
# =========================
def load_smp_series(csv_path: str) -> pd.Series:
    """
    SMP.csv (기간, 01시~24시, 최대, 최소, 가중평균 형식)를 읽어
    시간별 SMP 시리즈(원/kWh)로 변환.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SMP CSV가 없음: {csv_path}")

    # 인코딩 탐색
    for enc in ["utf-8-sig", "cp949"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError("SMP CSV 읽기 실패")

    # 열 이름 정리
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if "기간" in c or "date" in c.lower()), None)
    if date_col is None:
        raise ValueError("‘기간’ 또는 날짜 열을 찾을 수 없습니다.")

    # melt로 wide → long 변환
    hour_cols = [c for c in df.columns if any(str(i).zfill(2) in c for i in range(1,25))]
    df_long = df.melt(id_vars=[date_col], value_vars=hour_cols,
                      var_name="hour", value_name="price")

    # 날짜 + 시각 결합
    def parse_datetime(row):
        h = str(row["hour"]).replace("시", "").strip()
        try:
            h_int = int(h)
        except:
            return pd.NaT
        return pd.to_datetime(f"{row[date_col]} {h_int:02d}:00", errors="coerce")

    df_long["dt"] = df_long.apply(parse_datetime, axis=1)
    df_long["price"] = pd.to_numeric(df_long["price"], errors="coerce")

    df_long = df_long.dropna(subset=["dt", "price"])
    df_long = df_long.sort_values("dt")

    s = df_long.set_index("dt")["price"]

    # 리샘플링/보간/타임존
    s = s.resample("1H").mean().interpolate("time").ffill().bfill()
    if s.index.tz is None:
        s.index = s.index.tz_localize("Asia/Seoul", nonexistent="shift_forward", ambiguous="NaT")

    return s  # Series(dt → SMP[원/kWh])


def escalate_series_by_cagr(base_series: pd.Series, base_year: int, year: int, cagr: float) -> pd.Series:
    """대표연도 시계열을 해당 연도로 스케일(가격 × (1+CAGR)^(Δt))"""
    factor = (1.0 + cagr) ** (year - base_year)
    s = base_series.copy()
    # 연도만 교체(월/일/시 유지)
    s.index = s.index.map(lambda t: t.replace(year=year))
    return s * factor

# =========================
# 5) 시간 분해: PV·V2G 시리즈 만들기
# =========================
def build_pv_hourly_series(year: int, annual_pv_kwh: float) -> pd.Series:
    """연간 kWh를 1시간 해상도로 분배(간단 월×시간 가중치)"""
    idx = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="1H", tz="Asia/Seoul", inclusive="left")
    df = pd.DataFrame(index=idx)

    # 월별 가중치(예시). 필요하면 실제 월별 일사/발전 비중으로 교체
    month_weights = np.array([0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.11,0.10,0.08,0.05,0.03])
    month_weights = month_weights / month_weights.sum()
    df["m_w"] = [month_weights[t.month-1] for t in df.index]

    # 일중 가중치: 7~18시만 발전(정오 피크)
    hour = df.index.hour
    daytime = ((hour>=7)&(hour<=18)).astype(int)
    t = (hour-7) / (18-7)
    shape = np.where(daytime==1, np.sin(np.pi * t), 0.0)
    shape[shape<0]=0.0
    df["h_w"] = shape

    w = df["m_w"] * df["h_w"]
    if w.sum() == 0:
        return pd.Series(0.0, index=idx)
    return (w / w.sum()) * annual_pv_kwh

def pv_export_series(pv_hourly_kwh: pd.Series, self_use_ratio: float) -> pd.Series:
    return pv_hourly_kwh * (1.0 - self_use_ratio)

def build_v2g_hourly_series(year: int,
                            num_chargers: int,
                            kwh_per_charger_day: float,
                            operating_days: int,
                            degradation: float,
                            discharge_hours=(17,18,19,20,21),
                            price_weighted: bool=False,
                            smp_for_year: pd.Series|None=None) -> pd.Series:
    """
    하루 방전량 = (#충전기 × 1대당 일일방전량 × 열화계수).
    기본은 선택한 시간대에 균등 분배.
    price_weighted=True면 같은 날 SMP 비례로 분배.
    """
    idx = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="1H", tz="Asia/Seoul", inclusive="left")
    s = pd.Series(0.0, index=idx)

    E_day = num_chargers * kwh_per_charger_day * degradation
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D", tz="Asia/Seoul")

    # 단순히 앞에서부터 operating_days일만 운전(원하면 평일만/무작위 등으로 바꿔도 됨)
    ops_mask = pd.Series(0, index=days)
    ops_mask.iloc[:min(operating_days, len(ops_mask))] = 1

    H = len(discharge_hours)
    for day, active in ops_mask.items():
        if active != 1 or H == 0:
            continue

        hours_ts = [day + pd.Timedelta(hours=h) for h in discharge_hours]

        if price_weighted and smp_for_year is not None:
            # 해당 일자 SMP 합으로 비례 분배
            prices = []
            for t in hours_ts:
                if t in smp_for_year.index:
                    prices.append(float(smp_for_year.loc[t]))
                else:
                    # 존재 안 하면 가까운 값 사용
                    nearest = smp_for_year.index.get_indexer([t], method="nearest")
                    prices.append(float(smp_for_year.iloc[nearest[0]]))
            prices = np.array(prices, dtype=float)
            weights = prices / prices.sum() if prices.sum() > 0 else np.ones_like(prices)/len(prices)
        else:
            weights = np.ones(H) / H

        for t, w in zip(hours_ts, weights):
            # 인덱스 존재 보정
            if t not in s.index:
                # 근접 시간으로 매핑
                nearest = s.index.get_indexer([t], method="nearest")
                t = s.index[nearest[0]]
            s.loc[t] += E_day * float(w)

    return s

def revenue_from_smp(smp_price: pd.Series,
                     pv_export: pd.Series,
                     v2g_discharge: pd.Series) -> tuple[float, float]:
    """(총수익, PV수익만) 반환. 둘 다 원 단위."""
    s = smp_price.astype(float)
    pv = pv_export.reindex(s.index, method="nearest").fillna(0.0)
    v2g = v2g_discharge.reindex(s.index, method="nearest").fillna(0.0)
    pv_rev   = float((pv  * s).sum())
    total_rev= float(((pv+v2g) * s).sum())
    return total_rev, pv_rev

# =========================
# 6) 기본 파라미터
# =========================
def make_v2g_model_params():
    return {
        "self_use_ratio": 0.60,           # 자가소비 비율
        # V2G
        "num_v2g_chargers": 6,
        "v2g_charger_unit_cost": 25_000_000,
        "v2g_daily_discharge_per_charger_kwh": 35,
        "degradation_factor": 0.9,
        "v2g_operating_days": 300,
        # 단가(연복리 상승; SMP를 쓰면 SMP에 적용)
        "tariff_base_year": 2025,
        "pv_base_price": 160,             # (미사용: SMP 사용 시)
        "v2g_price_gap": 30,              # (미사용: SMP 사용 시)
        "price_cagr": 0.043,
        # O&M
        "om_ratio": 0.015,
    }

# =========================
# 7) 현금흐름 빌드
#   - SMP가 있으면 시간대별로 수익 산출
#   - 없으면 기존 단가(고정/연상승) 방식으로 계산
# =========================
def build_yearly_cashflows_from_csv(install_year: int, current_year: int, p: dict,
                                    pv_kwh_by_year: dict,
                                    smp_base_series: pd.Series|None = None,
                                    smp_base_year: int|None = None,
                                    use_price_weighted_v2g: bool = True):
    # 공통 비용요소
    capex_total   = p["num_v2g_chargers"] * p["v2g_charger_unit_cost"]
    annual_om_cost= capex_total * p["om_ratio"]
    self_use      = float(p["self_use_ratio"])

    years = list(range(install_year, current_year + 1))
    yearly_cash, cumulative = [], []
    pv_revenues, v2g_revenues, om_costs, capex_list = [], [], [], []

    if len(pv_kwh_by_year) == 0:
        raise ValueError("CSV에서 계산된 PV 연간 발전량이 없습니다.")
    min_y, max_y = min(pv_kwh_by_year.keys()), max(pv_kwh_by_year.keys())

    cum = 0.0

    # V2G 연간 kWh “상한”은 동일하지만, 시간대 배치에 따라 '수익'이 달라짐.
    # 시간단 SMP가 없다면 기존 평균단가 방식으로 계산.
    for i, year in enumerate(years):
        # PV 연간 kWh (범위 밖은 경계값)
        y_key = min(max(year, min_y), max_y)
        annual_pv_kwh = pv_kwh_by_year[y_key]
        # 연간 PV → 시간분해 → 자가소비 반영
        pv_hourly = build_pv_hourly_series(year, annual_pv_kwh)
        pv_export = pv_export_series(pv_hourly, self_use)

        # V2G 시간 분해
        if smp_base_series is not None and smp_base_year is not None:
            smp_y = escalate_series_by_cagr(smp_base_series, smp_base_year, year, p["price_cagr"])
            v2g_hourly = build_v2g_hourly_series(
                year,
                p["num_v2g_chargers"],
                p["v2g_daily_discharge_per_charger_kwh"],
                p["v2g_operating_days"],
                p["degradation_factor"],
                price_weighted=use_price_weighted_v2g,
                smp_for_year=smp_y if use_price_weighted_v2g else None
            )
            total_rev_y, pv_rev_y = revenue_from_smp(smp_y, pv_export, v2g_hourly)
            v2g_rev_y = total_rev_y - pv_rev_y
        else:
            # (백업) 평균단가 방식
            annual_pv_surplus_kwh = annual_pv_kwh * (1 - self_use)
            daily_v2g_kwh = p["num_v2g_chargers"] * p["v2g_daily_discharge_per_charger_kwh"]
            annual_v2g_kwh = daily_v2g_kwh * p["v2g_operating_days"] * p["degradation_factor"]

            pv_price_y  = price_with_cagr(p["pv_base_price"], p["tariff_base_year"], year, p["price_cagr"])
            v2g_price_y = pv_price_y + p["v2g_price_gap"]

            pv_rev_y  = annual_pv_surplus_kwh * pv_price_y
            v2g_rev_y = annual_v2g_kwh       * v2g_price_y
            total_rev_y = pv_rev_y + v2g_rev_y

        om_y    = annual_om_cost
        capex_y = capex_total if i == 0 else 0.0
        cf_y    = total_rev_y - om_y - capex_y
        cum    += cf_y

        yearly_cash.append(cf_y)
        cumulative.append(cum)
        pv_revenues.append(pv_rev_y)
        v2g_revenues.append(v2g_rev_y)
        om_costs.append(om_y)
        capex_list.append(capex_y)

    # 탄소절감 kWh는 평균치로(단순화)
    avg_pv_surplus_kwh = np.mean([pv_kwh_by_year[min(max(y, min_y), max_y)] * (1 - self_use) for y in years])

    # V2G 연간 kWh(정보 표시에만 사용)
    daily_v2g_kwh = p["num_v2g_chargers"] * p["v2g_daily_discharge_per_charger_kwh"]
    annual_v2g_kwh = daily_v2g_kwh * p["v2g_operating_days"] * p["degradation_factor"]

    return {
        "years": years,
        "yearly_cash": yearly_cash,
        "cumulative": cumulative,
        "pv_revenues": pv_revenues,
        "v2g_revenues": v2g_revenues,
        "om_costs": om_costs,
        "capex_list": capex_list,
        "annual_pv_surplus_kwh": avg_pv_surplus_kwh,
        "annual_v2g_kwh": annual_v2g_kwh,
    }

# =========================
# 8) Streamlit App
# =========================
def main():
    st.title("V2G 투자 대비 연도별/누적 현금흐름 (CSV 일사합 + 시간대별 SMP 옵션)")

    # ── 파일 경로(사이드바에 노출 안 함): 레포 루트에 파일 배치 ──
    base_dir = os.path.dirname(os.path.abspath(__file__))
    irr_csv_path = os.path.join(base_dir, "jeju.csv")  # 일사합
    smp_csv_path = os.path.join(base_dir, "SMP.csv")   # SMP(시간대별 단가)

    # ① 모듈/면적/PR
    st.sidebar.header("시스템 설정")
    c1, c2 = st.sidebar.columns(2)
    panel_w   = c1.number_input("패널 폭(m)", value=1.46, step=0.01, format="%.2f")
    panel_h   = c2.number_input("패널 높이(m)", value=0.98, step=0.01, format="%.2f")
    n_modules = st.sidebar.number_input("모듈 수(장)", value=250, step=5, min_value=1)

    st.sidebar.caption("PR은 고정 계수(기본) 또는 사용자가 직접 지정")
    use_manual_pr = st.sidebar.checkbox("PR 수동 지정", value=False)
    if use_manual_pr:
        pr_manual = st.sidebar.number_input("PR (0~1)", value=0.78, min_value=0.0, max_value=1.0, step=0.01)
    else:
        pr_manual = None
    pr_base, availability, soiling, inv_eff = 0.82, 0.98, 0.02, 0.97

    # ② 시뮬레이션 기간/요금/비율
    st.sidebar.header("시뮬레이션/정산 설정")
    install_year = st.sidebar.number_input("설치 연도", value=2025, step=1)
    current_year = st.sidebar.number_input("마지막 연도", value=2045, step=1, min_value=install_year)

    params = make_v2g_model_params()
    params["num_v2g_chargers"] = st.sidebar.number_input("V2G 충전기 대수", value=params["num_v2g_chargers"], step=1, min_value=1)
    params["v2g_daily_discharge_per_charger_kwh"] = st.sidebar.number_input(
        "1대당 일일 방전량 (kWh)", value=params["v2g_daily_discharge_per_charger_kwh"], step=1, min_value=1
    )
    params["v2g_operating_days"] = st.sidebar.number_input(
        "연간 운영일 수", value=params["v2g_operating_days"], step=10, min_value=1, max_value=365
    )
    # 열화율 노출 원하면 주석 해제
    # params["degradation_factor"] = st.sidebar.number_input("열화·가용 보정", value=params["degradation_factor"], min_value=0.0, max_value=1.0, step=0.01)

    params["self_use_ratio"] = st.sidebar.slider("PV 자가소비 비율", 0.0, 1.0, params["self_use_ratio"], 0.05)

    # SMP 사용 옵션 (가격 연상승률은 SMP에 적용)
    use_smp = st.sidebar.checkbox("시간대별 SMP로 정산", value=True)
    params["price_cagr"] = st.sidebar.number_input("단가 연평균 상승률(CAGR)", value=params["price_cagr"], step=0.001, format="%.3f")
    smp_base_year = st.sidebar.number_input("SMP 기준 연도(스케일 기준)", value=2024, step=1)

    # V2G 방전 전략
    v2g_price_weighted = st.sidebar.checkbox("V2G 가격가중 방전(비싼 시간대 더 많이)", value=True)

    # 재무 지표용 할인율
    discount_rate = st.sidebar.number_input("할인율(연)", value=0.08, min_value=0.0, max_value=0.5, step=0.005, format="%.3f")

    # ── CSV 로드 & PV 연간 kWh 계산 ──
    irr_df = load_irradiance_csv(irr_csv_path)
    pv_by_year, area_m2, PR_used = compute_pv_kwh_by_year(
        irr_df,
        panel_width_m=panel_w, panel_height_m=panel_h, n_modules=int(n_modules),
        pr_base=pr_base, availability=availability, soiling=soiling, inv_eff=inv_eff,
        pr_manual=pr_manual
    )

    # ── SMP 시계열 준비(옵션) ──
    smp_series = None
    if use_smp:
        try:
            smp_series = load_smp_series(smp_csv_path)
        except Exception as e:
            st.warning(f"SMP 읽기 실패: {e}\n→ SMP 미사용 방식으로 대체 계산합니다.")
            smp_series = None
            use_smp = False

    # ── 현금흐름 계산 ──
    cf = build_yearly_cashflows_from_csv(
        install_year, current_year, params, pv_by_year,
        smp_base_series=smp_series if use_smp else None,
        smp_base_year=smp_base_year if use_smp else None,
        use_price_weighted_v2g=v2g_price_weighted
    )
    years       = cf["years"]
    yearly_cash = cf["yearly_cash"]
    cumulative  = cf["cumulative"]

    # 손익분기 연도
    break_even_year = next((y for y, cum in zip(years, cumulative) if cum >= 0), None)

    # 탄소절감량(kgCO2e) – 평균 연간 kWh로 추정
    clean_kwh_per_year = cf["annual_pv_surplus_kwh"] + cf["annual_v2g_kwh"]
    total_clean_kwh    = clean_kwh_per_year * len(years)
    total_co2_kg       = total_clean_kwh * EMISSION_FACTOR_KG_PER_KWH

    # 재무지표
    npv_val = npv(discount_rate, yearly_cash)
    irr_val = irr_bisection(yearly_cash)
    dpp_val = discounted_payback(yearly_cash, discount_rate)

    # --- KPI (1행: 재무) ---
    k1, k2, k3, sp1 = st.columns([1, 1, 1, 0.4])
    with k1:
        st.markdown('<div style="font-size:0.85rem;color:#666;">NPV</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{npv_val:,.0f} 원</div>', unsafe_allow_html=True)
        st.caption(f"할인율 {discount_rate*100:.1f}%")
    with k2:
        st.markdown('<div style="font-size:0.85rem;color:#666;">IRR</div>', unsafe_allow_html=True)
        irr_txt = f"{irr_val*100:.2f} %" if irr_val is not None else "정의 불가"
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{irr_txt}</div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div style="font-size:0.85rem;color:#666;">할인 회수기간</div>', unsafe_allow_html=True)
        dpp_txt = f"{dpp_val:.2f} 년" if dpp_val is not None else "미회수"
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{dpp_txt}</div>', unsafe_allow_html=True)

    # --- KPI (2행: 시스템/환경) ---
    r1, r2, r3, sp2 = st.columns([1, 1, 1, 0.4])
    with r1:
        be_text = f"{break_even_year}년" if break_even_year else "아직 미도달"
        st.markdown('<div style="font-size:0.85rem;color:#666;">손익분기 연도</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{be_text}</div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div style="font-size:0.85rem;color:#666;">마지막 연도 누적</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{cumulative[-1]:,.0f} 원</div>', unsafe_allow_html=True)
    with r3:
        st.markdown('<div style="font-size:0.85rem;color:#666;">누적 탄소절감량</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{total_co2_kg:,.0f} kgCO₂e</div>', unsafe_allow_html=True)

    # --- 그래프: 누적 라인 ---
    st.subheader("누적 현금흐름")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, cumulative, marker="o", linewidth=2.2)
    ax.set_xlabel("연도"); ax.set_ylabel("누적 금액(원)")
    ax.yaxis.set_major_formatter(FuncFormatter(won_formatter))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("V2G + PV 누적 현금흐름" + (" (SMP 정산)" if use_smp else ""))
    if break_even_year is not None:
        ax.axvline(break_even_year, color="green", linestyle="--", alpha=0.7)
        ax.text(break_even_year, 0, f"손익분기 {break_even_year}", color="green", va="bottom", ha="left")
    st.pyplot(fig)

    # --- 그래프: 연도별 누적 막대 ---
    st.subheader("연도별 순현금흐름 (누적)")
    x_labels = [f"{y}년" for y in years]
    colors = ["red" if cum < 0 else "royalblue" for cum in cumulative]
    bar_fig = go.Figure(data=[go.Bar(x=x_labels, y=cumulative,
                                     marker=dict(color=colors),
                                     text=[f"{v:,.0f}원" for v in cumulative],
                                     textposition="outside")])
    if break_even_year is not None:
        be_label = f"{break_even_year}년"
        bar_fig.add_shape(type="line", x0=be_label, x1=be_label, y0=0, y1=1,
                          xref="x", yref="paper", line=dict(color="green", width=2, dash="dash"))
        bar_fig.add_annotation(x=be_label, y=1, xref="x", yref="paper",
                               text=f"손익분기 {break_even_year}년", showarrow=False, yanchor="bottom",
                               font=dict(color="green"))
    bar_fig.update_layout(title="연도별 순현금흐름 (누적)", yaxis=dict(tickformat=","), bargap=0.25)
    st.plotly_chart(bar_fig, use_container_width=True)

    # --- 표: 연도별 금액 ---
    st.subheader("연도별 금액 확인")
    df_table = pd.DataFrame({
        "연도": years,
        "순현금흐름(원)": yearly_cash,
        "누적(원)": cumulative,
        "PV 수입(원)": cf["pv_revenues"],
        "V2G 수입(원)": cf["v2g_revenues"],
        "O&M 비용(원)": cf["om_costs"],
        "CAPEX(원)": cf["capex_list"],
    })
    st.dataframe(df_table, use_container_width=True)

    # 참고 정보
    st.caption(
        f"총 모듈 면적: {area_m2:.1f} m² | 사용 PR: {PR_used:.3f} "
        f"| CSV(일사합) 연도 범위: {min(pv_by_year.keys())}~{max(pv_by_year.keys())} "
        f"| 정산: {'SMP(시간대별)' if use_smp else '평균단가'}"
    )

if __name__ == "__main__":
    main()
