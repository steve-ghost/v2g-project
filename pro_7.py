import os
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pvlib import location
from pvlib import irradiance, atmosphere, pvsystem, temperature

# -----------------------------
# 상수: kWh당 탄소배출계수 (kg CO2e/kWh)
# -----------------------------
EMISSION_FACTOR_KG_PER_KWH = 0.495  # 국내 전력 1kWh 생산시 약 0.495 kgCO2e

# =========================
# 1) 한글 폰트 (NanumGothic.ttf 를 앱 폴더에 넣어두면 자동 사용)
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
# 2) 유틸/지표 계산 함수
# =========================
def generate_hourly_pv_kwh_from_jeju_csv(file_obj, pv_kw=125):
    df = pd.read_csv(file_obj)

    # ✅ 1) 컬럼 공백 제거 + BOM 제거
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    # ✅ 2) 일시 컬럼 자동 탐색
    possible_time_cols = ["일시", "일시(UTC)", "date", "시간", "timestamp"]
    time_col = None
    for col in df.columns:
        if col in possible_time_cols:
            time_col = col
            break

    if time_col is None:
        raise KeyError(f"CSV에서 일시 컬럼을 찾지 못했습니다. 실제 컬럼: {df.columns.tolist()}")

    df[time_col] = pd.to_datetime(df[time_col])

    # ✅ 3) 일사합(MJ/m2) 컬럼 자동 탐색
    possible_irr_cols = ["일사합(MJ/m2)", "일사합", "GHI", "Irradiance", "일사"]
    irr_col = None
    for col in df.columns:
        if col in possible_irr_cols:
            irr_col = col
            break

    if irr_col is None:
        raise KeyError(f"CSV에서 일사합(irradiance) 컬럼을 찾지 못했습니다. 실제 컬럼: {df.columns.tolist()}")

    # ✅ 4) MJ/m2 → kWh/m2 변환
    df["irr_kwh_m2"] = df[irr_col] * 0.2777778  # 1 MJ = 0.27778 kWh

    # ✅ 5) 패널 용량 × 효율 가정해서 실제 발전량 계산
    performance_ratio = 0.75
    df["pv_kwh"] = df["irr_kwh_m2"] * pv_kw * performance_ratio

    # ✅ 6) 시간순 정렬 & set_index
    df = df.sort_values(time_col)
    df = df.set_index(time_col)

    return df["pv_kwh"]

    # GHI 일별값을 시간별로 재분배(가중: 일사곡선 기반)
    def distribute_daily_to_hourly(day, ghi_day):
        mask = (hourly_index.date == day.date())
        hours = hourly_index[mask]
        pos = solpos[mask]
        zen = pos['zenith']
        # 낮 시간만 양수
        w = np.clip(np.cos(np.radians(zen)), 0, None)
        if w.sum() == 0:
            return pd.Series(np.zeros(len(hours)), index=hours)
        return pd.Series(ghi_day * (w / w.sum()), index=hours)

    hourly_ghi = pd.concat(
        [distribute_daily_to_hourly(day, ghi_day) for day, ghi_day in daily_ghi.items()]
    )

    # PVlib 입력 위한 POA 변환
    poa = irradiance.get_total_irradiance(
        surface_tilt=25,
        surface_azimuth=180,
        dni=None,
        ghi=hourly_ghi,
        dhi=None,
        solar_zenith=solpos['zenith'].reindex(hourly_ghi.index),
        solar_azimuth=solpos['azimuth'].reindex(hourly_ghi.index)
    )["poa_global"]

    # PV 성능 파라미터
    temp_cell = temperature.sapm_cell(poa, temp_air=25, wind_speed=1)
    effective_irr = poa

    # 모듈 성능 모델 (단순화)
    pv_power_w = pv_kw * 1000 * (effective_irr / 1000)  # 1sun = 1000W/m2
    pv_power_kwh = (pv_power_w / 1000)

    return pv_power_kwh

def price_with_cagr(base_price, base_year, year, cagr):
    """기준연도 대비 연복리(cagr) 상승 단가"""
    return base_price * (1 + cagr) ** (year - base_year)

def npv(rate: float, cashflows: list[float]) -> float:
    """NPV: 첫 해(설치연도) 현금흐름이 cashflows[0] (t=0) 기준"""
    if rate <= -0.999999:
        return float("nan")
    return float(sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows)))

def irr_bisection(cashflows: list[float], lo=-0.99, hi=3.0, tol=1e-7, max_iter=200):
    """
    IRR: 부호변화가 있어야 수렴이 잘 됨. 이분법으로 안정적으로 탐색.
    반환값이 None이면 IRR이 정의되지 않거나 범위 내에서 찾지 못한 경우.
    """
    def f(r):
        try:
            return npv(r, cashflows)
        except Exception:
            return np.nan

    f_lo, f_hi = f(lo), f(hi)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
        return None  # 근 찾기 곤란(부호변화 없음)

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if np.isnan(f_mid):
            return None
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return mid

def discounted_payback(cashflows: list[float], rate: float):
    """
    할인 회수기간(DPP): 할인 누적현금흐름이 0을 넘는 최초 시점까지의 연수.
    선형 보간으로 부분 연도 반영. 없으면 None.
    """
    disc = []
    cum = 0.0
    for t, cf in enumerate(cashflows):
        val = cf / ((1 + rate) ** t)
        cum += val
        disc.append(cum)

    # 최초로 0 이상이 되는 시점 찾기
    for k, v in enumerate(disc):
        if v >= 0:
            if k == 0:
                return 0.0
            prev = disc[k - 1]
            # k-1년도 말엔 prev<0, k년도 말엔 v>=0
            # 그 사이 어느 시점에 0이 되는지 선형 보간
            frac = 0.0 if v == prev else (-prev) / (v - prev)
            return (k - 1) + max(0.0, min(1.0, frac))
    return None

def won_formatter(x, pos):
    return f"{int(x):,}"


# =========================
# 3) 기본 파라미터
# =========================
def make_v2g_model_params():
    return {
        # PV
        "pv_capacity_kw": 125,            # 정보용(직접 계산엔 annual_kwh 사용)
        "pv_annual_kwh": 153_300,         # 연간 발전량 (kWh)
        "self_use_ratio": 0.60,           # 자가소비 비율

        # V2G
        "num_v2g_chargers": 6,            # 대수
        "v2g_charger_unit_cost": 25_000_000,  # 대당 CAPEX
        "v2g_daily_discharge_per_charger_kwh": 35,
        "degradation_factor": 0.9,
        "v2g_operating_days": 300,        # 연 가동일

        # 단가(연복리 상승)
        "tariff_base_year": 2025,
        "pv_base_price": 160,             # 원/kWh
        "v2g_price_gap": 30,              # PV 대비 V2G 프리미엄
        "price_cagr": 0.043,              # 전력단가 상승률

        # O&M
        "om_ratio": 0.015,                # CAPEX 대비 연간 O&M 비율
    }


# =========================
# 4) 현금흐름 빌드
# =========================
def build_yearly_cashflows(install_year: int, current_year: int, p: dict):
    # PV: 잉여 판매량
    annual_pv_kwh = p["pv_annual_kwh"]
    annual_pv_surplus_kwh = annual_pv_kwh * (1 - p["self_use_ratio"])

    # V2G: 연간 방전량
    daily_v2g_kwh = p["num_v2g_chargers"] * p["v2g_daily_discharge_per_charger_kwh"]
    annual_v2g_kwh = daily_v2g_kwh * p["v2g_operating_days"] * p["degradation_factor"]

    # CAPEX/O&M
    capex_total = p["num_v2g_chargers"] * p["v2g_charger_unit_cost"]
    annual_om_cost = capex_total * p["om_ratio"]

    years = list(range(install_year, current_year + 1))
    yearly_cash, cumulative = [], []
    pv_revenues, v2g_revenues, om_costs, capex_list = [], [], [], []

    cum = 0.0
    for i, year in enumerate(years):
        pv_price_y = price_with_cagr(p["pv_base_price"], p["tariff_base_year"], year, p["price_cagr"])
        v2g_price_y = pv_price_y + p["v2g_price_gap"]

        revenue_pv_y = annual_pv_surplus_kwh * pv_price_y
        revenue_v2g_y = annual_v2g_kwh * v2g_price_y
        annual_revenue_y = revenue_pv_y + revenue_v2g_y

        om_y = annual_om_cost
        capex_y = capex_total if i == 0 else 0

        cf = annual_revenue_y - om_y - capex_y
        cum += cf

        yearly_cash.append(cf)
        cumulative.append(cum)
        pv_revenues.append(revenue_pv_y)
        v2g_revenues.append(revenue_v2g_y)
        om_costs.append(om_y)
        capex_list.append(capex_y)

    return {
        "years": years,
        "yearly_cash": yearly_cash,
        "cumulative": cumulative,
        "pv_revenues": pv_revenues,
        "v2g_revenues": v2g_revenues,
        "om_costs": om_costs,
        "capex_list": capex_list,
        # 탄소절감 kWh 계산용
        "annual_pv_surplus_kwh": annual_pv_surplus_kwh,
        "annual_v2g_kwh": annual_v2g_kwh,
    }


# =========================
# 5) Streamlit App
# =========================

def main():
    st.title("V2G 투자 대비 연도별/누적 현금흐름")

    params = make_v2g_model_params()

    # ----- PV 데이터 업로드 -----
    uploaded = st.sidebar.file_uploader("jeju.csv 업로드", type=["csv"])

    if uploaded is None:
        st.warning("jeju.csv 파일을 업로드해주세요.")
        return   # ✅ 이 return은 반드시 main() 내부여야 함.

    # ✅ 업로드된 파일 읽기
    hourly_pv = generate_hourly_pv_kwh_from_jeju_csv(
        uploaded, pv_kw=params["pv_capacity_kw"]
    )
    params["pv_annual_kwh"] = hourly_pv.sum()






    # ----- 사이드바 입력 -----
    st.sidebar.header("시뮬레이션 입력")
    # ----- PVlib 기반 시간별 발전량 업로드 -----
    csv_path = ".devcontainer/jeju.csv"  # 또는 GitHub RAW URL
    hourly_pv = generate_hourly_pv_kwh_from_jeju_csv(csv_path, pv_kw=params["pv_capacity_kw"])
    params["pv_annual_kwh"] = hourly_pv.sum()


    if uploaded is not None:
        hourly_pv = generate_hourly_pv_kwh_from_jeju_csv(uploaded, pv_kw=params["pv_capacity_kw"])
        annual_pv_from_pvlib = hourly_pv.sum()

        st.sidebar.success(f"PVlib 기반 연간 발전량 계산됨: {annual_pv_from_pvlib:,.0f} kWh")
        params["pv_annual_kwh"] = annual_pv_from_pvlib

        install_year = st.sidebar.number_input("설치 연도", value=2025, step=1)
        current_year = st.sidebar.number_input("마지막 연도", value=2045, step=1, min_value=install_year)

        params["num_v2g_chargers"] = st.sidebar.number_input("V2G 충전기 대수", value=params["num_v2g_chargers"], step=1, min_value=1)
        params["v2g_daily_discharge_per_charger_kwh"] = st.sidebar.number_input(
            "1대당 일일 방전량 (kWh)", value=params["v2g_daily_discharge_per_charger_kwh"], step=1, min_value=1
        )
        params["v2g_operating_days"] = st.sidebar.number_input(
            "연간 운영일 수", value=params["v2g_operating_days"], step=10, min_value=1, max_value=365
        )
        params["pv_annual_kwh"] = st.sidebar.number_input("연간 PV 발전량 (kWh)", value=params["pv_annual_kwh"], step=1000, min_value=0)
        params["self_use_ratio"] = st.sidebar.slider("PV 자가소비 비율", min_value=0.0, max_value=1.0, value=params["self_use_ratio"], step=0.05)
        params["pv_base_price"] = st.sidebar.number_input("PV 기준단가 (원/kWh)", value=params["pv_base_price"], step=5, min_value=0)
        params["price_cagr"] = st.sidebar.number_input("전력단가 연평균 상승률", value=params["price_cagr"], step=0.001, format="%.3f")

    # ★ 재무 지표용 할인율
    discount_rate = st.sidebar.number_input(
        "할인율(연)", value=0.08, min_value=0.0, max_value=0.5, step=0.005, format="%.3f", help="NPV/할인회수기간 계산에 사용"
    )

    # ----- 계산 -----
    cf = build_yearly_cashflows(install_year, current_year, params)
    years = cf["years"]
    yearly_cash = cf["yearly_cash"]
    cumulative = cf["cumulative"]

    # 손익분기 연도(회계적): 누적이 0 이상 최초 연도
    break_even_year = None
    for y, cum in zip(years, cumulative):
        if cum >= 0:
            break_even_year = y
            break

    # 탄소절감량 (kgCO2e)
    clean_kwh_per_year = cf["annual_pv_surplus_kwh"] + cf["annual_v2g_kwh"]
    total_clean_kwh = clean_kwh_per_year * len(years)
    total_co2_kg = total_clean_kwh * EMISSION_FACTOR_KG_PER_KWH

    # ----- 재무지표 -----
    npv_val = npv(discount_rate, yearly_cash)
    irr_val = irr_bisection(yearly_cash)  # None 가능
    dpp_val = discounted_payback(yearly_cash, discount_rate)  # None 가능

    # ----- KPI (1행: 재무지표) -----
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

    # ----- KPI (2행: 기존 지표) -----
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

    # ----- 누적 라인 차트 (matplotlib) -----
    st.subheader("누적 현금흐름")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, cumulative, marker="o", linewidth=2.2)
    ax.set_xlabel("연도"); ax.set_ylabel("누적 금액(원)")
    ax.yaxis.set_major_formatter(FuncFormatter(won_formatter))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("V2G + PV 누적 현금흐름")
    if break_even_year is not None:
        ax.axvline(break_even_year, color="green", linestyle="--", alpha=0.7)
        ax.text(break_even_year, 0, f"손익분기 {break_even_year}", color="green", va="bottom", ha="left")
    st.pyplot(fig)

    # ----- 누적 막대 (연도별 누적) -----
    st.subheader("연도별 순현금흐름 (누적)")
    x_labels = [f"{y}년" for y in years]
    colors = ["red" if cum < 0 else "royalblue" for cum in cumulative]
    bar_fig = go.Figure(
        data=[go.Bar(x=x_labels, y=cumulative, marker=dict(color=colors),
                     text=[f"{v:,.0f}원" for v in cumulative], textposition="outside")]
    )
    if break_even_year is not None:
        be_label = f"{break_even_year}년"
        bar_fig.add_shape(type="line", x0=be_label, x1=be_label, y0=0, y1=1,
                          xref="x", yref="paper", line=dict(color="green", width=2, dash="dash"))
        bar_fig.add_annotation(x=be_label, y=1, xref="x", yref="paper",
                               text=f"손익분기 {break_even_year}년", showarrow=False, yanchor="bottom",
                               font=dict(color="green"))
    bar_fig.update_layout(title="연도별 순현금흐름 (누적)", yaxis=dict(tickformat=","), bargap=0.25)
    st.plotly_chart(bar_fig, use_container_width=True)

    # ----- 표 -----
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


if __name__ == "__main__":
    main()
