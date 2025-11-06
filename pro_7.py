import os
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# kWh 당 탄소배출계수 (kg CO2e/kWh)
EMISSION_FACTOR_KG_PER_KWH = 0.495


# =========================
# 1. 한글 폰트
# =========================
def set_korean_font():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, "NanumGothic.ttf")

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
    else:
        plt.rcParams["axes.unicode_minus"] = False


set_korean_font()


# =========================
# 전력단가 CAGR 적용 함수
# =========================
def price_with_cagr(base_price, base_year, year, cagr):
    return base_price * (1 + cagr) ** (year - base_year)


def make_v2g_model_params():
    return {
        "pv_capacity_kw": 125,
        "pv_annual_kwh": 153_300,
        "self_use_ratio": 0.60,
        "num_v2g_chargers": 6,
        "v2g_charger_unit_cost": 25_000_000,
        "v2g_daily_discharge_per_charger_kwh": 35,
        "degradation_factor": 0.9,
        "v2g_operating_days": 300,
        "tariff_base_year": 2025,
        "pv_base_price": 160,
        "v2g_price_gap": 30,
        "price_cagr": 0.043,
        "om_ratio": 0.015,
        "discount_rate": 0.06,   # ✅ 추가됨
    }


# =========================
# 연도별 CF 계산 + 할인현금흐름 추가
# =========================
def build_yearly_cashflows(install_year: int, current_year: int, p: dict):

    # PV
    annual_pv_kwh = p["pv_annual_kwh"]
    annual_pv_surplus_kwh = annual_pv_kwh * (1 - p["self_use_ratio"])

    # V2G
    daily_v2g_kwh = p["num_v2g_chargers"] * p["v2g_daily_discharge_per_charger_kwh"]
    annual_v2g_kwh = daily_v2g_kwh * p["v2g_operating_days"] * p["degradation_factor"]

    # CAPEX / O&M
    capex_total = p["num_v2g_chargers"] * p["v2g_charger_unit_cost"]
    annual_om_cost = capex_total * p["om_ratio"]

    years = list(range(install_year, current_year + 1))

    yearly_cash = []
    cumulative = []
    discounted_cf = []
    cumulative_discounted = []

    pv_revenues = []
    v2g_revenues = []
    om_costs = []
    capex_list = []

    cum = 0
    cum_d = 0

    for idx, year in enumerate(years):

        pv_price_y = price_with_cagr(
            p["pv_base_price"], p["tariff_base_year"], year, p["price_cagr"]
        )
        v2g_price_y = pv_price_y + p["v2g_price_gap"]

        revenue_pv = annual_pv_surplus_kwh * pv_price_y
        revenue_v2g = annual_v2g_kwh * v2g_price_y

        om_cost = annual_om_cost
        capex = capex_total if idx == 0 else 0

        cf = revenue_pv + revenue_v2g - om_cost - capex
        cum += cf

        # ✅ 할인 적용
        discount_factor = (1 + p["discount_rate"]) ** idx
        dcf = cf / discount_factor
        cum_d += dcf

        yearly_cash.append(cf)
        cumulative.append(cum)
        discounted_cf.append(dcf)
        cumulative_discounted.append(cum_d)

        pv_revenues.append(revenue_pv)
        v2g_revenues.append(revenue_v2g)
        om_costs.append(om_cost)
        capex_list.append(capex)

    # ✅ NPV
    npv = cumulative_discounted[-1]

    # ✅ IRR
    try:
        irr = np.irr(yearly_cash)
    except:
        irr = None

    # ✅ Discounted Payback Period
    dpbp = None
    for y, val in zip(years, cumulative_discounted):
        if val >= 0:
            dpbp = y
            break

    return {
        "years": years,
        "yearly_cash": yearly_cash,
        "cumulative": cumulative,
        "discounted_cf": discounted_cf,
        "cumulative_discounted": cumulative_discounted,
        "pv_revenues": pv_revenues,
        "v2g_revenues": v2g_revenues,
        "om_costs": om_costs,
        "capex_list": capex_list,
        "annual_pv_surplus_kwh": annual_pv_surplus_kwh,
        "annual_v2g_kwh": annual_v2g_kwh,
        "npv": npv,
        "irr": irr,
        "dpbp": dpbp,
    }


def won_formatter(x, pos):
    return f"{int(x):,}"


# =========================
# Streamlit App
# =========================
def main():
    st.title("⚡ V2G + PV 경제성 분석 (NPV · IRR · Payback 포함)")

    params = make_v2g_model_params()

    st.sidebar.header("시뮬레이션 입력")
    install_year = st.sidebar.number_input("설치 연도", value=2025)
    current_year = st.sidebar.number_input("마지막 연도", value=2045)

    params["discount_rate"] = st.sidebar.number_input(
        "할인율(Discount Rate)", value=params["discount_rate"], step=0.005, format="%.3f"
    )

    # 나머지 입력 유지
    params["pv_annual_kwh"] = st.sidebar.number_input("연간 PV 발전량(kWh)", value=params["pv_annual_kwh"])
    params["self_use_ratio"] = st.sidebar.slider("PV 자가소비 비율", 0.0, 1.0, params["self_use_ratio"])
    params["num_v2g_chargers"] = st.sidebar.number_input("V2G 충전기 대수",
                                                         value=params["num_v2g_chargers"], step=1)

    # ===== 계산 =====
    cf = build_yearly_cashflows(install_year, current_year, params)

    years = cf["years"]

    # ✅ 경제성 KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("NPV", f"{cf['npv']:,.0f} 원")
    col2.metric("IRR", f"{cf['irr']*100:.2f} %" if cf["irr"] is not None else "N/A")
    col3.metric("Discounted Payback", f"{cf['dpbp']}년" if cf["dpbp"] else "미도달")

    # ✅ 탄소 절감량
    clean_kwh = (cf["annual_pv_surplus_kwh"] + cf["annual_v2g_kwh"]) * len(years)
    co2_saved = clean_kwh * EMISSION_FACTOR_KG_PER_KWH
    st.metric("누적 탄소절감량", f"{co2_saved:,.0f} kgCO₂e")

    # ✅ 누적 현재가치
    st.subheader("누적 현재가치 (DCF)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, cf["cumulative_discounted"], marker="o")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("연도")
    ax.set_ylabel("현재가치(원)")
    st.pyplot(fig)

    # ✅ 표 출력
    df = pd.DataFrame({
        "연도": years,
        "순현금흐름": cf["yearly_cash"],
        "할인현금흐름": cf["discounted_cf"],
        "누적": cf["cumulative"],
        "누적(할인)": cf["cumulative_discounted"]
    })
    st.subheader("연도별 현금흐름")
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
