import os
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import pandas as pd


# =========================
# 1. 한글 폰트: repo에 올려둔 NanumGothic.ttf 강제 사용
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
    }


def build_yearly_cashflows(install_year: int, current_year: int, p: dict):
    # PV
    annual_pv_kwh = p["pv_annual_kwh"]
    annual_pv_surplus_kwh = annual_pv_kwh * (1 - p["self_use_ratio"])

    # V2G
    daily_v2g_kwh = p["num_v2g_chargers"] * p["v2g_daily_discharge_per_charger_kwh"]
    annual_v2g_kwh = (
        daily_v2g_kwh * p["v2g_operating_days"] * p["degradation_factor"]
    )

    # CAPEX / O&M
    capex_total = p["num_v2g_chargers"] * p["v2g_charger_unit_cost"]
    annual_om_cost = capex_total * p["om_ratio"]

    years = list(range(install_year, current_year + 1))
    yearly_cash = []
    cumulative = []
    pv_revenues = []
    v2g_revenues = []
    om_costs = []
    capex_list = []

    cum = 0
    for i, year in enumerate(years):
        pv_price_y = price_with_cagr(
            p["pv_base_price"], p["tariff_base_year"], year, p["price_cagr"]
        )
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
    }


def won_formatter(x, pos):
    return f"{int(x):,}"


def main():
    st.title("V2G 투자 대비 연도별/누적 현금흐름")

    params = make_v2g_model_params()

    # ===== 사이드바 입력 =====
    st.sidebar.header("시뮬레이션 입력")
    install_year = st.sidebar.number_input("설치 연도", value=2025, step=1)
    current_year = st.sidebar.number_input(
        "마지막 연도", value=2045, step=1, min_value=install_year
    )

    params["num_v2g_chargers"] = st.sidebar.number_input(
        "V2G 충전기 대수",
        value=params["num_v2g_chargers"],
        step=1,
        min_value=1,
    )
    params["v2g_daily_discharge_per_charger_kwh"] = st.sidebar.number_input(
        "1대당 일일 방전량(kWh)",
        value=params["v2g_daily_discharge_per_charger_kwh"],
        step=1,
        min_value=1,
    )
    params["v2g_operating_days"] = st.sidebar.number_input(
        "연간 운영일 수",
        value=params["v2g_operating_days"],
        step=10,
        min_value=1,
        max_value=365,
    )
    params["pv_annual_kwh"] = st.sidebar.number_input(
        "연간 PV 발전량(kWh)",
        value=params["pv_annual_kwh"],
        step=1000,
        min_value=0,
    )
    params["self_use_ratio"] = st.sidebar.slider(
        "PV 자가소비 비율",
        min_value=0.0,
        max_value=1.0,
        value=params["self_use_ratio"],
        step=0.05,
    )
    params["pv_base_price"] = st.sidebar.number_input(
        "PV 기준단가(원/kWh)",
        value=params["pv_base_price"],
        step=5,
        min_value=0,
    )
    params["price_cagr"] = st.sidebar.number_input(
        "전력단가 연평균 상승률",
        value=params["price_cagr"],
        step=0.001,
        format="%.3f",
    )

    # ===== 계산 =====
    cf_data = build_yearly_cashflows(install_year, current_year, params)
    years = cf_data["years"]
    yearly_cash = cf_data["yearly_cash"]
    cumulative = cf_data["cumulative"]

    # 손익분기 연도
    break_even_year = None
    for y, cum_val in zip(years, cumulative):
        if cum_val >= 0:
            break_even_year = y
            break

    # ===== KPI =====
    col1, col2 = st.columns(2)
    if break_even_year is not None:
        col1.metric("손익분기 연도", f"{break_even_year}년")
    else:
        col1.metric("손익분기 연도", "아직 미도달")
    
    val_str = "{:,.0f}".format(cumulative[-1])  # 소수점 0자리로 강제
    col2.metric("마지막 연도 누적", f"{val_str} 원")

    # ===== 1) 누적 현금흐름 (matplotlib) =====
    st.subheader("누적 현금흐름")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, cumulative, marker="o", linewidth=2.2)
    ax.set_xlabel("연도")
    ax.set_ylabel("누적 금액(원)")
    ax.yaxis.set_major_formatter(FuncFormatter(won_formatter))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("V2G 투자 대비 연도별/누적 현금흐름")
    if break_even_year is not None:
        ax.axvline(break_even_year, color="green", linestyle="--", alpha=0.7)
        ax.text(
            break_even_year,
            0,
            f"손익분기 {break_even_year}",
            color="green",
            va="bottom",
            ha="left",
        )
    st.pyplot(fig)

    # ===== 2) 워터폴 =====
    st.subheader("연도별 순현금흐름")
    wf = go.Figure(
        go.Waterfall(
            name="연도별 현금흐름",
            orientation="v",
            x=[f"{y}년" for y in years],
            measure=["relative"] * len(years),
            y=yearly_cash,
            text=[f"{v:,.0f}원" for v in yearly_cash],
            textposition="outside",
        )
    )
    wf.update_layout(
        title="연도별 순현금흐름",
        yaxis=dict(tickformat=","),
    )
    st.plotly_chart(wf, use_container_width=True)

    # ===== 표 =====
    st.subheader("연도별 금액 확인")
    df_table = pd.DataFrame(
        {
            "연도": years,
            "순현금흐름(원)": yearly_cash,
            "누적(원)": cumulative,
            "PV 수입(원)": cf_data["pv_revenues"],
            "V2G 수입(원)": cf_data["v2g_revenues"],
            "O&M 비용(원)": cf_data["om_costs"],
            "CAPEX(원)": cf_data["capex_list"],
        }
    )
    st.dataframe(df_table)


if __name__ == "__main__":
    main()
