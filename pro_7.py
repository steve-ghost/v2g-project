# app.py
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
MJ_PER_M2_TO_KWH_PER_M2    = 0.27778 # MJ/m² → kWh/m²

# =========================
# 1) 한글 폰트 (NanumGothic.ttf 를 앱 폴더에 두면 자동 사용)
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
        try:
            return npv(r, cashflows)
        except Exception:
            return np.nan
    f_lo, f_hi = f(lo), f(hi)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
        return None
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
    disc = []
    cum = 0.0
    for t, cf in enumerate(cashflows):
        val = cf / ((1 + rate) ** t)
        cum += val
        disc.append(cum)
    for k, v in enumerate(disc):
        if v >= 0:
            if k == 0:
                return 0.0
            prev = disc[k - 1]
            frac = 0.0 if v == prev else (-prev) / (v - prev)
            return (k - 1) + max(0.0, min(1.0, frac))
    return None

def won_formatter(x, pos):
    return f"{int(x):,}"

# =========================
# 3) CSV → 연도별 PV kWh 계산
# =========================
def load_irradiance_csv(csv_path: str) -> pd.DataFrame:
    """
    연도별 '일사합(MJ/m²)' 파일을 읽어:
    - 연도 컬럼 후보: ['연도','year','일시']
    - 일사합 컬럼 후보: ['일사합(mj/m2)','일사합(mj/m²)','ghi_mj','ghi(mj/m2)','ghi(mj/m²)','solar_mj','solar(mj/m2)']
    반환: ['year','ghi_mj_m2']
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")

    tried = []
    df = None
    for enc in ["utf-8-sig", "cp949"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            tried.append((enc, str(e)))
    if df is None:
        raise ValueError(f"CSV 읽기 실패: {tried}")

    df.columns = [str(c).strip().lower() for c in df.columns]

    # 연도
    year_candidates = ["연도", "year", "일시"]
    year_col = next((c for c in year_candidates if c in df.columns), None)
    if year_col is None:
        raise ValueError(f"[연도 컬럼 없음] 실제 열: {list(df.columns)}")

    # 일사합
    ghi_candidates = [
        "일사합(mj/m2)", "일사합(mj/m²)", "ghi_mj", "ghi(mj/m2)", "ghi(mj/m²)",
        "solar_mj", "solar(mj/m2)"
    ]
    ghi_col = next((c for c in ghi_candidates if c in df.columns), None)
    if ghi_col is None:
        maybe = [c for c in df.columns if ("mj" in c and "m" in c)]
        raise ValueError(f"[일사합 컬럼 없음] 실제 열: {list(df.columns)} / 유사 후보: {maybe}")

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
    연도별 일사합(MJ/m²) → kWh/m² 변환 후
    PV 연간 발전량(kWh) = GHI_kWh/m² × 총면적(m²) × PR
    """
    area_m2 = float(panel_width_m) * float(panel_height_m) * int(n_modules)
    PR = float(pr_manual) if pr_manual is not None else (pr_base * availability * (1.0 - soiling) * inv_eff)

    ghi_kwh_m2 = irr_df["ghi_mj_m2"].astype(float) * MJ_PER_M2_TO_KWH_PER_M2
    pv_kwh = ghi_kwh_m2 * area_m2 * PR

    out = dict(zip(irr_df["year"].astype(int).tolist(), pv_kwh.astype(float).tolist()))
    return out, area_m2, PR

# =========================
# 4) 기본 파라미터
# =========================
def make_v2g_model_params():
    return {
        # PV는 CSV로 자동 계산
        "self_use_ratio": 0.60,  # 자가소비 비율

        # V2G (사이드바에서 조정 가능)
        "num_v2g_chargers": 6,
        "v2g_charger_unit_cost": 25_000_000,
        "v2g_daily_discharge_per_charger_kwh": 35,
        "degradation_factor": 0.9,
        "v2g_operating_days": 300,

        # 잉여 전력 중 V2G로 전환하는 비율(0~1) — 기본은 1.0
        "v2g_share_of_surplus": 1.0,

        # 단가/상승률
        "tariff_base_year": 2025,
        "pv_base_price": 160,
        "v2g_price_gap": 30,
        "price_cagr": 0.043,

        # O&M
        "om_ratio": 0.015,
    }

# =========================
# 5) 현금흐름 빌드 (CSV 기반 PV + 연도별 V2G 캡 로직)
# =========================
def build_yearly_cashflows_from_csv(install_year: int, current_year: int, p: dict,
                                    pv_kwh_by_year: dict):
    """
    pv_kwh_by_year: {year: kWh}
    기간 밖 연도는 경계값으로 클램핑.
    '그 해의 PV 잉여'와 '그 해의 V2G 용량'의 교집합으로 V2G kWh를 결정.
    나머지 잉여는 계통판매(PV 수익)로 반영.
    """
    # CAPEX/O&M
    capex_total = p["num_v2g_chargers"] * p["v2g_charger_unit_cost"]
    annual_om_cost = capex_total * p["om_ratio"]

    years = list(range(install_year, current_year + 1))
    yearly_cash, cumulative = [], []
    pv_revenues, v2g_revenues, om_costs, capex_list = [], [], [], []

    if len(pv_kwh_by_year) == 0:
        raise ValueError("CSV에서 계산된 PV 연간 발전량이 없습니다.")
    min_y, max_y = min(pv_kwh_by_year.keys()), max(pv_kwh_by_year.keys())

    self_use = float(p["self_use_ratio"])
    v2g_share = float(p.get("v2g_share_of_surplus", 1.0))

    cum = 0.0
    for i, year in enumerate(years):
        # 1) PV(연도별)
        y_key = min(max(year, min_y), max_y)
        pv_kwh_y = pv_kwh_by_year[y_key]

        pv_self_y    = pv_kwh_y * self_use
        pv_surplus_y = max(0.0, pv_kwh_y - pv_self_y)

        # 2) 그 해 V2G 용량 상한
        v2g_capacity_y = (
            p["num_v2g_chargers"]
            * p["v2g_daily_discharge_per_charger_kwh"]
            * p["v2g_operating_days"]
            * p["degradation_factor"]
        )

        # 3) 실제 V2G 처리량(잉여의 일부를 V2G로 전환) + 잔여 잉여는 계통판매
        v2g_kwh_y    = min(pv_surplus_y * v2g_share, v2g_capacity_y)
        pv_grid_kwh_y = max(0.0, pv_surplus_y - v2g_kwh_y)

        # 4) 단가
        pv_price_y  = price_with_cagr(p["pv_base_price"], p["tariff_base_year"], year, p["price_cagr"])
        v2g_price_y = pv_price_y + p["v2g_price_gap"]

        # 5) 수입
        revenue_pv_y  = pv_grid_kwh_y * pv_price_y
        revenue_v2g_y = v2g_kwh_y    * v2g_price_y
        annual_revenue_y = revenue_pv_y + revenue_v2g_y

        # 6) 비용
        om_y    = annual_om_cost
        capex_y = capex_total if i == 0 else 0.0

        # 7) 순현금흐름
        cf = annual_revenue_y - om_y - capex_y
        cum += cf

        yearly_cash.append(cf)
        cumulative.append(cum)
        pv_revenues.append(revenue_pv_y)
        v2g_revenues.append(revenue_v2g_y)
        om_costs.append(om_y)
        capex_list.append(capex_y)

    # 보고용 평균치(탄소 계산 간략화)
    avg_pv_surplus_kwh = np.mean([
        max(0.0, pv_kwh_by_year[min(max(y, min_y), max_y)] * (1 - self_use)) for y in years
    ])
    avg_v2g_kwh = np.mean([
        min(
            max(0.0, pv_kwh_by_year[min(max(y, min_y), max_y)] * (1 - self_use)) * v2g_share,
            p["num_v2g_chargers"] * p["v2g_daily_discharge_per_charger_kwh"] * p["v2g_operating_days"] * p["degradation_factor"]
        )
        for y in years
    ])

    return {
        "years": years,
        "yearly_cash": yearly_cash,
        "cumulative": cumulative,
        "pv_revenues": pv_revenues,
        "v2g_revenues": v2g_revenues,
        "om_costs": om_costs,
        "capex_list": capex_list,
        "avg_pv_surplus_kwh": avg_pv_surplus_kwh,
        "avg_v2g_kwh": avg_v2g_kwh,
    }

# =========================
# 6) Streamlit App
# =========================
def main():
    st.title("V2G + PV 경제성 (CSV 일사합 기반 • 연도별 잉여→V2G 캡 적용)")

    # --- CSV 고정 로드 (사이드바에서 경로 입력 제거) ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "jeju.csv")

    try:
        irr_df = load_irradiance_csv(csv_path)
    except Exception as e:
        st.error(f"CSV 로드 오류: {e}")
        st.stop()

    # --- 사이드바: 시스템 파라미터 ---
    st.sidebar.header("시스템/요금 설정")

    # 패널/면적/PR
    c1, c2 = st.sidebar.columns(2)
    panel_w = c1.number_input("패널 폭(m)", value=1.46, step=0.01, format="%.2f")
    panel_h = c2.number_input("패널 높이(m)", value=0.98, step=0.01, format="%.2f")
    n_modules = st.sidebar.number_input("모듈 수(장)", value=250, step=5, min_value=1)

    st.sidebar.caption("PR은 고정 계수(기본) 또는 수동 지정")
    use_manual_pr = st.sidebar.checkbox("PR 수동 지정", value=False)
    pr_manual = st.sidebar.number_input("PR (0~1)", value=0.78, min_value=0.0, max_value=1.0, step=0.01) if use_manual_pr else None
    # 고정 계수(기본)
    pr_base, availability, soiling, inv_eff = 0.82, 0.98, 0.02, 0.97

    # 시뮬레이션 기간
    install_year = st.sidebar.number_input("설치 연도", value=2025, step=1)
    current_year = st.sidebar.number_input("마지막 연도", value=2045, step=1, min_value=install_year)

    # 요금/비율
    params = make_v2g_model_params()
    params["self_use_ratio"] = st.sidebar.slider("PV 자가소비 비율", 0.0, 1.0, params["self_use_ratio"], 0.05)
    params["pv_base_price"]  = st.sidebar.number_input("PV 기준단가 (원/kWh)", value=params["pv_base_price"], step=5, min_value=0)
    params["price_cagr"]     = st.sidebar.number_input("전력단가 연평균 상승률", value=params["price_cagr"], step=0.001, format="%.3f")

    # V2G 파라미터 (요청대로 복원)
    params["num_v2g_chargers"] = st.sidebar.number_input("V2G 충전기 대수", value=params["num_v2g_chargers"], step=1, min_value=1)
    params["v2g_daily_discharge_per_charger_kwh"] = st.sidebar.number_input(
        "1대당 일일 방전량 (kWh)", value=params["v2g_daily_discharge_per_charger_kwh"], step=1, min_value=1
    )
    params["v2g_operating_days"] = st.sidebar.number_input(
        "연간 운영일 수", value=params["v2g_operating_days"], step=10, min_value=1, max_value=365
    )
    params["degradation_factor"] = st.sidebar.number_input(
        "운영 보정(열화 등)", value=params["degradation_factor"], step=0.01, min_value=0.0, max_value=1.0, format="%.2f"
    )
    params["v2g_share_of_surplus"] = st.sidebar.slider("잉여 전력 중 V2G 전환 비율", 0.0, 1.0, params["v2g_share_of_surplus"], 0.05)

    # 재무 지표 할인율
    discount_rate = st.sidebar.number_input("할인율(연)", value=0.08, min_value=0.0, max_value=0.5, step=0.005, format="%.3f")

    # --- PV 연간 kWh 계산 (CSV 기반) ---
    pv_by_year, area_m2, PR_used = compute_pv_kwh_by_year(
        irr_df,
        panel_width_m=panel_w, panel_height_m=panel_h, n_modules=int(n_modules),
        pr_base=pr_base, availability=availability, soiling=soiling, inv_eff=inv_eff,
        pr_manual=pr_manual
    )

    # --- 현금흐름 계산 ---
    cf = build_yearly_cashflows_from_csv(install_year, current_year, params, pv_by_year)
    years = cf["years"]; yearly_cash = cf["yearly_cash"]; cumulative = cf["cumulative"]

    # 손익분기 연도
    break_even_year = next((y for y, cum in zip(years, cumulative) if cum >= 0), None)

    # 탄소절감량(kgCO2e) — 평균치로 간단 추정
    total_clean_kwh = (cf["avg_pv_surplus_kwh"] + cf["avg_v2g_kwh"]) * len(years)
    total_co2_kg = total_clean_kwh * EMISSION_FACTOR_KG_PER_KWH

    # 재무지표
    npv_val = npv(discount_rate, yearly_cash)
    irr_val = irr_bisection(yearly_cash)
    dpp_val = discounted_payback(yearly_cash, discount_rate)

    # ----- KPI (1행: 재무) -----
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

    # ----- KPI (2행: 시스템/환경) -----
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

    # ----- 그래프: 누적 라인 -----
    st.subheader("누적 현금흐름")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, cumulative, marker="o", linewidth=2.2)
    ax.set_xlabel("연도"); ax.set_ylabel("누적 금액(원)")
    ax.yaxis.set_major_formatter(FuncFormatter(won_formatter))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("V2G + PV 누적 현금흐름 (CSV 기반)")
    if break_even_year is not None:
        ax.axvline(break_even_year, color="green", linestyle="--", alpha=0.7)
        ax.text(break_even_year, 0, f"손익분기 {break_even_year}", color="green", va="bottom", ha="left")
    st.pyplot(fig)

    # ----- 그래프: 연도별 누적 막대 -----
    st.subheader("연도별 순현금흐름 (누적)")
    x_labels = [f"{y}년" for y in years]
    colors = ["red" if cum < 0 else "royalblue" for cum in cumulative]
    bar_fig = go.Figure(data=[go.Bar(
        x=x_labels, y=cumulative,
        marker=dict(color=colors),
        text=[f"{v:,.0f}원" for v in cumulative],
        textposition="outside"
    )])
    if break_even_year is not None:
        be_label = f"{break_even_year}년"
        bar_fig.add_shape(type="line", x0=be_label, x1=be_label, y0=0, y1=1,
                          xref="x", yref="paper", line=dict(color="green", width=2, dash="dash"))
        bar_fig.add_annotation(x=be_label, y=1, xref="x", yref="paper",
                               text=f"손익분기 {break_even_year}년", showarrow=False,
                               yanchor="bottom", font=dict(color="green"))
    bar_fig.update_layout(title="연도별 순현금흐름 (누적)", yaxis=dict(tickformat=","), bargap=0.25)
    st.plotly_chart(bar_fig, use_container_width=True)

    # ----- 표: 연도별 금액 -----
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
        f"총 모듈 면적: {area_m2:.1f} m² | 사용 PR: {PR_used:.3f} | "
        f"CSV 연도 범위: {min(pv_by_year.keys())}~{max(pv_by_year.keys())} "
        f"(범위 밖 연도는 경계값으로 대체)"
    )

if __name__ == "__main__":
    main()
