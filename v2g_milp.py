#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2G Scheduling MILP (Pyomo)
- Inputs: Jeju PV (jeju.csv with columns: 일시, 일사합(MJ/m2)), SMP (HOME_전력거래_계통한계가격_시간별SMP.csv)
- Output: v2g_schedule_out.csv with hourly optimal schedule
- Solver: HiGHS (recommended) with fallback to CBC/GLPK
"""
import argparse
import pandas as pd
import numpy as np
import math
from pathlib import Path

from pyomo.environ import (
    ConcreteModel, Var, NonNegativeReals, Binary, Reals,
    Objective, Constraint, maximize, value, SolverFactory
)

ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]

def read_csv_smart(path):
    last_err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def ensure_datetime_index(df, col_names):
    for c in col_names:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().any():
                df = df.copy()
                df[c] = dt
                df = df.set_index(c).sort_index()
                return df
    if ("날짜" in df.columns) and ("시간" in df.columns):
        dt = pd.to_datetime(df["날짜"].astype(str) + " " + df["시간"].astype(str), errors="coerce")
        df = df.copy()
        df.index = dt
        df = df.sort_index()
        return df
    raise ValueError("Datetime column not found among: {}".format(col_names))

def daily_to_hourly_ghi(daily_ghi_kwhm2, lat_deg=33.5):
    hourly = []
    idx = []
    for d, Gd in daily_ghi_kwhm2.items():
        n = d.timetuple().tm_yday
        lat = math.radians(lat_deg)
        delta = math.radians(23.45*np.sin(np.deg2rad(360*(284+n)/365)))
        ws = np.degrees(np.arccos(-np.tan(lat)*np.tan(delta)))
        sunrise = 12 - ws/15
        sunset  = 12 + ws/15
        profile = np.zeros(24)
        hours = np.arange(24)
        mask = (hours >= np.floor(sunrise)) & (hours <= np.ceil(sunset))
        if Gd > 0 and mask.sum() > 0:
            h = hours[mask]
            x = (h - 12)/((sunset - sunrise)/2 + 1e-9)
            w = np.cos(np.pi*x/2.0)**2
            w = w / w.sum()
            profile[mask] = w
        ghih = profile * Gd
        for h in range(24):
            idx.append(d + pd.Timedelta(hours=h))
            hourly.append(ghih[h])
    return pd.Series(hourly, index=pd.DatetimeIndex(idx), name="GHI_kWhm2")

def pv_from_ghi_hourly(ghi_h_kWhm2, pv_kw=125.0, pr=0.8, tf=1.10, alpha=0.22):
    poa = ghi_h_kWhm2 * tf
    pv_kWh = pv_kw * pr * alpha * poa
    return pd.Series(pv_kWh.values, index=ghi_h_kWhm2.index, name="PV_kWh")

def load_pv_from_jeju(jeju_csv, lat_deg=33.5, pv_kw=125.0, pr=0.8, tf=1.10, alpha=0.22):
    df = read_csv_smart(./jeju.csv)
    col_ghi = None
    for cand in ["일사합(MJ/m2)","일사합(MJ/m²)","일사합","MJ_m2","GHI_MJ_m2"]:
        if cand in df.columns:
            col_ghi = cand; break
    if col_ghi is None:
        raise ValueError("일사합(MJ/m2) column not found in jeju.csv")
    df = ensure_datetime_index(df, ["일시","일자","일시(YYYY-MM-DD)","date"])
    daily_kWhm2 = (df[col_ghi].astype(float).resample("D").sum() * 0.27778).rename("GHI_kWhm2_day")
    daily_kWhm2 = daily_kWhm2.dropna()
    ghi_hourly = daily_to_hourly_ghi(daily_kWhm2, lat_deg=lat_deg)
    pv_hourly = pv_from_ghi_hourly(ghi_hourly, pv_kw=pv_kw, pr=pr, tf=tf, alpha=alpha)
    return pv_hourly

def load_smp(smp_csv):
    df = read_csv_smart(./SMP.csv)
    try_cols = ["일시","거래시간","시간","TIMESTAMP","datetime","Date","날짜"]
    df = ensure_datetime_index(df, try_cols)
    price_col = None
    for c in ["SMP","계통한계가격","SMP(원/kWh)","가격","Price","price"]:
        if c in df.columns:
            price_col = c; break
    if price_col is None:
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols)==1:
            price_col = num_cols[0]
        else:
            raise ValueError("SMP price column not found.")
    smp_hourly = df[price_col].astype(float).resample("H").mean().rename("SMP_Won_per_kWh").ffill().bfill()
    return smp_hourly

def default_availability(index, weekday_day=0.7, weekday_night=0.3, weekend_day=0.5, weekend_night=0.2):
    vals = []
    for ts in index:
        is_weekend = ts.weekday() >= 5
        hr = ts.hour
        if 9 <= hr <= 18:
            vals.append(weekend_day if is_weekend else weekday_day)
        else:
            vals.append(weekend_night if is_weekend else weekday_night)
    return pd.Series(vals, index=index, name="availability")

def solve_v2g(price, pv_kWh, load_kWh=None, availability=None,
              N=60, P_ch=11.0, P_dis=11.0,
              E_cap=2000.0, soc0=1000.0, soc_min=400.0, soc_max=1800.0,
              eta_ch=0.95, eta_dis=0.95, import_cost=140.0, deg_cost=40.0,
              P_xfmr=1000.0, P_exp=1000.0, dt=1.0):
    idx = price.index.intersection(pv_kWh.index)
    if load_kWh is not None:
        idx = idx.intersection(load_kWh.index)
    if availability is not None:
        idx = idx.intersection(availability.index)
    price = price.loc[idx]
    pv_kWh = pv_kWh.loc[idx]
    if load_kWh is None:
        load_kWh = pd.Series(0.0, index=idx)
    else:
        load_kWh = load_kWh.loc[idx]
    if availability is None:
        availability = default_availability(idx)
    else:
        availability = availability.loc[idx]

    T = len(idx)
    Mbig = N*max(P_ch,P_dis)

    m = ConcreteModel()
    m.T = range(T)
    m.x_ch  = Var(m.T, domain=NonNegativeReals)
    m.x_dis = Var(m.T, domain=NonNegativeReals)
    m.p_imp = Var(m.T, domain=NonNegativeReals)
    m.p_exp = Var(m.T, domain=NonNegativeReals)
    m.soc   = Var(m.T, domain=Reals)
    m.u     = Var(m.T, domain=Binary)

    def obj_rule(m):
        rev = sum(price.iloc[t]*m.p_exp[t]*dt for t in m.T)
        cost_imp = sum(import_cost*m.p_imp[t]*dt for t in m.T)
        cost_deg = sum(deg_cost*m.x_dis[t]*dt for t in m.T)
        return rev - cost_imp - cost_deg
    m.OBJ = Objective(rule=obj_rule, sense=maximize)

    def power_balance(m,t):
        return pv_kWh.iloc[t] + m.p_imp[t] + m.x_dis[t] == load_kWh.iloc[t] + m.x_ch[t] + m.p_exp[t]
    m.balance = Constraint(m.T, rule=power_balance)

    def ch_cap(m,t):
        return m.x_ch[t] <= availability.iloc[t] * N * P_ch
    m.ch_cap = Constraint(m.T, rule=ch_cap)

    def dis_cap(m,t):
        return m.x_dis[t] <= availability.iloc[t] * N * P_dis
    m.dis_cap = Constraint(m.T, rule=dis_cap)

    def anti1(m,t): return m.x_ch[t] <= Mbig * m.u[t]
    def anti2(m,t): return m.x_dis[t] <= Mbig * (1 - m.u[t])
    m.anti1 = Constraint(m.T, rule=anti1)
    m.anti2 = Constraint(m.T, rule=anti2)

    def soc_dyn(m,t):
        if t==0:
            return m.soc[t] == soc0 + eta_ch*m.x_ch[t]*dt - (1/eta_dis)*m.x_dis[t]*dt
        return m.soc[t] == m.soc[t-1] + eta_ch*m.x_ch[t]*dt - (1/eta_dis)*m.x_dis[t]*dt
    m.soc_dyn = Constraint(m.T, rule=soc_dyn)

    def soc_lo(m,t): return m.soc[t] >= soc_min
    def soc_hi(m,t): return m.soc[t] <= soc_max
    m.soc_lo = Constraint(m.T, rule=soc_lo)
    m.soc_hi = Constraint(m.T, rule=soc_hi)

    def xfmr_lim(m,t): return m.p_imp[t] <= P_xfmr
    def exp_lim(m,t):  return m.p_exp[t] <= P_exp
    m.xfmr = Constraint(m.T, rule=xfmr_lim)
    m.expr = Constraint(m.T, rule=exp_lim)

    solver = None
    last_err = None
    for solver_name in ["highs", "cbc", "glpk"]:
        try:
            solver = SolverFactory(solver_name)
            res = solver.solve(m, tee=False)
            break
        except Exception as e:
            last_err = e
            solver = None
    if solver is None:
        raise RuntimeError(f"No MILP solver available. Install HiGHS (highspy) or CBC/GLPK. Last error: {last_err}")

    res_df = pd.DataFrame({
        "time": idx,
        "price_Won_per_kWh": price.values,
        "pv_kWh": pv_kWh.values,
        "load_kWh": load_kWh.values,
        "avail": availability.values,
        "charge_kW": [m.x_ch[t].value for t in m.T],
        "discharge_kW": [m.x_dis[t].value for t in m.T],
        "import_kW": [m.p_imp[t].value for t in m.T],
        "export_kW": [m.p_exp[t].value for t in m.T],
        "soc_kWh": [m.soc[t].value for t in m.T],
        "mode_u": [m.u[t].value for t in m.T],
    })
    res_df["revenue_Won"] = res_df["export_kW"]*res_df["price_Won_per_kWh"]
    res_df["import_cost_Won"] = res_df["import_kW"]*140.0
    res_df["deg_cost_Won"] = res_df["discharge_kW"]*40.0
    res_df["profit_Won"] = res_df["revenue_Won"] - res_df["import_cost_Won"] - res_df["deg_cost_Won"]
    res_df["cum_profit_Won"] = res_df["profit_Won"].cumsum()
    obj_profit = float(value(m.OBJ))

    return {"schedule": res_df, "profit": obj_profit}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smp", type=str, default="HOME_전력거래_계통한계가격_시간별SMP.csv")
    ap.add_argument("--pv", type=str, default="jeju.csv")
    ap.add_argument("--lat", type=float, default=33.5)
    ap.add_argument("--install_kw", type=float, default=125.0)
    ap.add_argument("--pr", type=float, default=0.80)
    ap.add_argument("--tf", type=float, default=1.10)
    ap.add_argument("--alpha", type=float, default=0.22)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--p_ch", type=float, default=11.0)
    ap.add_argument("--p_dis", type=float, default=11.0)
    ap.add_argument("--cap_kwh", type=float, default=2000.0)
    ap.add_argument("--soc0", type=float, default=1000.0)
    ap.add_argument("--soc_min", type=float, default=400.0)
    ap.add_argument("--soc_max", type=float, default=1800.0)
    ap.add_argument("--eta_ch", type=float, default=0.95)
    ap.add_argument("--eta_dis", type=float, default=0.95)
    ap.add_argument("--import_cost", type=float, default=140.0)
    ap.add_argument("--deg_cost", type=float, default=40.0)
    ap.add_argument("--p_xfmr", type=float, default=1000.0)
    ap.add_argument("--p_exp", type=float, default=1000.0)
    args = ap.parse_args()

    smp_path = Path(args.smp)
    pv_path  = Path(args.pv)
    if smp_path.exists():
        price = load_smp(str(smp_path))
    else:
        hours = pd.date_range("2025-06-01", periods=168, freq="H")
        price = pd.Series(150 + 70*np.sin(np.linspace(0, 4*np.pi, len(hours))), index=hours, name="SMP_Won_per_kWh")
    if pv_path.exists():
        pv_hourly = load_pv_from_jeju(str(pv_path), lat_deg=args.lat, pv_kw=args.install_kw, pr=args.pr, tf=args.tf, alpha=args.alpha)
    else:
        idx = price.index
        pv_hourly = pd.Series(np.maximum(0, 200*np.sin((idx.hour-6)/6.0*np.pi)), index=idx, name="PV_kWh")

    result = solve_v2g(
        price=price, pv_kWh=pv_hourly, load_kWh=None, availability=None,
        N=args.n, P_ch=args.p_ch, P_dis=args.p_dis, E_cap=args.cap_kwh,
        soc0=args.soc0, soc_min=args.soc_min, soc_max=args.soc_max,
        eta_ch=args.eta_ch, eta_dis=args.eta_dis,
        import_cost=args.import_cost, deg_cost=args.deg_cost,
        P_xfmr=args.p_xfmr, P_exp=args.p_exp
    )
    out_csv = "v2g_schedule_out.csv"
    result["schedule"].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Optimal profit (₩):", round(result["profit"], 1))
    print("Wrote:", out_csv)

if __name__ == "__main__":
    main()
