import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timezone

st.set_page_config(page_title="Microgrid Control MVP", layout="wide")

BACKEND = st.secrets.get("backend_url", "http://localhost:8000")
site_id_default = "accra_office"

st.title("Microgrid Control — MVP")
st.caption("Decision engine + interface layer | Forecast → Optimize → Dispatch (simulated)")

with st.sidebar:
    st.header("Config")
    backend = st.text_input("Backend URL", value=BACKEND)
    site_id = st.text_input("Site ID", value=site_id_default)
    horizon = st.number_input("Horizon (steps)", min_value=4, max_value=96, value=24)
    interval_min = st.selectbox("Interval (min)", options=[5, 10, 15, 30, 60], index=2)
    min_soc = st.slider("Min SOC", 0.0, 1.0, 0.2, step=0.05)
    grid_cap = st.number_input("Grid Import Cap (kW, 0 = none)", min_value=0, value=0)
    allow_export = st.checkbox("Allow Export", value=False)

st.write("### 1) Synthetic Data")
colA, colB, colC = st.columns(3)
with colA:
    if st.button("Generate Synthetic Data (CLI)"):
        st.info("Run the generator from the simulator/ folder as described in README.")
with colB:
    uploaded = st.file_uploader("Upload telemetry JSON (from generator)", type=["json"], accept_multiple_files=False)
with colC:
    if st.button("Load Latest Data from File (parse)") and uploaded:
        rows = json.loads(uploaded.getvalue().decode("utf-8"))
        r = requests.post(f"{backend}/ingest/telemetry", json={"rows": rows})
        st.write(r.json())

st.write("### 2) Forecasts")
c1, c2 = st.columns(2)
with c1:
    if st.button("Run Load Forecast"):
        r = requests.get(f"{backend}/forecast/load", params={"site_id": site_id, "h": horizon, "interval_min": interval_min})
        f_load = r.json()
        st.session_state["f_load"] = f_load
        st.success("Load forecast ready.")
        st.json(f_load)
with c2:
    if st.button("Run PV Forecast"):
        r = requests.get(f"{backend}/forecast/pv", params={"site_id": site_id, "h": horizon, "interval_min": interval_min})
        f_pv = r.json()
        st.session_state["f_pv"] = f_pv
        st.success("PV forecast ready.")
        st.json(f_pv)

st.write("### 3) Optimization")
price_default = [0.15]*int(horizon)
default_price_text = st.session_state.get("price_str") or ",".join([str(x) for x in price_default])
price_str = st.text_area("Price vector (USD/kWh, comma-separated)", value=default_price_text, height=80)

if st.button("Run Optimization Plan"):
    f_load = st.session_state.get("f_load")
    f_pv = st.session_state.get("f_pv")
    if not f_load or not f_pv:
        st.error("Please run forecasts first.")
    else:
        try:
            price = [float(x.strip()) for x in price_str.split(",") if x.strip()][:len(f_load["values_kw"])]
            start_ts = datetime.now(timezone.utc).isoformat()
            payload = {
                "site_id": site_id,
                "start_ts": start_ts,
                "interval_min": interval_min,
                "price_usd_per_kwh": price,
                "load_kw": f_load["values_kw"],
                "pv_kw": f_pv["values_kw"],
                "init_soc": 0.5,
                "batt_max_charge_kw": 100.0,
                "batt_max_discharge_kw": 100.0,
                "batt_capacity_kwh": 400.0,
                "eta_c": 0.95,
                "eta_d": 0.95,
                "min_soc": min_soc,
                "grid_cap_kw": (grid_cap if grid_cap>0 else None),
                "allow_export": allow_export,
                "degr_cost_usd_per_kwh": 0.02
            }
            r = requests.post(f"{backend}/optimize/plan", json=payload)
            plan = r.json()
            st.session_state["plan"] = plan
            st.success("Plan computed.")
            st.json(plan)
        except Exception as e:
            st.error(f"Failed to compute plan: {e}")

st.write("### 4) Last Plan")
if st.button("Fetch Last Plan"):
    r = requests.get(f"{backend}/plan/last", params={"site_id": site_id})
    st.json(r.json())

st.write("---")
st.caption("Tip: This MVP uses a simple seasonal forecaster and an LP optimizer with a rule-based fallback for reliability on new systems.")


st.write("### 5) Publish Commands (MQTT)")
colx, coly = st.columns(2)
with colx:
    broker = st.text_input("MQTT Broker Host", value="localhost")
with coly:
    broker_port = st.number_input("MQTT Port", value=1883, min_value=1, max_value=65535)
plan = st.session_state.get("plan")
if st.button("Publish Battery Command (first step)"):
    if not plan:
        st.error("No plan available. Run optimization first.")
    else:
        first = plan["schedule"][0]
        payload = {
            "broker_host": broker,
            "broker_port": broker_port,
            "site_id": plan["site_id"],
            "asset_id": "batt_1",
            "cmd_type": "set_power_kw",
            "value": float(first["p_batt_kw"]),
            "valid_until": plan["start_ts"]
        }
        r = requests.post(f"{backend}/commands/publish", json=payload)
        st.json(r.json())

if st.button("Publish Diesel Command (first step)"):
    if not plan:
        st.error("No plan available. Run optimization first.")
    else:
        first = plan["schedule"][0]
        p_diesel = float(first.get("p_diesel_kw", 0.0))
        payload = {
            "broker_host": broker,
            "broker_port": broker_port,
            "site_id": plan["site_id"],
            "asset_id": "diesel_1",
            "cmd_type": "set_power_kw",
            "value": p_diesel,
            "valid_until": plan["start_ts"]
        }
        r = requests.post(f"{backend}/commands/publish", json=payload)
        st.json(r.json())


st.write("### Tariff Editor")
with st.expander("Edit Tariff (Time-of-Use, demand charge, diesel cost)"):
    # Load current tariff
    if st.button("Load Tariff"):
        r = requests.get(f"{backend}/tariff/get", params={"site_id": site_id})
        if r.ok:
            st.session_state["tariff"] = r.json()
            st.success("Tariff loaded.")
            st.json(st.session_state["tariff"])
        else:
            st.error("Failed to load tariff")

    t = st.session_state.get("tariff") or {
        "site_id": site_id,
        "timezone": "UTC",
        "weekday_periods": [{"name":"Off-peak","start_hour":0,"end_hour":8,"price_usd_per_kwh":0.12},
                            {"name":"Mid","start_hour":8,"end_hour":16,"price_usd_per_kwh":0.18},
                            {"name":"Peak","start_hour":16,"end_hour":21,"price_usd_per_kwh":0.28},
                            {"name":"Off-peak","start_hour":21,"end_hour":24,"price_usd_per_kwh":0.12}],
        "weekend_periods": None,
        "demand_charge_lambda": 5.0,
        "diesel_cost_usd_per_kwh": 0.35
    }

    tz = st.text_input("Timezone (IANA)", value=t.get("timezone","UTC"))
    st.write("Weekday Periods")
    # Render simple inputs for up to 4 periods
    wp = t.get("weekday_periods", [])
    while len(wp) < 4: wp.append({"name":f"P{len(wp)+1}","start_hour":0,"end_hour":24,"price_usd_per_kwh":0.15})
    cols = st.columns([2,1,1,1])
    new_wp = []
    for i in range(4):
        with st.container():
            with cols[0]:
                name = st.text_input(f"W Name {i+1}", value=wp[i]["name"], key=f"wname{i}")
            with cols[1]:
                sh = st.number_input(f"W Start {i+1}", min_value=0, max_value=23, value=int(wp[i]["start_hour"]), key=f"wsh{i}")
            with cols[2]:
                eh = st.number_input(f"W End {i+1}", min_value=1, max_value=24, value=int(wp[i]["end_hour"]), key=f"weh{i}")
            with cols[3]:
                pr = st.number_input(f"W Price {i+1}", min_value=0.0, value=float(wp[i]["price_usd_per_kwh"]), key=f"wpr{i}")
        new_wp.append({"name":name,"start_hour":int(sh),"end_hour":int(eh),"price_usd_per_kwh":float(pr)})

    weekend_enable = st.checkbox("Enable weekend periods", value=bool(t.get("weekend_periods")))
    new_we = None
    if weekend_enable:
        we = t.get("weekend_periods") or new_wp
        while len(we) < 4: we.append({"name":f"P{len(we)+1}","start_hour":0,"end_hour":24,"price_usd_per_kwh":0.12})
        cols2 = st.columns([2,1,1,1])
        new_we = []
        for i in range(4):
            with cols2[0]:
                name = st.text_input(f"WE Name {i+1}", value=we[i]["name"], key=f"wename{i}")
            with cols2[1]:
                sh = st.number_input(f"WE Start {i+1}", min_value=0, max_value=23, value=int(we[i]["start_hour"]), key=f"wesh{i}")
            with cols2[2]:
                eh = st.number_input(f"WE End {i+1}", min_value=1, max_value=24, value=int(we[i]["end_hour"]), key=f"weeh{i}")
            with cols2[3]:
                pr = st.number_input(f"WE Price {i+1}", min_value=0.0, value=float(we[i]["price_usd_per_kwh"]), key=f"wepr{i}")
            new_we.append({"name":name,"start_hour":int(sh),"end_hour":int(eh),"price_usd_per_kwh":float(pr)})

    demand_lambda = st.number_input("Demand charge proxy (lambda, USD/kW)", min_value=0.0, value=float(t.get("demand_charge_lambda",5.0)))
    diesel_cost = st.number_input("Diesel cost (USD/kWh)", min_value=0.0, value=float(t.get("diesel_cost_usd_per_kwh",0.35)))

    if st.button("Save Tariff"):
        body = {
            "site_id": site_id,
            "timezone": tz,
            "weekday_periods": new_wp,
            "weekend_periods": new_we if weekend_enable else None,
            "demand_charge_lambda": float(demand_lambda),
            "diesel_cost_usd_per_kwh": float(diesel_cost)
        }
        r = requests.post(f"{backend}/tariff/set", json=body)
        if r.ok:
            st.session_state["tariff"] = r.json()
            st.success("Tariff saved.")
            st.json(r.json())
        else:
            st.error("Failed to save tariff.")

    st.write("Generate Price Vector")
    if st.button("Build Prices for Next Horizon From Tariff"):
        start_ts = datetime.now(timezone.utc).isoformat()
        r = requests.get(f"{backend}/tariff/price_vector", params={
            "site_id": site_id, "start_ts": start_ts, "horizon": int(horizon), "interval_min": int(interval_min)
        })
        if r.ok:
            vals = r.json().get("values_usd_per_kwh", [])
            st.session_state["built_prices"] = vals
            st.success("Built prices from tariff. Injected into optimizer input area below.")
            # inject into the optimization input
            st.session_state["price_str"] = ",".join([str(x) for x in vals])
        else:
            st.error("Failed to build price vector from tariff.")
