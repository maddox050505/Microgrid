# backend/app.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import math

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Microgrid MVP Backend")

# In-memory storage for tariffs (by site_id)
TARIFFS: Dict[str, Dict[str, Any]] = {}


# ---------------------------
# Helpers
# ---------------------------

def _parse_iso(ts: str) -> datetime:
    """Parse ISO string, force UTC."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _build_tou_price_vector(start_ts: str, horizon: int, interval_min: int) -> Dict[str, Any]:
    """
    Simple TOU (time-of-use) price curve:
    - Off-peak (0–7, 19–24): $0.14/kWh
    - Peak (7–19):          $0.28/kWh
    """
    start = _parse_iso(start_ts)
    step = timedelta(minutes=interval_min)
    values: List[float] = []

    for i in range(horizon):
        t = start + i * step
        if 7 <= t.hour < 19:
            price = 0.28
        else:
            price = 0.14
        values.append(price)

    return {
        "start_ts": start.isoformat(),
        "interval_min": interval_min,
        "values_usd_per_kwh": values,
    }


def _synthetic_load_and_pv(
    horizon: int, interval_min: int, application: str = "commercial"
) -> Dict[str, Dict[str, Any]]:
    """
    Generate synthetic load (kW) and PV (kW) profiles, similar to your
    Streamlit fallback.
    """
    step = timedelta(minutes=interval_min)
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    app_lower = (application or "").lower()

    base_kw = 2.0 if "res" in app_lower else (20.0 if "com" in app_lower else 100.0)

    load_vals: List[float] = []
    pv_vals: List[float] = []

    for i in range(horizon):
        t = start + i * step
        # normalized position in the day
        frac = (t.hour + t.minute / 60.0) / 24.0 * 2 * math.pi

        # daily cycle
        day_cycle = 0.6 + 0.5 * math.sin(frac - math.pi / 2)
        evening_bump = 0.25 * math.exp(-0.5 * ((t.hour - 19) / 2.0) ** 2)
        load = max(0.1, base_kw * (0.7 + 0.5 * day_cycle) + base_kw * evening_bump)

        # PV peak at 13:00
        pv_peak_kw = 0.6 if "res" in app_lower else 5.0
        pv_shape = math.exp(-0.5 * ((t.hour + t.minute / 60.0 - 13.0) / 3.0) ** 2)
        pv = max(0.0, pv_peak_kw * pv_shape)

        load_vals.append(float(load))
        pv_vals.append(float(pv))

    load = {
        "start_ts": start.isoformat(),
        "interval_min": interval_min,
        "values_kw": load_vals,
    }
    pv = {
        "start_ts": start.isoformat(),
        "interval_min": interval_min,
        "values_kw": pv_vals,
    }

    return {"load": load, "pv": pv}


# ---------------------------
# Tariff endpoints
# ---------------------------

@app.post("/tariff/set")
async def set_tariff(payload: Dict[str, Any]):
    """
    Store a tariff definition. The UI sends:
    {
      "site_id": "...",
      "timezone": "UTC",
      "weekday_periods": [...],
      "weekend_periods": null / [...],
      "demand_charge_lambda": ...,
      "diesel_cost_usd_per_kwh": ...
    }
    """
    site_id = payload.get("site_id") or "site"
    TARIFFS[site_id] = payload
    return {"status": "ok", "site_id": site_id}


@app.get("/tariff/price_vector")
async def get_price_vector(
    site_id: str,
    start_ts: str,
    horizon: int = 24,
    interval_min: int = 15,
):
    """
    Return a price vector for this site_id. For now, we ignore the stored
    tariff and just return a simple TOU curve. The Streamlit UI only needs
    a vector of prices.
    """
    if horizon <= 0:
        raise HTTPException(status_code=400, detail="horizon must be > 0")
    if interval_min <= 0:
        raise HTTPException(status_code=400, detail="interval_min must be > 0")

    return _build_tou_price_vector(start_ts=start_ts, horizon=horizon, interval_min=interval_min)


# ---------------------------
# Forecast endpoints
# ---------------------------

@app.post("/forecast/load")
async def forecast_load(payload: Dict[str, Any]):
    """
    Request body example:
    {
      "site_id": "north_america_01803_commercial",
      "horizon": 24,
      "interval_min": 15
    }
    """
    site_id = payload.get("site_id", "site")
    horizon = int(payload.get("horizon", 24))
    interval_min = int(payload.get("interval_min", 15))

    if horizon <= 0 or interval_min <= 0:
        raise HTTPException(status_code=400, detail="Invalid horizon or interval_min")

    # For now we ignore site_id and application in forecasts and just use a generic pattern.
    synthetic = _synthetic_load_and_pv(horizon=horizon, interval_min=interval_min, application=site_id)
    return synthetic["load"]


@app.post("/forecast/pv")
async def forecast_pv(payload: Dict[str, Any]):
    """
    Same request shape as /forecast/load.
    """
    site_id = payload.get("site_id", "site")
    horizon = int(payload.get("horizon", 24))
    interval_min = int(payload.get("interval_min", 15))

    if horizon <= 0 or interval_min <= 0:
        raise HTTPException(status_code=400, detail="Invalid horizon or interval_min")

    synthetic = _synthetic_load_and_pv(horizon=horizon, interval_min=interval_min, application=site_id)
    return synthetic["pv"]


# ---------------------------
# Optimization endpoint
# ---------------------------

@app.post("/optimize/plan")
async def optimize_plan(payload: Dict[str, Any]):
    """
    Request body example:
    {
      "site_id": "site",
      "horizon": 96,
      "interval_min": 15,
      "price_usd_per_kwh": [...],
      "load_kw": [...],
      "pv_kw": [...]
    }
    Simplest possible "optimization":
    - Use PV first.
    - Remaining load comes from the grid.
    - No battery or diesel for now (but we structure schedule so UI can expand later).
    """
    site_id = payload.get("site_id", "site")
    horizon = int(payload.get("horizon", 0))
    interval_min = int(payload.get("interval_min", 15))

    prices = payload.get("price_usd_per_kwh") or []
    load_kw = payload.get("load_kw") or []
    pv_kw = payload.get("pv_kw") or []

    H = min(len(prices), len(load_kw), len(pv_kw))
    if horizon > 0:
        H = min(H, horizon)

    if H <= 0:
        raise HTTPException(status_code=400, detail="No valid horizon from input data.")

    schedule: List[Dict[str, Any]] = []
    soc = 0.5  # dummy state of charge

    for i in range(H):
        load = float(load_kw[i])
        pv = float(pv_kw[i])
        # First use PV to cover load; we ignore feed-in for now
        grid = max(load - pv, 0.0)

        row = {
            "t_idx": i,
            "p_grid_kw": grid,
            "p_batt_kw": 0.0,   # no battery yet
            "p_diesel_kw": 0.0, # no generator
            "soc": soc,
        }
        schedule.append(row)

    start_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()

    return {
        "site_id": site_id,
        "start_ts": start_ts,
        "interval_min": interval_min,
        "schedule": schedule,
    }


# ---------------------------
# Savings endpoint
# ---------------------------

@app.post("/analyze/savings")
async def analyze_savings(payload: Dict[str, Any]):
    """
    Request body example (from UI):
    {
      "site_id": "...",
      "interval_min": 15,
      "price_usd_per_kwh": [...],
      "load_kw": [...],
      "pv_kw": [...],
      "schedule": [...],
      "diesel_cost_usd_per_kwh": 0.35,
      "demand_charge_lambda": 5.0,
      "degr_cost_usd_per_kwh": 0.02
    }
    """
    prices = payload.get("price_usd_per_kwh") or []
    load_kw = payload.get("load_kw") or []
    pv_kw = payload.get("pv_kw") or []
    schedule = payload.get("schedule") or []
    interval_min = int(payload.get("interval_min", 15))

    H = min(len(prices), len(load_kw), len(pv_kw), len(schedule))
    if H <= 0:
        raise HTTPException(status_code=400, detail="Insufficient data to analyze savings.")

    step_hours = interval_min / 60.0

    baseline_cost = 0.0
    optimized_cost = 0.0

    for i in range(H):
        price = float(prices[i])
        load = float(load_kw[i])
        pv = float(pv_kw[i])

        # Baseline: assume grid supplies load minus PV (no storage)
        base_grid = max(load - pv, 0.0)
        baseline_cost += base_grid * price * step_hours

        sch = schedule[i]
        grid_opt = float(sch.get("p_grid_kw", 0.0))
        optimized_cost += grid_opt * price * step_hours

    savings_abs = baseline_cost - optimized_cost
    savings_pct = (savings_abs / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

    return {
        "baseline_cost": baseline_cost,
        "optimized_cost": optimized_cost,
        "savings_abs": savings_abs,
        "savings_pct": savings_pct,
    }
