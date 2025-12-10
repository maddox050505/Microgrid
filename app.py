

# ui/streamlit_app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

# =========================
# Imports & page config
# =========================
import io
import os
import re
import json
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import random
import streamlit as st
import pandas as pd
import base64
import json
import streamlit as st
from openai import OpenAI

from PIL import Image, ImageFilter, ImageOps 

import streamlit as st
from datetime import date

import streamlit as st

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not set. Please configure it in Render.")
    st.stop()

client = OpenAI(api_key=api_key)

def inject_geolocator():
    st.markdown("""
    <script>
        // Request browser location
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const coords = pos.coords.latitude + "," + pos.coords.longitude;
                window.parent.postMessage({type: "geo", coords: coords}, "*");
            },
            (err) => {
                window.parent.postMessage({type: "geo", coords: null}, "*");
            }
        );
    </script>
    """, unsafe_allow_html=True)

    # Listen for JS → Streamlit message
    geo = st.session_state.get("geo_coords")

    msg = st.experimental_get_query_params().get("geo_message")
    if msg and not geo:
        st.session_state["geo_coords"] = msg[0]

HARD_OPTIMIZATION_BACKEND_ONLY = True

st.set_page_config(page_title="Microgrid MVP", layout="wide")

st.markdown("""
<style>
/* ===== Global layout & background ===== */
html, body, .stApp {
  background: radial-gradient(circle at top, #0f172a 0, #020617 48%, #000 100%) !important;
  color: #f9fafb !important;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
/* Center main app column */
.main-shell {
  max-width: 1120px;
  margin: 0 auto;
  padding: 24px 8px 40px;
}
/* Remove extra padding around app */
.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 1.5rem !important;
}
/* ===== Typography ===== */
h1, h2, h3, h4, h5, h6 {
  color: #f9fafb !important;
  letter-spacing: 0.03em;
}
p, label, span, div {
  color: #e5e7eb !important;
}
/* ===== Inputs & widgets ===== */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] div,
.stDateInput input {
  background-color: #020617 !important;
  color: #f9fafb !important;
  border-radius: 10px !important;
  border: 1px solid #1f2937 !important;
  padding: 6px 10px !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
  border-color: #4f46e5 !important;
  box-shadow: 0 0 0 1px #4f46e5 !important;
}
/* Selectbox label spacing */
label[data-testid="stWidgetLabel"] {
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #9ca3af !important;
}
/* ===== Buttons ===== */
.stButton button {
  background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%) !important;
  color: #f9fafb !important;
  border: none !important;
  border-radius: 999px !important;
  padding: 0.55rem 1.4rem !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  box-shadow: 0 10px 25px rgba(37, 99, 235, 0.35) !important;
}
.stButton button:hover {
  filter: brightness(1.06);
  transform: translateY(-1px);
}
.stButton button:active {
  transform: translateY(0);
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.8) !important;
}
/* ===== Tables & dataframes ===== */
[data-testid="stDataFrameResizable"],
[data-testid="stTable"] {
  border-radius: 10px !important;
  border: 1px solid #1f2937 !important;
  background: rgba(15, 23, 42, 0.85) !important;
}
/* ===== Metrics & alerts ===== */
[data-testid="stMetricValue"] {
  color: #f9fafb !important;
  font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
  color: #9ca3af !important;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.7rem;
}
.stAlert {
  background: rgba(15, 23, 42, 0.9) !important;
  border-radius: 12px !important;
  border: 1px solid #1f2937 !important;
}
/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
  background: #020617 !important;
  border-right: 1px solid #111827 !important;
}
section[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
}
section[data-testid="stSidebar"] .stButton button {
  width: 100%;
  box-shadow: none !important;
}
/* ===== Code blocks ===== */
pre, code {
  background: #020617 !important;
  color: #e5e7eb !important;
  border-radius: 8px !important;
}
/* ===== Progress pills (your wizard) ===== */
.progress-pill {
  background: #111827;
  color: #e5e7eb;
  border-radius: 999px;
  border: 1px solid #1f2937;
}
.progress-pill.active {
  background: linear-gradient(135deg, #4f46e5, #0ea5e9);
  color: #f9fafb;
  border-color: transparent;
}
.progress-pill.done {
  background: #16a34a;
  color: #ecfdf5;
  border-color: transparent;
}
/* ===== Hide Streamlit chrome & links ===== */
#MainMenu { display: none !important; }
header { display: none !important; }
footer { display: none !important; }
[data-testid="stAppToolbar"] { display: none !important; }
a[href^="https://streamlit.io"],
a[href^="https://docs.streamlit.io"],
a[href^="https://discuss.streamlit.io"],
a[href^="https://github.com/streamlit"] {
  display: none !important;
}
footer div a { display: none !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Optional libs (silent)
# -------------------------
pdfplumber = None
convert_from_bytes = None
Image = None
pytesseract = None
camelot = None  # optional PDF table reader
tabula = None   # optional PDF table reader

try:
    import pdfplumber  # type: ignore
except Exception:
    pass
try:
    from pdf2image import convert_from_bytes  # type: ignore
except Exception:
    pass
try:
    from PIL import Image  # type: ignore
except Exception:
    pass
try:
    import pytesseract  # type: ignore
    import platform as _plat
    if _plat.system() == "Darwin":
        for _p in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
            if os.path.exists(_p):
                pytesseract.pytesseract.tesseract_cmd = _p
                break
except Exception:
    pass
try:
    import camelot  # type: ignore
except Exception:
    pass
try:
    import tabula  # type: ignore
except Exception:
    pass

# =========================
# Globals / constants
# =========================
REGION_OPTIONS = ["", "North America", "Europe", "Africa", "Asia", "South America", "Oceania", "Other"]
APP_OPTIONS = ["", "Residential", "Commercial", "Industrial", "Public/Institutional"]

def get_energy_mode() -> str:
    prof = st.session_state.get("profile", {}) or {}
    return prof.get("energy_mode", "Electricity only")

# 🔌 Dual-mode energy selector
ENERGY_MODES = ["Electricity only", "Electricity + Natural Gas"]

WIZ_STEPS = ["profile", "upload", "tariff_prices", "forecasts", "optimize", "dashboard"]
INTRO_PAGES = ["intro", "compare", "actions"]

# ---- BACKEND CONFIG (HARD-CODED) ----
BACKEND = "https://amadinm-badkendinatorr.hf.space"  # <- FastAPI backend Space
API_KEY = os.getenv("API_KEY", "")
session = requests.Session()
if API_KEY:
    session.headers.update({"x-api-key": API_KEY})

st.caption(f"Backend URL in use: {BACKEND}")

def _backend_base() -> str:
    # Always use the same base for all API calls
    return BACKEND

# =========================
# Session init & nav helpers
# =========================
def init_state():
    ss = st.session_state

    ss.setdefault("step", "login")
    ss.setdefault("authenticated", False)
    ss.setdefault("username", "")

    prof = ss.setdefault("profile", {})
    for k, v in {
        "region": "",
        "zip": "",
        "application": "",
        "interval_minutes": 15,
        "energy_mode": "Electricity only",
    }.items():
        prof.setdefault(k, v)

    ss.setdefault("site_id", "site")
    ss.setdefault("bill_usage_df", None)
    ss.setdefault("bill_fields", {})
    ss.setdefault("price_vector", None)
    ss.setdefault("forecast_load", None)
    ss.setdefault("forecast_pv", None)
    ss.setdefault("plan", None)
    ss.setdefault("last_savings", None)
    ss.setdefault("res_mode", False)
    ss.setdefault("last_tariff", {})
    ss.setdefault("parse_log", [])
    ss.setdefault("bill_monthly_cost", 0.0)
    ss.setdefault("bill_monthly_kwh", 0.0)

# ✅ call this ONCE at module import time, before login_gate()
init_state()

# ---------- Simple per-user persistence (disk snapshot) ----------
PERSIST_KEYS = [
    "profile",
    "site_id",
    "bill_usage_df",
    "bill_fields",
    "price_vector",
    "forecast_load",
    "forecast_pv",
    "plan",
    "last_savings",
    "bill_monthly_cost",
    "bill_monthly_kwh",
    "prev_bill_monthly_cost",
    "prev_bill_monthly_kwh",
]

def _user_data_path() -> Optional[str]:
    user = st.session_state.get("username")
    if not user:
        return None
    # store under a small local folder
    return os.path.join("user_data", f"{user}.json")

def _handle_js_message():
    params = st.experimental_get_query_params()
    msg = params.get("geo_message")
    if msg:
        st.session_state["geo_coords"] = msg[0]

def save_user_snapshot():
    """
    Save key analysis state for the logged-in user so that on next login
    we can jump straight back to their bill + charts.
    """
    path = _user_data_path()
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    snap = {k: st.session_state.get(k) for k in PERSIST_KEYS}
    try:
        with open(path, "w") as f:
            json.dump(snap, f, default=str)
    except Exception:
        # fail silently – UX shouldn’t break if disk write fails
        pass

def load_user_snapshot() -> bool:
    """
    Load previously saved state for this user into session_state.
    Returns True if something was loaded, False otherwise.
    """
    path = _user_data_path()
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            snap = json.load(f)
        if isinstance(snap, dict):
            for k, v in snap.items():
                st.session_state[k] = v
        return True
    except Exception:
        return False

backend_base_now = _backend_base()
st.caption(f"Backend base resolved to: {backend_base_now}")

def _roll_bill_history():
    """
    When a new bill is confirmed, move the current bill into 'previous'
    so we can compare month-over-month on the dashboard.
    """
    s = st.session_state
    cur_cost = s.get("bill_monthly_cost")
    cur_kwh = s.get("bill_monthly_kwh")

    if cur_cost is not None:
        s["prev_bill_monthly_cost"] = cur_cost
        s["prev_bill_monthly_kwh"] = cur_kwh

def _log(msg: str):
    st.session_state["parse_log"].append(msg)

def _populate_zip_from_backend():
    """
    Ask the backend (and, if needed, ipapi.co) once per session for a best-guess ZIP.
    Only fills profile['zip'] if it's currently empty.
    Also syncs into the zip text_input widget state so it actually shows up.
    """
    # Avoid repeated network calls
    if st.session_state.get("_backend_geo_done", False):
        return
    st.session_state["_backend_geo_done"] = True

    prof = st.session_state.setdefault("profile", {})

    # If user already has a ZIP in profile or widget, don't override
    if prof.get("zip") or st.session_state.get("profile_zip_input"):
        return

    guessed_zip = ""

    # --- Try backend first ---
    try:
        base = _backend_base()
        resp = session.get(f"{base}/geo/guess_zip", timeout=3)
        if resp.ok:
            data = resp.json()
            guessed_zip = str(data.get("zip") or "").strip()
    except Exception:
        pass

    # --- Fallback: direct ipapi from Streamlit space ---
    if not guessed_zip:
        try:
            r2 = requests.get("https://ipapi.co/json", timeout=3)
            if r2.ok:
                d2 = r2.json()
                guessed_zip = str(d2.get("postal") or "").strip()
        except Exception:
            pass

    # If we got something, push it into profile + widget state
    if guessed_zip:
        prof["zip"] = guessed_zip
        st.session_state["profile"] = prof
        # set widget state so the text_input shows it
        if "profile_zip_input" not in st.session_state:
            st.session_state["profile_zip_input"] = guessed_zip

def step_index() -> int:
    return WIZ_STEPS.index(st.session_state["step"]) if st.session_state["step"] in WIZ_STEPS else 0

def go_step(s: str):
    st.session_state["step"] = s
    st.rerun()

def _local_savings_from_plan() -> Optional[dict]:
    """
    Fallback: compute baseline vs optimized cost directly from
    price_vector + forecast_load + plan.schedule.
    """
    s = st.session_state
    prices = (s.get("price_vector") or {}).get("values_usd_per_kwh")
    load   = (s.get("forecast_load")  or {}).get("values_kw")
    plan   = s.get("plan")

    if not (prices and load and plan and plan.get("schedule")):
        return None

    H = min(len(prices), len(load), len(plan["schedule"]))
    if H <= 0:
        return None

    interval_min = int(s.get("profile", {}).get("interval_minutes", 15))
    step_hours = interval_min / 60.0

    prices_arr = np.array(prices[:H], dtype=float)
    load_arr   = np.array(load[:H], dtype=float)
    grid_arr   = np.array([float(row.get("p_grid_kw", 0.0)) for row in plan["schedule"][:H]])

    baseline_cost  = float(np.sum(prices_arr * load_arr * step_hours))
    optimized_cost = float(np.sum(prices_arr * grid_arr * step_hours))
    savings_abs    = baseline_cost - optimized_cost
    savings_pct    = (savings_abs / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

    return {
        "baseline_cost": baseline_cost,
        "optimized_cost": optimized_cost,
        "savings_abs": savings_abs,
        "savings_pct": savings_pct,
    }

def go_next():
    i = step_index()
    if i < len(WIZ_STEPS) - 1:
        go_step(WIZ_STEPS[i + 1])

def go_prev():
    i = step_index()
    if i > 0:
        go_step(WIZ_STEPS[i - 1])

def render_progress():
    labels = ["Profile", "Upload", "Tariff", "Forecasts", "Optimize", "Dashboard"]
    i = step_index()
    cols = st.columns(len(labels))
    for idx, c in enumerate(cols):
        with c:
            active = idx == i
            done = idx < i
            pill_class = "progress-pill active" if active else ("progress-pill done" if done else "progress-pill")
            st.markdown(
                f"""
                <div class="{pill_class}" style="
                    text-align:center;padding:8px 10px;border-radius:999px;font-weight:700;">
                    {idx+1}. {labels[idx]}
                </div>
                """,
                unsafe_allow_html=True,
            )

def _derive_site_id(region: str, zip_code: str, application: str) -> str:
    base = (zip_code or "site").replace(" ", "").replace("-", "")
    return f"{(region or '').lower().replace(' ','_')}_{base}_{(application or '').lower().replace('/','_')}"

def get_energy_mode() -> str:
    prof = st.session_state.get("profile", {}) or {}
    return prof.get("energy_mode", "Electricity only")

# =========================
# Residential mode helpers
# =========================
def is_residential_grid_only() -> bool:
    prof = st.session_state.get("profile", {}) or {}
    app = (prof.get("application") or "").strip().lower()
    return app.startswith("res") or "residential" in app

def ensure_res_mode() -> bool:
    st.session_state["res_mode"] = is_residential_grid_only()
    return st.session_state["res_mode"]

def taper_tariff_payload(payload: dict) -> dict:
    if st.session_state.get("res_mode", False):
        out = dict(payload)
        out["demand_charge_lambda"] = 0.0
        out.pop("diesel_cost_usd_per_kwh", None)
        return out
    return payload

def taper_plan(plan: dict | None) -> dict | None:
    if not (plan and isinstance(plan, dict)):
        return plan
    if not st.session_state.get("res_mode", False):
        return plan
    sched = plan.get("schedule") or []
    if not isinstance(sched, list):
        return plan
    newp = dict(plan)
    newp["schedule"] = [{**row, "p_diesel_kw": 0.0} for row in sched]
    return newp

# =========================
# Time-axis helpers (12h)
# =========================
def hour_12_formatter():
    fmt = "%-I%p"
    try:
        datetime(2025,1,1).strftime(fmt)
    except ValueError:
        fmt = "%#I%p"
    return mdates.DateFormatter(fmt)

def _apply_readable_time_axis(ax, interval_hours: int = 2):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(hour_12_formatter())
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_horizontalalignment("right")
    ax.figure.tight_layout()

# =========================
# >>> Typewriter utilities (ONE-TIME per page) <<<
# =========================
from uuid import uuid4
import streamlit.components.v1 as components

def typewriter(text: str, speed_ms: int = 28, cursor: str = "▌", height: int = 56, weight: int = 800, size: str = "1.6rem"):
    _id = f"tw-{uuid4().hex}"
    _safe = json.dumps(text)
    _cursor = json.dumps(cursor)
    html = f"""
    <div id="{_id}" style="
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        font-weight: {weight};
        letter-spacing: .2px;
        color: #f5f5f5;
        white-space: pre-wrap;
        line-height: 1.25;
        font-size: {size};
        margin-bottom: 8px;
    " aria-live="polite"></div>
    <script>
      (function() {{
        const el = document.getElementById("{_id}");
        if (!el) return;
        const txt = {_safe};
        const cursor = {_cursor};
        let i = 0;
        let on = true;
        const blink = setInterval(() => {{
          on = !on;
          if (i >= txt.length) {{
            el.innerText = txt + (on ? " " + cursor : " ");
          }}
        }}, 450);
        const typer = setInterval(() => {{
          if (i < txt.length) {{
            el.innerText = txt.slice(0, i + 1) + " " + cursor;
            i++;
          }} else {{
            clearInterval(typer);
          }}
        }}, {speed_ms});
      }})();
    </script>
    """
    components.html(html, height=height)

def _static_header(text: str, weight: int = 900, size: str = "1.8rem"):
    st.markdown(
        f'<div style="font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;'
        f'font-weight:{weight}; color:#f5f5f5; font-size:{size}; margin-bottom:8px;">{text}</div>',
        unsafe_allow_html=True
    )

def tw_header_once(page_key: str, text: str):
    """
    Play the typewriter once for this logical page. On subsequent reruns,
    render a static header so interactions (text inputs/selects) don't retrigger it.
    """
    flag_key = f"_tw_done_{page_key}"
    if st.session_state.get(flag_key, False):
        _static_header(text)
    else:
        typewriter(text, speed_ms=22, height=60, weight=900, size="1.8rem")
        st.session_state[flag_key] = True

# =========================
# AI-POWERED SMART BILL READER
# (unchanged below except where called)
# =========================
def _normalize_timeseries_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    cols = {str(c).strip().lower(): c for c in df.columns}
    ts_candidate = next((cols[k] for k in [
        "timestamp","datetime","date_time","time","date","period_start","interval_start","start_time"
    ] if k in cols), None)
    kwh_candidate = next((cols[k] for k in [
        "kwh","usage","energy","consumption","kwh_used","energy_kwh","kwh_consumption","energy_use"
    ] if k in cols), None)
    if ts_candidate and kwh_candidate:
        out = df.rename(columns={ts_candidate:"timestamp", kwh_candidate:"kwh"}).copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        out["kwh"] = pd.to_numeric(out["kwh"], errors="coerce")
        out = out.dropna(subset=["timestamp","kwh"]).sort_values("timestamp")
        return out[["timestamp","kwh"]] if not out.empty else None
    for tk in ["total_kwh","kwh_total","total_energy_kwh","usage_total","billing_kwh","energy_total_kwh"]:
        if tk in cols:
            s = pd.to_numeric(df[cols[tk]], errors="coerce").dropna()
            if not s.empty:
                return pd.DataFrame({"kwh":[float(s.iloc[0])]})
    return None

def _extract_fields_from_text(text: str) -> Dict[str, float | str]:
    fields: Dict[str, float | str] = {}
    m = re.search(r"(amount\s+due|total\s+amount|current\s+charges)[^\d$]{0,25}\$?\s*([\d,]+(?:\.\d+)?)", text, re.I)
    if m: fields["amount_due_usd"] = float(m.group(2).replace(",", ""))
    m = re.search(r"(total\s+(usage|energy|kwh)[^\d]{0,40})([\d,]+(?:\.\d+)?)", text, re.I)
    if m: fields["total_kwh"] = float(m.group(3).replace(",", ""))
    m = re.search(r"\baccount\s*(?:no\.?|number)\s*[:#]?\s*([A-Z0-9\-]{6,})", text, re.I)
    if m: fields["account_number"] = m.group(1)
    d = re.search(r"(service|billing)\s*(period|dates?)\s*[:\-]?\s*([A-Za-z0-9 ,\/\-–]+)", text, re.I)
    if d: fields["billing_period_raw"] = d.group(3).strip()
    return fields

def _safe_bytes(uploaded) -> Tuple[bytes, str, str]:
    try:
        b = uploaded.read()
        try:
            uploaded.seek(0)
        except Exception:
            pass
    except Exception:
        b = bytes(uploaded or b"")
    name_lower = (getattr(uploaded, "name", "") or "upload").lower()
    mime = getattr(uploaded, "type", "") or ""
    return b, name_lower, mime

def _read_tabular_bytes(file_bytes: bytes, name_lower: str) -> Optional[pd.DataFrame]:
    try:
        if name_lower.endswith(".csv"):
            return _normalize_timeseries_df(pd.read_csv(io.BytesIO(file_bytes)))
        if name_lower.endswith((".xls",".xlsx")):
            return _normalize_timeseries_df(pd.read_excel(io.BytesIO(file_bytes)))
        if name_lower.endswith(".json"):
            j = json.load(io.BytesIO(file_bytes))
            df = pd.json_normalize(j)
            return _normalize_timeseries_df(df)
        if name_lower.endswith((".parquet",".pq",".pqt",".parq")):
            df = pd.read_parquet(io.BytesIO(file_bytes))
            return _normalize_timeseries_df(df)
    except Exception:
        return None
    return None

def _read_html_like(file_bytes: bytes) -> Optional[pd.DataFrame]:
    try:
        tables = pd.read_html(io.BytesIO(file_bytes))
        for t in tables:
            ts = _normalize_timeseries_df(t)
            if ts is not None:
                return ts
    except Exception:
        pass
    return None

def _pdf_tables_via_camelot(file_bytes: bytes) -> List[pd.DataFrame]:
    out = []
    if camelot is None:
        return out
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tables = camelot.read_pdf(tmp.name, pages="all", flavor="lattice")
            for t in tables:
                try:
                    df = t.df
                    df.columns = df.iloc[0]
                    df = df[1:]
                    out.append(df)
                except Exception:
                    pass
    except Exception:
        pass
    return out

def _pdf_tables_via_tabula(file_bytes: bytes) -> List[pd.DataFrame]:
    out = []
    if tabula is None:
        return out
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            dfs = tabula.read_pdf(tmp.name, pages="all", multiple_tables=True)
            for df in dfs or []:
                try:
                    out.append(df)
                except Exception:
                    pass
    except Exception:
        pass
    return out

def _read_pdf(file_bytes: bytes) -> Tuple[Optional[pd.DataFrame], Dict, str]:
    usage = None
    text = ""
    fields: Dict[str, float | str] = {}
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                texts = []
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t:
                        texts.append(t)
                    for tbl in (page.extract_tables() or []):
                        try:
                            df = pd.DataFrame(tbl[1:], columns=tbl[0])
                            ts = _normalize_timeseries_df(df)
                            if ts is not None and (usage is None or len(ts) > len(usage)):
                                usage = ts
                        except Exception:
                            pass
                text = "\n".join(texts)
                if text:
                    fields.update(_extract_fields_from_text(text))
            _log("pdfplumber text extracted.")
        except Exception:
            _log("pdfplumber failed.")
    if usage is None:
        for df in _pdf_tables_via_camelot(file_bytes) + _pdf_tables_via_tabula(file_bytes):
            ts = _normalize_timeseries_df(df)
            if ts is not None and (usage is None or len(ts) > len(usage)):
                usage = ts
        if usage is not None:
            _log("PDF table parsed via Camelot/Tabula.")
    if usage is None and convert_from_bytes and pytesseract:
        try:
            pages = convert_from_bytes(file_bytes, dpi=220)
            ocr_texts = []
            for pg in pages:
                ocr_texts.append(pytesseract.image_to_string(pg))
            text_ocr = "\n".join(ocr_texts)
            if text_ocr:
                text = text or text_ocr
                fields.update(_extract_fields_from_text(text_ocr))
            _log("OCR text extracted from PDF.")
        except Exception:
            _log("OCR on PDF failed.")
    if usage is None:
        html_ts = _read_html_like(file_bytes)
        if html_ts is not None:
            usage = html_ts
            _log("PDF->HTML-like table parsed.")
    if usage is None and "total_kwh" in fields:
        usage = pd.DataFrame({"kwh":[fields["total_kwh"]]})
    return usage, fields, text or ""

def _read_image(file_bytes: bytes) -> Tuple[Optional[pd.DataFrame], Dict, str]:
    """
    Aggressive OCR for images:
    - Upscales small images
    - Converts to grayscale + auto-contrast
    - Light denoising / sharpening
    - Lets LLM handle messy text later
    """
    usage = None
    text = ""
    fields: Dict[str, float | str] = {}

    if not (Image and pytesseract):
        _log("Image OCR unavailable (no PIL/pytesseract).")
        return None, fields, text

    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("L")  # grayscale

        # Upscale if small; Tesseract likes ~300 dpi equivalent
        w, h = img.size
        scale = 1.0
        if max(w, h) < 1200:
            scale = 1200.0 / max(w, h)
        if scale > 1.05:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Light denoise + sharpen + auto-contrast
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = img.filter(ImageFilter.SHARPEN)
        img = ImageOps.autocontrast(img)

        # Stronger Tesseract config for uniform text blocks
        ocr_config = "--psm 6 --oem 3"  # assume a block of text; LSTM engine
        text = pytesseract.image_to_string(img, config=ocr_config)

        if text:
            _log("OCR text extracted from image (enhanced).")
            fields.update(_extract_fields_from_text(text))

        # Very simple direct timeseries extraction (if the image has clear time/kWh rows)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        rows = []
        for ln in lines:
            # e.g. "13:00  0.54 kWh"
            m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?\b.*?([\d\.]+)\s*kwh\b", ln, re.I)
            if m:
                hh = int(m.group(1))
                mm = int(m.group(2) or "0")
                ap = (m.group(3) or "").lower()
                if ap in ("pm",) and hh < 12:
                    hh += 12
                if ap in ("am",) and hh == 12:
                    hh = 0
                kwh = float(m.group(4))
                rows.append((hh, mm, kwh))

        if rows:
            start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            ts = [start + timedelta(hours=h, minutes=m) for h, m, _ in rows]
            usage = pd.DataFrame(
                {"timestamp": pd.to_datetime(ts, utc=True),
                 "kwh": [r[2] for r in rows]}
            )
    except Exception as e:
        _log(f"Image OCR failed: {e}")

    return usage, fields, text

def _extract_json_from_text(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{"); end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            return None
    start = s.find("["); end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            return None
    return None

def extract_bill_from_image(uploaded_file):
    file_bytes = uploaded_file.read()
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    mime = uploaded_file.type or "image/png"

    prompt = (
        "Extract total_kwh and amount_due_usd as JSON ..."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
    )

    data = json.loads(resp.choices[0].message.content)
    return data.get("total_kwh"), data.get("amount_due_usd")

def _llm_parse_bill_text_to_df(text: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Use the OpenAI v1 client only (no legacy openai.ChatCompletion).
    Takes any non-empty bill text and tries to return either:
      - a timeseries DataFrame with columns ['timestamp','kwh'], or
      - a single-row DataFrame with total_kwh.
    Also returns a dict of extra fields (amount_due, account_number, billing_period).
    """
    # Normalize input
    text = (text or "").strip()
    if not text:
        _log("LLM parser skipped: empty text.")
        return None, {}

    # Get OpenAI API key from environment
    key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")

    if not key:
        st.error("OPENAI_API_KEY is not set. Configure it as an environment variable.")
        st.stop()

    client = OpenAI(api_key=key)

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    _log(f"LLM parser invoked. model={model}")

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a robust bill-to-data parser. Return STRICT JSON ONLY. "
                        "Schema: either "
                        "{'timeseries':[{'timestamp':ISO8601,'kwh':float},...], ...} "
                        "OR {'total_kwh':float, ...}. "
                        "Optional fields: 'amount_due_usd','account_number',"
                        "'billing_period':{'start':ISO8601,'end':ISO8601}. "
                        "NO commentary, no prose, just JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

        content = (resp.choices[0].message.content or "").strip()
        if not content:
            _log("LLM returned empty content.")
            return None, {}

        data = _extract_json_from_text(content)
        if not isinstance(data, dict):
            _log("LLM output was not a dict after JSON extraction.")
            return None, {}

    except Exception as e:
        _log(f"LLM parser error: {e}")
        return None, {}

        fields: Dict[str, Any] = {}

        # Optional fields
        if "amount_due_usd" in data:
            try:
                fields["amount_due_usd"] = float(data["amount_due_usd"])
            except Exception:
                pass

        if "account_number" in data:
            fields["account_number"] = str(data["account_number"])

        if isinstance(data.get("billing_period"), dict):
            fields["billing_period"] = data["billing_period"]

        # Timeseries path
        if isinstance(data.get("timeseries"), list):
            df = pd.DataFrame(data["timeseries"])
            if not {"timestamp", "kwh"}.issubset(df.columns):
                _log("LLM timeseries missing required columns.")
                return None, fields

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
            df = df.dropna(subset=["timestamp", "kwh"]).sort_values("timestamp")

            if df.empty:
                _log("LLM timeseries dataframe ended up empty after cleaning.")
                return None, fields

            return df[["timestamp", "kwh"]], fields

        # total_kwh-only path
        if "total_kwh" in data:
            try:
                tk = float(data["total_kwh"])
                fields["total_kwh"] = tk
                df = pd.DataFrame({"kwh": [tk]})
                return df, fields
            except Exception:
                _log("LLM total_kwh could not be converted to float.")
                return None, fields

        _log("LLM JSON did not contain 'timeseries' or 'total_kwh'.")
        return None, fields

    except Exception as e:
        _log(f"LLM parse failed (v1 client): {e}")
        return None, {}

from openai import OpenAI  # add near top if not already there


def llm_recommend_windows(
    bill_monthly_kwh: float,
    bill_monthly_cost: float,
    price_vector: dict,
) -> Optional[dict]:
    """
    Use OpenAI to:
      - sanity-check baseline cost
      - estimate daily/monthly/yearly savings from shifting load
      - return explicit cheap windows to run heavy loads
    Input: bill summary + price vector (start_ts, interval_min, values_usd_per_kwh)
    Output: dict like:
    {
      "baseline_daily_cost_usd": float,
      "optimized_daily_cost_usd": float,
      "daily_savings_usd": float,
      "monthly_savings_usd": float,
      "yearly_savings_usd": float,
      "cheap_windows": [
         {"start": "22:00", "end": "06:00", "label": "Run washer/dryer"},
         ...
      ]
    }
    """
    values = (price_vector or {}).get("values_usd_per_kwh") or []
    if not values or bill_monthly_cost <= 0:
        return None

    # Find API key the same way you do for the bill parser
    key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")

    if not key:
        st.error("OPENAI_API_KEY is not set. Configure it as an environment variable.")
        st.stop()

client = OpenAI(api_key=key)

    if not key:
        _log("LLM optimizer skipped: no OPENAI_API_KEY.")
        return None

    client = OpenAI(api_key=key)
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    payload = {
        "bill_monthly_kwh": bill_monthly_kwh,
        "bill_monthly_cost": bill_monthly_cost,
        "price_vector": {
            "start_ts": price_vector.get("start_ts"),
            "interval_min": price_vector.get("interval_min"),
            "values_usd_per_kwh": values,
        },
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an energy analyst. "
                        "Given a monthly bill and a time-of-use price vector, "
                        "propose a realistic demand-shifting strategy that minimizes cost "
                        "without making up impossible savings.\n\n"
                        "Return STRICT JSON ONLY with this schema:\n"
                        "{\n"
                        '  "baseline_daily_cost_usd": float,  # your best estimate\n'
                        '  "optimized_daily_cost_usd": float,\n'
                        '  "daily_savings_usd": float,\n'
                        '  "monthly_savings_usd": float,\n'
                        '  "yearly_savings_usd": float,\n'
                        '  "cheap_windows": [\n'
                        '     {"start": "HH:MM", "end": "HH:MM", "label": "short tip"},\n'
                        "     ...\n"
                        "  ]\n"
                        "}\n"
                        "Do not include any explanation text, just JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload),
                },
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        data = _extract_json_from_text(content)
        if not isinstance(data, dict):
            _log("LLM optimizer: JSON parse failed.")
            return None
        # Light sanity checks
        for k in [
            "baseline_daily_cost_usd",
            "optimized_daily_cost_usd",
            "daily_savings_usd",
            "monthly_savings_usd",
            "yearly_savings_usd",
        ]:
            if k in data:
                try:
                    data[k] = float(data[k])
                except Exception:
                    data[k] = 0.0
        if not isinstance(data.get("cheap_windows"), list):
            data["cheap_windows"] = []
        return data
    except Exception as e:
        _log(f"LLM optimizer failed: {e}")
        return None        

def ai_read_bill(uploaded, allow_llm: bool = True) -> Tuple[Optional[pd.DataFrame], Dict, str]:
    st.session_state["parse_log"] = []
    file_bytes, name_lower, mime = _safe_bytes(uploaded)
    _log(f"Incoming file: {getattr(uploaded,'name','(unnamed)')} type={mime or '(unknown)'}")
    tab = _read_tabular_bytes(file_bytes, name_lower)
    if tab is not None:
        _log("Parsed as tabular file (CSV/Excel/JSON/Parquet).")
        return tab, {}, ""
    html_ts = _read_html_like(file_bytes)
    if html_ts is not None:
        _log("Parsed as HTML-like table.")
        return html_ts, {}, ""
    if name_lower.endswith(".pdf") or "pdf" in mime:
        usage, fields, text = _read_pdf(file_bytes)
        if usage is not None:
            _log("PDF yielded timeseries/total.")
            return usage, fields, text
        if allow_llm and text:
            df, extra = _llm_parse_bill_text_to_df(text)
            if df is not None:
                _log("LLM successfully parsed PDF text.")
                return df, {**fields, **extra}, text
        _log("PDF parse failed.")
        return None, fields, text
    if name_lower.endswith((".png",".jpg",".jpeg",".tiff",".bmp",".webp",".gif")) or mime.startswith("image/"):
        usage, fields, text = _read_image(file_bytes)
        if usage is not None:
            _log("Image OCR yielded timeseries/total.")
            return usage, fields, text
        if allow_llm and text:
            df, extra = _llm_parse_bill_text_to_df(text)
            if df is not None:
                _log("LLM successfully parsed image OCR text.")
                return df, {**fields, **extra}, text
        _log("Image parse failed.")
        return None, fields, text
    try:
        raw_text_guess = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        raw_text_guess = ""
    if raw_text_guess:
        fields = _extract_fields_from_text(raw_text_guess)
        m = re.search(r"total\s*(kwh|energy)[^\d]{0,40}([\d,]+(?:\.\d+)?)", raw_text_guess, re.I)
        if m:
            total = float(m.group(2).replace(",", ""))
            _log("Plain text contained total_kwh.")
            return pd.DataFrame({"kwh":[total]}), {"total_kwh": total} | fields, raw_text_guess
        if allow_llm:
            df, extra = _llm_parse_bill_text_to_df(raw_text_guess)
            if df is not None:
                _log("LLM parsed plain text.")
                return df, {**fields, **extra}, raw_text_guess
    _log("All parsers exhausted.")
    return None, {}, raw_text_guess

def read_energy_bill_any(uploaded_file, allow_llm: bool = True):
    usage, fields, text = ai_read_bill(uploaded_file, allow_llm=allow_llm)
    if isinstance(usage, pd.DataFrame) and not usage.empty:
        if "timestamp" in usage.columns and "kwh" in usage.columns:
            return usage[["timestamp","kwh"]].copy(), fields
        if "kwh" in usage.columns and len(usage) == 1 and "timestamp" not in usage.columns:
            return pd.DataFrame([{"total_kwh": float(usage.iloc[0]["kwh"])}]), fields
    raise ValueError(
        "Could not parse this bill with built-in + AI readers. Try a clearer scan or enable LLM parsing with OPENAI_API_KEY."
    )

# =========================
# API helpers (unchanged)
# =========================
def ensure_prices(
    start_ts: Optional[str] = None,
    horizon: Optional[int] = None,
    interval_min: Optional[int] = None,
) -> Optional[dict]:
    """
    Ensure we have a price vector for ~24 hours based on the current interval.
    Priority:
    1) Reuse existing st.session_state["price_vector"] if present.
    2) Try backend /tariff/price_vector.
    3) If backend fails or returns flat prices, build a local TOU price vector
       from our inferred tariff (ZIP/region/application).
    """
    # 1) Reuse if already there
    pv = st.session_state.get("price_vector")
    if pv:
        return pv

    # 2) Resolve interval + horizon
    prof = st.session_state.get("profile", {}) or {}
    if interval_min is None:
        interval_min = int(prof.get("interval_minutes", 15))
    if horizon is None:
        horizon = int(24 * 60 / interval_min)

    # 3) Start timestamp
    if start_ts is None:
        start_ts = datetime.now(timezone.utc).replace(
            second=0, microsecond=0
        ).isoformat()

    base = _backend_base()
    vec: Optional[dict] = None

    # 4) Try backend first
    try:
        r = session.get(
            f"{base}/tariff/price_vector",
            params={
                "site_id": st.session_state.get("site_id", "site"),
                "start_ts": start_ts,
                "horizon": int(horizon),
                "interval_min": int(interval_min),
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        vals = data.get("values_usd_per_kwh")
        if isinstance(vals, list) and len(vals) > 0:
            # If backend vector has variation, keep it; if flat, we'll override.
            uniq = sorted({round(float(v), 5) for v in vals})
            if len(uniq) > 1:
                st.session_state["price_vector"] = data
                return data
            else:
                st.info(
                    "Backend returned a flat price vector – using local TOU approximation instead."
                )
        else:
            st.warning("Backend returned an invalid price vector payload; falling back locally.")
    except Exception as e:
        st.warning(f"Price vector unavailable from backend: {e}")

    # 5) Local fallback: infer tariff from ZIP/region/app and build a TOU vector
    #    Use last_tariff if we already calculated it; otherwise estimate from profile.
    tariff = st.session_state.get("last_tariff")
    if not tariff:
        zip_code = (prof.get("zip") or "").strip()
        region = prof.get("region", "")
        application = prof.get("application", "")
        tariff = _estimate_tariff_from_zip(zip_code, region=region, application=application)

    if not tariff:
        st.error("Could not infer a tariff locally; price vector is unavailable.")
        return None

    vec = _local_build_price_vector_from_tariff(
        tariff=tariff,
        start_ts=start_ts,
        horizon=int(horizon),
        interval_min=int(interval_min),
    )

    if isinstance(vec, dict) and vec.get("values_usd_per_kwh"):
        st.session_state["price_vector"] = vec
        st.info("Using locally-derived TOU price curve based on your ZIP and application.")
        return vec

    st.error("Local TOU price vector could not be built.")
    return None

def ensure_forecasts(horizon: int = 24, interval_min: int = 15):
    fl = st.session_state.get("forecast_load")
    fp = st.session_state.get("forecast_pv")
    base = _backend_base()
    if not fl:
        try:
            resp = session.post(f"{base}/forecast/load", json={
                "site_id": st.session_state.get("site_id", "site"),
                "horizon": int(horizon),
                "interval_min": int(interval_min),
            }, timeout=20)
            resp.raise_for_status()
            st.session_state["forecast_load"] = resp.json()
            fl = st.session_state["forecast_load"]
        except Exception as e:
            st.warning(f"Load forecast failed: {e}")
    if not fp:
        try:
            resp = session.post(f"{base}/forecast/pv", json={
                "site_id": st.session_state.get("site_id", "site"),
                "horizon": int(horizon),
                "interval_min": int(interval_min),
            }, timeout=20)
            resp.raise_for_status()
            st.session_state["forecast_pv"] = resp.json()
            fp = st.session_state["forecast_pv"]
        except Exception as e:
            st.info(f"PV forecast unavailable: {e}")
    need_fl = (not fl) or ("values_kw" not in fl)
    need_fp = (not fp) or ("values_kw" not in fp)
    if need_fl or need_fp:
        pv = st.session_state.get("price_vector") or {}
        try:
            start_ts = pv.get("start_ts") or datetime.now(timezone.utc).isoformat()
            start = datetime.fromisoformat(str(start_ts).replace("Z","+00:00"))
        except Exception:
            start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        steps = int(horizon)
        t = np.arange(steps, dtype=float)
        app = (st.session_state.get("profile", {}).get("application","") or "").lower()
        base_kw = 2.0 if "res" in app else (20.0 if "com" in app else 100.0)
        day_cycle = 0.6 + 0.5*np.sin((t/steps)*2*np.pi - np.pi/2)
        eve_bump = 0.25*np.exp(-0.5*((t - 0.75*steps)/(0.12*steps))**2)
        synthetic_load = np.clip(base_kw*(0.7 + 0.5*day_cycle) + base_kw*eve_bump, 0.1, None)
        pv_peak_kw = (0.6 if "res" in app else 5.0)
        hours = t * (interval_min/60.0)
        pv_shape = np.exp(-0.5*((hours - 13.0)/3.0)**2)
        synthetic_pv = np.clip(pv_peak_kw * pv_shape, 0.0, None)
        ts_list = [(start + timedelta(minutes=i*interval_min)).isoformat() for i in range(steps)]
        if need_fl:
            fl = {"start_ts": ts_list[0], "interval_min": interval_min, "values_kw": synthetic_load.tolist()}
            st.session_state["forecast_load"] = fl
            st.info("Using synthetic load forecast (backend unreachable).")
        if need_fp:
            fp = {"start_ts": ts_list[0], "interval_min": interval_min, "values_kw": synthetic_pv.tolist()}
            st.session_state["forecast_pv"] = fp
            st.info("Using synthetic PV forecast (backend unreachable).")
    return st.session_state.get("forecast_load"), st.session_state.get("forecast_pv")

def ensure_plan() -> Optional[dict]:
    if st.session_state.get("plan"):
        return st.session_state["plan"]
    pv = st.session_state.get("price_vector")
    fl = st.session_state.get("forecast_load")
    fp = st.session_state.get("forecast_pv")
    if not (pv and fl):
        return None
    H = min(
        len(pv.get("values_usd_per_kwh", [])),
        len(fl.get("values_kw", [])),
        len(fp.get("values_kw", [])) if fp else 10**9,
    )
    if H <= 0:
        return None
    try:
        req = {
            "site_id": st.session_state.get("site_id", "site"),
            "horizon": int(H),
            "interval_min": int(st.session_state.get("profile", {}).get("interval_minutes", 15)),
            "price_usd_per_kwh": pv["values_usd_per_kwh"][:H],
            "load_kw": fl["values_kw"][:H],
            "pv_kw": (fp["values_kw"][:H] if fp else [0.0] * H),
        }
        r = session.post(f"{_backend_base()}/optimize/plan", json=req, timeout=25)
        r.raise_for_status()
        plan = r.json()
        plan = taper_plan(plan)
        st.session_state["plan"] = plan
        return plan
    except Exception as e:
        st.warning(f"Optimization failed: {e}")
        return None

def ensure_savings(force: bool = False) -> Optional[dict]:
    """
    Compute savings from the backend if possible.
    - If `force=False`, we re-use a previously-computed non-empty result.
    - If backend fails or reports zero savings, fall back to local calculation
      from price_vector + forecast_load + plan.schedule.
    """
    s = st.session_state

    # Only reuse cached result if caller didn't ask to force recompute
    if (not force) and isinstance(s.get("last_savings"), dict):
        return s["last_savings"]

    prices = s.get("price_vector")
    load_fc = s.get("forecast_load")
    pv_fc = s.get("forecast_pv")
    plan = s.get("plan")

    if not (prices and load_fc and plan and plan.get("schedule")):
        return None

    pv_vals = pv_fc["values_kw"] if pv_fc else [0.0] * len(load_fc["values_kw"])
    H = min(
        len(prices.get("values_usd_per_kwh", [])),
        len(load_fc.get("values_kw", [])),
        len(pv_vals),
        len(plan.get("schedule", [])),
    )
    if H <= 0:
        return None

    try:
        req = {
            "site_id": plan.get("site_id", s.get("site_id", "site")),
            "interval_min": int(plan.get("interval_min", s.get("profile", {}).get("interval_minutes", 15))),
            "price_usd_per_kwh": prices["values_usd_per_kwh"][:H],
            "load_kw": load_fc["values_kw"][:H],
            "pv_kw": pv_vals[:H],
            "schedule": plan["schedule"][:H],
            "diesel_cost_usd_per_kwh": 0.35,
            "demand_charge_lambda": 5.0,
            "degr_cost_usd_per_kwh": 0.02,
        }
        r = session.post(f"{_backend_base()}/analyze/savings", json=req, timeout=20)
        r.raise_for_status()
        res = r.json()

        abs_sv = float(res.get("savings_abs", 0.0) or 0.0)

        # If backend says "zero savings", try our local calculation instead
        if abs_sv <= 0.0:
            local = _local_savings_from_plan()
            if local:
                s["last_savings"] = local
                return local

        s["last_savings"] = res
        return res

    except Exception as e:
        # Backend failed – fall back to local calculation
        local = _local_savings_from_plan()
        if local:
            s["last_savings"] = local
            return local

        st.info(f"Savings analysis not available now: {e}")
        return None

def warm_analysis_pipeline():
    """
    Make sure price_vector, forecasts, plan, and savings are in place
    so the Dashboard has everything it needs when you land there
    (even if you skip the Tariff / Forecast / Optimize pages).
    Safe to call multiple times.
    """
    prof = st.session_state.get("profile", {}) or {}
    zip_code = (prof.get("zip") or "").strip()
    region = prof.get("region", "")
    application = prof.get("application", "")
    interval_min = int(prof.get("interval_minutes", 15))
    horizon_steps = int(24 * 60 / interval_min)  # ~24h horizon

    # ---------- 1) Tariff + Price Vector ----------
    if not st.session_state.get("price_vector"):
        # If we have a ZIP, build a tariff from it
        if zip_code:
            tariff = _estimate_tariff_from_zip(zip_code, region=region, application=application)
        else:
            # Fallback: use last_tariff or a default
            tariff = st.session_state.get("last_tariff") or _estimate_tariff_from_zip(
                zip_code="", region=region, application=application
            )

        st.session_state["last_tariff"] = tariff

        # Build a local TOU-style price vector for the next 24 hours
        start_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
        vec = _local_build_price_vector_from_tariff(
            tariff,
            start_ts=start_ts,
            horizon=horizon_steps,
            interval_min=interval_min,
        )
        if vec.get("values_usd_per_kwh"):
            st.session_state["price_vector"] = vec

    # ---------- 2) Forecasts ----------
    # (will use backend if available, otherwise synthetic)
    ensure_forecasts(horizon=24, interval_min=interval_min)

    # ---------- 3) Optimization Plan ----------
    ensure_plan()

    # ---------- 4) Savings ----------
    ensure_savings()

# =========================
# Login gate
# =========================
USERS = {
    "client01": "demoPass01!",
    "client02": "demoPass02!",
    "client03": "demoPass03!",
    "client04": "demoPass04!",
    "client05": "demoPass05!",
}

def login_gate():
    # extra safety – but init_state() is already called at top-level
    if "authenticated" not in st.session_state:
        init_state()

    if st.session_state.get("authenticated", False):
        return

    tw_header_once("login", "🔒 Microgrid Dashboard Login")
    u = st.text_input("Username", key="login_user").strip()
    p = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login", key="login_btn"):
        if u in USERS and USERS[u] == p:
            st.session_state["authenticated"] = True
            st.session_state["username"] = u

            has_snapshot = load_user_snapshot()

            if has_snapshot and st.session_state.get("plan") and st.session_state.get("last_savings"):
                st.session_state["step"] = "dashboard"
            else:
                st.session_state["step"] = "profile"

            st.success(f"Welcome, {u}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

login_gate()

with st.sidebar:
    st.success(f"Logged in as {st.session_state.get('username','')}")

        # --- Quick navigation ---
    if st.button("📊 Dashboard"):
        go_step("dashboard")
        st.rerun()

    if st.button("📈 Forecasts"):
        go_step("forecasts")
        st.rerun()

    if st.button("⚙️ Optimization Plan"):
        go_step("optimize")
        st.rerun()

    st.markdown("---")
    
    # Start over with a fresh bill
    if st.button("Start new analysis"):
        for k in [
            "bill_usage_df",
            "bill_fields",
            "price_vector",
            "forecast_load",
            "forecast_pv",
            "plan",
            "last_savings",
            "bill_monthly_cost",
            "bill_monthly_kwh",
        ]:
            st.session_state.pop(k, None)
        st.session_state["step"] = "upload"
        save_user_snapshot()  # snapshot now reflects "empty" state
        st.rerun()

    if st.button("Logout"):
        st.session_state.clear()
        init_state()
        st.rerun()

if st.session_state.get("step", "login") == "login":
        go_step("upload")

# =========================
# LLM-based Tariff Estimator (optional)
# =========================
def _llm_tariff_from_location(zip_code: str, region: str = "", application: str = "") -> Optional[dict]:
    """
    Use OpenAI to estimate a realistic time-of-use tariff for this location + application.
    This is *approximate* (based on model knowledge, not live utility data).
    Returns a tariff dict matching our internal shape or None on failure.
    """
    zip_code = (zip_code or "").strip()
    region = (region or "").strip()
    application = (application or "").strip()

    # Reuse the same key discovery logic as the bill parser
    key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")

if not key:
    st.error("OPENAI_API_KEY is not set. Configure it as an environment variable.")
    st.stop()

client = OpenAI(api_key=key)

    if not key:
        # No key = no LLM tariff
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)

        model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        # We ask the model for a *simple JSON tariff* we can feed into the rest of the app
        system_msg = (
            "You are an electricity tariff estimator. "
            "Given a location and application type, you return a plausible time-of-use tariff, "
            "NOT exact live data. Your output MUST be valid JSON only, no commentary. "
            "Schema:\n"
            "{\n"
            '  "timezone": "string",\n'
            '  "weekday_periods": [\n'
            '    {"name": "Off-peak", "start_hour": 0,  "end_hour": 7,  "price_usd_per_kwh": 0.12},\n'
            '    {"name": "Mid-peak","start_hour": 7,  "end_hour": 16, "price_usd_per_kwh": 0.20},\n'
            '    {"name": "Peak",    "start_hour": 16, "end_hour": 21, "price_usd_per_kwh": 0.32},\n'
            '    {"name": "Off-peak","start_hour": 21, "end_hour": 24, "price_usd_per_kwh": 0.12}\n'
            "  ],\n"
            '  "weekend_periods": null OR same structure as weekday_periods,\n'
            '  "demand_charge_lambda": 0.0–20.0,\n'
            '  "diesel_cost_usd_per_kwh": 0.2–0.6\n'
            "}\n"
            "Prices must be in USD. If information is uncertain, return a *reasonable* generic TOU pattern."
        )

        user_msg = (
            f"Location: ZIP/postal '{zip_code}', region '{region}'. "
            f"Application type: '{application}' (e.g. Residential, Commercial, Industrial). "
            "Return a single JSON object only, following the schema above."
        )

        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)

        # Basic sanity check
        if not isinstance(data, dict):
            return None
        if "weekday_periods" not in data or not isinstance(data["weekday_periods"], list):
            return None
        # Fill defaults if missing
        data.setdefault("timezone", "UTC")
        data.setdefault("weekend_periods", None)
        data.setdefault("demand_charge_lambda", 5.0)
        data.setdefault("diesel_cost_usd_per_kwh", 0.35)

        return data

    except Exception as e:
        # Fail quietly – we'll fall back to the catalog-based estimator
        st.info(f"LLM tariff estimator unavailable: {e}")
        return None
            

# =========================
# Tariff estimator (unchanged)
# =========================
def _estimate_tariff_from_zip(zip_code: str, region: str = "", application: str = "") -> dict:
    z = (zip_code or "").strip()
    reg = (region or "").strip().lower()
    app = (application or "").strip().lower()

    # 1) Try LLM-based tariff first (approximate, but location-aware)
    llm_tariff = _llm_tariff_from_location(zip_code=z, region=region, application=application)
    if isinstance(llm_tariff, dict) and llm_tariff.get("weekday_periods"):
        # Use it directly, but still store into session
        st.session_state["last_tariff"] = llm_tariff
        return llm_tariff

    # 2) Fall back to your hard-coded catalog (previous behavior)
    catalog = {
        "941": {"res_offpeak": 0.18, "res_peak": 0.45, "com_offpeak": 0.16, "com_peak": 0.40, "peak_window": (16, 21)},
        "100": {"res_offpeak": 0.20, "res_peak": 0.39, "com_offpeak": 0.18, "com_peak": 0.36, "peak_window": (15, 20)},
        "733": {"res_offpeak": 0.11, "res_peak": 0.25, "com_offpeak": 0.10, "com_peak": 0.22, "peak_window": (14, 19)},
        "gh":  {"res_offpeak": 0.11, "res_peak": 0.20, "com_offpeak": 0.12, "com_peak": 0.22, "peak_window": (17, 22)},
        "ng":  {"res_offpeak": 0.10, "res_peak": 0.19, "com_offpeak": 0.11, "com_peak": 0.21, "peak_window": (17, 21)},
        "ke":  {"res_offpeak": 0.13, "res_peak": 0.24, "com_offpeak": 0.12, "com_peak": 0.22, "peak_window": (18, 21)},
    }

    key = z[:3] if z[:3].isdigit() else ""
    if not key:
        if reg in ("ghana", "africa") or z.lower().startswith("gh"): key = "gh"
        elif reg in ("nigeria",) or z.lower().startswith("ng"):      key = "ng"
        elif reg in ("kenya",) or z.lower().startswith("ke"):        key = "ke"

    entry = catalog.get(key)

    # 🔴 FORCE STRONG TOU IF NO SPECIAL ENTRY – THIS IS WHAT WILL FIX YOUR FLAT LINE
    if not entry:
        if reg in ("north america", "us", "united states", "canada", "usa"):
            entry = {
                "res_offpeak": 0.14,
                "res_peak":    0.34,
                "com_offpeak": 0.12,
                "com_peak":    0.30,
                "peak_window": (15, 20),  # 3pm–8pm
            }
        elif reg in ("europe", "eu", "united kingdom", "uk"):
            entry = {"res_offpeak": 0.20, "res_peak": 0.45, "com_offpeak": 0.18, "com_peak": 0.40, "peak_window": (17, 20)}
        elif reg in ("africa",):
            entry = {"res_offpeak": 0.12, "res_peak": 0.24, "com_offpeak": 0.12, "com_peak": 0.26, "peak_window": (17, 21)}
        else:
            entry = {"res_offpeak": 0.14, "res_peak": 0.32, "com_offpeak": 0.13, "com_peak": 0.28, "peak_window": (16, 20)}

    is_res = ("res" in app) or (application.lower() == "residential")
    offpeak = entry["res_offpeak"] if is_res else entry["com_offpeak"]
    peak    = entry["res_peak"]    if is_res else entry["com_peak"]
    peak_start, peak_end = entry["peak_window"]

    demand_lambda = 12.0 if "industrial" in app else (8.0 if ("com" in app or "office" in app) else 4.0)
    diesel_cost   = 0.35

    t = {
        "timezone": "UTC",
        "weekday_periods": [
            {"name": "Off-peak", "start_hour": 0,          "end_hour": peak_start, "price_usd_per_kwh": float(offpeak)},
            {"name": "Peak",     "start_hour": peak_start, "end_hour": peak_end,   "price_usd_per_kwh": float(peak)},
            {"name": "Off-peak", "start_hour": peak_end,   "end_hour": 24,         "price_usd_per_kwh": float(offpeak)},
        ],
        "weekend_periods": None,
        "demand_charge_lambda": float(demand_lambda),
        "diesel_cost_usd_per_kwh": float(diesel_cost),
    }

    st.session_state["last_tariff"] = t
    return t

    key = z[:3] if z[:3].isdigit() else ""
    if not key:
        if reg in ("ghana", "africa") or z.lower().startswith("gh"): key = "gh"
        elif reg in ("nigeria",) or z.lower().startswith("ng"):      key = "ng"
        elif reg in ("kenya",) or z.lower().startswith("ke"):        key = "ke"

    entry = catalog.get(key)

    # 🔴 FORCE STRONG TOU IF NO SPECIAL ENTRY – THIS IS WHAT WILL FIX YOUR FLAT LINE
    if not entry:
        if reg in ("north america", "us", "united states", "canada", "usa"):
            # Nice obvious TOU: cheap nights, expensive 3–8pm
            entry = {
                "res_offpeak": 0.14,
                "res_peak":    0.34,
                "com_offpeak": 0.12,
                "com_peak":    0.30,
                "peak_window": (15, 20),  # 3pm–8pm
            }
        elif reg in ("europe", "eu", "united kingdom", "uk"):
            entry = {"res_offpeak": 0.20, "res_peak": 0.45, "com_offpeak": 0.18, "com_peak": 0.40, "peak_window": (17, 20)}
        elif reg in ("africa",):
            entry = {"res_offpeak": 0.12, "res_peak": 0.24, "com_offpeak": 0.12, "com_peak": 0.26, "peak_window": (17, 21)}
        else:
            entry = {"res_offpeak": 0.14, "res_peak": 0.32, "com_offpeak": 0.13, "com_peak": 0.28, "peak_window": (16, 20)}

    is_res = ("res" in app) or (application.lower() == "residential")
    offpeak = entry["res_offpeak"] if is_res else entry["com_offpeak"]
    peak    = entry["res_peak"]    if is_res else entry["com_peak"]
    peak_start, peak_end = entry["peak_window"]

    demand_lambda = 12.0 if "industrial" in app else (8.0 if ("com" in app or "office" in app) else 4.0)
    diesel_cost   = 0.35

    t = {
        "timezone": "UTC",
        "weekday_periods": [
            {"name": "Off-peak", "start_hour": 0,          "end_hour": peak_start, "price_usd_per_kwh": float(offpeak)},
            {"name": "Peak",     "start_hour": peak_start, "end_hour": peak_end,   "price_usd_per_kwh": float(peak)},
            {"name": "Off-peak", "start_hour": peak_end,   "end_hour": 24,         "price_usd_per_kwh": float(offpeak)},
        ],
        "weekend_periods": None,
        "demand_charge_lambda": float(demand_lambda),
        "diesel_cost_usd_per_kwh": float(diesel_cost),
    }

    st.session_state["last_tariff"] = t
    return t

def _auto_prepare_tariff_and_prices():
    prof = st.session_state.get("profile", {}) or {}
    zip_code = (prof.get("zip") or "").strip()
    region = prof.get("region", "")
    application = prof.get("application", "")

    if not zip_code:
        st.info("No ZIP in profile – skipping auto-tariff.")
        return

    tariff = _estimate_tariff_from_zip(zip_code, region=region, application=application)
    st.session_state["last_tariff"] = tariff

    payload = {
        "site_id": st.session_state.get("site_id", "site"),
        "timezone": tariff.get("timezone", "UTC"),
        "weekday_periods": tariff["weekday_periods"],
        "weekend_periods": tariff.get("weekend_periods"),
        "demand_charge_lambda": tariff["demand_charge_lambda"],
        "diesel_cost_usd_per_kwh": tariff["diesel_cost_usd_per_kwh"],
    }
    payload = taper_tariff_payload(payload)

    base = _backend_base()

    # ---- 1) push tariff to backend ----
    try:
        r_set = session.post(f"{base}/tariff/set", json=payload, timeout=10)
        r_set.raise_for_status()
        st.caption("✅ Tariff pushed to optimizer backend.")
    except Exception as e:
        st.warning(f"⚠️ Could not push tariff to backend ({base}/tariff/set): {e}")

    # ---- 2) build price vector (backend first, then local fallback) ----
    interval_min = int(prof.get("interval_minutes", 15))
    horizon_steps = max(1, int(24 * 60 / interval_min))
    start_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()

    vec = None
    try:
        r_vec = session.get(
            f"{base}/tariff/price_vector",
            params={
                "site_id": st.session_state.get("site_id", "site"),
                "start_ts": start_ts,
                "horizon": int(horizon_steps),
                "interval_min": int(interval_min),
            },
            timeout=10,
        )
        r_vec.raise_for_status()
        vec = r_vec.json()
        st.caption("✅ Price vector fetched from backend.")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch price vector from backend: {e}")
        vec = None

    # If backend vector missing or flat, use local TOU builder
    if not (isinstance(vec, dict) and vec.get("values_usd_per_kwh")):
        st.info("Using local tariff approximation for price vector.")
        vec = _local_build_price_vector_from_tariff(
            tariff,
            start_ts=start_ts,
            horizon=horizon_steps,
            interval_min=interval_min,
        )
    else:
        uniq = sorted({round(float(x), 5) for x in vec.get("values_usd_per_kwh", [])})
        if len(uniq) <= 1:
            st.info("Backend prices are flat – rebuilding from local TOU.")
            vec = _local_build_price_vector_from_tariff(
                tariff,
                start_ts=start_ts,
                horizon=horizon_steps,
                interval_min=interval_min,
            )

    if isinstance(vec, dict) and vec.get("values_usd_per_kwh"):
        st.session_state["price_vector"] = vec
        st.caption(f"💡 Price vector ready with {len(vec['values_usd_per_kwh'])} steps.")
    else:
        st.error("❌ Failed to create a price vector (backend + local).")  

import streamlit as st
from datetime import date

def parse_bill_with_ai(uploaded_file):
    # TODO: your OCR + OpenAI extraction pipeline
    # Should return (data: dict, confidence: float, errors: list[str])
    return {}, 0.0, ["not implemented"]

def bill_input_form(parsed=None, errors=None, test_mode=False):
    parsed = parsed or {}
    errors = errors or []

    st.subheader("Step 2 – Review Your Bill Details")

    if test_mode:
        st.info("Test mode is ON. Using randomized sample data.")
        # overwrite parsed with fake numbers
        parsed = {
            "total_kwh": 1234,
            "total_amount_usd": 187.56,
            "billing_period_start": date(2025, 11, 1),
            "billing_period_end": date(2025, 11, 30),
            "peak_kw_demand": 42.3,
        }

    col1, col2 = st.columns(2)

    with col1:
        total_kwh = st.number_input(
            "Total kWh used this period",
            min_value=0.0,
            value=float(parsed.get("total_kwh") or 0.0),
            step=1.0
        )

        peak_kw = st.number_input(
            "Peak kW demand (if shown)",
            min_value=0.0,
            value=float(parsed.get("peak_kw_demand") or 0.0),
            step=0.1
        )

    with col2:
        total_amount = st.number_input(
            "Total bill amount ($)",
            min_value=0.0,
            value=float(parsed.get("total_amount_usd") or 0.0),
            step=0.01,
            format="%.2f"
        )

        billing_period_start = st.date_input(
            "Billing period start",
            value=parsed.get("billing_period_start") or date.today().replace(day=1)
        )

        billing_period_end = st.date_input(
            "Billing period end",
            value=parsed.get("billing_period_end") or date.today()
        )

    if errors:
        st.warning("We had trouble reading some fields. Please confirm everything before continuing.")

    proceed = st.button("Run Optimization")

    if proceed:
        # Final validation before proceeding
        if total_kwh <= 0 or total_amount <= 0:
            st.error("Please enter a positive kWh and bill amount.")
            return None

        return {
            "total_kwh": total_kwh,
            "total_amount_usd": total_amount,
            "peak_kw_demand": peak_kw or None,
            "billing_period_start": billing_period_start,
            "billing_period_end": billing_period_end,
        }

    return None

def _local_build_price_vector_from_tariff(tariff: dict, start_ts: str, horizon: int, interval_min: int) -> dict:
    if not tariff or not isinstance(tariff.get("weekday_periods"), list):
        return {}
    try:
        start = datetime.fromisoformat(str(start_ts).replace("Z","+00:00"))
    except Exception:
        start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    idx = pd.date_range(start=start, periods=horizon, freq=f"{int(interval_min)}min")
    prices = []
    periods = tariff["weekday_periods"]
    for ts in idx:
        hr = ts.hour
        price = None
        for p in periods:
            if int(p["start_hour"]) <= hr < int(p["end_hour"]):
                price = float(p["price_usd_per_kwh"]); break
        prices.append(price if price is not None else float(periods[-1]["price_usd_per_kwh"]))
    return {"start_ts": start.isoformat(), "interval_min": int(interval_min), "values_usd_per_kwh": prices}

# =========================
# Views
# =========================
def view_upload():
    import random  # local import so this function is self-contained

    render_progress()
    tw_header_once("upload", "2) Upload your latest energy bill")

    st.caption(
        "Upload a recent bill (PDF or image). "
        "We’ll extract your monthly usage and cost automatically."
    )

    # ---------- Test Mode (no AI calls, randomized sample data) ----------
    test_mode = st.checkbox(
        "Test mode (use sample numbers, no AI bill reading)",
        help="Use this to demo Microgrid without using API credits.",
    )

    def manual_entry(reason: str, default_kwh: float = 0.0, default_cost: float = 0.0):
        """
        Unified manual entry / review form.
        Used both for true fallback and for 'partial' parses.
        """
        if reason:
            st.warning(reason)

        st.markdown("You can still continue — just enter or confirm the details below.")

        kwh = st.number_input(
            "Monthly Usage (kWh)",
            min_value=0.0,
            value=float(default_kwh or 0.0),
            key="manual_monthly_kwh",
        )
        cost = st.number_input(
            "Monthly Cost ($)",
            min_value=0.0,
            value=float(default_cost or 0.0),
            key="manual_monthly_cost",
        )

        if st.button("Continue with these values →", key="manual_continue_btn"):
            st.session_state["bill_monthly_kwh"] = kwh
            st.session_state["bill_monthly_cost"] = cost
            _go_to_dashboard_with_current_bill()
            st.stop()

        # If they haven't clicked continue yet, just stop this branch
        st.stop()

    # ---------- If Test Mode is ON, skip uploads & AI ----------
    if test_mode:
        st.info("Test mode is ON. Using randomized sample values instead of reading a real bill.")

        # Random but reasonable defaults
        sample_kwh = float(random.randint(400, 2000))
        sample_cost = float(random.randint(50, 400))

        st.metric("Sample Monthly Usage", f"{sample_kwh:,.0f} kWh")
        st.metric("Sample Monthly Cost", f"${sample_cost:,.2f}")

        # Let the user tweak before continuing
        kwh = st.number_input(
            "Confirm Monthly Usage (kWh)",
            min_value=0.0,
            value=sample_kwh,
            key="test_confirm_kwh",
        )
        cost = st.number_input(
            "Confirm Monthly Cost ($)",
            min_value=0.0,
            value=sample_cost,
            key="test_confirm_cost",
        )

        if st.button("Continue with test values →", key="test_continue_btn"):
            st.session_state["bill_monthly_kwh"] = kwh
            st.session_state["bill_monthly_cost"] = cost
            _go_to_dashboard_with_current_bill()
            st.stop()

        # In test mode we don't require an upload at all
        st.caption("You can adjust the sample values above and continue, no bill upload needed.")
        return

    # ---------- Normal mode: require a bill upload ----------
    bill = st.file_uploader(
        "Upload your bill",
        type=["pdf", "png", "jpg", "jpeg", "heic", "webp"],
        accept_multiple_files=False,
    )

    if not bill:
        st.caption("Upload a bill to continue, or turn on Test mode above.")
        return

    # =========================================================
    #  IMAGE BILLS (OpenAI Vision / image extractor path)
    # =========================================================
    if bill.type and bill.type.startswith("image/"):
        try:
            st.info("Reading your bill image with AI…")
            total_kwh, amount_due = extract_bill_from_image(bill)
        except Exception as e:
            # Log the real error server-side, but don't show it to the user
            print("AI bill reading (image) error:", e)
            return manual_entry(
                "We couldn't read this bill image automatically due to a technical issue."
            )

        # If we got absolutely nothing back
        if total_kwh is None and amount_due is None:
            return manual_entry(
                "We couldn't extract usage or cost from this bill image."
            )

        st.success("Bill successfully read ✔")

        # Store what we have in session
        if total_kwh is not None:
            st.metric("Detected Monthly Usage", f"{float(total_kwh):,.0f} kWh")
            st.session_state["bill_monthly_kwh"] = float(total_kwh)
        else:
            st.warning("Usage not found on the image. Please enter it below.")
            st.session_state["bill_monthly_kwh"] = 0.0

        if amount_due is not None:
            st.metric("Detected Monthly Cost", f"${float(amount_due):,.2f}")
            st.session_state["bill_monthly_cost"] = float(amount_due)
        else:
            st.warning("Cost not found on the image. Please enter it below.")
            st.session_state["bill_monthly_cost"] = 0.0

        # Always give the user a chance to confirm/override
        kwh = st.number_input(
            "Confirm Monthly Usage (kWh)",
            min_value=0.0,
            value=float(st.session_state.get("bill_monthly_kwh", 0.0)),
            key="img_confirm_kwh",
        )
        cost = st.number_input(
            "Confirm Monthly Cost ($)",
            min_value=0.0,
            value=float(st.session_state.get("bill_monthly_cost", 0.0)),
            key="img_confirm_cost",
        )

        if st.button("Continue →", key="img_continue_btn"):
            st.session_state["bill_monthly_kwh"] = kwh
            st.session_state["bill_monthly_cost"] = cost
            _go_to_dashboard_with_current_bill()
            return

        return  # wait for user to click the button

    # =========================================================
    #  NON-IMAGE BILLS (PDF / other) → read_energy_bill_any
    # =========================================================
    try:
        # If you already have this helper:
        df, fields = read_energy_bill_any(bill, allow_llm=True)
        st.session_state["bill_usage_df"] = df
        st.session_state["bill_fields"] = fields

        total_kwh = fields.get("total_kwh")
        amount_due = fields.get("amount_due_usd")

        # If we got both, show metrics & confirm
        if total_kwh is not None or amount_due is not None:
            st.success("Bill successfully read ✔")

            if total_kwh is not None:
                st.metric("Detected Monthly Usage", f"{float(total_kwh):,.0f} kWh")

            if amount_due is not None:
                st.metric("Detected Monthly Cost", f"${float(amount_due):,.2f}")

            # Use manual_entry-style confirm UI with prefilled values
            default_kwh = float(total_kwh) if total_kwh is not None else 0.0
            default_cost = float(amount_due) if amount_due is not None else 0.0

            # Reuse the same manual form but with prefilled values (no scary error message)
            st.markdown("Please confirm or adjust your bill details below.")
            kwh = st.number_input(
                "Confirm Monthly Usage (kWh)",
                min_value=0.0,
                value=default_kwh,
                key="pdf_confirm_kwh",
            )
            cost = st.number_input(
                "Confirm Monthly Cost ($)",
                min_value=0.0,
                value=default_cost,
                key="pdf_confirm_cost",
            )

            if st.button("Go to Dashboard →", key="pdf_continue_btn"):
                st.session_state["bill_monthly_kwh"] = kwh
                st.session_state["bill_monthly_cost"] = cost
                _go_to_dashboard_with_current_bill()
                return

            return

        # Partial or failed PDF parse → clean manual entry
        return manual_entry(
            "We couldn't fully extract usage or cost from this bill. "
            "Please enter the details so we can continue.",
        )

    except Exception as e:
        # Again, log real error but show friendly message
        print("AI bill reading (PDF) error:", e)
        return manual_entry(
            "We couldn't read this bill automatically due to a technical issue."
        )

def _go_to_dashboard_with_current_bill():
    """Common path: we already have bill_monthly_kwh / bill_monthly_cost set."""
    _auto_prepare_tariff_and_prices()

    interval_min = int(
        st.session_state.get("profile", {}).get("interval_minutes", 15)
    )
    horizon_steps = max(1, int(24 * 60 / interval_min))

    ensure_forecasts(horizon=horizon_steps, interval_min=interval_min)
    ensure_plan()

    st.session_state["step"] = "dashboard"
    save_user_snapshot()
    st.rerun()

def view_profile():
    render_progress()
    tw_header_once("profile", "1) Tell us where and how you use energy")

    prof = st.session_state.setdefault("profile", {})
    _populate_zip_from_backend()

    # --- Region / ZIP / Application / Interval (your existing widgets) ---
    r_idx = REGION_OPTIONS.index(prof.get("region", "")) if prof.get("region", "") in REGION_OPTIONS else 0
    a_idx = APP_OPTIONS.index(prof.get("application", "")) if prof.get("application", "") in APP_OPTIONS else 0

    if prof.get("zip") and "profile_zip_input" not in st.session_state:
        st.session_state["profile_zip_input"] = prof["zip"]

    colA, colB = st.columns(2)
    with colA:
        region = st.selectbox("REGION", REGION_OPTIONS, index=r_idx, key="profile_region")
        zip_code = st.text_input("ZIP/POSTAL CODE", key="profile_zip_input")
    with colB:
        application = st.selectbox("APPLICATION", APP_OPTIONS, index=a_idx, key="profile_app")
        interval_minutes = st.number_input(
            "INTERVAL (MINUTES)",
            min_value=5,
            max_value=60,
            value=int(prof.get("interval_minutes", 15)),
            step=5,
            key="profile_interval",
        )

    zip_code_clean = (zip_code or "").strip()

    st.session_state["profile"] = {
        "region": region,
        "zip": zip_code_clean,
        "application": application,
        "interval_minutes": int(interval_minutes),
    }
    st.session_state["site_id"] = _derive_site_id(region or "", zip_code_clean or "", application or "")

    ensure_res_mode()

    st.caption(f"Site ID: {st.session_state['site_id']}")

     # ---------- NAV BUTTONS (this is what brings the Next button back) ----------
    colL, colR = st.columns([1, 1])

    with colL:
        if step_index() > 0 and st.button("← Back"):
            go_prev()

    with colR:
        if zip_code_clean:
            # Only show an active Next button if ZIP is present
            if st.button("Next →"):
                go_next()
        else:
            # Show hint when ZIP is missing
            st.caption(
                "We couldn't determine your ZIP automatically — please enter it to continue."
            )

def render_tariff_explainer(tariff: dict | None, price_vector: dict | None, interval_min: int) -> None:
    st.markdown("### Tariff explainer")

    if isinstance(tariff, dict) and isinstance(tariff.get("weekday_periods"), list):
        st.write("**Your defined periods (weekday):**")
        df_periods = pd.DataFrame([{
            "Name": p["name"],
            "Start": f"{int(p['start_hour']):02d}:00",
            "End": f"{int(p['end_hour']):02d}:00",
            "$/kWh": float(p["price_usd_per_kwh"]),
        } for p in tariff["weekday_periods"]])
        st.dataframe(df_periods, use_container_width=True)
        try:
            uniq = sorted({round(float(p["price_usd_per_kwh"]), 5) for p in tariff["weekday_periods"]})
            if len(uniq) >= 2:
                cheapest = min(uniq); priciest = max(uniq)
                st.info(f"Cheapest hours ~ **${cheapest:.2f}/kWh**; priciest ~ **${priciest:.2f}/kWh**.")
        except Exception:
            pass

    if not (isinstance(price_vector, dict) and price_vector.get("values_usd_per_kwh")):
        st.caption("No price vector available yet to chart hourly variation.")
        return

    vals = price_vector["values_usd_per_kwh"]
    start_ts = price_vector.get("start_ts")
    try:
        start = datetime.fromisoformat(str(start_ts).replace("Z", "+00:00")) if start_ts else datetime.now(timezone.utc)
    except Exception:
        start = datetime.now(timezone.utc)

    periods = len(vals)
    idx = pd.date_range(start=start, periods=periods, freq=f"{int(interval_min)}min")
    df = pd.DataFrame({"ts": idx, "price_usd_per_kwh": pd.to_numeric(pd.Series(vals), errors="coerce").fillna(0.0)})
    df["hour"] = df["ts"].dt.hour

    hourly = df.groupby("hour", as_index=False)["price_usd_per_kwh"].mean().sort_values("hour")
    st.write("**Average price by hour of day**")
    st.bar_chart(hourly.set_index("hour"))

    kwh_per_step = 1.0 * (interval_min / 60.0)
    step_cost = df["price_usd_per_kwh"] * kwh_per_step
    daily_cost_example = step_cost.sum()

    cheapest_hr = hourly.loc[hourly["price_usd_per_kwh"].idxmin(), "hour"]
    priciest_hr = hourly.loc[hourly["price_usd_per_kwh"].idxmax(), "hour"]
    cheapest_price = float(hourly["price_usd_per_kwh"].min())
    priciest_price = float(hourly["price_usd_per_kwh"].max())
    shift_kwh = 10.0
    shift_savings = (priciest_price - cheapest_price) * shift_kwh

    st.markdown(
        f"""**Concrete $ examples**
- Steady **1 kW** all day → **${daily_cost_example:,.2f}** for energy (at these prices).
- Shift **{shift_kwh:.0f} kWh** from **{priciest_hr:02d}:00** (~${priciest_price:.2f}/kWh) to **{cheapest_hr:02d}:00** (~${cheapest_price:.2f}/kWh) → save **${shift_savings:,.2f}**."""
    )

    if st.session_state.get("res_mode", False):
        st.caption("Residential mode: demand charges & generator economics disabled.")

def view_tariff_prices():
    render_progress()
    tw_header_once("tariff_prices", "3) Tariff & Prices (Auto from ZIP)")

    prof = st.session_state.get("profile", {}) or {}
    zip_code = (prof.get("zip") or "").strip()
    region = prof.get("region", "") or "North America"
    application = prof.get("application", "") or "Commercial"

    # === CRITICAL BEHAVIOR ===
    # ZIP MUST be provided. But instead of ERROR + RETURN (which broke flow),
    # we show a friendly prompt AND STILL auto-advance when the user continues.
    if not zip_code:
        st.warning("Please enter your ZIP code in the Profile page so we can load your real tariff.")
        st.info("We will proceed with a placeholder until your ZIP is available.")
        zip_code = "02110"  # DEFAULT that never breaks logic (Boston Downtown)
        st.session_state["profile"]["zip"] = zip_code

    # --- Build tariff from ZIP / region / app ---
    tariff = _estimate_tariff_from_zip(zip_code, region=region, application=application)
    st.session_state["last_tariff"] = tariff

    st.subheader("Tariff we inferred from your ZIP")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Region:** {region or '—'}")
        st.write(f"**Application:** {application or '—'}")
        st.write(f"**ZIP:** {zip_code or '—'}")
    with c2:
        st.write(f"**Demand λ ($/kW):** {tariff['demand_charge_lambda']}")
        st.write(f"**Diesel cost ($/kWh):** {tariff['diesel_cost_usd_per_kwh']}")

    wk = tariff["weekday_periods"]
    df_periods = pd.DataFrame([{
        "Name": p["name"],
        "Start": f"{p['start_hour']:02d}:00",
        "End": f"{p['end_hour']:02d}:00",
        "$/kWh": p["price_usd_per_kwh"],
    } for p in wk])
    st.dataframe(df_periods, use_container_width=True)

    # --- Push tariff to backend (for optimization) ---
    payload = {
        "site_id": st.session_state.get("site_id", "site"),
        "timezone": tariff.get("timezone", "UTC"),
        "weekday_periods": tariff["weekday_periods"],
        "weekend_periods": tariff.get("weekend_periods"),
        "demand_charge_lambda": tariff["demand_charge_lambda"],
        "diesel_cost_usd_per_kwh": tariff["diesel_cost_usd_per_kwh"],
    }
    payload = taper_tariff_payload(payload)

    base = _backend_base()
    try:
        session.post(f"{base}/tariff/set", json=payload, timeout=10)
    except Exception as e:
        st.info(f"Tariff set warning: {e}")

       # --- Build price vector: backend first, then local override if flat ---
    interval_local = int(st.session_state.get("profile", {}).get("interval_minutes", 15))
    horizon_local = int(24 * 60 / interval_local)  # full 24 hours in steps

    try:
        start_ts = (st.session_state.get("forecast_load") or {}).get("start_ts") or \
                   datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
        r = session.get(
            f"{base}/tariff/price_vector",
            params={
                "site_id": st.session_state.get("site_id", "site"),
                "start_ts": start_ts,
                "horizon": int(horizon_local),
                "interval_min": int(interval_local),
            },
            timeout=15,
        )
        r.raise_for_status()
        vec = r.json()
    except Exception as e:
        st.warning(f"Price vector build failed (backend): {e}")
        vec = None

    # --- If backend vector is missing OR flat, rebuild locally from tariff ---
    if isinstance(vec, dict) and vec.get("values_usd_per_kwh"):
        uniq = sorted({round(x, 5) for x in vec["values_usd_per_kwh"]})
        if len(uniq) <= 1:
            # backend gave you essentially one price → flat line
            vec = _local_build_price_vector_from_tariff(
                st.session_state.get("last_tariff", {}),
                start_ts,
                horizon_local,
                interval_local,
            )
            if vec.get("values_usd_per_kwh"):
                st.caption("Backend returned flat prices — using local TOU approximation instead.")
    else:
        # No backend vec: always fall back to local
        vec = _local_build_price_vector_from_tariff(
            st.session_state.get("last_tariff", {}),
            start_ts,
            horizon_local,
            interval_local,
        )
        if vec.get("values_usd_per_kwh"):
            st.caption("Backend unavailable — using local tariff approximation.")

    # --- Store vec and show a quick preview ---
    if isinstance(vec, dict) and vec.get("values_usd_per_kwh"):
        st.session_state["price_vector"] = vec
        st.success("Price vector ready")

        df_vec = pd.DataFrame({"$/kWh": vec["values_usd_per_kwh"]})
        st.dataframe(df_vec, use_container_width=True)

        uniq = sorted({round(x, 5) for x in vec["values_usd_per_kwh"]})
        st.caption(f"Unique price levels: {len(uniq)} → {uniq[:6]}{' …' if len(uniq) > 6 else ''}")
    else:
        st.error("Could not build a price vector. You can still continue, but forecasts/plan may not run.")

    # --- Nav buttons ---
    colL, colR = st.columns([1, 1])
    with colL:
        if step_index() > 0 and st.button("← Back"):
            go_prev()
    with colR:
        if st.session_state.get("price_vector") and st.button("Next →"):
            go_next()
        elif not st.session_state.get("price_vector"):
            st.caption("We’ll try again on the next step if data becomes available.")
        
def view_forecasts():
    render_progress()
    tw_header_once("forecasts", "4) Forecasts")

    interval = int(st.session_state.get("profile", {}).get("interval_minutes", 15))
    steps_per_day = int(24 * 60 / interval)
    fl, fp = ensure_forecasts(horizon=steps_per_day, interval_min=interval)

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Load forecast")
        if fl and "values_kw" in fl:
            st.dataframe(pd.DataFrame({"load_kw": fl["values_kw"]}), use_container_width=True)
            st.metric("Avg Load (kW)", round(sum(fl["values_kw"]) / len(fl["values_kw"]), 1))
            st.metric("Max Load (kW)", round(max(fl["values_kw"]), 1))
            st.markdown(
                "- **What this is:** Expected site demand (kW) at each time-step.\n"
                "- **How we use it:** Drives how we size storage and shift flexible loads into cheaper hours."
            )
        else:
            st.error("No load forecast yet.")

    with cols[1]:
        st.subheader("PV forecast")
        if fp and "values_kw" in fp:
            st.dataframe(pd.DataFrame({"pv_kw": fp["values_kw"]}), use_container_width=True)
            st.metric("Avg PV (kW)", round(sum(fp["values_kw"]) / len(fp["values_kw"]), 1))
            st.metric("Max PV (kW)", round(max(fp["values_kw"]), 1))
            st.markdown(
                "- **What this is:** Estimated solar production (kW) at each time-step.\n"
                "- **How we use it:** Shows when clean/cheap onsite power is available to charge or offset grid use."
            )
        else:
            st.info("PV forecast not available.")

    # Optional note about tariff flatness, using price_vector safely
    price_vec = st.session_state.get("price_vector") or {}
    values = price_vec.get("values_usd_per_kwh")
    if values:
        vals = np.array(values, dtype=float)
        if len(vals) > 1 and np.std(vals) < 1e-3:
            st.caption(
                "Your tariff is effectively flat over this horizon – most savings will come "
                "from demand/peak management, not time-of-use shifting."
            )

    colL, colR = st.columns([1, 1])
    with colL:
        if step_index() > 0 and st.button("← Back"):
            go_prev()
    with colR:
        if (fl and "values_kw" in fl) and st.button("Next →"):
            go_next()
        elif not (fl and "values_kw" in fl):
            st.caption("We need at least a load forecast to continue.")

def view_optimize():
    render_progress()
    tw_header_once("optimize", "5) Optimization Plan")

    plan = ensure_plan()

    if plan and isinstance(plan, dict) and "schedule" in plan:
        st.success(f"Plan ready ({len(plan['schedule'])} steps)")
        st.dataframe(pd.DataFrame(plan["schedule"][:10]), use_container_width=True)
        try:
            pk_grid = max(float(row.get("p_grid_kw", 0.0)) for row in plan["schedule"])
            st.metric("Peak grid (kW)", f"{pk_grid:.1f}")
        except Exception:
            pass
        st.markdown(
            "- **What this plan is:** A time-step schedule that decides how much power comes from the grid, battery, solar, "
            "and (if enabled) generator at each interval.\n"
            "- **How to read it:** Use `p_grid_kw` to see remaining grid draw, `p_batt_kw` (>0 = charging, <0 = discharging), "
            "`soc` to track battery state-of-charge, and any PV/generator columns to see how onsite assets are used.\n"
            "- **Why it matters for solar:** Turns tariffs + load + PV into a concrete dispatch and savings story you can drop "
            "directly into proposals (sizing, peak shaving, backup coverage, payback)."
        )
    else:
        st.error("No plan yet. Ensure Prices and Forecasts are available.")

    colL, colR = st.columns([1, 1])
    with colL:
        if step_index() > 0 and st.button("← Back"):
            go_prev()
    with colR:
        if plan and st.button("Next →"):
            go_next()
        elif not plan:
            st.caption("We need a valid plan to continue.")            

def view_dashboard():
    tw_header_once("dashboard", "6) Dashboard & Savings")

    s = st.session_state

    # ----- Build site options safely -----
    # If you already store something like s["site_options"] or s["sites"],
    # we’ll use that. Otherwise, fall back to a single default.
    site_options = (
        s.get("site_options")
        or s.get("sites")
        or ["My site"]
    )

    # Make sure it's a list, not e.g. a single string
    if isinstance(site_options, str):
        site_options = [site_options]

    with st.container():
        st.markdown(
            "<div style='margin-top: 1.25rem; margin-bottom: 0.35rem; "
            "font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; "
            "color: rgba(255,255,255,0.65);'>SHOW SAVINGS FOR:</div>",
            unsafe_allow_html=True,
        )

        site = st.selectbox(
            label="Show savings for",
            options=site_options,
            index=0,
            key="show_savings_for",
            label_visibility="collapsed",
        )

    # small spacer before the main card
    st.write("")

    interval_min = int(s.get("profile", {}).get("interval_minutes", 15))
    energy_mode = get_energy_mode()
    st.caption(f"Mode: **{energy_mode}**")

    # --------- Make sure we have all core ingredients (even if user came straight from Upload) ---------
    # 1) Price vector from tariff
    if not s.get("price_vector"):
        horizon_steps = max(1, int(24 * 60 / interval_min))  # 24h worth of steps
        ensure_prices(horizon=horizon_steps, interval_min=interval_min)

    price_vec = s.get("price_vector") or {}
    prices = price_vec.get("values_usd_per_kwh")

    # 2) Forecasts (24h)
    horizon_steps = max(1, int(24 * 60 / interval_min))
    fl, fp = ensure_forecasts(horizon=horizon_steps, interval_min=interval_min)
    load_fc = (fl or {}).get("values_kw")

    # 3) Optimization plan (uses current price_vector + forecasts)
    plan = ensure_plan()

    # --------- If something is still missing, show friendly message and bail ----------
    if not (plan and prices and load_fc and plan.get("schedule")):
        st.info(
            "To compute savings we need all of:\n"
            "- A price vector (Tariff)\n"
            "- A load forecast (Forecasts)\n"
            "- An optimization plan (Optimize)\n\n"
            "Those should now load automatically once the backend responds."
        )
        save_user_snapshot()
        return

    schedule = plan.get("schedule", [])
    H = min(len(prices), len(load_fc), len(schedule))
    if H <= 0:
        st.info("We couldn’t align prices, load, and schedule over a valid horizon.")
        save_user_snapshot()
        return

    step_hours = interval_min / 60.0
    prices_arr = np.array(prices[:H], dtype=float)
    load_arr = np.array(load_fc[:H], dtype=float)
    grid_arr = np.array([float(row.get("p_grid_kw", load_arr[i])) for i, row in enumerate(schedule[:H])])

    # --------- Compute baseline vs optimized over the actual run horizon ----------
    # Baseline: grid = load (no optimization)
    baseline_run = float(np.sum(prices_arr * load_arr * step_hours))
    # Optimized: grid = p_grid_kw from schedule
    optimized_run = float(np.sum(prices_arr * grid_arr * step_hours))
    savings_run = max(0.0, baseline_run - optimized_run)

    hours = H * step_hours  # actual horizon in hours

    # --------- Prefer backend savings if they are richer, but don’t invent time scaling ----------
    backend = ensure_savings() or {}
    backend_abs = float(backend.get("savings_abs", 0.0) or 0.0)

    if backend_abs > 0:
        # Use backend’s cost breakdown if present
        baseline_run = float(backend.get("baseline_cost", baseline_run))
        optimized_run = float(backend.get("optimized_cost", optimized_run))
        savings_run = max(0.0, backend_abs)

    # Treat this run as a single representative day (design goal: horizon ≈ 24h)
    daily_baseline = baseline_run
    daily_savings = savings_run
    daily_optimized = max(0.0, daily_baseline - daily_savings)

    # --------- Anchor to the monthly bill if present ----------
    bill_monthly_cost = float(s.get("bill_monthly_cost", 0.0) or 0.0)

    # Period selector for user-facing numbers
    period = st.selectbox("Show savings for:", ["Daily", "Monthly", "Yearly"], index=1)

    if period == "Daily":
        main_savings = daily_savings
        if bill_monthly_cost > 0:
            baseline_display = bill_monthly_cost / 30.0
        else:
            baseline_display = daily_baseline
    elif period == "Monthly":
        main_savings = daily_savings * 30.0
        if bill_monthly_cost > 0:
            baseline_display = bill_monthly_cost
        else:
            baseline_display = daily_baseline * 30.0
    else:  # Yearly
        main_savings = daily_savings * 365.0
        if bill_monthly_cost > 0:
            baseline_display = bill_monthly_cost * 12.0
        else:
            baseline_display = daily_baseline * 365.0

    # Never let savings exceed baseline
    main_savings = max(0.0, min(main_savings, baseline_display))
    optimized_display = max(0.0, baseline_display - main_savings)
    pct_display = (main_savings / baseline_display * 100.0) if baseline_display > 0 else 0.0

    # --------- Hero savings card ----------
    st.markdown(
        f"""
        <div style="background:linear-gradient(180deg,#111827 0%,#0b1220 100%);
                    border-radius:18px;padding:22px 20px;color:#fff;
                    box-shadow:0 6px 20px rgba(0,0,0,.35);
                    margin-bottom:16px;border:1px solid #1f2937;">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
            <div>
              <div style="font-size:14px;opacity:.85;">Projected {period.lower()} savings</div>
              <div style="font-size:40px;font-weight:800;line-height:1.1;">
                ${main_savings:,.0f}
              </div>
              <div style="display:inline-block;
                          background:rgba(34,197,94,.18);
                          padding:6px 10px;border-radius:999px;font-size:12px;
                          border:1px solid rgba(34,197,94,.5);">
                {pct_display:.1f}% less than your baseline
              </div>
            </div>
            <div style="text-align:right;font-size:12px;opacity:.95;">
              <div>Site: <b>{s.get("site_id","—")}</b></div>
              <div>Interval: <b>{interval_min} min</b></div>
              <div>Run horizon: <b>{hours:.1f} h</b></div>
              <div>Run baseline: <b>${baseline_run:,.0f}</b></div>
              <div>Run optimized: <b>${optimized_run:,.0f}</b></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------- Summary metrics ----------
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"{period} Baseline", f"${baseline_display:,.0f}")
    with c2:
        st.metric(f"{period} Optimized", f"${optimized_display:,.0f}")
    with c3:
        delta_label = f"-${main_savings:,.0f}" if main_savings > 0 else "—"
        st.metric(f"{period} Savings", f"${main_savings:,.0f}", delta_label)

    # --------- Optimized spend curve over the run horizon ----------
    st.subheader("Optimized spend over this run horizon")

    H_plot = H
    prices_arr_plot = prices_arr[:H_plot]
    grid_arr_plot = grid_arr[:H_plot]
    step_cost_opt = prices_arr_plot * grid_arr_plot * step_hours

    try:
        start_ts = (fl or {}).get("start_ts") or datetime.now(timezone.utc).replace(
            second=0, microsecond=0
        ).isoformat()
        start_dt = datetime.fromisoformat(str(start_ts).replace("Z", "+00:00"))
    except Exception:
        start_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    idx = pd.date_range(start=start_dt, periods=H_plot, freq=f"{int(interval_min)}min")
    df_trend = pd.DataFrame(
        {"ts": idx, "OptimizedCost": np.cumsum(step_cost_opt)}
    ).set_index("ts")
    st.line_chart(df_trend)

    # --------- Month-over-month spend change note ----------
    prev_cost = float(s.get("prev_bill_monthly_cost", 0.0) or 0.0)

    if period == "Monthly" and bill_monthly_cost > 0 and prev_cost > 0:
        diff = bill_monthly_cost - prev_cost
        pct = abs(diff) / prev_cost * 100.0 if prev_cost > 0 else 0.0

        if diff > 0:
            direction = "up"
            color = "#f97316"  # orange-ish
        elif diff < 0:
            direction = "down"
            color = "#22c55e"  # green
        else:
            direction = "flat"
            color = "#9ca3af"

        if direction == "flat" or pct < 0.1:
            note = "Your monthly energy spend is roughly flat compared to your last bill."
        else:
            note = f"Your monthly energy spend is **{direction} {pct:.1f}%** versus your last bill."

        st.markdown(
            f"""
            <div style="margin-top:8px;font-size:12px;color:{color};opacity:0.9;">
              {note}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------- Cheapest hours highlight ----------
    st.markdown("---")
    st.subheader("Cheapest hours to run flexible loads")

    vals = price_vec.get("values_usd_per_kwh")
    if not vals:
        st.caption("We need a valid price vector to highlight cheapest hours.")
        save_user_snapshot()
        return

    vals_arr = np.array(vals, dtype=float)
    if len(vals_arr) < 2 or np.allclose(vals_arr, vals_arr[0]):
        st.info(
            "Your tariff is effectively flat over this horizon – "
            "there are no clearly 'cheapest hours' to highlight."
        )
        save_user_snapshot()
        return

    try:
        start_ts_p = price_vec.get("start_ts") or start_dt.isoformat()
        start_dt_p = datetime.fromisoformat(str(start_ts_p).replace("Z", "+00:00"))
    except Exception:
        start_dt_p = start_dt

    interval_min_vec = int(price_vec.get("interval_min", interval_min))
    idx_p = pd.date_range(
        start=start_dt_p, periods=len(vals_arr), freq=f"{interval_min_vec}min"
    )
    df_p = pd.DataFrame({"ts": idx_p, "price": vals_arr}).set_index("ts")

    q_low = np.quantile(df_p["price"], 0.25)
    q_high = np.quantile(df_p["price"], 0.75)
    cheap_mask = df_p["price"] <= q_low
    expensive_mask = df_p["price"] >= q_high

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(df_p.index, df_p["price"], linewidth=2)
    ax.set_xlabel("Time of day")
    ax.set_ylabel("$ / kWh")
    _apply_readable_time_axis(ax, interval_hours=2)

    def _shade(mask, color, alpha=0.15):
        on = None
        times = mask.index
        for i, flag in enumerate(mask):
            if flag and on is None:
                on = times[i]
            if (not flag and on is not None) or (flag and i == len(mask) - 1):
                off = times[i] if not flag else (
                    times[i] + timedelta(minutes=interval_min_vec)
                )
                ax.axvspan(on, off, color=color, alpha=alpha)
                on = None

    _shade(cheap_mask, "green", alpha=0.18)
    _shade(expensive_mask, "red", alpha=0.10)
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
- **Green bands** → cheapest hours: shift as much flexible load as possible here.  
- **Red bands** → most expensive hours: avoid starting big loads here (chillers, pumps, EVs, etc.).  
"""
    )

    # --------- Persist snapshot for next login ----------
    save_user_snapshot()
        
# =========================
# Intro + Comparison + Actions (only visible at 'optimize')
# =========================
def _can_show_compare_ui() -> bool:
    # Show the comparison / actions UI on both Optimize *and* Dashboard steps
    return st.session_state.get("step") in ("optimize", "dashboard")


if _can_show_compare_ui():
    # Ensure we have a page state
    if "page" not in st.session_state or st.session_state.page not in INTRO_PAGES:
        st.session_state.page = "intro"

    # Shared dark chart helper
    def _dark_chart(ax):
        plt.style.use("dark_background")
        ax.set_facecolor("#0f0f0f")
        ax.figure.set_facecolor("#0b0b0b")
        ax.grid(color="#333", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.tick_params(colors="#eaeaea")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.xaxis.label.set_color("#ffffff")
        ax.yaxis.label.set_color("#ffffff")
        return ax

    # ---------- INTRO PAGE ----------
    if st.session_state.page == "intro":
        tw_header_once("intro", "How this works")
        st.markdown(
            """
1) We analyze your historical usage and tariff.  
2) We generate an optimized schedule that shifts flexible loads into cheaper hours.  
3) Then you'll see **Current vs Optimized** spend, followed by **Daily Actions**.
"""
        )
        if st.button("Show my comparison"):
            st.session_state.page = "compare"
            st.rerun()

    # ---------- COMPARE PAGE ----------
    if st.session_state.page == "compare":
        tw_header_once("compare", "Spending: Current vs Optimized")
        st.caption("Hourly view based on your actual tariff, forecast and optimization plan.")

        # Pull state we need
        prices  = (st.session_state.get("price_vector")  or {}).get("values_usd_per_kwh")
        load_fc = (st.session_state.get("forecast_load") or {}).get("values_kw")
        plan    =  st.session_state.get("plan")

        if not (prices and load_fc and plan and plan.get("schedule")):
            st.warning(
                "To see a real comparison, we need:\n"
                "- A price vector (Tariff step)\n"
                "- A load forecast (Forecasts step)\n"
                "- An optimization plan (Optimize step)"
            )
        else:
            # ---------- Build baseline vs optimized cost series ----------
            H = min(len(prices), len(load_fc), len(plan["schedule"]))
            interval_min = int(st.session_state.get("profile", {}).get("interval_minutes", 15))

            # Time axis
            start_ts = (st.session_state.get("forecast_load") or {}).get("start_ts") \
                       or datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
            try:
                start_dt = datetime.fromisoformat(str(start_ts).replace("Z", "+00:00"))
            except Exception:
                start_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)

            idx = pd.date_range(start=start_dt, periods=H, freq=f"{interval_min}min")

            prices_arr = np.array(prices[:H], dtype=float)
            load_arr   = np.array(load_fc[:H], dtype=float)
            grid_opt   = np.array([float(r.get("p_grid_kw", 0.0)) for r in plan["schedule"][:H]])

            step_hours = interval_min / 60.0

            # Baseline: grid = load (no optimization)
            baseline_step_cost  = prices_arr * load_arr * step_hours
            # Optimized: grid = p_grid_kw from schedule
            optimized_step_cost = prices_arr * grid_opt * step_hours

            baseline_df  = pd.DataFrame({"ts": idx, "spend": baseline_step_cost}).set_index("ts")
            optimized_df = pd.DataFrame({"ts": idx, "spend": optimized_step_cost}).set_index("ts")

            hourly_baseline  = baseline_df.resample("1H")["spend"].sum()
            hourly_optimized = optimized_df.resample("1H")["spend"].sum()

            # ---------- Plot baseline ----------
            st.subheader("Current (baseline) hourly spend")
            fig, ax = plt.subplots(figsize=(9, 3.9))
            _dark_chart(ax)
            ax.plot(hourly_baseline.index, hourly_baseline.values, linewidth=2)
            ax.set_xlabel("Time of day")
            ax.set_ylabel("Spend ($)")
            _apply_readable_time_axis(ax, interval_hours=2)
            st.pyplot(fig, clear_figure=True)

            # ---------- Plot optimized ----------
            st.subheader("Optimized hourly spend")
            fig2, ax2 = plt.subplots(figsize=(9, 3.9))
            _dark_chart(ax2)
            ax2.plot(hourly_optimized.index, hourly_optimized.values, linewidth=2)
            ax2.set_xlabel("Time of day")
            ax2.set_ylabel("Spend ($)")
            _apply_readable_time_axis(ax2, interval_hours=2)
            st.pyplot(fig2, clear_figure=True)

            # ---------- Summary numbers ----------
            total_current   = float(hourly_baseline.sum())
            total_optimized = float(hourly_optimized.sum())
            delta           = total_current - total_optimized
            pct             = (delta / total_current * 100.0) if total_current > 0 else 0.0

            st.success(
                f"Estimated savings over this run horizon: **${delta:,.2f}** "
                f"(Current ${total_current:,.2f} → Optimized ${total_optimized:,.2f}, {pct:,.1f}% reduction)."
            )

        st.markdown("---")
        if st.button("Show my Daily Actions →"):
            st.session_state.page = "actions"
            st.rerun()

    # ---------- ACTIONS PAGE ----------
    if st.session_state.page == "actions":
        tw_header_once("actions", "Daily Actions to Save Energy")
        st.caption("Quick steps + visual windows to schedule flexible loads.")

        # Simple synthetic price profile for illustration
        start = datetime(2025, 1, 1, 0, 0)
        idx = pd.date_range(start, start + timedelta(hours=24), freq="60min", inclusive="left")
        price = np.where((idx.hour >= 7) & (idx.hour < 19), 0.28, 0.14)
        df = pd.DataFrame({"ts": idx, "price": price}).set_index("ts")
        cheap_mask = df["price"] == df["price"].min()
        expensive_mask = df["price"] == df["price"].max()

        fig, ax = plt.subplots(figsize=(10, 3.8))
        _dark_chart(ax)
        ax.plot(df.index, df["price"], linewidth=2)
        ax.set_xlabel("Time of Day")
        ax.set_ylabel("$ / kWh")
        _apply_readable_time_axis(ax, interval_hours=2)

        def _shade_blocks(mask, color, alpha=0.12):
            on = None
            times = mask.index
            for i, flag in enumerate(mask):
                if flag and on is None:
                    on = times[i]
                if (not flag and on is not None) or (flag and i == len(mask) - 1):
                    off = times[i] if not flag else (times[i] + timedelta(hours=1))
                    ax.axvspan(on, off, color=color, alpha=alpha)
                    on = None

        _shade_blocks(cheap_mask, color="#22c55e", alpha=0.18)
        _shade_blocks(expensive_mask, color="#ef4444", alpha=0.12)

        st.pyplot(fig, clear_figure=True)

        st.subheader("Your quick wins (today)")
        st.markdown(
            """
- **Run high-load appliances after 9:00pm** (dryer, dishwasher, EV charging).  
- **Pre-cool or pre-heat** in cheap hours; coast through peak (7:00am–7:00pm).  
- **Charge batteries noon–4:00pm** if you have PV surplus or cheap midday rates.  
- **Avoid peak spikes**: stagger big loads; don’t start them together at 6–7pm.  
- **Hold a 25% battery reserve** for the evening unless prices are unusually low.
"""
        )

# =========================
# Router
# =========================
page = st.session_state.get("step", "profile")

# Wrap all main pages in a centered shell for a more "product" feel
with st.container():
    st.markdown("<div class='main-shell'>", unsafe_allow_html=True)

    if page == "profile":
        view_profile()
    elif page == "upload":
        view_upload()
    elif page == "tariff_prices":
        view_tariff_prices()
    elif page == "forecasts":
        view_forecasts()
    elif page == "optimize":
        view_optimize()
    elif page == "dashboard":
        view_dashboard()
    else:
        go_step("profile")

    st.markdown("</div>", unsafe_allow_html=True)
