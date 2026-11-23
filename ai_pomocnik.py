"""Built-in AI Earth Observation Assistant (Gemini 2.5 Flash REST + Smart Local Fallback).

Converts natural-language Slovenian or English remote-sensing requests into a strict,
validated geodetic execution plan and runs the corresponding backend pipeline strictly
in projected metric coordinates (D96/TM ``EPSG:3794`` or UTM 33N ``EPSG:32633``):

- Primary parser: Google Gemini 2.5 Flash REST API (via ``requests``, zero heavy SDK).
- Built-in local fallback: deterministic Slovenian/English regex + keyword + declension
  parser that works 100% offline with zero API key required.
- Geodetic orchestrator: geocodes place -> builds metric AOI -> runs RGB scene, spectral
  index, multi-decade time series, or rapid natural disaster assessment -> produces exact
  metric statistics, GeoTIFF/CSV export bytes, technical report summary, and Cartography handoff.
"""

from __future__ import annotations

import calendar
import gc
import json
import os
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import requests
import xarray as xr

from analize import (
    SAR_METRICS,
    SPECTRAL_INDICES,
    apply_cloud_mask,
    compute_index,
    compute_index_stats,
    get_required_bands_for_sensor,
)
from casovne_vrste import extract_time_series
from geodezija_crs import (
    CRS_D96TM,
    bbox_from_point_wgs84,
    export_to_geotiff_bytes,
)
from nesrece import (
    DISASTER_FLOOD,
    DISASTER_FOREST,
    DISASTER_WILDFIRE,
    run_disaster_assessment,
)
from stac_loader import (
    DEFAULT_MIN_AOI_COVERAGE_PCT,
    HISTORICAL_MIN_DATE,
    INTERVAL_14D,
    INTERVAL_ALL,
    INTERVAL_MONTHLY,
    INTERVAL_SEASONAL,
    INTERVAL_YEARLY,
    SENSOR_LANDSAT,
    SENSOR_S1,
    SENSOR_S2,
    load_scene,
    search_scenes,
)

# ---------------------------------------------------------------------------
# Constants, Schema Allowed Values & Quick Example Prompts
# ---------------------------------------------------------------------------
DEFAULT_GEMINI_MODEL: str = "gemini-3.5-flash"
FALLBACK_GEMINI_MODELS: tuple[str, ...] = ("gemini-3.5-flash", "gemini-2.5-flash")
GEMINI_MODELS_LIST_URL: str = "https://generativelanguage.googleapis.com/v1beta/models"

GEMINI_MODEL_OPTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "gemini-3.5-flash",
        "label_sl": "gemini-3.5-flash (Zelo hiter, najmanjša poraba tokenov — privzeto)",
        "label_en": "gemini-3.5-flash (Very fast, lowest token usage — default)",
    },
    {
        "id": "gemini-3.6-flash",
        "label_sl": "gemini-3.6-flash (Hiter, uravnotežen)",
        "label_en": "gemini-3.6-flash (Fast, balanced)",
    },
    {
        "id": "gemini-3.8-flash",
        "label_sl": "gemini-3.8-flash (Napredni Flash)",
        "label_en": "gemini-3.8-flash (Advanced Flash)",
    },
    {
        "id": "gemini-4-hydrogen",
        "label_sl": "gemini-4-hydrogen (Gemini 4 Hydrogen)",
        "label_en": "gemini-4-hydrogen (Gemini 4 Hydrogen)",
    },
    {
        "id": "gemini-2.5-flash",
        "label_sl": "gemini-2.5-flash (Klasični stabilni)",
        "label_en": "gemini-2.5-flash (Classic stable)",
    },
    {
        "id": "custom",
        "label_sl": "Po meri (Custom Model ID)",
        "label_en": "Custom (Custom Model ID)",
    },
)


def build_gemini_endpoint(model_id: str | None = None) -> str:
    """Build the Gemini REST API generateContent URL for any selected or custom model ID."""
    clean = str(model_id or DEFAULT_GEMINI_MODEL).strip()
    if clean.startswith("models/"):
        clean = clean[len("models/") :]
    if not clean or clean == "custom":
        clean = DEFAULT_GEMINI_MODEL
    return f"https://generativelanguage.googleapis.com/v1beta/models/{clean}:generateContent"


def list_available_gemini_models(api_key: str, timeout: int = 10) -> list[str]:
    """Query Google AI Studio REST API and return exact model IDs supporting ``generateContent``."""
    clean_key = str(api_key or "").strip()
    if not clean_key:
        raise ValueError("API key is empty.")

    resp = requests.get(
        GEMINI_MODELS_LIST_URL,
        params={"key": clean_key, "pageSize": 100},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    raw_models = payload.get("models") or []

    out: list[str] = []
    seen: set[str] = set()
    for entry in raw_models:
        methods = entry.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        name = str(entry.get("name") or "").strip()
        if name.startswith("models/"):
            name = name[len("models/") :]
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


GEMINI_ENDPOINT: str = build_gemini_endpoint(DEFAULT_GEMINI_MODEL)

VALID_SENSORS: tuple[str, ...] = (SENSOR_S2, SENSOR_LANDSAT, SENSOR_S1)
VALID_TASKS: tuple[str, ...] = (
    "scene_rgb",
    "spectral_index",
    "time_series",
    "disaster_flood",
    "disaster_fire",
    "disaster_forest",
)
VALID_INDICES: tuple[str, ...] = ("NDVI", "NDWI", "MNDWI", "NDBI", "NBR", "VV")
VALID_RESOLUTIONS: tuple[int, ...] = (10, 20, 30, 60)

INTERVAL_LABEL_TO_KEY: dict[str, str] = {
    "Vsi posnetki": INTERVAL_ALL,
    "Vsakih 14 dni": INTERVAL_14D,
    "Mesečno": INTERVAL_MONTHLY,
    "Sezonsko (3 mes.)": INTERVAL_SEASONAL,
    "Letno": INTERVAL_YEARLY,
}
VALID_INTERVAL_LABELS: tuple[str, ...] = tuple(INTERVAL_LABEL_TO_KEY.keys())

AI_EXAMPLE_PROMPTS: tuple[dict[str, str], ...] = (
    {
        "id": "celje_flood",
        "sl": "🌊 Pokaži poplave v Celju avgusta 2023",
        "en": "🌊 Show floods in Celje in August 2023",
    },
    {
        "id": "kras_fire",
        "sl": "🔥 Analiziraj požar na Krasu julija 2022",
        "en": "🔥 Analyze wildfire on Kras in July 2022",
    },
    {
        "id": "bled_ndvi",
        "sl": "🌿 Izračunaj NDVI za Bled poleti 2025",
        "en": "🌿 Compute NDVI for Bled in summer 2025",
    },
    {
        "id": "lj_ts_30y",
        "sl": "📈 30-letna časovna vrsta vegetacije za Ljubljano (1995–2025)",
        "en": "📈 30-year vegetation time series for Ljubljana (1995–2025)",
    },
)

# Built-in Slovenian places & declension stem matching (normalized ASCII stems -> canonical name + coords)
_SLOVENIAN_PLACE_STEMS: tuple[tuple[str, str, float, float], ...] = (
    ("ljubljansko barj", "Ljubljansko barje", 46.0050, 14.4600),
    ("barj", "Ljubljansko barje", 46.0050, 14.4600),
    ("savinjsk", "Celje", 46.2397, 15.2677),
    ("kostanjevic", "Kras", 45.8480, 13.6350),
    ("cerj", "Kras", 45.8480, 13.6350),
    ("kras", "Kras", 45.8480, 13.6350),
    ("ljubljan", "Ljubljana", 46.0569, 14.5058),
    ("maribor", "Maribor", 46.5547, 15.6459),
    ("celj", "Celje", 46.2397, 15.2677),
    ("kranjsk", "Kranjska Gora", 46.4845, 13.7857),
    ("kranj", "Kranj", 46.2389, 14.3556),
    ("koper", "Koper", 45.5481, 13.7302),
    ("kopr", "Koper", 45.5481, 13.7302),
    ("novo mest", "Novo mesto", 45.8040, 15.1689),
    ("novem mest", "Novo mesto", 45.8040, 15.1689),
    ("nova goric", "Nova Gorica", 45.9560, 13.6484),
    ("novi goric", "Nova Gorica", 45.9560, 13.6484),
    ("velenj", "Velenje", 46.3592, 15.1103),
    ("mursk", "Murska Sobota", 46.6625, 16.1664),
    ("prekmur", "Murska Sobota", 46.6625, 16.1664),
    ("ptuj", "Ptuj", 46.4200, 15.8702),
    ("bled", "Bled", 46.3683, 14.1146),
    ("bohinj", "Bohinj", 46.2784, 13.8606),
    ("triglav", "Triglav", 46.3789, 13.8367),
    ("cerknis", "Cerkniško jezero", 45.7745, 14.3755),
    ("cerknic", "Cerkniško jezero", 45.7745, 14.3755),
    ("postojn", "Postojna", 45.7744, 14.2153),
    ("kocev", "Kočevje", 45.6433, 14.8633),
    ("pohorj", "Pohorje", 46.5167, 15.4667),
    ("kamnik", "Kamnik", 46.2259, 14.6121),
    ("kamnisk", "Kamnik", 46.2259, 14.6121),
    ("domzal", "Domžale", 46.1377, 14.5937),
    ("skofj", "Škofja Loka", 46.1655, 14.3063),
    ("jesenic", "Jesenice", 46.4306, 14.0669),
    ("trbovlj", "Trbovlje", 46.1550, 15.0533),
    ("krsk", "Krško", 45.9592, 15.4917),
    ("brezic", "Brežice", 45.9033, 15.5911),
    ("piran", "Piran", 45.5283, 13.5683),
    ("portoroz", "Portorož", 45.5143, 13.5921),
    ("izol", "Izola", 45.5369, 13.6619),
    ("sezan", "Sežana", 45.7092, 13.8733),
    ("ajdovscin", "Ajdovščina", 45.8860, 13.9095),
    ("idrij", "Idrija", 46.0028, 14.0306),
    ("tolmin", "Tolmin", 46.1830, 13.7332),
    ("bovec", "Bovec", 46.3380, 13.5524),
    ("bovc", "Bovec", 46.3380, 13.5524),
    ("slovenj grad", "Slovenj Gradec", 46.5103, 15.0806),
    ("ravn", "Ravne na Koroškem", 46.5431, 14.9692),
    ("lendav", "Lendava", 46.5649, 16.4509),
    ("ormoz", "Ormož", 46.4114, 16.1544),
    ("radovljic", "Radovljica", 46.3444, 14.1744),
    ("logarsk", "Logarska dolina", 46.3947, 14.6292),
    ("vipav", "Vipava", 45.8467, 13.9631),
)

_MONTH_PATTERNS: tuple[tuple[str, int], ...] = (
    ("januar", 1), ("january", 1), ("jan ", 1),
    ("februar", 2), ("february", 2), ("feb ", 2),
    ("marec", 3), ("marc", 3), ("march", 3),
    ("april", 4),
    ("maj", 5), ("may", 5),
    ("junij", 6), ("june", 6),
    ("julij", 7), ("july", 7),
    ("avgust", 8), ("august", 8),
    ("septemb", 9),
    ("oktob", 10), ("octob", 10),
    ("novemb", 11),
    ("decemb", 12),
)


def _normalize_ascii(text: str) -> str:
    """Lowercase and strip diacritics for robust Slovenian/English keyword matching."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower().strip()


# ---------------------------------------------------------------------------
# API Key Discovery
# ---------------------------------------------------------------------------
def get_active_gemini_api_key(user_key: str | None = None) -> tuple[str | None, str]:
    """Resolve Gemini API key from session input, env vars (GitHub Codespaces / HF Spaces Secrets), or Streamlit secrets.

    Checks ``os.environ.get("GEMINI_API_KEY")`` before ``st.secrets`` so GitHub Codespaces
    and Hugging Face Spaces deployments work cleanly without touching ``.streamlit/secrets.toml``,
    and wraps ``st.secrets`` access safely so ``StreamlitSecretNotFoundError`` is never raised.

    Returns ``(api_key_or_none, source_tag)`` where ``source_tag`` is one of
    ``"session"``, ``"env"``, ``"secrets"``, or ``"none"``.
    """
    if user_key and str(user_key).strip():
        return str(user_key).strip(), "session"

    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and str(env_key).strip():
        return str(env_key).strip(), "env"

    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            sec_key = st.secrets.get("GEMINI_API_KEY")
            if sec_key and str(sec_key).strip():
                return str(sec_key).strip(), "secrets"
    except Exception:
        pass

    return None, "none"


# ---------------------------------------------------------------------------
# Strict Plan Validation & Geodetic Normalization
# ---------------------------------------------------------------------------
def _parse_iso_date(raw: Any, fallback: date) -> date:
    if isinstance(raw, date) and not isinstance(raw, datetime):
        d = raw
    elif isinstance(raw, datetime):
        d = raw.date()
    else:
        try:
            d = datetime.strptime(str(raw).strip()[:10], "%Y-%m-%d").date()
        except Exception:
            d = fallback
    today = date.today()
    if d < HISTORICAL_MIN_DATE:
        d = HISTORICAL_MIN_DATE
    if d > today:
        d = today
    return d


def validate_and_normalize_plan(raw_plan: dict[str, Any], engine: str = "local") -> dict[str, Any]:
    """Validate and enforce strict geodetic/sensor rules on any parsed JSON plan."""
    today = date.today()
    place_name = str(raw_plan.get("place_name") or "Ljubljana").strip() or "Ljubljana"

    try:
        radius_m = int(round(float(raw_plan.get("radius_m", 4000))))
    except Exception:
        radius_m = 4000
    radius_m = int(np.clip(radius_m, 500, 12000))

    task_type = str(raw_plan.get("task_type") or "spectral_index").strip().lower()
    if task_type not in VALID_TASKS:
        task_type = "spectral_index"

    index_name = str(raw_plan.get("index_name") or "NDVI").strip().upper()
    if index_name not in VALID_INDICES:
        index_name = "VV" if task_type == "disaster_flood" else "NDVI"

    default_start = today - timedelta(days=90)
    start_d = _parse_iso_date(raw_plan.get("start_date"), default_start)
    end_d = _parse_iso_date(raw_plan.get("end_date"), today)
    if end_d < start_d:
        start_d, end_d = end_d, start_d
    event_d = _parse_iso_date(raw_plan.get("event_date"), end_d)

    # Enforce physically valid sensor rules
    sensor = str(raw_plan.get("sensor") or SENSOR_S2).strip().lower()
    if sensor not in VALID_SENSORS:
        sensor = SENSOR_S2

    if task_type == "disaster_flood" or index_name == "VV":
        sensor = SENSOR_S1
        index_name = "VV"
    elif task_type in ("disaster_fire", "disaster_forest"):
        if event_d < date(2016, 1, 1):
            sensor = SENSOR_LANDSAT
        elif sensor == SENSOR_S1:
            sensor = SENSOR_S2
        index_name = "NBR" if task_type == "disaster_fire" else "NDVI"
    else:
        if start_d < date(2016, 1, 1):
            sensor = SENSOR_LANDSAT

    # Normalize temporal interval label
    interval = str(raw_plan.get("interval") or "Mesečno").strip()
    if interval not in INTERVAL_LABEL_TO_KEY:
        span_days = (end_d - start_d).days
        if span_days >= 1825:
            interval = "Letno"
        elif span_days >= 730:
            interval = "Sezonsko (3 mes.)"
        elif span_days >= 150:
            interval = "Mesečno"
        else:
            interval = "Vsi posnetki"

    # Normalize spatial resolution (16 GB RAM profile: native 10 m for Sentinel, 30 m for Landsat)
    try:
        res_m = int(raw_plan.get("resolution_m", 30 if sensor == SENSOR_LANDSAT else 10))
    except Exception:
        res_m = 30 if sensor == SENSOR_LANDSAT else 10
    if res_m not in VALID_RESOLUTIONS:
        res_m = min(VALID_RESOLUTIONS, key=lambda r: abs(r - res_m))
    if sensor == SENSOR_LANDSAT and res_m < 30:
        res_m = 30

    expl_sl = str(raw_plan.get("explanation_sl") or "").strip()
    expl_en = str(raw_plan.get("explanation_en") or "").strip()
    if not expl_sl:
        expl_sl = (
            f"Načrt: {task_type} ({index_name}) za območje {place_name} (radij {radius_m} m) "
            f"s senzorjem {sensor} v obdobju {start_d.isoformat()} – {end_d.isoformat()}."
        )
    if not expl_en:
        expl_en = (
            f"Plan: {task_type} ({index_name}) for {place_name} (radius {radius_m} m) "
            f"using {sensor} over {start_d.isoformat()} – {end_d.isoformat()}."
        )

    return {
        "place_name": place_name,
        "radius_m": radius_m,
        "sensor": sensor,
        "task_type": task_type,
        "index_name": index_name,
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "event_date": event_d.isoformat(),
        "interval": interval,
        "resolution_m": res_m,
        "explanation_sl": expl_sl,
        "explanation_en": expl_en,
        "parser_engine": engine,
    }


# ---------------------------------------------------------------------------
# Engine 1: Gemini 2.5 Flash REST API Parser
# ---------------------------------------------------------------------------
_GEMINI_SYSTEM_PROMPT = """You are a geodetic Earth Observation (remote sensing) planner for Slovenia and Europe.
Convert the user's Slovenian or English request into a STRICT JSON object with these exact keys:
{
  "place_name": string (nominative place name, e.g. "Celje", "Kras", "Bled", "Ljubljana"),
  "radius_m": integer (1000 to 10000, default 4000, use 5000-6000 for floods/wildfires),
  "sensor": "sentinel-2-l2a" | "landsat-c2-l2" | "sentinel-1-rtc",
  "task_type": "scene_rgb" | "spectral_index" | "time_series" | "disaster_flood" | "disaster_fire" | "disaster_forest",
  "index_name": "NDVI" | "NDWI" | "MNDWI" | "NDBI" | "NBR" | "VV",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "event_date": "YYYY-MM-DD",
  "interval": "Vsi posnetki" | "Vsakih 14 dni" | "Mesečno" | "Sezonsko (3 mes.)" | "Letno",
  "resolution_m": 10 | 20 | 30 | 60,
  "explanation_sl": string (concise 1-2 sentence technical explanation in Slovenian),
  "explanation_en": string (concise 1-2 sentence technical explanation in English)
}
STRICT SCIENTIFIC RULES:
1. For floods / inundation / radar / all-weather -> sensor="sentinel-1-rtc", task_type="disaster_flood", index_name="VV".
2. For wildfires / burn severity / Kras fire -> task_type="disaster_fire", index_name="NBR".
3. For windthrow / clearcut / forest loss / drought disturbance -> task_type="disaster_forest", index_name="NDVI".
4. Whenever start_date is before 2016-01-01 -> MUST use sensor="landsat-c2-l2" and resolution_m=30.
5. For multi-year trends / time series (>5 years) -> task_type="time_series", interval="Letno".
6. Known Slovenian events: Savinja/Celje floods = 2023-08-04, Barje floods = 2023-08-05, Kras wildfire = 2022-07-22.
Return ONLY valid JSON without markdown code fences."""


def parse_eo_request_gemini(
    query: str,
    api_key: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout: int = 15,
) -> dict[str, Any]:
    """Call Gemini REST API with JSON output mode for ``model`` and validate the resulting plan."""
    clean_model = str(model or DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    if clean_model.startswith("models/"):
        clean_model = clean_model[len("models/") :]
    endpoint_url = build_gemini_endpoint(clean_model)

    today_iso = date.today().isoformat()
    payload = {
        "systemInstruction": {"parts": [{"text": _GEMINI_SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Today is {today_iso}. User request: {query}"}],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
        },
    }
    resp = requests.post(
        endpoint_url,
        params={"key": api_key},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    text = body["candidates"][0]["content"]["parts"][0]["text"]
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
    parsed = json.loads(cleaned)
    plan = validate_and_normalize_plan(parsed, engine="gemini")
    plan["gemini_model"] = clean_model
    return plan


# ---------------------------------------------------------------------------
# Engine 2: Built-in Smart Local Fallback Parser (100% Offline, Deterministic)
# ---------------------------------------------------------------------------
def _extract_place_local(query: str, norm_q: str) -> str:
    # 1. Check built-in Slovenian place stems (longest stem first)
    for stem, canonical, _lat, _lon in _SLOVENIAN_PLACE_STEMS:
        if stem in norm_q:
            return canonical

    # 2. Check preposition-based capitalized word extraction (e.g. "za Maribor", "v Celju", "in Vienna", "for Bled")
    prep_match = re.search(
        r"\b(?:za|v|na|okoli|pri|in|for|around|near|over)\s+([A-ZČŠŽ][a-zčšž]+(?:\s+[A-ZČŠŽ][a-zčšž]+)?)",
        query,
    )
    if prep_match:
        cand = prep_match.group(1).strip()
        if _normalize_ascii(cand) not in {"ndvi", "ndwi", "mndwi", "ndbi", "nbr", "rgb", "sar", "landsat", "sentinel"}:
            return cand

    return "Ljubljana"


def _extract_dates_local(norm_q: str) -> tuple[date, date, date, str]:
    """Extract (start_date, end_date, event_date, default_interval) from Slovenian/English text."""
    today = date.today()

    # Check explicit year range: e.g. 1995-2025, 1995–2025, 1995 to 2025, od 1995 do 2025
    yr_range = re.search(r"\b(19[89]\d|20[0-2]\d)\s*(?:[-–—]|do|to)\s*(19[89]\d|20[0-2]\d)\b", norm_q)
    if yr_range:
        y1, y2 = int(yr_range.group(1)), int(yr_range.group(2))
        if y2 < y1:
            y1, y2 = y2, y1
        s_d = date(y1, 1, 1)
        e_d = min(date(y2, 12, 31), today)
        ev_d = e_d
        span_y = y2 - y1
        interval = "Letno" if span_y >= 5 else ("Sezonsko (3 mes.)" if span_y >= 2 else "Mesečno")
        return s_d, e_d, ev_d, interval

    # Check "N-letna" / "N-year" duration (e.g. "30-letna")
    dur_match = re.search(r"\b(\d{1,2})\s*-?\s*(?:letn|let|year)", norm_q)
    years_found = [int(y) for y in re.findall(r"\b(19[89]\d|20[0-2]\d)\b", norm_q)]

    # Detect month
    month_num: int | None = None
    for pat, m_idx in _MONTH_PATTERNS:
        if pat in norm_q:
            month_num = m_idx
            break

    # Detect season
    season: str | None = None
    if any(k in norm_q for k in ("polet", "summer")):
        season = "summer"
    elif any(k in norm_q for k in ("pomlad", "spomlad", "spring")):
        season = "spring"
    elif any(k in norm_q for k in ("jesen", "autumn", "fall")):
        season = "autumn"
    elif any(k in norm_q for k in ("pozim", "zim", "winter")):
        season = "winter"

    if dur_match and not month_num and not season:
        n_years = max(1, min(int(dur_match.group(1)), 42))
        end_yr = max(years_found) if years_found else today.year
        s_d = max(HISTORICAL_MIN_DATE, date(end_yr - n_years, 1, 1))
        e_d = min(date(end_yr, 12, 31), today)
        interval = "Letno" if n_years >= 5 else "Sezonsko (3 mes.)"
        return s_d, e_d, e_d, interval

    target_year = years_found[0] if years_found else today.year
    if target_year > today.year:
        target_year = today.year

    if month_num is not None:
        last_day = calendar.monthrange(target_year, month_num)[1]
        s_d = date(target_year, month_num, 1)
        e_d = min(date(target_year, month_num, last_day), today)
        # Known Slovenian historical disaster peak dates
        if target_year == 2023 and month_num == 8 and any(k in norm_q for k in ("poplav", "flood")):
            ev_d = date(2023, 8, 5) if "barj" in norm_q else date(2023, 8, 4)
        elif target_year == 2022 and month_num == 7 and any(k in norm_q for k in ("pozar", "fire", "kras")):
            ev_d = date(2022, 7, 22)
        else:
            ev_d = min(date(target_year, month_num, min(15, last_day)), today)
        return s_d, e_d, ev_d, "Vsi posnetki"

    if season == "summer":
        s_d = date(target_year, 6, 1)
        e_d = min(date(target_year, 8, 31), today)
        if s_d > today:
            s_d = date(today.year - 1, 6, 1)
            e_d = date(today.year - 1, 8, 31)
        return s_d, e_d, e_d, "Mesečno"
    if season == "spring":
        s_d = date(target_year, 3, 1)
        e_d = min(date(target_year, 5, 31), today)
        return s_d, e_d, e_d, "Mesečno"
    if season == "autumn":
        s_d = date(target_year, 9, 1)
        e_d = min(date(target_year, 11, 30), today)
        return s_d, e_d, e_d, "Mesečno"
    if season == "winter":
        s_d = date(target_year, 1, 1)
        e_d = min(date(target_year, 2, 28), today)
        return s_d, e_d, e_d, "Mesečno"

    if years_found:
        s_d = date(target_year, 5, 1)
        e_d = min(date(target_year, 9, 30), today)
        if s_d > today:
            s_d = date(target_year, 1, 1)
            e_d = today
        return s_d, e_d, e_d, "Mesečno"

    # Default: last 90 days
    return today - timedelta(days=90), today, today, "Mesečno"


def parse_eo_request_local(query: str) -> dict[str, Any]:
    """Deterministic offline Slovenian/English remote-sensing query parser."""
    norm_q = _normalize_ascii(query)
    place_name = _extract_place_local(query, norm_q)
    start_d, end_d, event_d, interval = _extract_dates_local(norm_q)

    # 1. Detect task_type & index_name
    is_flood = any(k in norm_q for k in ("poplav", "flood", "inundat", "razlitj"))
    is_fire = any(k in norm_q for k in ("pozar", "wildfire", "fire", "burn", "dnbr", "opozar"))
    is_forest = any(k in norm_q for k in ("vetrolom", "posek", "zled", "susa", "windthrow", "clearcut", "deforest", "dndvi"))
    is_ts = any(
        k in norm_q
        for k in (
            "casovn",
            "time series",
            "timeseries",
            "trend",
            "skozi cas",
            "spreminj",
            "over time",
            "letna",
            "year",
            "zgodovin",
            "histor",
        )
    ) or ((end_d - start_d).days >= 400 and not (is_flood or is_fire or is_forest))

    # Index detection
    if "mndwi" in norm_q:
        index_name = "MNDWI"
    elif "ndwi" in norm_q or any(k in norm_q for k in ("vodn", "jezer", "water", "lake", "river", "reka")):
        index_name = "NDWI"
    elif "ndbi" in norm_q or any(k in norm_q for k in ("pozidav", "urban", "mesto", "city", "built")):
        index_name = "NDBI"
    elif "nbr" in norm_q or is_fire:
        index_name = "NBR"
    elif "vv" in norm_q or "sar" in norm_q or "radar" in norm_q or is_flood:
        index_name = "VV"
    else:
        index_name = "NDVI"

    if is_flood:
        task_type = "disaster_flood"
        sensor = SENSOR_S1
        radius_m = 6000 if place_name == "Celje" else 5000
        resolution_m = 20
        expl_sl = (
            f"Zaznana analiza poplav za območje {place_name} (datum dogodka {event_d.isoformat()}). "
            f"Izbran je radarski senzor Sentinel-1 SAR RTC (prodira skozi oblake) za ločitev stalnih voda od novo poplavljenih površin."
        )
        expl_en = (
            f"Detected flood assessment for {place_name} (event date {event_d.isoformat()}). "
            f"Selected all-weather Sentinel-1 SAR RTC radar to separate permanent water from newly flooded land."
        )
    elif is_fire:
        task_type = "disaster_fire"
        sensor = SENSOR_LANDSAT if event_d < date(2016, 1, 1) else SENSOR_S2
        radius_m = 6000 if place_name == "Kras" else 5000
        resolution_m = 30 if sensor == SENSOR_LANDSAT else 20
        expl_sl = (
            f"Zaznana analiza požarišča za območje {place_name} ({event_d.isoformat()}). "
            f"Izračun razlike dNBR (NBR_pred − NBR_po) in klasifikacija po USGS lestvici stopnje poškodovanosti."
        )
        expl_en = (
            f"Detected wildfire burn severity analysis for {place_name} ({event_d.isoformat()}). "
            f"Computing cloud-masked dNBR (NBR_pre − NBR_post) classified by official USGS Fire Severity levels."
        )
    elif is_forest:
        task_type = "disaster_forest"
        sensor = SENSOR_LANDSAT if event_d < date(2016, 1, 1) else SENSOR_S2
        radius_m = 5000
        resolution_m = 30 if sensor == SENSOR_LANDSAT else 20
        expl_sl = (
            f"Zaznana analiza izgube vegetacije / gozda za območje {place_name} ({event_d.isoformat()}). "
            f"Izračun ΔNDVI (NDVI_po − NDVI_pred) za kartiranje vetroloma, poseka ali suše."
        )
        expl_en = (
            f"Detected forest/vegetation disturbance analysis for {place_name} ({event_d.isoformat()}). "
            f"Computing cloud-masked ΔNDVI (NDVI_post − NDVI_pre) for windthrow, clearcut, or drought mapping."
        )
    elif is_ts:
        task_type = "time_series"
        sensor = SENSOR_LANDSAT if start_d < date(2016, 1, 1) else (SENSOR_S1 if index_name == "VV" else SENSOR_S2)
        radius_m = 3500
        resolution_m = 30 if sensor == SENSOR_LANDSAT else 20
        expl_sl = (
            f"Zaznana večletna časovna vrsta indeksa {index_name} za {place_name} ({start_d.isoformat()} – {end_d.isoformat()}). "
            f"Senzor: {sensor}, vzorčenje: {interval} (z izločanjem delnih posnetkov < 90 % AOI)."
        )
        expl_en = (
            f"Detected multi-temporal {index_name} time series for {place_name} ({start_d.isoformat()} – {end_d.isoformat()}). "
            f"Sensor: {sensor}, sampling: {interval} (excluding partial swaths < 90% AOI)."
        )
    elif any(k in norm_q for k in ("rgb", "naravn", "true color", "slik", "posnetek")) and not any(
        k in norm_q for k in ("ndvi", "ndwi", "mndwi", "ndbi", "nbr", "indeks", "vegetacij")
    ):
        task_type = "scene_rgb"
        sensor = SENSOR_LANDSAT if start_d < date(2016, 1, 1) else SENSOR_S2
        radius_m = 3500
        resolution_m = 30 if sensor == SENSOR_LANDSAT else 10
        expl_sl = (
            f"Prenos najčistejšega satelitskega posnetka v naravnih barvah (RGB) za {place_name} "
            f"({start_d.isoformat()} – {end_d.isoformat()}) s samodejnim mozaiciranjem ploščic."
        )
        expl_en = (
            f"Streaming the clearest True Color RGB satellite scene for {place_name} "
            f"({start_d.isoformat()} – {end_d.isoformat()}) with automatic same-day tile mosaicking."
        )
    else:
        task_type = "spectral_index"
        sensor = SENSOR_LANDSAT if start_d < date(2016, 1, 1) else (SENSOR_S1 if index_name == "VV" else SENSOR_S2)
        radius_m = 3500
        resolution_m = 30 if sensor == SENSOR_LANDSAT else 10
        expl_sl = (
            f"Izračun spektralnega indeksa {index_name} za območje {place_name} "
            f"({start_d.isoformat()} – {end_d.isoformat()}) z oblačnim maskiranjem in točno metrično statistiko."
        )
        expl_en = (
            f"Computing cloud-masked {index_name} spectral index for {place_name} "
            f"({start_d.isoformat()} – {end_d.isoformat()}) with exact metric area statistics."
        )

    # Explicit radius override if user wrote e.g. "5 km" or "3000 m"
    km_m = re.search(r"\b(\d+(?:[.,]\d+)?)\s*km\b", norm_q)
    m_m = re.search(r"\b(\d{3,5})\s*m\b", norm_q)
    if km_m:
        radius_m = int(round(float(km_m.group(1).replace(",", ".")) * 1000.0))
    elif m_m:
        radius_m = int(m_m.group(1))

    return validate_and_normalize_plan(
        {
            "place_name": place_name,
            "radius_m": radius_m,
            "sensor": sensor,
            "task_type": task_type,
            "index_name": index_name,
            "start_date": start_d.isoformat(),
            "end_date": end_d.isoformat(),
            "event_date": event_d.isoformat(),
            "interval": interval,
            "resolution_m": resolution_m,
            "explanation_sl": expl_sl,
            "explanation_en": expl_en,
        },
        engine="local",
    )


def parse_eo_request(
    query: str,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> dict[str, Any]:
    """Parse a natural-language EO query using the selected Gemini model or smart local fallback.

    If the selected model returns HTTP 404/400 or fails, gracefully cascades through
    ``gemini-3.5-flash`` -> ``gemini-2.5-flash`` -> built-in offline local parser,
    recording which model actually responded.
    """
    resolved_key, _source = get_active_gemini_api_key(api_key)
    req_model = str(model or DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    if req_model.startswith("models/"):
        req_model = req_model[len("models/") :]
    if req_model == "custom" or not req_model:
        req_model = DEFAULT_GEMINI_MODEL

    if resolved_key:
        candidates: list[str] = [req_model]
        for fb in FALLBACK_GEMINI_MODELS:
            if fb not in candidates:
                candidates.append(fb)

        first_status: int | None = None
        for idx, cand_model in enumerate(candidates):
            try:
                plan = parse_eo_request_gemini(query, api_key=resolved_key, model=cand_model)
                if idx == 0:
                    plan["fallback_note_sl"] = ""
                    plan["fallback_note_en"] = ""
                else:
                    code_txt = f" (HTTP {first_status})" if first_status else ""
                    plan["fallback_note_sl"] = (
                        f"Model `{req_model}` ni bil dosegljiv{code_txt}; uspešen samodejni preklop na `{cand_model}`."
                    )
                    plan["fallback_note_en"] = (
                        f"Model `{req_model}` unavailable{code_txt}; automatically responded via `{cand_model}`."
                    )
                return plan
            except Exception as exc:
                if idx == 0:
                    first_status = getattr(getattr(exc, "response", None), "status_code", None)
                continue

        plan_loc = parse_eo_request_local(query)
        code_txt = f" (HTTP {first_status})" if first_status else ""
        plan_loc["fallback_note_sl"] = (
            f"Gemini API (`{req_model}` in rezervni modeli) trenutno ni dosegljiv{code_txt}; "
            f"uporabljen je bil vgrajeni lokalni pametni parser."
        )
        plan_loc["fallback_note_en"] = (
            f"Gemini API (`{req_model}` and fallback models) unavailable{code_txt}; "
            f"used built-in smart local parser."
        )
        return plan_loc

    plan_loc = parse_eo_request_local(query)
    plan_loc["fallback_note_sl"] = ""
    plan_loc["fallback_note_en"] = ""
    return plan_loc


# ---------------------------------------------------------------------------
# Geodetic Execution Engine & Technical Report Summary
# ---------------------------------------------------------------------------
def _default_geocode_place(place_name: str) -> tuple[float, float, str] | None:
    norm_q = _normalize_ascii(place_name)
    for stem, canonical, lat, lon in _SLOVENIAN_PLACE_STEMS:
        if stem in norm_q or _normalize_ascii(canonical) == norm_q:
            return lat, lon, f"{canonical}, Slovenija"
    return None


def build_technical_summary(
    plan: dict[str, Any],
    result: dict[str, Any],
    lang: str = "sl",
) -> str:
    """Build a crisp copy-ready geodetic technical summary for seminar/thesis reports."""
    crs = result.get("crs", CRS_D96TM)
    task = plan["task_type"]
    place = result.get("resolved_place", plan["place_name"])
    sensor = plan["sensor"]
    res_m = plan["resolution_m"]

    lines: list[str] = []
    if lang == "sl":
        lines.append("=== TEHNIČNI POVZETEK GEODETSKE ANALIZE (DALJINSKO ZAZNAVANJE) ===")
        lines.append(f"Območje (AOI): {place} (φ={result['lat']:.5f}°, λ={result['lon']:.5f}°, radij={plan['radius_m']} m)")
        lines.append(f"Koordinatni sistem (CRS): {crs} | Prostorska ločljivost: {res_m} m")
        lines.append(f"Senzor: {sensor} | Analiza: {task} ({plan['index_name']})")
    else:
        lines.append("=== GEODETIC REMOTE SENSING TECHNICAL SUMMARY ===")
        lines.append(f"Area of Interest (AOI): {place} (φ={result['lat']:.5f}°, λ={result['lon']:.5f}°, radius={plan['radius_m']} m)")
        lines.append(f"Coordinate Reference System: {crs} | Spatial Resolution: {res_m} m")
        lines.append(f"Sensor: {sensor} | Analysis: {task} ({plan['index_name']})")

    if task in ("disaster_flood", "disaster_fire", "disaster_forest"):
        dis = result["disaster_result"]
        d_sum = dis["summary"]
        if lang == "sl":
            lines.append(f"Datumi posnetkov: PRED {dis['date_pre']} -> MED/PO {dis['date_event']}")
            lines.append(f"Skupna prizadeta površina: {d_sum['affected_area_ha']:,.2f} ha ({d_sum['affected_area_km2']:.3f} km², {d_sum['affected_pct']:.2f} % AOI)")
            if task == "disaster_flood":
                lines.append(f"Stalna voda (pred poplavo): {d_sum.get('permanent_water_ha', 0.0):,.2f} ha")
        else:
            lines.append(f"Scene Dates: PRE {dis['date_pre']} -> EVENT/POST {dis['date_event']}")
            lines.append(f"Total Affected Area: {d_sum['affected_area_ha']:,.2f} ha ({d_sum['affected_area_km2']:.3f} km², {d_sum['affected_pct']:.2f}% of AOI)")
            if task == "disaster_flood":
                lines.append(f"Permanent Water (pre-flood): {d_sum.get('permanent_water_ha', 0.0):,.2f} ha")
    elif task == "time_series":
        ts_df: pd.DataFrame = result["ts_df"]
        n_obs = len(ts_df)
        d1 = str(ts_df["date_str"].iloc[0]) if n_obs else plan["start_date"]
        d2 = str(ts_df["date_str"].iloc[-1]) if n_obs else plan["end_date"]
        mean_val = float(ts_df["mean"].mean()) if n_obs else float("nan")
        if lang == "sl":
            lines.append(f"Obdobje časovne vrste: {d1} – {d2} ({n_obs} veljavnih datumov, interval: {plan['interval']})")
            lines.append(f"Povprečna vrednost {plan['index_name']}: {mean_val:.3f}")
        else:
            lines.append(f"Time Series Span: {d1} – {d2} ({n_obs} valid dates, interval: {plan['interval']})")
            lines.append(f"Mean {plan['index_name']} value: {mean_val:.3f}")
    elif task == "spectral_index":
        st_d = result["stats"]
        if lang == "sl":
            lines.append(f"Datum posnetka: {result['date_str']} ({result.get('tile', '')})")
            lines.append(
                f"Statistika {plan['index_name']}: povprečje={st_d['mean']:.3f}, mediana={st_d['median']:.3f}, "
                f"min={st_d['min']:.3f}, max={st_d['max']:.3f}, veljavna površina={st_d['valid_area_ha']:,.2f} ha"
            )
        else:
            lines.append(f"Scene Date: {result['date_str']} ({result.get('tile', '')})")
            lines.append(
                f"{plan['index_name']} Statistics: mean={st_d['mean']:.3f}, median={st_d['median']:.3f}, "
                f"min={st_d['min']:.3f}, max={st_d['max']:.3f}, valid area={st_d['valid_area_ha']:,.2f} ha"
            )
    else:
        if lang == "sl":
            lines.append(f"Datum posnetka: {result['date_str']} ({result.get('tile', '')})")
        else:
            lines.append(f"Scene Date: {result['date_str']} ({result.get('tile', '')})")

    lines.append(
        "Vir podatkov: Copernicus / USGS (Microsoft Planetary Computer STAC)"
        if lang == "sl"
        else "Data Source: Copernicus / USGS (Microsoft Planetary Computer STAC)"
    )
    return "\n".join(lines)


def execute_ai_plan(
    plan: dict[str, Any],
    *,
    target_epsg: str = CRS_D96TM,
    geocode_fn: Callable[[str], tuple[float, float, str] | None] | None = None,
    lang: str = "sl",
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Execute a validated AI EO plan strictly in metric projection (EPSG:3794 / EPSG:32633)."""

    def _prog(pct: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, msg)

    _prog(0.08, f"Geokodiram lokacijo: {plan['place_name']}..." if lang == "sl" else f"Geocoding location: {plan['place_name']}...")
    geo = geocode_fn(plan["place_name"]) if geocode_fn is not None else _default_geocode_place(plan["place_name"])
    if geo is None:
        geo = _default_geocode_place(plan["place_name"])
    if geo is None:
        raise RuntimeError(
            f"Lokacije '{plan['place_name']}' ni bilo mogoče najti. Poskusite z bolj znanim krajem ali koordinatami."
            if lang == "sl"
            else f"Could not geocode '{plan['place_name']}'. Try a nearby town name or coordinates."
        )

    lat, lon, resolved_label = float(geo[0]), float(geo[1]), str(geo[2])
    bbox_4326, bbox_3794 = bbox_from_point_wgs84(lat, lon, float(plan["radius_m"]))
    task = plan["task_type"]
    sensor = plan["sensor"]
    res_m = int(plan["resolution_m"])
    epsg_num = str(target_epsg).split(":")[-1]

    base_out: dict[str, Any] = {
        "plan": plan,
        "crs": target_epsg,
        "lat": lat,
        "lon": lon,
        "resolved_place": resolved_label,
        "bbox_4326": bbox_4326,
        "bbox_3794": bbox_3794,
    }

    # --- A. Disaster Assessment Tasks (Flood SAR / Wildfire dNBR / Forest dNDVI) ---
    if task in ("disaster_flood", "disaster_fire", "disaster_forest"):
        dis_map = {
            "disaster_flood": DISASTER_FLOOD,
            "disaster_fire": DISASTER_WILDFIRE,
            "disaster_forest": DISASTER_FOREST,
        }
        ev_date = datetime.strptime(plan["event_date"], "%Y-%m-%d").date()
        dis_out = run_disaster_assessment(
            disaster_type=dis_map[task],
            bbox_4326=bbox_4326,
            event_date=ev_date,
            target_crs=target_epsg,
            resolution=max(res_m, 30 if sensor == SENSOR_LANDSAT else 10),
            search_window_days=60,
            lang=lang,
            progress_cb=lambda f, m: _prog(0.15 + 0.80 * f, m),
        )
        base_out.update(
            {
                "kind": "disaster",
                "disaster_result": dis_out,
                "tif_bytes": dis_out["tif_bytes"],
                "tif_filename": dis_out["tif_filename"],
            }
        )
        base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
        _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
        return base_out

    # --- B. Multi-Temporal Time Series ---
    if task == "time_series":
        _prog(0.18, "Iščem satelitski arhiv za časovno vrsto..." if lang == "sl" else "Searching satellite archive for time series...")
        interval_key = INTERVAL_LABEL_TO_KEY.get(plan["interval"], INTERVAL_YEARLY)
        max_cc = 100.0 if sensor == SENSOR_S1 else 25.0
        items, df = search_scenes(
            sensor=sensor,
            bbox_4326=bbox_4326,
            start_date=plan["start_date"],
            end_date=plan["end_date"],
            max_cloud_cover=max_cc,
            exclude_l7_slc_off=(sensor == SENSOR_LANDSAT),
            min_coverage_pct=DEFAULT_MIN_AOI_COVERAGE_PCT,
            interval_mode=interval_key,
        )
        if not items or df.empty:
            raise RuntimeError(
                "Za izbrano obdobje in lokacijo ni bilo najdenih posnetkov za časovno vrsto."
                if lang == "sl"
                else "No scenes found for the requested time series period and location."
            )
        # Cap to 24 dates for crisp responsiveness
        if len(items) > 24:
            pick_idx = np.linspace(0, len(items) - 1, 24, dtype=int)
            items = [items[i] for i in pick_idx]

        def _ts_cb(cur: int, total: int, d_str: str) -> None:
            _prog(0.25 + 0.70 * (cur / max(total, 1)), f"{d_str} ({cur}/{total})")

        ts_df = extract_time_series(
            items=items,
            bbox_4326=bbox_4326,
            index_name=plan["index_name"],
            target_epsg=target_epsg,
            resolution=max(res_m, 30 if sensor == SENSOR_LANDSAT else 10),
            mask_clouds=(sensor != SENSOR_S1),
            progress_callback=_ts_cb,
            sensor=sensor,
        )
        if ts_df.empty:
            raise RuntimeError(
                "Noben posnetek v obdobju ni imel dovolj ne-oblačnih pikslov za časovno vrsto."
                if lang == "sl"
                else "No scenes had enough cloud-free pixels to build the time series."
            )
        csv_bytes = ts_df.to_csv(index=False).encode("utf-8")
        csv_fname = f"AI_TimeSeries_{plan['index_name']}_{plan['place_name']}_{plan['start_date']}_{plan['end_date']}.csv"
        base_out.update(
            {
                "kind": "time_series",
                "ts_df": ts_df,
                "csv_bytes": csv_bytes,
                "csv_filename": csv_fname,
            }
        )
        base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
        _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
        return base_out

    # --- C. Single-Date Spectral Index or RGB Scene ---
    _prog(0.22, "Iščem najčistejši satelitski posnetek..." if lang == "sl" else "Searching for the clearest satellite scene...")
    max_cc = 100.0 if sensor == SENSOR_S1 else 20.0
    items, df = search_scenes(
        sensor=sensor,
        bbox_4326=bbox_4326,
        start_date=plan["start_date"],
        end_date=plan["end_date"],
        max_cloud_cover=max_cc,
        exclude_l7_slc_off=(sensor == SENSOR_LANDSAT),
        min_coverage_pct=DEFAULT_MIN_AOI_COVERAGE_PCT,
        interval_mode=INTERVAL_ALL,
    )
    if not items or df.empty:
        items, df = search_scenes(
            sensor=sensor,
            bbox_4326=bbox_4326,
            start_date=plan["start_date"],
            end_date=plan["end_date"],
            max_cloud_cover=45.0 if sensor != SENSOR_S1 else 100.0,
            exclude_l7_slc_off=(sensor == SENSOR_LANDSAT),
            min_coverage_pct=70.0,
            interval_mode=INTERVAL_ALL,
        )
    if not items or df.empty:
        raise RuntimeError(
            f"Ni najdenega posnetka za {plan['place_name']} v obdobju {plan['start_date']} – {plan['end_date']}."
            if lang == "sl"
            else f"No scene found for {plan['place_name']} between {plan['start_date']} and {plan['end_date']}."
        )

    # Pick clearest scene (lowest cloud cover, then highest AOI coverage)
    best_i = 0
    if sensor != SENSOR_S1:
        cc_vals = df["cloud_cover"].fillna(50.0).to_numpy()
        best_i = int(np.argmin(cc_vals))
    best_group = items[best_i]
    best_row = df.iloc[best_i]
    date_str = str(best_row["date_str"])
    tile_str = str(best_row.get("mgrs_tile", ""))

    if task == "spectral_index":
        _prog(0.55, f"Prenašam pasove in računam {plan['index_name']} ({date_str})..." if lang == "sl" else f"Streaming bands and computing {plan['index_name']} ({date_str})...")
        if sensor == SENSOR_S1:
            ds = load_scene(
                sensor=sensor,
                item=best_group,
                bbox_4326=bbox_4326,
                bands=["vv"],
                target_epsg=target_epsg,
                resolution=res_m,
            )
            idx_da = ds["VV"]
        else:
            req_bands = get_required_bands_for_sensor(
                sensor=sensor,
                metric_name=plan["index_name"],
                include_cloud_mask=True,
            )
            ds = load_scene(
                sensor=sensor,
                item=best_group,
                bbox_4326=bbox_4326,
                bands=req_bands,
                target_epsg=target_epsg,
                resolution=res_m,
            )
            ds_masked, _ = apply_cloud_mask(ds, sensor=sensor)
            idx_da = compute_index(ds_masked, plan["index_name"])
        stats = compute_index_stats(idx_da, resolution_m=float(res_m))
        tif_bytes = export_to_geotiff_bytes(idx_da, target_epsg=target_epsg)
        tif_fname = f"AI_{plan['index_name']}_{plan['place_name']}_{date_str}_EPSG{epsg_num}_{res_m}m.tif"
        gc.collect()
        base_out.update(
            {
                "kind": "spectral_index",
                "index_da": idx_da,
                "stats": stats,
                "date_str": date_str,
                "tile": tile_str,
                "tif_bytes": tif_bytes,
                "tif_filename": tif_fname,
            }
        )
        base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
        _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
        return base_out

    # Scene RGB
    _prog(0.55, f"Prenašam RGB posnetek ({date_str})..." if lang == "sl" else f"Streaming RGB scene ({date_str})...")
    rgb_bands = ["vv", "vh"] if sensor == SENSOR_S1 else ["B04", "B03", "B02"]
    rgb_vars = ["VV", "VH"] if sensor == SENSOR_S1 else ["B04", "B03", "B02"]
    ds_rgb = load_scene(
        sensor=sensor,
        item=best_group,
        bbox_4326=bbox_4326,
        bands=rgb_bands,
        target_epsg=target_epsg,
        resolution=res_m,
    )
    tif_bytes = export_to_geotiff_bytes(ds_rgb[rgb_vars], target_epsg=target_epsg)
    tif_fname = f"AI_RGB_{plan['place_name']}_{date_str}_EPSG{epsg_num}_{res_m}m.tif"
    gc.collect()
    base_out.update(
        {
            "kind": "scene_rgb",
            "scene_ds": ds_rgb,
            "bands": rgb_vars,
            "date_str": date_str,
            "tile": tile_str,
            "tif_bytes": tif_bytes,
            "tif_filename": tif_fname,
        }
    )
    base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
    _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
    return base_out
