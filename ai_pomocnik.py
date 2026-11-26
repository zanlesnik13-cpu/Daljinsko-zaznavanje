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
import math
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
    compute_snow_glacier_stats,
    get_required_bands_for_sensor,
)
from casovne_vrste import (
    align_and_compute_delta,
    build_multidate_visual_frames,
    compile_timelapse_gif_fps,
    compute_change_summary,
    compute_cloud_free_composite,
    compute_dynamic_frame_delta,
    extract_time_series,
    render_multi_date_comparison,
)
from geodezija_crs import (
    CRS_D96TM,
    bbox_from_point_wgs84,
    export_to_geotiff_bytes,
    get_auto_utm_epsg,
    get_ram_limits,
    get_safe_processing_resolution,
    is_in_slovenia,
    is_low_ram_cloud,
)
from gnss_plan import run_gnss_mission_plan
from kartografija import DEFAULT_LAYOUT_CONFIG, LAYOUT_PRESETS, merge_layout_config
from map_overlay import raster_to_map_overlay_png
from nesrece import (
    DISASTER_FLOOD,
    DISASTER_FOREST,
    DISASTER_WILDFIRE,
    build_slippy_rgba_map,
    run_disaster_assessment,
)
from relief import (
    build_preset_transect_endpoints,
    compute_terrain_derivatives,
    extract_transect_profile,
    load_copernicus_dem,
)
from stac_loader import (
    DEFAULT_MIN_AOI_COVERAGE_PCT,
    HISTORICAL_MIN_DATE,
    INTERVAL_14D,
    INTERVAL_ALL,
    INTERVAL_MONTHLY,
    INTERVAL_SEASONAL,
    INTERVAL_YEARLY,
    MODIS_SENSORS,
    SEASON_GLACIER_ABLATION,
    SEASON_SUMMER,
    SENSOR_LANDSAT,
    SENSOR_MODIS_LST,
    SENSOR_MODIS_SR,
    SENSOR_S1,
    SENSOR_S2,
    assess_scene_analysis_reliability,
    get_default_reliable_t1_index,
    load_scene,
    search_scenes,
)

from gee_engine import (
    GEE_SUPPORTED_INDICES,
    gee_run_temporal_analysis,
    is_gee_ready,
)

# ---------------------------------------------------------------------------
# Constants, Schema Allowed Values & Quick Example Prompts
# ---------------------------------------------------------------------------
DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"
FALLBACK_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
)
GEMINI_MODELS_LIST_URL: str = "https://generativelanguage.googleapis.com/v1beta/models"

GEMINI_MODEL_OPTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "gemini-3.8-flash",
        "label_sl": "3.8 Flash",
        "label_en": "3.8 Flash",
    },
    {
        "id": "gemini-3.6-flash",
        "label_sl": "3.6 Flash",
        "label_en": "3.6 Flash",
    },
    {
        "id": "gemini-3.5-flash",
        "label_sl": "3.5 Flash",
        "label_en": "3.5 Flash",
    },
    {
        "id": "gemini-4-hydrogen",
        "label_sl": "4 Hydrogen",
        "label_en": "4 Hydrogen",
    },
    {
        "id": "gemini-2.5-flash",
        "label_sl": "2.5 Flash",
        "label_en": "2.5 Flash",
    },
    {
        "id": "custom",
        "label_sl": "Po meri",
        "label_en": "Custom",
    },
)


def resolve_gemini_model_id(choice: str | None, custom_id: str | None = None) -> str:
    """Resolve a preset selector value, live-discovered model ID, or custom model ID to a clean Gemini model name."""
    raw = str(choice or DEFAULT_GEMINI_MODEL).strip()
    if raw.lower() == "custom":
        raw = str(custom_id or "").strip()
    if raw.startswith("models/"):
        raw = raw[len("models/") :]
    return raw or DEFAULT_GEMINI_MODEL


def build_gemini_endpoint(model_id: str | None = None) -> str:
    """Build the Gemini REST API generateContent URL for any selected or custom model ID."""
    clean = resolve_gemini_model_id(model_id)
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


def verify_gemini_api_connection(
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout: int = 10,
) -> dict[str, Any]:
    """Verify whether ``api_key`` is valid and whether ``model`` is reachable on that key."""
    clean_model = resolve_gemini_model_id(model)
    live_models = list_available_gemini_models(api_key, timeout=timeout)
    model_found = clean_model in live_models
    fallback = next((m for m in FALLBACK_GEMINI_MODELS if m in live_models), live_models[0] if live_models else DEFAULT_GEMINI_MODEL)
    return {
        "ok": True,
        "model": clean_model,
        "model_found": model_found,
        "total_models": len(live_models),
        "fallback": fallback,
    }


GEMINI_ENDPOINT: str = build_gemini_endpoint(DEFAULT_GEMINI_MODEL)

VALID_SENSORS: tuple[str, ...] = (
    SENSOR_S2,
    SENSOR_LANDSAT,
    SENSOR_S1,
    SENSOR_MODIS_SR,
    SENSOR_MODIS_LST,
)
VALID_TASKS: tuple[str, ...] = (
    "scene_rgb",
    "spectral_index",
    "cloud_free_composite",
    "time_series",
    "time_series_and_comparison",
    "change_comparison",
    "disaster_flood",
    "disaster_fire",
    "disaster_forest",
    "terrain_dem",
    "gnss_plan",
)
VALID_INDICES: tuple[str, ...] = (
    "NDVI",
    "EVI",
    "SAVI",
    "NDMI",
    "NDRE",
    "NDWI",
    "MNDWI",
    "NDSI",
    "NDBI",
    "BSI",
    "NBR",
    "BAI",
    "LST",
    "VV",
)
VALID_RESOLUTIONS: tuple[int, ...] = (10, 20, 30, 60, 100, 250, 500, 1000, 2000)

FUSION_SINGLE: str = "single"
FUSION_TEMPORAL_LANDSAT_S2: str = "temporal_landsat_s2"
FUSION_SYNERGY_SAR_OPTICAL: str = "synergy_sar_optical"
FUSION_SYNERGY_OPTICAL_THERMAL: str = "synergy_optical_thermal"
FUSION_SYNERGY_OPTICAL_DEM: str = "synergy_optical_dem"
VALID_FUSION_MODES: tuple[str, ...] = (
    FUSION_SINGLE,
    FUSION_TEMPORAL_LANDSAT_S2,
    FUSION_SYNERGY_SAR_OPTICAL,
    FUSION_SYNERGY_OPTICAL_THERMAL,
    FUSION_SYNERGY_OPTICAL_DEM,
)
SENSOR_DEM: str = "cop-dem-glo-30"

# Continental / polar regions that automatically route to MODIS 500m-1km + Polar Stereographic / UTM
# tuple: (keywords, canonical_name, lat, lon, default_radius_m, default_res_m, default_index)
_CONTINENTAL_POLAR_REGIONS: tuple[tuple[tuple[str, ...], str, float, float, int, int, str], ...] = (
    (
        ("grenland", "greenland", "jakobshavn", "ilulissat", "sermeq"),
        "Greenland Ice Sheet",
        72.0,
        -40.0,
        550_000,
        1000,
        "NDSI",
    ),
    (
        ("antarktik", "antarctic", "thwaites", "pine island", "larsen"),
        "West Antarctica (Thwaites / Peninsula)",
        -75.0,
        -106.0,
        550_000,
        1000,
        "NDSI",
    ),
    (
        ("amazon", "rondoni", "amazonsk"),
        "Amazon Rainforest (Rondônia, Brazil)",
        -10.5,
        -62.5,
        220_000,
        500,
        "NDVI",
    ),
)


def _detect_continental_polar_region(
    query: str,
    place_name: str = "",
) -> tuple[str, float, float, int, int, str] | None:
    """Detect continental or polar AOI requests (Greenland, Antarctica, Amazon) for MODIS routing."""
    hay = _normalize_ascii(f"{query} {place_name}")
    for kws, cname, lat_v, lon_v, rad_m, res_m, idx_c in _CONTINENTAL_POLAR_REGIONS:
        if any(kw in hay for kw in kws):
            return cname, lat_v, lon_v, rad_m, res_m, idx_c
    return None


# Global landmarks & common typos -> (canonical_name, lat, lon)
# Prevents ambiguous place names (e.g. "Boca Chica" -> Dominican Republic instead of SpaceX Starbase Texas!)
_KNOWN_GLOBAL_LANDMARKS: tuple[tuple[tuple[str, ...], str, float, float], ...] = (
    (
        ("grenland", "greenland", "jakobshavn"),
        "Greenland Ice Sheet",
        72.0,
        -40.0,
    ),
    (
        ("antarktik", "antarctic", "thwaites"),
        "West Antarctica (Thwaites / Peninsula)",
        -75.0,
        -106.0,
    ),
    (
        ("amazon", "rondoni", "amazonsk"),
        "Amazon Rainforest (Rondônia, Brazil)",
        -10.5,
        -62.5,
    ),
    (
        ("starbase", "boca chic", "spacex", "sapacex"),
        "SpaceX Starbase, Boca Chica, Texas, USA",
        25.9972,
        -97.1561,
    ),
    (
        ("pasterz",),
        "Pasterze Glacier, Austria",
        47.0860,
        12.7240,
    ),
    (
        ("giga berlin", "gigafactory berlin", "tesla berlin", "grunheide"),
        "Tesla Gigafactory Berlin-Brandenburg, Germany",
        52.3984,
        13.7945,
    ),
    (
        ("palm jumeirah",),
        "Palm Jumeirah, Dubai, UAE",
        25.1124,
        55.1390,
    ),
    (
        ("panama canal", "panamski prekop"),
        "Panama Canal, Panama",
        9.0800,
        -79.6800,
    ),
    (
        ("suez canal", "sueski prekop"),
        "Suez Canal, Egypt",
        30.5852,
        32.2654,
    ),
    (
        ("aral sea", "aralsko jezero"),
        "Aral Sea, Kazakhstan/Uzbekistan",
        45.0000,
        59.5000,
    ),
    (
        ("cape canaveral", "kennedy space"),
        "Kennedy Space Center, Florida, USA",
        28.5729,
        -80.6490,
    ),
)

_EXPANSION_COMPARISON_CUES: tuple[str, ...] = (
    "siril",
    "siri",
    "razsir",
    "zrasl",
    "rast ",
    "rasti",
    "gradnj",
    "gradil",
    "umik",
    "krci",
    "izsek",
    "sprememb",
    "skozi let",
    "skozi cas",
    "prej in potem",
    "prej/potem",
    "over the years",
    "over time",
    "through the years",
    "expansion",
    "expanded",
    "expand",
    "growth",
    "grew",
    "retreat",
    "shrink",
    "construction",
    "deforest",
    "before and after",
)


def resolve_ai_metric_crs(lat: float, lon: float) -> str:
    """Return exact local metric EPSG string: ``EPSG:3794`` in Slovenia, local auto-UTM globally."""
    if is_in_slovenia(float(lat), float(lon)):
        return CRS_D96TM
    epsg_int, _ = get_auto_utm_epsg(float(lat), float(lon))
    return f"EPSG:{int(epsg_int)}"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometers between two WGS84 points."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(max(0.0, a))))

INTERVAL_LABEL_TO_KEY: dict[str, str] = {
    "Vsi posnetki": INTERVAL_ALL,
    "Vsakih 14 dni": INTERVAL_14D,
    "Mesečno": INTERVAL_MONTHLY,
    "Sezonsko (3 mes.)": INTERVAL_SEASONAL,
    "Letno": INTERVAL_YEARLY,
}
VALID_INTERVAL_LABELS: tuple[str, ...] = tuple(INTERVAL_LABEL_TO_KEY.keys())


def choose_adaptive_interval(
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    user_requested_interval: str | None = None,
) -> str:
    """Choose an optimal temporal sampling interval based on date span unless explicitly requested.

    Rules:
    - If ``user_requested_interval`` is a valid interval label, return it directly.
    - ``span_days > 730`` (> 2 years, e.g. 2018–2026) -> ``"Letno"`` (1 clear scene/year, same season)
    - ``365 < span_days <= 730`` (1–2 years) -> ``"Sezonsko (3 mes.)"``
    - ``60 < span_days <= 365`` (2–12 months) -> ``"Mesečno"``
    - ``span_days <= 60`` (<= 2 months) -> ``"Vsakih 14 dni"``
    """
    if user_requested_interval:
        clean_req = str(user_requested_interval).strip()
        if clean_req in INTERVAL_LABEL_TO_KEY:
            return clean_req

    def _to_d(v: date | datetime | str) -> date:
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v).strip()[:10])

    s_d = _to_d(start_date)
    e_d = _to_d(end_date)
    span_days = abs((e_d - s_d).days)
    if span_days > 730:
        return "Letno"
    if span_days > 365:
        return "Sezonsko (3 mes.)"
    if span_days > 60:
        return "Mesečno"
    return "Vsakih 14 dni"


def get_relevant_indices(primary_index: str, action: str = "time_series") -> list[str]:
    """Return ordered visual layers ``[RGB/SAR, primary, complementary...]`` for any EO topic.

    - Glacier / Snow (``NDSI``) -> ``["RGB", "NDSI", "NDWI"]``
    - Urban / Industrial (``NDBI``) -> ``["RGB", "NDBI", "NDVI"]``
    - Vegetation / Forest (``NDVI``) -> ``["RGB", "NDVI", "NDWI"]``
    - Water / Lakes (``NDWI`` / ``MNDWI``) -> ``["RGB", primary, "NDVI"]``
    - Wildfire (``NBR`` / ``dNBR`` / ``disaster_fire``) -> ``["RGB", "SWIR", "NBR"]``
    - Surface Temperature (``LST``) -> ``["RGB", "LST", "NDVI"]``
    - Flood / SAR (``VV`` / ``SAR`` / ``disaster_flood``) -> ``["SAR", "VV"]``
    """
    idx_u = str(primary_index or "NDVI").strip().upper()
    act_l = str(action or "").strip().lower()

    if act_l == "disaster_flood" or idx_u in ("VV", "VH", "SAR"):
        return ["SAR", "VV"]
    if act_l == "disaster_fire" or idx_u in ("NBR", "DNBR"):
        return ["RGB", "SWIR", "NBR"]
    if idx_u == "NDSI":
        return ["RGB", "NDSI", "NDWI"]
    if idx_u == "NDBI":
        return ["RGB", "NDBI", "NDVI"]
    if idx_u == "NDVI":
        return ["RGB", "NDVI", "NDWI"]
    if idx_u in ("NDWI", "MNDWI"):
        return ["RGB", idx_u, "NDVI"]
    if idx_u == "LST":
        return ["RGB", "LST", "NDVI"]
    return ["RGB", idx_u, "NDVI" if idx_u != "NDVI" else "NDWI"]

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
# Note: More specific town/valley stems MUST appear before broader regional stems (e.g. Črna na Koroškem before Koroška).
_SLOVENIAN_PLACE_STEMS: tuple[tuple[str, str, float, float], ...] = (
    ("ljubljansko barj", "Ljubljansko barje", 46.0050, 14.4600),
    ("ljubljanskem barj", "Ljubljansko barje", 46.0050, 14.4600),
    ("barj", "Ljubljansko barje", 46.0050, 14.4600),
    ("crna na korosk", "Črna na Koroškem", 46.4706, 14.8500),
    ("crni na korosk", "Črna na Koroškem", 46.4706, 14.8500),
    ("slovenj grad", "Slovenj Gradec", 46.5103, 15.0806),
    ("ravn", "Ravne na Koroškem", 46.5431, 14.9692),
    ("mezic", "Mežica", 46.5214, 14.8544),
    ("prevalj", "Prevalje", 46.5469, 14.9208),
    ("dravograd", "Dravograd", 46.5881, 15.0192),
    ("korosk", "Slovenj Gradec", 46.5103, 15.0806),
    ("logarsk", "Logarska dolina", 46.3947, 14.6292),
    ("solcav", "Solčava", 46.4200, 14.6917),
    ("mozir", "Mozirje", 46.3394, 14.9633),
    ("nazarj", "Nazarje", 46.3175, 14.9539),
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
    ("pomur", "Murska Sobota", 46.6625, 16.1664),
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
    ("medvod", "Medvode", 46.1422, 14.4111),
    ("menges", "Mengeš", 46.1669, 14.5750),
    ("komend", "Komenda", 46.2047, 14.5394),
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
    ("lendav", "Lendava", 46.5649, 16.4509),
    ("ormoz", "Ormož", 46.4114, 16.1544),
    ("radovljic", "Radovljica", 46.3444, 14.1744),
    ("vipav", "Vipava", 45.8467, 13.9631),
    # Broad historical Slovenian regions (resolved to central regional hub for geocoding)
    ("gorenjsk", "Kranj", 46.2389, 14.3556),
    ("dolenjsk", "Novo mesto", 45.8040, 15.1689),
    ("stajersk", "Maribor", 46.5547, 15.6459),
    ("primorsk", "Nova Gorica", 45.9560, 13.6484),
    ("notranjsk", "Postojna", 45.7744, 14.2153),
    ("posav", "Krško", 45.9592, 15.4917),
    ("zasav", "Trbovlje", 46.1550, 15.0533),
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


def _normalize_slovenian_place_name(raw_name: str) -> str:
    """Normalize declined Slovenian town/region strings (e.g. 'na koroškem', 'v Celju', 'na Gorenjskem') to nominative names."""
    cleaned = re.sub(
        r"^(?:na|v|pri|okoli|za|in|near|around|over)\s+",
        "",
        str(raw_name or "").strip(),
        flags=re.IGNORECASE,
    ).strip()
    if not cleaned:
        return "Ljubljana"
    norm = _normalize_ascii(cleaned)
    for stem, canonical, _lat, _lon in _SLOVENIAN_PLACE_STEMS:
        if stem in norm:
            return canonical
    return cleaned


def _match_known_global_landmark(text: str) -> tuple[str, float, float] | None:
    """Match famous global landmarks (and common typos) to unambiguous ``(name, lat, lon)``."""
    norm = _normalize_ascii(text)
    if not norm:
        return None
    for stems, canonical, lat, lon in _KNOWN_GLOBAL_LANDMARKS:
        if any(st in norm for st in stems):
            return (canonical, float(lat), float(lon))
    return None


def _coerce_valid_lat_lon(raw_lat: Any, raw_lon: Any) -> tuple[float | None, float | None]:
    """Return ``(lat, lon)`` as floats if both are valid finite WGS84 coordinates, else ``(None, None)``."""
    if raw_lat is None or raw_lon is None:
        return None, None
    try:
        la = float(raw_lat)
        lo = float(raw_lon)
        if np.isfinite(la) and np.isfinite(lo) and -90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0:
            if not (abs(la) < 1e-6 and abs(lo) < 1e-6):
                return round(la, 6), round(lo, 6)
    except (TypeError, ValueError):
        pass
    return None, None


def detect_fusion_mode(
    *,
    task_type: str,
    sensor: str,
    index_name: str,
    start_d: date,
    end_d: date,
    query: str = "",
    requested_mode: str | None = None,
) -> tuple[str, list[str], str, str]:
    """Determine optimal multi-sensor satellite fusion mode, sensor list, and bilingual rationale.

    Returns ``(fusion_mode, sensors, rationale_sl, rationale_en)``.
    """
    norm_q = _normalize_ascii(query)
    req_m = str(requested_mode or "").strip().lower()
    y1, y2 = start_d.year, end_d.year

    # 0. Continental / Polar MODIS Mode (unified 2000–present 8-day composites; never force 30m/10m fusion over 500km!)
    if sensor in MODIS_SENSORS:
        r_sl = (
            "Za analizo območja kontinentalnega ali polarnega merila je bil izbran 8-dnevni mozaik **Terra MODIS (500 m–1 km)** "
            "v ustrezni polarni stereografski (**EPSG:3413 / EPSG:3031**) ali lokalni UTM projekciji, ki omogoča hiter, "
            "brezoblačen pregled celotnega ledenega pokrova ali porečja znotraj 1 GB RAM-a brez popačenj UTM na visokih širinah."
        )
        r_en = (
            "For continental or polar-scale analysis, **Terra MODIS 8-day composites (500 m–1 km)** were selected "
            "in Polar Stereographic (**EPSG:3413 / EPSG:3031**) or local UTM projection, enabling fast, cloud-minimized "
            "coverage of entire ice sheets or river basins within 1 GB RAM without high-latitude UTM distortion."
        )
        return FUSION_SINGLE, [sensor], r_sl, r_en

    # 1. Multi-Decade Temporal Fusion: Landsat (1984–2016) + Sentinel-2 (2017–present)
    is_temporal_task = task_type in ("time_series", "time_series_and_comparison", "change_comparison")
    spans_s2_boundary = start_d < date(2017, 1, 1) and end_d >= date(2017, 1, 1)
    if (
        (is_temporal_task and spans_s2_boundary and index_name not in ("VV", "LST"))
        or req_m == FUSION_TEMPORAL_LANDSAT_S2
    ):
        r_sl = (
            f"Za obdobje {y1}–2016 je bil uporabljen **Landsat Collection 2 L2 (30 m)**, ker Sentinel-2 takrat še ni deloval; "
            f"od leta 2017 do {y2} pa **Sentinel-2 L2A (10–20 m)** zaradi bistveno višje prostorske ločljivosti in ostrejših podrobnosti."
        )
        r_en = (
            f"**Landsat Collection 2 L2 (30 m)** covers the historical {y1}–2016 period before Sentinel-2 existed, "
            f"seamlessly transitioning to **Sentinel-2 L2A (10–20 m)** from 2017 to {y2} for sharper spatial detail."
        )
        return FUSION_TEMPORAL_LANDSAT_S2, [SENSOR_LANDSAT, SENSOR_S2], r_sl, r_en

    # 2. All-Weather Radar + Optical Synergy: Sentinel-1 SAR + Sentinel-2 L2A
    has_radar_cue = any(k in norm_q for k in ("radar", "sar", "sentinel-1", "sentinel 1"))
    has_optical_cue = any(k in norm_q for k in ("optic", "rgb", "sentinel-2", "sentinel 2", "barvn", "ndvi", "ndwi"))
    has_metallic_cue = any(k in norm_q for k in ("starbase", "pristanisc", "luka", "port", "tovarn", "factory", "most", "bridge"))
    if (
        req_m == FUSION_SYNERGY_SAR_OPTICAL
        or task_type == "disaster_flood"
        or (has_radar_cue and has_optical_cue)
        or (has_radar_cue and has_metallic_cue)
    ):
        r_sl = (
            "**Sentinel-1 SAR RTC** (C-pasovni radar) prodira skozi nevihtne oblake in zanesljivo zaznava poplavno vodo "
            "ter kovinske strukture, najbližji jasen posnetek **Sentinel-2 L2A (10 m)** pa na isti karti dodaja barvni optični RGB "
            "in kontekst pokrovnosti tal (NDWI/NDVI)."
        )
        r_en = (
            "**Sentinel-1 SAR RTC** (C-band radar) penetrates storm clouds to reliably detect floodwater and metallic structures, "
            "while the nearest clear **Sentinel-2 L2A (10 m)** scene adds True Color RGB and optical land-cover context (NDWI/NDVI) on the same map."
        )
        return FUSION_SYNERGY_SAR_OPTICAL, [SENSOR_S1, SENSOR_S2], r_sl, r_en

    # 3. High-Res Optical + Thermal Synergy: Sentinel-2 L2A (10 m) + Landsat ST_B10 LST (30 m)
    has_thermal_cue = (
        index_name == "LST"
        or any(k in norm_q for k in ("toplotni otok", "heat island", "uhi", "temperat", "vrocin", "toplot", "thermal", "lst"))
    )
    has_uhi_or_cooling_cue = any(
        k in norm_q
        for k in ("toplotni otok", "heat island", "uhi", "park", "hlajenj", "cooling", "susa", "drought", "drobir", "debris", "gozd", "vegetacij", "mesto")
    )
    if (
        req_m == FUSION_SYNERGY_OPTICAL_THERMAL
        or (has_thermal_cue and has_uhi_or_cooling_cue and end_d >= date(2017, 1, 1))
        or (index_name == "LST" and task_type == "spectral_index" and end_d >= date(2017, 1, 1))
    ):
        r_sl = (
            "**Sentinel-2 L2A (10 m)** zagotavlja oster optični prikaz (RGB in NDVI/NDSI) parkov, gozdov in pozidave, "
            "**Landsat Collection 2 L2 (termalni kanal ST_B10, 30 m)** pa meri dejansko temperaturo površja (LST v °C), "
            "kar omogoča kvantitativno oceno toplotnega otoka in hladilnega učinka vegetacije."
        )
        r_en = (
            "**Sentinel-2 L2A (10 m)** provides sharp optical detail (RGB and NDVI/NDSI) of parks, forests, and built-up surfaces, "
            "while **Landsat Collection 2 L2 (thermal ST_B10, 30 m)** measures Land Surface Temperature (LST in °C) to quantify "
            "Urban Heat Island intensity and vegetation cooling."
        )
        return FUSION_SYNERGY_OPTICAL_THERMAL, [SENSOR_S2, SENSOR_LANDSAT], r_sl, r_en

    # 4. Optical + Relief Synergy: Sentinel-2/Landsat + Copernicus DEM 30 m
    has_dem_synergy_cue = any(
        k in norm_q
        for k in ("snezna meja", "snowline", "plaz", "landslide", "strm", "gorski gozd", "mountain", "visinsk")
    )
    if (
        req_m == FUSION_SYNERGY_OPTICAL_DEM
        or (task_type == "spectral_index" and (index_name == "NDSI" or has_dem_synergy_cue))
    ):
        opt_lbl = "Landsat (30 m)" if sensor == SENSOR_LANDSAT else "Sentinel-2 L2A (10–20 m)"
        r_sl = (
            f"Optični posnetek **{opt_lbl}** je združen s **Copernicus DEM 30 m** za izračun nadmorskih višin, "
            f"naklonov terena in snežne meje [m n.v.] na istem metričnem koordinatnem sistemu."
        )
        r_en = (
            f"Optical **{opt_lbl}** imagery is fused with **Copernicus DEM 30 m** to quantify elevation, "
            f"terrain slope, and snowline altitude [m a.s.l.] in the same metric projection."
        )
        return FUSION_SYNERGY_OPTICAL_DEM, [sensor, SENSOR_DEM], r_sl, r_en

    # 5. Standard single-sensor mode
    return FUSION_SINGLE, [sensor], "", ""


def validate_and_normalize_plan(
    raw_plan: dict[str, Any],
    engine: str = "local",
    query: str = "",
) -> dict[str, Any]:
    """Validate and enforce strict geodetic/sensor rules on any parsed JSON plan."""
    today = date.today()
    raw_place = str(raw_plan.get("place_name") or raw_plan.get("location") or "Ljubljana").strip()
    cont_hit = _detect_continental_polar_region(query, raw_place)
    lm_hit = _match_known_global_landmark(raw_place)
    if cont_hit is not None:
        place_name = cont_hit[0]
    elif lm_hit is not None:
        place_name = lm_hit[0]
    else:
        place_name = _normalize_slovenian_place_name(raw_place)

    lat_val, lon_val = _coerce_valid_lat_lon(raw_plan.get("lat"), raw_plan.get("lon"))
    if cont_hit is not None and lat_val is None:
        lat_val, lon_val = cont_hit[1], cont_hit[2]
    elif lat_val is None and lm_hit is not None:
        lat_val, lon_val = lm_hit[1], lm_hit[2]

    task_type = str(raw_plan.get("task_type") or "spectral_index").strip().lower()
    if task_type not in VALID_TASKS:
        task_type = "spectral_index"

    include_comparison = bool(raw_plan.get("include_comparison", False)) or (
        task_type == "time_series_and_comparison"
    )
    if task_type == "time_series" and include_comparison:
        task_type = "time_series_and_comparison"

    index_name = str(raw_plan.get("index_name") or (cont_hit[5] if cont_hit else "NDVI")).strip().upper()
    if index_name not in VALID_INDICES:
        index_name = "VV" if task_type == "disaster_flood" else (cont_hit[5] if cont_hit else "NDVI")

    default_start = today - timedelta(days=90)
    start_d = _parse_iso_date(raw_plan.get("start_date"), default_start)
    end_d = _parse_iso_date(raw_plan.get("end_date"), today)
    if end_d < start_d:
        start_d, end_d = end_d, start_d
    event_d = _parse_iso_date(raw_plan.get("event_date"), end_d)

    # Enforce physically valid sensor rules (including Continental / Polar MODIS!)
    sensor = str(raw_plan.get("sensor") or SENSOR_S2).strip().lower()
    if sensor not in VALID_SENSORS:
        sensor = SENSOR_S2

    if cont_hit is not None or sensor in MODIS_SENSORS:
        sensor = SENSOR_MODIS_LST if index_name == "LST" else SENSOR_MODIS_SR
        if start_d < date(2000, 2, 18):
            start_d = date(2000, 2, 18)
        if cont_hit is not None and not raw_plan.get("index_name"):
            index_name = cont_hit[5]
    elif task_type == "disaster_flood" or index_name == "VV":
        sensor = SENSOR_S1
        index_name = "VV"
    elif index_name == "LST":
        sensor = SENSOR_LANDSAT
    elif task_type in ("disaster_fire", "disaster_forest"):
        if event_d < date(2016, 1, 1):
            sensor = SENSOR_LANDSAT
        elif sensor == SENSOR_S1:
            sensor = SENSOR_S2
        index_name = "NBR" if task_type == "disaster_fire" else "NDVI"
    else:
        if start_d < date(2016, 1, 1):
            sensor = SENSOR_LANDSAT

    is_modis_plan = sensor in MODIS_SENSORS
    try:
        radius_m = int(round(float(raw_plan.get("radius_m", cont_hit[3] if cont_hit else 4000))))
    except Exception:
        radius_m = cont_hit[3] if cont_hit else 4000
    if is_modis_plan:
        if cont_hit is not None and radius_m < 50_000:
            radius_m = cont_hit[3]
        radius_m = int(np.clip(radius_m, 25_000, 1_200_000))
    else:
        radius_m = int(np.clip(radius_m, 500, 12_000))

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

    # Normalize spatial resolution
    default_res_for_sensor = (
        1000
        if sensor == SENSOR_MODIS_LST
        else (
            (cont_hit[4] if cont_hit else 500)
            if sensor == SENSOR_MODIS_SR
            else (30 if sensor == SENSOR_LANDSAT else 10)
        )
    )
    try:
        res_m = int(raw_plan.get("resolution_m", default_res_for_sensor))
    except Exception:
        res_m = default_res_for_sensor
    if is_modis_plan:
        if res_m < 500:
            res_m = default_res_for_sensor
        if sensor == SENSOR_MODIS_LST and res_m < 1000:
            res_m = 1000
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

    target_crs_hint = resolve_ai_metric_crs(lat_val, lon_val) if (lat_val is not None and lon_val is not None) else None

    fusion_mode, sensors_list, rat_sl, rat_en = detect_fusion_mode(
        task_type=task_type,
        sensor=sensor,
        index_name=index_name,
        start_d=start_d,
        end_d=end_d,
        query=query or str(raw_plan.get("original_query") or ""),
        requested_mode=raw_plan.get("fusion_mode"),
    )
    if str(raw_plan.get("fusion_rationale_sl") or "").strip() and fusion_mode != FUSION_SINGLE:
        rat_sl = str(raw_plan["fusion_rationale_sl"]).strip()
    if str(raw_plan.get("fusion_rationale_en") or "").strip() and fusion_mode != FUSION_SINGLE:
        rat_en = str(raw_plan["fusion_rationale_en"]).strip()

    return {
        "place_name": place_name,
        "location": place_name,
        "lat": lat_val,
        "lon": lon_val,
        "target_epsg": target_crs_hint,
        "radius_m": radius_m,
        "sensor": sensor,
        "sensors": sensors_list,
        "fusion_mode": fusion_mode,
        "fusion_rationale": rat_sl,
        "fusion_rationale_sl": rat_sl,
        "fusion_rationale_en": rat_en,
        "task_type": task_type,
        "include_comparison": include_comparison,
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


def _extract_gemini_text(body: dict[str, Any]) -> str:
    """Safely join ALL non-thought text parts returned by Gemini 2.5 / 3.x / 4 REST API."""
    candidates = body.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    non_thought = [
        str(p.get("text", ""))
        for p in parts
        if isinstance(p, dict) and p.get("text") and not p.get("thought")
    ]
    joined = "".join(non_thought).strip()
    if joined:
        return joined
    fallback_all = [
        str(p.get("text", ""))
        for p in parts
        if isinstance(p, dict) and p.get("text")
    ]
    return "".join(fallback_all).strip()


# ---------------------------------------------------------------------------
# Engine 1: Gemini REST API Planner
# ---------------------------------------------------------------------------
_GEMINI_SYSTEM_PROMPT = """You are a global geodetic Earth Observation (remote sensing) planner for Slovenia, Europe, and worldwide locations.
Convert the user's Slovenian or English request into a STRICT JSON object with these exact keys:
{
  "place_name": string (unambiguous full place name including region/state and country, e.g. "SpaceX Starbase, Boca Chica, Texas, USA", "Črna na Koroškem, Slovenia", "Pasterze Glacier, Austria" — NEVER a bare ambiguous town name!),
  "lat": number | null (exact WGS84 latitude whenever you know the landmark/place, e.g. 25.9972 for SpaceX Starbase, 46.0569 for Ljubljana),
  "lon": number | null (exact WGS84 longitude whenever you know the landmark/place, e.g. -97.1561 for SpaceX Starbase, 14.5058 for Ljubljana),
  "radius_m": integer (1000 to 10000, default 4000, use 5000-6000 for floods/wildfires),
  "sensor": "sentinel-2-l2a" | "landsat-c2-l2" | "sentinel-1-rtc",
  "task_type": "scene_rgb" | "spectral_index" | "cloud_free_composite" | "time_series" | "time_series_and_comparison" | "change_comparison" | "disaster_flood" | "disaster_fire" | "disaster_forest" | "terrain_dem" | "gnss_plan",
  "include_comparison": boolean (true for multi-year spatial growth/expansion/retreat/deforestation/construction),
  "index_name": "NDVI" | "NDWI" | "MNDWI" | "NDBI" | "NBR" | "NDSI" | "LST" | "VV",
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
4. For Copernicus DEM / elevation / slope / aspect / hillshade / terrain profile -> task_type="terrain_dem", resolution_m=30.
5. For GNSS mission planning / skyplot / PDOP / GPS Galileo visibility / obstacle mask -> task_type="gnss_plan".
6. For cloud-free median composite / brezoblačni mozaik -> task_type="cloud_free_composite".
7. For 2-date Before/After swipe comparison -> task_type="change_comparison".
8. For LST / surface temperature -> sensor="landsat-c2-l2", index_name="LST", resolution_m=30.
9. For NDSI / snow / glaciers -> index_name="NDSI".
10. Whenever start_date is before 2016-01-01 -> MUST use sensor="landsat-c2-l2" and resolution_m=30.
11. For multi-year spatial growth, factory/urban expansion, glacier retreat, deforestation, or "how it changed over the years" -> task_type="time_series_and_comparison", include_comparison=true, interval="Letno" (use index_name="NDBI" for factories/urban expansion, "NDSI" for glaciers, "NDVI" for vegetation).
12. Always provide accurate numeric "lat" and "lon" for specific landmarks/facilities (e.g. SpaceX Starbase Boca Chica Texas = 25.9972, -97.1561).
13. Known Slovenian events: Koroška/Savinja/Celje/Kamnik/Škofja Loka floods = 2023-08-04, Barje floods = 2023-08-05, Kras wildfire = 2022-07-22.
14. Always normalize declined Slovenian places/regions to nominative geocodable names: "na Koroškem"/"Koroška" -> "Slovenj Gradec" (or "Črna na Koroškem"), "v Celju"/"v Savinjski dolini" -> "Celje", "na Krasu" -> "Kras", "na Gorenjskem" -> "Kranj", "v Prekmurju" -> "Murska Sobota".
Return ONLY valid JSON without markdown code fences."""


def parse_eo_request_gemini(
    query: str,
    api_key: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout: int = 20,
) -> dict[str, Any]:
    """Call Gemini REST API with JSON output mode for ``model`` and validate the resulting plan."""
    clean_model = resolve_gemini_model_id(model)
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
            "maxOutputTokens": 2048,
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
    text = _extract_gemini_text(body)
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
    # 0. Check known global landmarks & common typos first (e.g. SpaceX Starbase Boca Chica Texas)
    lm_hit = _match_known_global_landmark(norm_q)
    if lm_hit is not None:
        return lm_hit[0]

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
        if target_year == 2023 and any(k in norm_q for k in ("poplav", "flood")):
            ev_d = date(2023, 8, 5) if "barj" in norm_q else date(2023, 8, 4)
            return date(2023, 8, 1), date(2023, 8, 31), ev_d, "Vsi posnetki"
        if target_year == 2022 and any(k in norm_q for k in ("pozar", "fire", "kras")):
            return date(2022, 7, 1), date(2022, 7, 31), date(2022, 7, 22), "Vsi posnetki"
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
    lm_hit = _match_known_global_landmark(norm_q)
    place_name = lm_hit[0] if lm_hit is not None else _extract_place_local(query, norm_q)
    lm_lat = lm_hit[1] if lm_hit is not None else None
    lm_lon = lm_hit[2] if lm_hit is not None else None
    start_d, end_d, event_d, interval = _extract_dates_local(norm_q)

    # 1. Detect task_type & index_name
    is_gnss = any(k in norm_q for k in ("gnss", "gps", "galileo", "glonass", "pdop", "hdop", "skyplot", "izmer", "maska ovir", "satelitov"))
    is_dem = any(k in norm_q for k in ("dem", "dmv", "relief", "naklon", "visinsk", "hillshade", "slope", "aspect", "elevation", "terrain", "nadmorsk", "profil terena"))
    is_composite = any(k in norm_q for k in ("kompozit", "mozaik", "brezoblac", "median", "cloud-free", "cloud free", "composite"))
    is_change = any(k in norm_q for k in ("primerjaj", "primerjav", "prej in potem", "prej/potem", "compare", "comparison", "before and after", "swipe", "zaves"))
    is_flood = any(k in norm_q for k in ("poplav", "flood", "inundat", "razlitj"))
    is_fire = any(k in norm_q for k in ("pozar", "wildfire", "fire", "burn", "dnbr", "opozar"))
    is_forest = any(k in norm_q for k in ("vetrolom", "posek", "zled", "susa", "windthrow", "clearcut", "deforest", "dndvi"))
    is_expansion_change = any(k in norm_q for k in _EXPANSION_COMPARISON_CUES)
    is_ts = any(
        k in norm_q
        for k in (
            "casovn",
            "time series",
            "timeseries",
            "trend",
            "skozi cas",
            "skozi let",
            "spreminj",
            "over time",
            "over the years",
            "through the years",
            "letna",
            "year",
            "zgodovin",
            "histor",
        )
    ) or is_expansion_change or ((end_d - start_d).days >= 400 and not (is_flood or is_fire or is_forest or is_change or is_dem or is_gnss))

    # If user asks about multi-year growth/expansion/retreat without explicit years, use a multi-year window
    if (is_expansion_change or "skozi let" in norm_q or "over the years" in norm_q) and (end_d - start_d).days < 365:
        today_d = date.today()
        if any(k in norm_q for k in ("ledenik", "glacier", "pasterz", "aral")):
            start_d = date(1985, 1, 1)
        else:
            start_d = date(2018, 1, 1)
        end_d = today_d
        event_d = today_d
        interval = "Letno"

    # Index detection
    if "ndsi" in norm_q or any(
        k in norm_q
        for k in (
            "sneg",
            "snez",
            "ledenik",
            "leden",
            "ledu",
            "leda",
            "ledom",
            "snow",
            "glacier",
            "ice sheet",
            "ice cap",
            "sea ice",
            "grenland",
            "greenland",
            "antarktik",
            "antarctic",
        )
    ):
        index_name = "NDSI"
    elif "lst" in norm_q or any(k in norm_q for k in ("temperat", "vrocin", "toplot", "thermal", "heat island")):
        index_name = "LST"
    elif "mndwi" in norm_q:
        index_name = "MNDWI"
    elif "ndwi" in norm_q or any(k in norm_q for k in ("vodn", "jezer", "water", "lake", "river", "reka")):
        index_name = "NDWI"
    elif "ndbi" in norm_q or any(
        k in norm_q
        for k in ("pozidav", "urban", "mesto", "city", "built", "tovarn", "factory", "starbase", "gradnj", "gigafactory", "objekt")
    ):
        index_name = "NDBI"
    elif "nbr" in norm_q or is_fire:
        index_name = "NBR"
    elif "vv" in norm_q or "sar" in norm_q or "radar" in norm_q or is_flood:
        index_name = "VV"
    else:
        index_name = "NDVI"

    if is_gnss:
        task_type = "gnss_plan"
        sensor = SENSOR_S2
        radius_m = 2500
        resolution_m = 10
        expl_sl = (
            f"Zaznano načrtovanje GNSS izmere za stojišče {place_name} na datum {event_d.isoformat()} "
            f"(GPS + Galileo + GLONASS, PDOP/HDOP/VDOP in 360° maska ovir iz Copernicus DEM 30 m + Sentinel-2 10 m)."
        )
        expl_en = (
            f"Detected GNSS mission planning for station {place_name} on {event_d.isoformat()} "
            f"(GPS + Galileo + GLONASS, PDOP/HDOP/VDOP, and 360° obstruction mask from Copernicus DEM 30 m + Sentinel-2 10 m)."
        )
    elif is_dem:
        task_type = "terrain_dem"
        sensor = SENSOR_S2
        radius_m = 4000
        resolution_m = 30
        expl_sl = (
            f"Zaznana analiza reliefa Copernicus DEM 30 m za območje {place_name} "
            f"(nadmorska višina, naklon [°], ekspozicija [°], senčenje ter višinski profil)."
        )
        expl_en = (
            f"Detected Copernicus DEM 30 m terrain analysis for {place_name} "
            f"(elevation, slope [°], aspect [°], hillshade, and elevation profile)."
        )
    elif is_flood:
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
    elif is_composite:
        task_type = "cloud_free_composite"
        sensor = SENSOR_LANDSAT if start_d < date(2016, 1, 1) else SENSOR_S2
        radius_m = 4000
        resolution_m = 30 if sensor == SENSOR_LANDSAT else 10
        expl_sl = (
            f"Izdelava brezoblačnega medianega mozaika ({index_name} / RGB) za območje {place_name} "
            f"v obdobju {start_d.isoformat()} – {end_d.isoformat()}."
        )
        expl_en = (
            f"Building a cloud-free temporal median composite ({index_name} / RGB) for {place_name} "
            f"over {start_d.isoformat()} – {end_d.isoformat()}."
        )
    elif is_change:
        task_type = "change_comparison"
        sensor = SENSOR_LANDSAT if start_d < date(2016, 1, 1) else (SENSOR_S1 if index_name == "VV" else SENSOR_S2)
        radius_m = 3500
        resolution_m = 30 if sensor == SENSOR_LANDSAT else 10
        expl_sl = (
            f"Interaktivna primerjava Prej / Potem ({index_name}) z zoomabilno zaveso in Δ statistiko "
            f"za {place_name} ({start_d.isoformat()} – {end_d.isoformat()})."
        )
        expl_en = (
            f"Interactive Before / After comparison ({index_name}) with zoomable swipe map and Δ statistics "
            f"for {place_name} ({start_d.isoformat()} – {end_d.isoformat()})."
        )
    elif is_ts:
        task_type = "time_series_and_comparison" if is_expansion_change else "time_series"
        sensor = SENSOR_LANDSAT if start_d < date(2016, 1, 1) else (SENSOR_S1 if index_name == "VV" else SENSOR_S2)
        radius_m = 3500
        resolution_m = 30 if sensor == SENSOR_LANDSAT else 10
        expl_sl = (
            f"Zaznana večletna časovna vrsta ({index_name}) + interaktivna vizualna primerjava Prej / Potem (RGB in {index_name}) "
            f"za {place_name} ({start_d.isoformat()} – {end_d.isoformat()}). Senzor: {sensor}, vzorčenje: {interval}."
            if is_expansion_change
            else f"Zaznana večletna časovna vrsta indeksa {index_name} za {place_name} ({start_d.isoformat()} – {end_d.isoformat()}). "
            f"Senzor: {sensor}, vzorčenje: {interval} (z izločanjem delnih posnetkov < 90 % AOI)."
        )
        expl_en = (
            f"Detected multi-year {index_name} time series + interactive Before / After visual comparison (RGB & {index_name}) "
            f"for {place_name} ({start_d.isoformat()} – {end_d.isoformat()}). Sensor: {sensor}, sampling: {interval}."
            if is_expansion_change
            else f"Detected multi-temporal {index_name} time series for {place_name} ({start_d.isoformat()} – {end_d.isoformat()}). "
            f"Sensor: {sensor}, sampling: {interval} (excluding partial swaths < 90% AOI)."
        )
    elif any(k in norm_q for k in ("rgb", "naravn", "true color", "slik", "posnetek")) and not any(
        k in norm_q for k in ("ndvi", "ndwi", "mndwi", "ndbi", "nbr", "ndsi", "lst", "indeks", "vegetacij", "sneg", "temperat")
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
        sensor = SENSOR_LANDSAT if (start_d < date(2016, 1, 1) or index_name == "LST") else (SENSOR_S1 if index_name == "VV" else SENSOR_S2)
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
            "lat": lm_lat,
            "lon": lm_lon,
            "radius_m": radius_m,
            "sensor": sensor,
            "task_type": task_type,
            "include_comparison": bool(task_type == "time_series_and_comparison" or (task_type == "time_series" and is_expansion_change)),
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
        query=query,
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
    elif task in ("time_series", "time_series_and_comparison"):
        ts_df: pd.DataFrame = result["ts_df"]
        n_obs = len(ts_df)
        d1 = str(ts_df["date_str"].iloc[0]) if n_obs else plan["start_date"]
        d2 = str(ts_df["date_str"].iloc[-1]) if n_obs else plan["end_date"]
        mean_val = float(ts_df["mean"].mean()) if n_obs else float("nan")
        if lang == "sl":
            lines.append(f"Obdobje časovne vrste: {d1} – {d2} ({n_obs} veljavnih datumov, interval: {plan['interval']})")
            lines.append(f"Povprečna vrednost {plan['index_name']}: {mean_val:.3f}")
            if result.get("engine") == "gee":
                lines.append(f"Pogon obdelave: Google Earth Engine (strežniška obdelava, projekt: {result.get('gee_project_id') or 'GEE'})")
                cs = result.get("change_summary") or {}
                if cs.get("decrease_area_km2") is not None:
                    lines.append(
                        f"GEE prostorska sprememba ({cs.get('t1_date')} -> {cs.get('t2_date')}): "
                        f"upad {cs['decrease_area_km2']:,.2f} km² ({cs['decrease_area_ha']:,.0f} ha, {cs['decrease_pct']:.1f} % AOI), "
                        f"porast {cs['increase_area_km2']:,.2f} km² ({cs['increase_pct']:.1f} % AOI)"
                    )
            elif result.get("gee_fallback_note"):
                lines.append(str(result["gee_fallback_note"]))
        else:
            lines.append(f"Time Series Span: {d1} – {d2} ({n_obs} valid dates, interval: {plan['interval']})")
            lines.append(f"Mean {plan['index_name']} value: {mean_val:.3f}")
            if result.get("engine") == "gee":
                lines.append(f"Processing Engine: Google Earth Engine (server-side processing, project: {result.get('gee_project_id') or 'GEE'})")
                cs = result.get("change_summary") or {}
                if cs.get("decrease_area_km2") is not None:
                    lines.append(
                        f"GEE spatial change ({cs.get('t1_date')} -> {cs.get('t2_date')}): "
                        f"decrease {cs['decrease_area_km2']:,.2f} km² ({cs['decrease_area_ha']:,.0f} ha, {cs['decrease_pct']:.1f}% of AOI), "
                        f"increase {cs['increase_area_km2']:,.2f} km² ({cs['increase_pct']:.1f}% of AOI)"
                    )
            elif result.get("gee_fallback_note"):
                lines.append(str(result["gee_fallback_note"]))
    elif task == "spectral_index":
        st_d = result["stats"]
        if lang == "sl":
            lines.append(f"Datum posnetka: {result['date_str']} ({result.get('tile', '')})")
            lines.append(
                f"Statistika {plan['index_name']}: povprečje={st_d['mean']:.3f}, mediana={st_d['median']:.3f}, "
                f"min={st_d['min']:.3f}, max={st_d['max']:.3f}, veljavna površina={st_d['valid_area_ha']:,.2f} ha"
            )
            if isinstance(result.get("snow_stats"), dict):
                sn = result["snow_stats"]
                lines.append(
                    f"Sneg/ledeniki (NDSI): {sn.get('snow_area_ha', 0.0):,.2f} ha ({sn.get('snow_pct', 0.0):.1f} %), "
                    f"snežna meja={sn.get('snowline_min_m', '—')} m n. m."
                )
        else:
            lines.append(f"Scene Date: {result['date_str']} ({result.get('tile', '')})")
            lines.append(
                f"{plan['index_name']} Statistics: mean={st_d['mean']:.3f}, median={st_d['median']:.3f}, "
                f"min={st_d['min']:.3f}, max={st_d['max']:.3f}, valid area={st_d['valid_area_ha']:,.2f} ha"
            )
            if isinstance(result.get("snow_stats"), dict):
                sn = result["snow_stats"]
                lines.append(
                    f"Snow/Glacier (NDSI): {sn.get('snow_area_ha', 0.0):,.2f} ha ({sn.get('snow_pct', 0.0):.1f}%), "
                    f"snowline={sn.get('snowline_min_m', '—')} m a.s.l."
                )
    elif task == "cloud_free_composite":
        cm = result.get("composite_meta") or {}
        if lang == "sl":
            lines.append(f"Brezoblačni mediani mozaik: {cm.get('date_start', '')} – {cm.get('date_end', '')} ({cm.get('scenes_used', 0)} posnetkov)")
            lines.append(f"Veljavna površina: {cm.get('valid_area_ha', 0.0):,.2f} ha ({cm.get('valid_pct', 0.0):.1f} %), med. opazovanj/px: {cm.get('median_obs_per_px', 0.0):.1f}")
        else:
            lines.append(f"Cloud-Free Median Composite: {cm.get('date_start', '')} – {cm.get('date_end', '')} ({cm.get('scenes_used', 0)} scenes)")
            lines.append(f"Valid Area: {cm.get('valid_area_ha', 0.0):,.2f} ha ({cm.get('valid_pct', 0.0):.1f}%), median obs/px: {cm.get('median_obs_per_px', 0.0):.1f}")
    elif task == "change_comparison":
        comp = result.get("comparison_result") or {}
        c_dates = comp.get("dates") or []
        csum = result.get("change_summary") or {}
        if lang == "sl":
            lines.append(f"Primerjana datuma: {c_dates[0] if c_dates else ''} -> {c_dates[-1] if c_dates else ''}")
            if csum:
                lines.append(f"Povprečna sprememba Δ{plan['index_name']}: {csum.get('mean', 0.0):+.3f} | Porast: {csum.get('pos_area_ha', 0.0):,.2f} ha | Upad: {csum.get('neg_area_ha', 0.0):,.2f} ha")
        else:
            lines.append(f"Compared Dates: {c_dates[0] if c_dates else ''} -> {c_dates[-1] if c_dates else ''}")
            if csum:
                lines.append(f"Mean Δ{plan['index_name']}: {csum.get('mean', 0.0):+.3f} | Increase: {csum.get('pos_area_ha', 0.0):,.2f} ha | Decrease: {csum.get('neg_area_ha', 0.0):,.2f} ha")
    elif task == "terrain_dem":
        ds = result.get("dem_stats") or {}
        if lang == "sl":
            lines.append(
                f"Copernicus DEM 30 m: višina min={ds.get('min_h', 0):.1f} m, povprečje={ds.get('mean_h', 0):.1f} m, max={ds.get('max_h', 0):.1f} m (razpon {ds.get('relief_diff_m', 0):.1f} m)"
            )
            lines.append(
                f"Naklon terena: povprečje={ds.get('mean_slope_deg', 0):.2f}°, max={ds.get('max_slope_deg', 0):.2f}° | Strmo >20°: {ds.get('steep_20_ha', 0):,.2f} ha ({ds.get('steep_20_pct', 0):.1f} %)"
            )
        else:
            lines.append(
                f"Copernicus DEM 30 m: elevation min={ds.get('min_h', 0):.1f} m, mean={ds.get('mean_h', 0):.1f} m, max={ds.get('max_h', 0):.1f} m (relief {ds.get('relief_diff_m', 0):.1f} m)"
            )
            lines.append(
                f"Terrain Slope: mean={ds.get('mean_slope_deg', 0):.2f}°, max={ds.get('max_slope_deg', 0):.2f}° | Steep >20°: {ds.get('steep_20_ha', 0):,.2f} ha ({ds.get('steep_20_pct', 0):.1f}%)"
            )
    elif task == "gnss_plan":
        gs = (result.get("gnss_result") or {}).get("summary") or {}
        wins = ", ".join(gs.get("optimal_windows") or []) or "—"
        if lang == "sl":
            lines.append(f"Datum načrtovanja GNSS: {plan['event_date']} (GPS + Galileo + GLONASS)")
            lines.append(
                f"Povprečno vidni sateliti (z ovirami / odprto): {gs.get('avg_vis_real', 0):.1f} / {gs.get('avg_vis_open', 0):.1f} | "
                f"Min PDOP: {gs.get('min_pdop_real', 0):.2f} | Zastiranje neba: {gs.get('sky_blocked_pct', 0):.1f} %"
            )
            lines.append(f"Optimalna okna izmere (PDOP ≤ 2,5): {wins}")
        else:
            lines.append(f"GNSS Planning Date: {plan['event_date']} (GPS + Galileo + GLONASS)")
            lines.append(
                f"Avg Visible Satellites (Real / Open Sky): {gs.get('avg_vis_real', 0):.1f} / {gs.get('avg_vis_open', 0):.1f} | "
                f"Min PDOP: {gs.get('min_pdop_real', 0):.2f} | Sky Blocked: {gs.get('sky_blocked_pct', 0):.1f}%"
            )
            lines.append(f"Optimal Measurement Windows (PDOP ≤ 2.5): {wins}")
    else:
        if lang == "sl":
            lines.append(f"Datum posnetka: {result.get('date_str', '')} ({result.get('tile', '')})")
        else:
            lines.append(f"Scene Date: {result.get('date_str', '')} ({result.get('tile', '')})")

    lines.append(
        "Vir podatkov: Copernicus / USGS (Microsoft Planetary Computer STAC)"
        if lang == "sl"
        else "Data Source: Copernicus / USGS (Microsoft Planetary Computer STAC)"
    )
    return "\n".join(lines)


def _sfloat(val: Any, default: float = 0.0) -> float:
    """Safely convert any value (including ``None`` or ``NaN``) to a finite ``float``."""
    if val is None:
        return float(default)
    try:
        out = float(val)
        return out if np.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _sint(val: Any, default: int = 0) -> int:
    """Safely convert any value (including ``None``) to an ``int``."""
    if val is None:
        return int(default)
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return int(default)


def build_analysis_context_dict(
    plan: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Extract a clean, JSON-serializable structured summary of any executed EO/geodetic analysis for chat memory."""
    kind = str(result.get("kind", ""))
    ctx: dict[str, Any] = {
        "kind": kind,
        "place_name": str(plan.get("place_name", "")),
        "resolved_place": str(result.get("resolved_place", plan.get("place_name", ""))),
        "lat": round(_sfloat(result.get("lat"), 0.0), 5),
        "lon": round(_sfloat(result.get("lon"), 0.0), 5),
        "radius_m": _sint(plan.get("radius_m"), 4000),
        "crs": str(result.get("crs", CRS_D96TM)),
        "sensor": str(plan.get("sensor", SENSOR_S2)),
        "sensors": list(plan.get("sensors") or [plan.get("sensor", SENSOR_S2)]),
        "fusion_mode": str(plan.get("fusion_mode") or FUSION_SINGLE),
        "fusion_rationale_sl": str(plan.get("fusion_rationale_sl") or ""),
        "fusion_rationale_en": str(plan.get("fusion_rationale_en") or ""),
        "resolution_m": _sint(plan.get("resolution_m"), 10),
        "task_type": str(plan.get("task_type", "")),
        "index_name": str(plan.get("index_name", "")),
        "start_date": str(plan.get("start_date", "")),
        "end_date": str(plan.get("end_date", "")),
        "event_date": str(plan.get("event_date", "")),
        "technical_summary": str(result.get("technical_summary", "")),
    }
    if isinstance(result.get("synergy_metrics"), dict) and result["synergy_metrics"]:
        ctx["synergy_metrics"] = dict(result["synergy_metrics"])

    if kind == "disaster" and isinstance(result.get("disaster_result"), dict):
        dis = result["disaster_result"]
        d_sum = dis.get("summary") or {}
        bdf = dis.get("breakdown_df")
        breakdown_rows: list[dict[str, Any]] = []
        if isinstance(bdf, pd.DataFrame) and not bdf.empty:
            for _, row in bdf.iterrows():
                breakdown_rows.append(
                    {
                        "class_label": str(row.get("class_label", "")),
                        "area_ha": round(_sfloat(row.get("area_ha"), 0.0), 2),
                        "area_km2": round(_sfloat(row.get("area_km2"), 0.0), 3),
                        "pct": round(_sfloat(row.get("pct"), 0.0), 2),
                    }
                )
        ctx.update(
            {
                "disaster_type": str(dis.get("disaster_type", "")),
                "date_pre": str(dis.get("date_pre", "")),
                "date_event": str(dis.get("date_event", "")),
                "affected_area_ha": round(_sfloat(d_sum.get("affected_area_ha"), 0.0), 2),
                "affected_area_km2": round(_sfloat(d_sum.get("affected_area_km2"), 0.0), 3),
                "affected_pct": round(_sfloat(d_sum.get("affected_pct"), 0.0), 2),
                "permanent_water_ha": round(_sfloat(d_sum.get("permanent_water_ha"), 0.0), 2),
                "mean_dnbr": d_sum.get("mean_dnbr"),
                "mean_dndvi": d_sum.get("mean_dndvi"),
                "severity_breakdown": breakdown_rows,
            }
        )
    elif kind == "spectral_index" and isinstance(result.get("stats"), dict):
        st_d = result["stats"]
        ctx.update(
            {
                "date_str": str(result.get("date_str", "")),
                "tile": str(result.get("tile", "")),
                "stats": {
                    "min": round(_sfloat(st_d.get("min"), 0.0), 4),
                    "max": round(_sfloat(st_d.get("max"), 0.0), 4),
                    "mean": round(_sfloat(st_d.get("mean"), 0.0), 4),
                    "median": round(_sfloat(st_d.get("median"), 0.0), 4),
                    "std": round(_sfloat(st_d.get("std"), 0.0), 4),
                    "valid_area_ha": round(_sfloat(st_d.get("valid_area_ha"), 0.0), 2),
                    "valid_pct": round(_sfloat(st_d.get("valid_pct"), 0.0), 2),
                },
            }
        )
        if isinstance(result.get("snow_stats"), dict):
            ctx["snow_stats"] = result["snow_stats"]
    elif kind == "scene_rgb":
        ctx.update(
            {
                "date_str": str(result.get("date_str", "")),
                "tile": str(result.get("tile", "")),
            }
        )
    elif kind == "cloud_free_composite":
        ctx["composite_meta"] = dict(result.get("composite_meta") or {})
    elif kind == "change_comparison":
        comp = result.get("comparison_result") or {}
        ctx.update(
            {
                "dates": list(comp.get("dates") or []),
                "layer_mode": str(comp.get("layer_mode", "")),
                "replacements": list(comp.get("replacements") or []),
                "change_summary": dict(result.get("change_summary") or {}),
            }
        )
    elif kind == "terrain_dem":
        ctx["dem_stats"] = dict(result.get("dem_stats") or {})
    elif kind == "gnss_plan":
        g_res = result.get("gnss_result") or {}
        g_sum = dict(g_res.get("summary") or {})
        for k, v in list(g_sum.items()):
            if isinstance(v, float) and not np.isfinite(v):
                g_sum[k] = None
        ctx.update(
            {
                "orbit_source": str(g_res.get("orbit_source", "")),
                "gnss_summary": g_sum,
            }
        )
    elif kind in ("time_series", "time_series_and_comparison") and isinstance(result.get("ts_df"), pd.DataFrame):
        ts_df: pd.DataFrame = result["ts_df"]
        excluded_dates: list[str] = list(result.get("excluded_archive_dates") or [])
        if "analysis_reliable" in ts_df.columns:
            rel_mask = ts_df["analysis_reliable"].astype(bool)
            for d_ex in ts_df.loc[~rel_mask, "date_str"].astype(str).tolist():
                if d_ex not in excluded_dates:
                    excluded_dates.append(d_ex)
            rel_df = ts_df.loc[rel_mask].reset_index(drop=True)
            if len(rel_df) < 2:
                rel_df = ts_df
        else:
            rel_df = ts_df
        excluded_dates = sorted(set(excluded_dates))
        n_obs = len(rel_df)
        vals = rel_df["mean"].to_numpy(dtype=np.float64) if n_obs else np.array([], dtype=np.float64)
        fin_vals = vals[np.isfinite(vals)]
        slope = None
        if n_obs >= 3 and fin_vals.size >= 3:
            slope = round(float(np.polyfit(np.arange(fin_vals.size, dtype=np.float64), fin_vals, 1)[0]), 5)
        ctx.update(
            {
                "n_dates": int(n_obs),
                "total_visual_dates": int(len(ts_df)),
                "excluded_archive_dates": excluded_dates,
                "date_start": str(rel_df["date_str"].iloc[0]) if n_obs else str(plan.get("start_date", "")),
                "date_end": str(rel_df["date_str"].iloc[-1]) if n_obs else str(plan.get("end_date", "")),
                "interval": str(plan.get("interval", "")),
                "mean": round(float(np.nanmean(fin_vals)), 4) if fin_vals.size else None,
                "min": round(float(np.nanmin(fin_vals)), 4) if fin_vals.size else None,
                "max": round(float(np.nanmax(fin_vals)), 4) if fin_vals.size else None,
                "first_val": round(float(fin_vals[0]), 4) if fin_vals.size else None,
                "last_val": round(float(fin_vals[-1]), 4) if fin_vals.size else None,
                "trend_slope_per_obs": slope,
                "has_visual_comparison": bool(result.get("comparison_rgb") or result.get("comparison_index")),
            }
        )
        cs_ts = result.get("change_summary") or {}
        if cs_ts:
            ctx["delta_mean"] = round(_sfloat(cs_ts.get("mean"), 0.0), 4)
            ctx["pos_area_ha"] = round(_sfloat(cs_ts.get("pos_area_ha"), 0.0), 2)
            ctx["neg_area_ha"] = round(_sfloat(cs_ts.get("neg_area_ha"), 0.0), 2)
        g_stats = result.get("glacier_change_stats")
        if isinstance(g_stats, dict) and g_stats:
            ctx["glacier_change_stats"] = dict(g_stats)
    return ctx


def resolve_plan_coordinates(
    plan: dict[str, Any],
    *,
    geocode_fn: Callable[[str], tuple[float, float, str] | None] | None = None,
    lang: str = "sl",
) -> tuple[float, float, str]:
    """Resolve unambiguous ``(lat, lon, resolved_label)`` for an AI plan.

    If the plan already carries valid numeric ``(lat, lon)`` from Gemini or the
    global landmark table, Nominatim is checked only as a label/refinement helper
    and is strictly rejected if it fails or lands more than 30 km away from the
    AI's known landmark coordinates (preventing e.g. 'Boca Chica' from hijacking
    SpaceX Starbase Texas to the Dominican Republic).
    """
    place_name = str(plan.get("place_name") or plan.get("location") or "Ljubljana").strip()
    ai_lat, ai_lon = _coerce_valid_lat_lon(plan.get("lat"), plan.get("lon"))

    # Also check known global landmarks if AI didn't supply lat/lon
    if ai_lat is None:
        lm_hit = _match_known_global_landmark(place_name)
        if lm_hit is not None:
            place_name, ai_lat, ai_lon = lm_hit[0], lm_hit[1], lm_hit[2]

    geo = None
    try:
        geo = geocode_fn(place_name) if geocode_fn is not None else _default_geocode_place(place_name)
    except Exception:
        geo = None
    if geo is None:
        try:
            geo = _default_geocode_place(place_name)
        except Exception:
            geo = None

    if ai_lat is not None and ai_lon is not None:
        if geo is not None:
            g_lat, g_lon, g_lbl = float(geo[0]), float(geo[1]), str(geo[2])
            if _haversine_km(ai_lat, ai_lon, g_lat, g_lon) <= 30.0:
                return g_lat, g_lon, g_lbl or place_name
        return float(ai_lat), float(ai_lon), place_name

    if geo is not None:
        return float(geo[0]), float(geo[1]), str(geo[2])

    raise RuntimeError(
        f"Lokacije '{place_name}' ni bilo mogoče najti. Poskusite z bolj znanim krajem ali koordinatami."
        if lang == "sl"
        else f"Could not geocode '{place_name}'. Try a nearby town name or coordinates."
    )


def _resolve_ai_target_season(plan: dict[str, Any]) -> str:
    """Select late-summer glacier ablation window (Jul 15 – Sep 15) for glacier/NDSI analyses, else Summer."""
    idx_u = str(plan.get("index_name") or "").strip().upper()
    blob = _normalize_ascii(
        " ".join(
            str(plan.get(k) or "")
            for k in ("place_name", "location", "explanation_sl", "explanation_en")
        )
    )
    if idx_u == "NDSI" or any(
        k in blob for k in ("ledenik", "glacier", "pasterz", "grossglockner", "grosglockner", "triglavski ledenik", "krcenj", "umikanj")
    ):
        return SEASON_GLACIER_ABLATION
    return SEASON_SUMMER


def execute_temporal_fusion(
    plan: dict[str, Any],
    *,
    base_out: dict[str, Any],
    bbox_4326: Sequence[float],
    target_epsg: str,
    lang: str = "sl",
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Engine A — Unified Multi-Decade Temporal Fusion: Landsat C2 L2 (1984–2016) + Sentinel-2 L2A (2017–present).

    Harmonizes historical 30 m Landsat and modern 10–20 m Sentinel-2 into ONE chronological DataFrame,
    ONE RAM-safe visual frame stack (<= 10 frames on 1 GB cloud, <= 14 on High-RAM), ONE cross-resolution
    pixel-aligned Δ change map (``t2_S2 - t1_Landsat``), and ONE multi-decade Timelapse GIF.
    """
    def _prog(pct: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, msg)

    epsg_num = str(target_epsg).split(":")[-1]
    layer_m = str(plan.get("index_name") or "NDVI").upper()
    start_s = str(plan["start_date"])
    end_s = str(plan["end_date"])
    hist_end = "2016-12-31"
    mod_start = "2017-01-01"
    ai_target_season = _resolve_ai_target_season(plan)

    adaptive_interval = choose_adaptive_interval(
        start_s,
        end_s,
        user_requested_interval=(plan.get("interval") if plan.get("interval_explicit") else None),
    )
    interval_key = INTERVAL_LABEL_TO_KEY.get(adaptive_interval, INTERVAL_YEARLY)
    relevant_layers = get_relevant_indices(layer_m, str(plan.get("task_type") or "time_series"))
    extra_idx_list = [c for c in relevant_layers if c not in ("RGB", "SAR", layer_m)]

    low_ram = is_low_ram_cloud()
    max_ls_frames = 5 if low_ram else 7
    max_s2_frames = 5 if low_ram else 7
    vis_res_s2 = max(int(plan.get("resolution_m", 10)), 20 if low_ram else 10)

    # 1. Search historical Landsat (start_date .. 2016-12-31)
    _prog(
        0.14,
        f"Iščem zgodovinske Landsat posnetke ({start_s[:4]}–2016)..."
        if lang == "sl"
        else f"Searching historical Landsat archive ({start_s[:4]}–2016)...",
    )
    ls_items, ls_df_search = search_scenes(
        sensor=SENSOR_LANDSAT,
        bbox_4326=bbox_4326,
        start_date=start_s,
        end_date=hist_end,
        max_cloud_cover=25.0,
        exclude_l7_slc_off=True,
        min_coverage_pct=DEFAULT_MIN_AOI_COVERAGE_PCT,
        interval_mode=interval_key,
        target_season=ai_target_season,
    )
    if not ls_items or ls_df_search.empty:
        ls_items, ls_df_search = search_scenes(
            sensor=SENSOR_LANDSAT,
            bbox_4326=bbox_4326,
            start_date=start_s,
            end_date=hist_end,
            max_cloud_cover=45.0,
            exclude_l7_slc_off=True,
            min_coverage_pct=65.0,
            interval_mode=interval_key,
            target_season=ai_target_season,
        )
    if len(ls_items) > 12:
        p_idx = np.linspace(0, len(ls_items) - 1, 12, dtype=int)
        ls_items = [ls_items[i] for i in p_idx]
        ls_df_search = ls_df_search.iloc[p_idx].reset_index(drop=True)

    # 2. Search modern Sentinel-2 (2017-01-01 .. end_date)
    _prog(
        0.22,
        f"Iščem sodobne Sentinel-2 posnetke (2017–{end_s[:4]})..."
        if lang == "sl"
        else f"Searching modern Sentinel-2 archive (2017–{end_s[:4]})...",
    )
    s2_items, s2_df_search = search_scenes(
        sensor=SENSOR_S2,
        bbox_4326=bbox_4326,
        start_date=mod_start,
        end_date=end_s,
        max_cloud_cover=25.0,
        exclude_l7_slc_off=False,
        min_coverage_pct=DEFAULT_MIN_AOI_COVERAGE_PCT,
        interval_mode=interval_key,
        target_season=ai_target_season,
    )
    if not s2_items or s2_df_search.empty:
        s2_items, s2_df_search = search_scenes(
            sensor=SENSOR_S2,
            bbox_4326=bbox_4326,
            start_date=mod_start,
            end_date=end_s,
            max_cloud_cover=45.0,
            exclude_l7_slc_off=False,
            min_coverage_pct=65.0,
            interval_mode=interval_key,
            target_season=ai_target_season,
        )
    if len(s2_items) > 12:
        p_idx = np.linspace(0, len(s2_items) - 1, 12, dtype=int)
        s2_items = [s2_items[i] for i in p_idx]
        s2_df_search = s2_df_search.iloc[p_idx].reset_index(drop=True)

    # 3. Extract harmonized time series from both archives
    ts_parts: list[pd.DataFrame] = []
    if ls_items:
        _prog(0.30, "Računam zgodovinsko časovno vrsto (Landsat 30 m)..." if lang == "sl" else "Extracting historical Landsat (30 m) time series...")
        ts_ls = extract_time_series(
            items=ls_items,
            bbox_4326=bbox_4326,
            index_name=layer_m,
            target_epsg=target_epsg,
            resolution=30,
            mask_clouds=True,
            sensor=SENSOR_LANDSAT,
            candidates_by_date=ls_df_search.attrs.get("candidates_by_bucket"),
        )
        if not ts_ls.empty:
            ts_ls = ts_ls.copy()
            ts_ls["sensor_family"] = "Landsat (30 m)"
            ts_ls["resolution_m"] = 30
            ts_parts.append(ts_ls)

    if s2_items:
        _prog(0.46, "Računam sodobno časovno vrsto (Sentinel-2 10–20 m)..." if lang == "sl" else "Extracting modern Sentinel-2 (10–20 m) time series...")
        ts_s2 = extract_time_series(
            items=s2_items,
            bbox_4326=bbox_4326,
            index_name=layer_m,
            target_epsg=target_epsg,
            resolution=vis_res_s2,
            mask_clouds=True,
            sensor=SENSOR_S2,
            candidates_by_date=s2_df_search.attrs.get("candidates_by_bucket"),
        )
        if not ts_s2.empty:
            ts_s2 = ts_s2.copy()
            ts_s2["sensor_family"] = "Sentinel-2 (10–20 m)"
            ts_s2["resolution_m"] = vis_res_s2
            ts_parts.append(ts_s2)

    if not ts_parts:
        raise RuntimeError(
            "Za izbrano obdobje ni bilo najdenih čistih posnetkov Landsat ali Sentinel-2."
            if lang == "sl"
            else "No cloud-free Landsat or Sentinel-2 scenes found for the requested period."
        )

    ts_df = pd.concat(ts_parts, ignore_index=True)
    ts_df = ts_df.sort_values("date_str", ascending=True).reset_index(drop=True)
    csv_bytes = ts_df.to_csv(index=False).encode("utf-8")
    csv_fname = f"AI_Fusion_Landsat_S2_{layer_m}_{plan['place_name']}_{start_s}_{end_s}.csv"

    # 4. Build unified RAM-safe chronological visual frames (Landsat + Sentinel-2)
    frames: list[dict[str, Any]] = []
    da_first = None
    da_last = None

    if ls_items:
        _prog(0.60, "Pripravljam zgodovinske satelitske posnetke (Landsat)..." if lang == "sl" else "Rendering historical Landsat frames...")
        ls_dates = [str(d) for d in ls_df_search["date_str"].tolist()]
        ls_bundle = build_multidate_visual_frames(
            item_groups=ls_items,
            date_labels=ls_dates,
            bbox_4326=bbox_4326,
            index_name=layer_m,
            target_epsg=target_epsg,
            resolution=30,
            sensor=SENSOR_LANDSAT,
            extra_indices=extra_idx_list,
            max_frames=max_ls_frames,
            candidates_by_date=ls_df_search.attrs.get("candidates_by_bucket"),
            auto_cloud_fallback=True,
            min_clear_pct=70.0,
            max_thumb_width=560 if low_ram else 680,
            overlay_max_width=1000 if low_ram else 1300,
            default_fps=1.5,
        )
        frames.extend(ls_bundle.get("frames") or [])
        if ls_bundle.get("da_first") is not None:
            da_first = ls_bundle["da_first"]
        if ls_bundle.get("da_last") is not None:
            da_last = ls_bundle["da_last"]
        gc.collect()

    if s2_items:
        _prog(0.76, "Pripravljam sodobne satelitske posnetke (Sentinel-2)..." if lang == "sl" else "Rendering modern Sentinel-2 frames...")
        s2_dates = [str(d) for d in s2_df_search["date_str"].tolist()]
        s2_bundle = build_multidate_visual_frames(
            item_groups=s2_items,
            date_labels=s2_dates,
            bbox_4326=bbox_4326,
            index_name=layer_m,
            target_epsg=target_epsg,
            resolution=vis_res_s2,
            sensor=SENSOR_S2,
            extra_indices=extra_idx_list,
            max_frames=max_s2_frames,
            candidates_by_date=s2_df_search.attrs.get("candidates_by_bucket"),
            auto_cloud_fallback=True,
            min_clear_pct=75.0,
            max_thumb_width=560 if low_ram else 680,
            overlay_max_width=1000 if low_ram else 1300,
            default_fps=1.5,
        )
        frames.extend(s2_bundle.get("frames") or [])
        if da_first is None and s2_bundle.get("da_first") is not None:
            da_first = s2_bundle["da_first"]
        if s2_bundle.get("da_last") is not None:
            da_last = s2_bundle["da_last"]
        gc.collect()

    frames = sorted(frames, key=lambda fr: str(fr.get("date") or ""))
    default_t1_idx = get_default_reliable_t1_index(frames)
    if len(frames) >= 2 and default_t1_idx >= len(frames) - 1:
        default_t1_idx = 0

    excluded_archive_dates: list[str] = sorted(
        {
            str(f.get("date"))
            for f in frames
            if isinstance(f, dict) and f.get("analysis_reliable") is False and f.get("date")
        }
        | (
            set(ts_df.loc[~ts_df["analysis_reliable"].astype(bool), "date_str"].astype(str).tolist())
            if "analysis_reliable" in ts_df.columns
            else set()
        )
    )

    # 5. Cross-resolution pixel-aligned Δ change map (water-aware for NDSI -> detects terminus-to-lake retreat!)
    diff_da = None
    ch_summary: dict[str, Any] = {}
    delta_uri = None
    delta_bnds = None
    delta_vmin = -0.6
    delta_vmax = 0.6
    delta_cmap = "RdBu"
    glacier_class_uri = None
    glacier_class_bnds = None
    glacier_change_stats = None
    tif_bytes = None
    tif_fname = None

    if len(frames) >= 2:
        _prog(0.88, "Usklajujem rastrski mreži (Landsat 30 m ↔ Sentinel-2) za karto sprememb Δ..." if lang == "sl" else "Aligning Landsat 30 m & Sentinel-2 grids for Δ change map...")
        dyn_out = compute_dynamic_frame_delta(
            frames[default_t1_idx],
            frames[-1],
            target_epsg=target_epsg,
            index_name=layer_m,
            max_width=1000 if low_ram else 1300,
        )
        if isinstance(dyn_out, dict):
            diff_da = dyn_out["diff_da"]
            ch_summary = dyn_out["change_summary"]
            delta_uri = dyn_out["delta_overlay_uri"]
            delta_bnds = dyn_out["delta_overlay_bounds"]
            delta_vmin = dyn_out["delta_vmin"]
            delta_vmax = dyn_out["delta_vmax"]
            delta_cmap = dyn_out["delta_cmap"]
            glacier_class_uri = dyn_out.get("glacier_class_uri")
            glacier_class_bnds = dyn_out.get("glacier_class_bounds")
            glacier_change_stats = dyn_out.get("glacier_change_stats")
        elif da_first is not None and da_last is not None:
            delta_out = align_and_compute_delta(
                da_first,
                da_last,
                target_epsg=target_epsg,
                index_name=layer_m,
                threshold=0.12,
                max_width=1000 if low_ram else 1300,
            )
            diff_da = delta_out["diff_da"]
            ch_summary = delta_out["change_summary"]
            delta_uri = delta_out["delta_overlay_uri"]
            delta_bnds = delta_out["delta_overlay_bounds"]
            delta_vmin = delta_out["delta_vmin"]
            delta_vmax = delta_out["delta_vmax"]
            delta_cmap = delta_out["delta_cmap"]
        if diff_da is not None:
            tif_bytes = export_to_geotiff_bytes(diff_da, target_epsg=target_epsg)
            tif_fname = f"AI_Fusion_Delta_{layer_m}_{plan['place_name']}_{frames[default_t1_idx]['date']}_to_{frames[-1]['date']}_EPSG{epsg_num}.tif"

    # 6. Unified Multi-Decade Timelapse GIFs
    rgb_pngs = [f["rgb_png_bytes"] for f in frames if f.get("rgb_png_bytes")]
    idx_pngs = [f["index_png_bytes"] for f in frames if f.get("index_png_bytes")]
    gif_rgb = compile_timelapse_gif_fps(rgb_pngs, fps=1.5) if len(rgb_pngs) >= 2 else None
    gif_idx = compile_timelapse_gif_fps(idx_pngs, fps=1.5) if len(idx_pngs) >= 2 else None

    n_ls_obs = int((ts_df["sensor_family"] == "Landsat (30 m)").sum()) if "sensor_family" in ts_df.columns else 0
    n_s2_obs = int((ts_df["sensor_family"] == "Sentinel-2 (10–20 m)").sum()) if "sensor_family" in ts_df.columns else 0
    synergy_metrics = {
        "fusion_type": FUSION_TEMPORAL_LANDSAT_S2,
        "landsat_obs": n_ls_obs,
        "sentinel2_obs": n_s2_obs,
        "transition_year": 2017,
        "total_frames": len(frames),
    }
    if isinstance(glacier_change_stats, dict):
        synergy_metrics.update(glacier_change_stats)

    gc.collect()
    base_out.update(
        {
            "kind": "time_series_and_comparison" if len(frames) >= 2 else "time_series",
            "adaptive_interval": adaptive_interval,
            "relevant_layers": relevant_layers,
            "extra_indices": extra_idx_list,
            "ts_df": ts_df,
            "csv_bytes": csv_bytes,
            "csv_filename": csv_fname,
            "frames": frames,
            "default_t1_index": default_t1_idx,
            "excluded_archive_dates": excluded_archive_dates,
            "timelapse_rgb_gif_bytes": gif_rgb,
            "timelapse_index_gif_bytes": gif_idx,
            "comparison_rgb": None,
            "comparison_index": None,
            "comparison_result": None,
            "change_summary": ch_summary,
            "delta_overlay_uri": delta_uri,
            "delta_overlay_bounds": delta_bnds,
            "delta_vmin": delta_vmin,
            "delta_vmax": delta_vmax,
            "delta_cmap": delta_cmap,
            "glacier_class_uri": glacier_class_uri,
            "glacier_class_bounds": glacier_class_bnds,
            "glacier_change_stats": glacier_change_stats,
            "da_last": da_last,
            "index_da": diff_da if diff_da is not None else da_last,
            "tif_bytes": tif_bytes,
            "tif_filename": tif_fname,
            "synergy_metrics": synergy_metrics,
        }
    )
    base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
    base_out["analysis_context"] = build_analysis_context_dict(plan, base_out)
    _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
    return base_out


def _attach_sar_optical_synergy(
    *,
    bbox_4326: Sequence[float],
    ref_date: date,
    target_epsg: str,
    lang: str = "sl",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Engine B1 — Fetch the nearest clear Sentinel-2 L2A optical scene (RGB + NDWI) to pair with Sentinel-1 SAR."""
    extra_layers: list[dict[str, Any]] = []
    syn_metrics: dict[str, Any] = {"fusion_type": FUSION_SYNERGY_SAR_OPTICAL}
    try:
        s_start = (ref_date - timedelta(days=45)).isoformat()
        s_end = (ref_date + timedelta(days=45)).isoformat()
        s2_items, s2_df = search_scenes(
            sensor=SENSOR_S2,
            bbox_4326=bbox_4326,
            start_date=s_start,
            end_date=s_end,
            max_cloud_cover=25.0,
            min_coverage_pct=70.0,
            interval_mode=INTERVAL_ALL,
        )
        if not s2_items or s2_df.empty:
            s2_items, s2_df = search_scenes(
                sensor=SENSOR_S2,
                bbox_4326=bbox_4326,
                start_date=s_start,
                end_date=s_end,
                max_cloud_cover=50.0,
                min_coverage_pct=60.0,
                interval_mode=INTERVAL_ALL,
            )
        if s2_items and not s2_df.empty:
            cc_vals = s2_df["cloud_cover"].fillna(50.0).to_numpy()
            best_i = int(np.argmin(cc_vals))
            s2_date = str(s2_df.iloc[best_i]["date_str"])
            s2_res = 20 if is_low_ram_cloud() else 10
            ds_s2 = load_scene(
                sensor=SENSOR_S2,
                item=s2_items[best_i],
                bbox_4326=bbox_4326,
                bands=["B04", "B03", "B02", "B08", "SCL"],
                target_epsg=target_epsg,
                resolution=s2_res,
            )
            if all(b in ds_s2.data_vars for b in ("B04", "B03", "B02")):
                rgb_u, rgb_b = raster_to_map_overlay_png(
                    ds_s2,
                    src_epsg=target_epsg,
                    is_rgb=True,
                    bands=["B04", "B03", "B02"],
                    use_percentile_stretch=True,
                    max_width=1400,
                )
                extra_layers.append(
                    {
                        "code": "S2_RGB",
                        "name": f"🛰️ Sentinel-2 RGB ({s2_date})",
                        "image": rgb_u,
                        "bounds": rgb_b,
                    }
                )
            ds_m, _ = apply_cloud_mask(ds_s2, sensor=SENSOR_S2)
            ndwi_da = compute_index(ds_m, "NDWI")
            ndwi_spec = SPECTRAL_INDICES["NDWI"]
            ndwi_u, ndwi_b = raster_to_map_overlay_png(
                ndwi_da,
                src_epsg=target_epsg,
                is_rgb=False,
                cmap_name=str(ndwi_spec["cmap"]),
                vmin=float(ndwi_spec["vmin"]),
                vmax=float(ndwi_spec["vmax"]),
                max_width=1400,
            )
            extra_layers.append(
                {
                    "code": "S2_NDWI",
                    "name": f"💧 Sentinel-2 NDWI ({s2_date})",
                    "image": ndwi_u,
                    "bounds": ndwi_b,
                }
            )
            ndwi_vals = np.asarray(ndwi_da.values, dtype=np.float32)
            fin_w = ndwi_vals[np.isfinite(ndwi_vals)]
            syn_metrics["s2_optical_date"] = s2_date
            syn_metrics["s2_resolution_m"] = s2_res
            if fin_w.size > 0:
                syn_metrics["s2_ndwi_mean"] = round(float(np.mean(fin_w)), 4)
            del ds_s2, ds_m, ndwi_da
            gc.collect()
    except Exception:
        pass
    return extra_layers, syn_metrics


def _execute_optical_thermal_synergy(
    plan: dict[str, Any],
    *,
    base_out: dict[str, Any],
    bbox_4326: Sequence[float],
    target_epsg: str,
    lang: str = "sl",
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Engine B2 — Fuse Sentinel-2 L2A (10 m RGB + NDVI) with Landsat C2 L2 Thermal (ST_B10 LST °C, 30 m).

    Computes quantitative Urban Heat Island cooling delta (LST_urban - LST_veg) and Pearson r(NDVI, LST).
    """
    def _prog(pct: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, msg)

    epsg_num = str(target_epsg).split(":")[-1]
    start_s = str(plan["start_date"])
    end_s = str(plan["end_date"])

    # 1. Load clearest Landsat C2 L2 thermal scene (ST_B10 -> LST °C)
    _prog(0.22, "Prenašam termalni posnetek Landsat (ST_B10 → LST °C)..." if lang == "sl" else "Streaming Landsat thermal scene (ST_B10 → LST °C)...")
    ls_items, ls_df = search_scenes(
        sensor=SENSOR_LANDSAT,
        bbox_4326=bbox_4326,
        start_date=start_s,
        end_date=end_s,
        max_cloud_cover=25.0,
        exclude_l7_slc_off=True,
        min_coverage_pct=70.0,
        interval_mode=INTERVAL_ALL,
    )
    if not ls_items or ls_df.empty:
        ls_items, ls_df = search_scenes(
            sensor=SENSOR_LANDSAT,
            bbox_4326=bbox_4326,
            start_date=start_s,
            end_date=end_s,
            max_cloud_cover=50.0,
            exclude_l7_slc_off=True,
            min_coverage_pct=60.0,
            interval_mode=INTERVAL_ALL,
        )
    if not ls_items or ls_df.empty:
        raise RuntimeError(
            f"Ni najdenega termalnega posnetka Landsat za {plan['place_name']} v obdobju {start_s} – {end_s}."
            if lang == "sl"
            else f"No Landsat thermal scene found for {plan['place_name']} between {start_s} and {end_s}."
        )

    best_ls_i = int(np.argmin(ls_df["cloud_cover"].fillna(50.0).to_numpy()))
    ls_date = str(ls_df.iloc[best_ls_i]["date_str"])
    ls_tile = str(ls_df.iloc[best_ls_i].get("mgrs_tile", ""))
    ls_bands = get_required_bands_for_sensor(SENSOR_LANDSAT, "LST", include_cloud_mask=True)
    ds_ls = load_scene(
        sensor=SENSOR_LANDSAT,
        item=ls_items[best_ls_i],
        bbox_4326=bbox_4326,
        bands=list(dict.fromkeys([*ls_bands, "B04", "B03", "B02"])),
        target_epsg=target_epsg,
        resolution=30,
    )
    ds_ls_m, _ = apply_cloud_mask(ds_ls, sensor=SENSOR_LANDSAT)
    lst_da = compute_index(ds_ls_m, "LST")
    lst_spec = SPECTRAL_INDICES["LST"]
    lst_uri, lst_bnds = raster_to_map_overlay_png(
        lst_da,
        src_epsg=target_epsg,
        is_rgb=False,
        cmap_name=str(lst_spec["cmap"]),
        vmin=float(lst_spec["vmin"]),
        vmax=float(lst_spec["vmax"]),
        max_width=1600,
    )
    lst_stats = compute_index_stats(lst_da, resolution_m=30.0)

    # 2. Load clearest Sentinel-2 L2A (10 m RGB + NDVI) for crisp optical & vegetation context
    _prog(0.55, "Prenašam visokoločljivi Sentinel-2 (10 m RGB + NDVI) za korelacijo z LST..." if lang == "sl" else "Streaming Sentinel-2 (10 m RGB + NDVI) for LST correlation...")
    sat_uri: str | None = None
    sat_bnds: list[list[float]] | None = None
    extra_layers: list[dict[str, Any]] = []
    syn_metrics: dict[str, Any] = {
        "fusion_type": FUSION_SYNERGY_OPTICAL_THERMAL,
        "landsat_thermal_date": ls_date,
        "lst_mean_c": round(float(lst_stats.get("mean", 0.0)), 2),
        "lst_min_c": round(float(lst_stats.get("min", 0.0)), 2),
        "lst_max_c": round(float(lst_stats.get("max", 0.0)), 2),
    }

    s2_items, s2_df = search_scenes(
        sensor=SENSOR_S2,
        bbox_4326=bbox_4326,
        start_date=start_s,
        end_date=end_s,
        max_cloud_cover=25.0,
        min_coverage_pct=70.0,
        interval_mode=INTERVAL_ALL,
    )
    if s2_items and not s2_df.empty:
        best_s2_i = int(np.argmin(s2_df["cloud_cover"].fillna(50.0).to_numpy()))
        s2_date = str(s2_df.iloc[best_s2_i]["date_str"])
        s2_res = 20 if is_low_ram_cloud() else 10
        ds_s2 = load_scene(
            sensor=SENSOR_S2,
            item=s2_items[best_s2_i],
            bbox_4326=bbox_4326,
            bands=["B04", "B03", "B02", "B08", "SCL"],
            target_epsg=target_epsg,
            resolution=s2_res,
        )
        if all(b in ds_s2.data_vars for b in ("B04", "B03", "B02")):
            sat_uri, sat_bnds = raster_to_map_overlay_png(
                ds_s2,
                src_epsg=target_epsg,
                is_rgb=True,
                bands=["B04", "B03", "B02"],
                use_percentile_stretch=True,
                max_width=1600,
            )
        ds_s2_m, _ = apply_cloud_mask(ds_s2, sensor=SENSOR_S2)
        ndvi_da = compute_index(ds_s2_m, "NDVI")
        ndvi_spec = SPECTRAL_INDICES["NDVI"]
        ndvi_u, ndvi_b = raster_to_map_overlay_png(
            ndvi_da,
            src_epsg=target_epsg,
            is_rgb=False,
            cmap_name=str(ndvi_spec["cmap"]),
            vmin=float(ndvi_spec["vmin"]),
            vmax=float(ndvi_spec["vmax"]),
            max_width=1400,
        )
        extra_layers.append(
            {
                "code": "S2_NDVI",
                "name": f"🌿 Sentinel-2 NDVI ({s2_res} m, {s2_date})",
                "image": ndvi_u,
                "bounds": ndvi_b,
            }
        )
        syn_metrics["s2_optical_date"] = s2_date
        syn_metrics["s2_resolution_m"] = s2_res

        # Cross-sensor quantitative UHI cooling correlation (align 10 m NDVI -> 30 m LST grid)
        try:
            ndvi_aligned = ndvi_da.rio.reproject_match(lst_da)
            n_arr = np.asarray(ndvi_aligned.values, dtype=np.float32)
            t_arr = np.asarray(lst_da.values, dtype=np.float32)
            valid = np.isfinite(n_arr) & np.isfinite(t_arr)
            if int(np.sum(valid)) >= 25:
                nv = n_arr[valid]
                tv = t_arr[valid]
                veg_mask = nv > 0.5
                urb_mask = nv < 0.2
                if int(np.sum(veg_mask)) >= 10:
                    syn_metrics["lst_veg_mean_c"] = round(float(np.mean(tv[veg_mask])), 2)
                if int(np.sum(urb_mask)) >= 10:
                    syn_metrics["lst_urban_mean_c"] = round(float(np.mean(tv[urb_mask])), 2)
                if "lst_veg_mean_c" in syn_metrics and "lst_urban_mean_c" in syn_metrics:
                    syn_metrics["uhi_cooling_delta_c"] = round(
                        float(syn_metrics["lst_urban_mean_c"] - syn_metrics["lst_veg_mean_c"]), 2
                    )
                if float(np.std(nv)) > 1e-5 and float(np.std(tv)) > 1e-5:
                    r_val = float(np.corrcoef(nv, tv)[0, 1])
                    if np.isfinite(r_val):
                        syn_metrics["pearson_r_ndvi_lst"] = round(r_val, 3)
            del ndvi_aligned
        except Exception:
            pass
        del ds_s2, ds_s2_m, ndvi_da
    elif all(b in ds_ls.data_vars for b in ("B04", "B03", "B02")):
        sat_uri, sat_bnds = raster_to_map_overlay_png(
            ds_ls,
            src_epsg=target_epsg,
            is_rgb=True,
            bands=["B04", "B03", "B02"],
            use_percentile_stretch=True,
            max_width=1600,
        )

    tif_bytes = export_to_geotiff_bytes(lst_da, target_epsg=target_epsg)
    tif_fname = f"AI_Fusion_LST_S2_{plan['place_name']}_{ls_date}_EPSG{epsg_num}.tif"
    del ds_ls, ds_ls_m
    gc.collect()

    base_out.update(
        {
            "kind": "spectral_index",
            "index_da": lst_da,
            "stats": lst_stats,
            "snow_stats": None,
            "date_str": ls_date,
            "sat_date_str": syn_metrics.get("s2_optical_date", ls_date),
            "tile": ls_tile,
            "sat_overlay_uri": sat_uri,
            "sat_overlay_bounds": sat_bnds,
            "analysis_overlay_uri": lst_uri,
            "analysis_overlay_bounds": lst_bnds,
            "extra_overlay_layers": extra_layers,
            "synergy_metrics": syn_metrics,
            "tif_bytes": tif_bytes,
            "tif_filename": tif_fname,
        }
    )
    base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
    base_out["analysis_context"] = build_analysis_context_dict(plan, base_out)
    _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
    return base_out


def execute_ai_plan(
    plan: dict[str, Any],
    *,
    target_epsg: str | None = None,
    geocode_fn: Callable[[str], tuple[float, float, str] | None] | None = None,
    lang: str = "sl",
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Execute ANY validated AI EO/geodetic plan strictly in local metric projection (EPSG:3794 in SLO, auto-UTM globally).

    Supports all app capabilities:
    ``disaster_flood``, ``disaster_fire``, ``disaster_forest``, ``spectral_index`` (incl. NDSI+DEM
    snowline & LST °C), ``scene_rgb``, ``cloud_free_composite``, ``time_series``,
    ``time_series_and_comparison``, ``change_comparison``, ``terrain_dem``, and ``gnss_plan``.
    """

    def _prog(pct: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, msg)

    _prog(0.08, f"Geokodiram lokacijo: {plan['place_name']}..." if lang == "sl" else f"Geocoding location: {plan['place_name']}...")
    lat, lon, resolved_label = resolve_plan_coordinates(plan, geocode_fn=geocode_fn, lang=lang)

    # Always enforce exact local metric projection: EPSG:3794 in Slovenia, local auto-UTM globally!
    target_epsg = resolve_ai_metric_crs(lat, lon)
    bbox_4326, bbox_3794 = bbox_from_point_wgs84(lat, lon, float(plan["radius_m"]))
    task = plan["task_type"]
    sensor = plan["sensor"]
    res_m = int(plan["resolution_m"])
    epsg_num = str(target_epsg).split(":")[-1]

    base_out: dict[str, Any] = {
        "plan": {**plan, "lat": lat, "lon": lon, "target_epsg": target_epsg},
        "crs": target_epsg,
        "epsg_code": int(epsg_num),
        "lat": lat,
        "lon": lon,
        "resolved_place": resolved_label,
        "bbox_4326": bbox_4326,
        "bbox_3794": bbox_3794,
    }

    # --- A. Copernicus DEM 30 m Terrain Analysis ---
    if task == "terrain_dem":
        _prog(0.25, "Prenašam Copernicus DEM 30 m..." if lang == "sl" else "Streaming Copernicus DEM 30 m...")
        dem_da = load_copernicus_dem(bbox_4326=bbox_4326, target_epsg=target_epsg, resolution=30.0)
        _prog(0.55, "Računam naklon, ekspozicijo, senčenje in višinski profil..." if lang == "sl" else "Computing slope, aspect, hillshade, and elevation profile...")
        deriv = compute_terrain_derivatives(dem_da)
        dem_stats = deriv["stats"]
        elev_da = deriv["elevation"]
        slope_da = deriv["slope_deg"]
        hill_da = deriv["hillshade"]

        elev_uri, elev_bnds = raster_to_map_overlay_png(
            elev_da,
            src_epsg=target_epsg,
            is_rgb=False,
            cmap_name="terrain",
            vmin=float(dem_stats["min_h"]),
            vmax=max(float(dem_stats["max_h"]), float(dem_stats["min_h"]) + 10.0),
            max_width=1800,
        )
        slope_uri, slope_bnds = raster_to_map_overlay_png(
            slope_da,
            src_epsg=target_epsg,
            is_rgb=False,
            cmap_name="YlOrRd",
            vmin=0.0,
            vmax=45.0,
            max_width=1800,
        )
        hill_uri, hill_bnds = raster_to_map_overlay_png(
            hill_da,
            src_epsg=target_epsg,
            is_rgb=False,
            cmap_name="Greys_r",
            vmin=0.0,
            vmax=255.0,
            max_width=1800,
        )
        profile_df = None
        try:
            p_start, p_end = build_preset_transect_endpoints(elev_da, mode="we")
            profile_df, _profile_summary = extract_transect_profile(elev_da, p_start, p_end, n_samples=160)
        except Exception:
            profile_df = None

        tif_bytes = export_to_geotiff_bytes(elev_da, target_epsg=target_epsg)
        tif_fname = f"AI_DEM30m_{plan['place_name']}_EPSG{epsg_num}.tif"
        gc.collect()
        base_out.update(
            {
                "kind": "terrain_dem",
                "index_da": elev_da,
                "dem_stats": dem_stats,
                "dem_elev_uri": elev_uri,
                "dem_elev_bounds": elev_bnds,
                "dem_slope_uri": slope_uri,
                "dem_slope_bounds": slope_bnds,
                "dem_hill_uri": hill_uri,
                "dem_hill_bounds": hill_bnds,
                "profile_df": profile_df,
                "tif_bytes": tif_bytes,
                "tif_filename": tif_fname,
            }
        )
        base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
        base_out["analysis_context"] = build_analysis_context_dict(plan, base_out)
        _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
        return base_out

    # --- B. GNSS Mission Planning ---
    if task == "gnss_plan":
        plan_dt = datetime.strptime(plan["event_date"], "%Y-%m-%d").date()
        gnss_out = run_gnss_mission_plan(
            lat_deg=lat,
            lon_deg=lon,
            plan_date=plan_dt,
            constellations=("GPS", "Galileo", "GLONASS"),
            cutoff_deg=10.0,
            ant_height_m=2.0,
            use_dem=True,
            use_s2_obstacles=True,
            tz_name="Europe/Ljubljana",
            step_minutes=15,
            target_epsg=target_epsg,
            progress_cb=lambda f, m: _prog(0.15 + 0.80 * f, m),
        )
        tl_df: pd.DataFrame = gnss_out["timeline_df"]
        csv_bytes = tl_df.to_csv(index=False).encode("utf-8")
        csv_fname = f"AI_GNSS_{plan['place_name']}_{plan_dt.isoformat()}.csv"
        gc.collect()
        base_out.update(
            {
                "kind": "gnss_plan",
                "gnss_result": gnss_out,
                "csv_bytes": csv_bytes,
                "csv_filename": csv_fname,
            }
        )
        base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
        base_out["analysis_context"] = build_analysis_context_dict(plan, base_out)
        _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
        return base_out

    # --- C. Disaster Assessment Tasks (Flood SAR / Wildfire dNBR / Forest dNDVI) ---
    if task in ("disaster_flood", "disaster_fire", "disaster_forest"):
        dis_map = {
            "disaster_flood": DISASTER_FLOOD,
            "disaster_fire": DISASTER_WILDFIRE,
            "disaster_forest": DISASTER_FOREST,
        }
        dt_k = dis_map[task]
        ev_date = datetime.strptime(plan["event_date"], "%Y-%m-%d").date()
        dis_out = run_disaster_assessment(
            disaster_type=dt_k,
            bbox_4326=bbox_4326,
            event_date=ev_date,
            target_crs=target_epsg,
            resolution=max(res_m, 30 if sensor == SENSOR_LANDSAT else 10),
            search_window_days=60,
            lang=lang,
            progress_cb=lambda f, m: _prog(0.15 + 0.78 * f, m),
        )
        analysis_uri, analysis_bnds = raster_to_map_overlay_png(
            dis_out["class_da"],
            src_epsg=target_epsg,
            is_rgb=False,
            discrete_rgba_map=build_slippy_rgba_map(dt_k),
            max_width=1800,
        )
        extra_syn_layers: list[dict[str, Any]] = []
        syn_metrics_dis: dict[str, Any] = {}
        if str(plan.get("fusion_mode") or "") == FUSION_SYNERGY_SAR_OPTICAL:
            _prog(0.92, "Dodajam najbližji jasen optični posnetek Sentinel-2 (RGB + NDWI)..." if lang == "sl" else "Attaching nearest clear Sentinel-2 optical scene (RGB + NDWI)...")
            extra_syn_layers, syn_metrics_dis = _attach_sar_optical_synergy(
                bbox_4326=bbox_4326,
                ref_date=ev_date,
                target_epsg=target_epsg,
                lang=lang,
            )
            syn_metrics_dis["sar_event_date"] = str(dis_out.get("date_event", ""))
            syn_metrics_dis["affected_area_ha"] = round(float((dis_out.get("summary") or {}).get("affected_area_ha", 0.0)), 2)

        base_out.update(
            {
                "kind": "disaster",
                "disaster_result": dis_out,
                "sat_overlay_uri": dis_out.get("overlay_uri_post"),
                "sat_overlay_bounds": dis_out.get("overlay_bounds_post"),
                "sat_overlay_uri_pre": dis_out.get("overlay_uri_pre"),
                "sat_overlay_bounds_pre": dis_out.get("overlay_bounds_pre"),
                "sat_date_str": str(dis_out.get("date_event", "")),
                "analysis_overlay_uri": analysis_uri,
                "analysis_overlay_bounds": analysis_bnds,
                "extra_overlay_layers": extra_syn_layers,
                "synergy_metrics": syn_metrics_dis,
                "tif_bytes": dis_out["tif_bytes"],
                "tif_filename": dis_out["tif_filename"],
            }
        )
        base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
        base_out["analysis_context"] = build_analysis_context_dict(plan, base_out)
        _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
        return base_out

    # --- D. Cloud-Free Temporal Median Composite ---
    if task == "cloud_free_composite":
        opt_sensor = SENSOR_LANDSAT if sensor == SENSOR_LANDSAT else SENSOR_S2
        _prog(0.20, "Iščem najčistejše posnetke za brezoblačni mozaik..." if lang == "sl" else "Searching clearest scenes for cloud-free composite...")
        items, df = search_scenes(
            sensor=opt_sensor,
            bbox_4326=bbox_4326,
            start_date=plan["start_date"],
            end_date=plan["end_date"],
            max_cloud_cover=35.0,
            exclude_l7_slc_off=(opt_sensor == SENSOR_LANDSAT),
            min_coverage_pct=85.0,
            interval_mode=INTERVAL_ALL,
        )
        if not items or df.empty:
            raise RuntimeError(
                f"Ni najdenih optičnih posnetkov za mozaik ({plan['start_date']} – {plan['end_date']})."
                if lang == "sl"
                else f"No optical scenes found for cloud-free composite ({plan['start_date']} – {plan['end_date']})."
            )
        comp_mode = "RGB_NIR"
        comp_obj, comp_meta = compute_cloud_free_composite(
            items=items,
            bbox_4326=bbox_4326,
            mode=comp_mode,
            target_epsg=target_epsg,
            resolution=max(res_m, 30 if opt_sensor == SENSOR_LANDSAT else 10),
            sensor=opt_sensor,
            max_scenes=6,
            progress_callback=lambda c, tot, d: _prog(0.28 + 0.62 * (c / max(tot, 1)), f"{d} ({c}/{tot})"),
        )
        if comp_obj is None:
            raise RuntimeError(
                "Brezoblačnega mozaika ni bilo mogoče sestaviti (preveč oblakov v izbranem obdobju)."
                if lang == "sl"
                else "Could not assemble cloud-free composite (too much cloud cover in period)."
            )
        sat_uri, sat_bnds = raster_to_map_overlay_png(
            comp_obj,
            src_epsg=target_epsg,
            is_rgb=True,
            bands=["B04", "B03", "B02"],
            use_percentile_stretch=True,
            gamma=1.0,
            max_width=1800,
        )
        tif_bytes = export_to_geotiff_bytes(comp_obj, target_epsg=target_epsg)
        tif_fname = f"AI_Composite_RGB_{plan['place_name']}_{comp_meta.get('date_start', '')}_{comp_meta.get('date_end', '')}_EPSG{epsg_num}.tif"
        gc.collect()
        base_out.update(
            {
                "kind": "cloud_free_composite",
                "scene_ds": comp_obj,
                "bands": ["B04", "B03", "B02"],
                "composite_meta": comp_meta,
                "sat_overlay_uri": sat_uri,
                "sat_overlay_bounds": sat_bnds,
                "tif_bytes": tif_bytes,
                "tif_filename": tif_fname,
            }
        )
        base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
        base_out["analysis_context"] = build_analysis_context_dict(plan, base_out)
        _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
        return base_out

    # --- E. 2-Date Before / After Change Comparison (Zoomable Swipe Map + Δ Stats) ---
    if task == "change_comparison":
        _prog(0.18, "Iščem par najčistejših posnetkov Prej / Potem..." if lang == "sl" else "Searching cleanest Before / After scene pair...")
        span_days = (
            datetime.strptime(plan["end_date"], "%Y-%m-%d").date()
            - datetime.strptime(plan["start_date"], "%Y-%m-%d").date()
        ).days
        int_mode = INTERVAL_YEARLY if span_days >= 360 else INTERVAL_ALL
        max_cc = 100.0 if sensor == SENSOR_S1 else 30.0
        items, df = search_scenes(
            sensor=sensor,
            bbox_4326=bbox_4326,
            start_date=plan["start_date"],
            end_date=plan["end_date"],
            max_cloud_cover=max_cc,
            exclude_l7_slc_off=(sensor == SENSOR_LANDSAT),
            min_coverage_pct=85.0,
            interval_mode=int_mode,
            target_season=SEASON_SUMMER,
        )
        if not items or len(items) < 2:
            items, df = search_scenes(
                sensor=sensor,
                bbox_4326=bbox_4326,
                start_date=plan["start_date"],
                end_date=plan["end_date"],
                max_cloud_cover=50.0 if sensor != SENSOR_S1 else 100.0,
                exclude_l7_slc_off=(sensor == SENSOR_LANDSAT),
                min_coverage_pct=70.0,
                interval_mode=INTERVAL_ALL,
            )
        if not items or len(items) < 2:
            raise RuntimeError(
                f"Za primerjavo sta potrebna vsaj 2 posnetka v obdobju {plan['start_date']} – {plan['end_date']}."
                if lang == "sl"
                else f"At least 2 scenes are required for comparison between {plan['start_date']} and {plan['end_date']}."
            )
        # Pick oldest and newest date in the matched list
        sorted_idx = sorted(range(len(df)), key=lambda i: str(df.iloc[i]["date_str"]))
        i_first, i_last = sorted_idx[0], sorted_idx[-1]
        chosen_groups = [items[i_first], items[i_last]]
        chosen_dates = [str(df.iloc[i_first]["date_str"]), str(df.iloc[i_last]["date_str"])]
        raw_by_date = df.attrs.get("raw_items_by_date") or {}
        all_cands = list(raw_by_date.values()) if isinstance(raw_by_date, dict) and raw_by_date else list(items)
        layer_m = str(plan["index_name"])
        _prog(0.50, f"Pripravljam primerjavo {chosen_dates[0]} ↔ {chosen_dates[-1]} ({layer_m})..." if lang == "sl" else f"Building {chosen_dates[0]} ↔ {chosen_dates[-1]} comparison ({layer_m})...")
        comp_res = render_multi_date_comparison(
            item_groups=chosen_groups,
            date_labels=chosen_dates,
            bbox_4326=bbox_4326,
            layer_mode=layer_m,
            target_epsg=target_epsg,
            resolution=float(max(res_m, 30 if sensor == SENSOR_LANDSAT else 10)),
            sensor=sensor,
            mask_clouds=(sensor != SENSOR_S1),
            all_candidates=all_cands,
            candidates_by_date=df.attrs.get("candidates_by_bucket"),
            auto_cloud_fallback=(sensor != SENSOR_S1),
            min_clear_pct=75.0,
        )
        diff_da = comp_res.get("diff_da")
        ch_summary = (
            compute_change_summary(
                diff_da,
                resolution_m=float(max(res_m, 30 if sensor == SENSOR_LANDSAT else 10)),
                threshold=2.0 if sensor == SENSOR_S1 else (2.0 if layer_m == "LST" else 0.15),
            )
            if diff_da is not None
            else {}
        )
        tif_bytes = export_to_geotiff_bytes(diff_da, target_epsg=target_epsg) if diff_da is not None else None
        r_dates = comp_res.get("dates") or chosen_dates
        tif_fname = f"AI_Delta_{layer_m}_{plan['place_name']}_{r_dates[0]}_to_{r_dates[-1]}_EPSG{epsg_num}.tif"
        gc.collect()
        base_out.update(
            {
                "kind": "change_comparison",
                "comparison_result": comp_res,
                "change_summary": ch_summary,
                "index_da": diff_da,
                "tif_bytes": tif_bytes,
                "tif_filename": tif_fname,
            }
        )
        base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
        base_out["analysis_context"] = build_analysis_context_dict(plan, base_out)
        _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
        return base_out

    # --- F. Multi-Temporal Time Series (+ Multi-Date Intermediate Frames, Swipe Map & Timelapse GIF) ---
    if task in ("time_series", "time_series_and_comparison") or (
        task == "change_comparison" and str(plan.get("fusion_mode") or "") == FUSION_TEMPORAL_LANDSAT_S2
    ):
        # Optional Google Earth Engine (GEE) server-side routing for large regional/continental or explicit GEE queries
        _q_norm_gee = _normalize_ascii(str(plan.get("user_query") or plan.get("explanation_sl") or "")).lower()
        _user_asked_gee = any(k in _q_norm_gee for k in ("google earth engine", "earth engine", " gee", "gee "))
        _is_large_for_gee = (
            _sfloat(plan.get("radius_m"), 0.0) >= 40_000.0
            or sensor in MODIS_SENSORS
            or _detect_continental_polar_region(_q_norm_gee, str(plan.get("place_name") or "")) is not None
        )
        _idx_for_gee = str(plan.get("index_name") or "NDVI").strip().upper()
        if is_gee_ready() and _idx_for_gee in GEE_SUPPORTED_INDICES and (_user_asked_gee or _is_large_for_gee):
            _prog(
                0.10,
                "🌍 Povezan z Google Earth Engine — zaganjam strežniško obdelavo..."
                if lang == "sl"
                else "🌍 Connected to Google Earth Engine — starting server-side analysis...",
            )
            _sy_gee = _parse_iso_date(plan.get("start_date"), date(2000, 1, 1)).year
            _ey_gee = _parse_iso_date(plan.get("end_date"), date.today()).year
            _gee_sensor_mode = "modis" if (_sfloat(plan.get("radius_m"), 0.0) >= 400_000.0 and _sy_gee >= 2000) else "auto_fusion"
            _gee_season = _resolve_ai_target_season(plan)
            gee_res = gee_run_temporal_analysis(
                bbox_4326=bbox_4326,
                start_year=_sy_gee,
                end_year=_ey_gee,
                index_code=_idx_for_gee,
                sensor_mode=_gee_sensor_mode,
                target_season=_gee_season,
                max_frames=9,
                progress_cb=progress_cb,
                lang=lang,
            )
            _ts_gee = gee_res["ts_df"]
            _csv_gee = _ts_gee.to_csv(index=False).encode("utf-8-sig")
            _csv_fn_gee = f"GEE_{_idx_for_gee}_{plan['place_name']}_{_sy_gee}_{_ey_gee}.csv"
            base_out["plan"]["fusion_rationale_sl"] = (
                "Za visokonatančno obdelavo velikega območja je bila uporabljena strežniška platforma "
                "**Google Earth Engine (GEE)** (Landsat C2 L2 + Sentinel-2 SR / MODIS), ki izračuna "
                "sezonske mediano-mozaike, Δ spremembe in površine v km² neposredno na Googlovih strežnikih."
            )
            base_out["plan"]["fusion_rationale_en"] = (
                "Server-side **Google Earth Engine (GEE)** processing (Landsat C2 L2 + Sentinel-2 SR / MODIS) "
                "was used for high-accuracy large-scale computation of seasonal median composites, Δ change maps, "
                "and exact pixelArea() statistics in km² directly on Google's servers."
            )
            base_out.update(
                {
                    "kind": "time_series_and_comparison" if len(gee_res["frames"]) >= 2 else "time_series",
                    "engine": "gee",
                    "gee_project_id": gee_res.get("gee_project_id", ""),
                    "adaptive_interval": "Letno",
                    "relevant_layers": ["RGB", _idx_for_gee] + list(gee_res.get("extra_indices") or []),
                    "extra_indices": list(gee_res.get("extra_indices") or []),
                    "ts_df": _ts_gee,
                    "csv_bytes": _csv_gee,
                    "csv_filename": _csv_fn_gee,
                    "frames": gee_res["frames"],
                    "default_t1_index": gee_res["default_t1_index"],
                    "excluded_archive_dates": gee_res["excluded_archive_dates"],
                    "timelapse_rgb_gif_bytes": gee_res["timelapse_rgb_gif_bytes"],
                    "timelapse_index_gif_bytes": gee_res["timelapse_index_gif_bytes"],
                    "comparison_rgb": None,
                    "comparison_index": None,
                    "comparison_result": None,
                    "change_summary": gee_res["change_summary"],
                    "delta_overlay_uri": gee_res["delta_overlay_uri"],
                    "delta_overlay_bounds": gee_res["delta_overlay_bounds"],
                    "delta_vmin": gee_res["delta_vmin"],
                    "delta_vmax": gee_res["delta_vmax"],
                    "delta_cmap": gee_res["delta_cmap"],
                    "synergy_metrics": {
                        "fusion_type": "gee_server_side",
                        "engine": "gee",
                        "gee_project_id": gee_res.get("gee_project_id", ""),
                        "reduce_scale_m": gee_res.get("reduce_scale_m", 120),
                        "total_frames": len(gee_res["frames"]),
                    },
                }
            )
            base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
            base_out["analysis_context"] = build_analysis_context_dict(plan, base_out)
            _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
            return base_out
        if _user_asked_gee and not is_gee_ready():
            base_out["gee_fallback_note"] = (
                "Opomba: Google Earth Engine še ni nastavljen v stranski vrstici, zato je bila analiza samodejno izvedena prek Microsoft Planetary Computer (STAC/MODIS)."
                if lang == "sl"
                else "Note: Google Earth Engine is not configured in the sidebar yet, so the analysis ran automatically via Microsoft Planetary Computer (STAC/MODIS)."
            )

        if str(plan.get("fusion_mode") or "") == FUSION_TEMPORAL_LANDSAT_S2:
            return execute_temporal_fusion(
                plan,
                base_out=base_out,
                bbox_4326=bbox_4326,
                target_epsg=target_epsg,
                lang=lang,
                progress_cb=progress_cb,
            )
        adaptive_interval = choose_adaptive_interval(
            plan["start_date"],
            plan["end_date"],
            user_requested_interval=(plan.get("interval") if plan.get("interval_explicit") else None),
        )
        base_out["plan"]["interval"] = adaptive_interval
        _prog(
            0.14,
            f"Iščem satelitski arhiv ({adaptive_interval})..."
            if lang == "sl"
            else f"Searching satellite archive ({adaptive_interval})...",
        )
        interval_key = INTERVAL_LABEL_TO_KEY.get(adaptive_interval, INTERVAL_YEARLY)
        ai_target_season = _resolve_ai_target_season(plan)
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
            target_season=ai_target_season,
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
                interval_mode=interval_key,
                target_season=ai_target_season,
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
            df = df.iloc[pick_idx].reset_index(drop=True)

        want_visuals = bool(
            task == "time_series_and_comparison"
            or plan.get("include_comparison")
            or len(items) >= 2
        )
        ts_prog_span = 0.36 if want_visuals else 0.72

        def _ts_cb(cur: int, total: int, d_str: str) -> None:
            _prog(0.18 + ts_prog_span * (cur / max(total, 1)), f"{d_str} ({cur}/{total})")

        eff_ts_res = max(res_m, 30 if sensor == SENSOR_LANDSAT else 10)
        ts_df = extract_time_series(
            items=items,
            bbox_4326=bbox_4326,
            index_name=plan["index_name"],
            target_epsg=target_epsg,
            resolution=eff_ts_res,
            mask_clouds=(sensor != SENSOR_S1),
            progress_callback=_ts_cb,
            sensor=sensor,
            candidates_by_date=df.attrs.get("candidates_by_bucket"),
        )
        if ts_df.empty:
            raise RuntimeError(
                "Noben posnetek v obdobju ni imel dovolj ne-oblačnih pikslov za časovno vrsto."
                if lang == "sl"
                else "No scenes had enough cloud-free pixels to build the time series."
            )

        csv_bytes = ts_df.to_csv(index=False).encode("utf-8")
        csv_fname = f"AI_TimeSeries_{plan['index_name']}_{plan['place_name']}_{plan['start_date']}_{plan['end_date']}.csv"

        # Step B: Build RAM-safe chronological multi-date frame stack (RGB + Index) & Timelapse GIFs!
        frames: list[dict[str, Any]] = []
        gif_rgb_bytes: bytes | None = None
        gif_idx_bytes: bytes | None = None
        comp_rgb: dict[str, Any] | None = None
        comp_idx: dict[str, Any] | None = None
        ch_summary: dict[str, Any] = {}
        diff_da = None
        da_last = None
        tif_bytes: bytes | None = None
        tif_fname: str | None = None

        delta_uri: str | None = None
        delta_bnds: list[list[float]] | None = None
        delta_vmin: float = -0.5
        delta_vmax: float = 0.5
        delta_cmap: str = "RdBu"
        glacier_class_uri: str | None = None
        glacier_class_bounds: list[list[float]] | None = None
        glacier_change_stats: dict[str, float] | None = None
        layer_m = str(plan["index_name"])
        relevant_layers = get_relevant_indices(layer_m, task)
        extra_idx_list = [l for l in relevant_layers if l not in ("RGB", "SAR", layer_m)]

        if want_visuals and len(ts_df) >= 2:
            raw_by_date = df.attrs.get("raw_items_by_date") or {}
            date_to_group: dict[str, Any] = {}
            for i_row in range(min(len(items), len(df))):
                date_to_group[str(df.iloc[i_row]["date_str"])] = items[i_row]
            if isinstance(raw_by_date, dict):
                for dk, dv in raw_by_date.items():
                    date_to_group.setdefault(str(dk), dv)

            ts_dates_sorted = [str(d) for d in ts_df["date_str"].tolist()]
            matched_groups: list[Any] = []
            matched_dates: list[str] = []
            for d_str in ts_dates_sorted:
                grp = date_to_group.get(d_str)
                if grp is not None:
                    matched_groups.append(grp)
                    matched_dates.append(d_str)
            if len(matched_groups) < 2:
                matched_groups = list(items)
                matched_dates = [str(df.iloc[i]["date_str"]) for i in range(min(len(items), len(df)))]

            all_cands = list(raw_by_date.values()) if isinstance(raw_by_date, dict) and raw_by_date else list(items)
            low_ram = is_low_ram_cloud()
            max_vis_frames = 9 if low_ram else 14
            area_est_km2 = ((2.0 * float(plan["radius_m"])) ** 2) / 1e6
            safe_res, _ = get_safe_processing_resolution(res_m, area_est_km2, sensor=sensor)
            vis_res_m = float(max(safe_res, 20 if low_ram else res_m, 30 if sensor == SENSOR_LANDSAT else 10))

            def _vis_cb(cur: int, total: int, d_str: str) -> None:
                _prog(
                    0.56 + 0.36 * (cur / max(total, 1)),
                    f"Nalagam vmesni posnetek {d_str} ({cur}/{total})..."
                    if lang == "sl"
                    else f"Loading intermediate scene {d_str} ({cur}/{total})...",
                )

            mf_out = build_multidate_visual_frames(
                item_groups=matched_groups,
                date_labels=matched_dates,
                bbox_4326=bbox_4326,
                index_name=layer_m,
                target_epsg=target_epsg,
                resolution=vis_res_m,
                sensor=sensor,
                extra_indices=extra_idx_list,
                max_frames=max_vis_frames,
                all_candidates=all_cands,
                candidates_by_date=df.attrs.get("candidates_by_bucket"),
                auto_cloud_fallback=(sensor != SENSOR_S1),
                min_clear_pct=75.0,
                max_thumb_width=640 if low_ram else 760,
                overlay_max_width=1000 if low_ram else 1500,
                default_fps=1.5,
                progress_callback=_vis_cb,
            )
            frames = list(mf_out.get("frames") or [])

            # HARD GUARANTEE: if strict cloud threshold yielded < 2 frames, retry earliest & latest pair cleanly
            if len(frames) < 2 and len(matched_groups) >= 2:
                mf_out = build_multidate_visual_frames(
                    item_groups=[matched_groups[0], matched_groups[-1]],
                    date_labels=[matched_dates[0], matched_dates[-1]],
                    bbox_4326=bbox_4326,
                    index_name=layer_m,
                    target_epsg=target_epsg,
                    resolution=vis_res_m,
                    sensor=sensor,
                    extra_indices=extra_idx_list,
                    max_frames=2,
                    all_candidates=all_cands,
                    auto_cloud_fallback=False,
                    min_clear_pct=10.0,
                    max_thumb_width=640 if low_ram else 760,
                    overlay_max_width=1000 if low_ram else 1500,
                    default_fps=1.5,
                )
                frames = list(mf_out.get("frames") or [])

            gif_rgb_bytes = mf_out.get("timelapse_rgb_gif_bytes")
            gif_idx_bytes = mf_out.get("timelapse_index_gif_bytes")
            diff_da = mf_out.get("diff_da")
            da_last = mf_out.get("da_last")
            delta_uri = mf_out.get("delta_overlay_uri")
            delta_bnds = mf_out.get("delta_overlay_bounds")
            delta_vmin = float(mf_out.get("delta_vmin", -0.5))
            delta_vmax = float(mf_out.get("delta_vmax", 0.5))
            delta_cmap = str(mf_out.get("delta_cmap") or "RdBu")
            glacier_class_uri = mf_out.get("glacier_class_uri")
            glacier_class_bounds = mf_out.get("glacier_class_bounds")
            glacier_change_stats = mf_out.get("glacier_change_stats")
            mf_reps = list(mf_out.get("replacements") or [])

            default_t1_idx_ss = int(mf_out.get("default_t1_index", get_default_reliable_t1_index(frames)))
            if len(frames) >= 2 and default_t1_idx_ss >= len(frames) - 1:
                default_t1_idx_ss = 0
            if len(frames) >= 2:
                f0, f_last = frames[default_t1_idx_ss], frames[-1]
                if f0.get("rgb_overlay_uri") and f_last.get("rgb_overlay_uri"):
                    comp_rgb = {
                        "dates": [f0["date"], f_last["date"]],
                        "overlay_uris": [f0["rgb_overlay_uri"], f_last["rgb_overlay_uri"]],
                        "overlay_bounds": [f0["bounds_4326"], f_last["bounds_4326"]],
                        "replacements": mf_reps,
                    }
                if f0.get("index_overlay_uri") and f_last.get("index_overlay_uri"):
                    comp_idx = {
                        "dates": [f0["date"], f_last["date"]],
                        "overlay_uris": [f0["index_overlay_uri"], f_last["index_overlay_uri"]],
                        "overlay_bounds": [f0["bounds_4326"], f_last["bounds_4326"]],
                        "replacements": mf_reps,
                        "diff_da": diff_da,
                    }

            if diff_da is not None:
                ch_summary = compute_change_summary(
                    diff_da,
                    resolution_m=vis_res_m,
                    threshold=2.0 if sensor == SENSOR_S1 else (2.0 if layer_m == "LST" else 0.15),
                )
                tif_bytes = export_to_geotiff_bytes(diff_da, target_epsg=target_epsg)
                r_d0 = frames[default_t1_idx_ss]["date"] if frames else ts_dates_sorted[0]
                r_d1 = frames[-1]["date"] if frames else ts_dates_sorted[-1]
                tif_fname = f"AI_Delta_{layer_m}_{plan['place_name']}_{r_d0}_to_{r_d1}_EPSG{epsg_num}.tif"

        excluded_archive_dates_ss: list[str] = sorted(
            {
                str(f.get("date"))
                for f in frames
                if isinstance(f, dict) and f.get("analysis_reliable") is False and f.get("date")
            }
            | (
                set(ts_df.loc[~ts_df["analysis_reliable"].astype(bool), "date_str"].astype(str).tolist())
                if "analysis_reliable" in ts_df.columns
                else set()
            )
        )
        result_kind = (
            "time_series_and_comparison"
            if (len(frames) >= 2 or comp_rgb is not None or comp_idx is not None)
            else "time_series"
        )
        gc.collect()
        base_out.update(
            {
                "kind": result_kind,
                "adaptive_interval": adaptive_interval,
                "relevant_layers": relevant_layers,
                "extra_indices": extra_idx_list,
                "ts_df": ts_df,
                "csv_bytes": csv_bytes,
                "csv_filename": csv_fname,
                "frames": frames,
                "default_t1_index": get_default_reliable_t1_index(frames),
                "excluded_archive_dates": excluded_archive_dates_ss,
                "timelapse_rgb_gif_bytes": gif_rgb_bytes,
                "timelapse_index_gif_bytes": gif_idx_bytes,
                "comparison_rgb": comp_rgb,
                "comparison_index": comp_idx,
                "comparison_result": comp_idx or comp_rgb,
                "change_summary": ch_summary,
                "delta_overlay_uri": delta_uri,
                "delta_overlay_bounds": delta_bnds,
                "delta_vmin": delta_vmin,
                "delta_vmax": delta_vmax,
                "delta_cmap": delta_cmap,
                "glacier_class_uri": glacier_class_uri,
                "glacier_class_bounds": glacier_class_bounds,
                "glacier_change_stats": glacier_change_stats,
                "da_last": da_last,
                "index_da": diff_da if diff_da is not None else da_last,
                "tif_bytes": tif_bytes,
                "tif_filename": tif_fname,
            }
        )
        base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
        base_out["analysis_context"] = build_analysis_context_dict(plan, base_out)
        _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
        return base_out

    # --- G. Single-Date Spectral Index (incl. NDSI+DEM Snowline & LST °C) or RGB Scene ---
    if str(plan.get("fusion_mode") or "") == FUSION_SYNERGY_OPTICAL_THERMAL:
        return _execute_optical_thermal_synergy(
            plan,
            base_out=base_out,
            bbox_4326=bbox_4326,
            target_epsg=target_epsg,
            lang=lang,
            progress_cb=progress_cb,
        )

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
        sat_uri: str | None = None
        sat_bnds: list[list[float]] | None = None
        snow_stats: dict[str, Any] | None = None
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
            try:
                sat_uri, sat_bnds = raster_to_map_overlay_png(
                    ds["VV"],
                    src_epsg=target_epsg,
                    is_rgb=False,
                    cmap_name="Greys_r",
                    vmin=-22.0,
                    vmax=-2.0,
                    max_width=1800,
                )
            except Exception:
                sat_uri, sat_bnds = None, None
        else:
            req_bands = get_required_bands_for_sensor(
                sensor=sensor,
                metric_name=plan["index_name"],
                include_cloud_mask=True,
            )
            rel_layers = get_relevant_indices(plan["index_name"], task)
            comp_codes = [c for c in rel_layers if c not in ("RGB", "SAR", plan["index_name"])]
            # Fetch True Color RGB + primary + complementary index bands in ONE single STAC request
            extra_b = ["B04", "B03", "B02"]
            if plan["index_name"] == "NDSI":
                extra_b.append("B08")
            for cc in comp_codes:
                if cc == "SWIR":
                    extra_b.extend(["B12", "B08", "B04"])
                elif cc in SPECTRAL_INDICES:
                    extra_b.extend(get_required_bands_for_sensor(sensor, cc, include_cloud_mask=False))
            fetch_bands = list(dict.fromkeys([*req_bands, *extra_b]))
            ds = load_scene(
                sensor=sensor,
                item=best_group,
                bbox_4326=bbox_4326,
                bands=fetch_bands,
                target_epsg=target_epsg,
                resolution=res_m,
            )
            try:
                if all(b in ds.data_vars for b in ("B04", "B03", "B02")):
                    sat_uri, sat_bnds = raster_to_map_overlay_png(
                        ds,
                        src_epsg=target_epsg,
                        is_rgb=True,
                        bands=["B04", "B03", "B02"],
                        use_percentile_stretch=True,
                        gamma=1.0,
                        max_width=1800,
                    )
            except Exception:
                sat_uri, sat_bnds = None, None
            ds_masked, _ = apply_cloud_mask(ds, sensor=sensor)
            idx_da = compute_index(ds_masked, plan["index_name"])
            extra_overlay_layers: list[dict[str, Any]] = []
            for cc in comp_codes:
                try:
                    if cc == "SWIR" and all(b in ds.data_vars for b in ("B12", "B08", "B04")):
                        sw_u, sw_b = raster_to_map_overlay_png(
                            ds, src_epsg=target_epsg, is_rgb=True, bands=["B12", "B08", "B04"], max_width=1400
                        )
                        extra_overlay_layers.append({"code": "SWIR", "name": f"🔥 SWIR ({date_str})", "image": sw_u, "bounds": sw_b})
                    elif cc in SPECTRAL_INDICES:
                        c_da = compute_index(ds_masked, cc)
                        c_sp = SPECTRAL_INDICES[cc]
                        c_u, c_b = raster_to_map_overlay_png(
                            c_da,
                            src_epsg=target_epsg,
                            is_rgb=False,
                            cmap_name=str(c_sp["cmap"]),
                            vmin=float(c_sp["vmin"]),
                            vmax=float(c_sp["vmax"]),
                            max_width=1400,
                        )
                        extra_overlay_layers.append({"code": cc, "name": f"📊 {cc} ({date_str})", "image": c_u, "bounds": c_b})
                        del c_da
                except Exception:
                    pass
            synergy_metrics_sp: dict[str, Any] = {}
            if plan["index_name"] == "NDSI" or str(plan.get("fusion_mode") or "") == FUSION_SYNERGY_OPTICAL_DEM:
                try:
                    dem_sn = load_copernicus_dem(bbox_4326=bbox_4326, target_epsg=target_epsg, resolution=30.0)
                    dem_deriv = compute_terrain_derivatives(dem_sn)
                    d_st = dem_deriv["stats"]
                    synergy_metrics_sp = {
                        "fusion_type": FUSION_SYNERGY_OPTICAL_DEM,
                        "dem_min_m": round(float(d_st.get("min_h", 0.0)), 1),
                        "dem_mean_m": round(float(d_st.get("mean_h", 0.0)), 1),
                        "dem_max_m": round(float(d_st.get("max_h", 0.0)), 1),
                        "dem_mean_slope_deg": round(float(d_st.get("mean_slope", 0.0)), 1),
                    }
                    hs_u, hs_b = raster_to_map_overlay_png(
                        dem_deriv["hillshade"],
                        src_epsg=target_epsg,
                        is_rgb=False,
                        cmap_name="Greys_r",
                        vmin=0.0,
                        vmax=255.0,
                        max_width=1400,
                    )
                    extra_overlay_layers.append(
                        {"code": "DEM_HS", "name": "🏔️ Copernicus DEM Hillshade (30 m)", "image": hs_u, "bounds": hs_b}
                    )
                    if plan["index_name"] == "NDSI":
                        _, sn_dict = compute_snow_glacier_stats(ds_masked, dem_da=dem_sn, resolution_m=float(res_m))
                        snow_stats = {
                            k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                            for k, v in sn_dict.items()
                        }
                        if snow_stats.get("snowline_m") is not None:
                            synergy_metrics_sp["snowline_m"] = snow_stats["snowline_m"]
                except Exception:
                    snow_stats = None
        if sensor == SENSOR_S1:
            extra_overlay_layers = []
            synergy_metrics_sp = {}
            if str(plan.get("fusion_mode") or "") == FUSION_SYNERGY_SAR_OPTICAL:
                try:
                    ref_d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                except Exception:
                    ref_d = date.today()
                extra_overlay_layers, synergy_metrics_sp = _attach_sar_optical_synergy(
                    bbox_4326=bbox_4326,
                    ref_date=ref_d,
                    target_epsg=target_epsg,
                    lang=lang,
                )

        idx_key = str(plan["index_name"])
        spec_ai = (
            SAR_METRICS.get(idx_key, SAR_METRICS["VV"])
            if sensor == SENSOR_S1
            else SPECTRAL_INDICES.get(idx_key, SPECTRAL_INDICES["NDVI"])
        )
        analysis_uri, analysis_bnds = raster_to_map_overlay_png(
            idx_da,
            src_epsg=target_epsg,
            is_rgb=False,
            cmap_name=str(spec_ai["cmap"]),
            vmin=float(spec_ai["vmin"]),
            vmax=float(spec_ai["vmax"]),
            max_width=1800,
        )
        stats = compute_index_stats(idx_da, resolution_m=float(res_m))
        tif_bytes = export_to_geotiff_bytes(idx_da, target_epsg=target_epsg)
        tif_fname = f"AI_{plan['index_name']}_{plan['place_name']}_{date_str}_EPSG{epsg_num}_{res_m}m.tif"
        gc.collect()
        base_out.update(
            {
                "kind": "spectral_index",
                "index_da": idx_da,
                "stats": stats,
                "snow_stats": snow_stats,
                "date_str": date_str,
                "sat_date_str": date_str,
                "tile": tile_str,
                "sat_overlay_uri": sat_uri,
                "sat_overlay_bounds": sat_bnds,
                "analysis_overlay_uri": analysis_uri,
                "analysis_overlay_bounds": analysis_bnds,
                "extra_overlay_layers": extra_overlay_layers,
                "synergy_metrics": synergy_metrics_sp,
                "tif_bytes": tif_bytes,
                "tif_filename": tif_fname,
            }
        )
        base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
        base_out["analysis_context"] = build_analysis_context_dict(plan, base_out)
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
    sat_uri, sat_bnds = raster_to_map_overlay_png(
        ds_rgb,
        src_epsg=target_epsg,
        is_rgb=True,
        bands=rgb_vars,
        use_percentile_stretch=True,
        gamma=1.0,
        max_width=1800,
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
            "sat_date_str": date_str,
            "tile": tile_str,
            "sat_overlay_uri": sat_uri,
            "sat_overlay_bounds": sat_bnds,
            "tif_bytes": tif_bytes,
            "tif_filename": tif_fname,
        }
    )
    base_out["technical_summary"] = build_technical_summary(plan, base_out, lang=lang)
    base_out["analysis_context"] = build_analysis_context_dict(plan, base_out)
    _prog(1.0, "Zaključeno." if lang == "sl" else "Complete.")
    return base_out


# ---------------------------------------------------------------------------
# Multi-Turn Conversational EO Agent (Router + Rich Response Generator)
# ---------------------------------------------------------------------------
def _call_gemini_rest_with_fallback(
    payload: dict[str, Any],
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout: int = 30,
) -> tuple[str, str, str, str]:
    """Send a generateContent request to Gemini REST API with automatic model fallback cascade.

    Extracts and concatenates ALL non-thought text parts cleanly so Gemini 3.x thinking models
    never truncate or drop response fragments.

    Returns ``(response_text, used_model, fallback_note_sl, fallback_note_en)``.
    """
    req_model = resolve_gemini_model_id(model)
    candidates: list[str] = [req_model]
    for fb in FALLBACK_GEMINI_MODELS:
        if fb not in candidates:
            candidates.append(fb)

    first_status: int | None = None
    last_exc: Exception | None = None
    for idx, cand_model in enumerate(candidates):
        try:
            resp = requests.post(
                build_gemini_endpoint(cand_model),
                params={"key": api_key},
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            text = _extract_gemini_text(body)
            if not text:
                raise ValueError("Empty text response from Gemini candidate.")
            if idx == 0:
                return text, cand_model, "", ""
            code_txt = f" (HTTP {first_status})" if first_status else ""
            return (
                text,
                cand_model,
                f"Model `{req_model}` ni bil dosegljiv{code_txt}; samodejni preklop na `{cand_model}`.",
                f"Model `{req_model}` unavailable{code_txt}; automatically switched to `{cand_model}`.",
            )
        except Exception as exc:
            last_exc = exc
            if idx == 0:
                first_status = getattr(getattr(exc, "response", None), "status_code", None)
            continue

    raise RuntimeError(f"All Gemini models failed ({req_model}): {last_exc}")


_GEMINI_ROUTER_SYSTEM_PROMPT = """You are the intent router, geodetic Earth Observation planner, and QGIS-style Cartographic Composer controller for a Slovenian/English Remote Sensing Assistant.
Analyze the user's latest message together with recent chat history, last executed analysis context, and current map layout config, and choose ONE of 4 actions:
1. "run_analysis": User wants to fetch/compute a satellite, DEM terrain, or GNSS mission analysis (with enough details or after answering a clarification).
2. "generate_map": User wants to create, export, style, or iteratively modify a printable 300 DPI geodetic map / layout / PDF / PNG / poster from the active analysis or workspace layer (e.g. "Naredi karto iz teh rezultatov", "Izdelaj pokončno A4 karto s temno temo, severnico zgoraj levo, pregledno karto Slovenije in križci na 1 km", "Premakni legendo spodaj", "Dodaj naslov ...", "Skrij nogo in vklopi pregledno karto").
3. "ask_clarification": User wants to run an analysis, BUT critical details are missing or genuinely ambiguous (e.g., "Naredi analizo" with no location/topic, or "Primerjaj dve leti za Maribor" without specifying years or index). Do NOT guess blindly — ask a concise, friendly clarifying question with 2–3 concrete suggestions!
4. "chat_qa": User is asking a follow-up question about results, requesting an explanation/methodology/comparison of sensors, asking to draft a report paragraph, or having a general conversation.

Return STRICT JSON ONLY (no markdown fences) with this exact schema:
{
  "action": "run_analysis" | "generate_map" | "ask_clarification" | "chat_qa",
  "clarification_question": null OR string,
  "target_layer": "latest_ai" | "satellite_rgb" | "analysis_mask" | "workspace_layer",
  "reset_layout": false,
  "layout_overrides": null OR {
    // INCLUDE ONLY KEYS EXPLICITLY REQUESTED OR CHANGED BY THE USER (omit unchanged keys so previous tweaks are preserved!):
    "paper_format": "A4_LANDSCAPE" | "A4_PORTRAIT" | "A3_LANDSCAPE" | "SQUARE_1_1",
    "theme": "classic_light" | "minimal_white" | "dark_technical",
    "show_title": true | false,
    "title": string,
    "subtitle": string,
    "author": string,
    "title_position": "top_left" | "top_center" | "inside_top_left" | "inside_top_center" | "none",
    "show_footer": true | false,
    "custom_note_text": string,
    "custom_note_position": "bottom_left" | "bottom_right" | "below_legend" | "none",
    "show_grid": true | false,
    "grid_spacing": "auto" | 250 | 500 | 1000 | 2000 | 5000,
    "grid_style": "lines" | "crosses" | "ticks_only",
    "grid_color": "light" | "dark" | "teal",
    "grid_alpha": number (0.1 to 1.0),
    "show_aoi_outline": true | false,
    "aoi_outline_color": "#0F766E" | "#DC2626" | "#FFFFFF" | "#000000",
    "show_north_arrow": true | false,
    "north_position": "top_right" | "top_left" | "bottom_right" | "bottom_left",
    "north_style": "classic_compass" | "minimal_arrow" | "geodetic_star",
    "north_size": "small" | "medium" | "large",
    "show_scale_bar": true | false,
    "scale_position": "bottom_left" | "bottom_right" | "top_left" | "top_right",
    "scale_style": "alternating_bar" | "simple_line",
    "show_numeric_scale": true | false,
    "show_legend": true | false,
    "legend_position": "right_panel" | "bottom_panel" | "inside_bottom_right" | "inside_top_right" | "inside_bottom_left",
    "legend_title": string,
    "show_metadata_in_legend": true | false,
    "show_inset_map": true | false,
    "inset_position": "bottom_right" | "top_right" | "bottom_left" | "top_left" | "right_panel_bottom"
  },
  "plan": null OR {
    "place_name": string (unambiguous full place name including region/state and country, e.g. "SpaceX Starbase, Boca Chica, Texas, USA", "Črna na Koroškem, Slovenia", "Pasterze Glacier, Austria" — NEVER a bare ambiguous town name!),
    "location": string (same unambiguous full place name),
    "lat": number | null (exact WGS84 latitude whenever you know the landmark/place, e.g. 25.9972 for SpaceX Starbase, 46.0569 for Ljubljana),
    "lon": number | null (exact WGS84 longitude whenever you know the landmark/place, e.g. -97.1561 for SpaceX Starbase, 14.5058 for Ljubljana),
    "radius_m": integer (1000 to 10000, default 4000, use 5000-6000 for floods/wildfires),
    "sensor": "sentinel-2-l2a" | "landsat-c2-l2" | "sentinel-1-rtc",
    "sensors": ["sentinel-2-l2a" | "landsat-c2-l2" | "sentinel-1-rtc" | "cop-dem-glo-30"],
    "fusion_mode": "single" | "temporal_landsat_s2" | "synergy_sar_optical" | "synergy_optical_thermal" | "synergy_optical_dem",
    "fusion_rationale_sl": string (why these satellite systems were combined, empty if single),
    "fusion_rationale_en": string (why these satellite systems were combined, empty if single),
    "task_type": "scene_rgb" | "spectral_index" | "cloud_free_composite" | "time_series" | "time_series_and_comparison" | "change_comparison" | "disaster_flood" | "disaster_fire" | "disaster_forest" | "terrain_dem" | "gnss_plan",
    "include_comparison": boolean (true whenever user asks about spatial growth, expansion, shrinking, construction, deforestation, glacier retreat, or change over the years),
    "index_name": "NDVI" | "NDWI" | "MNDWI" | "NDBI" | "NBR" | "NDSI" | "LST" | "VV",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "event_date": "YYYY-MM-DD",
    "interval": "Vsi posnetki" | "Vsakih 14 dni" | "Mesečno" | "Sezonsko (3 mes.)" | "Letno",
    "resolution_m": 10 | 20 | 30 | 60,
    "explanation_sl": string (concise 1-2 sentence Slovenian plan note),
    "explanation_en": string (concise 1-2 sentence English plan note)
  }
}

ALL 11 SUPPORTED TASK TYPES FOR "run_analysis":
- "disaster_flood": Floods/inundation/radar -> sensor="sentinel-1-rtc", index_name="VV".
- "disaster_fire": Wildfire/burn severity -> index_name="NBR".
- "disaster_forest": Windthrow/clearcut/forest loss -> index_name="NDVI".
- "spectral_index": Single-date index ("NDVI", "NDWI", "MNDWI", "NDBI", "NBR", "NDSI" snow/glaciers + DEM snowline, "LST" surface temp °C from Landsat ST_B10, "VV").
- "scene_rgb": Single-date True Color RGB or SAR composite.
- "cloud_free_composite": Cloud-free temporal median composite over a period.
- "time_series": Multi-year or seasonal time series chart + trend (Sentinel-2 2017+ or Landsat 1984+).
- "time_series_and_comparison": Multi-year time series chart + trend PAIRED with an interactive zoomable Before/After swipe map (earliest t1 ↔ latest t2 in RGB & index) + Δ stats. ALWAYS use this for spatial growth, factory/urban expansion ("širila", "zrasla"), glacier retreat ("umikala"), deforestation, or "changes over the years"!
- "change_comparison": 2-date Before/After comparison with zoomable swipe map + Δ stats.
- "terrain_dem": Copernicus DEM 30 m elevation, slope [°], aspect [°], hillshade, elevation profile (resolution_m=30).
- "gnss_plan": GNSS Mission Planning (GPS/Galileo/GLONASS visibility, PDOP/HDOP/VDOP, DEM+S2 obstacle mask, optimal windows).

CARTOGRAPHY RULES FOR "generate_map":
- Use "generate_map" whenever the user asks to create a printable map/poster/PDF/PNG from existing results OR tweak map elements (paper format, theme, title, subtitle, author, note, footer, north arrow, scale bar, legend, grid lines/crosses/spacing, AOI outline, inset locator map).
- Put ONLY the requested/changed keys inside "layout_overrides" so iterative edits merge cleanly on top of previous settings!
- Set "target_layer": "satellite_rgb" if user explicitly asks for the RGB/satellite image map, "analysis_mask" if they ask for the flood/fire/index mask, "workspace_layer" if they refer to a layer loaded in Workspaces 1–3, else "latest_ai".
- Set "reset_layout": true ONLY if the user explicitly asks to reset layout to defaults.

SCIENTIFIC & GLOBAL GEOCODING RULES:
- ALWAYS provide accurate numeric "lat" and "lon" AND an unambiguous full "place_name"/"location" with region/country for specific landmarks/facilities (e.g. SpaceX Starbase Boca Chica Texas = 25.9972, -97.1561, NOT Boca Chica Dominican Republic!).
- LST -> sensor="landsat-c2-l2", resolution_m=30.
- start_date < 2016-01-01 -> sensor="landsat-c2-l2", resolution_m=30.
- Multi-year spatial growth/expansion/retreat -> task_type="time_series_and_comparison", include_comparison=true, interval="Letno" (NDBI for factories/urban expansion, NDSI for glaciers, NDVI for vegetation).
- Known Slovenian events: Koroška/Savinja/Celje/Kamnik/Škofja Loka floods = 2023-08-04, Barje floods = 2023-08-05, Kras wildfire = 2022-07-22.
- Normalize declined Slovenian places/regions to nominative geocodable names: "na Koroškem"/"Koroška" -> "Slovenj Gradec" (or "Črna na Koroškem"), "v Celju"/"v Savinjski dolini" -> "Celje", "na Krasu" -> "Kras", "na Gorenjskem" -> "Kranj", "v Prekmurju" -> "Murska Sobota".
- If user replies after a previous clarifying question, combine the previous user request + new answer into a complete "run_analysis" plan!"""


_VALID_TARGET_LAYERS: tuple[str, ...] = (
    "latest_ai",
    "satellite_rgb",
    "analysis_mask",
    "workspace_layer",
)


def sanitize_layout_overrides(raw_overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Filter and normalize ONLY explicitly provided ``MapLayoutConfig`` override keys."""
    if not isinstance(raw_overrides, dict):
        return {}
    clean: dict[str, Any] = {}
    valid_keys = set(DEFAULT_LAYOUT_CONFIG.keys())
    # Allow preset shorthand if passed
    preset_name = str(raw_overrides.get("preset") or "").strip().lower()
    if preset_name in LAYOUT_PRESETS:
        clean.update(LAYOUT_PRESETS[preset_name])

    for k, v in raw_overrides.items():
        if k not in valid_keys or v is None:
            continue
        if isinstance(v, str) and not v.strip() and k in (
            "paper_format",
            "theme",
            "title_position",
            "custom_note_position",
            "grid_style",
            "grid_color",
            "aoi_outline_color",
            "north_position",
            "north_style",
            "north_size",
            "scale_position",
            "scale_style",
            "legend_position",
            "inset_position",
        ):
            continue
        clean[k] = v

    if not clean:
        return {}

    # Validate each supplied key through merge_layout_config without injecting untouched defaults
    validated_full = merge_layout_config(clean)
    return {k: validated_full[k] for k in clean if k in validated_full}


def apply_incremental_layout_overrides(
    base_layout: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
    *,
    reset_layout: bool = False,
) -> dict[str, Any]:
    """Incrementally merge ``overrides`` onto ``base_layout`` preserving all untouched settings."""
    base = merge_layout_config() if (reset_layout or not isinstance(base_layout, dict)) else merge_layout_config(base_layout)
    clean_ov = sanitize_layout_overrides(overrides)
    if clean_ov:
        base.update(clean_ov)
    return merge_layout_config(base)


_MAP_GEN_PATTERNS: tuple[str, ...] = (
    "naredi kart",
    "izdelaj kart",
    "ustvari kart",
    "pripravi kart",
    "generiraj kart",
    "natisni kart",
    "izvozi kart",
    "karto za tisk",
    "karto iz ",
    "karto tega",
    "karto teh",
    "tiskaln",
    "pdf kart",
    "png kart",
    "pokoncno a4",
    "pokoncna a4",
    "lezeco a4",
    "lezeca a4",
    "a3 kart",
    "kvadratn",
    "poster",
    "premakni severnic",
    "premakni legend",
    "premakni meril",
    "premakni pregledn",
    "premakni naslov",
    "vklopi pregledn",
    "izklopi pregledn",
    "dodaj pregledn",
    "pregledno karto",
    "pregledna karta",
    "inset map",
    "overview map",
    "temno temo",
    "temna tema",
    "svetlo temo",
    "svetla tema",
    "minimalisticn",
    "dark theme",
    "minimal theme",
    "classic theme",
    "krizc",
    "samo oznak",
    "brez noge",
    "skrij nogo",
    "brez severnic",
    "skrij severnic",
    "brez legend",
    "skrij legend",
    "brez meril",
    "skrij meril",
    "brez mrez",
    "skrij mrez",
    "severnico ",
    "severnica ",
    "legendo ",
    "legenda ",
    "graficno meril",
    "make a map",
    "create a map",
    "generate a map",
    "generate map",
    "export map",
    "print map",
    "print layout",
    "move north",
    "move legend",
    "move scale",
    "portrait a4",
    "landscape a4",
    "a3 landscape",
)


def _detect_corner_in_span(span: str) -> str | None:
    """Detect a map corner ('top_left', 'top_right', 'bottom_left', 'bottom_right') in normalized text."""
    has_top = any(w in span for w in ("zgoraj", "zgornj", "top", "upper"))
    has_bot = any(w in span for w in ("spodaj", "spodnj", "bottom", "lower"))
    has_left = any(w in span for w in ("levo", "levi", "left"))
    has_right = any(w in span for w in ("desno", "desni", "right"))
    if has_top and has_left:
        return "top_left"
    if has_top and has_right:
        return "top_right"
    if has_bot and has_left:
        return "bottom_left"
    if has_bot and has_right:
        return "bottom_right"
    return None


def parse_map_layout_overrides_local(message: str) -> tuple[dict[str, Any], str, bool]:
    """Deterministic Slovenian/English parser mapping natural-language map instructions to ``MapLayoutConfig``."""
    raw = str(message or "").strip()
    norm = _normalize_ascii(raw)
    ov: dict[str, Any] = {}

    reset_layout = any(k in norm for k in ("ponastavi", "privzet", "reset layout", "na novo", "fresh layout"))

    # Target layer preference
    target_layer = "latest_ai"
    if any(k in norm for k in ("rgb", "satelitskega posnetka", "satelitski posnetek", "true color", "prave barve")):
        target_layer = "satellite_rgb"
    elif any(k in norm for k in ("mask", "poplavnega obseg", "poplavn", "pozar", "indeks", "ndvi", "ndwi", "ndsi", "lst", "dem")):
        target_layer = "analysis_mask"
    elif any(k in norm for k in ("delovnega prostora", "workspace", "tega sloja", "active layer")):
        target_layer = "workspace_layer"

    # Quick presets
    if any(k in norm for k in ("krizna karta", "krizno karto", "crisis preset", "crisis map")):
        ov.update(LAYOUT_PRESETS["crisis"])
    elif any(k in norm for k in ("cisti poster", "poster preset", "clean poster")):
        ov.update(LAYOUT_PRESETS["poster"])
    elif any(k in norm for k in ("magistrsk", "elaborat", "thesis preset")):
        ov.update(LAYOUT_PRESETS["thesis"])

    # 1. Paper format
    if any(k in norm for k in ("pokonc", "portrait")):
        ov["paper_format"] = "A4_PORTRAIT"
    elif "a3" in norm:
        ov["paper_format"] = "A3_LANDSCAPE"
    elif any(k in norm for k in ("kvadrat", "1:1", "square")):
        ov["paper_format"] = "SQUARE_1_1"
    elif any(k in norm for k in ("lezec", "landscape")):
        ov["paper_format"] = "A4_LANDSCAPE"

    # 2. Theme
    if any(k in norm for k in ("temn", "dark theme", "dark_technical", "temno temo", "temna tema")):
        ov["theme"] = "dark_technical"
    elif any(k in norm for k in ("minimal", "brez okvirj", "frameless")):
        ov["theme"] = "minimal_white"
    elif any(k in norm for k in ("svetl", "klasic", "classic_light", "light theme")):
        ov["theme"] = "classic_light"

    # 3. Title, subtitle, author, note, footer
    if any(k in norm for k in ("brez naslov", "skrij naslov", "no title", "hide title")):
        ov["show_title"] = False
        ov["title_position"] = "none"
    else:
        m_title = re.search(
            r"(?:z\s+naslovom|naslov\s*:|title\s*:?|naslovom)\s*[\"«'„]([^\"»'“]+)[\"»'“]",
            raw,
            flags=re.IGNORECASE,
        )
        if not m_title:
            m_title = re.search(
                r"(?:z\s+naslovom|naslov\s*:|title\s*:)\s*([^,;\n]+)",
                raw,
                flags=re.IGNORECASE,
            )
        if m_title:
            ov["show_title"] = True
            ov["title"] = m_title.group(1).strip().strip("\"'«»")
        if "naslov" in norm or "title" in norm:
            if any(k in norm for k in ("znotraj", "inside")):
                ov["title_position"] = "inside_top_center" if any(c in norm for c in ("sredin", "center")) else "inside_top_left"
            elif any(c in norm for c in ("sredin", "center")):
                ov["title_position"] = "top_center"

    m_sub = re.search(r"(?:podnaslov\s*:?|subtitle\s*:?)\s*[\"«'„]([^\"»'“]+)[\"»'“]", raw, flags=re.IGNORECASE)
    if m_sub:
        ov["subtitle"] = m_sub.group(1).strip()

    m_auth = re.search(r"(?:avtor\s*:?|author\s*:?)\s*[\"«'„]([^\"»'“]+)[\"»'“]", raw, flags=re.IGNORECASE)
    if m_auth:
        ov["author"] = m_auth.group(1).strip()

    m_note = re.search(r"(?:opomb[ao]?\s*:?|podpis\s*:?|note\s*:?|caption\s*:?)\s*[\"«'„]([^\"»'“]+)[\"»'“]", raw, flags=re.IGNORECASE)
    if m_note:
        ov["custom_note_text"] = m_note.group(1).strip()
        if "pod legend" in norm or "below legend" in norm:
            ov["custom_note_position"] = "below_legend"
        elif "spodaj levo" in norm or "bottom left" in norm:
            ov["custom_note_position"] = "bottom_left"
        else:
            ov["custom_note_position"] = "bottom_right"

    if any(k in norm for k in ("brez noge", "skrij nogo", "no footer", "hide footer")):
        ov["show_footer"] = False
    elif any(k in norm for k in ("vklopi nogo", "prikazi nogo", "show footer")):
        ov["show_footer"] = True

    # 4. North arrow
    if any(k in norm for k in ("brez severnic", "skrij severnic", "no north", "hide north")):
        ov["show_north_arrow"] = False
    else:
        m_n = re.search(r"(?:severnic\w*|north\s*arrow|north)(.{0,45})", norm)
        if m_n:
            ov["show_north_arrow"] = True
            span_n = m_n.group(0)
            c_n = _detect_corner_in_span(span_n)
            if c_n:
                ov["north_position"] = c_n
            if any(w in span_n for w in ("zvezd", "star", "kompas")):
                ov["north_style"] = "geodetic_star"
            elif "minimal" in span_n:
                ov["north_style"] = "minimal_arrow"
            elif "klasic" in span_n or "classic" in span_n:
                ov["north_style"] = "classic_compass"
            if any(w in span_n for w in ("velik", "large")):
                ov["north_size"] = "large"
            elif any(w in span_n for w in ("majhn", "small")):
                ov["north_size"] = "small"

    # 5. Scale bar
    if any(k in norm for k in ("brez meril", "skrij meril", "no scale", "hide scale")):
        ov["show_scale_bar"] = False
    else:
        m_s = re.search(r"(?:meril\w*|scale\s*bar|scale)(.{0,45})", norm)
        if m_s:
            ov["show_scale_bar"] = True
            span_s = m_s.group(0)
            c_s = _detect_corner_in_span(span_s)
            if c_s:
                ov["scale_position"] = c_s
            if any(w in span_s for w in ("crt", "line")):
                ov["scale_style"] = "simple_line"
            elif any(w in span_s for w in ("izmenic", "alternat")):
                ov["scale_style"] = "alternating_bar"
    if any(k in norm for k in ("brez numericnega", "no numeric scale")):
        ov["show_numeric_scale"] = False

    # 6. Legend
    if any(k in norm for k in ("brez legend", "skrij legend", "no legend", "hide legend")):
        ov["show_legend"] = False
    else:
        m_l = re.search(r"(?:legend\w*)(.{0,48})", norm)
        if m_l:
            ov["show_legend"] = True
            span_l = m_l.group(0)
            if any(w in span_l for w in ("znotraj", "inside")):
                c_l = _detect_corner_in_span(span_l)
                if c_l == "top_right":
                    ov["legend_position"] = "inside_top_right"
                elif c_l == "bottom_left":
                    ov["legend_position"] = "inside_bottom_left"
                else:
                    ov["legend_position"] = "inside_bottom_right"
            elif any(w in span_l for w in ("spodaj", "spodnj", "bottom")):
                ov["legend_position"] = "bottom_panel"
            elif any(w in span_l for w in ("desn", "right")):
                ov["legend_position"] = "right_panel"

    # 7. Inset / Overview locator map
    if any(k in norm for k in ("brez pregledn", "skrij pregledn", "izklopi pregledn", "no inset", "hide inset")):
        ov["show_inset_map"] = False
    else:
        m_in = re.search(r"(?:pregledn\w*(?:\s+kart\w*)?(?:\s+slovenij\w*)?|inset\s*map|inset|overview\s*map|lokacijsk\w*\s+kart\w*)(.{0,40})", norm)
        if m_in:
            ov["show_inset_map"] = True
            span_in = m_in.group(0)
            if any(w in span_in for w in ("desni panel", "right panel")):
                ov["inset_position"] = "right_panel_bottom"
            else:
                c_in = _detect_corner_in_span(span_in)
                if c_in:
                    ov["inset_position"] = c_in

    # 8. Coordinate grid & AOI outline
    if any(k in norm for k in ("brez mrez", "skrij mrez", "izklopi mrez", "no grid", "hide grid")):
        ov["show_grid"] = False
    else:
        if any(k in norm for k in ("krizc", "cross")):
            ov["show_grid"] = True
            ov["grid_style"] = "crosses"
        elif any(k in norm for k in ("samo oznak", "ticks only", "ticks_only")):
            ov["show_grid"] = True
            ov["grid_style"] = "ticks_only"
        elif re.search(r"(?:mrez\w*|grid).{0,24}(?:crt|line)", norm):
            ov["show_grid"] = True
            ov["grid_style"] = "lines"

        # Grid spacing (250 m, 500 m, 1 km / 1000 m, 2 km / 2000 m, 5 km / 5000 m)
        m_gsp = re.search(r"(?:na\s+|razmak\s+|mrez\w*[^0-9]{0,16}|krizc\w*[^0-9]{0,16}|grid[^0-9]{0,16})(250|500|1000|2000|5000|1|2|5)\s*(km|m)\b", norm)
        if m_gsp:
            num_v = int(m_gsp.group(1))
            unit_v = m_gsp.group(2)
            meters = num_v * 1000 if unit_v == "km" else num_v
            if meters in (250, 500, 1000, 2000, 5000):
                ov["show_grid"] = True
                ov["grid_spacing"] = meters

        m_gcol = re.search(r"(?:mrez\w*|grid|krizc\w*)(.{0,30})", norm)
        if m_gcol:
            sg = m_gcol.group(0)
            if "teal" in sg or "zelen" in sg:
                ov["grid_color"] = "teal"
            elif "temn" in sg or "dark" in sg or "crn" in sg:
                ov["grid_color"] = "dark"
            elif "svetl" in sg or "bel" in sg or "light" in sg:
                ov["grid_color"] = "light"

    if any(k in norm for k in ("brez aoi", "skrij aoi", "brez obrob", "hide aoi", "no aoi")):
        ov["show_aoi_outline"] = False
    elif any(k in norm for k in ("rdeca obrob", "rdeco obrob", "red outline")):
        ov["show_aoi_outline"] = True
        ov["aoi_outline_color"] = "#DC2626"
    elif any(k in norm for k in ("bela obrob", "belo obrob", "white outline")):
        ov["show_aoi_outline"] = True
        ov["aoi_outline_color"] = "#FFFFFF"
    elif any(k in norm for k in ("crna obrob", "crno obrob", "black outline")):
        ov["show_aoi_outline"] = True
        ov["aoi_outline_color"] = "#000000"

    return sanitize_layout_overrides(ov), target_layer, reset_layout


_QA_PATTERNS: tuple[str, ...] = (
    "kaj pomeni",
    "kaj je ",
    "kaj so ",
    "zakaj",
    "kako deluje",
    "kako se izracun",
    "kako deluj",
    "razlozi",
    "pojasni",
    "ali je bolje",
    "bi bilo bolje",
    "kaksna je razlika",
    "razlika med",
    "napisi odstavek",
    "napisi povzetek",
    "napisi besedilo",
    "za porocilo",
    "za seminarsko",
    "za diplomo",
    "interpretiraj",
    "komentiraj",
    "ta rezultat",
    "te stevilke",
    "tega rezultata",
    "what does",
    "what is ",
    "why ",
    "how does",
    "how is ",
    "explain",
    "interpret",
    "summarize",
    "write a paragraph",
    "for my report",
    "would it be better",
    "difference between",
    "zivijo",
    "pozdravljen",
    "dober dan",
    "hvala",
    "hello",
    "hi ",
    "thanks",
    "thank you",
)

_RUN_ACTION_PATTERNS: tuple[str, ...] = (
    "pokazi",
    "prikazi",
    "analiziraj",
    "izracunaj",
    "naredi",
    "prenesi",
    "poisci",
    "oceni",
    "izdelaj",
    "narisi",
    "zazeni",
    "nacrtuj",
    "primerjaj",
    "show ",
    "analyze",
    "analyse",
    "compute",
    "calculate",
    "run ",
    "fetch",
    "plot ",
    "load ",
    "map ",
    "plan ",
    "compare",
)

_EO_SUBJECT_PATTERNS: tuple[str, ...] = (
    "poplav",
    "flood",
    "pozar",
    "wildfire",
    "fire",
    "vetrolom",
    "posek",
    "windthrow",
    "clearcut",
    "ndvi",
    "ndwi",
    "mndwi",
    "ndbi",
    "nbr",
    "dnbr",
    "dndvi",
    "ndsi",
    "lst",
    "sneg",
    "ledenik",
    "snow",
    "glacier",
    "temperat",
    "vrocin",
    "toplot",
    "casovn",
    "time series",
    "rgb",
    "posnetek",
    "vegetacij",
    "kompozit",
    "mozaik",
    "brezoblac",
    "composite",
    "dem",
    "dmv",
    "relief",
    "naklon",
    "visinsk",
    "hillshade",
    "slope",
    "terrain",
    "elevation",
    "gnss",
    "gps",
    "galileo",
    "glonass",
    "pdop",
    "skyplot",
    "izmer",
)


def _route_user_message_local(
    message: str,
    chat_history: Sequence[dict[str, Any]] | None = None,
    last_context: dict[str, Any] | None = None,
    lang: str = "sl",
) -> dict[str, Any]:
    """Deterministic offline intent router separating 'generate_map', 'run_analysis', 'ask_clarification', and 'chat_qa'."""
    raw_msg = str(message or "").strip()
    raw_norm = _normalize_ascii(raw_msg)
    raw_is_pure_qa = (
        (any(p in raw_norm for p in _QA_PATTERNS) or raw_norm.endswith("?"))
        and not any(p in raw_norm for p in _RUN_ACTION_PATTERNS)
        and not any(p in raw_norm for p in _MAP_GEN_PATTERNS)
    )
    effective_msg = raw_msg

    # If the previous assistant turn asked a clarifying question (and the user isn't asking a pure QA question),
    # combine the user's answer with the prior user turn
    hist = list(chat_history or [])
    if (
        not raw_is_pure_qa
        and len(hist) >= 2
        and hist[-1].get("role") == "assistant"
        and hist[-1].get("action") == "ask_clarification"
    ):
        prev_user = next((str(m.get("content", "")) for m in reversed(hist[:-1]) if m.get("role") == "user"), "")
        if prev_user:
            effective_msg = f"{prev_user} {effective_msg}".strip()

    norm_q = _normalize_ascii(effective_msg)
    has_qa_cue = any(p in norm_q for p in _QA_PATTERNS) or norm_q.endswith("?")
    has_run_verb = any(p in norm_q for p in _RUN_ACTION_PATTERNS)
    has_eo_subject = any(p in norm_q for p in _EO_SUBJECT_PATTERNS) or any(
        c in norm_q for c in _EXPANSION_COMPARISON_CUES
    )
    years_in_q = re.findall(r"\b(19[89]\d|20[0-2]\d)\b", norm_q)
    has_year_or_month = bool(years_in_q) or any(
        pat in norm_q for pat, _ in _MONTH_PATTERNS
    ) or any(s in norm_q for s in ("polet", "pomlad", "jesen", "zim", "summer", "winter", "spring", "autumn"))
    has_known_place = (
        any(stem in norm_q for stem, _, _, _ in _SLOVENIAN_PLACE_STEMS)
        or (_match_known_global_landmark(norm_q) is not None)
    )
    has_prep_place = bool(
        re.search(
            r"\b(?:za|v|na|okoli|pri|in|for|around|near|over)\s+([A-ZČŠŽ][a-zčšž]+)",
            effective_msg,
        )
    )
    has_place = has_known_place or has_prep_place
    refers_same_area = any(
        p in norm_q for p in ("isto obmocj", "isti lokacij", "tam", "same area", "same location")
    )

    # 0. Printable Cartographic Map Creation / Iterative Layout Editing -> generate_map
    map_overrides, map_target_layer, map_reset = parse_map_layout_overrides_local(raw_msg)
    has_map_cue = any(p in raw_norm for p in _MAP_GEN_PATTERNS) or (
        bool(map_overrides) and not (has_eo_subject and (has_year_or_month or has_known_place))
    )
    if has_map_cue and not raw_is_pure_qa:
        return {
            "action": "generate_map",
            "intent": "generate_map",
            "plan": None,
            "layout_overrides": map_overrides,
            "target_layer": map_target_layer,
            "reset_layout": map_reset,
            "clarification_question": None,
            "engine": "local",
            "gemini_model": None,
            "fallback_note_sl": "",
            "fallback_note_en": "",
        }

    # 1. Pure explanation / follow-up question / writing request -> chat_qa
    if has_qa_cue and not (has_run_verb and has_eo_subject):
        return {
            "action": "chat_qa",
            "intent": "chat_qa",
            "plan": None,
            "layout_overrides": None,
            "target_layer": "latest_ai",
            "reset_layout": False,
            "clarification_question": None,
            "engine": "local",
            "gemini_model": None,
            "fallback_note_sl": "",
            "fallback_note_en": "",
        }

    # 2. Ambiguous / underspecified analysis requests -> ask_clarification (DO NOT guess blindly!)
    is_vague_generic = (
        has_run_verb
        and not has_eo_subject
        and not has_place
        and not refers_same_area
    ) or (
        any(k in norm_q for k in ("naredi analiz", "izvedi analiz", "run analysis", "do an analysis"))
        and not has_eo_subject
        and not has_place
    )
    is_vague_compare = (
        any(k in norm_q for k in ("primerjaj", "compare"))
        and len(years_in_q) < 2
        and not any(k in norm_q for k in ("poplav", "pozar", "flood", "fire"))
        and not any(k in norm_q for k in ("ndvi", "ndwi", "ndsi", "lst", "rgb", "nbr", "mndwi", "ndbi", "vv"))
    )
    if is_vague_generic:
        q_clar = (
            "Z veseljem pripravim analizo! Prosim, povejte mi še **lokacijo** in **vrsto analize**, npr.:\n"
            "- 🌊 *»Pokaži poplave na Koroškem leta 2023«* (Sentinel-1 SAR)\n"
            "- 🌿 *»Izračunaj NDVI za Bled poleti 2025«* (Sentinel-2 10 m)\n"
            "- ⛰️ *»Analiza reliefa (DEM) za Triglav«* ali 📡 *»Načrtuj GNSS izmero za Ljubljano«*"
            if lang == "sl"
            else (
                "I'd be happy to run an analysis! Could you specify the **location** and **analysis type**? For example:\n"
                "- 🌊 *\"Show floods in Celje in August 2023\"* (Sentinel-1 SAR)\n"
                "- 🌿 *\"Compute NDVI for Bled in summer 2025\"* (Sentinel-2 10 m)\n"
                "- ⛰️ *\"Terrain DEM analysis for Triglav\"* or 📡 *\"Plan GNSS survey for Ljubljana\"*"
            )
        )
        return {
            "action": "ask_clarification",
            "intent": "ask_clarification",
            "plan": None,
            "layout_overrides": None,
            "target_layer": "latest_ai",
            "reset_layout": False,
            "clarification_question": q_clar,
            "engine": "local",
            "gemini_model": None,
            "fallback_note_sl": "",
            "fallback_note_en": "",
        }
    if is_vague_compare:
        loc_hint = _extract_place_local(effective_msg, norm_q) if has_place else ("izbrano območje" if lang == "sl" else "your location")
        q_clar = (
            f"Za primerjavo območja **{loc_hint}** mi prosim sporočite **kateri dve leti (ali obdobji)** in **kateri sloj** naj primerjam:\n"
            f"- 🌿 *»Primerjaj NDVI za {loc_hint} med 2019 in 2025«*\n"
            f"- 🛰️ *»Primerjaj RGB za {loc_hint} med 2020 in 2024«*\n"
            f"- 🌡️ *»Primerjaj temperaturo LST za {loc_hint} med 2015 in 2024«*"
            if lang == "sl"
            else (
                f"To compare **{loc_hint}**, could you specify **which two years/dates** and **which layer** you'd like?\n"
                f"- 🌿 *\"Compare NDVI for {loc_hint} between 2019 and 2025\"*\n"
                f"- 🛰️ *\"Compare RGB for {loc_hint} between 2020 and 2024\"*"
            )
        )
        return {
            "action": "ask_clarification",
            "intent": "ask_clarification",
            "plan": None,
            "layout_overrides": None,
            "target_layer": "latest_ai",
            "reset_layout": False,
            "clarification_question": q_clar,
            "engine": "local",
            "gemini_model": None,
            "fallback_note_sl": "",
            "fallback_note_en": "",
        }

    # 3. Explicit spatial/temporal/DEM/GNSS execution request -> run_analysis
    should_run = (
        (has_run_verb and (has_eo_subject or has_place or refers_same_area))
        or (has_eo_subject and (has_place or has_year_or_month or refers_same_area))
    )
    if should_run:
        plan = parse_eo_request_local(effective_msg)
        if (
            refers_same_area
            and not has_place
            and isinstance(last_context, dict)
            and last_context.get("place_name")
        ):
            plan["place_name"] = str(last_context["place_name"])
            if last_context.get("radius_m"):
                plan["radius_m"] = int(last_context["radius_m"])
        return {
            "action": "run_analysis",
            "intent": "run_analysis",
            "plan": plan,
            "fusion_mode": plan.get("fusion_mode", FUSION_SINGLE),
            "sensors": list(plan.get("sensors") or [plan.get("sensor", SENSOR_S2)]),
            "fusion_rationale": plan.get("fusion_rationale_en" if lang == "en" else "fusion_rationale_sl", ""),
            "layout_overrides": None,
            "target_layer": "latest_ai",
            "reset_layout": False,
            "clarification_question": None,
            "engine": "local",
            "gemini_model": None,
            "fallback_note_sl": "",
            "fallback_note_en": "",
        }

    return {
        "action": "chat_qa",
        "intent": "chat_qa",
        "plan": None,
        "layout_overrides": None,
        "target_layer": "latest_ai",
        "reset_layout": False,
        "clarification_question": None,
        "engine": "local",
        "gemini_model": None,
        "fallback_note_sl": "",
        "fallback_note_en": "",
    }


def route_user_message(
    message: str,
    chat_history: Sequence[dict[str, Any]] | None = None,
    last_context: dict[str, Any] | None = None,
    *,
    last_map_layout: dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    lang: str = "sl",
) -> dict[str, Any]:
    """Route a user chat message into ``'run_analysis'``, ``'generate_map'``, ``'ask_clarification'``, or ``'chat_qa'``."""
    resolved_key, _ = get_active_gemini_api_key(api_key)
    if resolved_key:
        history_lines: list[str] = []
        for turn in list(chat_history or [])[-6:]:
            role_lbl = "User" if turn.get("role") == "user" else "Assistant"
            txt = str(turn.get("content", "")).strip()
            if txt:
                history_lines.append(f"{role_lbl}: {txt[:500]}")
        ctx_snippet = (
            json.dumps(
                {k: v for k, v in (last_context or {}).items() if k != "technical_summary"},
                ensure_ascii=False,
            )
            if last_context
            else "null"
        )
        map_snippet = json.dumps(last_map_layout, ensure_ascii=False) if isinstance(last_map_layout, dict) else "null"
        lang_hint = "Slovenian" if lang == "sl" else "English"
        user_blob = (
            f"Today is {date.today().isoformat()}. Preferred language: {lang_hint}.\n"
            f"Last executed analysis context: {ctx_snippet}\n"
            f"Current active MapLayoutConfig: {map_snippet}\n"
            f"Recent chat history:\n" + ("\n".join(history_lines) if history_lines else "(empty)") + "\n\n"
            f"Latest user message: {message}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": _GEMINI_ROUTER_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_blob}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
                "maxOutputTokens": 2048,
            },
        }
        try:
            raw_text, used_model, fb_sl, fb_en = _call_gemini_rest_with_fallback(
                payload, api_key=resolved_key, model=model, timeout=22
            )
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.DOTALL).strip()
            parsed = json.loads(cleaned)
            action = str(parsed.get("action") or parsed.get("intent") or "chat_qa").strip().lower()
            if action == "generate_map":
                raw_ov = parsed.get("layout_overrides") if isinstance(parsed.get("layout_overrides"), dict) else {}
                clean_ov = sanitize_layout_overrides(raw_ov)
                # Fill any subtle local cues if Gemini omitted them
                t_layer = str(parsed.get("target_layer") or "latest_ai").strip().lower()
                if t_layer not in _VALID_TARGET_LAYERS:
                    t_layer = "latest_ai"
                return {
                    "action": "generate_map",
                    "intent": "generate_map",
                    "plan": None,
                    "layout_overrides": clean_ov,
                    "target_layer": t_layer,
                    "reset_layout": bool(parsed.get("reset_layout", False)),
                    "clarification_question": None,
                    "engine": "gemini",
                    "gemini_model": used_model,
                    "fallback_note_sl": fb_sl,
                    "fallback_note_en": fb_en,
                }
            if action in ("run_analysis", "time_series_and_comparison") and isinstance(parsed.get("plan"), dict):
                raw_pl = dict(parsed["plan"])
                norm_m = _normalize_ascii(message)
                if action == "time_series_and_comparison" or (
                    str(raw_pl.get("task_type") or "").strip().lower() == "time_series"
                    and any(c in norm_m for c in _EXPANSION_COMPARISON_CUES)
                ):
                    raw_pl["task_type"] = "time_series_and_comparison"
                    raw_pl["include_comparison"] = True
                plan = validate_and_normalize_plan(raw_pl, engine="gemini", query=message)
                plan["gemini_model"] = used_model
                plan["fallback_note_sl"] = fb_sl
                plan["fallback_note_en"] = fb_en
                return {
                    "action": "run_analysis",
                    "intent": "run_analysis",
                    "plan": plan,
                    "fusion_mode": plan.get("fusion_mode", FUSION_SINGLE),
                    "sensors": list(plan.get("sensors") or [plan.get("sensor", SENSOR_S2)]),
                    "fusion_rationale": plan.get("fusion_rationale_en" if lang == "en" else "fusion_rationale_sl", ""),
                    "layout_overrides": None,
                    "target_layer": "latest_ai",
                    "reset_layout": False,
                    "clarification_question": None,
                    "engine": "gemini",
                    "gemini_model": used_model,
                    "fallback_note_sl": fb_sl,
                    "fallback_note_en": fb_en,
                }
            if action == "ask_clarification":
                clar_q = str(parsed.get("clarification_question") or "").strip()
                if clar_q:
                    return {
                        "action": "ask_clarification",
                        "intent": "ask_clarification",
                        "plan": None,
                        "layout_overrides": None,
                        "target_layer": "latest_ai",
                        "reset_layout": False,
                        "clarification_question": clar_q,
                        "engine": "gemini",
                        "gemini_model": used_model,
                        "fallback_note_sl": fb_sl,
                        "fallback_note_en": fb_en,
                    }
            return {
                "action": "chat_qa",
                "intent": "chat_qa",
                "plan": None,
                "layout_overrides": None,
                "target_layer": "latest_ai",
                "reset_layout": False,
                "clarification_question": None,
                "engine": "gemini",
                "gemini_model": used_model,
                "fallback_note_sl": fb_sl,
                "fallback_note_en": fb_en,
            }
        except Exception:
            pass

    return _route_user_message_local(
        message,
        chat_history=chat_history,
        last_context=last_context,
        lang=lang,
    )


def _generate_local_conversational_reply(
    user_message: str,
    analysis_context: dict[str, Any] | None,
    lang: str = "sl",
    is_post_analysis: bool = False,
) -> str:
    """Generate a comprehensive, domain-aware Slovenian/English geodetic report without an external API."""
    if is_post_analysis and isinstance(analysis_context, dict):
        kind = str(analysis_context.get("kind", ""))
        place = str(analysis_context.get("resolved_place") or analysis_context.get("place_name") or "")
        crs = str(analysis_context.get("crs", CRS_D96TM))
        sensor = str(analysis_context.get("sensor", ""))
        res_m = analysis_context.get("resolution_m", 10)
        idx = str(analysis_context.get("index_name", ""))

        if kind == "disaster":
            d_type = str(analysis_context.get("disaster_type", ""))
            d_pre = str(analysis_context.get("date_pre", ""))
            d_ev = str(analysis_context.get("date_event", ""))
            ha = float(analysis_context.get("affected_area_ha", 0.0))
            km2 = float(analysis_context.get("affected_area_km2", 0.0))
            pct = float(analysis_context.get("affected_pct", 0.0))
            if lang == "sl":
                if d_type == DISASTER_FLOOD:
                    bg = (
                        "Za zaznavo poplav je bil uporabljen radarski senzor **Sentinel-1 SAR RTC (C-pas)**, "
                        "ki prodira skozi nevihtne oblake in padavine ter z nizkim povratnim sipanjem na mirni vodni gladini "
                        "zanesljivo loči novo poplavljene površine od stalnih vodotokov."
                    )
                    caveat = (
                        "**Opozorilo glede interpretacije:** V strmih alpskih in predalpskih dolinah (npr. Koroška / Mežiška dolina) "
                        "lahko radarska geometrija (senčenje in skrajšanje pobočij — *layover/shadow*) ter gosta urbana pozidava "
                        "delno podcenita dejanski obseg poplave v ozkih soteskah."
                    )
                    extra = f" Stalna voda pred dogodkom je znašala **{analysis_context.get('permanent_water_ha', 0.0):,.2f} ha**."
                    sugg = (
                        "- *»Kaj pomeni razlika med stalno vodo in novo poplavljeno površino na tej karti?«*\n"
                        "- *»Pripravi analizo reliefa (Copernicus DEM) za isto območje, da ocenimo strmino pobočij.«*\n"
                        "- *»Napiši kratek strokovni odstavek o tem rezultatu za moje poročilo.«*"
                    )
                elif d_type == DISASTER_WILDFIRE:
                    bg = (
                        "Analiza požarišča temelji na razliki spektralnega indeksa **dNBR (NBR_pred − NBR_po)**, "
                        "ki izkorišča močan padec odbojnosti v bližnjem infrardečem pasu (NIR) in dvig v kratkovalovnem infrardečem pasu (SWIR) na ožganem terenu, "
                        "klasificirano po uradni **USGS lestvici stopnje poškodovanosti**."
                    )
                    caveat = "**Opozorilo glede interpretacije:** Manjši lokalni žepi dima ali senc lahko lokalno vplivajo na razred nizke poškodovanosti."
                    extra = ""
                    sugg = (
                        "- *»Zakaj je toliko površine v razredu visoke ali zmerne poškodovanosti?«*\n"
                        "- *»Pokaži časovno vrsto NDVI za isto območje (2021–2025), da vidimo obnovo gozda.«*"
                    )
                else:
                    bg = (
                        "Analiza poškodovanosti gozda temelji na razliki **ΔNDVI (NDVI_po − NDVI_pred)**, "
                        "ki zazna nenaden padec klorofilne aktivnosti pri vetrolomu, sečnji, lubadarju ali hudi suši."
                    )
                    caveat = "**Opozorilo glede interpretacije:** Sezonske razlike v fenologiji listavcev lahko delno vplivajo na zmerne padce ΔNDVI."
                    extra = ""
                    sugg = (
                        "- *»Razloži pragove ΔNDVI za zmerno in močno izgubo vegetacije.«*\n"
                        "- *»Izračunaj še NBR za isto območje.«*"
                    )
                return (
                    f"### Rezultati analize za območje **{place}**\n\n"
                    f"**Ozadje in metodologija:** {bg}\n\n"
                    f"**Podatki in projekcija:** `{sensor}` (**{res_m} m**), primerjava **PRED `{d_pre}`** → **MED/PO `{d_ev}`** v metričnem koordinatnem sistemu **{crs}**.\n\n"
                    f"**Podrobna interpretacija rezultatov:**\n"
                    f"- **Ocenjena prizadeta površina:** **{ha:,.2f} ha** (**{km2:.3f} km²**, oziroma **{pct:.2f} %** analiziranega območja).{extra}\n"
                    f"- {caveat}\n"
                    f"- Na interaktivni karti spodaj sta v legendi slojev na voljo **oba sloja** — dejanski satelitski posnetek in maska prizadetosti.\n\n"
                    f"**Predlogi za nadaljevanje:**\n{sugg}"
                )
            else:
                bg = (
                    "All-weather **Sentinel-1 SAR RTC** radar penetrates storm clouds to separate newly inundated land from permanent water."
                    if d_type == DISASTER_FLOOD
                    else "Cloud-masked optical change detection (**dNBR / ΔNDVI**) quantifies ground severity classes."
                )
                return (
                    f"### Analysis Results for **{place}**\n\n"
                    f"**Background & Methodology:** {bg}\n\n"
                    f"- **Data & CRS:** `{sensor}` ({res_m} m), **PRE `{d_pre}`** → **EVENT/POST `{d_ev}`**, projected in **{crs}**.\n"
                    f"- **Total Affected Area:** **{ha:,.2f} ha** (**{km2:.3f} km²**, **{pct:.2f}%** of the AOI).\n"
                    f"- **Map tip:** Use the top-right Layer Control to toggle both the **analyzed satellite scene** and the **impact mask** independently.\n\n"
                    f"**Suggested follow-ups:**\n"
                    f"- *\"Explain the severity classes and sensor limitations in steep terrain.\"*\n"
                    f"- *\"Draft a concise report paragraph summarizing these numbers.\"*"
                )

        if kind == "terrain_dem":
            ds = analysis_context.get("dem_stats") or {}
            if lang == "sl":
                return (
                    f"### Analiza reliefa (Copernicus DEM 30 m) — **{place}**\n\n"
                    f"**Metodologija in podatki:** Uporabljen je digitalni model višin **Copernicus GLO-30 (30 m)** v metrični projekciji **{crs}** za izračun nadmorskih višin, naklonov, ekspozicije, analitičnega senčenja in vzdolžnega profila terena.\n\n"
                    f"**Podrobna interpretacija rezultatov:**\n"
                    f"- **Nadmorska višina:** min **`{ds.get('min_h', 0):.1f} m`**, povprečje **`{ds.get('mean_h', 0):.1f} m`**, max **`{ds.get('max_h', 0):.1f} m`** (relativna višinska razlika **`{ds.get('relief_diff_m', 0):.1f} m`**).\n"
                    f"- **Naklon pobočij:** povprečno **`{ds.get('mean_slope_deg', 0):.2f}°`**, največji naklon **`{ds.get('max_slope_deg', 0):.2f}°`**.\n"
                    f"- **Strma pobočja (> 20°):** **{ds.get('steep_20_ha', 0):,.2f} ha** (**{ds.get('steep_20_pct', 0):.1f} %** AOI); zelo strma (> 30°): **{ds.get('steep_30_ha', 0):,.2f} ha**.\n\n"
                    f"**Predlogi za nadaljevanje:**\n"
                    f"- *»Načrtuj GNSS izmero za {place}, da preverimo vpliv tega reliefa na vidnost satelitov.«*\n"
                    f"- *»Izračunaj še NDVI ali NDSI (sneg) za isto območje.«*"
                )
            return (
                f"### Terrain Analysis (Copernicus DEM 30 m) — **{place}**\n\n"
                f"- **Elevation:** min `{ds.get('min_h', 0):.1f} m`, mean `{ds.get('mean_h', 0):.1f} m`, max `{ds.get('max_h', 0):.1f} m` (relief `{ds.get('relief_diff_m', 0):.1f} m`).\n"
                f"- **Slope:** mean `{ds.get('mean_slope_deg', 0):.2f}°`, max `{ds.get('max_slope_deg', 0):.2f}°` | Steep >20°: **{ds.get('steep_20_ha', 0):,.2f} ha** ({ds.get('steep_20_pct', 0):.1f}%)."
            )

        if kind == "gnss_plan":
            gs = analysis_context.get("gnss_summary") or {}
            wins = ", ".join(gs.get("optimal_windows") or []) or ("ni optimalnega okna" if lang == "sl" else "none")
            if lang == "sl":
                return (
                    f"### Načrtovanje 24-urne GNSS izmere — **{place}** (`{analysis_context.get('event_date', '')}`)\n\n"
                    f"**Metodologija in podatki:** Izračun 24-urne geometrije satelitov **GPS + Galileo + GLONASS** z upoštevanjem realne **360° maske ovir** iz **Copernicus DEM 30 m** (teren v 2,5 km) in zadnjega **Sentinel-2 10 m** (gozd in stavbe v 150 m) v projekciji **{crs}**.\n\n"
                    f"**Podrobna interpretacija rezultatov:**\n"
                    f"- **Povprečno število vidnih satelitov:** **`{gs.get('avg_vis_real', 0):.1f}`** (z ovirami) proti **`{gs.get('avg_vis_open', 0):.1f}`** (odprto nebo).\n"
                    f"- **Geometrija PDOP:** minimalni PDOP **`{gs.get('min_pdop_real', 0):.2f}`**, povprečni PDOP **`{gs.get('mean_pdop_real', 0):.2f}`**.\n"
                    f"- **Zastiranje neba zaradi reliefa in ovir:** **`{gs.get('sky_blocked_pct', 0):.1f} %`**.\n"
                    f"- **Optimalna okna za RTK / statično izmero (PDOP ≤ 2,5 in ≥ 6 sat.):** **{wins}**.\n\n"
                    f"**Predlogi za nadaljevanje:**\n"
                    f"- *»Razloži, kaj pomeni PDOP in zakaj želimo vrednost pod 2,5.«*\n"
                    f"- *»Pokaži analizo reliefa (DEM) za isto območje.«*"
                )
            return (
                f"### 24-Hour GNSS Mission Plan — **{place}** (`{analysis_context.get('event_date', '')}`)\n\n"
                f"- **Avg Visible Satellites:** `{gs.get('avg_vis_real', 0):.1f}` (obstructed) vs `{gs.get('avg_vis_open', 0):.1f}` (open sky).\n"
                f"- **PDOP:** min `{gs.get('min_pdop_real', 0):.2f}`, mean `{gs.get('mean_pdop_real', 0):.2f}` | **Sky Blocked:** `{gs.get('sky_blocked_pct', 0):.1f}%`.\n"
                f"- **Optimal Windows (PDOP ≤ 2.5):** **{wins}**."
            )

        if kind == "cloud_free_composite":
            cm = analysis_context.get("composite_meta") or {}
            if lang == "sl":
                return (
                    f"### Brezoblačni mediani mozaik — **{place}** (`{cm.get('date_start', '')}` – `{cm.get('date_end', '')}`)\n\n"
                    f"Združenih je bilo **{cm.get('scenes_used', 0)} najčistejših posnetkov** (`{sensor}`, **{res_m} m**, **{crs}**) z oblačnim maskiranjem in slikovnim medianiranjem po pikslih.\n\n"
                    f"- **Veljavna pokritost brez oblakov:** **{cm.get('valid_area_ha', 0):,.2f} ha** (**{cm.get('valid_pct', 0):.1f} %** AOI, mediana **{cm.get('median_obs_per_px', 0):.1f}** jasnih opazovanj na piksel).\n\n"
                    f"**Predlogi za nadaljevanje:**\n"
                    f"- *»Izračunaj še NDVI ali NDBI za isto območje.«*\n"
                    f"- *»Pošlji ta brezoblačni mozaik v Izdelavo karte za 300 DPI PDF izvoz.«*"
                )
            return (
                f"### Cloud-Free Median Composite — **{place}** (`{cm.get('date_start', '')}` – `{cm.get('date_end', '')}`)\n\n"
                f"Combined **{cm.get('scenes_used', 0)} clearest scenes** (`{sensor}`, {res_m} m, {crs}) — valid cloud-free area **{cm.get('valid_area_ha', 0):,.2f} ha** ({cm.get('valid_pct', 0):.1f}%)."
            )

        if kind == "change_comparison":
            c_dates = analysis_context.get("dates") or []
            csum = analysis_context.get("change_summary") or {}
            d1_s = c_dates[0] if c_dates else ""
            d2_s = c_dates[-1] if c_dates else ""
            if lang == "sl":
                return (
                    f"### Primerjava sprememb Prej / Potem (**{idx}**) — **{place}** (`{d1_s}` ↔ `{d2_s}`)\n\n"
                    f"Izvedena je bila piksel-na-piksel primerjava posnetkov **`{d1_s}`** in **`{d2_s}`** (`{sensor}`, **{res_m} m**, **{crs}**) s samodejnim preverjanjem jasnosti nad AOI.\n\n"
                    f"- **Povprečna sprememba Δ{idx}:** **`{csum.get('mean', 0.0):+.3f}`**\n"
                    f"- **Porast:** **{csum.get('pos_area_ha', 0.0):,.2f} ha** ({csum.get('pos_pct', 0.0):.1f} %) | **Upad:** **{csum.get('neg_area_ha', 0.0):,.2f} ha** ({csum.get('neg_pct', 0.0):.1f} %)\n"
                    f"- Spodaj lahko z interaktivno zoomabilno zaveso `↔` natančno pregledate spremembe na ravni parcel.\n\n"
                    f"**Predlogi za nadaljevanje:**\n"
                    f"- *»Pokaži še celotno večletno časovno vrsto {idx} za {place}.«*"
                )
            return (
                f"### Before / After Change Comparison (**{idx}**) — **{place}** (`{d1_s}` ↔ `{d2_s}`)\n\n"
                f"- **Mean Δ{idx}:** `{csum.get('mean', 0.0):+.3f}` | **Increase:** `{csum.get('pos_area_ha', 0.0):,.2f} ha` | **Decrease:** `{csum.get('neg_area_ha', 0.0):,.2f} ha`."
            )

        if kind == "spectral_index":
            st_d = analysis_context.get("stats") or {}
            d_str = str(analysis_context.get("date_str", ""))
            mean_v = float(st_d.get("mean", 0.0))
            med_v = float(st_d.get("median", 0.0))
            min_v = float(st_d.get("min", 0.0))
            max_v = float(st_d.get("max", 0.0))
            ha_v = float(st_d.get("valid_area_ha", 0.0))
            unit_s = " °C" if idx == "LST" else (" dB" if idx == "VV" else "")
            sn = analysis_context.get("snow_stats")
            snow_line = ""
            if isinstance(sn, dict):
                snow_line = (
                    f"\n- **Snežna/ledeniška pokritost (NDSI + DEM):** **{sn.get('snow_area_ha', 0):,.2f} ha** "
                    f"(**{sn.get('snow_pct', 0):.1f} %** AOI), spodnja snežna meja: **`{sn.get('snowline_min_m', '—')} m n. m.`**."
                    if lang == "sl"
                    else f"\n- **Snow/Glacier Cover (NDSI + DEM):** **{sn.get('snow_area_ha', 0):,.2f} ha** ({sn.get('snow_pct', 0):.1f}%), snowline: **`{sn.get('snowline_min_m', '—')} m a.s.l.`**."
                )
            if lang == "sl":
                return (
                    f"### Izračun indeksa **{idx}** za **{place}**\n\n"
                    f"Analiza je bila izvedena na najčistejšem posnetku **`{d_str}`** (`{sensor}`, **{res_m} m**) v geodetski projekciji **{crs}** z oblačnim maskiranjem.\n\n"
                    f"- **Povprečje / Mediana:** **`{mean_v:.3f}{unit_s}`** / **`{med_v:.3f}{unit_s}`** (razpon `{min_v:.3f}{unit_s}` do `{max_v:.3f}{unit_s}`)\n"
                    f"- **Veljavna analizirana površina:** **{ha_v:,.2f} ha**{snow_line}\n"
                    f"- **Interaktivna karta:** Spodaj sta pripravljena **oba sloja** — pravi barvni satelitski posnetek (RGB) in sloj indeksa **{idx}**, ki ju lahko ločeno vklapljate v legendi slojev.\n\n"
                    f"**Predlogi za nadaljevanje:**\n"
                    f"- *»Kaj pomeni povprečna vrednost {idx} = {mean_v:.3f}{unit_s} za to območje?«*\n"
                    f"- *»Izračunaj še NDWI ali analizo reliefa (DEM) za isto območje.«*"
                )
            return (
                f"### **{idx}** Spectral Analysis for **{place}**\n\n"
                f"Computed on the clearest scene (**`{d_str}`**, `{sensor}` at **{res_m} m**) in **{crs}** with cloud masking applied.\n\n"
                f"- **Mean / Median:** **`{mean_v:.3f}{unit_s}`** / **`{med_v:.3f}{unit_s}`** (range `{min_v:.3f}{unit_s}` to `{max_v:.3f}{unit_s}`)\n"
                f"- **Valid Analyzed Area:** **{ha_v:,.2f} ha**{snow_line}\n"
                f"- **Interactive Map:** Both the **True Color RGB satellite scene** and the **{idx} overlay** are toggleable in the map's Layer Control."
            )

        if kind in ("time_series", "time_series_and_comparison"):
            n_d = _sint(analysis_context.get("n_dates"), 0)
            d1 = str(analysis_context.get("date_start", ""))
            d2 = str(analysis_context.get("date_end", ""))
            mean_v = _sfloat(analysis_context.get("mean"), 0.0)
            min_ts = _sfloat(analysis_context.get("min"), 0.0)
            max_ts = _sfloat(analysis_context.get("max"), 0.0)
            slope = analysis_context.get("trend_slope_per_obs")
            excl_arch = list(analysis_context.get("excluded_archive_dates") or [])
            arch_note_sl = ""
            arch_note_en = ""
            if excl_arch:
                arch_str = ", ".join(excl_arch[:4]) + (" …" if len(excl_arch) > 4 else "")
                arch_note_sl = (
                    f"\n- **Radiometrična zanesljivost:** Starejši ali nasičeni posnetki pred letom 1990 "
                    f"(`{arch_str}`) so na voljo za vizualni pregled v primerjalniku, galeriji in Timelapse animaciji, "
                    f"medtem ko kvantitativna analiza sprememb (Δ), trend in izračun površin temeljijo na "
                    f"zanesljivejših podatkih od **`{d1}`** naprej."
                )
                arch_note_en = (
                    f"\n- **Radiometric Reliability:** Earlier or saturated archive scenes (`{arch_str}`) remain "
                    f"available for visual inspection in the comparator, gallery, and Timelapse animation, while the "
                    f"quantitative change map (Δ), trend regression, and area statistics are computed from the "
                    f"reliable baseline starting **`{d1}`**."
                )
            if lang == "sl":
                trend_txt = f" Linearni trend znaša **`{_sfloat(slope):+.4f}` na opazovanje**." if slope is not None else ""
                return (
                    f"### Časovna vrsta **{idx}** za **{place}** (`{d1}` – `{d2}`)\n\n"
                    f"Iz satelitskega arhiva `{sensor}` (**{res_m} m**, **{crs}**) je bilo obdelanih **{n_d} kakovostnih datumov** (interval: `{analysis_context.get('interval', '')}` z istim letnim časom in samodejno zamenjavo oblačnih datumov).\n\n"
                    f"- **Povprečna vrednost v obdobju:** **`{mean_v:.3f}`** (min `{min_ts:.3f}`, max `{max_ts:.3f}`).{trend_txt}{arch_note_sl}\n\n"
                    f"**Predlogi za nadaljevanje:**\n"
                    f"- *»Napiši kratek povzetek tega trenda za moje poročilo.«*\n"
                    f"- *»Primerjaj {idx} za {place} med {d1[:4]} in {d2[:4]} z interaktivno zaveso.«*"
                )
            return (
                f"### **{idx}** Time Series for **{place}** (`{d1}` – `{d2}`)\n\n"
                f"Processed **{n_d} reliable dates** from `{sensor}` (**{res_m} m**, **{crs}**).\n\n"
                f"- **Period Mean:** **`{mean_v:.3f}`** (min `{min_ts:.3f}`, max `{max_ts:.3f}`).{arch_note_en}"
            )

        if kind == "scene_rgb":
            d_str = str(analysis_context.get("date_str", ""))
            if lang == "sl":
                return (
                    f"### Satelitski posnetek za **{place}** (`{d_str}`)\n\n"
                    f"Naložen je najčistejši satelitski posnetek (`{sensor}`, **{res_m} m**, projekcija **{crs}**). "
                    f"Spodaj ga lahko približate na karti, prenesete kot georeferenciran GeoTIFF ali pošljete v modul za izdelavo 300 DPI karte.\n\n"
                    f"**Predlog:** *»Izračunaj še NDVI ali brezoblačni mozaik za isto območje.«*"
                )
            return (
                f"### Satellite Scene for **{place}** (`{d_str}`)\n\n"
                f"Loaded the clearest scene (`{sensor}`, **{res_m} m**, **{crs}**). You can inspect it interactively below, download the GeoTIFF, or send it to 300 DPI Cartography."
            )

        if kind == "carto_map":
            mc = analysis_context.get("layout_config") or {}
            scale_d = int(analysis_context.get("scale_denom", 10000))
            scale_str = f"{scale_d:,}".replace(",", " ")
            grid_m = int(analysis_context.get("grid_spacing_m", 1000))
            fmt_s = str(mc.get("paper_format", "A4_LANDSCAPE"))
            theme_s = str(mc.get("theme", "classic_light"))
            north_s = (
                f"{mc.get('north_position', 'top_right')} ({mc.get('north_style', 'classic_compass')})"
                if mc.get("show_north_arrow", True)
                else ("skrita" if lang == "sl" else "hidden")
            )
            scale_pos_s = (
                str(mc.get("scale_position", "bottom_left"))
                if mc.get("show_scale_bar", True)
                else ("skrito" if lang == "sl" else "hidden")
            )
            leg_s = (
                str(mc.get("legend_position", "right_panel"))
                if mc.get("show_legend", True)
                else ("skrita" if lang == "sl" else "hidden")
            )
            inset_s = (
                f"da ({mc.get('inset_position', 'right_panel_bottom')})"
                if mc.get("show_inset_map", False)
                else ("ne" if lang == "sl" else "off")
            )
            g_style = str(mc.get("grid_style", "lines")) if mc.get("show_grid", True) else ("brez" if lang == "sl" else "off")
            map_title = str(mc.get("title") or analysis_context.get("layer_title") or "")
            if lang == "sl":
                return (
                    f"### 🗺️ Pripravljena geodetska karta — **{map_title}**\n\n"
                    f"Karta je izrisana v strogi metrični projekciji **{crs}** (1:1 razmerje brez distorzije) in pripravljena za tisk v **300 DPI (PDF in PNG)**.\n\n"
                    f"- **Format in tema:** `{fmt_s}` · `{theme_s}`\n"
                    f"- **Izračunano merilo:** **M ≈ 1 : {scale_str}** · **Mreža:** `{g_style}` ({grid_m} m)\n"
                    f"- **Elementi:** severnica `{north_s}`, merilo `{scale_pos_s}`, legenda `{leg_s}`, pregledna karta `{inset_s}`\n\n"
                    f"Če želite premakniti katerikoli element, spremeniti barvo mreže, dodati opombo ali preklopiti na temno/svetlo temo, mi preprosto napišite!"
                )
            return (
                f"### 🗺️ Print-Ready Geodetic Map — **{map_title}**\n\n"
                f"Rendered in exact 1:1 metric projection **{crs}** with **300 DPI PDF & PNG** exports ready below.\n\n"
                f"- **Format & Theme:** `{fmt_s}` · `{theme_s}`\n"
                f"- **Computed Scale:** **M ≈ 1 : {scale_str}** · **Grid:** `{g_style}` ({grid_m} m)\n"
                f"- **Elements:** North arrow `{north_s}`, Scale bar `{scale_pos_s}`, Legend `{leg_s}`, Inset map `{inset_s}`\n\n"
                f"Tell me if you'd like to move any element, switch themes, toggle the locator map, or add a figure caption!"
            )

    # General QA / follow-up fallback when no Gemini API key is configured
    norm_q = _normalize_ascii(user_message)
    parts: list[str] = []
    if lang == "sl":
        if any(k in norm_q for k in ("porocil", "odstavek", "povzetek", "seminarsk")) and analysis_context:
            parts.append("**Predlog odstavka za poročilo (na podlagi zadnje analize):**\n")
            parts.append(f"> {analysis_context.get('technical_summary', '').replace(chr(10), ' ')}")
        elif any(k in norm_q for k in ("sar", "radar", "poplav")):
            parts.append(
                "**Sentinel-1 SAR (C-pas, 5,4 GHz)** je aktiven radarski senzor, ki oddaja lasten mikrovalovni signal in prodira skozi oblake ter dež. "
                "Mirna vodna gladina odbija radarski žarek zrcalno stran od satelita (zelo nizke vrednosti VV dB, pod −16 dB), zato poplavljene površine na SAR sliki izstopajo kot temna območja."
            )
        elif "ndvi" in norm_q:
            parts.append(
                "**NDVI = (NIR − Red) / (NIR + Red)** meri vitalnost in gostoto vegetacije na lestvici od **−1 do +1**:\n"
                "- `< 0,0`: voda, sneg ali oblaki\n"
                "- `0,0 – 0,2`: gola tla, asfalt, pozidane površine\n"
                "- `0,2 – 0,5`: redka vegetacija, travniki, sušni posevki\n"
                "- `> 0,6`: gost zdrav gozd ali bujna vegetacija"
            )
        elif any(k in norm_q for k in ("pdop", "gnss", "galileo", "skyplot")):
            parts.append(
                "**PDOP (Position Dilution of Precision)** meri geometrijsko kakovost razporeditve vidnih GNSS satelitov v 3D prostoru. "
                "Vrednosti **PDOP ≤ 2,5** ob najmanj 6 vidnih satelitih zagotavljajo zanesljivo geodetsko RTK / statično izmero."
            )
        elif any(k in norm_q for k in ("dnbr", "nbr", "pozar")):
            parts.append(
                "**dNBR = NBR_pred − NBR_po** primerja spektralni indeks ožganosti pred požarom in po njem (`NBR = (NIR − SWIR2) / (NIR + SWIR2)`). "
                "Pozitivne vrednosti dNBR nad `0,27` kažejo zmerno, nad `0,66` pa zelo visoko stopnjo poškodovanosti vegetacije in tal po USGS lestvici."
            )
        elif any(k in norm_q for k in ("landsat", "sentinel")):
            parts.append(
                "**Sentinel-2 (10 m)** ponuja najvišjo prostorsko ločljivost in 5-dnevno ponovljivost za obdobje od **2015 dalje**, "
                "medtem ko **Landsat Collection 2 (30 m)** omogoča neprekinjen zgodovinski arhiv od leta **1984** ter termalni kanal za izračun površinske temperature (**LST °C**)."
            )
        elif analysis_context:
            parts.append(
                f"Trenutno aktivna analiza v pogovoru je **{analysis_context.get('task_type')} ({analysis_context.get('index_name')})** "
                f"za **{analysis_context.get('resolved_place')}** v projekciji **{analysis_context.get('crs')}**."
            )
        else:
            parts.append(
                "Sem vaš geodetski AI pomočnik za daljinsko zaznavanje. Lahko zaženem analizo poplav (SAR), požarišč (dNBR), spektralnih indeksov (NDVI, NDWI, NDSI, LST), "
                "brezoblačnih mozaikov, primerjav Prej/Potem, časovnih vrst, Copernicus DEM 30 m reliefa ali načrtovanja GNSS izmere — samo opišite lokacijo in želeno analizo!"
            )

        if analysis_context and not any(k in norm_q for k in ("porocil", "odstavek", "povzetek")):
            if analysis_context.get("kind") == "disaster":
                parts.append(
                    f"\n*Kontekst zadnje analize ({analysis_context.get('resolved_place')}):* prizadeta površina **{analysis_context.get('affected_area_ha', 0):,.2f} ha** ({analysis_context.get('affected_pct', 0):.2f} % AOI)."
                )
            elif analysis_context.get("kind") == "spectral_index":
                st_c = analysis_context.get("stats") or {}
                parts.append(
                    f"\n*Kontekst zadnje analize ({analysis_context.get('resolved_place')}):* povprečje **{analysis_context.get('index_name')} = {st_c.get('mean', 0):.3f}**, veljavna površina **{st_c.get('valid_area_ha', 0):,.2f} ha**."
                )
        parts.append(
            "\n\n*ℹ️ Za popolno generativno razpravo v naravnem jeziku vnesite brezplačen Google Gemini API ključ v stranski vrstici (`⚙️ Gemini API in model`).*"
        )
        return "\n".join(parts)

    if analysis_context:
        parts.append(
            f"Active analysis context: **{analysis_context.get('task_type')} ({analysis_context.get('index_name')})** "
            f"for **{analysis_context.get('resolved_place')}** in **{analysis_context.get('crs')}**."
        )
    else:
        parts.append(
            "I'm your Geodetic Earth Observation assistant. I can run SAR floods, dNBR wildfires, NDVI/NDWI/NDSI/LST, cloud-free composites, Before/After comparisons, time series, Copernicus DEM terrain analysis, and GNSS mission planning."
        )
    return "\n\n".join(parts)


def _format_fusion_rationale_section(analysis_context: dict[str, Any] | None, lang: str = "sl") -> str:
    """Format a clear, educational Markdown subsection explaining why multiple satellite systems were combined."""
    if not isinstance(analysis_context, dict):
        return ""
    f_mode = str(analysis_context.get("fusion_mode") or FUSION_SINGLE)
    if f_mode == FUSION_SINGLE:
        return ""

    rationale = str(
        analysis_context.get("fusion_rationale_en" if lang == "en" else "fusion_rationale_sl") or ""
    ).strip()
    syn = analysis_context.get("synergy_metrics") if isinstance(analysis_context.get("synergy_metrics"), dict) else {}

    extra_bullets: list[str] = []
    if f_mode == FUSION_TEMPORAL_LANDSAT_S2:
        n_ls = syn.get("landsat_obs")
        n_s2 = syn.get("sentinel2_obs")
        if n_ls is not None and n_s2 is not None:
            extra_bullets.append(
                f"- **Število uporabljenih posnetkov:** `{n_ls}` zgodovinskih posnetkov Landsat (30 m) + `{n_s2}` sodobnih posnetkov Sentinel-2 (10–20 m)."
                if lang == "sl"
                else f"- **Harmonized observations:** `{n_ls}` historical Landsat (30 m) scenes + `{n_s2}` modern Sentinel-2 (10–20 m) scenes."
            )
        extra_bullets.append(
            "- **Znanstveno opozorilo (opomba k fuziji):** Zaradi razlike v prostorski ločljivosti (30 m vs. 10–20 m) in manjših razlik v spektralnih pasovih senzorjev OLI/TM in MSI sta rastrski mreži za karto sprememb Δ samodejno usklajeni v isti metrični projekciji."
            if lang == "sl"
            else "- **Scientific caveat:** Due to pixel-size differences (30 m vs. 10–20 m) and slight spectral bandpass differences between Landsat OLI/TM and Sentinel-2 MSI, grids are co-registered in the same metric projection for pixel-wise Δ maps."
        )
    elif f_mode == FUSION_SYNERGY_OPTICAL_THERMAL:
        if syn.get("lst_urban_mean_c") is not None and syn.get("lst_veg_mean_c") is not None:
            delta_c = syn.get("uhi_cooling_delta_c", round(syn["lst_urban_mean_c"] - syn["lst_veg_mean_c"], 2))
            extra_bullets.append(
                f"- **Kvantitativni učinek toplotnega otoka (UHI):** Povprečna temperatura pozidanih/golih površin (`NDVI < 0,2`) znaša **{syn['lst_urban_mean_c']:.1f} °C**, v gostih parkih/gozdovih (`NDVI > 0,5`) pa **{syn['lst_veg_mean_c']:.1f} °C** → hladilni učinek vegetacije: **−{abs(delta_c):.1f} °C**."
                if lang == "sl"
                else f"- **Quantitative Urban Heat Island (UHI) cooling:** Built-up/bare areas (`NDVI < 0.2`) average **{syn['lst_urban_mean_c']:.1f} °C** vs. dense vegetation/parks (`NDVI > 0.5`) at **{syn['lst_veg_mean_c']:.1f} °C** → vegetation cooling effect: **−{abs(delta_c):.1f} °C**."
            )
        if syn.get("pearson_r_ndvi_lst") is not None:
            extra_bullets.append(
                f"- **Korelacija NDVI ↔ LST:** Pearsonov koeficient `r = {syn['pearson_r_ndvi_lst']:+.3f}`."
                if lang == "sl"
                else f"- **NDVI ↔ LST correlation:** Pearson `r = {syn['pearson_r_ndvi_lst']:+.3f}`."
            )
        extra_bullets.append(
            "- **Znanstveno opozorilo:** Termalni kanal Landsat `ST_B10` ima 30 m (izvorno 100 m) ločljivost, medtem ko Sentinel-2 zagotavlja 10 m optični kontekst; za statistično korelacijo je NDVI prevzorčen na termalno mrežo."
            if lang == "sl"
            else "- **Scientific caveat:** Landsat `ST_B10` thermal imagery is provided at 30 m (native 100 m TIR) while Sentinel-2 provides 10 m optical detail; NDVI is aligned to the 30 m thermal grid for correlation."
        )
    elif f_mode == FUSION_SYNERGY_SAR_OPTICAL:
        if syn.get("s2_optical_date"):
            extra_bullets.append(
                f"- **Optični kontekst:** Na isto karto je dodan najbližji jasen posnetek Sentinel-2 L2A z dne `{syn['s2_optical_date']}` (RGB + NDWI)."
                if lang == "sl"
                else f"- **Optical context:** Paired with the nearest clear Sentinel-2 L2A scene from `{syn['s2_optical_date']}` (RGB + NDWI) on the same map."
            )
    elif f_mode == FUSION_SYNERGY_OPTICAL_DEM:
        if syn.get("dem_mean_m") is not None:
            extra_bullets.append(
                f"- **Reliefni kontekst (Copernicus DEM 30 m):** Nadmorska višina `{syn.get('dem_min_m')}–{syn.get('dem_max_m')} m` (povprečje `{syn.get('dem_mean_m')} m`), povprečni naklon `{syn.get('dem_mean_slope_deg')}°`."
                if lang == "sl"
                else f"- **Terrain context (Copernicus DEM 30 m):** Elevation `{syn.get('dem_min_m')}–{syn.get('dem_max_m')} m` (mean `{syn.get('dem_mean_m')} m`), mean slope `{syn.get('dem_mean_slope_deg')}°`."
            )

    heading = (
        "### 🛰️ Zakaj je bila uporabljena kombinacija več satelitskih sistemov"
        if lang == "sl"
        else "### 🛰️ Why Multiple Satellite Systems Were Combined"
    )
    body_lines = [heading, "", rationale]
    if extra_bullets:
        body_lines.append("")
        body_lines.extend(extra_bullets)
    return "\n\n" + "\n".join(body_lines)


def generate_conversational_reply(
    user_message: str,
    chat_history: Sequence[dict[str, Any]] | None = None,
    analysis_context: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    lang: str = "sl",
    is_post_analysis: bool = False,
) -> dict[str, Any]:
    """Generate a rich, thorough, non-truncated multi-turn conversational geodetic EO reply."""
    resolved_key, _ = get_active_gemini_api_key(api_key)
    lang_name = "Slovenian" if lang == "sl" else "English"
    is_multi_sensor = bool(
        isinstance(analysis_context, dict)
        and str(analysis_context.get("fusion_mode") or FUSION_SINGLE) != FUSION_SINGLE
    )

    if resolved_key:
        if is_post_analysis and isinstance(analysis_context, dict) and analysis_context.get("kind") == "carto_map":
            sys_prompt = (
                f"You are an expert Geodetic Cartographer & Remote Sensing AI assistant. "
                f"Write a crisp, helpful response in natural {lang_name} using clean Markdown.\n"
                f"A publication-ready 300 DPI geodetic map was JUST generated from the user's request. "
                f"Summarize the exact applied cartographic configuration from the JSON (paper format, exact computed scale M ≈ 1 : ..., "
                f"metric CRS, theme, element placements for North arrow, Scale bar, Legend, Grid style & spacing, Inset map, custom note if any), "
                f"and warmly invite the user to ask for any further layout tweaks in natural language (e.g., moving elements, switching theme, changing grid spacing, or adding a caption)."
            )
        elif is_post_analysis:
            fusion_instruction = (
                "\n- **MANDATORY MULTI-SENSOR SUBSECTION**: Because `fusion_mode != 'single'`, you MUST include a dedicated subsection titled "
                + (
                    "**🛰️ Zakaj je bila uporabljena kombinacija več satelitskih sistemov**"
                    if lang == "sl"
                    else "**🛰️ Why Multiple Satellite Systems Were Combined**"
                )
                + " clearly explaining which satellite systems were fused, WHY each was chosen (e.g., Landsat 30 m for pre-2017 history + Sentinel-2 10–20 m for 2017+, Sentinel-1 SAR penetrating clouds + Sentinel-2 optical RGB/NDWI, or Sentinel-2 10 m optical + Landsat ST_B10 LST °C for Urban Heat Island cooling), all quantitative cross-sensor metrics from `synergy_metrics`, and honest scientific caveats (pixel size / bandpass differences)."
                if is_multi_sensor
                else ""
            )
            archive_dates_ctx = (
                list(analysis_context.get("excluded_archive_dates") or [])
                if isinstance(analysis_context, dict)
                else []
            )
            archive_instruction = (
                "\n- **RADIOMETRIC RELIABILITY NOTE**: `excluded_archive_dates` is non-empty "
                f"({', '.join(archive_dates_ctx[:5])}). Explicitly explain in Methodology/Interpretation that those "
                "earlier pre-1990 or saturated archive scenes are kept available for visual inspection in the swipe "
                "comparator, gallery, and Timelapse animation, while the quantitative change map (Δ), trend regression, "
                "and area-loss statistics are computed strictly from the radiometrically reliable baseline (>= 1990)."
                if archive_dates_ctx
                else ""
            )
            sys_prompt = (
                f"You are an expert, insightful Geodetic & Remote Sensing (Earth Observation) AI specialist. "
                f"Write a thorough, complete, well-structured report in clear, natural {lang_name} using clean Markdown.\n"
                f"A real geodetic / satellite analysis was JUST executed for the user's request. Exact computed results are provided in JSON below.\n"
                f"Organize your response into these 4 clear sections (with crisp headings in {lang_name}):\n"
                f"1. **Ozadje dogodka in območja** (or Background & Geographic Context in EN): Concise, insightful geographic/historical context about the analyzed place or event (e.g., Mežiška dolina / Črna na Koroškem during the catastrophic August 2023 floods, Kras karst terrain during July 2022 wildfires, Triglav alpine glaciology, etc.).\n"
                f"2. **Metodologija in podatki** (Methodology & Data): Exact sensor/dataset used, WHY it was chosen (e.g., Sentinel-1 SAR radar penetrates storm clouds; Sentinel-2 10 m optical; Landsat thermal ST_B10; Copernicus DEM 30 m; GNSS multi-constellation with 360° DEM+S2 obstruction mask), exact acquisition dates, spatial resolution, and projected metric CRS (`EPSG:3794` / local UTM).{fusion_instruction}{archive_instruction}\n"
                f"3. **Podrobna interpretacija rezultatov** (Detailed Interpretation of Results): Thoroughly explain EVERY key computed number from the JSON (`ha`, `km²`, `%`, min/mean/median/max, severity classes, snowline altitude, slope/relief, UHI cooling delta °C, or PDOP/satellite visibility & optimal windows), what those values physically mean on the ground, AND important scientific limitations/caveats. NEVER invent numbers not present in the JSON.\n"
                f"4. **Predlogi za nadaljevanje** (Suggested Next Steps): 2–3 concrete follow-up analyses or questions the user can ask next in this chat."
            )
        else:
            sys_prompt = (
                f"You are an expert, friendly Geodetic & Remote Sensing (Earth Observation) AI specialist. "
                f"Respond thoroughly and clearly in natural {lang_name} using clean Markdown.\n"
                f"Use the provided recent analysis context (if any) to answer the user's follow-up question accurately with exact numbers, "
                f"explain geodetic/EO concepts clearly, or draft publication-ready report paragraphs when asked. Never fabricate metrics not present in the context."
            )

        ctx_block = (
            json.dumps(analysis_context, ensure_ascii=False)
            if isinstance(analysis_context, dict) and analysis_context
            else "None (no analysis executed yet in this session)."
        )
        full_sys = f"{sys_prompt}\n\nCURRENT EXECUTED ANALYSIS CONTEXT (JSON):\n{ctx_block}"

        contents: list[dict[str, Any]] = []
        for turn in list(chat_history or [])[-8:]:
            role = "user" if turn.get("role") == "user" else "model"
            txt = str(turn.get("content", "")).strip()
            if txt:
                contents.append({"role": role, "parts": [{"text": txt[:2500]}]})

        if not (
            contents
            and contents[-1]["role"] == "user"
            and contents[-1]["parts"][0]["text"].strip() == user_message.strip()
        ):
            contents.append({"role": "user", "parts": [{"text": user_message}]})

        payload = {
            "systemInstruction": {"parts": [{"text": full_sys}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 4096,
            },
        }
        try:
            reply_text, used_model, fb_sl, fb_en = _call_gemini_rest_with_fallback(
                payload, api_key=resolved_key, model=model, timeout=35
            )
            if reply_text:
                return {
                    "text": reply_text,
                    "engine": "gemini",
                    "gemini_model": used_model,
                    "fallback_note_sl": fb_sl,
                    "fallback_note_en": fb_en,
                }
        except Exception:
            pass

    local_txt = _generate_local_conversational_reply(
        user_message=user_message,
        analysis_context=analysis_context,
        lang=lang,
        is_post_analysis=is_post_analysis,
    )
    if is_post_analysis and is_multi_sensor:
        local_txt = local_txt + _format_fusion_rationale_section(analysis_context, lang=lang)
    return {
        "text": local_txt,
        "engine": "local",
        "gemini_model": None,
        "fallback_note_sl": "",
        "fallback_note_en": "",
    }

