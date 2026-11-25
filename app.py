"""Glavna spletna aplikacija za daljinsko zaznavanje (Multi-Sensor EO — Slovenija).

Geodetsko natančna obdelava posnetkov Sentinel-2 L2A (10 m, 2015–danes),
Landsat Collection 2 Level-2 (30 m, 1984–danes) in Sentinel-1 SAR RTC (10 m,
radarski posnetki skozi oblake) v D96/TM (EPSG:3794) in UTM 33N (EPSG:32633)
z neposrednim pretočnim nalaganjem COG iz Microsoft Planetary Computer STAC API.
Podpira interaktivno risanje poligonov (`rio.clip`), iskanje krajev (Nominatim),
interaktivni rastrski prikazovalnik (Plotly zoom/pan) ter dvojezični vmesnik (SL | EN).
"""

from __future__ import annotations

import base64
import gc
import importlib
import json
import os
import re
import sys
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any, Callable, Sequence

import folium
from folium.plugins import Draw, MeasureControl
import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import streamlit.components.v1 as components
import xarray as xr
from shapely.geometry import box as shapely_box
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from streamlit_folium import st_folium

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

_LOCAL_MODULES_ORDER: tuple[str, ...] = (
    "github_sync",
    "auth",
    "projekti",
    "geodezija_crs",
    "map_overlay",
    "kartografija",
    "stac_loader",
    "analize",
    "casovne_vrste",
    "relief",
    "nesrece",
    "ai_pomocnik",
    "gnss_plan",
    "i18n",
)


def _reload_stale_local_modules() -> None:
    """Reload cached local modules when their .py files changed on disk (Streamlit Cloud hot-reload)."""
    any_reloaded = False
    for mod_name in _LOCAL_MODULES_ORDER:
        mod = sys.modules.get(mod_name)
        mod_path = os.path.join(_APP_DIR, f"{mod_name}.py")
        if mod is not None and os.path.exists(mod_path):
            try:
                disk_mtime = os.path.getmtime(mod_path)
                loaded_mtime = getattr(mod, "__dz_loaded_mtime__", None)
                if any_reloaded or loaded_mtime != disk_mtime:
                    reloaded = importlib.reload(mod)
                    setattr(reloaded, "__dz_loaded_mtime__", disk_mtime)
                    any_reloaded = True
            except Exception:
                pass


_reload_stale_local_modules()

from auth import (
    authenticate_user,
    load_user_profile,
    register_user,
    save_user_profile,
    sync_user_data_to_github,
)
from github_sync import (
    get_github_sync_status,
    github_restore_all_on_startup,
    is_github_sync_configured,
)
from projekti import (
    delete_map_layout,
    delete_user_project,
    export_user_bundle_json,
    import_user_bundle_json,
    list_map_layouts,
    list_user_projects,
    load_last_session,
    save_last_session,
    save_map_layout,
    save_user_project,
)
from ai_pomocnik import (
    AI_EXAMPLE_PROMPTS,
    DEFAULT_GEMINI_MODEL,
    GEMINI_MODEL_OPTIONS,
    apply_incremental_layout_overrides,
    build_analysis_context_dict,
    execute_ai_plan,
    generate_conversational_reply,
    get_active_gemini_api_key,
    list_available_gemini_models,
    parse_eo_request,
    resolve_gemini_model_id,
    route_user_message,
    verify_gemini_api_connection,
)
from analize import (
    OPTICAL_SIGNATURE_BANDS,
    SAR_METRICS,
    SPECTRAL_INDICES,
    apply_cloud_mask,
    compute_index,
    compute_index_stats,
    compute_quick_land_cover,
    compute_sar_water_mask,
    compute_snow_glacier_stats,
    extract_sar_metric,
    extract_spectral_signature,
    get_required_bands_for_sensor,
)
from casovne_vrste import (
    compile_timelapse_gif,
    compile_timelapse_gif_fps,
    compute_change_detection,
    compute_change_summary,
    compute_cloud_free_composite,
    extract_time_series,
    generate_timelapse_frames,
    render_change_swipe_pair,
    render_multi_date_comparison,
)
from geodezija_crs import (
    CRS_D96TM,
    CRS_UTM33N,
    EPSG_D96TM,
    EPSG_WGS84,
    bbox_from_point_d96tm,
    bbox_from_point_wgs84,
    clamp_polygon_to_max_area,
    clip_to_aoi_polygon,
    export_to_geotiff_bytes,
    format_crs_label,
    get_auto_utm_epsg,
    get_best_metric_crs,
    get_ram_limits,
    get_runtime_environment,
    get_safe_processing_resolution,
    is_codespaces,
    is_in_slovenia,
    transform_coords,
    validate_aoi_geometry,
    validate_aoi_size,
)
from gnss_plan import (
    CONSTELLATIONS as GNSS_CONSTELLATIONS,
    CONSTELLATION_COLORS as GNSS_CONST_COLORS,
    run_gnss_mission_plan,
)
from i18n import t
from kartografija import (
    DEFAULT_LAYOUT_CONFIG,
    LAYOUT_PRESETS,
    format_metric_number,
    merge_layout_config,
    render_geodetic_map,
)
from map_overlay import (
    build_horizontal_colorbar_html,
    build_satellite_slippy_map,
    build_zoomable_swipe_map,
    expand_bbox_for_panning,
    raster_to_map_overlay_png,
)
from nesrece import (
    DISASTER_FLOOD,
    DISASTER_FOREST,
    DISASTER_TYPES,
    DISASTER_WILDFIRE,
    SLOVENIA_CASE_STUDIES,
    build_carto_discrete_items,
    build_slippy_rgba_map,
    get_class_specs,
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
    SEASON_SUMMER,
    SENSOR_LANDSAT,
    SENSOR_S1,
    SENSOR_S2,
    TEMPORAL_INTERVAL_KEYS,
    YEARLY_SEASON_KEYS,
    YEARLY_SEASON_SPECS,
    filter_by_temporal_interval,
    load_copernicus_dem,
    load_scene,
    search_scenes,
)
from relief import (
    TERRAIN_LAYER_SPECS,
    build_preset_transect_endpoints,
    compute_terrain_derivatives,
    extract_transect_profile,
    load_copernicus_dem as load_relief_dem,
)

for _m_name in _LOCAL_MODULES_ORDER:
    _m_obj = sys.modules.get(_m_name)
    _m_path = os.path.join(_APP_DIR, f"{_m_name}.py")
    if _m_obj is not None and os.path.exists(_m_path):
        try:
            setattr(_m_obj, "__dz_loaded_mtime__", os.path.getmtime(_m_path))
        except Exception:
            pass

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Konfiguracija strani (mora biti prvi Streamlit klic)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Daljinsko Zaznavanje — Multi-Sensor EO (SLO)",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tehnični / kartografski CSS (brez vijoličnih barv in neonskih prelivov)
st.markdown(
    """
<style>
:root {
    --geo-teal: #0F766E;
    --geo-teal-dark: #115E59;
    --geo-teal-soft: rgba(15, 118, 110, 0.08);
    --geo-slate: #0F172A;
    --geo-muted: #475569;
    --geo-border: #E2E8F0;
    --geo-surface: #FFFFFF;
    --geo-bg: #F8FAFC;
    --geo-radius: 8px;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, [data-testid="stChatMessageContainer"] {
    overflow-anchor: none !important;
}

.block-container {
    padding-top: 2.85rem !important;
    padding-bottom: 2rem !important;
    max-width: 1440px;
}

/* Kompakten, čist horizontalni naslovni pas */
.geo-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.65rem 1.25rem;
    background-color: var(--geo-surface);
    border: 1px solid var(--geo-border);
    border-left: 4px solid var(--geo-teal);
    border-radius: var(--geo-radius);
    padding: 0.65rem 1.05rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.geo-header-left {
    min-width: 240px;
    flex: 1 1 auto;
}
.geo-header-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--geo-slate);
    letter-spacing: -0.015em;
    margin: 0;
    line-height: 1.25;
}
.geo-header-sub {
    font-size: 0.8rem;
    color: var(--geo-muted);
    margin: 0.12rem 0 0 0;
    line-height: 1.3;
}
.geo-badges {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
}
.geo-badge {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.74rem;
    font-weight: 600;
    color: var(--geo-teal-dark);
    background-color: var(--geo-teal-soft);
    border: 1px solid rgba(15, 118, 110, 0.22);
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    white-space: nowrap;
}
.geo-badge-slate {
    color: var(--geo-slate);
    background-color: #F1F5F9;
    border: 1px solid #CBD5E1;
}

/* Inline RAM statusni indikator v stranski vrstici */
.ram-pill {
    font-size: 0.78rem;
    font-weight: 600;
    border-radius: 6px;
    padding: 0.34rem 0.6rem;
    margin: 0.35rem 0 0.45rem 0;
    line-height: 1.35;
}
.ram-pill-ok {
    color: #166534;
    background-color: #F0FDF4;
    border: 1px solid #BBF7D0;
}
.ram-pill-warn {
    color: #92400E;
    background-color: #FFFBEB;
    border: 1px solid #FDE68A;
}
.ram-pill-err {
    color: #991B1B;
    background-color: #FEF2F2;
    border: 1px solid #FECACA;
}

/* Monospace kartica za geodetske koordinate */
.coord-box {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.76rem;
    line-height: 1.48;
    color: var(--geo-slate);
    background-color: var(--geo-surface);
    border: 1px solid var(--geo-border);
    border-left: 3px solid var(--geo-teal);
    border-radius: 6px;
    padding: 0.45rem 0.62rem;
    margin: 0.3rem 0 0.45rem 0;
}
.coord-box strong {
    color: var(--geo-teal-dark);
}

/* Strukturirane kartice z obrobo (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--geo-surface);
    border: 1px solid var(--geo-border) !important;
    border-radius: var(--geo-radius) !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}

/* Kartice metrik */
div[data-testid="stMetric"] {
    background-color: var(--geo-surface);
    border: 1px solid var(--geo-border);
    border-radius: var(--geo-radius);
    padding: 0.6rem 0.85rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}
div[data-testid="stMetric"] label {
    color: var(--geo-muted);
    font-size: 0.78rem;
    font-weight: 600;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--geo-slate);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 1.18rem;
    font-weight: 700;
}

/* Stranska vrstica z jasno ločnico in kompaktnim razmikom (tablice) */
section[data-testid="stSidebar"] {
    border-right: 1px solid #CBD5E1;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.1rem !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--geo-slate);
    font-size: 0.98rem;
    letter-spacing: -0.01em;
    margin-bottom: 0.25rem;
}

/* Gumbi prilagojeni za namizja in tablice (Samsung Galaxy Tab S11 Ultra) */
.stButton > button, .stDownloadButton > button {
    min-height: 2.65rem;
    font-weight: 600;
    border-radius: var(--geo-radius);
    border: 1px solid var(--geo-border);
    transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--geo-teal);
    color: var(--geo-teal-dark);
}
.stButton > button[kind="primary"] {
    background-color: var(--geo-teal);
    border-color: var(--geo-teal);
    color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--geo-teal-dark);
    border-color: var(--geo-teal-dark);
    color: #FFFFFF;
}
.stDownloadButton > button {
    background-color: var(--geo-surface);
    color: var(--geo-teal-dark);
    border-color: var(--geo-teal);
}
.stDownloadButton > button:hover {
    background-color: var(--geo-teal-soft);
    border-color: var(--geo-teal-dark);
}

/* Zavihki in podzavihki */
button[data-baseweb="tab"] {
    font-weight: 600;
    color: var(--geo-muted);
    min-height: 2.55rem;
    padding: 0.45rem 0.95rem;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--geo-teal-dark);
}
div[data-testid="stExpander"] {
    border: 1px solid var(--geo-border);
    border-radius: var(--geo-radius);
    background-color: var(--geo-surface);
}
/* Minimalna statusna kartica aktivnega konteksta v stranski vrstici */
.sb-status-card {
    background-color: var(--geo-surface);
    border: 1px solid var(--geo-border);
    border-left: 3.5px solid var(--geo-teal);
    border-radius: var(--geo-radius);
    padding: 0.62rem 0.78rem;
    margin: 0.35rem 0 0.55rem 0;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.sb-status-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.045em;
    color: var(--geo-teal-dark);
    margin-bottom: 0.4rem;
}
.sb-status-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.24rem 0.65rem;
    align-items: baseline;
    font-size: 0.78rem;
    line-height: 1.38;
}
.sb-status-key {
    color: var(--geo-muted);
    font-weight: 500;
    white-space: nowrap;
}
.sb-status-val {
    color: var(--geo-slate);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-weight: 600;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.sb-env-badge {
    display: inline-block;
    margin-top: 0.45rem;
    padding: 0.16rem 0.5rem;
    border-radius: 999px;
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    background-color: var(--geo-surface-alt);
    color: var(--geo-muted);
    border: 1px solid var(--geo-border);
}
.sb-env-badge-hi {
    background-color: var(--geo-teal-soft);
    color: var(--geo-teal-dark);
    border-color: #99F6E4;
}
/* Prepreči prikaz besedilnega kurzorja / virtualne tipkovnice v spustnih menijih na tablicah */
[data-baseweb="select"] input {
    caret-color: transparent !important;
    cursor: pointer !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Konstante in prednastavitve (stabilni ključi neodvisni od jezika)
# ---------------------------------------------------------------------------
SLOVENIA_CENTER_LAT: float = 46.1512
SLOVENIA_CENTER_LON: float = 14.9955
SLOVENIA_DEFAULT_ZOOM: int = 8

PHOTON_SEARCH_URL: str = "https://photon.komoot.io/api/"

NOMINATIM_SEARCH_URL: str = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT: str = "DaljinskoZaznavanjeSLO/1.0 (https://github.com/zanlesnik13-cpu/Daljinsko-zaznavanje)"

PLOTLY_VIEWER_HEIGHT: int = 580
PLOTLY_VIEWER_CONFIG: dict[str, Any] = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

SLOVENIA_PRESETS: dict[str, dict[str, Any]] = {
    "Ljubljana (Barje/Center)": {
        "lat": 46.0569,
        "lon": 14.5058,
        "radius_m": 2500,
        "desc_key": "preset_ljubljana_desc",
    },
    "Triglav / Bohinjsko jezero": {
        "lat": 46.2830,
        "lon": 13.8580,
        "radius_m": 3500,
        "desc_key": "preset_triglav_desc",
    },
    "Maribor / Pohorje": {
        "lat": 46.5547,
        "lon": 15.6459,
        "radius_m": 3000,
        "desc_key": "preset_maribor_desc",
    },
    "Koper / Sečoveljske soline": {
        "lat": 45.4930,
        "lon": 13.6130,
        "radius_m": 3000,
        "desc_key": "preset_koper_desc",
    },
    "Kras (požarišče)": {
        "lat": 45.8550,
        "lon": 13.6450,
        "radius_m": 4000,
        "desc_key": "preset_kras_desc",
    },
    "Mura / Prekmurje": {
        "lat": 46.6620,
        "lon": 16.1660,
        "radius_m": 3500,
        "desc_key": "preset_mura_desc",
    },
}

SLOVENIA_PLACES_FALLBACK: dict[str, tuple[float, float]] = {
    "Ljubljana": (46.0569, 14.5058),
    "Maribor": (46.5547, 15.6459),
    "Celje": (46.2397, 15.2677),
    "Kranj": (46.2389, 14.3556),
    "Koper": (45.5481, 13.7302),
    "Novo mesto": (45.8040, 15.1689),
    "Nova Gorica": (45.9560, 13.6484),
    "Velenje": (46.3592, 15.1103),
    "Murska Sobota": (46.6625, 16.1664),
    "Ptuj": (46.4201, 15.8702),
    "Bled": (46.3683, 14.1146),
    "Bohinj": (46.2788, 13.8842),
    "Bohinjsko jezero": (46.2830, 13.8580),
    "Triglav": (46.3789, 13.8367),
    "Postojna": (45.7744, 14.2153),
    "Kamnik": (46.2259, 14.6121),
    "Domžale": (46.1377, 14.5937),
    "Škofja Loka": (46.1655, 14.3063),
    "Jesenice": (46.4306, 14.0669),
    "Izola": (45.5369, 13.6619),
    "Piran": (45.5283, 13.5683),
    "Portorož": (45.5143, 13.5921),
    "Slovenj Gradec": (46.5103, 15.0804),
    "Krško": (45.9591, 15.4917),
    "Brežice": (45.9033, 15.5911),
    "Kočevje": (45.6433, 14.8633),
    "Idrija": (46.0028, 14.0306),
    "Tolmin": (46.1830, 13.7332),
    "Bovec": (46.3380, 13.5524),
    "Kranjska Gora": (46.4845, 13.7857),
    "Logarska dolina": (46.3910, 14.6290),
    "Kras": (45.8550, 13.6450),
    "Sečovlje": (45.4930, 13.6130),
    "Sečoveljske soline": (45.4930, 13.6130),
    "Pohorje": (46.5180, 15.4750),
    "Cerknica": (45.7931, 14.3625),
    "Cerkniško jezero": (45.7520, 14.3750),
    "Ribnica": (45.7386, 14.7275),
    "Trbovlje": (46.1550, 15.0533),
    "Zagorje": (46.1318, 14.9969),
    "Zagorje ob Savi": (46.1318, 14.9969),
    "Litija": (46.0586, 14.8225),
    "Grosuplje": (45.9556, 14.6589),
    "Vrhnika": (45.9635, 14.2948),
    "Ajdovščina": (45.8860, 13.9095),
    "Sežana": (45.7092, 13.8733),
    "Radovljica": (46.3444, 14.1744),
    "Ljubljansko barje": (45.9950, 14.4600),
    "Velika planina": (46.2940, 14.6470),
}

SENSOR_KEYS: list[str] = [SENSOR_S2, SENSOR_LANDSAT, SENSOR_S1]
SENSOR_META: dict[str, dict[str, Any]] = {
    SENSOR_S2: {
        "label_key": "sensor_s2_label",
        "pill_key": "sensor_s2_pill",
        "short_name": "Sentinel-2 L2A",
        "file_prefix": "S2L2A",
        "min_date": date(2015, 6, 23),
        "res_options": [10, 20, 30, 60],
        "default_res": 10,
    },
    SENSOR_LANDSAT: {
        "label_key": "sensor_landsat_label",
        "pill_key": "sensor_landsat_pill",
        "short_name": "Landsat C2 L2",
        "file_prefix": "LandsatC2L2",
        "min_date": date(1984, 1, 1),
        "res_options": [30, 60],
        "default_res": 30,
    },
    SENSOR_S1: {
        "label_key": "sensor_s1_label",
        "pill_key": "sensor_s1_pill",
        "short_name": "Sentinel-1 SAR RTC",
        "file_prefix": "S1RTC",
        "min_date": date(2014, 10, 1),
        "res_options": [10, 20, 30, 60],
        "default_res": 10,
    },
}

BASEMAP_KEYS: list[str] = ["satellite", "topo", "osm"]
BASEMAP_SPECS: dict[str, dict[str, Any]] = {
    "satellite": {
        "label_key": "basemap_satellite",
        "short_label_key": "basemap_satellite_short",
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri World Imagery",
        "max_zoom": 19,
    },
    "topo": {
        "label_key": "basemap_topo",
        "short_label_key": "basemap_topo_short",
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)",
        "max_zoom": 17,
    },
    "osm": {
        "label_key": "basemap_osm",
        "short_label_key": "basemap_osm_short",
        "tiles": "OpenStreetMap",
        "attr": None,
        "max_zoom": 19,
    },
}

LOC_MODE_KEYS: list[str] = ["place", "d96", "wgs84"]
LOC_MODE_I18N: dict[str, str] = {
    "place": "loc_mode_place",
    "d96": "loc_mode_d96",
    "wgs84": "loc_mode_wgs84",
}

AOI_MODE_KEYS: list[str] = ["map", "draw", "manual", "preset", "geojson"]
AOI_MODE_I18N: dict[str, str] = {
    "map": "aoi_mode_map",
    "draw": "aoi_mode_draw",
    "manual": "aoi_mode_manual",
    "preset": "aoi_mode_preset",
    "geojson": "aoi_mode_geojson",
}

CRS_MODE_KEYS: list[str] = ["auto", "d96", "utm"]
CRS_MODE_I18N: dict[str, str] = {
    "auto": "crs_mode_auto",
    "d96": "crs_mode_d96",
    "utm": "crs_mode_utm",
}
CRS_KEYS: list[str] = [CRS_D96TM, CRS_UTM33N]
CRS_I18N: dict[str, str] = {
    CRS_D96TM: "crs_d96_label",
    CRS_UTM33N: "crs_utm_label",
}
CRS_SHORT_I18N: dict[str, str] = {
    CRS_D96TM: "crs_short_d96",
    CRS_UTM33N: "crs_short_utm",
}
CRS_PILL_I18N: dict[str, str] = {
    CRS_D96TM: "crs_pill_d96",
    CRS_UTM33N: "crs_pill_utm",
}

OPTICAL_COMPOSITE_SPECS: dict[str, dict[str, Any]] = {
    "RGB": {
        "bands": ["B04", "B03", "B02"],
        "label_key": "vis_rgb_label",
        "pill_key": "vis_rgb_pill",
        "desc_key": "vis_rgb_desc",
    },
    "NIR": {
        "bands": ["B08", "B04", "B03"],
        "label_key": "vis_nir_label",
        "pill_key": "vis_nir_pill",
        "desc_key": "vis_nir_desc",
    },
    "SWIR": {
        "bands": ["B12", "B08", "B04"],
        "label_key": "vis_swir_label",
        "pill_key": "vis_swir_pill",
        "desc_key": "vis_swir_desc",
    },
}

SAR_VIS_SPECS: dict[str, dict[str, Any]] = {
    "VV": {
        "bands": ["vv"],
        "var_names": ["VV"],
        "label_key": "sar_vis_vv_label",
        "pill_key": "sar_vis_vv_pill",
        "desc_key": "sar_vis_vv_desc",
    },
    "VH": {
        "bands": ["vh"],
        "var_names": ["VH"],
        "label_key": "sar_vis_vh_label",
        "pill_key": "sar_vis_vh_pill",
        "desc_key": "sar_vis_vh_desc",
    },
    "SAR_FC": {
        "bands": ["vv", "vh"],
        "var_names": ["VV", "VH"],
        "label_key": "sar_vis_fc_label",
        "pill_key": "sar_vis_fc_pill",
        "desc_key": "sar_vis_fc_desc",
    },
}

INDEX_KEYS: list[str] = ["NDVI", "NDWI", "MNDWI", "NDBI", "NBR", "NDSI"]
SAR_METRIC_KEYS: list[str] = ["VV", "VH", "VV_VH"]

SNOW_GLACIER_PRESETS: dict[str, dict[str, Any]] = {
    "triglav_kanin": {
        "label_key": "snow_preset_triglav",
        "lat": 46.378,
        "lon": 13.836,
        "radius_m": 4000,
    },
    "pasterze": {
        "label_key": "snow_preset_pasterze",
        "lat": 47.086,
        "lon": 12.725,
        "radius_m": 5000,
    },
    "aletsch": {
        "label_key": "snow_preset_aletsch",
        "lat": 46.440,
        "lon": 8.075,
        "radius_m": 7000,
    },
}

SNOW_MASK_DISCRETE_RGBA: dict[int, tuple[int, int, int, int]] = {
    0: (226, 232, 240, 80),    # Kopno / Brez snega (#E2E8F0 subtle)
    1: (14, 165, 233, 235),    # Sneg / Led / Ledenik (#0EA5E9 crisp ice cyan)
    255: (0, 0, 0, 0),         # NoData transparent
}

RELIEF_LAYER_KEYS: list[str] = ["elevation", "slope_deg", "aspect", "hillshade"]
RELIEF_LAYER_I18N: dict[str, str] = {
    "elevation": "relief_layer_elevation",
    "slope_deg": "relief_layer_slope_deg",
    "aspect": "relief_layer_aspect",
    "hillshade": "relief_layer_hillshade",
}
PROFILE_PRESET_KEYS: list[str] = ["w_e", "s_n", "sw_ne", "custom"]
PROFILE_PRESET_I18N: dict[str, str] = {
    "w_e": "profile_preset_we",
    "s_n": "profile_preset_sn",
    "sw_ne": "profile_preset_swne",
    "custom": "profile_preset_custom",
}


def _get_optical_index_keys(sensor: str) -> list[str]:
    """Return optical index keys, including thermal LST [°C] whenever Landsat C2 L2 is active."""
    if sensor == SENSOR_LANDSAT:
        return [*INDEX_KEYS, "LST"]
    return list(INDEX_KEYS)


def _fmt_index_pill(code: str, lang: str) -> str:
    if code == "LST":
        return t("idx_LST_pill", lang=lang)
    if code == "VV_VH":
        return t("sar_VV_VH_pill", lang=lang)
    return code

INTERVAL_KEYS: list[str] = list(TEMPORAL_INTERVAL_KEYS)
INTERVAL_I18N: dict[str, str] = {
    INTERVAL_ALL: "interval_all",
    INTERVAL_14D: "interval_14d",
    INTERVAL_MONTHLY: "interval_monthly",
    INTERVAL_SEASONAL: "interval_seasonal",
    INTERVAL_YEARLY: "interval_yearly",
}

VIEWER_MODE_KEYS: list[str] = ["slippy", "plotly"]
VIEWER_MODE_I18N: dict[str, str] = {
    "slippy": "viewer_mode_slippy",
    "plotly": "viewer_mode_plotly",
}

SAR_WATER_DISCRETE_RGBA: dict[int, tuple[int, int, int, int]] = {
    0: (203, 213, 225, 130),   # Kopno / Land (#CBD5E1 semi-transparent)
    1: (3, 105, 161, 240),     # Voda / Poplava (#0369A1 crisp ocean blue)
    255: (0, 0, 0, 0),         # NoData transparent
}

CARTO_FORMAT_KEYS: list[str] = ["A4_LANDSCAPE", "A4_PORTRAIT", "A3_LANDSCAPE", "SQUARE_1_1"]
CARTO_FORMAT_I18N: dict[str, str] = {
    "A4_LANDSCAPE": "carto_fmt_a4_land",
    "A4_PORTRAIT": "carto_fmt_a4_port",
    "A3_LANDSCAPE": "carto_fmt_a3_land",
    "SQUARE_1_1": "carto_fmt_square",
}
CARTO_THEME_KEYS: list[str] = ["classic_light", "minimal_white", "dark_technical"]
CARTO_THEME_I18N: dict[str, str] = {
    "classic_light": "carto_theme_classic",
    "minimal_white": "carto_theme_minimal",
    "dark_technical": "carto_theme_dark",
}
CARTO_TITLE_POS_KEYS: list[str] = ["top_left", "top_center", "inside_top_left", "inside_top_center", "none"]
CARTO_TITLE_POS_I18N: dict[str, str] = {
    "top_left": "carto_tpos_top_left",
    "top_center": "carto_tpos_top_center",
    "inside_top_left": "carto_tpos_in_tl",
    "inside_top_center": "carto_tpos_in_tc",
    "none": "carto_tpos_none",
}
CARTO_NOTE_POS_KEYS: list[str] = ["bottom_left", "bottom_right", "below_legend", "none"]
CARTO_NOTE_POS_I18N: dict[str, str] = {
    "bottom_left": "carto_npos_bl",
    "bottom_right": "carto_npos_br",
    "below_legend": "carto_npos_leg",
    "none": "carto_npos_none",
}
CARTO_CORNER_KEYS: list[str] = ["top_right", "top_left", "bottom_right", "bottom_left"]
CARTO_CORNER_I18N: dict[str, str] = {
    "top_right": "carto_pos_tr",
    "top_left": "carto_pos_tl",
    "bottom_right": "carto_pos_br",
    "bottom_left": "carto_pos_bl",
}
CARTO_NORTH_STYLE_KEYS: list[str] = ["classic_compass", "minimal_arrow", "geodetic_star"]
CARTO_NORTH_STYLE_I18N: dict[str, str] = {
    "classic_compass": "carto_nstyle_classic",
    "minimal_arrow": "carto_nstyle_minimal",
    "geodetic_star": "carto_nstyle_star",
}
CARTO_NORTH_SIZE_KEYS: list[str] = ["small", "medium", "large"]
CARTO_NORTH_SIZE_I18N: dict[str, str] = {
    "small": "carto_nsize_small",
    "medium": "carto_nsize_medium",
    "large": "carto_nsize_large",
}
CARTO_SCALE_STYLE_KEYS: list[str] = ["alternating_bar", "simple_line"]
CARTO_SCALE_STYLE_I18N: dict[str, str] = {
    "alternating_bar": "carto_sstyle_alt",
    "simple_line": "carto_sstyle_line",
}
CARTO_LEGEND_POS_KEYS: list[str] = [
    "right_panel",
    "bottom_panel",
    "inside_bottom_right",
    "inside_top_right",
    "inside_bottom_left",
]
CARTO_LEGEND_POS_I18N: dict[str, str] = {
    "right_panel": "carto_lpos_right",
    "bottom_panel": "carto_lpos_bottom",
    "inside_bottom_right": "carto_lpos_in_br",
    "inside_top_right": "carto_lpos_in_tr",
    "inside_bottom_left": "carto_lpos_in_bl",
}
CARTO_INSET_POS_KEYS: list[str] = [
    "right_panel_bottom",
    "bottom_right",
    "top_right",
    "bottom_left",
    "top_left",
]
CARTO_INSET_POS_I18N: dict[str, str] = {
    "right_panel_bottom": "carto_ipos_right_bot",
    "bottom_right": "carto_pos_br",
    "top_right": "carto_pos_tr",
    "bottom_left": "carto_pos_bl",
    "top_left": "carto_pos_tl",
}
CARTO_GRID_OPTIONS: list[int] = [0, 250, 500, 1000, 2000, 5000]  # 0 = Auto
CARTO_GRID_TYPE_KEYS: list[str] = ["lines", "crosses", "ticks_only"]
CARTO_GRID_TYPE_I18N: dict[str, str] = {
    "lines": "carto_gtype_lines",
    "crosses": "carto_gtype_crosses",
    "ticks_only": "carto_gtype_ticks",
}
CARTO_GRID_STYLE_KEYS: list[str] = ["light", "dark", "teal"]
CARTO_GRID_STYLE_I18N: dict[str, str] = {
    "light": "carto_grid_light",
    "dark": "carto_grid_dark",
    "teal": "carto_grid_teal",
}
CARTO_AOI_COLOR_KEYS: list[str] = ["#0F766E", "#DC2626", "#FFFFFF", "#000000"]
CARTO_AOI_COLOR_I18N: dict[str, str] = {
    "#0F766E": "carto_aoi_col_teal",
    "#DC2626": "carto_aoi_col_red",
    "#FFFFFF": "carto_aoi_col_white",
    "#000000": "carto_aoi_col_black",
}

DISASTER_TYPE_KEYS: list[str] = list(DISASTER_TYPES)
DISASTER_TYPE_I18N: dict[str, str] = {
    DISASTER_FLOOD: "dis_type_flood",
    DISASTER_WILDFIRE: "dis_type_wildfire",
    DISASTER_FOREST: "dis_type_forest",
}
DISASTER_PRESET_KEYS: list[str] = list(SLOVENIA_CASE_STUDIES.keys())

WORKSPACE_KEYS: list[str] = [
    "ws_explore",
    "ws_analysis",
    "ws_temporal",
    "ws_geo",
    "ws_ai",
]
WS1_AOI_MODE_KEYS: list[str] = ["map", "draw", "preset", "geojson"]
WS1_AOI_MODE_I18N: dict[str, str] = {
    "map": "ws1_aoi_mode_map",
    "draw": "ws1_aoi_mode_draw",
    "preset": "ws1_aoi_mode_preset",
    "geojson": "ws1_aoi_mode_geojson",
}
YEARLY_SEASON_I18N: dict[str, str] = {
    "summer": "season_summer",
    "spring": "season_spring",
    "autumn": "season_autumn",
    "winter": "season_winter",
    "all_year": "season_all_year",
}


# ---------------------------------------------------------------------------
# Pomožne funkcije: stanje seje, sinhronizacija senzorja/datumov, iskanje STAC
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Inicializira spremenljivke seje."""
    today = date.today()
    default_start = today - timedelta(days=183)
    defaults: dict[str, Any] = {
        "lang": "sl",
        "active_workspace": "ws_explore",
        "pending_workspace": None,
        "active_location_label": "Ljubljana",
        "uploaded_geojson_aoi": None,
        "uploaded_geojson_name": None,
        "selected_sensor": SENSOR_S2,
        "active_sensor": SENSOR_S2,
        "sidebar_sensor_select": SENSOR_S2,
        "main_bar_sensor_select": SENSOR_S2,
        "tab3_sensor_select": SENSOR_S2,
        "search_start_date": default_start,
        "search_end_date": today,
        "sidebar_start_date": default_start,
        "sidebar_end_date": today,
        "tab3_start_date": default_start,
        "tab3_end_date": today,
        "search_max_cloud": 20,
        "sidebar_max_cloud": 20,
        "tab3_max_cloud": 20,
        "exclude_l7_slc_off": True,
        "sidebar_l7_slc_exclude": True,
        "tab3_l7_slc_exclude": True,
        "exclude_partial_swaths": True,
        "sidebar_exclude_partial": True,
        "tab3_exclude_partial": True,
        "temporal_interval": INTERVAL_ALL,
        "sidebar_interval_select": INTERVAL_ALL,
        "tab3_interval_select": INTERVAL_ALL,
        "yearly_target_season": SEASON_SUMMER,
        "sidebar_yearly_season": SEASON_SUMMER,
        "tab3_yearly_season": SEASON_SUMMER,
        "expand_pan_buffer": True,
        "sidebar_expand_pan_buffer": True,
        "map_basemap": "satellite",
        "loc_search_mode": "place",
        "sidebar_aoi_mode": "map",
        "pending_aoi_mode": None,
        "pending_reset_scene_widgets": False,
        "map_jump_version": 0,
        "map_lat": SLOVENIA_CENTER_LAT,
        "map_lon": SLOVENIA_CENTER_LON,
        "map_zoom": 10,
        "live_map_zoom": 10,
        "map_radius_m": 2000,
        "last_processed_click": None,
        "geocoded_place_msg": None,
        "drawn_polygon_4326": None,
        "draw_map_version": 0,
        "stac_items": [],
        "stac_summary_df": pd.DataFrame(),
        "stac_raw_summary_df": pd.DataFrame(),
        "stac_raw_items_by_date": {},
        "search_performed": False,
        "search_sensor": SENSOR_S2,
        "search_bbox_4326": None,
        "search_bbox_3794": None,
        "search_polygon_4326": None,
        "search_area_km2": None,
        # Tab 1 state
        "loaded_scene_ds": None,
        "loaded_scene_display_ds": None,
        "loaded_scene_overlay_cache": None,
        "loaded_scene_meta": None,
        "loaded_tif_bytes": None,
        "loaded_tif_filename": None,
        # Tab 2 state (optical, cloud-free composite & SAR)
        "tab2_index_da": None,
        "tab2_index_display_da": None,
        "tab2_index_overlay_cache": None,
        "tab2_index_stats": None,
        "tab2_index_meta": None,
        "tab2_tif_bytes": None,
        "tab2_tif_filename": None,
        "tab2_snow_mask_da": None,
        "tab2_snow_display_da": None,
        "tab2_snow_stats": None,
        "tab2_snow_tif_bytes": None,
        "tab2_snow_tif_filename": None,
        "tab2_relief_data": None,
        "tab2_relief_sent_to_carto": False,
        "sidebar_crs_mode": "auto",
        "tab2_comp_obj": None,
        "tab2_comp_meta": None,
        "tab2_comp_tif_bytes": None,
        "tab2_comp_tif_filename": None,
        "tab2_sar_ds": None,
        "tab2_sar_display_ds": None,
        "tab2_sar_meta": None,
        "tab2_sig_ds": None,
        "tab2_sig_meta": None,
        "tab2_lc_df": None,
        "tab2_lc_meta": None,
        "tab2_lc_tif_bytes": None,
        "tab2_lc_tif_filename": None,
        # Tab 3 state
        "tab3_ts_df": None,
        "tab3_ts_meta": None,
        "tab3_change_t1": None,
        "tab3_change_t2": None,
        "tab3_change_diff": None,
        "tab3_change_meta": None,
        "tab3_change_tif_bytes": None,
        "tab3_change_tif_filename": None,
        "tab3_swipe_png_t1": None,
        "tab3_swipe_png_t2": None,
        "tab3_comp_result": None,
        "tab3_timelapse_data": None,
        # Tab 4 Cartography / Print Layout state
        "tab4_carto_result": None,
        "tab4_carto_meta": None,
        "tab4_layout_config": merge_layout_config(),
        "tab4_last_render_sig": None,
        "tab4_pending_preset": None,
        "tab4_pending_layout_overrides": None,
        # Tab 5 Natural Disaster Assessment state
        "tab5_disaster_result": None,
        "tab5_last_preset": "custom",
        "tab5_sent_to_carto": False,
        # Tab 6 AI EO Assistant (Conversational Gemini Agent) state
        "tab6_ai_result": None,
        "tab6_sent_to_carto": False,
        "tab6_gemini_key": "",
        "tab6_query_input": "",
        "gemini_model": DEFAULT_GEMINI_MODEL,
        "gemini_custom_model": "",
        "gemini_discovered_models": [],
        "gemini_active_live_model": None,
        "ai_chat_messages": [],
        "ai_chat_results": {},
        "ai_last_analysis_context": None,
        "ai_pending_chat_prompt": None,
        "ai_pending_rerun_turn": None,
        "ai_editing_turn_idx": None,
        "ai_show_all_turns": False,
        "ai_api_verify_result": None,
        "ai_last_map_layout": None,
        # Tab 7 GNSS Mission Planning state
        "tab7_gnss_result": None,
        # Resolution guard
        "last_rec_res_key": None,
        "selected_resolution": 10,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _clear_loaded_products() -> None:
    """Sprosti vse naložene rastre iz seje (zaščita 1 GB RAM pri menjavi AOI/senzorja)."""
    for key in (
        "loaded_scene_ds",
        "loaded_scene_display_ds",
        "loaded_scene_overlay_cache",
        "loaded_scene_meta",
        "loaded_tif_bytes",
        "loaded_tif_filename",
        "tab2_index_da",
        "tab2_index_display_da",
        "tab2_index_overlay_cache",
        "tab2_index_stats",
        "tab2_index_meta",
        "tab2_tif_bytes",
        "tab2_tif_filename",
        "tab2_snow_mask_da",
        "tab2_snow_display_da",
        "tab2_snow_stats",
        "tab2_snow_tif_bytes",
        "tab2_snow_tif_filename",
        "tab2_relief_data",
        "tab2_comp_obj",
        "tab2_comp_meta",
        "tab2_comp_tif_bytes",
        "tab2_comp_tif_filename",
        "tab2_sar_ds",
        "tab2_sar_display_ds",
        "tab2_sar_meta",
        "tab2_sig_ds",
        "tab2_sig_meta",
        "tab2_lc_df",
        "tab2_lc_meta",
        "tab2_lc_tif_bytes",
        "tab2_lc_tif_filename",
        "tab3_ts_df",
        "tab3_ts_meta",
        "tab3_change_t1",
        "tab3_change_t2",
        "tab3_change_diff",
        "tab3_change_meta",
        "tab3_change_tif_bytes",
        "tab3_change_tif_filename",
        "tab3_swipe_png_t1",
        "tab3_swipe_png_t2",
        "tab3_comp_result",
        "tab3_timelapse_data",
        "tab4_carto_result",
        "tab4_carto_meta",
        "tab5_disaster_result",
    ):
        st.session_state[key] = None
    gc.collect()


def _collect_carto_sources(lang: str) -> list[dict[str, Any]]:
    """Samodejno odkrije vse trenutno naložene/izračunane rastre iz Zavihkov 1–3 za izdelavo karte."""
    sources: list[dict[str, Any]] = []

    # 1. Tab 1: Loaded satellite scene (RGB / NIR / SWIR / SAR VV / VH / SAR_FC)
    ds_t1 = st.session_state.get("loaded_scene_ds")
    meta_t1 = st.session_state.get("loaded_scene_meta")
    if ds_t1 is not None and isinstance(meta_t1, dict):
        vis_code = str(meta_t1.get("vis_code", "RGB"))
        is_sar = meta_t1.get("sensor") == SENSOR_S1
        bands_t1 = list(meta_t1.get("bands", []))
        vis_label = (
            t(SAR_VIS_SPECS[vis_code]["pill_key"], lang=lang)
            if (is_sar and vis_code in SAR_VIS_SPECS)
            else t(OPTICAL_COMPOSITE_SPECS.get(vis_code, OPTICAL_COMPOSITE_SPECS["RGB"])["pill_key"], lang=lang)
        )
        is_rgb_t1 = (not is_sar) or (vis_code == "SAR_FC") or (len(bands_t1) >= 2 and set(bands_t1) == {"VV", "VH"})
        sar_spec = SAR_METRICS.get(vis_code, SAR_METRICS["VV"]) if (is_sar and not is_rgb_t1) else None
        sources.append(
            {
                "id": "tab1_scene",
                "label": t(
                    "carto_layer_tab1",
                    lang=lang,
                    sensor=meta_t1.get("sensor_name", "Satellite"),
                    vis=vis_label,
                    date=meta_t1.get("date_str", ""),
                ),
                "raster": ds_t1,
                "crs": meta_t1.get("crs", CRS_D96TM),
                "is_rgb": is_rgb_t1,
                "bands": bands_t1,
                "use_percentile_stretch": bool(st.session_state.get("tab1_stretch", True)),
                "gamma": float(st.session_state.get("tab1_gamma", 1.0)),
                "cmap_name": sar_spec["cmap"] if sar_spec else "Greys_r",
                "vmin": float(sar_spec["vmin"]) if sar_spec else -22.0,
                "vmax": float(sar_spec["vmax"]) if sar_spec else -2.0,
                "discrete_items": None,
                "legend_title": vis_label,
                "legend_units": sar_spec["units"] if sar_spec else "",
                "stats_summary": [
                    (t("col_tile", lang=lang), str(meta_t1.get("mgrs_tile", ""))),
                    (
                        t("col_cloud", lang=lang),
                        f"{meta_t1['cloud_cover']:.1f} %"
                        if meta_t1.get("cloud_cover") is not None
                        else "SAR",
                    ),
                ],
                "default_title": f"{meta_t1.get('sensor_name', 'Satellite')} — {vis_label}",
                "default_subtitle": f"{meta_t1.get('date_str', '')} | {meta_t1.get('mgrs_tile', '')}",
                "sensor_label": str(meta_t1.get("sensor_name", "Satellite")),
                "date_label": str(meta_t1.get("date_str", "")),
                "resolution_m": float(meta_t1.get("resolution_m", 10.0)),
                "tif_bytes": st.session_state.get("loaded_tif_bytes"),
                "tif_filename": st.session_state.get("loaded_tif_filename") or "satellite_scene.tif",
                "file_stem": f"{meta_t1.get('sensor', 'sat')}_{vis_code}_{meta_t1.get('date_str', 'map')}",
            }
        )

    # 2. Tab 2B.1: Spectral Index
    idx_da = st.session_state.get("tab2_index_da")
    idx_meta = st.session_state.get("tab2_index_meta")
    idx_stats = st.session_state.get("tab2_index_stats")
    if idx_da is not None and isinstance(idx_meta, dict):
        idx_key = str(idx_meta.get("index_key", "NDVI"))
        spec = SPECTRAL_INDICES.get(idx_key, SPECTRAL_INDICES["NDVI"])
        stats_rows: list[tuple[str, str]] = []
        if isinstance(idx_stats, dict) and idx_stats.get("valid_pixels", 0) > 0:
            stats_rows = [
                ("Min", f"{idx_stats['min']:.3f}"),
                ("Mean" if lang == "en" else "Povprečje", f"{idx_stats['mean']:.3f}"),
                ("Median" if lang == "en" else "Mediana", f"{idx_stats['median']:.3f}"),
                ("Max", f"{idx_stats['max']:.3f}"),
                (t("m_valid_ha", lang=lang), f"{idx_stats['valid_area_ha']:,.1f} ha"),
            ]
        sources.append(
            {
                "id": "tab2_index",
                "label": t(
                    "carto_layer_tab2_idx",
                    lang=lang,
                    idx=idx_key,
                    sensor=idx_meta.get("sensor_name", "Optical"),
                    date=idx_meta.get("date_str", ""),
                ),
                "raster": idx_da,
                "crs": idx_meta.get("crs", CRS_D96TM),
                "is_rgb": False,
                "bands": None,
                "use_percentile_stretch": False,
                "gamma": 1.0,
                "cmap_name": spec["cmap"],
                "vmin": float(spec["vmin"]),
                "vmax": float(spec["vmax"]),
                "discrete_items": None,
                "legend_title": f"{idx_key} ({spec['formula']})",
                "legend_units": str(spec.get("units", "")),
                "stats_summary": stats_rows,
                "default_title": f"{idx_key} — {idx_meta.get('sensor_name', 'Satellite')}",
                "default_subtitle": f"{idx_meta.get('date_str', '')} | {idx_meta.get('mgrs_tile', '')}",
                "sensor_label": str(idx_meta.get("sensor_name", "Optical")),
                "date_label": str(idx_meta.get("date_str", "")),
                "resolution_m": float(idx_meta.get("resolution_m", 10.0)),
                "tif_bytes": st.session_state.get("tab2_tif_bytes"),
                "tif_filename": st.session_state.get("tab2_tif_filename") or f"{idx_key}.tif",
                "file_stem": f"{idx_key}_{idx_meta.get('date_str', 'map')}",
            }
        )

    # 2b. Tab 2B.1: Snow / Ice / Glacier Mask (NDSI)
    snow_mask_da = st.session_state.get("tab2_snow_mask_da")
    snow_stats = st.session_state.get("tab2_snow_stats")
    if snow_mask_da is not None and isinstance(snow_stats, dict) and isinstance(idx_meta, dict):
        sn_thr = float(snow_stats.get("ndsi_thresh", 0.40))
        sn_rows: list[tuple[str, str]] = [
            (t("snow_kpi_area", lang=lang), f"{snow_stats['snow_area_km2']:,.2f} km² ({snow_stats['snow_area_ha']:,.1f} ha)"),
            (t("snow_kpi_share", lang=lang), f"{snow_stats['snow_pct']:.1f} %"),
        ]
        if snow_stats.get("dem_available") and snow_stats.get("snowline_p10_m") is not None:
            sn_rows.append(
                (t("snow_kpi_snowline", lang=lang), f"{snow_stats['snowline_p10_m']:,.0f} m")
            )
        sources.append(
            {
                "id": "tab2_snow_mask",
                "label": t("carto_layer_tab2_snow", lang=lang, thr=sn_thr, date=idx_meta.get("date_str", "")),
                "raster": snow_mask_da,
                "crs": idx_meta.get("crs", CRS_D96TM),
                "is_rgb": False,
                "bands": None,
                "use_percentile_stretch": False,
                "gamma": 1.0,
                "cmap_name": "Blues",
                "vmin": 0.0,
                "vmax": 1.0,
                "discrete_items": [
                    (0, t("snow_mask_legend_free", lang=lang), "#E2E8F0"),
                    (1, t("snow_mask_legend_snow", lang=lang), "#0EA5E9"),
                ],
                "legend_title": f"NDSI ≥ {sn_thr:.2f}",
                "legend_units": "",
                "stats_summary": sn_rows,
                "default_title": f"Snow / Ice Extent (NDSI ≥ {sn_thr:.2f})" if lang == "en" else f"Obseg snega in ledu (NDSI ≥ {sn_thr:.2f})",
                "default_subtitle": f"{idx_meta.get('sensor_name', 'Optical')} | {idx_meta.get('date_str', '')}",
                "sensor_label": str(idx_meta.get("sensor_name", "Optical")),
                "date_label": str(idx_meta.get("date_str", "")),
                "resolution_m": float(idx_meta.get("resolution_m", 10.0)),
                "tif_bytes": st.session_state.get("tab2_snow_tif_bytes"),
                "tif_filename": st.session_state.get("tab2_snow_tif_filename") or "snow_mask.tif",
                "file_stem": f"SnowMask_NDSI_{idx_meta.get('date_str', 'map')}",
            }
        )

    # 2c. Tab 2: Copernicus DEM 30 m Terrain Layer (Elevation / Slope / Aspect / Hillshade)
    relief_data = st.session_state.get("tab2_relief_data")
    if isinstance(relief_data, dict) and relief_data.get("elevation") is not None:
        r_layer_key = str(st.session_state.get("tab2_relief_layer_sel", "elevation"))
        if r_layer_key not in RELIEF_LAYER_KEYS:
            r_layer_key = "elevation"
        r_da = relief_data[r_layer_key]
        r_stats = relief_data.get("stats", {})
        r_spec = TERRAIN_LAYER_SPECS[r_layer_key]
        r_crs = str(relief_data.get("crs", CRS_D96TM))
        r_lbl = t(RELIEF_LAYER_I18N[r_layer_key], lang=lang)
        if r_spec.get("auto_range"):
            r_vmin = float(r_stats.get("min_h", 0.0))
            r_vmax = max(r_vmin + 1.0, float(r_stats.get("max_h", 2500.0)))
        else:
            r_vmin = float(r_spec["vmin"])
            r_vmax = float(r_spec["vmax"])
        r_rows: list[tuple[str, str]] = [
            ("Min / Max [m]", f"{r_stats.get('min_h', 0.0):,.0f} – {r_stats.get('max_h', 0.0):,.0f} m"),
            ("ΔH", f"{r_stats.get('relief_diff_m', 0.0):,.0f} m"),
            (t("relief_kpi_slope", lang=lang), f"{r_stats.get('mean_slope_deg', 0.0):.1f}° / {r_stats.get('max_slope_deg', 0.0):.1f}°"),
            (t("relief_kpi_steep20", lang=lang), f"{r_stats.get('steep_20_ha', 0.0):,.1f} ha ({r_stats.get('steep_20_pct', 0.0):.1f}%)"),
        ]
        sources.append(
            {
                "id": "tab2_relief",
                "label": t("carto_layer_tab2_relief", lang=lang, layer=r_lbl, crs=format_crs_label(r_crs, short=True)),
                "raster": r_da,
                "crs": r_crs,
                "is_rgb": False,
                "bands": None,
                "use_percentile_stretch": False,
                "gamma": 1.0,
                "cmap_name": str(r_spec["cmap"]),
                "vmin": r_vmin,
                "vmax": r_vmax,
                "discrete_items": None,
                "legend_title": r_lbl,
                "legend_units": str(r_spec.get("units", "")),
                "stats_summary": r_rows,
                "default_title": f"Copernicus DEM — {r_lbl}",
                "default_subtitle": f"GLO-30 (30 m) | {format_crs_label(r_crs)}",
                "sensor_label": "Copernicus DEM 30 m",
                "date_label": "GLO-30",
                "resolution_m": float(r_stats.get("resolution_m", 30.0)),
                "tif_bytes": None,
                "tif_filename": f"CopernicusDEM_{r_layer_key}_{r_crs.replace('EPSG:', 'EPSG')}_30m.tif",
                "file_stem": f"CopernicusDEM_{r_layer_key}",
            }
        )

    # 3. Tab 2A: SAR Water / Flood Mask
    sar_ds = st.session_state.get("tab2_sar_ds")
    sar_meta = st.session_state.get("tab2_sar_meta")
    if sar_ds is not None and isinstance(sar_meta, dict):
        thr_db = float(st.session_state.get("tab2_sar_water_thr", -16.0))
        mask_da, water_stats = compute_sar_water_mask(
            sar_ds,
            threshold_db=thr_db,
            polarization="VV",
            resolution_m=float(sar_meta.get("resolution_m", 10.0)),
        )
        water_tif = export_to_geotiff_bytes(mask_da, target_epsg=sar_meta.get("crs", CRS_D96TM))
        sources.append(
            {
                "id": "tab2_sar_water",
                "label": t(
                    "carto_layer_tab2_sar",
                    lang=lang,
                    thr=thr_db,
                    date=sar_meta.get("date_str", ""),
                ),
                "raster": mask_da,
                "crs": sar_meta.get("crs", CRS_D96TM),
                "is_rgb": False,
                "bands": None,
                "use_percentile_stretch": False,
                "gamma": 1.0,
                "cmap_name": "Greys",
                "vmin": 0.0,
                "vmax": 1.0,
                "discrete_items": [
                    (1, f"{t('sar_water_legend_water', lang=lang)} (VV < {thr_db:.1f} dB)", "#0369A1"),
                    (0, t("sar_water_legend_land", lang=lang), "#CBD5E1"),
                ],
                "legend_title": t("sar_water_legend_water", lang=lang),
                "legend_units": "",
                "stats_summary": [
                    (t("m_water_ha", lang=lang), f"{water_stats['water_area_ha']:,.2f} ha"),
                    (t("m_water_pct", lang=lang), f"{water_stats['water_pct']:.2f} %"),
                    (t("m_valid_ha", lang=lang), f"{water_stats['valid_area_ha']:,.2f} ha"),
                ],
                "default_title": t("sar_water_map_title", lang=lang, thr=thr_db, date=sar_meta.get("date_str", "")),
                "default_subtitle": f"Sentinel-1 SAR RTC | {sar_meta.get('mgrs_tile', '')}",
                "sensor_label": "Sentinel-1 SAR RTC",
                "date_label": str(sar_meta.get("date_str", "")),
                "resolution_m": float(sar_meta.get("resolution_m", 10.0)),
                "tif_bytes": water_tif,
                "tif_filename": f"S1_WaterMask_{sar_meta.get('date_str', 'map')}.tif",
                "file_stem": f"S1_WaterMask_{sar_meta.get('date_str', 'map')}",
            }
        )

    # 4. Tab 2B.2: Cloud-Free Median Composite
    comp_obj = st.session_state.get("tab2_comp_obj")
    comp_meta = st.session_state.get("tab2_comp_meta")
    if comp_obj is not None and isinstance(comp_meta, dict):
        comp_mode = str(comp_meta.get("mode", "RGB_NIR"))
        d1 = str(comp_meta.get("date_start", ""))
        d2 = str(comp_meta.get("date_end", ""))
        if comp_mode == "RGB_NIR" and isinstance(comp_obj, xr.Dataset):
            comp_disp = str(st.session_state.get("tab2_comp_rgb_nir_toggle", "RGB"))
            comp_bands = ["B04", "B03", "B02"] if comp_disp == "RGB" else ["B08", "B04", "B03"]
            sources.append(
                {
                    "id": "tab2_composite",
                    "label": t("carto_layer_tab2_comp", lang=lang, mode=comp_disp, d1=d1, d2=d2),
                    "raster": comp_obj,
                    "crs": comp_meta.get("crs", CRS_D96TM),
                    "is_rgb": True,
                    "bands": comp_bands,
                    "use_percentile_stretch": bool(st.session_state.get("tab2_comp_stretch", True)),
                    "gamma": float(st.session_state.get("tab2_comp_gamma", 1.0)),
                    "cmap_name": "RdYlGn",
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "discrete_items": None,
                    "legend_title": f"{comp_disp} Median",
                    "legend_units": "",
                    "stats_summary": [
                        (t("comp_m_scenes", lang=lang), f"{comp_meta.get('scenes_used', 0)}"),
                        (t("m_valid_pct", lang=lang), f"{comp_meta.get('valid_pct', 0.0):.1f} %"),
                    ],
                    "default_title": f"{comp_meta.get('sensor_name', 'Satellite')} — {comp_disp} Median",
                    "default_subtitle": f"{d1} – {d2} ({comp_meta.get('scenes_used', 0)} scenes)",
                    "sensor_label": str(comp_meta.get("sensor_name", "Optical")),
                    "date_label": f"{d1} – {d2}",
                    "resolution_m": float(comp_meta.get("resolution_m", 10.0)),
                    "tif_bytes": st.session_state.get("tab2_comp_tif_bytes"),
                    "tif_filename": st.session_state.get("tab2_comp_tif_filename") or f"composite_{comp_disp}.tif",
                    "file_stem": f"Composite_{comp_disp}_{d1}_{d2}",
                }
            )
        elif isinstance(comp_obj, xr.DataArray):
            spec_c = SPECTRAL_INDICES.get(comp_mode, SPECTRAL_INDICES["NDVI"])
            sources.append(
                {
                    "id": "tab2_composite",
                    "label": t("carto_layer_tab2_comp", lang=lang, mode=comp_mode, d1=d1, d2=d2),
                    "raster": comp_obj,
                    "crs": comp_meta.get("crs", CRS_D96TM),
                    "is_rgb": False,
                    "bands": None,
                    "use_percentile_stretch": False,
                    "gamma": 1.0,
                    "cmap_name": spec_c["cmap"],
                    "vmin": float(spec_c["vmin"]),
                    "vmax": float(spec_c["vmax"]),
                    "discrete_items": None,
                    "legend_title": f"{comp_mode} Median",
                    "legend_units": "",
                    "stats_summary": [
                        (t("comp_m_scenes", lang=lang), f"{comp_meta.get('scenes_used', 0)}"),
                        (t("m_valid_pct", lang=lang), f"{comp_meta.get('valid_pct', 0.0):.1f} %"),
                    ],
                    "default_title": f"{comp_mode} Median — {comp_meta.get('sensor_name', 'Satellite')}",
                    "default_subtitle": f"{d1} – {d2}",
                    "sensor_label": str(comp_meta.get("sensor_name", "Optical")),
                    "date_label": f"{d1} – {d2}",
                    "resolution_m": float(comp_meta.get("resolution_m", 10.0)),
                    "tif_bytes": st.session_state.get("tab2_comp_tif_bytes"),
                    "tif_filename": st.session_state.get("tab2_comp_tif_filename") or f"composite_{comp_mode}.tif",
                    "file_stem": f"Composite_{comp_mode}_{d1}_{d2}",
                }
            )

    # 5. Tab 3: Change Detection (Δ Map)
    diff_da = st.session_state.get("tab3_change_diff")
    ch_meta = st.session_state.get("tab3_change_meta")
    if diff_da is not None and isinstance(ch_meta, dict):
        ch_idx = str(ch_meta.get("index_key", "NDVI"))
        is_sar_ch = ch_meta.get("sensor") == SENSOR_S1
        ch_spec = SAR_METRICS.get(ch_idx) if is_sar_ch else SPECTRAL_INDICES.get(ch_idx, SPECTRAL_INDICES["NDVI"])
        diff_cmap = ch_spec["diff_cmap"] if ch_spec else "RdBu"
        lim = 6.0 if is_sar_ch else 0.5
        units_ch = "dB" if is_sar_ch else ""
        d_t1 = str(ch_meta.get("date_t1", ""))
        d_t2 = str(ch_meta.get("date_t2", ""))
        s_cfg = SENSOR_META.get(ch_meta.get("sensor", SENSOR_S2), SENSOR_META[SENSOR_S2])
        sources.append(
            {
                "id": "tab3_change",
                "label": t("carto_layer_tab3_change", lang=lang, idx=ch_idx, d1=d_t1, d2=d_t2),
                "raster": diff_da,
                "crs": ch_meta.get("crs", CRS_D96TM),
                "is_rgb": False,
                "bands": None,
                "use_percentile_stretch": False,
                "gamma": 1.0,
                "cmap_name": diff_cmap,
                "vmin": -lim,
                "vmax": lim,
                "discrete_items": None,
                "legend_title": f"Δ{ch_idx} (T2 − T1)",
                "legend_units": units_ch,
                "stats_summary": [
                    ("T1", d_t1),
                    ("T2", d_t2),
                ],
                "default_title": f"Δ{ch_idx} ({d_t1} → {d_t2})",
                "default_subtitle": s_cfg["short_name"],
                "sensor_label": s_cfg["short_name"],
                "date_label": f"{d_t1} → {d_t2}",
                "resolution_m": float(ch_meta.get("resolution_m", 10.0)),
                "tif_bytes": st.session_state.get("tab3_change_tif_bytes"),
                "tif_filename": st.session_state.get("tab3_change_tif_filename") or f"delta_{ch_idx}.tif",
                "file_stem": f"Delta_{ch_idx}_{d_t1}_to_{d_t2}",
            }
        )

    # 6. Tab 5: Natural Disaster Impact Map (Flood SAR / Wildfire dNBR / Forest dNDVI)
    dis_res = st.session_state.get("tab5_disaster_result")
    if isinstance(dis_res, dict) and dis_res.get("class_da") is not None:
        d_type = str(dis_res.get("disaster_type", DISASTER_FLOOD))
        d_type_lbl = t(DISASTER_TYPE_I18N.get(d_type, "dis_type_flood"), lang=lang)
        d_pre = str(dis_res.get("date_pre", ""))
        d_ev = str(dis_res.get("date_event", ""))
        d_sum = dis_res.get("summary", {})
        s_cfg_d = SENSOR_META.get(dis_res.get("sensor", SENSOR_S1), SENSOR_META[SENSOR_S1])
        dis_stats_rows: list[tuple[str, str]] = [
            (
                t("dis_kpi_affected_ha", lang=lang),
                f"{d_sum.get('affected_area_ha', 0.0):,.2f} ha",
            ),
            (
                t("dis_kpi_affected_pct", lang=lang),
                f"{d_sum.get('affected_pct', 0.0):.2f} %",
            ),
        ]
        if d_type == DISASTER_FLOOD:
            dis_stats_rows.append(
                (
                    "Stalna voda" if lang == "sl" else "Permanent water",
                    f"{d_sum.get('permanent_water_ha', 0.0):,.2f} ha",
                )
            )
        sources.append(
            {
                "id": "tab5_disaster",
                "label": t("carto_layer_tab5_disaster", lang=lang, dtype=d_type_lbl, d1=d_pre, d2=d_ev),
                "raster": dis_res["class_da"],
                "crs": dis_res.get("crs", CRS_D96TM),
                "is_rgb": False,
                "bands": None,
                "use_percentile_stretch": False,
                "gamma": 1.0,
                "cmap_name": "Greys",
                "vmin": 0.0,
                "vmax": 4.0,
                "discrete_items": build_carto_discrete_items(d_type, lang=lang),
                "legend_title": d_type_lbl,
                "legend_units": "",
                "stats_summary": dis_stats_rows,
                "default_title": f"{d_type_lbl} — {d_ev}",
                "default_subtitle": f"{s_cfg_d['short_name']} | {d_pre} → {d_ev}",
                "sensor_label": s_cfg_d["short_name"],
                "date_label": f"{d_pre} → {d_ev}",
                "resolution_m": float(dis_res.get("resolution_m", 20.0)),
                "tif_bytes": dis_res.get("tif_bytes"),
                "tif_filename": dis_res.get("tif_filename") or f"disaster_{d_type}.tif",
                "file_stem": f"Disaster_{d_type}_{d_pre}_to_{d_ev}",
                "aoi_bbox_4326": dis_res.get("bbox_4326"),
                "aoi_polygon_4326": dis_res.get("polygon_4326"),
            }
        )

    # 7. Tab 6: AI EO Assistant Raster Output (all spatial AI analyses)
    ai_res = st.session_state.get("tab6_ai_result")
    if isinstance(ai_res, dict) and ai_res.get("kind") in (
        "disaster",
        "spectral_index",
        "scene_rgb",
        "cloud_free_composite",
        "change_comparison",
        "terrain_dem",
    ):
        ai_plan = ai_res["plan"]
        ai_place = str(ai_plan.get("place_name", "AOI"))
        ai_crs = str(ai_res.get("crs", CRS_D96TM))
        ai_sensor_cfg = SENSOR_META.get(ai_plan.get("sensor", SENSOR_S2), SENSOR_META[SENSOR_S2])
        ai_kind = ai_res["kind"]

        if ai_kind == "disaster":
            d_obj = ai_res["disaster_result"]
            dt = str(d_obj["disaster_type"])
            dt_lbl = t(DISASTER_TYPE_I18N.get(dt, "dis_type_flood"), lang=lang)
            ds_sum = d_obj["summary"]
            sources.append(
                {
                    "id": "tab6_ai",
                    "label": t("carto_layer_tab6_ai", lang=lang, title=f"{dt_lbl} — {ai_place}"),
                    "raster": d_obj["class_da"],
                    "crs": ai_crs,
                    "is_rgb": False,
                    "bands": None,
                    "use_percentile_stretch": False,
                    "gamma": 1.0,
                    "cmap_name": "Greys",
                    "vmin": 0.0,
                    "vmax": 4.0,
                    "discrete_items": build_carto_discrete_items(dt, lang=lang),
                    "legend_title": dt_lbl,
                    "legend_units": "",
                    "stats_summary": [
                        (t("dis_kpi_affected_ha", lang=lang), f"{ds_sum['affected_area_ha']:,.2f} ha"),
                        (t("dis_kpi_affected_pct", lang=lang), f"{ds_sum['affected_pct']:.2f} %"),
                    ],
                    "default_title": f"{dt_lbl} — {ai_place}",
                    "default_subtitle": f"{ai_sensor_cfg['short_name']} | {d_obj['date_pre']} → {d_obj['date_event']}",
                    "sensor_label": ai_sensor_cfg["short_name"],
                    "date_label": f"{d_obj['date_pre']} → {d_obj['date_event']}",
                    "resolution_m": float(d_obj.get("resolution_m", 20.0)),
                    "tif_bytes": d_obj.get("tif_bytes"),
                    "tif_filename": d_obj.get("tif_filename") or f"AI_{dt}.tif",
                    "file_stem": f"AI_{dt}_{ai_place}_{d_obj['date_event']}",
                    "aoi_bbox_4326": ai_res.get("bbox_4326"),
                    "aoi_polygon_4326": None,
                }
            )
        elif ai_kind == "spectral_index":
            idx_n = str(ai_plan.get("index_name", "NDVI"))
            is_sar_ai = ai_plan.get("sensor") == SENSOR_S1
            spec_ai = SAR_METRICS.get(idx_n, SAR_METRICS["VV"]) if is_sar_ai else SPECTRAL_INDICES.get(idx_n, SPECTRAL_INDICES["NDVI"])
            st_ai = ai_res.get("stats", {})
            sources.append(
                {
                    "id": "tab6_ai",
                    "label": t("carto_layer_tab6_ai", lang=lang, title=f"{idx_n} — {ai_place} ({ai_res.get('date_str', '')})"),
                    "raster": ai_res["index_da"],
                    "crs": ai_crs,
                    "is_rgb": False,
                    "bands": None,
                    "use_percentile_stretch": False,
                    "gamma": 1.0,
                    "cmap_name": spec_ai["cmap"],
                    "vmin": float(spec_ai["vmin"]),
                    "vmax": float(spec_ai["vmax"]),
                    "discrete_items": None,
                    "legend_title": idx_n,
                    "legend_units": spec_ai.get("units", ""),
                    "stats_summary": [
                        ("Mean" if lang == "en" else "Povprečje", f"{st_ai.get('mean', 0.0):.3f}"),
                        ("Median" if lang == "en" else "Mediana", f"{st_ai.get('median', 0.0):.3f}"),
                        (t("m_valid_ha", lang=lang), f"{st_ai.get('valid_area_ha', 0.0):,.1f} ha"),
                    ],
                    "default_title": f"{idx_n} — {ai_place}",
                    "default_subtitle": f"{ai_sensor_cfg['short_name']} | {ai_res.get('date_str', '')}",
                    "sensor_label": ai_sensor_cfg["short_name"],
                    "date_label": str(ai_res.get("date_str", "")),
                    "resolution_m": float(ai_plan.get("resolution_m", 10.0)),
                    "tif_bytes": ai_res.get("tif_bytes"),
                    "tif_filename": ai_res.get("tif_filename") or f"AI_{idx_n}.tif",
                    "file_stem": f"AI_{idx_n}_{ai_place}_{ai_res.get('date_str', '')}",
                    "aoi_bbox_4326": ai_res.get("bbox_4326"),
                    "aoi_polygon_4326": None,
                }
            )
        elif ai_kind == "scene_rgb":
            sources.append(
                {
                    "id": "tab6_ai",
                    "label": t("carto_layer_tab6_ai", lang=lang, title=f"RGB — {ai_place} ({ai_res.get('date_str', '')})"),
                    "raster": ai_res["scene_ds"],
                    "crs": ai_crs,
                    "is_rgb": True,
                    "bands": ai_res.get("bands", ["B04", "B03", "B02"]),
                    "use_percentile_stretch": True,
                    "gamma": 1.0,
                    "cmap_name": "Greys_r",
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "discrete_items": None,
                    "legend_title": "RGB" if ai_plan.get("sensor") != SENSOR_S1 else "SAR VV+VH",
                    "legend_units": "",
                    "stats_summary": [],
                    "default_title": f"{ai_sensor_cfg['short_name']} — {ai_place}",
                    "default_subtitle": f"{ai_res.get('date_str', '')} | {ai_res.get('tile', '')}",
                    "sensor_label": ai_sensor_cfg["short_name"],
                    "date_label": str(ai_res.get("date_str", "")),
                    "resolution_m": float(ai_plan.get("resolution_m", 10.0)),
                    "tif_bytes": ai_res.get("tif_bytes"),
                    "tif_filename": ai_res.get("tif_filename") or f"AI_RGB_{ai_place}.tif",
                    "file_stem": f"AI_RGB_{ai_place}_{ai_res.get('date_str', '')}",
                    "aoi_bbox_4326": ai_res.get("bbox_4326"),
                    "aoi_polygon_4326": None,
                }
            )
        elif ai_kind == "cloud_free_composite":
            cm = ai_res.get("composite_meta") or {}
            d_span = f"{cm.get('date_start', '')} → {cm.get('date_end', '')}"
            sources.append(
                {
                    "id": "tab6_ai",
                    "label": t("carto_layer_tab6_ai", lang=lang, title=f"Composite RGB — {ai_place} ({d_span})"),
                    "raster": ai_res["scene_ds"],
                    "crs": ai_crs,
                    "is_rgb": True,
                    "bands": ai_res.get("bands", ["B04", "B03", "B02"]),
                    "use_percentile_stretch": True,
                    "gamma": 1.0,
                    "cmap_name": "Greys_r",
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "discrete_items": None,
                    "legend_title": "Cloud-Free Median RGB",
                    "legend_units": "",
                    "stats_summary": [
                        ("Posnetki" if lang == "sl" else "Scenes", f"{cm.get('scenes_used', 0)}"),
                        (t("m_valid_ha", lang=lang), f"{float(cm.get('valid_area_ha', 0.0)):,.1f} ha"),
                    ],
                    "default_title": f"Brezoblačni mozaik — {ai_place}" if lang == "sl" else f"Cloud-Free Composite — {ai_place}",
                    "default_subtitle": f"{ai_sensor_cfg['short_name']} | {d_span}",
                    "sensor_label": ai_sensor_cfg["short_name"],
                    "date_label": d_span,
                    "resolution_m": float(ai_plan.get("resolution_m", 10.0)),
                    "tif_bytes": ai_res.get("tif_bytes"),
                    "tif_filename": ai_res.get("tif_filename") or f"AI_Composite_{ai_place}.tif",
                    "file_stem": f"AI_Composite_{ai_place}",
                    "aoi_bbox_4326": ai_res.get("bbox_4326"),
                    "aoi_polygon_4326": None,
                }
            )
        elif ai_kind == "change_comparison" and ai_res.get("index_da") is not None:
            comp_r = ai_res.get("comparison_result") or {}
            c_dates = comp_r.get("dates") or ["t1", "t2"]
            d_span = f"{c_dates[0]} → {c_dates[-1]}"
            idx_n = str(ai_plan.get("index_name", "NDVI"))
            cs = ai_res.get("change_summary") or {}
            sources.append(
                {
                    "id": "tab6_ai",
                    "label": t("carto_layer_tab6_ai", lang=lang, title=f"Δ {idx_n} — {ai_place} ({d_span})"),
                    "raster": ai_res["index_da"],
                    "crs": ai_crs,
                    "is_rgb": False,
                    "bands": None,
                    "use_percentile_stretch": False,
                    "gamma": 1.0,
                    "cmap_name": "RdBu",
                    "vmin": -0.5 if idx_n != "LST" else -10.0,
                    "vmax": 0.5 if idx_n != "LST" else 10.0,
                    "discrete_items": None,
                    "legend_title": f"Δ {idx_n}",
                    "legend_units": "°C" if idx_n == "LST" else ("dB" if ai_plan.get("sensor") == SENSOR_S1 else ""),
                    "stats_summary": [
                        ("Δ Mean", f"{float(cs.get('mean', 0.0)):+.3f}"),
                        ("+" if lang == "sl" else "Gain", f"{float(cs.get('pos_area_ha', 0.0)):,.1f} ha"),
                        ("-" if lang == "sl" else "Loss", f"{float(cs.get('neg_area_ha', 0.0)):,.1f} ha"),
                    ],
                    "default_title": f"Δ {idx_n} — {ai_place}",
                    "default_subtitle": f"{ai_sensor_cfg['short_name']} | {d_span}",
                    "sensor_label": ai_sensor_cfg["short_name"],
                    "date_label": d_span,
                    "resolution_m": float(ai_plan.get("resolution_m", 10.0)),
                    "tif_bytes": ai_res.get("tif_bytes"),
                    "tif_filename": ai_res.get("tif_filename") or f"AI_Delta_{idx_n}_{ai_place}.tif",
                    "file_stem": f"AI_Delta_{idx_n}_{ai_place}",
                    "aoi_bbox_4326": ai_res.get("bbox_4326"),
                    "aoi_polygon_4326": None,
                }
            )
        elif ai_kind == "terrain_dem" and ai_res.get("index_da") is not None:
            ds_t = ai_res.get("dem_stats") or {}
            min_h = float(ds_t.get("min_h", 0.0))
            max_h = max(float(ds_t.get("max_h", 1000.0)), min_h + 10.0)
            sources.append(
                {
                    "id": "tab6_ai",
                    "label": t("carto_layer_tab6_ai", lang=lang, title=f"Copernicus DEM 30 m — {ai_place}"),
                    "raster": ai_res["index_da"],
                    "crs": ai_crs,
                    "is_rgb": False,
                    "bands": None,
                    "use_percentile_stretch": False,
                    "gamma": 1.0,
                    "cmap_name": "terrain",
                    "vmin": min_h,
                    "vmax": max_h,
                    "discrete_items": None,
                    "legend_title": "Nadmorska višina" if lang == "sl" else "Elevation",
                    "legend_units": "m",
                    "stats_summary": [
                        ("Min–Max", f"{min_h:,.0f}–{max_h:,.0f} m"),
                        ("Naklon" if lang == "sl" else "Mean Slope", f"{float(ds_t.get('mean_slope_deg', 0.0)):.1f}°"),
                    ],
                    "default_title": f"Copernicus DEM 30 m — {ai_place}",
                    "default_subtitle": "Copernicus GLO-30 | 30 m",
                    "sensor_label": "Copernicus DEM 30 m",
                    "date_label": "GLO-30",
                    "resolution_m": 30.0,
                    "tif_bytes": ai_res.get("tif_bytes"),
                    "tif_filename": ai_res.get("tif_filename") or f"AI_DEM30m_{ai_place}.tif",
                    "file_stem": f"AI_DEM30m_{ai_place}",
                    "aoi_bbox_4326": ai_res.get("bbox_4326"),
                    "aoi_polygon_4326": None,
                }
            )

    return sources


def _clip_raster_to_exact_aoi(
    obj: xr.DataArray | xr.Dataset,
    bbox_4326: Sequence[float] | None,
    polygon_4326: dict[str, Any] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Izreže razširjeni (+35 %) raster natančno na izvirni AOI za točne površine, statistiko in GeoTIFF."""
    if polygon_4326 is not None:
        return clip_to_aoi_polygon(obj, polygon_4326, drop=True)
    if bbox_4326 is not None and len(bbox_4326) == 4:
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox_4326)
        box_geom = mapping(shapely_box(min_lon, min_lat, max_lon, max_lat))
        return clip_to_aoi_polygon(obj, box_geom, drop=True)
    return obj


def _get_tab1_overlay_entry(
    *,
    ds_disp: xr.Dataset,
    meta: dict[str, Any],
    use_percentile: bool,
    gamma: float,
    lang: str,
) -> tuple[str, list[list[float]], str]:
    """Vrne (in predpomni v seji) PNG prekrivni sloj v EPSG:4326 za naložen posnetek iz Zavihka 1."""
    is_sar = meta["sensor"] == SENSOR_S1
    vis_code = str(meta["vis_code"])
    cache_sig = (
        meta["scene_id"],
        vis_code,
        meta["crs"],
        meta["resolution_m"],
        bool(use_percentile),
        round(float(gamma), 2),
        bool(meta.get("expanded_pan", False)),
    )
    cached = st.session_state.get("loaded_scene_overlay_cache")
    if isinstance(cached, dict) and cached.get("sig") == cache_sig:
        png_uri, bounds = cached["png_uri"], cached["bounds"]
    else:
        if is_sar and vis_code in ("VV", "VH"):
            spec = SAR_METRICS[vis_code]
            png_uri, bounds = raster_to_map_overlay_png(
                ds_disp[vis_code],
                src_epsg=meta["crs"],
                is_rgb=False,
                cmap_name=str(spec["cmap"]),
                vmin=float(spec["vmin"]),
                vmax=float(spec["vmax"]),
            )
        elif is_sar:
            png_uri, bounds = raster_to_map_overlay_png(
                ds_disp,
                src_epsg=meta["crs"],
                is_rgb=True,
                bands=["VV", "VH"],
                use_percentile_stretch=bool(use_percentile),
                gamma=float(gamma),
                fixed_ranges=[(-22.0, -2.0), (-28.0, -8.0), (2.0, 14.0)],
            )
        else:
            png_uri, bounds = raster_to_map_overlay_png(
                ds_disp,
                src_epsg=meta["crs"],
                is_rgb=True,
                bands=list(meta["bands"]),
                use_percentile_stretch=bool(use_percentile),
                gamma=float(gamma),
            )
        st.session_state["loaded_scene_overlay_cache"] = {
            "sig": cache_sig,
            "png_uri": png_uri,
            "bounds": bounds,
        }

    vis_pill = (
        t(SAR_VIS_SPECS[vis_code]["pill_key"], lang=lang)
        if is_sar
        else t(OPTICAL_COMPOSITE_SPECS[vis_code]["pill_key"], lang=lang)
    )
    layer_name = f"🛰️ {meta['sensor_name']} {vis_pill} ({meta['date_str']})"
    return png_uri, bounds, layer_name


def _clamp_dates_for_sensor(sensor_key: str) -> None:
    """Zagotovi, da sta začetni in končni datum znotraj razpona [1984-01-01, danes]."""
    _ = sensor_key
    today = date.today()
    min_d: date = HISTORICAL_MIN_DATE
    cur_start: date = st.session_state.get("search_start_date", today - timedelta(days=183))
    cur_end: date = st.session_state.get("search_end_date", today)
    if cur_start < min_d:
        cur_start = min_d
    if cur_start > today:
        cur_start = today
    if cur_end < min_d:
        cur_end = today
    if cur_end > today:
        cur_end = today
    if cur_start > cur_end:
        cur_start = min_d

    st.session_state["search_start_date"] = cur_start
    st.session_state["search_end_date"] = cur_end
    st.session_state["sidebar_start_date"] = cur_start
    st.session_state["sidebar_end_date"] = cur_end
    st.session_state["tab3_start_date"] = cur_start
    st.session_state["tab3_end_date"] = cur_end


def _sync_sensor_state(source_key: str | None = None, explicit_sensor: str | None = None) -> None:
    """Sinhronizira izbrani senzor med stransko vrstico, glavno vrstico AOI in Zavihkom 3."""
    prev_sensor = str(st.session_state.get("selected_sensor", SENSOR_S2))
    raw_val = (
        explicit_sensor
        if explicit_sensor is not None
        else st.session_state.get(source_key or "selected_sensor", prev_sensor)
    )
    # Če uporabnik ponovno tapne že izbran gumb v segmented_control, ohranimo prejšnji senzor
    new_sensor = raw_val if raw_val in SENSOR_KEYS else prev_sensor

    st.session_state["selected_sensor"] = new_sensor
    st.session_state["active_sensor"] = new_sensor
    st.session_state["sidebar_sensor_select"] = new_sensor
    st.session_state["main_bar_sensor_select"] = new_sensor
    st.session_state["tab3_sensor_select"] = new_sensor

    if new_sensor != prev_sensor:
        st.session_state["stac_items"] = []
        st.session_state["stac_summary_df"] = pd.DataFrame()
        st.session_state["stac_raw_summary_df"] = pd.DataFrame()
        st.session_state["stac_raw_items_by_date"] = {}
        st.session_state["search_performed"] = False
        st.session_state["search_sensor"] = new_sensor
        st.session_state["last_rec_res_key"] = None
        default_res = SENSOR_META[new_sensor]["default_res"]
        st.session_state["selected_resolution"] = default_res
        st.session_state[f"sidebar_res_select_{new_sensor}"] = default_res
        _clamp_dates_for_sensor(new_sensor)
        _clear_loaded_products()


def _sync_start_date_state(source_key: str) -> None:
    new_d = st.session_state.get(source_key)
    if isinstance(new_d, date):
        st.session_state["search_start_date"] = new_d
        st.session_state["sidebar_start_date"] = new_d
        st.session_state["tab3_start_date"] = new_d


def _sync_end_date_state(source_key: str) -> None:
    new_d = st.session_state.get(source_key)
    if isinstance(new_d, date):
        st.session_state["search_end_date"] = new_d
        st.session_state["sidebar_end_date"] = new_d
        st.session_state["tab3_end_date"] = new_d


def _sync_max_cloud_state(source_key: str) -> None:
    val = st.session_state.get(source_key)
    if val is not None:
        c_int = int(val)
        st.session_state["search_max_cloud"] = c_int
        st.session_state["sidebar_max_cloud"] = c_int
        st.session_state["tab3_max_cloud"] = c_int


def _sync_partial_filter_state(source_key: str) -> None:
    val = bool(st.session_state.get(source_key, True))
    st.session_state["exclude_partial_swaths"] = val
    st.session_state["sidebar_exclude_partial"] = val
    st.session_state["tab3_exclude_partial"] = val


def _sync_l7_slc_state(source_key: str) -> None:
    val = bool(st.session_state.get(source_key, True))
    st.session_state["exclude_l7_slc_off"] = val
    st.session_state["sidebar_l7_slc_exclude"] = val
    st.session_state["tab3_l7_slc_exclude"] = val


def _sync_interval_state(source_key: str | None = None, explicit_interval: str | None = None) -> None:
    """Sinhronizira časovni korak med stransko vrstico in Zavihkom 3 ter takoj prefiltrira obstoječe rezultate."""
    prev_int = str(st.session_state.get("temporal_interval", INTERVAL_ALL))
    raw_val = (
        explicit_interval
        if explicit_interval is not None
        else st.session_state.get(source_key or "temporal_interval", prev_int)
    )
    new_int = raw_val if raw_val in INTERVAL_KEYS else prev_int

    st.session_state["temporal_interval"] = new_int
    st.session_state["sidebar_interval_select"] = new_int
    st.session_state["tab3_interval_select"] = new_int

    cur_season = str(st.session_state.get("yearly_target_season", SEASON_SUMMER))
    raw_df = st.session_state.get("stac_raw_summary_df")
    raw_by_date = st.session_state.get("stac_raw_items_by_date")
    if (
        st.session_state.get("search_performed", False)
        and isinstance(raw_df, pd.DataFrame)
        and not raw_df.empty
        and isinstance(raw_by_date, dict)
        and raw_by_date
    ):
        grouped_f, df_f = filter_by_temporal_interval(
            raw_df, raw_by_date, interval_mode=new_int, target_season=cur_season
        )
        df_f.attrs["raw_summary_df"] = raw_df
        df_f.attrs["raw_items_by_date"] = raw_by_date
        st.session_state["stac_items"] = grouped_f
        st.session_state["stac_summary_df"] = df_f
        st.session_state["pending_reset_scene_widgets"] = True


def _sync_yearly_season_state(source_key: str | None = None, explicit_season: str | None = None) -> None:
    """Sinhronizira ciljno obdobje leta (Poletje/Pomlad/Jesen/Zima/Celotno leto) in takoj prefiltrira letne rezultate."""
    prev_season = str(st.session_state.get("yearly_target_season", SEASON_SUMMER))
    raw_val = (
        explicit_season
        if explicit_season is not None
        else st.session_state.get(source_key or "yearly_target_season", prev_season)
    )
    new_season = raw_val if raw_val in YEARLY_SEASON_KEYS else prev_season

    st.session_state["yearly_target_season"] = new_season
    st.session_state["sidebar_yearly_season"] = new_season
    st.session_state["tab3_yearly_season"] = new_season

    cur_int = str(st.session_state.get("temporal_interval", INTERVAL_ALL))
    raw_df = st.session_state.get("stac_raw_summary_df")
    raw_by_date = st.session_state.get("stac_raw_items_by_date")
    if (
        cur_int == INTERVAL_YEARLY
        and st.session_state.get("search_performed", False)
        and isinstance(raw_df, pd.DataFrame)
        and not raw_df.empty
        and isinstance(raw_by_date, dict)
        and raw_by_date
    ):
        grouped_f, df_f = filter_by_temporal_interval(
            raw_df, raw_by_date, interval_mode=cur_int, target_season=new_season
        )
        df_f.attrs["raw_summary_df"] = raw_df
        df_f.attrs["raw_items_by_date"] = raw_by_date
        st.session_state["stac_items"] = grouped_f
        st.session_state["stac_summary_df"] = df_f
        st.session_state["pending_reset_scene_widgets"] = True


def _apply_tab3_date_preset(preset_code: str) -> None:
    """Hitro nastavi časovno obdobje (in pri 1984/1990+ samodejno preklopi na Landsat + Letno) v Zavihku 3."""
    today = date.today()
    if preset_code in ("1984_landsat", "1990_landsat"):
        _sync_sensor_state(explicit_sensor=SENSOR_LANDSAT)
        target_start = date(1984, 1, 1) if preset_code == "1984_landsat" else date(1990, 1, 1)
        _sync_interval_state(explicit_interval=INTERVAL_YEARLY)
    else:
        days_map = {"6m": 183, "2y": 730, "5y": 1826, "10y": 3652}
        target_start = today - timedelta(days=days_map.get(preset_code, 183))
        if target_start < HISTORICAL_MIN_DATE:
            target_start = HISTORICAL_MIN_DATE

    st.session_state["search_start_date"] = target_start
    st.session_state["search_end_date"] = today
    st.session_state["sidebar_start_date"] = target_start
    st.session_state["sidebar_end_date"] = today
    st.session_state["tab3_start_date"] = target_start
    st.session_state["tab3_end_date"] = today


def _apply_snow_glacier_preset(preset_key: str) -> None:
    """1-klik preklop na ledeniški / snežni primer: nastavi AOI, Avto-UTM CRS in indeks NDSI."""
    spec = SNOW_GLACIER_PRESETS.get(preset_key)
    if spec is None:
        return
    st.session_state["map_lat"] = float(spec["lat"])
    st.session_state["map_lon"] = float(spec["lon"])
    st.session_state["map_radius_m"] = int(spec["radius_m"])
    st.session_state["ws1_map_radius_slider"] = int(spec["radius_m"])
    st.session_state["active_location_label"] = t(spec["label_key"])
    st.session_state["pending_aoi_mode"] = "map"
    st.session_state["drawn_polygon_4326"] = None
    st.session_state["sidebar_crs_mode"] = "auto"
    st.session_state["tab2_index_select"] = "NDSI"
    st.session_state["tab3_ts_metric_sentinel-2-l2a"] = "NDSI"
    st.session_state["tab3_ts_metric_landsat-c2-l2"] = "NDSI"
    st.session_state["tab3_ch_metric_sentinel-2-l2a"] = "NDSI"
    st.session_state["tab3_ch_metric_landsat-c2-l2"] = "NDSI"
    st.session_state["tab3_tl_layer_sentinel-2-l2a"] = "NDSI"
    st.session_state["tab3_tl_layer_landsat-c2-l2"] = "NDSI"
    st.session_state["map_jump_version"] = int(st.session_state.get("map_jump_version", 0)) + 1
    st.session_state["search_performed"] = False
    st.session_state["stac_items"] = []
    st.session_state["stac_summary_df"] = pd.DataFrame()
    _clear_loaded_products()


def _format_interval_summary_caption(summary_df: pd.DataFrame, lang: str) -> str:
    """Sestavi jasen povzetek števila najdenih vs. izbranih reprezentativnih datumov in izločenih delnih preletov."""
    sel_n = int(len(summary_df))
    total_n = int(summary_df.attrs.get("total_dates_found", sel_n))
    partial_n = int(summary_df.attrs.get("filtered_partial_dates", 0))
    int_mode = str(summary_df.attrs.get("interval_mode", st.session_state.get("temporal_interval", INTERVAL_ALL)))
    int_label = t(INTERVAL_I18N.get(int_mode, "interval_all"), lang=lang)
    if int_mode == INTERVAL_YEARLY:
        season_k = str(summary_df.attrs.get("target_season", st.session_state.get("yearly_target_season", SEASON_SUMMER)))
        season_lbl = t(YEARLY_SEASON_I18N.get(season_k, "season_summer"), lang=lang)
        int_label = f"{int_label} · {season_lbl}"
    partial_note = t("partial_filtered_note", lang=lang, n=partial_n) if partial_n > 0 else ""
    return t(
        "interval_summary_caption",
        lang=lang,
        sel=sel_n,
        interval=int_label,
        total=total_n,
        partial_note=partial_note,
    )


def _execute_stac_search(
    *,
    sensor: str,
    bbox_4326: list[float],
    bbox_3794: list[float] | None,
    aoi_polygon_4326: dict[str, Any] | None,
    area_km2: float,
    start_date: date,
    end_date: date,
    max_cloud_cover: float,
    exclude_l7_slc_off: bool,
    exclude_partial_swaths: bool = True,
    interval_mode: str = INTERVAL_ALL,
    target_season: str | None = None,
    limit: int | None = None,
) -> None:
    """Izvede celotno zgodovinsko STAC iskanje s filtrom pokritosti AOI in časovnim vzorčenjem."""
    min_cov = DEFAULT_MIN_AOI_COVERAGE_PCT if exclude_partial_swaths else 0.0
    eff_season = str(
        target_season
        if target_season is not None
        else st.session_state.get("yearly_target_season", SEASON_SUMMER)
    )
    items_res, summary_res = search_scenes(
        sensor=sensor,
        bbox_4326=bbox_4326,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=float(max_cloud_cover),
        limit=limit,
        exclude_l7_slc_off=exclude_l7_slc_off,
        aoi_polygon_4326=aoi_polygon_4326,
        min_coverage_pct=min_cov,
        interval_mode=interval_mode,
        target_season=eff_season,
    )
    st.session_state["stac_items"] = items_res
    st.session_state["stac_summary_df"] = summary_res
    st.session_state["stac_raw_summary_df"] = summary_res.attrs.get("raw_summary_df", summary_res)
    st.session_state["stac_raw_items_by_date"] = summary_res.attrs.get(
        "raw_items_by_date", summary_res.attrs.get("items_by_date", {})
    )
    st.session_state["search_performed"] = True
    st.session_state["search_sensor"] = sensor
    st.session_state["search_bbox_4326"] = list(bbox_4326)
    st.session_state["search_bbox_3794"] = list(bbox_3794) if bbox_3794 else None
    st.session_state["search_polygon_4326"] = aoi_polygon_4326
    st.session_state["search_area_km2"] = area_km2
    st.session_state["pending_reset_scene_widgets"] = True
    _clear_loaded_products()


def _touch_selector(
    label: str,
    options: list[Any],
    *,
    default: Any = None,
    format_func: Callable[[Any], str] | None = None,
    key: str | None = None,
    help: str | None = None,
    on_change: Callable[..., Any] | None = None,
    args: tuple[Any, ...] | None = None,
    label_visibility: str = "visible",
    use_pills: bool = False,
) -> Any:
    """Dotiku prijazen izbirnik (segmented_control / pills z rezervnim horizontalnim radio).

    Na tablicah nikoli ne odpre virtualne tipkovnice in vedno vrne veljavno vrednost iz ``options``.
    """
    shadow_key = f"__touch_last_{key}" if key is not None else None
    last_valid = st.session_state.get(shadow_key) if shadow_key is not None else None
    fallback_val = (
        last_valid
        if (last_valid in options)
        else (default if (default in options) else options[0])
    )
    has_preset_state = key is not None and key in st.session_state
    if has_preset_state and st.session_state[key] not in options:
        st.session_state[key] = fallback_val

    widget_fn = None
    if use_pills and hasattr(st, "pills"):
        widget_fn = st.pills
    elif hasattr(st, "segmented_control"):
        widget_fn = st.segmented_control

    if widget_fn is not None:
        kw: dict[str, Any] = {
            "label": label,
            "options": options,
            "selection_mode": "single",
            "label_visibility": label_visibility,
        }
        if format_func is not None:
            kw["format_func"] = format_func
        if key is not None:
            kw["key"] = key
        if not has_preset_state:
            kw["default"] = fallback_val
        if help is not None:
            kw["help"] = help
        if on_change is not None:
            kw["on_change"] = on_change
            if args is not None:
                kw["args"] = args
        val = widget_fn(**kw)
        resolved = val if (val in options) else fallback_val
        if shadow_key is not None:
            st.session_state[shadow_key] = resolved
        return resolved

    r_kw: dict[str, Any] = {
        "label": label,
        "options": options,
        "horizontal": True,
        "label_visibility": label_visibility,
    }
    if format_func is not None:
        r_kw["format_func"] = format_func
    if key is not None:
        r_kw["key"] = key
    if not has_preset_state:
        r_kw["index"] = options.index(fallback_val)
    if help is not None:
        r_kw["help"] = help
    if on_change is not None:
        r_kw["on_change"] = on_change
        if args is not None:
            r_kw["args"] = args
    val = st.radio(**r_kw)
    resolved = val if (val in options) else fallback_val
    if shadow_key is not None:
        st.session_state[shadow_key] = resolved
    return resolved


def _scene_touch_picker(
    label: str,
    *,
    num_scenes: int,
    default_idx: int,
    scene_labels: Sequence[str],
    pill_labels: Sequence[str],
    key: str,
    lang: str,
    page_size: int = 12,
) -> int:
    """100 % dotiku prijazen izbirnik posnetka (◀ Prejšnji / Naslednji ▶ + st.pills), brez virtualne tipkovnice."""
    if num_scenes <= 0:
        return 0

    clamped_default = max(0, min(int(default_idx), num_scenes - 1))
    cur_raw = st.session_state.get(key, clamped_default)
    try:
        cur_idx = int(cur_raw)
    except (TypeError, ValueError):
        cur_idx = clamped_default
    if cur_idx < 0 or cur_idx >= num_scenes:
        cur_idx = clamped_default
    st.session_state[key] = cur_idx

    page_key = f"__scene_page_{key}"
    last_idx_key = f"__scene_last_idx_{key}"
    pill_widget_key = f"__scene_pill_{key}"
    num_pages = max(1, (num_scenes + page_size - 1) // page_size)
    idx_page = cur_idx // page_size

    if st.session_state.get(last_idx_key) != cur_idx or page_key not in st.session_state:
        st.session_state[page_key] = idx_page
        st.session_state[last_idx_key] = cur_idx
        st.session_state[pill_widget_key] = cur_idx
    else:
        cur_pg = int(st.session_state.get(page_key, idx_page))
        if cur_pg < 0 or cur_pg >= num_pages:
            st.session_state[page_key] = idx_page

    def _step_scene(delta: int) -> None:
        nxt = max(0, min(num_scenes - 1, int(st.session_state.get(key, 0)) + delta))
        st.session_state[key] = nxt
        st.session_state[page_key] = nxt // page_size
        st.session_state[last_idx_key] = nxt
        st.session_state[pill_widget_key] = nxt

    st.markdown(f"**{label}**")
    c_prev, c_badge, c_next = st.columns([0.95, 2.2, 0.95], vertical_alignment="center")
    with c_prev:
        st.button(
            t("btn_prev_scene", lang=lang),
            width="stretch",
            disabled=(cur_idx <= 0),
            key=f"__btn_prev_{key}",
            on_click=_step_scene,
            args=(-1,),
        )
    with c_badge:
        st.markdown(
            f'<div style="background:#F0FDFA;border:1px solid #99F6E4;border-radius:8px;'
            f'padding:6px 10px;text-align:center;color:#0F172A;font-size:0.86rem;">'
            f'<span style="color:#0F766E;font-weight:700;">{cur_idx + 1}/{num_scenes}</span> · '
            f'{scene_labels[cur_idx]}</div>',
            unsafe_allow_html=True,
        )
    with c_next:
        st.button(
            t("btn_next_scene", lang=lang),
            width="stretch",
            disabled=(cur_idx >= num_scenes - 1),
            key=f"__btn_next_{key}",
            on_click=_step_scene,
            args=(1,),
        )

    active_page = int(st.session_state.get(page_key, idx_page))
    if num_pages > 1:
        page_opts = list(range(num_pages))

        def _fmt_pg(p: int) -> str:
            s_i = p * page_size + 1
            e_i = min(num_scenes, (p + 1) * page_size)
            return f"{s_i}–{e_i}"

        active_page = int(
            _touch_selector(
                t("scene_page_label", lang=lang),
                options=page_opts,
                default=active_page,
                format_func=_fmt_pg,
                key=f"__pg_sel_{key}",
                label_visibility="collapsed",
                use_pills=True,
            )
        )
        st.session_state[page_key] = active_page

    start_i = active_page * page_size
    end_i = min(num_scenes, start_i + page_size)
    page_indices = list(range(start_i, end_i))

    if hasattr(st, "pills"):
        def _on_pill_pick() -> None:
            picked = st.session_state.get(pill_widget_key)
            if picked in page_indices:
                st.session_state[key] = int(picked)
                st.session_state[last_idx_key] = int(picked)

        if st.session_state.get(pill_widget_key) not in page_indices:
            st.session_state[pill_widget_key] = cur_idx if cur_idx in page_indices else None

        st.pills(
            label,
            options=page_indices,
            selection_mode="single",
            format_func=lambda idx: pill_labels[idx],
            key=pill_widget_key,
            label_visibility="collapsed",
            on_change=_on_pill_pick,
        )
    else:
        picked = st.radio(
            label,
            options=page_indices,
            index=page_indices.index(cur_idx) if cur_idx in page_indices else 0,
            format_func=lambda idx: pill_labels[idx],
            horizontal=True,
            key=f"__radio_{key}",
            label_visibility="collapsed",
        )
        if picked in page_indices:
            st.session_state[key] = int(picked)
            st.session_state[last_idx_key] = int(picked)

    return int(st.session_state[key])


def _multi_date_touch_picker(
    label: str,
    *,
    num_scenes: int,
    pill_labels: Sequence[str],
    key: str,
    lang: str,
    page_size: int = 24,
) -> list[int]:
    """100 % dotiku prijazen več-izbirni izbirnik 2 ali 3 datumov (st.pills multi), brez virtualne tipkovnice."""
    if num_scenes <= 0:
        return []

    default_sel = [num_scenes - 1, 0] if num_scenes >= 2 else [0]
    raw_sel = st.session_state.get(key)
    if not isinstance(raw_sel, list):
        cur_sel = list(default_sel)
    else:
        cur_sel = [int(i) for i in raw_sel if isinstance(i, (int, np.integer)) and 0 <= int(i) < num_scenes]
        if not cur_sel and num_scenes >= 2:
            cur_sel = list(default_sel)
    st.session_state[key] = cur_sel

    num_pages = max(1, (num_scenes + page_size - 1) // page_size)
    active_page = 0
    if num_pages > 1:
        page_opts = list(range(num_pages))
        active_page = int(
            _touch_selector(
                t("scene_page_label", lang=lang),
                options=page_opts,
                default=0,
                format_func=lambda p: f"{p * page_size + 1}–{min(num_scenes, (p + 1) * page_size)}",
                key=f"__multi_pg_{key}",
                label_visibility="collapsed",
                use_pills=True,
            )
        )

    start_i = active_page * page_size
    end_i = min(num_scenes, start_i + page_size)
    page_indices = list(range(start_i, end_i))
    page_set = set(page_indices)
    in_page_sel = [i for i in cur_sel if i in page_set]
    off_page_sel = [i for i in cur_sel if i not in page_set]

    widget_key = f"__multi_pills_{key}_p{active_page}"
    if hasattr(st, "pills"):
        if widget_key not in st.session_state:
            st.session_state[widget_key] = in_page_sel
        picked_page = st.pills(
            label,
            options=page_indices,
            selection_mode="multi",
            format_func=lambda idx: pill_labels[idx],
            key=widget_key,
        )
        picked_page_list = [int(i) for i in (picked_page or []) if i in page_set]
    else:
        picked_page_list = [
            int(i)
            for i in st.multiselect(
                label,
                options=page_indices,
                default=in_page_sel,
                format_func=lambda idx: pill_labels[idx],
                key=widget_key,
            )
        ]

    merged = list(dict.fromkeys([*off_page_sel, *picked_page_list]))
    st.session_state[key] = merged
    return merged


def _render_html_iframe(html_str: str, height: int = 1) -> None:
    """Izriše HTML v iframe z uporabo novejšega ``st.iframe`` (brez opozoril o zastarelosti) ali ``components.html``."""
    if hasattr(st, "iframe"):
        st.iframe(html_str, height=max(1, int(height)))
    else:
        components.html(html_str, height=int(height), scrolling=False)


def _inject_tablet_no_keyboard_js() -> None:
    """Nastavi readonly + inputmode='none' na vse BaseWeb selectbox vnose (prepreči tipkovnico na tablicah)."""
    _render_html_iframe(
        """
<script>
(function() {
  try {
    const doc = window.parent.document;
    function lockSelectInputs() {
      const inputs = doc.querySelectorAll('[data-baseweb="select"] input');
      inputs.forEach((el) => {
        if (el.getAttribute('inputmode') !== 'none') {
          el.setAttribute('readonly', 'readonly');
          el.setAttribute('inputmode', 'none');
        }
      });
    }
    lockSelectInputs();
    if (!window.parent.__dzNoKbdObserver) {
      const observer = new MutationObserver(lockSelectInputs);
      observer.observe(doc.body, { childList: true, subtree: true });
      window.parent.__dzNoKbdObserver = observer;
    }
  } catch (e) {}
})();
</script>
""",
        height=1,
    )


def _normalize_place_text(text: str) -> str:
    """Normalizira ime kraja (male črke, brez šumnikov/diakritičnih znakov) za zanesljivo ujemanje."""
    decomposed = unicodedata.normalize("NFKD", str(text or "").strip().lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped).strip()


_COORD_PAIR_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*[,;\s]+\s*([+-]?\d+(?:\.\d+)?)\s*$"
)


def _parse_direct_coords(query: str) -> tuple[float, float, str] | None:
    """Zazna prilepljene koordinate WGS84 (lat, lon) ali D96/TM (e, n > 1000)."""
    match = _COORD_PAIR_RE.match(str(query or "").strip())
    if not match:
        return None
    try:
        v1 = float(match.group(1))
        v2 = float(match.group(2))
    except ValueError:
        return None

    # D96/TM EPSG:3794 (vrednosti v metrih > 1000)
    if abs(v1) > 1000.0 and abs(v2) > 1000.0:
        try:
            lon, lat = transform_coords(v1, v2, EPSG_D96TM, EPSG_WGS84)
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return float(lat), float(lon), f"D96/TM (e = {v1:,.1f} m, n = {v2:,.1f} m)"
        except Exception:
            return None
        return None

    # WGS84 (lat, lon) v stopinjah
    if -90.0 <= v1 <= 90.0 and -180.0 <= v2 <= 180.0:
        return float(v1), float(v2), f"WGS84 ({v1:.5f}°, {v2:.5f}°)"
    return None


def _geocode_photon(query: str) -> tuple[float, float, str] | None:
    """Primarni spletni geokoder: Photon (Komoot), usmerjen proti Sloveniji."""
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    params = {
        "q": query,
        "limit": 5,
        "lat": SLOVENIA_CENTER_LAT,
        "lon": SLOVENIA_CENTER_LON,
    }
    resp = requests.get(PHOTON_SEARCH_URL, params=params, headers=headers, timeout=6)
    resp.raise_for_status()
    payload = resp.json()
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list) or not features:
        return None

    # Prednostno izberi zadetek v Sloveniji (countrycode == SI)
    chosen = features[0]
    for feat in features:
        props = feat.get("properties") or {}
        cc = str(props.get("countrycode", "")).upper()
        country = str(props.get("country", "")).lower()
        if cc == "SI" or "sloveni" in country:
            chosen = feat
            break

    coords = (chosen.get("geometry") or {}).get("coordinates") or []
    if len(coords) < 2:
        return None
    lon, lat = float(coords[0]), float(coords[1])
    props = chosen.get("properties") or {}
    name_parts = [
        str(props.get(k, "")).strip()
        for k in ("name", "city", "county", "country")
        if str(props.get(k, "")).strip()
    ]
    dedup_parts: list[str] = []
    for p in name_parts:
        if p not in dedup_parts:
            dedup_parts.append(p)
    display_name = ", ".join(dedup_parts) if dedup_parts else query
    return lat, lon, display_name


def geocode_place_nominatim(
    query: str,
    lang: str = "sl",
) -> tuple[float, float, str] | None:
    """Rezervni geokoder 1: OpenStreetMap Nominatim API (najprej Slovenija, nato svet)."""
    clean_q = str(query or "").strip()
    if not clean_q:
        return None

    headers = {
        "User-Agent": NOMINATIM_USER_AGENT,
        "Accept-Language": f"{lang},sl;q=0.9,en;q=0.8",
    }

    for country_filter in ("si", None):
        params: dict[str, Any] = {
            "q": clean_q,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 0,
        }
        if country_filter:
            params["countrycodes"] = country_filter

        resp = requests.get(
            NOMINATIM_SEARCH_URL,
            params=params,
            headers=headers,
            timeout=6,
        )
        resp.raise_for_status()
        results = resp.json()
        if isinstance(results, list) and len(results) > 0:
            hit = results[0]
            lat = float(hit["lat"])
            lon = float(hit["lon"])
            display_name = str(hit.get("display_name", clean_q))
            return lat, lon, display_name

    return None


def _geocode_local_slovenia(query: str) -> tuple[float, float, str] | None:
    """Rezervni geokoder 2: vgrajen seznam 45+ slovenskih krajev in naravnih znamenitosti."""
    norm_q = _normalize_place_text(query)
    if not norm_q:
        return None

    # 1. Natančno ujemanje brez šumnikov
    for place_name, (lat, lon) in SLOVENIA_PLACES_FALLBACK.items():
        if _normalize_place_text(place_name) == norm_q:
            return lat, lon, f"{place_name}, Slovenija"

    # 2. Delno ujemanje (začetek imena ali podniz)
    for place_name, (lat, lon) in SLOVENIA_PLACES_FALLBACK.items():
        norm_p = _normalize_place_text(place_name)
        if norm_p.startswith(norm_q) or norm_q in norm_p or norm_p in norm_q:
            return lat, lon, f"{place_name}, Slovenija"

    return None


@st.cache_data(ttl=3600, max_entries=32, show_spinner=False)
def geocode_place(
    query: str,
    lang: str = "sl",
) -> tuple[float, float, str] | None:
    """Zanesljiv 3-stopenjski geokoder s pametnim zaznavanjem koordinat.

    1. Pametno zaznavanje prilepljenih koordinat WGS84 (`lat, lon`) ali D96/TM (`e, n > 1000`).
    2. Primarni spletni geokoder: Photon API (Komoot, usmerjen v Slovenijo).
    3. Rezerva 1: OpenStreetMap Nominatim API.
    4. Rezerva 2: Vgrajen lokalni slovar 45+ slovenskih krajev (brez šumnikov).
    """
    clean_q = str(query or "").strip()
    if not clean_q:
        return None

    direct = _parse_direct_coords(clean_q)
    if direct is not None:
        return direct

    try:
        res_photon = _geocode_photon(clean_q)
        if res_photon is not None:
            return res_photon
    except Exception:
        pass

    try:
        res_nom = geocode_place_nominatim(clean_q, lang=lang)
        if res_nom is not None:
            return res_nom
    except Exception:
        pass

    return _geocode_local_slovenia(clean_q)


def _jump_to_location(lat: float, lon: float, label: str, zoom: int = 12) -> None:
    """Premakne središče glavnega zemljevida na ``(lat, lon)``, posodobi AOI in prikaže potrditev."""
    e_val, n_val = transform_coords(lon, lat, EPSG_WGS84, EPSG_D96TM)
    st.session_state["map_lat"] = round(float(lat), 6)
    st.session_state["map_lon"] = round(float(lon), 6)
    st.session_state["map_zoom"] = int(zoom)
    st.session_state["live_map_zoom"] = int(zoom)
    st.session_state["map_jump_version"] = int(st.session_state.get("map_jump_version", 0)) + 1
    st.session_state["last_processed_click"] = None
    st.session_state["pending_aoi_mode"] = "map"
    st.session_state["active_location_label"] = str(label).split(",")[0].strip() or str(label)
    st.session_state["geocoded_place_msg"] = t(
        "loc_jumped_banner",
        name=label,
        e=e_val,
        n=n_val,
        lat=float(lat),
        lon=float(lon),
    )
    st.rerun()


def _format_tile_label(tile: str, sensor: str = SENSOR_S2) -> str:
    clean = str(tile or "").strip()
    if not clean:
        return "—"
    if sensor == SENSOR_S2:
        parts = [p.strip() for p in clean.split("+") if p.strip()]
        formatted = [p if p.startswith("T") else f"T{p}" for p in parts]
        return " + ".join(formatted) if formatted else "—"
    return clean


def _d96tm_corners_to_wgs84_polygon(
    bbox_metric: list[float],
    metric_crs: int | str = EPSG_D96TM,
) -> list[list[float]]:
    min_e, min_n, max_e, max_n = bbox_metric
    corners_metric = [
        (min_e, min_n),
        (max_e, min_n),
        (max_e, max_n),
        (min_e, max_n),
        (min_e, min_n),
    ]
    poly_latlon: list[list[float]] = []
    for e_val, n_val in corners_metric:
        lon, lat = transform_coords(e_val, n_val, metric_crs, EPSG_WGS84)
        poly_latlon.append([lat, lon])
    return poly_latlon


def _polygon_bounds_to_bboxes(
    poly_geom_4326: dict[str, Any],
) -> tuple[list[float], list[float], float, float]:
    """Izračuna WGS84 in lokalni metrični okvir (D96/TM v SLO / lokalni UTM po svetu) ter središče."""
    geom = shape(poly_geom_4326)
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    bbox_4326 = [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]
    centroid = geom.centroid
    metric_crs, _ = get_best_metric_crs(float(centroid.y), float(centroid.x), mode="auto")
    e1, n1 = transform_coords(min_lon, min_lat, EPSG_WGS84, metric_crs)
    e2, n2 = transform_coords(max_lon, max_lat, EPSG_WGS84, metric_crs)
    bbox_metric = [min(e1, e2), min(n1, n2), max(e1, e2), max(n1, n2)]
    return bbox_4326, bbox_metric, float(centroid.y), float(centroid.x)


def _extract_drawn_polygon(map_out: Any) -> dict[str, Any] | None:
    """Izvleče zadnji narisani Polygon/MultiPolygon iz povratnih podatkov st_folium."""
    if not isinstance(map_out, dict):
        return None

    candidates: list[dict[str, Any]] = []
    last_active = map_out.get("last_active_drawing")
    if isinstance(last_active, dict):
        candidates.append(last_active)

    all_drawings = map_out.get("all_drawings")
    if isinstance(all_drawings, list):
        for item in reversed(all_drawings):
            if isinstance(item, dict):
                candidates.append(item)

    for feat in candidates:
        geom_obj = feat.get("geometry") if feat.get("type") == "Feature" else feat
        if isinstance(geom_obj, dict) and geom_obj.get("type") in ("Polygon", "MultiPolygon"):
            try:
                shp = shape(geom_obj)
                if not shp.is_empty and shp.area > 0:
                    return mapping(shp)
            except Exception:
                continue
    return None


def _create_base_folium_map(
    center_lat: float,
    center_lon: float,
    zoom_start: int = 10,
    basemap_key: str = "satellite",
) -> folium.Map:
    """Ustvari Folium zemljevid z natanko eno izbrano podlago (brez ponastavljanja ob kliku)."""
    spec = BASEMAP_SPECS.get(basemap_key, BASEMAP_SPECS["satellite"])
    max_z = int(spec["max_zoom"])
    clamped_zoom = max(1, min(int(zoom_start), max_z))

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=clamped_zoom,
        tiles=None,
        control_scale=True,
        max_zoom=max_z,
    )
    tile_kwargs: dict[str, Any] = {
        "tiles": spec["tiles"],
        "name": t(spec["label_key"]),
        "overlay": False,
        "control": False,
        "max_zoom": max_z,
    }
    if spec["attr"]:
        tile_kwargs["attr"] = spec["attr"]
    folium.TileLayer(**tile_kwargs).add_to(fmap)
    return fmap


def _build_aoi_folium_map(
    center_lat: float,
    center_lon: float,
    bbox_3794: list[float] | None = None,
    bbox_4326: list[float] | None = None,
    polygon_geojson_4326: dict[str, Any] | None = None,
    zoom_start: int = 10,
    interactive_hint: bool = True,
    basemap_key: str = "satellite",
) -> folium.Map:
    """Ustvari Folium zemljevid s kartografsko oznako AOI v temno petrol-zeleni barvi."""
    fmap = _create_base_folium_map(
        center_lat=center_lat,
        center_lon=center_lon,
        zoom_start=zoom_start,
        basemap_key=basemap_key,
    )

    if polygon_geojson_4326 is not None:
        folium.GeoJson(
            data={"type": "Feature", "geometry": polygon_geojson_4326, "properties": {}},
            style_function=lambda _feat: {
                "color": "#0F766E",
                "weight": 2.5,
                "fillColor": "#0F766E",
                "fillOpacity": 0.22,
            },
            tooltip=t("map_tooltip_aoi_gj"),
        ).add_to(fmap)
    elif bbox_3794 is not None:
        eff_map_crs, _ = get_best_metric_crs(float(center_lat), float(center_lon), mode="auto")
        poly_coords = _d96tm_corners_to_wgs84_polygon(bbox_3794, metric_crs=eff_map_crs)
        folium.Polygon(
            locations=poly_coords,
            color="#0F766E",
            weight=2.5,
            fill=True,
            fill_color="#0F766E",
            fill_opacity=0.18,
            tooltip=t("map_tooltip_aoi_d96"),
        ).add_to(fmap)
    elif bbox_4326 is not None:
        min_lon, min_lat, max_lon, max_lat = bbox_4326
        folium.Rectangle(
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            color="#0F766E",
            weight=2.5,
            fill=True,
            fill_color="#0F766E",
            fill_opacity=0.18,
            tooltip=t("map_tooltip_aoi_gj"),
        ).add_to(fmap)

    tooltip_txt = t("map_tooltip_center_move") if interactive_hint else t("map_tooltip_center")
    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=6,
        color="#0F172A",
        weight=2,
        fill=True,
        fill_color="#0F766E",
        fill_opacity=0.95,
        tooltip=tooltip_txt,
    ).add_to(fmap)

    MeasureControl(
        position="topleft",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
        primary_area_unit="hectares",
        secondary_area_unit="sqkilometers",
    ).add_to(fmap)

    return fmap


def _build_draw_folium_map(
    center_lat: float,
    center_lon: float,
    existing_polygon_4326: dict[str, Any] | None = None,
    zoom_start: int = 11,
    basemap_key: str = "satellite",
) -> folium.Map:
    """Ustvari interaktivni Folium zemljevid z orodjem Draw (poligon in pravokotnik)."""
    fmap = _create_base_folium_map(
        center_lat=center_lat,
        center_lon=center_lon,
        zoom_start=zoom_start,
        basemap_key=basemap_key,
    )

    if existing_polygon_4326 is not None:
        folium.GeoJson(
            data={"type": "Feature", "geometry": existing_polygon_4326, "properties": {}},
            name="AOI",
            style_function=lambda _feat: {
                "color": "#0F766E",
                "weight": 2.5,
                "fillColor": "#0F766E",
                "fillOpacity": 0.22,
            },
            tooltip=t("map_tooltip_aoi_gj"),
        ).add_to(fmap)

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": {
                "allowIntersection": False,
                "shapeOptions": {
                    "color": "#0F766E",
                    "weight": 2.5,
                    "fillColor": "#0F766E",
                    "fillOpacity": 0.22,
                },
            },
            "rectangle": {
                "shapeOptions": {
                    "color": "#0F766E",
                    "weight": 2.5,
                    "fillColor": "#0F766E",
                    "fillOpacity": 0.22,
                },
            },
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)

    MeasureControl(
        position="topright",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
        primary_area_unit="hectares",
        secondary_area_unit="sqkilometers",
    ).add_to(fmap)

    return fmap


def _parse_geojson_aoi(
    uploaded_file: Any,
    fallback_radius_m: float = 2000.0,
) -> tuple[list[float], list[float], dict[str, Any] | None]:
    """Prebere naložen GeoJSON in vrne ``(bbox_4326, bbox_3794, polygon_geojson_4326)``."""
    raw_text = uploaded_file.getvalue().decode("utf-8")
    data = json.loads(raw_text)

    geoms = []
    gtype = data.get("type")
    if gtype == "FeatureCollection":
        for feat in data.get("features", []):
            geom_obj = feat.get("geometry")
            if geom_obj:
                geoms.append(shape(geom_obj))
    elif gtype == "Feature":
        geom_obj = data.get("geometry")
        if geom_obj:
            geoms.append(shape(geom_obj))
    elif gtype in {
        "Polygon",
        "MultiPolygon",
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "GeometryCollection",
    }:
        geoms.append(shape(data))
    else:
        raise ValueError(t("geojson_type_error", gtype=gtype))

    if not geoms:
        raise ValueError(t("geojson_empty_error"))

    merged = unary_union(geoms)
    min_x, min_y, max_x, max_y = merged.bounds

    if not (
        -180.0 <= min_x <= 180.0
        and -180.0 <= max_x <= 180.0
        and -90.0 <= min_y <= 90.0
        and -90.0 <= max_y <= 90.0
    ):
        raise ValueError(t("geojson_crs_error"))

    if min_x == max_x or min_y == max_y:
        b4326, b3794 = bbox_from_point_wgs84(
            0.5 * (min_y + max_y),
            0.5 * (min_x + max_x),
            fallback_radius_m,
        )
        return b4326, b3794, None

    bbox_4326 = [float(min_x), float(min_y), float(max_x), float(max_y)]
    e1, n1 = transform_coords(min_x, min_y, EPSG_WGS84, EPSG_D96TM)
    e2, n2 = transform_coords(max_x, max_y, EPSG_WGS84, EPSG_D96TM)
    bbox_3794 = [min(e1, e2), min(n1, n2), max(e1, e2), max(n1, n2)]
    poly_dict = mapping(merged) if merged.geom_type in ("Polygon", "MultiPolygon") else None
    return bbox_4326, bbox_3794, poly_dict


# ---------------------------------------------------------------------------
# Interaktivni Plotly rastrski prikazovalnik (fiksna višina, zoom/pan, 1:1, sever zgoraj)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _mpl_to_plotly_colorscale(cmap_name: str, n: int = 32) -> list[list[Any]]:
    """Pretvori Matplotlib barvno lestvico v Plotly colorscale brez vijoličnih odtenkov."""
    cmap = matplotlib.colormaps[cmap_name]
    scale: list[list[Any]] = []
    for i in range(n):
        frac = float(i) / float(n - 1)
        r, g, b, _ = cmap(frac)
        scale.append([round(frac, 4), f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"])
    return scale


def _axis_labels_for_crs(target_crs: str) -> tuple[str, str]:
    if target_crs == CRS_D96TM:
        return t("axis_e_d96"), t("axis_n_d96")
    crs_lbl = format_crs_label(target_crs, short=False)
    return t("axis_e_utm_dyn", crs_label=crs_lbl), t("axis_n_utm_dyn", crs_label=crs_lbl)


def _stretch_channel(
    arr: np.ndarray,
    use_percentile_stretch: bool = True,
    fixed_range: tuple[float, float] = (0.0, 0.35),
    gamma: float = 1.0,
) -> np.ndarray:
    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    if use_percentile_stretch:
        p_low, p_high = np.percentile(finite_vals, [2.0, 98.0])
        if p_high <= p_low:
            p_low, p_high = float(finite_vals.min()), float(finite_vals.max())
    else:
        p_low, p_high = fixed_range

    denom = max(float(p_high - p_low), 1e-6)
    stretched = np.clip((arr - np.float32(p_low)) / np.float32(denom), 0.0, 1.0)
    if gamma != 1.0 and gamma > 0:
        stretched = np.power(np.clip(stretched, 0.0, 1.0), np.float32(1.0 / gamma))
    return stretched.astype(np.float32, copy=False)


def _compose_rgb_uint8(
    ds: xr.Dataset,
    bands: list[str],
    use_percentile_stretch: bool = True,
    gamma: float = 1.0,
    fixed_ranges: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Sestavi 3-kanalno sliko ``uint8 (H, W, 3)`` za interaktivni ``px.imshow``.

    Neveljavni piksli (npr. zunaj narisanega poligona) dobijo nevtralno svetlo
    kartografsko ozadje ``#F1F5F9`` (`241, 245, 249`).
    """
    channels: list[np.ndarray] = []
    valid_union = np.ones(ds[bands[0]].shape, dtype=bool)
    for b in bands:
        arr = ds[b].values.astype(np.float32, copy=False)
        valid_union &= np.isfinite(arr)
        channels.append(arr)

    bg_rgb = (241, 245, 249)
    uint8_channels: list[np.ndarray] = []
    for idx, arr in enumerate(channels):
        f_range = fixed_ranges[idx] if (fixed_ranges and idx < len(fixed_ranges)) else (0.0, 0.35)
        stretched = _stretch_channel(
            arr,
            use_percentile_stretch=use_percentile_stretch,
            fixed_range=f_range,
            gamma=gamma,
        )
        ch_u8 = np.where(
            valid_union,
            np.clip(np.round(stretched * 255.0), 0, 255),
            bg_rgb[idx],
        ).astype(np.uint8)
        uint8_channels.append(ch_u8)

    return np.stack(uint8_channels, axis=-1)


def render_interactive_raster_plotly(
    *,
    mode: str,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    title: str,
    target_crs: str,
    rgb_uint8: np.ndarray | None = None,
    z_data: np.ndarray | None = None,
    cmap_name: str = "RdYlGn",
    colorscale_override: list[list[Any]] | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    cbar_title: str = "",
    unit_suffix: str = "",
    show_scale: bool = True,
    height: int = PLOTLY_VIEWER_HEIGHT,
) -> go.Figure:
    """Ustvari interaktivni Plotly prikazovalnik z fiksno višino, zoom/pan in 1:1 merilom.

    Zagotavlja pravilno orientacijo **sever zgoraj (north-up)** tako, da večje
    vrednosti severne koordinate ``n [m]`` ležijo na vrhu prikaza (`autorange=True`).
    """
    x_label, y_label = _axis_labels_for_crs(target_crs)
    x_vals = np.asarray(x_coords, dtype=np.float64)
    y_vals = np.asarray(y_coords, dtype=np.float64)

    if mode == "rgb" and rgb_uint8 is not None:
        img = rgb_uint8
        # Če bi bile y koordinate naraščajoče, obrnemo vrstice, da je sever vedno zgoraj
        if y_vals.size > 1 and y_vals[0] < y_vals[-1]:
            img = img[::-1, :, :]
            y_vals = y_vals[::-1]

        fig = px.imshow(img, x=x_vals, y=y_vals)
        fig.update_traces(
            hovertemplate=(
                "<b>e</b>: %{x:,.1f} m<br>"
                "<b>n</b>: %{y:,.1f} m<br>"
                "<b>RGB</b>: (%{z})<extra></extra>"
            )
        )
    elif z_data is not None:
        cscale = (
            colorscale_override
            if colorscale_override is not None
            else _mpl_to_plotly_colorscale(cmap_name)
        )
        metric_lbl = cbar_title or "Value"
        fig = go.Figure(
            data=go.Heatmap(
                z=z_data,
                x=x_vals,
                y=y_vals,
                colorscale=cscale,
                zmin=zmin,
                zmax=zmax,
                showscale=show_scale,
                colorbar=dict(
                    title=dict(text=cbar_title, font=dict(size=12, color="#0F172A")),
                    tickfont=dict(size=11, color="#334155"),
                    thickness=16,
                    len=0.88,
                ),
                hoverongaps=False,
                hovertemplate=(
                    "<b>e</b>: %{x:,.1f} m<br>"
                    "<b>n</b>: %{y:,.1f} m<br>"
                    f"<b>{metric_lbl}</b>: %{{z:.3f}}{unit_suffix}<extra></extra>"
                ),
            )
        )
    else:
        raise ValueError("render_interactive_raster_plotly requires either rgb_uint8 or z_data.")

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color="#0F172A"),
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=12, color="#0F172A")),
            tickformat=",.0f",
            gridcolor="#CBD5E1",
            zeroline=False,
            constrain="domain",
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=12, color="#0F172A")),
            tickformat=",.0f",
            gridcolor="#CBD5E1",
            zeroline=False,
            autorange=True,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        dragmode="zoom",
        plot_bgcolor="#F1F5F9",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=65, r=25, t=52, b=50),
        height=height,
    )
    return fig


def _build_optical_rgb_plotly(
    ds: xr.Dataset,
    bands: list[str],
    composite_label: str,
    target_crs: str,
    use_percentile_stretch: bool,
    gamma: float,
    sensor_name: str = "Sentinel-2 L2A",
    tile_lbl: str = "—",
) -> go.Figure:
    rgb_u8 = _compose_rgb_uint8(ds, bands, use_percentile_stretch, gamma)
    res = float(ds.attrs.get("resolution_m", 10.0))
    date_str = str(ds.attrs.get("datetime", ""))[:10]
    title = (
        f"<b>{sensor_name} ({date_str} | {tile_lbl}) — {composite_label}</b><br>"
        f"<span style='font-size:12px;color:#475569'>"
        f"{t('fig_res_bands', crs=target_crs, res=res, bands='-'.join(bands))}</span>"
    )
    return render_interactive_raster_plotly(
        mode="rgb",
        x_coords=ds.coords["x"].values,
        y_coords=ds.coords["y"].values,
        rgb_uint8=rgb_u8,
        title=title,
        target_crs=target_crs,
    )


def _build_sar_scene_plotly(
    ds: xr.Dataset,
    vis_code: str,
    vis_label: str,
    target_crs: str,
    use_percentile_stretch: bool,
    gamma: float,
    tile_lbl: str = "—",
) -> go.Figure:
    """Interaktivni Plotly prikazovalnik za Sentinel-1 SAR (VV dB, VH dB ali SAR False Color)."""
    res = float(ds.attrs.get("resolution_m", 10.0))
    date_str = str(ds.attrs.get("datetime", ""))[:10]

    if vis_code in ("VV", "VH"):
        arr = ds[vis_code].values.astype(np.float32, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0 and use_percentile_stretch:
            vmin, vmax = np.percentile(finite, [2.0, 98.0])
            if vmax <= vmin:
                vmin, vmax = float(finite.min()), float(finite.max())
        else:
            vmin, vmax = (-22.0, -2.0) if vis_code == "VV" else (-28.0, -8.0)

        bands_txt = f"{vis_code} [dB]"
        title = (
            f"<b>Sentinel-1 SAR RTC ({date_str} | {tile_lbl}) — {vis_label}</b><br>"
            f"<span style='font-size:12px;color:#475569'>"
            f"{t('fig_res_bands', crs=target_crs, res=res, bands=bands_txt)}</span>"
        )
        return render_interactive_raster_plotly(
            mode="heatmap",
            x_coords=ds.coords["x"].values,
            y_coords=ds.coords["y"].values,
            z_data=arr,
            cmap_name="Greys_r",
            zmin=float(vmin),
            zmax=float(vmax),
            cbar_title=f"{vis_code} [dB]",
            unit_suffix=" dB",
            title=title,
            target_crs=target_crs,
        )

    # SAR False Color: R = VV [dB], G = VH [dB], B = VV - VH [dB]
    vv = ds["VV"].values.astype(np.float32, copy=False)
    vh = ds["VH"].values.astype(np.float32, copy=False)
    valid = np.isfinite(vv) & np.isfinite(vh)
    ratio = np.where(valid, vv - vh, np.nan).astype(np.float32)

    fc_ds = xr.Dataset(
        {
            "R": (("y", "x"), vv),
            "G": (("y", "x"), vh),
            "B": (("y", "x"), ratio),
        },
        coords=ds.coords,
    )
    rgb_u8 = _compose_rgb_uint8(
        fc_ds,
        ["R", "G", "B"],
        use_percentile_stretch=use_percentile_stretch,
        gamma=gamma,
        fixed_ranges=[(-20.0, 0.0), (-26.0, -6.0), (2.0, 14.0)],
    )
    bands_txt = "R:VV · G:VH · B:VV/VH [dB]"
    title = (
        f"<b>Sentinel-1 SAR RTC ({date_str} | {tile_lbl}) — {vis_label}</b><br>"
        f"<span style='font-size:12px;color:#475569'>"
        f"{t('fig_res_bands', crs=target_crs, res=res, bands=bands_txt)}</span>"
    )
    return render_interactive_raster_plotly(
        mode="rgb",
        x_coords=ds.coords["x"].values,
        y_coords=ds.coords["y"].values,
        rgb_uint8=rgb_u8,
        title=title,
        target_crs=target_crs,
    )


def _build_sar_water_plotly(
    mask_da: xr.DataArray,
    threshold_db: float,
    target_crs: str,
    date_str: str,
    tile_lbl: str,
    resolution_m: float,
) -> go.Figure:
    """Interaktivni Plotly prikazovalnik za dvoklasno SAR masko voda/poplav (0=Kopno, 1=Voda)."""
    raw = mask_da.values
    z_arr = np.where(raw == 255, np.nan, raw.astype(np.float32))
    # Diskretna kartografska lestvica: 0 = Kopno (#CBD5E1), 1 = Voda/Poplava (#0369A1)
    water_scale: list[list[Any]] = [
        [0.0, "#CBD5E1"],
        [0.499, "#CBD5E1"],
        [0.5, "#0369A1"],
        [1.0, "#0369A1"],
    ]
    title = (
        f"<b>{t('sar_water_map_title', thr=threshold_db, date=date_str)} | {tile_lbl}</b><br>"
        f"<span style='font-size:12px;color:#475569'>"
        f"{target_crs} | {resolution_m:.0f} m · "
        f"1 = {t('sar_water_legend_water')} (#0369A1) · 0 = {t('sar_water_legend_land')} (#CBD5E1)</span>"
    )
    fig = render_interactive_raster_plotly(
        mode="heatmap",
        x_coords=mask_da.coords["x"].values,
        y_coords=mask_da.coords["y"].values,
        z_data=z_arr,
        colorscale_override=water_scale,
        zmin=0.0,
        zmax=1.0,
        cbar_title=t("sar_water_legend_water"),
        title=title,
        target_crs=target_crs,
    )
    fig.data[0].colorbar.tickvals = [0.25, 0.75]
    fig.data[0].colorbar.ticktext = [t("sar_water_legend_land"), t("sar_water_legend_water")]
    return fig


def _build_index_plotly(
    index_da: xr.DataArray,
    index_key: str,
    target_crs: str,
    date_str: str,
    tile_lbl: str,
    resolution_m: float,
    sensor_name: str = "",
) -> go.Figure:
    spec = SPECTRAL_INDICES[index_key]
    prefix = f"{sensor_name} · " if sensor_name else ""
    title = (
        f"<b>{prefix}{index_key} ({date_str} | {tile_lbl})</b><br>"
        f"<span style='font-size:12px;color:#475569'>"
        f"{target_crs} | {resolution_m:.0f} m | {spec['formula']}</span>"
    )
    return render_interactive_raster_plotly(
        mode="heatmap",
        x_coords=index_da.coords["x"].values,
        y_coords=index_da.coords["y"].values,
        z_data=index_da.values,
        cmap_name=spec["cmap"],
        zmin=float(spec["vmin"]),
        zmax=float(spec["vmax"]),
        cbar_title=index_key,
        title=title,
        target_crs=target_crs,
    )


def _build_change_plotly(
    da_t1: xr.DataArray,
    da_t2: xr.DataArray,
    da_diff: xr.DataArray,
    index_key: str,
    target_crs: str,
    date_t1: str,
    date_t2: str,
    resolution_m: float,
    is_sar: bool = False,
    view_layer: str = "diff",
) -> go.Figure:
    """Interaktivni Plotly prikazovalnik za zaznavo sprememb (Δ, t1 ali t2) v fiksnem okviru."""
    spec = SAR_METRICS[index_key] if is_sar else SPECTRAL_INDICES[index_key]
    unit_suffix = " dB" if is_sar else ""
    unit_bracket = " [dB]" if is_sar else ""

    if view_layer == "t1":
        da_show = da_t1
        cmap_name = spec["cmap"]
        vmin, vmax = float(spec["vmin"]), float(spec["vmax"])
        cbar_lbl = f"{index_key}{unit_bracket}"
        panel_title = t("change_panel_t1", idx=f"{index_key}{unit_bracket}", d1=date_t1)
    elif view_layer == "t2":
        da_show = da_t2
        cmap_name = spec["cmap"]
        vmin, vmax = float(spec["vmin"]), float(spec["vmax"])
        cbar_lbl = f"{index_key}{unit_bracket}"
        panel_title = t("change_panel_t2", idx=f"{index_key}{unit_bracket}", d2=date_t2)
    else:
        da_show = da_diff
        cmap_name = spec["diff_cmap"]
        valid_diff = da_diff.values[np.isfinite(da_diff.values)]
        if is_sar:
            if valid_diff.size > 0:
                abs_lim = float(np.percentile(np.abs(valid_diff), 98.0))
                abs_lim = max(min(abs_lim, 12.0), 3.0)
            else:
                abs_lim = 6.0
        else:
            if valid_diff.size > 0:
                abs_lim = float(np.percentile(np.abs(valid_diff), 98.0))
                abs_lim = max(min(abs_lim, 1.0), 0.2)
            else:
                abs_lim = 0.5
        vmin, vmax = -abs_lim, abs_lim
        cbar_lbl = f"Δ{index_key}{unit_bracket}"
        panel_title = f"{t('change_panel_diff', idx=f'{index_key}{unit_bracket}')} ({date_t2} − {date_t1})"

    title = (
        f"<b>{panel_title}</b><br>"
        f"<span style='font-size:12px;color:#475569'>{target_crs} | {resolution_m:.0f} m</span>"
    )
    return render_interactive_raster_plotly(
        mode="heatmap",
        x_coords=da_show.coords["x"].values,
        y_coords=da_show.coords["y"].values,
        z_data=da_show.values,
        cmap_name=cmap_name,
        zmin=vmin,
        zmax=vmax,
        cbar_title=cbar_lbl,
        unit_suffix=unit_suffix,
        title=title,
        target_crs=target_crs,
    )


def _build_sar_vv_histogram(
    vv_da: xr.DataArray,
    threshold_db: float,
    date_str: str,
) -> go.Figure:
    """Interaktivni histogram SAR VV [dB] s pragom za klasifikacijo vodnih površin."""
    valid_vals = vv_da.values[np.isfinite(vv_da.values)]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=valid_vals,
            nbinsx=52,
            marker=dict(color="#0F766E", line=dict(color="#FFFFFF", width=0.5)),
            opacity=0.88,
            name="VV [dB]",
        )
    )
    fig.add_vline(
        x=float(threshold_db),
        line_width=2.2,
        line_dash="dash",
        line_color="#0369A1",
        annotation_text=f"VV < {threshold_db:.1f} dB",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#0369A1"),
    )
    fig.update_layout(
        title=dict(text=t("t2_hist_title", idx="VV [dB]", date=date_str), font=dict(size=14, color="#0F172A")),
        xaxis=dict(title=t("t2_hist_xaxis", idx="VV [dB]"), gridcolor="#E2E8F0", zerolinecolor="#94A3B8"),
        yaxis=dict(title=t("t2_hist_yaxis"), gridcolor="#E2E8F0"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=45, r=20, t=48, b=42),
        height=320,
        showlegend=False,
    )
    return fig


def _build_spectral_signature_chart(
    sig_df: pd.DataFrame,
    sensor_name: str,
    date_str: str,
) -> go.Figure:
    """Interaktivni Plotly linijski graf spektralnega podpisa (valovna dolžina vs. odbojnost)."""
    fig = go.Figure()
    labels = [f"{r['band_name']}<br>({r['band']})" for _, r in sig_df.iterrows()]
    fig.add_trace(
        go.Scatter(
            x=sig_df["wavelength_nm"],
            y=sig_df["reflectance"],
            mode="lines+markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=10, color="#334155"),
            line=dict(color="#0F766E", width=2.8),
            marker=dict(size=9, color="#0F766E", line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "λ = %{x} nm<br>"
                "ρ = %{y:.4f}<extra></extra>"
            ),
            name=sensor_name,
        )
    )
    fin_refl = sig_df["reflectance"].dropna() if not sig_df.empty else pd.Series(dtype=float)
    y_max = (
        min(1.05, max(0.55, float(fin_refl.max()) * 1.22))
        if not fin_refl.empty and np.isfinite(float(fin_refl.max()))
        else 0.55
    )
    fig.update_layout(
        title=dict(
            text=t("sig_chart_title", sensor=sensor_name, date=date_str),
            font=dict(size=14, color="#0F172A"),
        ),
        xaxis=dict(
            title=t("sig_xaxis"),
            tickvals=list(sig_df["wavelength_nm"]),
            ticktext=[f"{int(w)} nm" for w in sig_df["wavelength_nm"]],
            gridcolor="#E2E8F0",
            range=[420, 2320],
        ),
        yaxis=dict(
            title=t("sig_yaxis"),
            gridcolor="#E2E8F0",
            zerolinecolor="#94A3B8",
            range=[0.0, y_max],
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=52, r=24, t=52, b=46),
        height=350,
        showlegend=False,
    )
    return fig


def _build_land_cover_bar_chart(
    lc_df: pd.DataFrame,
    date_str: str,
    total_valid_ha: float,
    lang: str,
) -> go.Figure:
    """Horizontalni Plotly stolpčni graf površin [ha] in deležev [%] za 4 razrede rabe tal."""
    labels = [t(f"lc_{ck}", lang=lang) for ck in lc_df["class_key"]]
    text_ann = [f"{a:,.2f} ha ({p:.1f} %)" for a, p in zip(lc_df["area_ha"], lc_df["share_pct"])]
    fig = go.Figure(
        data=go.Bar(
            y=labels,
            x=lc_df["area_ha"],
            orientation="h",
            marker=dict(
                color=list(lc_df["color"]),
                line=dict(color="#FFFFFF", width=1),
            ),
            text=text_ann,
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>%{x:,.2f} ha<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=t("lc_chart_title", date=date_str, area=total_valid_ha),
            font=dict(size=14, color="#0F172A"),
        ),
        xaxis=dict(title=t("lc_col_area"), gridcolor="#E2E8F0"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=24, t=48, b=42),
        height=280,
        showlegend=False,
    )
    return fig


def _render_before_after_swipe_html(
    png_t1_bytes: bytes,
    png_t2_bytes: bytes,
    label_t1: str,
    label_t2: str,
    height: int = 520,
) -> None:
    """Izriše samostojni HTML5/CSS/JS drsnik Prej / Potem (t₁ levo, t₂ desno) z miško ali dotikom."""
    b64_t1 = base64.b64encode(png_t1_bytes).decode("ascii")
    b64_t2 = base64.b64encode(png_t2_bytes).decode("ascii")
    safe_t1 = label_t1.replace("<", "&lt;").replace(">", "&gt;")
    safe_t2 = label_t2.replace("<", "&lt;").replace(">", "&gt;")

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    background: #FFFFFF;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 4px;
  }}
  .swipe-shell {{
    position: relative;
    width: 100%;
    max-width: 760px;
    height: {height - 16}px;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    overflow: hidden;
    background: #F1F5F9;
    user-select: none;
    touch-action: pan-y;
  }}
  .swipe-img {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
    pointer-events: none;
  }}
  .swipe-left-wrap {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    clip-path: inset(0 50% 0 0);
    pointer-events: none;
  }}
  .swipe-divider {{
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 3px;
    margin-left: -1.5px;
    background: #FFFFFF;
    box-shadow: 0 0 6px rgba(15, 23, 42, 0.55);
    pointer-events: none;
  }}
  .swipe-handle {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #0F766E;
    border: 2.5px solid #FFFFFF;
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.35);
    pointer-events: none;
  }}
  .swipe-badge {{
    position: absolute;
    top: 12px;
    padding: 5px 10px;
    border-radius: 5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: 0.01em;
    pointer-events: none;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.35);
  }}
  .badge-left {{
    left: 12px;
    background: rgba(15, 23, 42, 0.88);
    border-left: 3px solid #94A3B8;
  }}
  .badge-right {{
    right: 12px;
    background: rgba(15, 118, 110, 0.92);
    border-right: 3px solid #F8FAFC;
  }}
  .swipe-range {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: ew-resize;
    z-index: 10;
    margin: 0;
  }}
</style>
</head>
<body>
  <div class="swipe-shell" id="shell">
    <img class="swipe-img" src="data:image/png;base64,{b64_t2}" alt="t2" />
    <div class="swipe-left-wrap" id="leftWrap">
      <img class="swipe-img" src="data:image/png;base64,{b64_t1}" alt="t1" />
    </div>
    <div class="swipe-divider" id="divider"></div>
    <div class="swipe-handle" id="handle">&#8596;</div>
    <div class="swipe-badge badge-left">{safe_t1}</div>
    <div class="swipe-badge badge-right">{safe_t2}</div>
    <input class="swipe-range" id="slider" type="range" min="0" max="100" value="50" aria-label="Before After Swipe" />
  </div>
  <script>
    const slider = document.getElementById('slider');
    const leftWrap = document.getElementById('leftWrap');
    const divider = document.getElementById('divider');
    const handle = document.getElementById('handle');
    function updatePos(val) {{
      const pct = Math.max(0, Math.min(100, Number(val)));
      leftWrap.style.clipPath = `inset(0 ${{100 - pct}}% 0 0)`;
      divider.style.left = pct + '%';
      handle.style.left = pct + '%';
    }}
    slider.addEventListener('input', (e) => updatePos(e.target.value));
    slider.addEventListener('change', (e) => updatePos(e.target.value));
    updatePos(50);
  </script>
</body>
</html>"""
    _render_html_iframe(html_doc, height=height)


def _build_index_histogram(
    index_da: xr.DataArray,
    stats: dict[str, float | int],
    index_key: str,
    date_str: str,
) -> go.Figure:
    valid_vals = index_da.values[np.isfinite(index_da.values)]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=valid_vals,
            nbinsx=48,
            marker=dict(color="#0F766E", line=dict(color="#FFFFFF", width=0.5)),
            opacity=0.88,
            name=index_key,
        )
    )
    if np.isfinite(stats["mean"]):
        fig.add_vline(
            x=float(stats["mean"]),
            line_width=2,
            line_dash="solid",
            line_color="#0F172A",
            annotation_text=f"{t('m_mean')}: {stats['mean']:.3f}",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#0F172A"),
        )
    if np.isfinite(stats["median"]):
        fig.add_vline(
            x=float(stats["median"]),
            line_width=1.8,
            line_dash="dash",
            line_color="#B45309",
            annotation_text=f"{t('m_median')}: {stats['median']:.3f}",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#B45309"),
        )

    fig.update_layout(
        title=dict(text=t("t2_hist_title", idx=index_key, date=date_str), font=dict(size=14, color="#0F172A")),
        xaxis=dict(title=t("t2_hist_xaxis", idx=index_key), gridcolor="#E2E8F0", zerolinecolor="#94A3B8"),
        yaxis=dict(title=t("t2_hist_yaxis"), gridcolor="#E2E8F0"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=45, r=20, t=48, b=42),
        height=330,
        showlegend=False,
    )
    return fig


def _build_time_series_chart(
    ts_df: pd.DataFrame,
    metric_label: str,
    target_crs: str,
    resolution_m: int,
    unit_suffix: str = "",
) -> go.Figure:
    fig = go.Figure()

    # Subtilen pas P10–P90 v kartografski petrol-zeleni barvi
    fig.add_trace(
        go.Scatter(
            x=ts_df["date_str"],
            y=ts_df["p90"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            name="P90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts_df["date_str"],
            y=ts_df["p10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(15, 118, 110, 0.14)",
            name=t("ts_trace_band"),
            hoverinfo="skip",
        )
    )

    customdata = np.stack(
        [ts_df["valid_pct"].values, ts_df["p10"].values, ts_df["p90"].values, ts_df["std"].values],
        axis=-1,
    )
    hover_valid = t("ts_hover_valid")

    fig.add_trace(
        go.Scatter(
            x=ts_df["date_str"],
            y=ts_df["mean"],
            mode="lines+markers",
            name=t("ts_trace_mean"),
            line=dict(color="#0F766E", width=2.5),
            marker=dict(size=7, color="#0F766E", line=dict(color="#FFFFFF", width=1.2)),
            customdata=customdata,
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{t('ts_trace_mean')}: %{{y:.3f}}{unit_suffix} (±%{{customdata[3]:.3f}})<br>"
                f"P10–P90: %{{customdata[1]:.3f}} – %{{customdata[2]:.3f}}{unit_suffix}<br>"
                f"{hover_valid}: %{{customdata[0]:.1f}}%<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ts_df["date_str"],
            y=ts_df["median"],
            mode="lines+markers",
            name=t("ts_trace_median"),
            line=dict(color="#0F172A", width=1.8, dash="dash"),
            marker=dict(size=5, color="#0F172A"),
            customdata=customdata,
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{t('ts_trace_median')}: %{{y:.3f}}{unit_suffix}<br>"
                f"{hover_valid}: %{{customdata[0]:.1f}}%<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=t("ts_chart_title", idx=metric_label, crs=target_crs, res=resolution_m),
            font=dict(size=15, color="#0F172A"),
        ),
        xaxis=dict(title=t("ts_xaxis"), gridcolor="#E2E8F0", tickangle=-20),
        yaxis=dict(
            title=t("ts_yaxis", idx=f"{metric_label}{unit_suffix}"),
            gridcolor="#E2E8F0",
            zerolinecolor="#94A3B8",
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#0F172A"),
        ),
        margin=dict(l=50, r=20, t=65, b=50),
        height=410,
    )
    return fig


def _epoch_sort_key(item_or_group: Any) -> str:
    """Izvleče ISO časovni žig za kronološko razvrščanje posameznih ali združenih STAC posnetkov."""
    first = item_or_group[0] if isinstance(item_or_group, (list, tuple)) else item_or_group
    dt = getattr(first, "datetime", None)
    if dt is not None:
        return dt.isoformat()
    return str(first.properties.get("datetime", ""))


def _select_evenly_spaced_items(items: list[Any], max_epochs: int) -> list[Any]:
    """Kronološko izbere največ ``max_epochs`` enakomerno razporejenih posnetkov."""
    if len(items) <= max_epochs:
        return list(items)
    ordered = sorted(items, key=_epoch_sort_key)
    indices = np.linspace(0, len(ordered) - 1, num=max_epochs, dtype=int)
    seen: set[int] = set()
    selected = []
    for idx in indices:
        i = int(idx)
        if i not in seen:
            seen.add(i)
            selected.append(ordered[i])
    return selected


def _resolve_active_aoi_state(
    lang: str,
) -> tuple[str, list[float] | None, list[float] | None, dict[str, Any] | None, float, float, str | None, str]:
    """Iz stanja seje izračuna aktivni AOI (bbox, poligon, središče in oznako lokacije)."""
    aoi_mode_key = str(st.session_state.get("sidebar_aoi_mode", "map"))
    if aoi_mode_key not in WS1_AOI_MODE_KEYS:
        aoi_mode_key = "map"
        st.session_state["sidebar_aoi_mode"] = "map"

    center_lat = float(st.session_state.get("map_lat", SLOVENIA_CENTER_LAT))
    center_lon = float(st.session_state.get("map_lon", SLOVENIA_CENTER_LON))
    radius_m = int(st.session_state.get("map_radius_m", 2000))
    bbox_4326: list[float] | None = None
    bbox_3794: list[float] | None = None
    aoi_polygon_4326: dict[str, Any] | None = None
    aoi_error: str | None = None
    loc_label = str(st.session_state.get("active_location_label") or f"{center_lat:.4f}°, {center_lon:.4f}°")

    if aoi_mode_key == "draw":
        drawn_poly = st.session_state.get("drawn_polygon_4326")
        if drawn_poly is not None:
            aoi_polygon_4326 = drawn_poly
            bbox_4326, bbox_3794, center_lat, center_lon = _polygon_bounds_to_bboxes(drawn_poly)
            loc_label = "Narisani poligon" if lang == "sl" else "Drawn Polygon"
        else:
            loc_label = "Narišite poligon..." if lang == "sl" else "Draw polygon..."
    elif aoi_mode_key == "geojson":
        gj = st.session_state.get("uploaded_geojson_aoi")
        if isinstance(gj, dict) and gj.get("bbox_4326") is not None:
            bbox_4326 = list(gj["bbox_4326"])
            bbox_3794 = list(gj["bbox_3794"]) if gj.get("bbox_3794") is not None else None
            aoi_polygon_4326 = gj.get("polygon")
            center_lon = 0.5 * (bbox_4326[0] + bbox_4326[2])
            center_lat = 0.5 * (bbox_4326[1] + bbox_4326[3])
            loc_label = str(gj.get("name") or "GeoJSON")
        else:
            loc_label = "GeoJSON (ni naložen)" if lang == "sl" else "GeoJSON (not loaded)"
    else:
        bbox_4326, bbox_3794 = bbox_from_point_wgs84(center_lat, center_lon, float(radius_m))

    return aoi_mode_key, bbox_4326, bbox_3794, aoi_polygon_4326, center_lat, center_lon, aoi_error, loc_label


def _sync_ws1_radius() -> None:
    val = st.session_state.get("ws1_map_radius_slider")
    if val is not None:
        max_r = int(get_ram_limits()["max_radius_m"])
        st.session_state["map_radius_m"] = min(int(val), max_r)


def _sync_ws1_aoi_mode() -> None:
    val = st.session_state.get("ws1_aoi_mode_ctrl")
    if val in WS1_AOI_MODE_KEYS:
        st.session_state["sidebar_aoi_mode"] = val


def _apply_slovenia_preset_ws1(preset_name: str) -> None:
    spec = SLOVENIA_PRESETS.get(preset_name)
    if spec is None:
        return
    max_r = int(get_ram_limits()["max_radius_m"])
    clamped_r = min(int(spec["radius_m"]), max_r)
    st.session_state["map_lat"] = float(spec["lat"])
    st.session_state["map_lon"] = float(spec["lon"])
    st.session_state["map_radius_m"] = clamped_r
    st.session_state["ws1_map_radius_slider"] = clamped_r
    st.session_state["active_location_label"] = preset_name
    st.session_state["map_zoom"] = 12
    st.session_state["live_map_zoom"] = 12
    st.session_state["map_jump_version"] = int(st.session_state.get("map_jump_version", 0)) + 1
    st.session_state["last_processed_click"] = None
    st.session_state["geocoded_place_msg"] = None


def _render_interactive_aoi_section(
    *,
    aoi_mode_key: str,
    center_lat: float,
    center_lon: float,
    bbox_3794: list[float] | None,
    bbox_4326: list[float] | None,
    aoi_polygon_4326: dict[str, Any] | None,
    lang: str,
) -> None:
    """Izriše enotno vrstico (Senzor | Podlaga | Kraj / D96/TM / WGS84 | Način AOI) in 600 px zemljevid AOI."""
    prev_basemap = str(st.session_state.get("map_basemap", "satellite"))
    if prev_basemap not in BASEMAP_KEYS:
        prev_basemap = "satellite"

    place_search_clicked = False
    place_query = ""
    d96_jump_clicked = False
    e_jump = 0.0
    n_jump = 0.0
    wgs_jump_clicked = False
    lat_jump = center_lat
    lon_jump = center_lon

    with st.container(border=True):
        # Vrstica 1: Satelitski senzor + Podlaga zemljevida (dotiku prijazni gumbi brez tipkovnice)
        col_sensor, col_bm = st.columns([1.35, 1.0], vertical_alignment="bottom")
        with col_sensor:
            _touch_selector(
                t("sensor_select_label"),
                options=SENSOR_KEYS,
                default=str(st.session_state.get("selected_sensor", SENSOR_S2)),
                format_func=lambda s, _lang=lang: t(SENSOR_META[s]["pill_key"], lang=_lang),
                key="main_bar_sensor_select",
                on_change=_sync_sensor_state,
                args=("main_bar_sensor_select",),
            )
        with col_bm:
            selected_basemap = _touch_selector(
                t("basemap_selector_label"),
                options=BASEMAP_KEYS,
                default=prev_basemap,
                format_func=lambda k, _lang=lang: t(BASEMAP_SPECS[k]["short_label_key"], lang=_lang),
                key="main_map_basemap_ctrl",
            )

        # Vrstica 2: Preklopnik načina iskanja (Kraj / D96/TM / WGS84) + obrazec za takojšen skok
        loc_mode = _touch_selector(
            t("loc_mode_label"),
            options=LOC_MODE_KEYS,
            default=str(st.session_state.get("loc_search_mode", "place")),
            format_func=lambda k, _lang=lang: t(LOC_MODE_I18N[k], lang=_lang),
            key="main_loc_search_mode_ctrl",
            label_visibility="collapsed",
        )
        st.session_state["loc_search_mode"] = loc_mode

        center_sig = f"{round(center_lat, 5)}_{round(center_lon, 5)}"

        if loc_mode == "place":
            with st.form("main_place_search_form", border=False):
                c_in, c_go = st.columns([3.4, 1.0], vertical_alignment="bottom")
                with c_in:
                    place_query = st.text_input(
                        t("place_search_label"),
                        value="",
                        placeholder=t("place_search_placeholder_smart"),
                        label_visibility="collapsed",
                        key="main_place_search_input",
                    )
                with c_go:
                    place_search_clicked = st.form_submit_button(
                        t("btn_place_search_short"),
                        type="primary",
                        width="stretch",
                    )

        elif loc_mode == "d96":
            e_cur, n_cur = transform_coords(center_lon, center_lat, EPSG_WGS84, EPSG_D96TM)
            with st.form("main_d96_jump_form", border=False):
                c_e, c_n, c_btn = st.columns([1.2, 1.2, 1.15], vertical_alignment="bottom")
                with c_e:
                    e_jump = st.number_input(
                        t("e_input_label"),
                        value=round(float(e_cur), 1),
                        step=100.0,
                        format="%.1f",
                        key=f"main_d96_e_{center_sig}",
                    )
                with c_n:
                    n_jump = st.number_input(
                        t("n_input_label"),
                        value=round(float(n_cur), 1),
                        step=100.0,
                        format="%.1f",
                        key=f"main_d96_n_{center_sig}",
                    )
                with c_btn:
                    d96_jump_clicked = st.form_submit_button(
                        t("btn_jump_coords"),
                        type="primary",
                        width="stretch",
                    )

        else:
            with st.form("main_wgs84_jump_form", border=False):
                c_lat, c_lon, c_btn = st.columns([1.2, 1.2, 1.15], vertical_alignment="bottom")
                with c_lat:
                    lat_jump = st.number_input(
                        t("lat_input_label"),
                        min_value=-90.0,
                        max_value=90.0,
                        value=round(float(center_lat), 6),
                        step=0.005,
                        format="%.6f",
                        key=f"main_wgs84_lat_{center_sig}",
                    )
                with c_lon:
                    lon_jump = st.number_input(
                        t("lon_input_label"),
                        min_value=-180.0,
                        max_value=180.0,
                        value=round(float(center_lon), 6),
                        step=0.005,
                        format="%.6f",
                        key=f"main_wgs84_lon_{center_sig}",
                    )
                with c_btn:
                    wgs_jump_clicked = st.form_submit_button(
                        t("btn_jump_coords"),
                        type="primary",
                        width="stretch",
                    )

        # Vrstica 3: Kompakten nadzor AOI (Klik+radij / Risanje / Prednastavitve SLO / GeoJSON)
        c_aoi_mode, c_aoi_ctrl = st.columns([1.45, 1.25], vertical_alignment="bottom")
        with c_aoi_mode:
            st.session_state["ws1_aoi_mode_ctrl"] = aoi_mode_key
            _touch_selector(
                t("ws1_aoi_mode_label"),
                options=WS1_AOI_MODE_KEYS,
                default=aoi_mode_key,
                format_func=lambda k, _lang=lang: t(WS1_AOI_MODE_I18N[k], lang=_lang),
                key="ws1_aoi_mode_ctrl",
                on_change=_sync_ws1_aoi_mode,
                use_pills=True,
            )
        with c_aoi_ctrl:
            ram_lims_aoi = get_ram_limits()
            max_rad_m = int(ram_lims_aoi["max_radius_m"])
            max_area_lim = float(ram_lims_aoi["max_area_km2"])
            if aoi_mode_key in ("map", "preset"):
                cur_rad = min(int(st.session_state.get("map_radius_m", 2000)), max_rad_m)
                st.session_state["map_radius_m"] = cur_rad
                if "ws1_map_radius_slider" not in st.session_state:
                    st.session_state["ws1_map_radius_slider"] = cur_rad
                elif int(st.session_state["ws1_map_radius_slider"]) > max_rad_m:
                    st.session_state["ws1_map_radius_slider"] = max_rad_m
                st.slider(
                    t("radius_label"),
                    min_value=500,
                    max_value=max_rad_m,
                    step=250,
                    key="ws1_map_radius_slider",
                    on_change=_sync_ws1_radius,
                    help=t("radius_help"),
                )
            elif aoi_mode_key == "draw":
                if st.session_state.get("drawn_polygon_4326") is not None:
                    if st.button(t("btn_clear_polygon"), width="stretch", key="ws1_btn_clear_drawn_poly"):
                        st.session_state["drawn_polygon_4326"] = None
                        st.session_state["draw_map_version"] = int(st.session_state["draw_map_version"]) + 1
                        st.rerun()
                else:
                    st.caption(f"✏️ {t('draw_waiting_status')}")
            else:
                uploaded_geojson = st.file_uploader(
                    t("geojson_upload_label"),
                    type=["geojson", "json"],
                    help=t("geojson_upload_help"),
                    label_visibility="collapsed",
                    key="ws1_geojson_uploader",
                )
                if uploaded_geojson is not None:
                    try:
                        gj_b4326, gj_b3794, gj_poly = _parse_geojson_aoi(uploaded_geojson)
                        gj_poly, was_clamped_gj, _ = clamp_polygon_to_max_area(
                            gj_poly, max_area_km2=max_area_lim
                        )
                        if was_clamped_gj:
                            from shapely.geometry import shape as _shp_gj

                            gj_b4326 = list(_shp_gj(gj_poly).bounds)
                            st.session_state["aoi_clamped_notice"] = max_area_lim
                        prev_gj = st.session_state.get("uploaded_geojson_aoi")
                        if not isinstance(prev_gj, dict) or prev_gj.get("name") != uploaded_geojson.name:
                            st.session_state["uploaded_geojson_aoi"] = {
                                "bbox_4326": gj_b4326,
                                "bbox_3794": gj_b3794,
                                "polygon": gj_poly,
                                "name": uploaded_geojson.name,
                            }
                            st.session_state["active_location_label"] = uploaded_geojson.name
                            st.rerun()
                    except Exception as exc:
                        st.error(t("geojson_error", err=exc))
                elif st.session_state.get("uploaded_geojson_aoi"):
                    st.caption(f"✅ {st.session_state['uploaded_geojson_aoi'].get('name', 'GeoJSON')}")

        if aoi_mode_key == "preset":
            preset_opts = list(SLOVENIA_PRESETS.keys())
            cur_preset = st.session_state.get("ws1_preset_select")
            if cur_preset not in preset_opts:
                cur_preset = preset_opts[0]
            chosen_preset = _touch_selector(
                t("preset_select_label"),
                options=preset_opts,
                default=cur_preset,
                key="ws1_preset_select",
                use_pills=True,
                label_visibility="collapsed",
            )
            if chosen_preset != st.session_state.get("__ws1_last_applied_preset"):
                st.session_state["__ws1_last_applied_preset"] = chosen_preset
                _apply_slovenia_preset_ws1(chosen_preset)
                st.rerun()
            st.caption(t(SLOVENIA_PRESETS[chosen_preset]["desc_key"]))

    if selected_basemap != prev_basemap:
        st.session_state["map_basemap"] = selected_basemap
        live_z = st.session_state.get("live_map_zoom")
        if live_z is not None:
            st.session_state["map_zoom"] = int(live_z)

    if place_search_clicked and place_query.strip():
        try:
            geo_res = geocode_place(place_query, lang=lang)
            if geo_res is not None:
                found_lat, found_lon, found_name = geo_res
                _jump_to_location(found_lat, found_lon, found_name, zoom=12)
                st.rerun()
            else:
                st.warning(t("place_not_found_warn", query=place_query.strip()))
        except Exception as exc:
            st.error(t("place_search_error", err=exc))

    if d96_jump_clicked:
        try:
            j_lon, j_lat = transform_coords(float(e_jump), float(n_jump), EPSG_D96TM, EPSG_WGS84)
            _jump_to_location(
                float(j_lat),
                float(j_lon),
                f"D96/TM ({float(e_jump):,.0f}, {float(n_jump):,.0f})",
                zoom=12,
            )
            st.rerun()
        except Exception as exc:
            st.error(t("place_search_error", err=exc))

    if wgs_jump_clicked:
        _jump_to_location(
            float(lat_jump),
            float(lon_jump),
            f"WGS84 ({float(lat_jump):.4f}°, {float(lon_jump):.4f}°)",
            zoom=12,
        )
        st.rerun()

    if st.session_state.get("geocoded_place_msg"):
        st.success(f"📍 {st.session_state['geocoded_place_msg']}")

    jump_ver = int(st.session_state.get("map_jump_version", 0))

    # Način 1: Interaktivno risanje poligona / pravokotnika na 600 px zemljevidu
    if aoi_mode_key == "draw":
        st.caption(t("draw_main_hint"))
        draw_zoom = int(st.session_state.get("map_zoom", 11))
        draw_map = _build_draw_folium_map(
            center_lat=center_lat,
            center_lon=center_lon,
            existing_polygon_4326=st.session_state.get("drawn_polygon_4326"),
            zoom_start=draw_zoom,
            basemap_key=selected_basemap,
        )
        draw_ver = int(st.session_state.get("draw_map_version", 0))
        draw_out = st_folium(
            draw_map,
            center=(center_lat, center_lon),
            zoom=draw_zoom,
            height=470,
            use_container_width=True,
            returned_objects=["last_active_drawing", "all_drawings", "zoom"],
            key=f"aoi_draw_map_{draw_ver}_{jump_ver}_{selected_basemap}",
        )
        if isinstance(draw_out, dict) and draw_out.get("zoom") is not None:
            try:
                st.session_state["live_map_zoom"] = int(draw_out["zoom"])
            except (TypeError, ValueError):
                pass
        extracted_poly = _extract_drawn_polygon(draw_out)
        if extracted_poly is not None:
            ram_max_draw = float(get_ram_limits()["max_area_km2"])
            extracted_poly, was_clamped_draw, _ = clamp_polygon_to_max_area(
                extracted_poly, max_area_km2=ram_max_draw
            )
            prev_poly = st.session_state.get("drawn_polygon_4326")
            if json.dumps(extracted_poly, sort_keys=True) != json.dumps(prev_poly, sort_keys=True):
                if isinstance(draw_out, dict) and draw_out.get("zoom") is not None:
                    try:
                        st.session_state["map_zoom"] = int(draw_out["zoom"])
                    except (TypeError, ValueError):
                        pass
                st.session_state["drawn_polygon_4326"] = extracted_poly
                if was_clamped_draw:
                    st.session_state["aoi_clamped_notice"] = ram_max_draw
                    st.session_state["draw_map_version"] = int(st.session_state.get("draw_map_version", 0)) + 1
                st.rerun()

        clamped_notice = st.session_state.pop("aoi_clamped_notice", None)
        if clamped_notice is not None:
            st.info(t("aoi_clamped_info", max_area=float(clamped_notice)))

    # Način 2: Velik interaktivni zemljevid (klik posodobi središče AOI)
    else:
        is_click_mode = aoi_mode_key in ("map", "preset")
        if is_click_mode:
            st.caption(t("map_click_hint"))

        zoom_val = int(st.session_state.get("map_zoom", 10))
        main_map = _build_aoi_folium_map(
            center_lat=center_lat,
            center_lon=center_lon,
            bbox_3794=bbox_3794,
            bbox_4326=bbox_4326,
            polygon_geojson_4326=aoi_polygon_4326,
            zoom_start=zoom_val,
            interactive_hint=is_click_mode,
            basemap_key=selected_basemap,
        )
        returned_objs = ["last_clicked", "zoom"] if is_click_mode else ["zoom"]
        map_out = st_folium(
            main_map,
            center=(center_lat, center_lon),
            zoom=zoom_val,
            height=470,
            use_container_width=True,
            returned_objects=returned_objs,
            key=f"main_interactive_aoi_map_{jump_ver}_{selected_basemap}",
        )
        if isinstance(map_out, dict):
            raw_zoom = map_out.get("zoom")
            if raw_zoom is not None:
                try:
                    st.session_state["live_map_zoom"] = int(raw_zoom)
                except (TypeError, ValueError):
                    pass

            if is_click_mode:
                clicked = map_out.get("last_clicked")
                if isinstance(clicked, dict) and "lat" in clicked and "lng" in clicked:
                    new_lat = round(float(clicked["lat"]), 6)
                    new_lon = round(float(clicked["lng"]), 6)
                    click_sig = (new_lat, new_lon)
                    if click_sig != st.session_state.get("last_processed_click"):
                        st.session_state["last_processed_click"] = click_sig
                        if abs(new_lat - center_lat) > 1e-6 or abs(new_lon - center_lon) > 1e-6:
                            st.session_state["map_lat"] = new_lat
                            st.session_state["map_lon"] = new_lon
                            st.session_state["active_location_label"] = f"{new_lat:.4f}°, {new_lon:.4f}°"
                            if raw_zoom is not None:
                                try:
                                    st.session_state["map_zoom"] = int(raw_zoom)
                                except (TypeError, ValueError):
                                    pass
                            elif st.session_state.get("live_map_zoom") is not None:
                                st.session_state["map_zoom"] = int(st.session_state["live_map_zoom"])
                            st.session_state["geocoded_place_msg"] = None
                            st.rerun()


# ---------------------------------------------------------------------------
# Glavni tok aplikacije
# ---------------------------------------------------------------------------
def _render_relief_and_profile_section(
    *,
    target_bbox: list[float] | None,
    target_polygon: dict[str, Any] | None,
    target_crs: str,
    bbox_metric: list[float] | None,
    center_lat: float,
    center_lon: float,
    aoi_valid: bool,
    lang: str,
    key_prefix: str = "tab2_rel",
) -> None:
    """Izriše modul ⛰️ Relief (Copernicus DEM 30 m) in Vzdolžni prerez (A → B) z opcijskim indeksom."""
    st.subheader(t("relief_header"))
    st.caption(t("relief_intro"))

    can_load = (target_bbox is not None) and aoi_valid
    load_dem_clicked = st.button(
        t("relief_btn_load"),
        type="primary",
        width="stretch",
        disabled=not can_load,
        key=f"{key_prefix}_btn_load_dem",
    )

    if load_dem_clicked and target_bbox is not None:
        with st.spinner(t("relief_spinner", crs=format_crs_label(target_crs))):
            try:
                st.session_state["tab2_relief_data"] = None
                st.session_state["tab2_relief_sent_to_carto"] = False
                gc.collect()

                dem_da = load_relief_dem(
                    bbox_4326=target_bbox,
                    target_epsg=target_crs,
                    resolution=30.0,
                    aoi_polygon_4326=target_polygon,
                )
                derivs = compute_terrain_derivatives(dem_da)
                derivs["crs"] = target_crs
                derivs["bbox_4326"] = list(target_bbox)
                derivs["bbox_metric"] = list(bbox_metric) if bbox_metric else None
                derivs["polygon_4326"] = target_polygon
                st.session_state["tab2_relief_data"] = derivs
                gc.collect()
            except Exception as exc:
                st.error(t("relief_error", err=exc))

    rel_data: dict[str, Any] | None = st.session_state.get("tab2_relief_data")
    if not isinstance(rel_data, dict) or rel_data.get("elevation") is None:
        return

    r_stats: dict[str, Any] = rel_data["stats"]
    r_crs = str(rel_data.get("crs", target_crs))
    dem_elev_da: xr.DataArray = rel_data["elevation"]

    # 1. KPI vrstica za relief
    rk1, rk2, rk3, rk4, rk5 = st.columns(5)
    rk1.metric(
        t("relief_kpi_elev"),
        f"{r_stats['mean_h']:,.0f} m",
        delta=f"{r_stats['min_h']:,.0f} – {r_stats['max_h']:,.0f} m",
        delta_color="off",
    )
    rk2.metric(t("relief_kpi_diff"), f"{r_stats['relief_diff_m']:,.0f} m")
    rk3.metric(
        t("relief_kpi_slope"),
        f"{r_stats['mean_slope_deg']:.1f}°",
        delta=f"max {r_stats['max_slope_deg']:.1f}°",
        delta_color="off",
    )
    rk4.metric(
        t("relief_kpi_steep20"),
        f"{r_stats['steep_20_ha']:,.1f} ha",
        delta=f"{r_stats['steep_20_pct']:.1f} %",
        delta_color="off",
    )
    rk5.metric(
        t("relief_kpi_steep30"),
        f"{r_stats['steep_30_ha']:,.1f} ha",
        delta=f"{r_stats['steep_30_pct']:.1f} %",
        delta_color="off",
    )

    # 2. Izbor sloja reliefa (dotiku prijazni žetoni) + prosojnost
    col_rl, col_rop = st.columns([1.6, 1.0], vertical_alignment="bottom")
    with col_rl:
        r_layer_key = _touch_selector(
            t("relief_layer_label"),
            options=RELIEF_LAYER_KEYS,
            default="elevation",
            format_func=lambda k, _lang=lang: t(RELIEF_LAYER_I18N[k], lang=_lang),
            key="tab2_relief_layer_sel",
            use_pills=True,
        )
    with col_rop:
        r_opacity = st.slider(
            t("relief_opacity_label"),
            min_value=0,
            max_value=100,
            value=85,
            step=5,
            key=f"{key_prefix}_opacity",
        )

    # 3. Nastavitev profila A → B (da ga lahko takoj narišemo tudi na interaktivnem zemljevidu!)
    st.divider()
    st.markdown(f"#### {t('profile_header')}")
    st.caption(t("profile_caption"))

    pr_preset = _touch_selector(
        t("profile_preset_label"),
        options=PROFILE_PRESET_KEYS,
        default="w_e",
        format_func=lambda k, _lang=lang: t(PROFILE_PRESET_I18N[k], lang=_lang),
        key=f"{key_prefix}_profile_preset",
        use_pills=True,
    )

    x_vals_d = np.asarray(dem_elev_da.coords["x"].values, dtype=np.float64)
    y_vals_d = np.asarray(dem_elev_da.coords["y"].values, dtype=np.float64)
    x_min_d, x_max_d = float(np.min(x_vals_d)), float(np.max(x_vals_d))
    y_min_d, y_max_d = float(np.min(y_vals_d)), float(np.max(y_vals_d))

    if pr_preset != "custom":
        pt_a, pt_b = build_preset_transect_endpoints(dem_elev_da, preset=pr_preset)
    else:
        def_a, def_b = build_preset_transect_endpoints(dem_elev_da, preset="w_e")
        pa1, pa2, pb1, pb2 = st.columns(4)
        with pa1:
            xa_in = st.number_input(
                t("profile_pt_a_e"),
                min_value=x_min_d,
                max_value=x_max_d,
                value=float(np.clip(def_a[0], x_min_d, x_max_d)),
                step=100.0,
                format="%.1f",
                key=f"{key_prefix}_xa",
            )
        with pa2:
            ya_in = st.number_input(
                t("profile_pt_a_n"),
                min_value=y_min_d,
                max_value=y_max_d,
                value=float(np.clip(def_a[1], y_min_d, y_max_d)),
                step=100.0,
                format="%.1f",
                key=f"{key_prefix}_ya",
            )
        with pb1:
            xb_in = st.number_input(
                t("profile_pt_b_e"),
                min_value=x_min_d,
                max_value=x_max_d,
                value=float(np.clip(def_b[0], x_min_d, x_max_d)),
                step=100.0,
                format="%.1f",
                key=f"{key_prefix}_xb",
            )
        with pb2:
            yb_in = st.number_input(
                t("profile_pt_b_n"),
                min_value=y_min_d,
                max_value=y_max_d,
                value=float(np.clip(def_b[1], y_min_d, y_max_d)),
                step=100.0,
                format="%.1f",
                key=f"{key_prefix}_yb",
            )
        pt_a = (float(xa_in), float(ya_in))
        pt_b = (float(xb_in), float(yb_in))

    # Pretvori A in B v WGS84 za prikaz oranžne linije A → B na interaktivnem Slippy zemljevidu
    transect_ll = None
    try:
        lon_a, lat_a = transform_coords(pt_a[0], pt_a[1], r_crs, EPSG_WGS84)
        lon_b, lat_b = transform_coords(pt_b[0], pt_b[1], r_crs, EPSG_WGS84)
        transect_ll = ((lat_a, lon_a), (lat_b, lon_b))
    except Exception:
        transect_ll = None

    # 4. Interaktivni Slippy zemljevid reliefa + barvna lestvica + GeoTIFF / Kartografija
    active_r_da: xr.DataArray = rel_data[r_layer_key]
    r_spec = TERRAIN_LAYER_SPECS[r_layer_key]
    r_lbl = t(RELIEF_LAYER_I18N[r_layer_key], lang=lang)
    if r_spec.get("auto_range"):
        r_vmin = float(r_stats.get("min_h", 0.0))
        r_vmax = max(r_vmin + 1.0, float(r_stats.get("max_h", 2500.0)))
    else:
        r_vmin = float(r_spec["vmin"])
        r_vmax = float(r_spec["vmax"])

    st.markdown(
        build_horizontal_colorbar_html(
            cmap_name=str(r_spec["cmap"]),
            vmin=r_vmin,
            vmax=r_vmax,
            label=f"Copernicus DEM 30 m — {r_lbl}",
            units=str(r_spec.get("units", "")),
        ),
        unsafe_allow_html=True,
    )

    r_uri, r_bnds = raster_to_map_overlay_png(
        active_r_da,
        src_epsg=r_crs,
        is_rgb=False,
        cmap_name=str(r_spec["cmap"]),
        vmin=r_vmin,
        vmax=r_vmax,
    )
    r_bbox_4326 = rel_data.get("bbox_4326") or target_bbox
    c_lat_r = 0.5 * (r_bbox_4326[1] + r_bbox_4326[3]) if r_bbox_4326 else center_lat
    c_lon_r = 0.5 * (r_bbox_4326[0] + r_bbox_4326[2]) if r_bbox_4326 else center_lon
    slippy_rel = build_satellite_slippy_map(
        center_lat=c_lat_r,
        center_lon=c_lon_r,
        zoom=None,
        basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
        overlay_layers=[
            {
                "name": f"⛰️ {r_lbl}",
                "image": r_uri,
                "bounds": r_bnds,
                "opacity": float(r_opacity) / 100.0,
                "show": True,
                "zindex": 18,
            }
        ],
        aoi_bbox_4326=r_bbox_4326,
        aoi_polygon_geojson=rel_data.get("polygon_4326"),
        aoi_bbox_3794=rel_data.get("bbox_metric"),
        aoi_outline_label=t("slippy_aoi_outline"),
        transect_latlon=transect_ll,
        lang=lang,
    )
    st_folium(
        slippy_rel,
        height=470,
        use_container_width=True,
        returned_objects=[],
        key=f"{key_prefix}_slippy_{r_layer_key}_{r_opacity}_{pr_preset}_{st.session_state.get('map_basemap', 'satellite')}",
    )

    # GeoTIFF + Pošlji v Izdelavo karte
    r_tif_bytes = export_to_geotiff_bytes(active_r_da, target_epsg=r_crs)
    r_epsg_tag = r_crs.replace("EPSG:", "EPSG")
    r_dl1, r_dl2 = st.columns(2)
    with r_dl1:
        st.download_button(
            label=t("relief_btn_dl_tif"),
            data=r_tif_bytes,
            file_name=f"CopernicusDEM_{r_layer_key}_{r_epsg_tag}_30m.tif",
            mime="image/tiff",
            width="stretch",
            key=f"{key_prefix}_dl_tif_{r_layer_key}",
        )
    with r_dl2:
        if st.button(
            t("relief_btn_send_carto"),
            width="stretch",
            key=f"{key_prefix}_send_carto",
        ):
            st.session_state["tab4_pending_source_select"] = "tab2_relief"
            st.session_state["tab2_relief_sent_to_carto"] = True
            st.session_state["pending_workspace"] = "ws_geo"
            st.rerun(scope="app")

    if st.session_state.get("tab2_relief_sent_to_carto"):
        st.success(t("relief_sent_carto_ok"))

    # 5. Dual-axis Plotly vzdolžni prerez (Elevation + opcijski aktivni indeks iz Zavihka 2)
    ov_da = st.session_state.get("tab2_index_da")
    ov_meta = st.session_state.get("tab2_index_meta")
    ov_available = (ov_da is not None) and isinstance(ov_meta, dict)
    use_ov = False
    ov_label = ""
    if ov_available:
        ov_k = str(ov_meta.get("index_key", "Index"))
        ov_unit = " [°C]" if ov_k == "LST" else ""
        ov_label = f"{ov_k}{ov_unit} ({ov_meta.get('date_str', '')})"
        use_ov = st.checkbox(
            t("profile_overlay_chk", name=ov_label),
            value=True,
            key=f"{key_prefix}_use_overlay",
        )

    if np.hypot(pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]) <= 1.0:
        st.warning(t("profile_same_pts_warn"))
    else:
        prof_df, prof_sum = extract_transect_profile(
            dem_elev_da,
            pt_a,
            pt_b,
            n_samples=150,
            overlay_da=(ov_da if use_ov else None),
            overlay_name=(ov_label if use_ov else None),
        )
        pk1, pk2, pk3, pk4 = st.columns(4)
        pk1.metric(
            t("profile_kpi_len"),
            f"{prof_sum['total_length_km']:.2f} km",
            delta=f"{prof_sum['total_length_m']:,.0f} m",
            delta_color="off",
        )
        pk2.metric(t("profile_kpi_ascent"), f"+{prof_sum['total_ascent_m']:,.0f} m")
        pk3.metric(t("profile_kpi_descent"), f"−{prof_sum['total_descent_m']:,.0f} m")
        pk4.metric(t("profile_kpi_slope"), f"{prof_sum['mean_slope_pct']:.1f} %")

        fig_prof = make_subplots(specs=[[{"secondary_y": True}]])
        fig_prof.add_trace(
            go.Scatter(
                x=prof_df["distance_m"],
                y=prof_df["elevation_m"],
                mode="lines",
                name=t("profile_yaxis_elev"),
                line=dict(color="#0F766E", width=2.4),
                fill="tozeroy",
                fillcolor="rgba(15, 118, 110, 0.18)",
                hovertemplate="d = %{x:,.0f} m<br>H = %{y:,.1f} m<extra></extra>",
            ),
            secondary_y=False,
        )
        if prof_sum.get("has_overlay") and "overlay_value" in prof_df.columns:
            fig_prof.add_trace(
                go.Scatter(
                    x=prof_df["distance_m"],
                    y=prof_df["overlay_value"],
                    mode="lines",
                    name=str(prof_sum["overlay_name"]),
                    line=dict(color="#EA580C", width=2.2),
                    hovertemplate="d = %{x:,.0f} m<br>" + str(prof_sum["overlay_name"]) + " = %{y:.3f}<extra></extra>",
                ),
                secondary_y=True,
            )
            fig_prof.update_yaxes(
                title_text=str(prof_sum["overlay_name"]),
                secondary_y=True,
                showgrid=False,
                color="#EA580C",
            )
        fig_prof.update_xaxes(title_text=t("profile_xaxis"), gridcolor="#E2E8F0")
        fig_prof.update_yaxes(
            title_text=t("profile_yaxis_elev"),
            secondary_y=False,
            gridcolor="#E2E8F0",
            color="#0F766E",
        )
        fig_prof.update_layout(
            title=t("profile_chart_title", crs=format_crs_label(r_crs, short=True)),
            height=410,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#0F172A"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=55, r=55, t=55, b=45),
        )
        st.plotly_chart(fig_prof, width="stretch", key=f"{key_prefix}_profile_plotly")

        st.download_button(
            label=t("profile_btn_dl_csv"),
            data=prof_df.to_csv(index=False).encode("utf-8"),
            file_name=f"profile_A_to_B_{r_epsg_tag}.csv",
            mime="text/csv",
            width="stretch",
            key=f"{key_prefix}_profile_dl_csv",
        )


# ---------------------------------------------------------------------------
# Hitri @st.fragment interaktivni prikazovalniki (brez ponovnega zagona app.py)
# ---------------------------------------------------------------------------
@st.fragment
def _render_tab1_scene_viewer_fragment(
    *,
    active_sensor: str,
    active_bbox: list[float] | None,
    active_polygon: dict[str, Any] | None,
    bbox_3794: list[float] | None,
    center_lat: float,
    center_lon: float,
    lang: str,
) -> None:
    """Fragment za pregledovalnik posnetka (Slippy Map / Plotly, prosojnost, razteg, gama)."""
    ds_show: xr.Dataset | None = st.session_state.get("loaded_scene_ds")
    _raw_disp = st.session_state.get("loaded_scene_display_ds")
    ds_disp_show: xr.Dataset | None = _raw_disp if _raw_disp is not None else ds_show
    meta: dict[str, Any] | None = st.session_state.get("loaded_scene_meta")

    if ds_show is None or meta is None or meta.get("sensor") != active_sensor:
        return

    st.divider()
    first_band = meta["bands"][0]
    h_px, w_px = ds_show[first_band].shape

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(t("m_date"), meta["date_str"])
    m2.metric(t("m_crs"), meta["crs"])
    m3.metric(t("m_res"), f"{meta['resolution_m']} m")
    m4.metric(t("m_dims"), f"{w_px} × {h_px} px")
    m5.metric(t("m_area"), f"{meta['area_km2']:.2f} km²")

    is_sar_active = active_sensor == SENSOR_S1
    gamma_disabled = is_sar_active and meta["vis_code"] in ("VV", "VH")

    with st.container(border=True):
        c_vm, c_op, c_st, c_gm = st.columns([1.35, 1.0, 0.85, 1.0], vertical_alignment="bottom")
        with c_vm:
            t1_view_mode = _touch_selector(
                t("viewer_mode_label"),
                options=VIEWER_MODE_KEYS,
                default="slippy",
                format_func=lambda k, _lang=lang: t(VIEWER_MODE_I18N[k], lang=_lang),
                key="tab1_viewer_mode",
            )
        with c_op:
            t1_opacity_pct = st.slider(
                t("opacity_layer_label"),
                min_value=0,
                max_value=100,
                value=100,
                step=5,
                disabled=(t1_view_mode != "slippy"),
                key="tab1_overlay_opacity",
            )
        with c_st:
            use_percentile = st.checkbox(
                t("stretch_label"),
                value=True,
                help=t("stretch_help"),
                key="tab1_stretch",
            )
        with c_gm:
            gamma_val = st.slider(
                t("gamma_label"),
                min_value=0.5,
                max_value=2.2,
                value=1.0,
                step=0.05,
                disabled=gamma_disabled,
                key="tab1_gamma",
            )

    if t1_view_mode == "slippy" and ds_disp_show is not None:
        st.caption(t("slippy_hint"))
        if active_sensor == SENSOR_S1 and meta["vis_code"] in ("VV", "VH"):
            s_spec = SAR_METRICS[meta["vis_code"]]
            st.markdown(
                build_horizontal_colorbar_html(
                    cmap_name=str(s_spec["cmap"]),
                    vmin=float(s_spec["vmin"]),
                    vmax=float(s_spec["vmax"]),
                    label=f"Sentinel-1 {meta['vis_code']}",
                    units="dB",
                ),
                unsafe_allow_html=True,
            )

        png_uri_t1, bounds_t1, layer_name_t1 = _get_tab1_overlay_entry(
            ds_disp=ds_disp_show,
            meta=meta,
            use_percentile=use_percentile,
            gamma=gamma_val,
            lang=lang,
        )
        c_lat_t1 = 0.5 * (active_bbox[1] + active_bbox[3]) if active_bbox is not None else center_lat
        c_lon_t1 = 0.5 * (active_bbox[0] + active_bbox[2]) if active_bbox is not None else center_lon
        slippy_t1 = build_satellite_slippy_map(
            center_lat=c_lat_t1,
            center_lon=c_lon_t1,
            zoom=None,
            basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
            overlay_layers=[
                {
                    "name": layer_name_t1,
                    "image": png_uri_t1,
                    "bounds": bounds_t1,
                    "opacity": float(t1_opacity_pct) / 100.0,
                    "show": True,
                    "zindex": 15,
                }
            ],
            aoi_bbox_4326=active_bbox,
            aoi_polygon_geojson=active_polygon,
            aoi_bbox_3794=st.session_state.get("search_bbox_3794") or bbox_3794,
            aoi_outline_label=t("slippy_aoi_outline"),
            lang=lang,
        )
        st_folium(
            slippy_t1,
            height=470,
            use_container_width=True,
            returned_objects=[],
            key=(
                f"tab1_slippy_{meta['scene_id']}_{meta['vis_code']}_"
                f"{t1_opacity_pct}_{use_percentile}_{gamma_val:.2f}_"
                f"{st.session_state.get('map_basemap', 'satellite')}"
            ),
        )
        st.caption(t("geodetic_separation_caption", crs=meta["crs"]))
    else:
        st.caption(t("viewer_zoom_hint"))
        if active_sensor == SENSOR_S1:
            vis_lbl = t(SAR_VIS_SPECS[meta["vis_code"]]["label_key"])
            fig_scene = _build_sar_scene_plotly(
                ds=ds_show,
                vis_code=meta["vis_code"],
                vis_label=vis_lbl,
                target_crs=meta["crs"],
                use_percentile_stretch=use_percentile,
                gamma=gamma_val,
                tile_lbl=meta["mgrs_tile"],
            )
        else:
            vis_lbl = t(OPTICAL_COMPOSITE_SPECS[meta["vis_code"]]["label_key"])
            fig_scene = _build_optical_rgb_plotly(
                ds=ds_show,
                bands=meta["bands"],
                composite_label=vis_lbl,
                target_crs=meta["crs"],
                use_percentile_stretch=use_percentile,
                gamma=gamma_val,
                sensor_name=meta["sensor_name"],
                tile_lbl=meta["mgrs_tile"],
            )

        st.plotly_chart(
            fig_scene,
            width="stretch",
            config=PLOTLY_VIEWER_CONFIG,
            key="tab1_plotly_raster_viewer",
        )

    cloud_disp = "— (SAR)" if meta["cloud_cover"] is None else f"{meta['cloud_cover']:.1f} %"
    if active_sensor == SENSOR_S2:
        pb_val = ds_show.attrs.get("processing_baseline", "—")
        off_yn = t("yes") if ds_show.attrs.get("radiometric_offset_applied") else t("no")
        cal_str = f"ESA PB {pb_val} (offset -1000: {off_yn})"
    elif active_sensor == SENSOR_LANDSAT:
        cal_str = "USGS C2 L2 (DN × 0.0000275 − 0.2)"
    else:
        cal_str = "Sentinel-1 RTC (10·log10 dB)"

    st.caption(
        t(
            "caption_radiometry",
            scene_id=meta["scene_id"],
            platform=meta["platform"],
            cloud=cloud_disp,
            pb=cal_str,
        )
    )

    tif_bytes = st.session_state.get("loaded_tif_bytes")
    if tif_bytes:
        st.download_button(
            label=t("btn_download_rgb_tif"),
            data=tif_bytes,
            file_name=st.session_state.get("loaded_tif_filename") or "scene_clip.tif",
            mime="image/tiff",
            width="stretch",
            help=t("help_download_rgb_tif", bands=", ".join(meta["bands"]), crs=meta["crs"]),
        )


@st.fragment
def _render_tab2_sar_viewer_fragment(
    *,
    active_bbox: list[float] | None,
    active_polygon: dict[str, Any] | None,
    bbox_3794: list[float] | None,
    center_lat: float,
    center_lon: float,
    lang: str,
) -> None:
    """Fragment za dinamični prag SAR maske voda/poplav in prosojnost Slippy Map."""
    ds_sar_loaded: xr.Dataset | None = st.session_state.get("tab2_sar_ds")
    _raw_sar_disp = st.session_state.get("tab2_sar_display_ds")
    ds_sar_disp: xr.Dataset | None = _raw_sar_disp if _raw_sar_disp is not None else ds_sar_loaded
    sar_meta: dict[str, Any] | None = st.session_state.get("tab2_sar_meta")
    if ds_sar_loaded is None or sar_meta is None:
        return

    st.divider()
    with st.container(border=True):
        c_thr, c_vm, c_op = st.columns([1.2, 1.3, 1.0], vertical_alignment="bottom")
        with c_thr:
            sar_thr_db = st.slider(
                t("sar_water_thr_label"),
                min_value=-25.0,
                max_value=-8.0,
                value=-16.0,
                step=0.5,
                help=t("sar_water_thr_help"),
                key="tab2_sar_water_thr",
            )
        with c_vm:
            t2_sar_view_mode = _touch_selector(
                t("viewer_mode_label"),
                options=VIEWER_MODE_KEYS,
                default="slippy",
                format_func=lambda k, _lang=lang: t(VIEWER_MODE_I18N[k], lang=_lang),
                key="tab2_sar_viewer_mode",
            )
        with c_op:
            t2_sar_opacity_pct = st.slider(
                t("opacity_index_label"),
                min_value=0,
                max_value=100,
                value=85,
                step=5,
                disabled=(t2_sar_view_mode != "slippy"),
                key="tab2_sar_overlay_opacity",
            )

    mask_da, water_stats = compute_sar_water_mask(
        ds_sar_loaded,
        threshold_db=float(sar_thr_db),
        polarization="VV",
        resolution_m=float(sar_meta["resolution_m"]),
    )
    vv_stats = compute_index_stats(ds_sar_loaded["VV"], resolution_m=float(sar_meta["resolution_m"]))

    if water_stats["valid_pixels"] == 0:
        st.warning(t("t2_no_valid_pixels"))
        return

    w1, w2, w3, w4 = st.columns(4)
    w1.metric(t("m_water_ha"), f"{water_stats['water_area_ha']:,.2f} ha")
    w2.metric(t("m_water_pct"), f"{water_stats['water_pct']:.2f} %")
    w3.metric(t("m_sar_vv_mean"), f"{vv_stats['mean']:.2f} dB")
    w4.metric(t("m_valid_ha"), f"{water_stats['valid_area_ha']:,.2f} ha")

    if t2_sar_view_mode == "slippy" and ds_sar_disp is not None:
        mask_disp_da, _ = compute_sar_water_mask(
            ds_sar_disp,
            threshold_db=float(sar_thr_db),
            polarization="VV",
            resolution_m=float(sar_meta["resolution_m"]),
        )
        st.caption(t("slippy_hint"))
        st.markdown(
            build_horizontal_colorbar_html(
                cmap_name="Greys",
                vmin=0.0,
                vmax=1.0,
                label=t("sar_water_map_title", thr=sar_thr_db, date=sar_meta["date_str"]),
                discrete_items=[
                    (t("sar_water_legend_water"), "#0369A1"),
                    (t("sar_water_legend_land"), "#CBD5E1"),
                ],
            ),
            unsafe_allow_html=True,
        )
        w_uri, w_bounds = raster_to_map_overlay_png(
            mask_disp_da,
            src_epsg=sar_meta["crs"],
            is_rgb=False,
            discrete_rgba_map=SAR_WATER_DISCRETE_RGBA,
        )
        c_lat_s1 = 0.5 * (active_bbox[1] + active_bbox[3]) if active_bbox is not None else center_lat
        c_lon_s1 = 0.5 * (active_bbox[0] + active_bbox[2]) if active_bbox is not None else center_lon
        slippy_s1 = build_satellite_slippy_map(
            center_lat=c_lat_s1,
            center_lon=c_lon_s1,
            zoom=None,
            basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
            overlay_layers=[
                {
                    "name": f"🌊 SAR Water Mask ({sar_meta['date_str']}, VV < {sar_thr_db:.1f} dB)",
                    "image": w_uri,
                    "bounds": w_bounds,
                    "opacity": float(t2_sar_opacity_pct) / 100.0,
                    "show": True,
                    "zindex": 20,
                }
            ],
            aoi_bbox_4326=active_bbox,
            aoi_polygon_geojson=active_polygon,
            aoi_bbox_3794=st.session_state.get("search_bbox_3794") or bbox_3794,
            aoi_outline_label=t("slippy_aoi_outline"),
            lang=lang,
        )
        st_folium(
            slippy_s1,
            height=470,
            use_container_width=True,
            returned_objects=[],
            key=(
                f"tab2_sar_slippy_{sar_meta['scene_id']}_{sar_thr_db:.1f}_"
                f"{t2_sar_opacity_pct}_{st.session_state.get('map_basemap', 'satellite')}"
            ),
        )
        st.caption(t("geodetic_separation_caption", crs=sar_meta["crs"]))
    else:
        st.caption(t("viewer_zoom_hint"))
        fig_water = _build_sar_water_plotly(
            mask_da=mask_da,
            threshold_db=float(sar_thr_db),
            target_crs=sar_meta["crs"],
            date_str=sar_meta["date_str"],
            tile_lbl=sar_meta["mgrs_tile"],
            resolution_m=float(sar_meta["resolution_m"]),
        )
        st.plotly_chart(
            fig_water,
            width="stretch",
            config=PLOTLY_VIEWER_CONFIG,
            key="tab2_sar_water_plotly_viewer",
        )

    hist_vv = _build_sar_vv_histogram(
        vv_da=ds_sar_loaded["VV"],
        threshold_db=float(sar_thr_db),
        date_str=sar_meta["date_str"],
    )
    st.plotly_chart(hist_vv, width="stretch", key="tab2_sar_vv_hist")

    water_tif_bytes = export_to_geotiff_bytes(mask_da, target_epsg=sar_meta["crs"])
    epsg_num = sar_meta["crs"].replace("EPSG:", "")
    thr_tag = f"{sar_thr_db:.1f}".replace("-", "m").replace(".", "p")
    water_fname = (
        f"S1RTC_{sar_meta['date_str']}_WaterMask_VV_{thr_tag}dB_"
        f"EPSG{epsg_num}_{sar_meta['resolution_m']}m.tif"
    )
    st.download_button(
        label=t("btn_download_sar_water_tif"),
        data=water_tif_bytes,
        file_name=water_fname,
        mime="image/tiff",
        width="stretch",
    )


@st.fragment
def _render_tab2_index_viewer_fragment(
    *,
    active_sensor: str,
    active_bbox: list[float] | None,
    active_polygon: dict[str, Any] | None,
    bbox_3794: list[float] | None,
    center_lat: float,
    center_lon: float,
    lang: str,
) -> None:
    """Fragment za prikaz spektralnega indeksa / NDSI snega / LST °C z nastavitvijo prosojnosti."""
    idx_da: xr.DataArray | None = st.session_state.get("tab2_index_da")
    _raw_idx_disp = st.session_state.get("tab2_index_display_da")
    idx_disp_da: xr.DataArray | None = _raw_idx_disp if _raw_idx_disp is not None else idx_da
    idx_stats: dict[str, Any] | None = st.session_state.get("tab2_index_stats")
    idx_meta: dict[str, Any] | None = st.session_state.get("tab2_index_meta")

    if (
        idx_da is None
        or idx_stats is None
        or idx_meta is None
        or idx_meta.get("sensor") != active_sensor
    ):
        return

    st.divider()
    if idx_stats["valid_pixels"] == 0:
        st.warning(t("t2_no_valid_pixels"))
        return

    sn_stats_show = st.session_state.get("tab2_snow_stats")
    sn_disp_show = st.session_state.get("tab2_snow_display_da")
    if idx_meta.get("index_key") == "NDSI" and isinstance(sn_stats_show, dict):
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            t("snow_kpi_area"),
            f"{sn_stats_show['snow_area_km2']:,.2f} km²",
            delta=f"{sn_stats_show['snow_area_ha']:,.1f} ha",
            delta_color="off",
        )
        s2.metric(
            t("snow_kpi_share"),
            f"{sn_stats_show['snow_pct']:.1f} %",
            delta=format_crs_label(idx_meta["crs"], short=True),
            delta_color="off",
        )
        if (
            sn_stats_show.get("dem_available")
            and sn_stats_show.get("snowline_p10_m") is not None
        ):
            s3.metric(
                t("snow_kpi_snowline"),
                f"{sn_stats_show['snowline_p10_m']:,.0f} m n.m.",
            )
            s4.metric(
                t("snow_kpi_elev_range"),
                f"{sn_stats_show['snow_median_elev_m']:,.0f} / {sn_stats_show['snow_max_elev_m']:,.0f} m",
            )
        else:
            s3.metric(t("snow_kpi_snowline"), t("snow_kpi_snowline_na"))
            s4.metric(t("m_valid_ha"), f"{idx_stats['valid_area_ha']:,.1f} ha")
        st.caption(t("snow_depth_scientific_note"))

    if idx_meta.get("index_key") == "LST":
        lst_range = float(idx_stats["max"]) - float(idx_stats["min"])
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(t("lst_kpi_min"), f"{idx_stats['min']:.1f} °C")
        k2.metric(t("lst_kpi_mean"), f"{idx_stats['mean']:.1f} °C")
        k3.metric(t("lst_kpi_max"), f"{idx_stats['max']:.1f} °C")
        k4.metric(t("lst_kpi_range"), f"{lst_range:.1f} °C")
        k5.metric(t("m_valid_ha"), f"{idx_stats['valid_area_ha']:,.2f} ha")
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(t("m_mean"), f"{idx_stats['mean']:.3f}")
        k2.metric(t("m_median"), f"{idx_stats['median']:.3f}")
        k3.metric(t("m_std"), f"{idx_stats['std']:.3f}")
        k4.metric(t("m_valid_pct"), f"{idx_stats['valid_pct']:.1f} %")
        k5.metric(t("m_valid_ha"), f"{idx_stats['valid_area_ha']:,.2f} ha")

    col_vm2, col_op2 = st.columns([1.65, 1.0], vertical_alignment="bottom")
    with col_vm2:
        t2_view_mode = _touch_selector(
            t("viewer_mode_label"),
            options=VIEWER_MODE_KEYS,
            default="slippy",
            format_func=lambda k, _lang=lang: t(VIEWER_MODE_I18N[k], lang=_lang),
            key="tab2_index_viewer_mode",
        )
    with col_op2:
        t2_opacity_pct = st.slider(
            t("opacity_index_label"),
            min_value=0,
            max_value=100,
            value=85,
            step=5,
            disabled=(t2_view_mode != "slippy"),
            key="tab2_index_overlay_opacity",
        )

    if t2_view_mode == "slippy" and idx_disp_da is not None:
        idx_key_s = str(idx_meta["index_key"])
        idx_spec = SPECTRAL_INDICES[idx_key_s]
        st.caption(t("slippy_hint"))
        st.markdown(
            build_horizontal_colorbar_html(
                cmap_name=str(idx_spec["cmap"]),
                vmin=float(idx_spec["vmin"]),
                vmax=float(idx_spec["vmax"]),
                label=f"{idx_meta['sensor_name']} — {idx_key_s} ({idx_meta['date_str']})",
                units=str(idx_spec.get("units", "")),
            ),
            unsafe_allow_html=True,
        )

        idx_cache_sig = (
            idx_meta.get("scene_id", idx_meta["date_str"]),
            idx_key_s,
            idx_meta["crs"],
            idx_meta["resolution_m"],
            bool(idx_meta.get("mask_clouds", True)),
            bool(idx_meta.get("expanded_pan", False)),
        )
        idx_cached = st.session_state.get("tab2_index_overlay_cache")
        if isinstance(idx_cached, dict) and idx_cached.get("sig") == idx_cache_sig:
            idx_uri, idx_bounds = idx_cached["png_uri"], idx_cached["bounds"]
        else:
            idx_uri, idx_bounds = raster_to_map_overlay_png(
                idx_disp_da,
                src_epsg=idx_meta["crs"],
                is_rgb=False,
                cmap_name=str(idx_spec["cmap"]),
                vmin=float(idx_spec["vmin"]),
                vmax=float(idx_spec["vmax"]),
            )
            st.session_state["tab2_index_overlay_cache"] = {
                "sig": idx_cache_sig,
                "png_uri": idx_uri,
                "bounds": idx_bounds,
            }

        t2_overlays: list[dict[str, Any]] = []
        _raw_t1_disp = st.session_state.get("loaded_scene_display_ds")
        t1_disp = _raw_t1_disp if _raw_t1_disp is not None else st.session_state.get("loaded_scene_ds")
        t1_meta = st.session_state.get("loaded_scene_meta")
        if t1_disp is not None and t1_meta is not None and t1_meta.get("sensor") == active_sensor:
            try:
                t1_uri, t1_bnds, t1_lbl = _get_tab1_overlay_entry(
                    ds_disp=t1_disp,
                    meta=t1_meta,
                    use_percentile=bool(st.session_state.get("tab1_stretch", True)),
                    gamma=float(st.session_state.get("tab1_gamma", 1.0)),
                    lang=lang,
                )
                t2_overlays.append(
                    {
                        "name": t1_lbl,
                        "image": t1_uri,
                        "bounds": t1_bnds,
                        "opacity": 1.0,
                        "show": False,
                        "zindex": 12,
                    }
                )
            except Exception:
                pass

        t2_overlays.append(
            {
                "name": f"📊 {idx_key_s} ({idx_meta['date_str']})",
                "image": idx_uri,
                "bounds": idx_bounds,
                "opacity": float(t2_opacity_pct) / 100.0,
                "show": not (idx_key_s == "NDSI" and sn_disp_show is not None),
                "zindex": 18,
            }
        )

        if idx_key_s == "NDSI" and sn_disp_show is not None and isinstance(sn_stats_show, dict):
            try:
                sn_uri, sn_bnds = raster_to_map_overlay_png(
                    sn_disp_show,
                    src_epsg=idx_meta["crs"],
                    is_rgb=False,
                    discrete_rgba_map=SNOW_MASK_DISCRETE_RGBA,
                )
                t2_overlays.append(
                    {
                        "name": t("snow_mask_layer_name", thr=float(sn_stats_show.get("ndsi_thresh", 0.40))),
                        "image": sn_uri,
                        "bounds": sn_bnds,
                        "opacity": float(t2_opacity_pct) / 100.0,
                        "show": True,
                        "zindex": 22,
                    }
                )
            except Exception:
                pass

        c_lat_t2 = 0.5 * (active_bbox[1] + active_bbox[3]) if active_bbox is not None else center_lat
        c_lon_t2 = 0.5 * (active_bbox[0] + active_bbox[2]) if active_bbox is not None else center_lon
        slippy_t2 = build_satellite_slippy_map(
            center_lat=c_lat_t2,
            center_lon=c_lon_t2,
            zoom=None,
            basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
            overlay_layers=t2_overlays,
            aoi_bbox_4326=active_bbox,
            aoi_polygon_geojson=active_polygon,
            aoi_bbox_3794=st.session_state.get("search_bbox_3794") or bbox_3794,
            aoi_outline_label=t("slippy_aoi_outline"),
            lang=lang,
        )
        st_folium(
            slippy_t2,
            height=470,
            use_container_width=True,
            returned_objects=[],
            key=(
                f"tab2_idx_slippy_{idx_meta.get('scene_id', idx_meta['date_str'])}_"
                f"{idx_key_s}_{t2_opacity_pct}_{len(t2_overlays)}_"
                f"{st.session_state.get('map_basemap', 'satellite')}"
            ),
        )
        st.caption(t("geodetic_separation_caption", crs=idx_meta["crs"]))
    else:
        st.caption(t("viewer_zoom_hint"))
        fig_idx = _build_index_plotly(
            index_da=idx_da,
            index_key=idx_meta["index_key"],
            target_crs=idx_meta["crs"],
            date_str=idx_meta["date_str"],
            tile_lbl=idx_meta["mgrs_tile"],
            resolution_m=float(idx_meta["resolution_m"]),
            sensor_name=idx_meta["sensor_name"],
        )
        st.plotly_chart(
            fig_idx,
            width="stretch",
            config=PLOTLY_VIEWER_CONFIG,
            key="tab2_index_plotly_viewer",
        )

    fig_hist = _build_index_histogram(
        index_da=idx_da,
        index_key=idx_meta["index_key"],
        date_str=idx_meta["date_str"],
    )
    st.plotly_chart(fig_hist, width="stretch", key="tab2_index_hist")

    idx_tif_bytes = st.session_state.get("tab2_tif_bytes")
    if idx_tif_bytes:
        st.download_button(
            label=t("btn_download_index_tif", idx=idx_meta["index_key"]),
            data=idx_tif_bytes,
            file_name=st.session_state.get("tab2_tif_filename") or f"{idx_meta['index_key']}.tif",
            mime="image/tiff",
            width="stretch",
            help=t("help_download_index_tif", idx=idx_meta["index_key"], crs=idx_meta["crs"]),
        )
    sn_tif_bytes = st.session_state.get("tab2_snow_tif_bytes")
    if idx_meta.get("index_key") == "NDSI" and sn_tif_bytes:
        st.download_button(
            label=t("btn_download_snow_tif"),
            data=sn_tif_bytes,
            file_name=st.session_state.get("tab2_snow_tif_filename") or "snow_mask_NDSI.tif",
            mime="image/tiff",
            width="stretch",
            key="tab2_dl_snow_mask_tif",
        )


@st.fragment
def _render_comp3_swipe_fragment(
    *,
    active_sensor: str,
    ch_threshold: float,
    lang: str,
    active_bbox: Sequence[float] | None = None,
    active_polygon: dict[str, Any] | None = None,
    bbox_3794: Sequence[float] | None = None,
    center_lat: float = 46.0569,
    center_lon: float = 14.5058,
) -> None:
    """Fragment za interaktivno 2–3 datumsko primerjavo (zoomabilna zavesa, pas in Δ karta)."""
    comp_show: dict[str, Any] | None = st.session_state.get("tab3_comp_result")
    if not (
        isinstance(comp_show, dict)
        and comp_show.get("sensor") == active_sensor
        and len(comp_show.get("swipe_pngs") or []) >= 2
    ):
        return

    c_dates: list[str] = list(comp_show["dates"])
    c_swipes: list[bytes] = list(comp_show["swipe_pngs"])
    c_strips: list[bytes] = list(comp_show["strip_pngs"])
    c_overlay_uris: list[str] = list(comp_show.get("overlay_uris") or [])
    c_overlay_bounds: list[list[list[float]]] = list(comp_show.get("overlay_bounds") or [])
    c_replacements: list[dict[str, Any]] = list(comp_show.get("replacements") or [])
    c_layer = str(comp_show["layer_mode"])
    c_is_sar = comp_show.get("sensor") == SENSOR_S1
    c_is_lst = (not c_is_sar) and c_layer == "LST"
    unit_b = " [dB]" if c_is_sar else (" [°C]" if c_is_lst else "")
    idx_lbl = f"{c_layer}{unit_b}"

    # Prikaz jasnih obvestil o samodejni zamenjavi oblačnih posnetkov nad AOI
    for rep in c_replacements:
        st.info(
            t(
                "comp3_replaced_badge",
                req=str(rep.get("requested_date", "")),
                orig_pct=float(rep.get("original_clear_pct", 0.0)),
                act=str(rep.get("actual_date", "")),
                act_pct=float(rep.get("actual_clear_pct", 0.0)),
            )
        )

    st.markdown(f"#### {t('swipe_header')}")
    has_zoom_swipe = (
        len(c_overlay_uris) == len(c_dates)
        and len(c_overlay_bounds) == len(c_dates)
        and all(bool(u) for u in c_overlay_uris)
    )
    st.caption(t("swipe_zoom_hint") if has_zoom_swipe else t("swipe_hint"))

    if len(c_dates) == 3:
        pair_map = {"0_1": (0, 1), "1_2": (1, 2), "0_2": (0, 2)}
        pair_choice = _touch_selector(
            t("comp3_swipe_pair_label"),
            options=["0_1", "1_2", "0_2"],
            default="0_1",
            format_func=lambda pk: (
                f"t{pair_map[pk][0] + 1} ({c_dates[pair_map[pk][0]]}) ↔ "
                f"t{pair_map[pk][1] + 1} ({c_dates[pair_map[pk][1]]})"
            ),
            key=f"tab3_comp3_pair_{active_sensor}",
            use_pills=True,
        )
        i_a, i_b = pair_map[pair_choice]
    else:
        i_a, i_b = 0, 1

    if has_zoom_swipe:
        swipe_opacity_pct = st.slider(
            t("opacity_layer_label"),
            min_value=20,
            max_value=100,
            value=100,
            step=5,
            format="%d %%",
            key=f"tab3_comp3_opacity_{active_sensor}",
        )
        comp_bbox = comp_show.get("bbox_4326") or active_bbox
        c_lat_s = (
            0.5 * (float(comp_bbox[1]) + float(comp_bbox[3]))
            if (comp_bbox is not None and len(comp_bbox) == 4)
            else float(center_lat)
        )
        c_lon_s = (
            0.5 * (float(comp_bbox[0]) + float(comp_bbox[2]))
            if (comp_bbox is not None and len(comp_bbox) == 4)
            else float(center_lon)
        )
        swipe_html = build_zoomable_swipe_map(
            overlay_uri_t1=c_overlay_uris[i_a],
            bounds_t1=c_overlay_bounds[i_a],
            overlay_uri_t2=c_overlay_uris[i_b],
            bounds_t2=c_overlay_bounds[i_b],
            label_t1=f"t{i_a + 1}: {c_dates[i_a]} ({idx_lbl})",
            label_t2=f"t{i_b + 1}: {c_dates[i_b]} ({idx_lbl})",
            center_lat=c_lat_s,
            center_lon=c_lon_s,
            basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
            aoi_bbox_4326=comp_bbox,
            aoi_polygon_geojson=active_polygon,
            aoi_bbox_3794=bbox_3794,
            aoi_outline_label=t("slippy_aoi_outline"),
            height=470,
            opacity=float(swipe_opacity_pct) / 100.0,
            lang=lang,
        )
        components.html(swipe_html, height=480, scrolling=False)
    else:
        _render_before_after_swipe_html(
            png_t1_bytes=c_swipes[i_a],
            png_t2_bytes=c_swipes[i_b],
            label_t1=f"t{i_a + 1}: {c_dates[i_a]} ({idx_lbl})",
            label_t2=f"t{i_b + 1}: {c_dates[i_b]} ({idx_lbl})",
            height=520,
        )

    st.markdown(f"#### {t('comp3_strip_header', n=len(c_dates))}")
    strip_cols = st.columns(len(c_dates))
    for s_idx, col_st in enumerate(strip_cols):
        with col_st:
            st.image(c_strips[s_idx], width="stretch")
            st.caption(f"**t{s_idx + 1}:** `{c_dates[s_idx]}` · {idx_lbl}")

    da_t1_s = comp_show.get("da_first")
    da_t2_s = comp_show.get("da_last")
    da_diff_s = comp_show.get("diff_da")
    if da_t1_s is not None and da_t2_s is not None and da_diff_s is not None:
        st.markdown(f"#### {t('comp3_delta_header', d_first=c_dates[0], d_last=c_dates[-1])}")
        summary = compute_change_summary(
            da_diff_s,
            resolution_m=float(comp_show["resolution_m"]),
            threshold=float(ch_threshold),
        )
        if summary["valid_pixels"] == 0:
            st.warning(t("t2_no_valid_pixels"))
        else:
            mean_str = (
                f"{summary['mean']:+.2f} dB"
                if c_is_sar
                else (f"{summary['mean']:+.2f} °C" if c_is_lst else f"{summary['mean']:+.3f}")
            )
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(t("change_m_mean"), mean_str)
            c2.metric(
                t("change_m_pos", thr=ch_threshold),
                f"{summary['pos_area_ha']:,.2f} ha",
                delta=f"{summary['pos_pct']:.1f} %",
            )
            c3.metric(
                t("change_m_neg", thr=ch_threshold),
                f"{summary['neg_area_ha']:,.2f} ha",
                delta=f"-{summary['neg_pct']:.1f} %",
                delta_color="inverse",
            )
            c4.metric(
                t("change_m_stable", thr=ch_threshold),
                f"{summary['stable_area_ha']:,.2f} ha",
                delta=f"{summary['stable_pct']:.1f} %",
                delta_color="off",
            )
            c5.metric(
                t("change_m_valid"),
                f"{summary['valid_area_ha']:,.2f} ha",
                delta=f"{summary['valid_pct']:.1f} %",
                delta_color="off",
            )

            layer_labels = {
                "diff": t("change_panel_diff", idx=idx_lbl),
                "t1": t("change_panel_t1", idx=idx_lbl, d1=c_dates[0]),
                "t2": t("change_panel_t2", idx=idx_lbl, d2=c_dates[-1]),
            }
            view_layer = st.radio(
                t("composite_select_label"),
                options=["diff", "t1", "t2"],
                index=0,
                horizontal=True,
                format_func=lambda k: layer_labels[k],
                label_visibility="collapsed",
                key="tab3_change_layer_radio",
            )

            st.caption(t("viewer_zoom_hint"))
            fig_ch = _build_change_plotly(
                da_t1=da_t1_s,
                da_t2=da_t2_s,
                da_diff=da_diff_s,
                index_key=c_layer,
                target_crs=str(comp_show["crs"]),
                date_t1=c_dates[0],
                date_t2=c_dates[-1],
                resolution_m=float(comp_show["resolution_m"]),
                is_sar=c_is_sar,
                view_layer=view_layer,
            )
            st.plotly_chart(
                fig_ch,
                width="stretch",
                config=PLOTLY_VIEWER_CONFIG,
                key="tab3_change_plotly_viewer",
            )

            diff_bytes = comp_show.get("tif_bytes")
            if diff_bytes:
                st.download_button(
                    label=t("btn_download_change_tif"),
                    data=diff_bytes,
                    file_name=comp_show.get("tif_filename") or "change_map.tif",
                    mime="image/tiff",
                    width="stretch",
                )


def _apply_carto_layout_to_widgets(overrides: dict[str, Any]) -> None:
    """Apply a ``MapLayoutConfig`` partial override dict onto Workspace 4 widget session keys before widget creation."""
    cfg = merge_layout_config(overrides)
    mapping = {
        "tab4_page_format": cfg["paper_format"],
        "tab4_theme": cfg["theme"],
        "tab4_chk_title": bool(cfg["show_title"]),
        "tab4_title_pos": cfg["title_position"],
        "tab4_chk_footer": bool(cfg["show_footer"]),
        "tab4_note_pos": cfg["custom_note_position"],
        "tab4_chk_grid": bool(cfg["show_grid"]),
        "tab4_grid_type": cfg["grid_style"],
        "tab4_grid_style": cfg["grid_color"],
        "tab4_grid_alpha": float(cfg["grid_alpha"]),
        "tab4_grid_spacing": 0 if cfg["grid_spacing"] == "auto" else int(cfg["grid_spacing"]),
        "tab4_chk_aoi": bool(cfg["show_aoi_outline"]),
        "tab4_aoi_color": cfg["aoi_outline_color"],
        "tab4_chk_north": bool(cfg["show_north_arrow"]),
        "tab4_north_pos": cfg["north_position"],
        "tab4_north_style": cfg["north_style"],
        "tab4_north_size": cfg["north_size"],
        "tab4_chk_scalebar": bool(cfg["show_scale_bar"]),
        "tab4_scale_pos": cfg["scale_position"],
        "tab4_scale_style": cfg["scale_style"],
        "tab4_chk_num_scale": bool(cfg["show_numeric_scale"]),
        "tab4_chk_legend": bool(cfg["show_legend"]),
        "tab4_legend_pos": cfg["legend_position"],
        "tab4_chk_leg_meta": bool(cfg["show_metadata_in_legend"]),
        "tab4_chk_inset": bool(cfg["show_inset_map"]),
        "tab4_inset_pos": cfg["inset_position"],
    }
    for wk, wv in mapping.items():
        if wk in overrides or True:
            st.session_state[wk] = wv
    if overrides.get("title"):
        st.session_state["tab4_title_input"] = str(overrides["title"])
    if overrides.get("subtitle") is not None and "subtitle" in overrides:
        st.session_state["tab4_subtitle_input"] = str(overrides["subtitle"])
    if overrides.get("author"):
        st.session_state["tab4_author_input"] = str(overrides["author"])
    if "custom_note_text" in overrides:
        st.session_state["tab4_note_input"] = str(overrides["custom_note_text"] or "")
    if "legend_title" in overrides and overrides["legend_title"]:
        st.session_state["tab4_leg_title_input"] = str(overrides["legend_title"])


@st.fragment
def _render_cartography_fragment(
    *,
    active_sensor: str,
    active_sensor_cfg: dict[str, Any],
    active_bbox: list[float] | None,
    active_polygon: dict[str, Any] | None,
    target_crs: str,
    selected_resolution: int,
    area_km2: float,
    search_performed: bool,
    items: list[Any],
    summary_df: pd.DataFrame,
    default_scene_idx: int,
    lang: str,
) -> None:
    """Fragment za QGIS-slog kartografski studio (300 DPI PDF/PNG/GeoTIFF) brez ponovnega zagona aplikacije."""
    st.subheader(t("carto_header"))
    st.caption(t("carto_intro"))

    carto_sources = _collect_carto_sources(lang)

    if not carto_sources:
        st.info(t("carto_no_layers_info"))
        if search_performed and items and not summary_df.empty:
            if st.button(
                t("carto_btn_quick_load"),
                type="primary",
                key="tab4_quick_load_btn",
            ):
                is_sar_ql = active_sensor == SENSOR_S1
                ql_vis = "SAR_FC" if is_sar_ql else "RGB"
                ql_bands = (
                    SAR_VIS_SPECS[ql_vis]["bands"]
                    if is_sar_ql
                    else OPTICAL_COMPOSITE_SPECS[ql_vis]["bands"]
                )
                ql_vars = (
                    SAR_VIS_SPECS[ql_vis]["var_names"]
                    if is_sar_ql
                    else OPTICAL_COMPOSITE_SPECS[ql_vis]["bands"]
                )
                ql_row = summary_df.iloc[default_scene_idx]
                ql_item = items[default_scene_idx]
                with st.spinner(t("carto_quick_load_spinner", crs=target_crs)):
                    try:
                        ds_ql = load_scene(
                            sensor=active_sensor,
                            item=ql_item,
                            bbox_4326=active_bbox,
                            bands=ql_bands,
                            target_epsg=target_crs,
                            resolution=int(selected_resolution),
                            aoi_polygon_4326=active_polygon,
                        )
                        ql_tif = export_to_geotiff_bytes(ds_ql[ql_vars], target_epsg=target_crs)
                        epsg_num_ql = target_crs.split(":")[-1]
                        ql_fname = (
                            f"{active_sensor_cfg['file_prefix']}_{ql_vis}_"
                            f"{ql_row['date_str']}_EPSG{epsg_num_ql}_{selected_resolution}m.tif"
                        )
                        st.session_state["loaded_scene_ds"] = ds_ql
                        st.session_state["loaded_scene_display_ds"] = ds_ql
                        st.session_state["loaded_scene_overlay_cache"] = None
                        st.session_state["loaded_scene_meta"] = {
                            "sensor": active_sensor,
                            "sensor_name": active_sensor_cfg["short_name"],
                            "date_str": str(ql_row["date_str"]),
                            "cloud_cover": (
                                None if pd.isna(ql_row["cloud_cover"]) else float(ql_row["cloud_cover"])
                            ),
                            "mgrs_tile": _format_tile_label(str(ql_row["mgrs_tile"]), sensor=active_sensor),
                            "platform": str(ql_row["platform"]),
                            "scene_id": str(ql_row["id"]),
                            "crs": target_crs,
                            "resolution_m": int(selected_resolution),
                            "bands": ql_vars,
                            "vis_code": ql_vis,
                            "area_km2": float(st.session_state["search_area_km2"] or area_km2),
                            "expanded_pan": False,
                        }
                        st.session_state["loaded_tif_bytes"] = ql_tif
                        st.session_state["loaded_tif_filename"] = ql_fname
                        gc.collect()
                        st.rerun()
                    except Exception as exc:
                        st.error(t("load_scene_error", err=exc))
        return

    # Apply any pending 1-click preset or AI layout overrides BEFORE instantiating widgets
    pending_preset = st.session_state.pop("tab4_pending_preset", None)
    if pending_preset in LAYOUT_PRESETS:
        cur_merged = dict(st.session_state.get("tab4_layout_config") or DEFAULT_LAYOUT_CONFIG)
        cur_merged.update(LAYOUT_PRESETS[pending_preset])
        _apply_carto_layout_to_widgets(cur_merged)

    pending_overrides = st.session_state.pop("tab4_pending_layout_overrides", None)
    if isinstance(pending_overrides, dict) and pending_overrides:
        cur_merged = dict(st.session_state.get("tab4_layout_config") or DEFAULT_LAYOUT_CONFIG)
        cur_merged.update(pending_overrides)
        _apply_carto_layout_to_widgets(cur_merged)

    # Initialize toggle/slider widget keys via setdefault so preset updates never trigger Streamlit default warnings
    for _wk, _wdef in (
        ("tab4_chk_title", True),
        ("tab4_chk_footer", True),
        ("tab4_chk_north", True),
        ("tab4_chk_scalebar", True),
        ("tab4_chk_num_scale", True),
        ("tab4_chk_legend", True),
        ("tab4_chk_leg_meta", True),
        ("tab4_chk_inset", False),
        ("tab4_chk_grid", True),
        ("tab4_grid_alpha", 0.45),
        ("tab4_chk_aoi", True),
    ):
        st.session_state.setdefault(_wk, _wdef)

    # Normalize legacy lowercase paper_format value if persisted in session state
    if st.session_state.get("tab4_page_format") not in CARTO_FORMAT_KEYS:
        legacy_f = str(st.session_state.get("tab4_page_format") or "A4_LANDSCAPE").lower()
        st.session_state["tab4_page_format"] = {
            "a4_landscape": "A4_LANDSCAPE",
            "a4_portrait": "A4_PORTRAIT",
            "a3_landscape": "A3_LANDSCAPE",
            "square": "SQUARE_1_1",
        }.get(legacy_f, "A4_LANDSCAPE")

    col_cfg, col_prev = st.columns([0.40, 0.60], gap="large")

    with col_cfg:
        src_ids = [s["id"] for s in carto_sources]
        src_map = {s["id"]: s for s in carto_sources}
        pending_src = st.session_state.pop("tab4_pending_source_select", None)
        if pending_src in src_ids:
            st.session_state["tab4_source_select"] = pending_src
            st.session_state["tab4_last_src_sig"] = None

        # --- EXPANDER 1: Layer, Paper & Theme + 1-Click Quick Presets ---
        with st.expander(t("carto_exp_1_layer"), expanded=True):
            selected_src_id = _touch_selector(
                t("carto_layer_label"),
                options=src_ids,
                default=src_ids[0],
                format_func=lambda sid: src_map[sid]["label"],
                key="tab4_source_select",
                use_pills=True,
            )
            src = src_map[selected_src_id]

            if st.session_state.get("tab4_last_src_sig") != src["label"]:
                st.session_state["tab4_title_input"] = src["default_title"]
                st.session_state["tab4_subtitle_input"] = src["default_subtitle"]
                st.session_state["tab4_leg_title_input"] = str(src.get("legend_title") or "")
                st.session_state["tab4_last_src_sig"] = src["label"]

            if "tab4_author_input" not in st.session_state:
                st.session_state["tab4_author_input"] = t("carto_author_default")
            if "tab4_note_input" not in st.session_state:
                st.session_state["tab4_note_input"] = ""
            if "tab4_leg_title_input" not in st.session_state:
                st.session_state["tab4_leg_title_input"] = str(src.get("legend_title") or "")

            carto_fmt = _touch_selector(
                t("carto_format_label"),
                options=CARTO_FORMAT_KEYS,
                default="A4_LANDSCAPE",
                format_func=lambda k, _lang=lang: t(CARTO_FORMAT_I18N[k], lang=_lang),
                key="tab4_page_format",
                use_pills=True,
            )
            carto_theme = _touch_selector(
                t("carto_theme_label"),
                options=CARTO_THEME_KEYS,
                default="classic_light",
                format_func=lambda k, _lang=lang: t(CARTO_THEME_I18N[k], lang=_lang),
                key="tab4_theme",
                use_pills=True,
            )

            st.caption(t("carto_presets_label"))
            pr_c1, pr_c2, pr_c3 = st.columns(3)
            with pr_c1:
                if st.button(t("carto_preset_thesis"), width="stretch", key="tab4_preset_thesis"):
                    st.session_state["tab4_pending_preset"] = "thesis"
                    st.rerun(scope="fragment")
            with pr_c2:
                if st.button(t("carto_preset_poster"), width="stretch", key="tab4_preset_poster"):
                    st.session_state["tab4_pending_preset"] = "poster"
                    st.rerun(scope="fragment")
            with pr_c3:
                if st.button(t("carto_preset_crisis"), width="stretch", key="tab4_preset_crisis"):
                    st.session_state["tab4_pending_preset"] = "crisis"
                    st.rerun(scope="fragment")

            _auth_u_ws4 = st.session_state.get("auth_user")
            if isinstance(_auth_u_ws4, dict) and _auth_u_ws4.get("username"):
                _uname_ws4 = str(_auth_u_ws4["username"])
                st.divider()
                st.caption(t("carto_saved_tpl_header", lang=lang))
                sv_c1, sv_c2 = st.columns([1.45, 0.95], gap="small")
                with sv_c1:
                    tpl_name_in = st.text_input(
                        t("carto_saved_tpl_header", lang=lang),
                        placeholder=t("carto_saved_tpl_name_ph", lang=lang),
                        label_visibility="collapsed",
                        key="tab4_saved_tpl_name_in",
                    )
                with sv_c2:
                    if st.button(
                        t("carto_saved_tpl_save_btn", lang=lang),
                        width="stretch",
                        key="tab4_saved_tpl_save_btn",
                    ):
                        clean_tn = str(tpl_name_in or "").strip()
                        if clean_tn:
                            cur_cfg_save = dict(
                                st.session_state.get("tab4_layout_config") or DEFAULT_LAYOUT_CONFIG
                            )
                            save_map_layout(_uname_ws4, clean_tn, cur_cfg_save, sync=True)
                            _maybe_auto_save_session()
                            st.success(t("carto_saved_tpl_saved_ok", lang=lang, name=clean_tn))
                            st.rerun(scope="fragment")

                _user_layouts = list_map_layouts(_uname_ws4)
                if _user_layouts:
                    _lnames = list(_user_layouts.keys())
                    sel_tpl = st.selectbox(
                        t("carto_saved_tpl_select_label", lang=lang),
                        options=_lnames,
                        key="tab4_saved_tpl_select",
                    )
                    tl_c1, tl_c2 = st.columns(2, gap="small")
                    with tl_c1:
                        if st.button(
                            t("carto_saved_tpl_load_btn", lang=lang),
                            width="stretch",
                            key="tab4_saved_tpl_load_btn",
                        ):
                            _cfg_load = (_user_layouts.get(sel_tpl) or {}).get("config")
                            if isinstance(_cfg_load, dict):
                                st.session_state["tab4_pending_layout_overrides"] = dict(_cfg_load)
                                st.toast(t("carto_saved_tpl_loaded_ok", lang=lang, name=sel_tpl))
                                st.rerun(scope="fragment")
                    with tl_c2:
                        if st.button(
                            t("carto_saved_tpl_del_btn", lang=lang),
                            width="stretch",
                            key="tab4_saved_tpl_del_btn",
                        ):
                            delete_map_layout(_uname_ws4, sel_tpl, sync=True)
                            st.success(t("carto_saved_tpl_deleted_ok", lang=lang, name=sel_tpl))
                            st.rerun(scope="fragment")

        # --- EXPANDER 2: Title, Notes & Footer ---
        with st.expander(t("carto_exp_2_text"), expanded=False):
            chk_title = st.toggle(t("carto_chk_title"), key="tab4_chk_title")
            title_pos = _touch_selector(
                t("carto_title_pos_label"),
                options=CARTO_TITLE_POS_KEYS,
                default="top_left",
                format_func=lambda k, _lang=lang: t(CARTO_TITLE_POS_I18N[k], lang=_lang),
                key="tab4_title_pos",
                use_pills=True,
            )
            carto_title = st.text_input(t("carto_title_label"), key="tab4_title_input")
            carto_subtitle = st.text_input(t("carto_subtitle_label"), key="tab4_subtitle_input")
            carto_author = st.text_input(t("carto_author_label"), key="tab4_author_input")

            carto_note = st.text_input(
                t("carto_note_label"),
                placeholder=t("carto_note_placeholder"),
                key="tab4_note_input",
            )
            note_pos = _touch_selector(
                t("carto_note_pos_label"),
                options=CARTO_NOTE_POS_KEYS,
                default="bottom_right",
                format_func=lambda k, _lang=lang: t(CARTO_NOTE_POS_I18N[k], lang=_lang),
                key="tab4_note_pos",
                use_pills=True,
            )
            chk_footer = st.toggle(t("carto_chk_footer"), key="tab4_chk_footer")

        # --- EXPANDER 3: Cartographic Elements & QGIS-Style Placement ---
        with st.expander(t("carto_exp_3_elements"), expanded=False):
            # North Arrow
            chk_north = st.toggle(t("carto_chk_north"), key="tab4_chk_north")
            north_pos = _touch_selector(
                t("carto_north_pos_label"),
                options=CARTO_CORNER_KEYS,
                default="top_right",
                format_func=lambda k, _lang=lang: t(CARTO_CORNER_I18N[k], lang=_lang),
                key="tab4_north_pos",
                use_pills=True,
            )
            north_style = _touch_selector(
                t("carto_north_style_label"),
                options=CARTO_NORTH_STYLE_KEYS,
                default="classic_compass",
                format_func=lambda k, _lang=lang: t(CARTO_NORTH_STYLE_I18N[k], lang=_lang),
                key="tab4_north_style",
                use_pills=True,
            )
            north_size = _touch_selector(
                t("carto_north_size_label"),
                options=CARTO_NORTH_SIZE_KEYS,
                default="medium",
                format_func=lambda k, _lang=lang: t(CARTO_NORTH_SIZE_I18N[k], lang=_lang),
                key="tab4_north_size",
                use_pills=True,
            )
            st.divider()

            # Scale Bar
            chk_scale = st.toggle(t("carto_chk_scalebar"), key="tab4_chk_scalebar")
            scale_pos = _touch_selector(
                t("carto_scale_pos_label"),
                options=CARTO_CORNER_KEYS,
                default="bottom_left",
                format_func=lambda k, _lang=lang: t(CARTO_CORNER_I18N[k], lang=_lang),
                key="tab4_scale_pos",
                use_pills=True,
            )
            scale_style = _touch_selector(
                t("carto_scale_style_label"),
                options=CARTO_SCALE_STYLE_KEYS,
                default="alternating_bar",
                format_func=lambda k, _lang=lang: t(CARTO_SCALE_STYLE_I18N[k], lang=_lang),
                key="tab4_scale_style",
                use_pills=True,
            )
            chk_num_scale = st.toggle(t("carto_chk_num_scale"), key="tab4_chk_num_scale")
            st.divider()

            # Legend
            chk_legend = st.toggle(t("carto_chk_legend"), key="tab4_chk_legend")
            legend_pos = _touch_selector(
                t("carto_leg_pos_label"),
                options=CARTO_LEGEND_POS_KEYS,
                default="right_panel",
                format_func=lambda k, _lang=lang: t(CARTO_LEGEND_POS_I18N[k], lang=_lang),
                key="tab4_legend_pos",
                use_pills=True,
            )
            carto_leg_title = st.text_input(t("carto_leg_title_label"), key="tab4_leg_title_input")
            chk_leg_meta = st.toggle(t("carto_chk_leg_meta"), key="tab4_chk_leg_meta")
            st.divider()

            # Overview / Inset Locator Map
            chk_inset = st.toggle(t("carto_chk_inset"), key="tab4_chk_inset")
            inset_pos = _touch_selector(
                t("carto_inset_pos_label"),
                options=CARTO_INSET_POS_KEYS,
                default="right_panel_bottom",
                format_func=lambda k, _lang=lang: t(CARTO_INSET_POS_I18N[k], lang=_lang),
                key="tab4_inset_pos",
                use_pills=True,
            )

        # --- EXPANDER 4: Coordinate Grid & AOI Frame ---
        with st.expander(t("carto_exp_4_grid"), expanded=False):
            chk_grid = st.toggle(t("carto_chk_grid"), key="tab4_chk_grid")
            grid_type = _touch_selector(
                t("carto_grid_type_label"),
                options=CARTO_GRID_TYPE_KEYS,
                default="lines",
                format_func=lambda k, _lang=lang: t(CARTO_GRID_TYPE_I18N[k], lang=_lang),
                key="tab4_grid_type",
                use_pills=True,
            )

            def _fmt_grid_opt(val: int, _lang: str = lang) -> str:
                if val == 0:
                    return t("carto_grid_auto", lang=_lang)
                return f"{val // 1000} km" if val >= 1000 else f"{val} m"

            carto_grid_val = _touch_selector(
                t("carto_grid_spacing_label"),
                options=CARTO_GRID_OPTIONS,
                default=0,
                format_func=_fmt_grid_opt,
                key="tab4_grid_spacing",
                use_pills=True,
            )
            carto_grid_color = _touch_selector(
                t("carto_grid_style_label"),
                options=CARTO_GRID_STYLE_KEYS,
                default="light",
                format_func=lambda k, _lang=lang: t(CARTO_GRID_STYLE_I18N[k], lang=_lang),
                key="tab4_grid_style",
                use_pills=True,
            )
            carto_grid_alpha = st.slider(
                t("carto_grid_alpha_label"),
                0.10,
                1.00,
                step=0.05,
                key="tab4_grid_alpha",
            )
            st.divider()
            chk_aoi = st.toggle(t("carto_chk_aoi"), key="tab4_chk_aoi")
            aoi_col = _touch_selector(
                t("carto_aoi_color_label"),
                options=CARTO_AOI_COLOR_KEYS,
                default="#0F766E",
                format_func=lambda k, _lang=lang: t(CARTO_AOI_COLOR_I18N[k], lang=_lang),
                key="tab4_aoi_color",
                use_pills=True,
            )

        # Build unified MapLayoutConfig from current composer controls
        active_layout_cfg = merge_layout_config(
            {
                "paper_format": str(carto_fmt),
                "theme": str(carto_theme),
                "show_title": bool(chk_title),
                "title": str(carto_title or ""),
                "subtitle": str(carto_subtitle or ""),
                "author": str(carto_author or ""),
                "title_position": str(title_pos),
                "show_footer": bool(chk_footer),
                "custom_note_text": str(carto_note or ""),
                "custom_note_position": str(note_pos),
                "show_grid": bool(chk_grid),
                "grid_spacing": "auto" if int(carto_grid_val) == 0 else int(carto_grid_val),
                "grid_style": str(grid_type),
                "grid_color": str(carto_grid_color),
                "grid_alpha": float(carto_grid_alpha),
                "show_aoi_outline": bool(chk_aoi),
                "aoi_outline_color": str(aoi_col),
                "show_north_arrow": bool(chk_north),
                "north_position": str(north_pos),
                "north_style": str(north_style),
                "north_size": str(north_size),
                "show_scale_bar": bool(chk_scale),
                "scale_position": str(scale_pos),
                "scale_style": str(scale_style),
                "show_numeric_scale": bool(chk_num_scale),
                "show_legend": bool(chk_legend),
                "legend_position": str(legend_pos),
                "legend_title": str(carto_leg_title or src.get("legend_title") or ""),
                "show_metadata_in_legend": bool(chk_leg_meta),
                "show_inset_map": bool(chk_inset),
                "inset_position": str(inset_pos),
                "markers": list((st.session_state.get("tab4_layout_config") or {}).get("markers") or []),
            }
        )
        st.session_state["tab4_layout_config"] = active_layout_cfg

        auto_live = st.toggle(t("carto_auto_live_toggle"), value=True, key="tab4_auto_live_preview")
        btn_gen_map = st.button(
            t("carto_btn_generate"),
            type="primary",
            width="stretch",
            key="tab4_btn_generate",
        )

        # Render on button click OR automatically when auto-live is enabled and settings changed
        cur_render_sig = (src["label"], str(sorted(active_layout_cfg.items())), lang)
        already_rendered = st.session_state.get("tab4_carto_result") is not None
        should_render = bool(btn_gen_map) or (
            auto_live and (not already_rendered or st.session_state.get("tab4_last_render_sig") != cur_render_sig)
        )

        if should_render:
            with st.spinner(t("carto_spinner")):
                try:
                    res_map = render_geodetic_map(
                        src,
                        layout_config=active_layout_cfg,
                        aoi_polygon_geojson=(
                            src["aoi_polygon_4326"] if "aoi_polygon_4326" in src else active_polygon
                        ),
                        aoi_bbox_4326=src.get("aoi_bbox_4326") or active_bbox,
                        aoi_legend_label=t("slippy_aoi_outline"),
                        lang=lang,
                    )
                    tif_out = src["tif_bytes"]
                    if tif_out is None:
                        tif_src = (
                            src["raster"][src["bands"]]
                            if (isinstance(src["raster"], xr.Dataset) and src["bands"])
                            else src["raster"]
                        )
                        tif_out = export_to_geotiff_bytes(tif_src, target_epsg=src["crs"])
                    epsg_tag = str(src["crs"]).replace(":", "")
                    stem = f"{src['file_stem']}_{epsg_tag}_{str(carto_fmt).lower()}"
                    st.session_state["tab4_carto_result"] = res_map
                    st.session_state["tab4_carto_meta"] = {
                        "crs": str(src["crs"]),
                        "stem": stem,
                        "tif_bytes": tif_out,
                        "tif_filename": src["tif_filename"],
                    }
                    st.session_state["tab4_last_render_sig"] = cur_render_sig
                    gc.collect()
                except Exception as exc:
                    st.error(t("carto_error", err=exc))

    with col_prev:
        st.markdown(f"#### {t('carto_preview_header')}")
        carto_res: dict[str, Any] | None = st.session_state.get("tab4_carto_result")
        carto_meta: dict[str, Any] | None = st.session_state.get("tab4_carto_meta")

        if carto_res is None or carto_meta is None:
            st.info(t("carto_prompt_generate"))
        else:
            st.image(carto_res["png_preview_bytes"], width="stretch")
            st.caption(
                t(
                    "carto_preview_caption",
                    scale=format_metric_number(carto_res["scale_denom"]),
                    grid=format_metric_number(carto_res["grid_spacing_m"]),
                    crs=carto_meta["crs"],
                )
            )

            dl_c1, dl_c2, dl_c3 = st.columns(3)
            with dl_c1:
                st.download_button(
                    label=t("carto_btn_dl_pdf"),
                    data=carto_res["pdf_bytes"],
                    file_name=f"{carto_meta['stem']}_300dpi.pdf",
                    mime="application/pdf",
                    width="stretch",
                    key="tab4_dl_pdf",
                )
            with dl_c2:
                st.download_button(
                    label=t("carto_btn_dl_png"),
                    data=carto_res["png_hires_bytes"],
                    file_name=f"{carto_meta['stem']}_300dpi.png",
                    mime="image/png",
                    width="stretch",
                    key="tab4_dl_png",
                )
            with dl_c3:
                st.download_button(
                    label=t("carto_btn_dl_tif"),
                    data=carto_meta["tif_bytes"],
                    file_name=carto_meta["tif_filename"],
                    mime="image/tiff",
                    width="stretch",
                    key="tab4_dl_tif",
                )


@st.fragment
def _render_gnss_skyplot_fragment(gnss_res: dict[str, Any], lang: str) -> None:
    """Fragment za 24-urni časovni drsnik in polarni GNSS Skyplot (takojšnje osveževanje)."""
    st.markdown(f"#### {t('gnss_skyplot_header')}")
    tl_df: pd.DataFrame = gnss_res["timeline_df"]
    t_labels = tl_df["time_str"].tolist()
    default_ep_idx = min(48, len(t_labels) - 1)
    chosen_ep_idx = st.select_slider(
        t("gnss_epoch_slider"),
        options=list(range(len(t_labels))),
        value=default_ep_idx,
        format_func=lambda idx: f"{t_labels[idx]} ({gnss_res['tz_name']})",
        key="tab7_skyplot_epoch_slider",
    )

    az_c = np.asarray(gnss_res["az_centers_deg"], dtype=np.float64)
    az_closed = np.append(az_c, az_c[0])
    terr_closed = np.append(gnss_res["terrain_mask_deg"], gnss_res["terrain_mask_deg"][0])
    obs_closed = np.append(gnss_res["obstacle_mask_deg"], gnss_res["obstacle_mask_deg"][0])
    cut_closed = np.full_like(az_closed, float(gnss_res["cutoff_deg"]))

    fig_sky = go.Figure()
    ring_az = np.concatenate([az_closed, az_closed[::-1]])
    if float(np.max(terr_closed)) > 0.5:
        terr_r = np.concatenate([np.zeros_like(az_closed), terr_closed[::-1]])
        fig_sky.add_trace(
            go.Scatterpolar(
                r=terr_r,
                theta=ring_az,
                fill="toself",
                fillcolor="rgba(100, 116, 139, 0.34)",
                line=dict(color="#475569", width=1.4),
                name=t("gnss_sky_terrain"),
            )
        )
    if float(np.max(obs_closed)) > 0.5:
        obs_r = np.concatenate([np.zeros_like(az_closed), obs_closed[::-1]])
        fig_sky.add_trace(
            go.Scatterpolar(
                r=obs_r,
                theta=ring_az,
                fill="toself",
                fillcolor="rgba(15, 118, 110, 0.28)",
                line=dict(color="#0F766E", width=1.5),
                name=t("gnss_sky_obs"),
            )
        )
    fig_sky.add_trace(
        go.Scatterpolar(
            r=cut_closed,
            theta=az_closed,
            mode="lines",
            line=dict(color="#DC2626", width=1.4, dash="dash"),
            name=t("gnss_sky_cutoff", deg=gnss_res["cutoff_deg"]),
        )
    )

    ep_sats = gnss_res["epoch_sat_states"][int(chosen_ep_idx)]
    for const in gnss_res["constellations"]:
        vis_c = [s for s in ep_sats if s["constellation"] == const and s["visible_real"]]
        if vis_c:
            fig_sky.add_trace(
                go.Scatterpolar(
                    r=[s["el_deg"] for s in vis_c],
                    theta=[s["az_deg"] for s in vis_c],
                    mode="markers+text",
                    text=[s["prn"] for s in vis_c],
                    textposition="top center",
                    textfont=dict(size=10, color="#0F172A"),
                    marker=dict(
                        size=10,
                        color=GNSS_CONST_COLORS.get(const, "#0F766E"),
                        line=dict(color="#FFFFFF", width=1.2),
                    ),
                    name=f"{const} ({len(vis_c)})",
                )
            )
    blocked_sats = [s for s in ep_sats if not s["visible_real"]]
    if blocked_sats:
        fig_sky.add_trace(
            go.Scatterpolar(
                r=[s["el_deg"] for s in blocked_sats],
                theta=[s["az_deg"] for s in blocked_sats],
                mode="markers+text",
                text=[s["prn"] for s in blocked_sats],
                textposition="top center",
                textfont=dict(size=9, color="#64748B"),
                marker=dict(size=9, symbol="x", color="#94A3B8", line=dict(width=1.5)),
                name=f"{t('gnss_sky_blocked')} ({len(blocked_sats)})",
            )
        )

    ep_row = tl_df.iloc[int(chosen_ep_idx)]
    pdop_ep_str = f"{ep_row['pdop_real']:.2f}" if pd.notna(ep_row["pdop_real"]) else "N/A"
    fig_sky.update_layout(
        title=dict(
            text=(
                f"Skyplot · {gnss_res['plan_date']} {t_labels[int(chosen_ep_idx)]} · "
                f"Visible: {int(ep_row['visible_real'])} / Open: {int(ep_row['visible_open'])} · PDOP: {pdop_ep_str}"
            ),
            font=dict(size=13, color="#0F172A"),
        ),
        polar=dict(
            bgcolor="#FFFFFF",
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=["N (0°)", "NE", "E (90°)", "SE", "S (180°)", "SW", "W (270°)", "NW"],
                gridcolor="#E2E8F0",
            ),
            radialaxis=dict(
                range=[90, 0],
                tickvals=[90, 60, 30, 0],
                ticktext=["90°", "60°", "30°", "0°"],
                gridcolor="#E2E8F0",
            ),
        ),
        height=540,
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_sky, width="stretch", key="tab7_skyplot_plotly")


def _resolve_ai_target_carto_layer(target_layer: str, lang: str) -> dict[str, Any] | None:
    """Resolve the best available raster layer for AI chat-driven map generation.

    Priority order:
    1. Latest spatial AI result (``tab6_ai``), respecting ``target_layer`` (``satellite_rgb`` vs ``analysis_mask``).
    2. Any active loaded layer from Workspaces 1, 2, or 3 (``_collect_carto_sources``).
    3. ``None`` if no spatial raster is loaded anywhere in the session yet.
    """
    sources = _collect_carto_sources(lang)
    ai_res = st.session_state.get("tab6_ai_result")

    # If user specifically asked for the satellite RGB scene of the latest AI result
    if target_layer == "satellite_rgb":
        if isinstance(ai_res, dict) and isinstance(ai_res.get("scene_ds"), xr.Dataset):
            ai_plan = ai_res.get("plan") or {}
            ai_place = str(ai_plan.get("place_name") or "AOI")
            ai_crs = str(ai_res.get("crs", CRS_D96TM))
            ai_sensor_cfg = SENSOR_META.get(ai_plan.get("sensor", SENSOR_S2), SENSOR_META[SENSOR_S2])
            return {
                "id": "tab6_ai",
                "label": t("carto_layer_tab6_ai", lang=lang, title=f"RGB — {ai_place}"),
                "raster": ai_res["scene_ds"],
                "crs": ai_crs,
                "is_rgb": True,
                "bands": ai_res.get("bands", ["B04", "B03", "B02"]),
                "use_percentile_stretch": True,
                "gamma": 1.0,
                "cmap_name": "Greys_r",
                "vmin": 0.0,
                "vmax": 1.0,
                "discrete_items": None,
                "legend_title": "RGB" if ai_plan.get("sensor") != SENSOR_S1 else "SAR VV+VH",
                "legend_units": "",
                "stats_summary": [],
                "default_title": f"{ai_sensor_cfg['short_name']} RGB — {ai_place}",
                "default_subtitle": f"{ai_res.get('date_str', '')} | {ai_res.get('tile', '')}",
                "sensor_label": ai_sensor_cfg["short_name"],
                "date_label": str(ai_res.get("date_str", "")),
                "resolution_m": float(ai_plan.get("resolution_m", 10.0)),
                "tif_bytes": ai_res.get("tif_bytes"),
                "tif_filename": ai_res.get("tif_filename") or f"AI_RGB_{ai_place}.tif",
                "file_stem": f"AI_RGB_{ai_place}",
                "aoi_bbox_4326": ai_res.get("bbox_4326"),
                "aoi_polygon_4326": None,
            }
        # Fall back to any RGB source loaded in Workspace 1/2
        for s in sources:
            if s.get("is_rgb"):
                return s

    if target_layer == "workspace_layer":
        for s in sources:
            if s["id"] != "tab6_ai":
                return s

    # Priority 1: latest AI spatial layer (`tab6_ai`)
    for s in sources:
        if s["id"] == "tab6_ai":
            return s

    # Priority 2: any loaded Workspace 1/2/3 layer
    return sources[0] if sources else None


KEEP_EXPANDED_TURNS: int = 2


def group_ai_chat_turns(chat_messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group flat ``ai_chat_messages`` into logical Q&A turns.

    Each turn dictionary contains:
    ``turn_idx``, ``user_msg_idx``, ``user_msg``, ``assistant_msg_idx``, ``assistant_msg``.
    """
    turns: list[dict[str, Any]] = []
    for idx, msg in enumerate(chat_messages or []):
        if not isinstance(msg, dict):
            continue
        role = "user" if msg.get("role") == "user" else "assistant"
        if role == "user":
            turns.append(
                {
                    "turn_idx": len(turns),
                    "user_msg_idx": idx,
                    "user_msg": msg,
                    "assistant_msg_idx": None,
                    "assistant_msg": None,
                }
            )
        else:
            if turns and turns[-1].get("assistant_msg") is None:
                turns[-1]["assistant_msg_idx"] = idx
                turns[-1]["assistant_msg"] = msg
            else:
                turns.append(
                    {
                        "turn_idx": len(turns),
                        "user_msg_idx": None,
                        "user_msg": None,
                        "assistant_msg_idx": idx,
                        "assistant_msg": msg,
                    }
                )
    return turns


def build_collapsed_turn_label(
    turn: dict[str, Any],
    *,
    chat_results: dict[str, Any] | None = None,
    lang: str = "sl",
    max_prompt_chars: int = 75,
) -> str:
    """Build a concise, scannable expander header for an older collapsed AI chat turn."""
    num = int(turn.get("turn_idx", 0)) + 1
    user_msg = turn.get("user_msg") or {}
    asst_msg = turn.get("assistant_msg") or {}

    raw_prompt = " ".join(str(user_msg.get("content") or "").split())
    if len(raw_prompt) > max_prompt_chars:
        snippet = raw_prompt[: max(max_prompt_chars - 1, 10)].rstrip() + "…"
    else:
        snippet = raw_prompt or "…"

    suffix = ""
    res_id = asst_msg.get("result_id")
    res_obj = (chat_results or {}).get(str(res_id)) if res_id else None
    plan = asst_msg.get("plan") if isinstance(asst_msg.get("plan"), dict) else None

    if isinstance(res_obj, dict) and res_obj.get("kind") == "carto_map":
        l_cfg = res_obj.get("layout_config") or {}
        m_title = str(l_cfg.get("title") or res_obj.get("stem") or t("ai_chat_map_badge", lang=lang)).strip()
        if len(m_title) > 40:
            m_title = m_title[:39].rstrip() + "…"
        suffix = f" — 🗺️ {m_title}"
    elif isinstance(res_obj, dict) or isinstance(plan, dict):
        place = str(
            (res_obj or {}).get("resolved_place")
            or (plan or {}).get("place_name")
            or ""
        ).strip()
        idx_tag = str(
            (res_obj or {}).get("index_name")
            or (plan or {}).get("index_name")
            or (plan or {}).get("task_type")
            or ""
        ).strip()
        s_date = str((plan or {}).get("start_date") or "").strip()
        e_date = str((plan or {}).get("end_date") or "").strip()
        period_tag = ""
        if len(s_date) >= 4 and len(e_date) >= 4:
            y1, y2 = s_date[:4], e_date[:4]
            period_tag = y1 if y1 == y2 else f"{y1}–{y2}"
        elif len(e_date) >= 4:
            period_tag = e_date[:4]

        meta_parts = [p for p in (idx_tag, period_tag) if p]
        meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
        if place or meta_str:
            suffix = f" — 🛰️ {place}{meta_str}".rstrip()

    return f'💬 #{num} · "{snippet}"{suffix}'


def _render_ai_chat_turn(
    turn: dict[str, Any],
    *,
    target_crs: str,
    lang: str,
) -> None:
    """Render a single AI Q&A turn with inline Edit & Resend controls on the user message."""
    t_idx = int(turn.get("turn_idx", 0))
    user_msg = turn.get("user_msg")
    user_msg_idx = turn.get("user_msg_idx")
    asst_msg = turn.get("assistant_msg")

    if isinstance(user_msg, dict):
        user_text = str(user_msg.get("content") or "")
        with st.chat_message("user"):
            st.markdown(user_text)
            if user_msg_idx is not None:
                _sp_u, col_edit, col_resend = st.columns([6.4, 1.3, 1.3], gap="small")
                with col_edit:
                    if st.button(
                        t("ai_turn_edit_btn", lang=lang),
                        key=f"edit_btn_{t_idx}",
                        width="stretch",
                    ):
                        st.session_state["ai_editing_turn_idx"] = t_idx
                        st.rerun()
                with col_resend:
                    if st.button(
                        t("ai_turn_resend_btn", lang=lang),
                        key=f"resend_btn_{t_idx}",
                        width="stretch",
                    ):
                        st.session_state["ai_editing_turn_idx"] = None
                        st.session_state["ai_pending_rerun_turn"] = {
                            "user_msg_idx": int(user_msg_idx),
                            "prompt": user_text,
                        }
                        st.rerun()

                if st.session_state.get("ai_editing_turn_idx") == t_idx:
                    edited_val = st.text_area(
                        t("ai_turn_edit_label", lang=lang),
                        value=user_text,
                        key=f"edit_ta_{t_idx}",
                        height=90,
                    )
                    ec_send, ec_cancel, _ec_sp = st.columns([2.2, 1.3, 4.5], gap="small")
                    with ec_send:
                        if st.button(
                            t("ai_turn_edit_send", lang=lang),
                            type="primary",
                            key=f"edit_submit_{t_idx}",
                            width="stretch",
                        ):
                            clean_ed = str(edited_val or "").strip()
                            if clean_ed:
                                st.session_state["ai_editing_turn_idx"] = None
                                st.session_state["ai_pending_rerun_turn"] = {
                                    "user_msg_idx": int(user_msg_idx),
                                    "prompt": clean_ed,
                                }
                                st.rerun()
                    with ec_cancel:
                        if st.button(
                            t("ai_turn_edit_cancel", lang=lang),
                            key=f"edit_cancel_{t_idx}",
                            width="stretch",
                        ):
                            st.session_state["ai_editing_turn_idx"] = None
                            st.rerun()

    if isinstance(asst_msg, dict):
        with st.chat_message("assistant"):
            st.markdown(str(asst_msg.get("content") or ""))
            fb_chat = str(asst_msg.get("fallback_note") or "").strip()
            if fb_chat:
                st.caption(f"ℹ️ {fb_chat}")
            res_id = asst_msg.get("result_id")
            if res_id:
                _render_ai_chat_analysis_fragment(
                    result_id=str(res_id),
                    target_crs=target_crs,
                    lang=lang,
                )


@st.fragment
def _render_ai_chat_analysis_fragment(
    *,
    result_id: str,
    target_crs: str,
    lang: str,
) -> None:
    """Fragment za prikaz rezultatov satelitske/geodetske analize ali izrisane 300 DPI karte znotraj klepeta AI agenta.

    Podpira vseh 10 zmožnosti aplikacije + neposredno generiranje 300 DPI geodetskih kart (QGIS-slog).
    Zemljevidi se izrišejo prek ``components.html`` brez dvosmernega ``st_folium`` povratnega klica,
    da drsenje strani nikoli ne skače.
    """
    results_store: dict[str, Any] = st.session_state.get("ai_chat_results") or {}
    ai_res_show: dict[str, Any] | None = results_store.get(result_id)
    if not isinstance(ai_res_show, dict):
        return

    # --- Direct AI Chat Cartographic Map Result (Part 3: Inline 150 DPI Preview + 300 DPI PDF/PNG + WS4 Handoff) ---
    if ai_res_show.get("kind") == "carto_map" and isinstance(ai_res_show.get("map_result"), dict):
        m_res = ai_res_show["map_result"]
        m_cfg = ai_res_show.get("layout_config") or {}
        m_crs = str(ai_res_show.get("crs", target_crs))
        m_stem = str(ai_res_show.get("stem") or "AI_Geodetic_Map")
        fmt_lbl = t(CARTO_FORMAT_I18N.get(str(m_cfg.get("paper_format", "A4_LANDSCAPE")), "carto_fmt_a4_land"))
        theme_lbl = t(CARTO_THEME_I18N.get(str(m_cfg.get("theme", "classic_light")), "carto_theme_classic"))
        scale_str = format_metric_number(int(m_res.get("scale_denom", 10000)))

        # Crisp 150 DPI preview image
        st.image(m_res["png_preview_bytes"], width="stretch")

        # Compact metadata pills (cartographic teal & slate, zero purple)
        pills_html = (
            f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 10px 0;">'
            f'<span style="background:#F0FDFA;color:#0F766E;border:1px solid #99F6E4;border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600;">📏 {t("ai_map_pill_scale")}: M ≈ 1 : {scale_str}</span>'
            f'<span style="background:#F8FAFC;color:#0F172A;border:1px solid #CBD5E1;border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600;">📄 {t("ai_map_pill_format")}: {fmt_lbl}</span>'
            f'<span style="background:#F8FAFC;color:#0F172A;border:1px solid #CBD5E1;border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600;">📐 {t("ai_plan_crs")}: {m_crs}</span>'
            f'<span style="background:#F8FAFC;color:#0F172A;border:1px solid #CBD5E1;border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600;">🎨 {t("ai_map_pill_theme")}: {theme_lbl}</span>'
            f'</div>'
        )
        st.markdown(pills_html, unsafe_allow_html=True)

        # 3 side-by-side action buttons: 300 DPI PDF, 300 DPI PNG, Fine-tune in Workspace 4
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.download_button(
                label=t("ai_map_btn_pdf"),
                data=m_res["pdf_bytes"],
                file_name=f"{m_stem}_300dpi.pdf",
                mime="application/pdf",
                width="stretch",
                key=f"ai_map_dl_pdf_{result_id}",
            )
        with mc2:
            st.download_button(
                label=t("ai_map_btn_png"),
                data=m_res["png_hires_bytes"],
                file_name=f"{m_stem}_300dpi.png",
                mime="image/png",
                width="stretch",
                key=f"ai_map_dl_png_{result_id}",
            )
        with mc3:
            if st.button(
                t("ai_map_btn_edit_ws4"),
                type="primary",
                width="stretch",
                key=f"ai_map_edit_ws4_{result_id}",
            ):
                src_id = str(ai_res_show.get("layer_source_id") or "tab6_ai")
                st.session_state["tab4_pending_source_select"] = src_id
                st.session_state["tab4_pending_layout_overrides"] = dict(m_cfg)
                st.session_state["tab4_layout_config"] = dict(m_cfg)
                st.session_state["tab4_carto_result"] = m_res
                st.session_state["tab4_carto_meta"] = {
                    "crs": m_crs,
                    "stem": m_stem,
                    "tif_bytes": ai_res_show.get("tif_bytes"),
                    "tif_filename": ai_res_show.get("tif_filename") or f"{m_stem}.tif",
                }
                st.session_state["pending_workspace"] = "ws_geo"
                st.rerun(scope="app")
        return

    if not ai_res_show.get("plan"):
        return

    pl = ai_res_show["plan"]
    ai_crs_show = str(ai_res_show.get("crs", target_crs))
    ai_kind_show = str(ai_res_show.get("kind", ""))
    eng_lbl = (
        f"Gemini ({pl.get('gemini_model', DEFAULT_GEMINI_MODEL)})"
        if pl.get("parser_engine") == "gemini"
        else t("ai_plan_engine_local")
    )
    fb_note = pl.get("fallback_note_en" if lang == "en" else "fallback_note_sl") or ""
    period_txt = (
        pl["event_date"]
        if pl["task_type"].startswith("disaster_") or pl["task_type"] == "gnss_plan"
        else ("Copernicus GLO-30" if pl["task_type"] == "terrain_dem" else f"{pl['start_date']} → {pl['end_date']}")
    )
    if ai_kind_show == "terrain_dem":
        ai_sensor_short = "Copernicus DEM 30 m"
    elif ai_kind_show == "gnss_plan":
        ai_sensor_short = "GPS + Galileo + GLONASS"
    else:
        ai_sensor_short = SENSOR_META.get(pl["sensor"], SENSOR_META[SENSOR_S2])["short_name"]

    # 1. Compact Geodetic Plan Pills (cartographic teal & slate, zero purple)
    plan_pills_html = (
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px 0;">'
        f'<span style="background:#F0FDFA;color:#0F766E;border:1px solid #99F6E4;border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600;">📍 {t("ai_plan_place")}: {ai_res_show.get("resolved_place", pl["place_name"])} ({pl["radius_m"]} m)</span>'
        f'<span style="background:#F8FAFC;color:#0F172A;border:1px solid #CBD5E1;border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600;">🛰️ {t("ai_plan_sensor")}: {ai_sensor_short}</span>'
        f'<span style="background:#F8FAFC;color:#0F172A;border:1px solid #CBD5E1;border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600;">📅 {t("ai_plan_period")}: {period_txt}</span>'
        f'<span style="background:#F8FAFC;color:#0F172A;border:1px solid #CBD5E1;border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600;">📐 {t("ai_plan_crs")}: {ai_crs_show} · {pl["resolution_m"]} m</span>'
        f'<span style="background:#FFFFFF;color:#475569;border:1px solid #CBD5E1;border-radius:999px;padding:3px 10px;font-size:11.5px;">{eng_lbl}</span>'
        f'</div>'
    )
    st.markdown(plan_pills_html, unsafe_allow_html=True)
    if fb_note:
        st.caption(f"ℹ️ {fb_note}")

    basemap_now = str(st.session_state.get("map_basemap", "satellite"))

    # 2. Task-specific KPI Row + Static-HTML Interactive Leaflet Map (or Plotly Chart)
    if ai_kind_show == "disaster":
        d_obj = ai_res_show["disaster_result"]
        d_sum = d_obj["summary"]
        dt_k = str(d_obj["disaster_type"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t("dis_kpi_dates"), f"{d_obj['date_pre']} → {d_obj['date_event']}")
        m2.metric(
            t("dis_kpi_affected_ha"),
            f"{d_sum['affected_area_ha']:,.2f} ha",
            delta=f"{d_sum['affected_area_km2']:.3f} km²",
            delta_color="off",
        )
        m3.metric(t("dis_kpi_affected_pct"), f"{d_sum['affected_pct']:.2f} %")
        m4.metric(t("ai_plan_crs"), ai_crs_show)

        ai_op_d = st.slider(
            t("opacity_layer_label"),
            0,
            100,
            85,
            5,
            key=f"ai_chat_dis_op_{result_id}",
        )
        lbl_k_ai = "en" if lang == "en" else "sl"
        sw_ai = [
            (str(sp[lbl_k_ai]), str(sp["hex"]))
            for sp in get_class_specs(dt_k)
            if sp["affected"] or (dt_k == DISASTER_FLOOD and sp["code"] == 1)
        ]
        st.markdown(
            build_horizontal_colorbar_html(
                "Greys",
                0.0,
                1.0,
                label=t(DISASTER_TYPE_I18N[dt_k]),
                discrete_items=sw_ai,
            ),
            unsafe_allow_html=True,
        )

        # Ensure BOTH toggleable Satellite Scene layer AND Disaster Mask layer are present
        overlay_layers: list[dict[str, Any]] = []
        sat_uri = ai_res_show.get("sat_overlay_uri") or d_obj.get("overlay_uri_post")
        sat_bnds = ai_res_show.get("sat_overlay_bounds") or d_obj.get("overlay_bounds_post")
        if sat_uri and sat_bnds:
            sat_lbl = (
                t("ai_layer_sat_sar", date=str(d_obj["date_event"]))
                if dt_k == DISASTER_FLOOD
                else f"🛰️ {ai_sensor_short} RGB ({d_obj['date_event']})"
            )
            overlay_layers.append(
                {
                    "name": sat_lbl,
                    "image": sat_uri,
                    "bounds": sat_bnds,
                    "opacity": 1.0,
                    "show": True,
                    "zindex": 12,
                }
            )

        mask_uri = ai_res_show.get("analysis_overlay_uri")
        mask_bnds = ai_res_show.get("analysis_overlay_bounds")
        if not (mask_uri and mask_bnds):
            mask_uri, mask_bnds = raster_to_map_overlay_png(
                d_obj["class_da"],
                src_epsg=ai_crs_show,
                is_rgb=False,
                discrete_rgba_map=build_slippy_rgba_map(dt_k),
            )
            ai_res_show["analysis_overlay_uri"] = mask_uri
            ai_res_show["analysis_overlay_bounds"] = mask_bnds

        mask_lbl_key = (
            "ai_layer_flood_mask"
            if dt_k == DISASTER_FLOOD
            else ("ai_layer_fire_mask" if dt_k == DISASTER_WILDFIRE else "ai_layer_forest_mask")
        )
        overlay_layers.append(
            {
                "name": t(mask_lbl_key),
                "image": mask_uri,
                "bounds": mask_bnds,
                "opacity": float(ai_op_d) / 100.0,
                "show": True,
                "zindex": 18,
            }
        )

        slippy_ai = build_satellite_slippy_map(
            center_lat=float(ai_res_show["lat"]),
            center_lon=float(ai_res_show["lon"]),
            zoom=None,
            basemap_mode=basemap_now,
            overlay_layers=overlay_layers,
            aoi_bbox_4326=ai_res_show["bbox_4326"],
            aoi_outline_label=t("slippy_aoi_outline"),
            lang=lang,
        )
        components.html(slippy_ai.get_root().render(), height=475, scrolling=False)
        st.caption(t("ai_layer_toggle_hint"))

    elif ai_kind_show == "spectral_index":
        st_ai = ai_res_show["stats"]
        idx_n = str(pl["index_name"])
        is_sar_ai = pl["sensor"] == SENSOR_S1
        spec_ai = (
            SAR_METRICS.get(idx_n, SAR_METRICS["VV"])
            if is_sar_ai
            else SPECTRAL_INDICES.get(idx_n, SPECTRAL_INDICES["NDVI"])
        )
        units_s = str(spec_ai.get("units") or "")
        unit_suf = f" {units_s}" if units_s else ""
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t("col_date"), str(ai_res_show["date_str"]))
        m2.metric("Povprečje" if lang == "sl" else "Mean", f"{st_ai['mean']:.3f}{unit_suf}")
        m3.metric(t("m_valid_ha"), f"{st_ai['valid_area_ha']:,.2f} ha")
        m4.metric(t("ai_plan_crs"), ai_crs_show)

        # NDSI Snow / Glacier + DEM Snowline KPIs when available
        snow_st = ai_res_show.get("snow_stats")
        if isinstance(snow_st, dict):
            s1, s2, s3 = st.columns(3)
            s1.metric(
                "❄️ Površina snega/ledu" if lang == "sl" else "❄️ Snow / Ice Area",
                f"{float(snow_st.get('snow_area_ha', 0.0)):,.1f} ha",
                delta=f"{float(snow_st.get('snow_area_km2', 0.0)):.2f} km²",
                delta_color="off",
            )
            s2.metric(
                "❄️ Delež AOI" if lang == "sl" else "❄️ Snow Cover %",
                f"{float(snow_st.get('snow_pct', 0.0)):.1f} %",
            )
            sl_m = snow_st.get("snowline_p10_m")
            s3.metric(
                "⛰️ Snežna meja (p10 DEM)" if lang == "sl" else "⛰️ Snowline (p10 DEM)",
                f"{float(sl_m):,.0f} m" if sl_m is not None else "—",
            )

        ai_op_i = st.slider(
            t("opacity_index_label"),
            0,
            100,
            85,
            5,
            key=f"ai_chat_idx_op_{result_id}",
        )
        st.markdown(
            build_horizontal_colorbar_html(
                spec_ai["cmap"],
                float(spec_ai["vmin"]),
                float(spec_ai["vmax"]),
                label=idx_n,
                units=units_s,
            ),
            unsafe_allow_html=True,
        )

        # Dual toggleable layers: True Color RGB / SAR scene + Computed Spectral Index
        overlay_layers = []
        sat_uri = ai_res_show.get("sat_overlay_uri")
        sat_bnds = ai_res_show.get("sat_overlay_bounds")
        if sat_uri and sat_bnds:
            sat_lbl = (
                t("ai_layer_sat_sar", date=str(ai_res_show["date_str"]))
                if is_sar_ai
                else t("ai_layer_sat_rgb", date=str(ai_res_show["date_str"]))
            )
            overlay_layers.append(
                {
                    "name": sat_lbl,
                    "image": sat_uri,
                    "bounds": sat_bnds,
                    "opacity": 1.0,
                    "show": True,
                    "zindex": 12,
                }
            )

        idx_uri = ai_res_show.get("analysis_overlay_uri")
        idx_bnds = ai_res_show.get("analysis_overlay_bounds")
        if not (idx_uri and idx_bnds):
            idx_uri, idx_bnds = raster_to_map_overlay_png(
                ai_res_show["index_da"],
                src_epsg=ai_crs_show,
                is_rgb=False,
                cmap_name=spec_ai["cmap"],
                vmin=float(spec_ai["vmin"]),
                vmax=float(spec_ai["vmax"]),
            )
            ai_res_show["analysis_overlay_uri"] = idx_uri
            ai_res_show["analysis_overlay_bounds"] = idx_bnds

        overlay_layers.append(
            {
                "name": f"📊 {idx_n} ({ai_res_show['date_str']})",
                "image": idx_uri,
                "bounds": idx_bnds,
                "opacity": float(ai_op_i) / 100.0,
                "show": True,
                "zindex": 18,
            }
        )
        for ex_i, ex_ov in enumerate(ai_res_show.get("extra_overlay_layers") or []):
            if isinstance(ex_ov, dict) and ex_ov.get("image") and ex_ov.get("bounds"):
                overlay_layers.append(
                    {
                        "name": str(ex_ov.get("name") or ex_ov.get("code") or f"Layer {ex_i + 1}"),
                        "image": ex_ov["image"],
                        "bounds": ex_ov["bounds"],
                        "opacity": float(ai_op_i) / 100.0,
                        "show": False,
                        "zindex": 15 + ex_i,
                    }
                )

        slippy_ai = build_satellite_slippy_map(
            center_lat=float(ai_res_show["lat"]),
            center_lon=float(ai_res_show["lon"]),
            zoom=None,
            basemap_mode=basemap_now,
            overlay_layers=overlay_layers,
            aoi_bbox_4326=ai_res_show["bbox_4326"],
            aoi_outline_label=t("slippy_aoi_outline"),
            lang=lang,
        )
        components.html(slippy_ai.get_root().render(), height=475, scrolling=False)
        st.caption(t("ai_layer_toggle_hint"))

    elif ai_kind_show == "scene_rgb":
        m1, m2, m3 = st.columns(3)
        m1.metric(t("col_date"), str(ai_res_show["date_str"]))
        m2.metric(t("col_tile"), str(ai_res_show.get("tile", "")))
        m3.metric(t("ai_plan_crs"), ai_crs_show)

        ai_op_r = st.slider(
            t("opacity_layer_label"),
            0,
            100,
            100,
            5,
            key=f"ai_chat_rgb_op_{result_id}",
        )
        sat_uri = ai_res_show.get("sat_overlay_uri")
        sat_bnds = ai_res_show.get("sat_overlay_bounds")
        if not (sat_uri and sat_bnds):
            sat_uri, sat_bnds = raster_to_map_overlay_png(
                ai_res_show["scene_ds"],
                src_epsg=ai_crs_show,
                is_rgb=True,
                bands=ai_res_show.get("bands", ["B04", "B03", "B02"]),
            )
            ai_res_show["sat_overlay_uri"] = sat_uri
            ai_res_show["sat_overlay_bounds"] = sat_bnds

        slippy_ai = build_satellite_slippy_map(
            center_lat=float(ai_res_show["lat"]),
            center_lon=float(ai_res_show["lon"]),
            zoom=None,
            basemap_mode=basemap_now,
            overlay_layers=[
                {
                    "name": t("ai_layer_sat_rgb", date=str(ai_res_show["date_str"])),
                    "image": sat_uri,
                    "bounds": sat_bnds,
                    "opacity": float(ai_op_r) / 100.0,
                    "show": True,
                    "zindex": 15,
                }
            ],
            aoi_bbox_4326=ai_res_show["bbox_4326"],
            aoi_outline_label=t("slippy_aoi_outline"),
            lang=lang,
        )
        components.html(slippy_ai.get_root().render(), height=475, scrolling=False)

    elif ai_kind_show == "cloud_free_composite":
        cm = ai_res_show.get("composite_meta") or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Št. posnetkov" if lang == "sl" else "Scenes Used", f"{int(cm.get('scenes_used', 0))}")
        m2.metric(t("m_valid_ha"), f"{float(cm.get('valid_area_ha', 0.0)):,.1f} ha")
        m3.metric(
            "Mediana opazovanj/px" if lang == "sl" else "Median Obs/px",
            f"{float(cm.get('median_obs_per_px', 0.0)):.1f}",
        )
        m4.metric(t("ai_plan_crs"), ai_crs_show)

        ai_op_c = st.slider(
            t("opacity_layer_label"),
            0,
            100,
            100,
            5,
            key=f"ai_chat_comp_op_{result_id}",
        )
        sat_uri = ai_res_show.get("sat_overlay_uri")
        sat_bnds = ai_res_show.get("sat_overlay_bounds")
        if sat_uri and sat_bnds:
            slippy_ai = build_satellite_slippy_map(
                center_lat=float(ai_res_show["lat"]),
                center_lon=float(ai_res_show["lon"]),
                zoom=None,
                basemap_mode=basemap_now,
                overlay_layers=[
                    {
                        "name": t("ai_layer_comp_rgb"),
                        "image": sat_uri,
                        "bounds": sat_bnds,
                        "opacity": float(ai_op_c) / 100.0,
                        "show": True,
                        "zindex": 15,
                    }
                ],
                aoi_bbox_4326=ai_res_show["bbox_4326"],
                aoi_outline_label=t("slippy_aoi_outline"),
                lang=lang,
            )
            components.html(slippy_ai.get_root().render(), height=475, scrolling=False)

    elif ai_kind_show == "change_comparison":
        comp_r = ai_res_show.get("comparison_result") or {}
        cs = ai_res_show.get("change_summary") or {}
        c_dates = comp_r.get("dates") or ["t1", "t2"]
        c_uris = comp_r.get("overlay_uris") or []
        c_bnds = comp_r.get("overlay_bounds") or []
        idx_lbl = str(pl.get("index_name", "NDVI"))

        for rep in comp_r.get("replacements") or []:
            st.info(
                t(
                    "comp3_replaced_badge",
                    req=rep.get("requested_date", ""),
                    orig_pct=float(rep.get("original_clear_pct", 0.0)),
                    act=rep.get("actual_date", ""),
                    act_pct=float(rep.get("actual_clear_pct", 0.0)),
                )
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Datuma" if lang == "sl" else "Dates", f"{c_dates[0]} ↔ {c_dates[-1]}")
        m2.metric(f"Δ {idx_lbl}", f"{float(cs.get('mean', 0.0)):+.3f}" if cs else "—")
        m3.metric(
            "Porast (+)" if lang == "sl" else "Increase (+)",
            f"{float(cs.get('pos_area_ha', 0.0)):,.1f} ha" if cs else "—",
        )
        m4.metric(
            "Upad (-)" if lang == "sl" else "Decrease (-)",
            f"{float(cs.get('neg_area_ha', 0.0)):,.1f} ha" if cs else "—",
        )

        ai_op_chg = st.slider(
            t("opacity_layer_label"),
            0,
            100,
            95,
            5,
            key=f"ai_chat_chg_op_{result_id}",
        )
        if len(c_uris) >= 2 and len(c_bnds) >= 2:
            swipe_html = build_zoomable_swipe_map(
                overlay_uri_t1=c_uris[0],
                bounds_t1=c_bnds[0],
                overlay_uri_t2=c_uris[-1],
                bounds_t2=c_bnds[-1],
                label_t1=f"t1: {c_dates[0]} ({idx_lbl})",
                label_t2=f"t2: {c_dates[-1]} ({idx_lbl})",
                center_lat=float(ai_res_show["lat"]),
                center_lon=float(ai_res_show["lon"]),
                basemap_mode=basemap_now,
                aoi_bbox_4326=ai_res_show.get("bbox_4326"),
                aoi_bbox_3794=ai_res_show.get("bbox_3794"),
                aoi_outline_label=t("slippy_aoi_outline"),
                height=470,
                opacity=float(ai_op_chg) / 100.0,
                lang=lang,
            )
            components.html(swipe_html, height=480, scrolling=False)

    elif ai_kind_show == "terrain_dem":
        ds_t = ai_res_show.get("dem_stats") or {}
        min_h = float(ds_t.get("min_h", 0.0))
        max_h = float(ds_t.get("max_h", 0.0))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Višina (min–max)" if lang == "sl" else "Elevation (min–max)",
            f"{min_h:,.0f}–{max_h:,.0f} m",
        )
        m2.metric(
            "Povp. naklon" if lang == "sl" else "Mean Slope",
            f"{float(ds_t.get('mean_slope_deg', 0.0)):.1f}°",
        )
        m3.metric(
            "Strmina > 20°" if lang == "sl" else "Steep > 20°",
            f"{float(ds_t.get('steep_20_ha', 0.0)):,.1f} ha",
        )
        m4.metric(t("ai_plan_crs"), ai_crs_show)

        ai_op_dem = st.slider(
            t("opacity_layer_label"),
            0,
            100,
            85,
            5,
            key=f"ai_chat_dem_op_{result_id}",
        )
        st.markdown(
            build_horizontal_colorbar_html(
                "terrain",
                min_h,
                max(max_h, min_h + 10.0),
                label="Nadmorska višina" if lang == "sl" else "Elevation",
                units="m",
            ),
            unsafe_allow_html=True,
        )

        dem_layers: list[dict[str, Any]] = []
        if ai_res_show.get("dem_hill_uri") and ai_res_show.get("dem_hill_bounds"):
            dem_layers.append(
                {
                    "name": t("ai_layer_dem_hillshade"),
                    "image": ai_res_show["dem_hill_uri"],
                    "bounds": ai_res_show["dem_hill_bounds"],
                    "opacity": 0.90,
                    "show": True,
                    "zindex": 11,
                }
            )
        if ai_res_show.get("dem_elev_uri") and ai_res_show.get("dem_elev_bounds"):
            dem_layers.append(
                {
                    "name": t("ai_layer_dem_elev"),
                    "image": ai_res_show["dem_elev_uri"],
                    "bounds": ai_res_show["dem_elev_bounds"],
                    "opacity": float(ai_op_dem) / 100.0,
                    "show": True,
                    "zindex": 16,
                }
            )
        if ai_res_show.get("dem_slope_uri") and ai_res_show.get("dem_slope_bounds"):
            dem_layers.append(
                {
                    "name": t("ai_layer_dem_slope"),
                    "image": ai_res_show["dem_slope_uri"],
                    "bounds": ai_res_show["dem_slope_bounds"],
                    "opacity": float(ai_op_dem) / 100.0,
                    "show": False,
                    "zindex": 20,
                }
            )
        if dem_layers:
            slippy_ai = build_satellite_slippy_map(
                center_lat=float(ai_res_show["lat"]),
                center_lon=float(ai_res_show["lon"]),
                zoom=None,
                basemap_mode=basemap_now,
                overlay_layers=dem_layers,
                aoi_bbox_4326=ai_res_show["bbox_4326"],
                aoi_outline_label=t("slippy_aoi_outline"),
                lang=lang,
            )
            components.html(slippy_ai.get_root().render(), height=475, scrolling=False)
            st.caption(t("ai_layer_toggle_hint"))

        prof_df = ai_res_show.get("profile_df")
        if (
            isinstance(prof_df, pd.DataFrame)
            and not prof_df.empty
            and "distance_m" in prof_df.columns
            and "elevation_m" in prof_df.columns
        ):
            fig_prof = go.Figure()
            fig_prof.add_trace(
                go.Scatter(
                    x=prof_df["distance_m"],
                    y=prof_df["elevation_m"],
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(15, 118, 110, 0.16)",
                    line=dict(color="#0F766E", width=2.4),
                    name="DEM [m]",
                )
            )
            fig_prof.update_layout(
                title=t("ai_profile_title", place=str(pl.get("place_name", ""))),
                xaxis_title="Razdalja [m]" if lang == "sl" else "Distance [m]",
                yaxis_title="Nadmorska višina [m]" if lang == "sl" else "Elevation [m a.s.l.]",
                height=310,
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#0F172A"),
                margin=dict(l=48, r=20, t=42, b=42),
            )
            st.plotly_chart(fig_prof, width="stretch", key=f"ai_chat_dem_prof_{result_id}")

    elif ai_kind_show == "gnss_plan":
        g_res = ai_res_show.get("gnss_result") or {}
        g_sum = g_res.get("summary") or {}
        tl_df = g_res.get("timeline_df")
        opt_wins = g_sum.get("optimal_windows") or []
        best_win = str(opt_wins[0]) if opt_wins else "—"
        min_pdop = g_sum.get("min_pdop_real")
        min_pdop_s = (
            f"{float(min_pdop):.2f}"
            if (min_pdop is not None and np.isfinite(float(min_pdop)))
            else "—"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Vidni sateliti (real/odprto)" if lang == "sl" else "Visible Sats (real/open)",
            f"{float(g_sum.get('avg_vis_real', 0.0)):.1f} / {float(g_sum.get('avg_vis_open', 0.0)):.1f}",
        )
        m2.metric("Min PDOP", min_pdop_s)
        m3.metric(
            "Zastrtost neba" if lang == "sl" else "Sky Obstruction",
            f"{float(g_sum.get('sky_blocked_pct', 0.0)):.1f} %",
        )
        m4.metric("Optimalno okno" if lang == "sl" else "Best Window", best_win)

        if isinstance(tl_df, pd.DataFrame) and not tl_df.empty:
            fig_g = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.10,
                subplot_titles=(
                    "Vidnost satelitov (24 h)" if lang == "sl" else "Satellite Visibility (24 h)",
                    "PDOP (realne ovire vs. odprto nebo)" if lang == "sl" else "PDOP (Real Obstacles vs. Open Sky)",
                ),
            )
            fig_g.add_trace(
                go.Scatter(
                    x=tl_df["time_str"],
                    y=tl_df["visible_open"],
                    mode="lines",
                    name="Odprto nebo" if lang == "sl" else "Open Sky",
                    line=dict(color="#94A3B8", width=1.6, dash="dot"),
                ),
                row=1,
                col=1,
            )
            fig_g.add_trace(
                go.Scatter(
                    x=tl_df["time_str"],
                    y=tl_df["visible_real"],
                    mode="lines",
                    name="Realno (DEM+S2)" if lang == "sl" else "Real (DEM+S2)",
                    line=dict(color="#0F766E", width=2.4),
                    fill="tozeroy",
                    fillcolor="rgba(15, 118, 110, 0.14)",
                ),
                row=1,
                col=1,
            )
            fig_g.add_trace(
                go.Scatter(
                    x=tl_df["time_str"],
                    y=tl_df["pdop_open"],
                    mode="lines",
                    name="PDOP (odprto)" if lang == "sl" else "PDOP (open)",
                    line=dict(color="#94A3B8", width=1.4, dash="dot"),
                ),
                row=2,
                col=1,
            )
            fig_g.add_trace(
                go.Scatter(
                    x=tl_df["time_str"],
                    y=tl_df["pdop_real"],
                    mode="lines",
                    name="PDOP (realno)" if lang == "sl" else "PDOP (real)",
                    line=dict(color="#DC2626", width=2.2),
                ),
                row=2,
                col=1,
            )
            fig_g.add_hline(y=2.5, line_dash="dash", line_color="#0F766E", row=2, col=1)
            fig_g.update_layout(
                height=460,
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#0F172A"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.20, xanchor="center", x=0.5),
                margin=dict(l=48, r=20, t=42, b=52),
            )
            st.plotly_chart(fig_g, width="stretch", key=f"ai_chat_gnss_plotly_{result_id}")

    elif ai_kind_show in ("time_series", "time_series_and_comparison"):
        ts_ai: pd.DataFrame = ai_res_show["ts_df"]
        if isinstance(ts_ai, pd.DataFrame) and "date_str" in ts_ai.columns and not ts_ai.empty:
            ts_ai = ts_ai.sort_values("date_str", ascending=True).reset_index(drop=True)
        idx_lbl_ts = str(pl.get("index_name", "NDVI"))
        frames_list: list[dict[str, Any]] = sorted(
            list(ai_res_show.get("frames") or []),
            key=lambda f: str(f.get("date") or ""),
        )
        comp_rgb = ai_res_show.get("comparison_rgb")
        comp_idx = ai_res_show.get("comparison_index")
        cs_ts = ai_res_show.get("change_summary") or {}
        has_multiframe = len(frames_list) >= 2
        has_legacy_swipe = (
            (isinstance(comp_rgb, dict) and len(comp_rgb.get("overlay_uris") or []) >= 2)
            or (isinstance(comp_idx, dict) and len(comp_idx.get("overlay_uris") or []) >= 2)
        )

        # Compute linear trend slope for KPI card
        trend_str = "—"
        if len(ts_ai) >= 3:
            x_ord_k = np.arange(len(ts_ai), dtype=np.float64)
            sl_k, _ = np.polyfit(x_ord_k, ts_ai["mean"].to_numpy(dtype=np.float64), 1)
            trend_str = f"{sl_k:+.4f}/obs"
        elif len(ts_ai) >= 2:
            d_val = float(ts_ai["mean"].iloc[-1]) - float(ts_ai["mean"].iloc[0])
            trend_str = f"{d_val:+.3f}"

        delta_str = (
            f"{float(cs_ts['mean']):+.3f}"
            if (isinstance(cs_ts, dict) and cs_ts.get("mean") is not None)
            else f"{ts_ai['mean'].mean():.3f}"
        )

        d_start_s = str(ts_ai["date_str"].iloc[0])
        d_end_s = str(ts_ai["date_str"].iloc[-1])
        period_kpi = (
            f"{d_start_s[:4]} → {d_end_s[:4]}"
            if (len(d_start_s) >= 4 and len(d_end_s) >= 4 and d_start_s[:4] != d_end_s[:4])
            else f"{d_start_s} → {d_end_s}"
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Št. datumov" if lang == "sl" else "Valid Dates", f"{len(ts_ai)}")
        m2.metric("Obdobje t1 → t2" if lang == "sl" else "Period t1 → t2", period_kpi)
        m3.metric("Trend", trend_str)
        m4.metric(
            (f"Δ {idx_lbl_ts}" if cs_ts else ("Povprečje" if lang == "sl" else "Mean")),
            delta_str,
        )
        m5.metric(t("ai_plan_crs"), format_crs_label(ai_crs_show, short=True))

        def _draw_ts_plotly_and_table() -> None:
            fig_ai_ts = go.Figure()
            fig_ai_ts.add_trace(
                go.Scatter(
                    x=ts_ai["date_str"],
                    y=ts_ai["mean"],
                    mode="lines+markers",
                    name=idx_lbl_ts,
                    line=dict(color="#0F766E", width=2.5),
                    marker=dict(size=7, color="#0F766E"),
                )
            )
            if len(ts_ai) >= 3:
                x_ord = np.arange(len(ts_ai), dtype=np.float64)
                slope, intercept = np.polyfit(x_ord, ts_ai["mean"].to_numpy(dtype=np.float64), 1)
                fig_ai_ts.add_trace(
                    go.Scatter(
                        x=ts_ai["date_str"],
                        y=slope * x_ord + intercept,
                        mode="lines",
                        name="Trend",
                        line=dict(color="#DC2626", width=1.8, dash="dash"),
                    )
                )
            fig_ai_ts.update_layout(
                title=t(
                    "ai_ts_chart_title",
                    idx=idx_lbl_ts,
                    place=pl["place_name"],
                    sensor=ai_sensor_short,
                    crs=ai_crs_show,
                ),
                xaxis_title=t("col_date"),
                yaxis_title=idx_lbl_ts,
                height=390,
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#0F172A"),
            )
            st.plotly_chart(fig_ai_ts, width="stretch", key=f"ai_chat_ts_plotly_{result_id}")
            tbl_cols = [c for c in ("date_str", "platform", "mean", "median", "std", "valid_pct") if c in ts_ai.columns]
            if tbl_cols:
                st.dataframe(ts_ai[tbl_cols], width="stretch", hide_index=True, height=min(240, 42 + 35 * len(ts_ai)))
            if ai_res_show.get("csv_bytes"):
                st.download_button(
                    label=t("ai_mf_dl_csv", lang=lang),
                    data=ai_res_show["csv_bytes"],
                    file_name=ai_res_show["csv_filename"],
                    mime="text/csv",
                    width="stretch",
                    key=f"ai_chat_dl_csv_tab_{result_id}",
                )

        if has_multiframe:
            t_sw, t_delta, t_tl, t_strip, t_ts = st.tabs(
                [
                    t("ai_mf_tab_swipe", lang=lang),
                    t("ai_mf_tab_delta", lang=lang),
                    t("ai_mf_tab_timelapse", lang=lang),
                    t("ai_mf_tab_strip", lang=lang),
                    t("ai_mf_tab_ts", lang=lang),
                ]
            )

            frame_dates = [str(f["date"]) for f in frames_list]
            n_fr = len(frames_list)
            has_rgb_frames = any(bool(f.get("rgb_overlay_uri")) for f in frames_list)
            has_idx_frames = any(bool(f.get("index_overlay_uri")) for f in frames_list)
            extra_codes_avail: list[str] = []
            for fr_chk in frames_list:
                for ek in (fr_chk.get("extra_overlays") or {}).keys():
                    if ek not in extra_codes_avail:
                        extra_codes_avail.append(str(ek))

            def _layer_pill_label(code: str) -> str:
                if code == "rgb":
                    return t("ai_mf_layer_rgb", lang=lang)
                if code == "index":
                    ico = "❄️" if idx_lbl_ts == "NDSI" else ("💧" if idx_lbl_ts in ("NDWI", "MNDWI") else "📊")
                    return f"{ico} {idx_lbl_ts}"
                ico_e = "💧" if code in ("NDWI", "MNDWI") else ("🔥" if code == "SWIR" else "🌿")
                return f"{ico_e} {code}"

            # --- TAB 1: Compare Any Two Dates (t1 ↔ t2) with Multi-Index Switcher ---
            with t_sw:
                dc1, dc2 = st.columns(2, gap="small")
                with dc1:
                    idx_left = st.selectbox(
                        t("ai_mf_date_left", lang=lang),
                        options=list(range(n_fr)),
                        index=0,
                        format_func=lambda i: (
                            f"{frames_list[i]['date']} ({idx_lbl_ts}={frames_list[i]['index_mean']:.2f})"
                            if frames_list[i].get("index_mean") is not None
                            else str(frames_list[i]["date"])
                        ),
                        key=f"ai_mf_left_idx_{result_id}",
                    )
                with dc2:
                    idx_right = st.selectbox(
                        t("ai_mf_date_right", lang=lang),
                        options=list(range(n_fr)),
                        index=n_fr - 1,
                        format_func=lambda i: (
                            f"{frames_list[i]['date']} ({idx_lbl_ts}={frames_list[i]['index_mean']:.2f})"
                            if frames_list[i].get("index_mean") is not None
                            else str(frames_list[i]["date"])
                        ),
                        key=f"ai_mf_right_idx_{result_id}",
                    )

                avail_sw_modes: list[str] = []
                if has_rgb_frames:
                    avail_sw_modes.append("rgb")
                if has_idx_frames:
                    avail_sw_modes.append("index")
                avail_sw_modes.extend(extra_codes_avail)

                ctrl_c1, ctrl_c2 = st.columns([1.35, 0.95], gap="small")
                with ctrl_c1:
                    if len(avail_sw_modes) > 1:
                        sw_mode = _touch_selector(
                            t("ai_mf_layer_label", lang=lang),
                            options=avail_sw_modes,
                            default=avail_sw_modes[0],
                            format_func=_layer_pill_label,
                            key=f"ai_chat_ts_sw_mode_{result_id}",
                            use_pills=True,
                        )
                    else:
                        sw_mode = avail_sw_modes[0]
                with ctrl_c2:
                    ai_op_ts_sw = st.slider(
                        t("opacity_layer_label"),
                        0,
                        100,
                        95,
                        5,
                        key=f"ai_chat_ts_sw_op_{result_id}",
                    )

                fr_l = frames_list[int(idx_left)]
                fr_r = frames_list[int(idx_right)]
                if sw_mode == "rgb":
                    uri_l = fr_l.get("rgb_overlay_uri") or fr_l.get("index_overlay_uri")
                    uri_r = fr_r.get("rgb_overlay_uri") or fr_r.get("index_overlay_uri")
                    sw_tag = "RGB"
                elif sw_mode == "index":
                    uri_l = fr_l.get("index_overlay_uri") or fr_l.get("rgb_overlay_uri")
                    uri_r = fr_r.get("index_overlay_uri") or fr_r.get("rgb_overlay_uri")
                    sw_tag = idx_lbl_ts
                else:
                    uri_l = (fr_l.get("extra_overlays") or {}).get(sw_mode) or fr_l.get("index_overlay_uri")
                    uri_r = (fr_r.get("extra_overlays") or {}).get(sw_mode) or fr_r.get("index_overlay_uri")
                    sw_tag = str(sw_mode)
                bnds_l = fr_l.get("bounds_4326")
                bnds_r = fr_r.get("bounds_4326")

                if uri_l and uri_r and bnds_l and bnds_r:
                    swipe_html = build_zoomable_swipe_map(
                        overlay_uri_t1=uri_l,
                        bounds_t1=bnds_l,
                        overlay_uri_t2=uri_r,
                        bounds_t2=bnds_r,
                        label_t1=f"t1: {fr_l['date']} ({sw_tag})",
                        label_t2=f"t2: {fr_r['date']} ({sw_tag})",
                        center_lat=float(ai_res_show["lat"]),
                        center_lon=float(ai_res_show["lon"]),
                        basemap_mode=basemap_now,
                        aoi_bbox_4326=ai_res_show.get("bbox_4326"),
                        aoi_bbox_3794=ai_res_show.get("bbox_3794"),
                        aoi_outline_label=t("slippy_aoi_outline"),
                        height=470,
                        opacity=float(ai_op_ts_sw) / 100.0,
                        lang=lang,
                    )
                    components.html(swipe_html, height=480, scrolling=False)
                    st.caption(t("swipe_zoom_hint"))

                try:
                    d_span_days = abs(
                        (date.fromisoformat(fr_r["date"][:10]) - date.fromisoformat(fr_l["date"][:10])).days
                    )
                    span_txt = (
                        f"{d_span_days / 365.25:.1f} {'let' if lang == 'sl' else 'yr'} ({d_span_days} {'dni' if lang == 'sl' else 'days'})"
                        if d_span_days >= 365
                        else f"{d_span_days} {'dni' if lang == 'sl' else 'days'}"
                    )
                except Exception:
                    span_txt = "—"

                delta_pair_txt = "—"
                if fr_l.get("index_mean") is not None and fr_r.get("index_mean") is not None:
                    delta_pair_txt = f"{float(fr_r['index_mean']) - float(fr_l['index_mean']):+.3f}"

                st.markdown(
                    f"**Δ {idx_lbl_ts} (t2 − t1):** `{delta_pair_txt}` · "
                    f"**{'Razmik' if lang == 'sl' else 'Span'}:** `{span_txt}` · "
                    f"`{fr_l['date']}` ↔ `{fr_r['date']}`"
                )

            # --- TAB 2: Spatial Change Map (Δ = t2 - t1) & All Toggleable Layers ---
            with t_delta:
                ai_op_delta = st.slider(
                    t("opacity_layer_label"),
                    0,
                    100,
                    90,
                    5,
                    key=f"ai_mf_delta_op_{result_id}",
                )
                d_cmap = str(ai_res_show.get("delta_cmap") or ("RdBu" if idx_lbl_ts in ("NDSI", "NDWI", "MNDWI") else "RdYlGn"))
                d_vmin = float(ai_res_show.get("delta_vmin", -0.5))
                d_vmax = float(ai_res_show.get("delta_vmax", 0.5))
                st.markdown(
                    build_horizontal_colorbar_html(
                        d_cmap,
                        d_vmin,
                        d_vmax,
                        label=f"Δ {idx_lbl_ts} (t2 − t1)",
                        units="°C" if idx_lbl_ts == "LST" else "",
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(
                    t("ai_mf_delta_legend_glacier", lang=lang)
                    if idx_lbl_ts == "NDSI"
                    else t("ai_mf_delta_legend_generic", lang=lang)
                )

                f_first, f_latest = frames_list[0], frames_list[-1]
                delta_layers: list[dict[str, Any]] = []
                if f_first.get("rgb_overlay_uri") and f_first.get("bounds_4326"):
                    delta_layers.append(
                        {
                            "name": f"🛰️ RGB (t1: {f_first['date']})",
                            "image": f_first["rgb_overlay_uri"],
                            "bounds": f_first["bounds_4326"],
                            "opacity": 1.0,
                            "show": False,
                            "zindex": 11,
                        }
                    )
                if f_latest.get("rgb_overlay_uri") and f_latest.get("bounds_4326"):
                    delta_layers.append(
                        {
                            "name": f"🛰️ RGB (t2: {f_latest['date']})",
                            "image": f_latest["rgb_overlay_uri"],
                            "bounds": f_latest["bounds_4326"],
                            "opacity": 1.0,
                            "show": True,
                            "zindex": 12,
                        }
                    )
                if f_latest.get("index_overlay_uri") and f_latest.get("bounds_4326"):
                    delta_layers.append(
                        {
                            "name": f"📊 {idx_lbl_ts} (t2: {f_latest['date']})",
                            "image": f_latest["index_overlay_uri"],
                            "bounds": f_latest["bounds_4326"],
                            "opacity": float(ai_op_delta) / 100.0,
                            "show": False,
                            "zindex": 14,
                        }
                    )
                for ex_i, ex_c in enumerate(extra_codes_avail):
                    ex_u = (f_latest.get("extra_overlays") or {}).get(ex_c)
                    if ex_u and f_latest.get("bounds_4326"):
                        delta_layers.append(
                            {
                                "name": f"{_layer_pill_label(ex_c)} (t2: {f_latest['date']})",
                                "image": ex_u,
                                "bounds": f_latest["bounds_4326"],
                                "opacity": float(ai_op_delta) / 100.0,
                                "show": False,
                                "zindex": 15 + ex_i,
                            }
                        )
                if ai_res_show.get("delta_overlay_uri") and ai_res_show.get("delta_overlay_bounds"):
                    delta_layers.append(
                        {
                            "name": t("ai_mf_delta_layer_name", lang=lang, idx=idx_lbl_ts),
                            "image": ai_res_show["delta_overlay_uri"],
                            "bounds": ai_res_show["delta_overlay_bounds"],
                            "opacity": float(ai_op_delta) / 100.0,
                            "show": True,
                            "zindex": 20,
                        }
                    )

                if delta_layers:
                    slippy_delta = build_satellite_slippy_map(
                        center_lat=float(ai_res_show["lat"]),
                        center_lon=float(ai_res_show["lon"]),
                        zoom=None,
                        basemap_mode=basemap_now,
                        overlay_layers=delta_layers,
                        aoi_bbox_4326=ai_res_show.get("bbox_4326"),
                        aoi_outline_label=t("slippy_aoi_outline"),
                        lang=lang,
                    )
                    components.html(slippy_delta.get_root().render(), height=475, scrolling=False)
                    st.caption(t("ai_layer_toggle_hint"))

            # --- TAB 3: Visual Timelapse (Animated GIF directly in Chat) ---
            with t_tl:
                rgb_pngs_all = [f["rgb_png_bytes"] for f in frames_list if f.get("rgb_png_bytes")]
                idx_pngs_all = [f["index_png_bytes"] for f in frames_list if f.get("index_png_bytes")]
                tl_modes: list[str] = []
                if len(rgb_pngs_all) >= 2:
                    tl_modes.append("rgb")
                if len(idx_pngs_all) >= 2:
                    tl_modes.append("index")

                if tl_modes:
                    tl_c1, tl_c2 = st.columns([1.0, 1.3], gap="small")
                    with tl_c1:
                        if len(tl_modes) > 1:
                            tl_mode = _touch_selector(
                                t("ai_mf_layer_label", lang=lang),
                                options=tl_modes,
                                default=tl_modes[0],
                                format_func=lambda m: (
                                    t("ai_mf_tl_rgb", lang=lang) if m == "rgb" else t("ai_mf_tl_idx", lang=lang)
                                ),
                                key=f"ai_mf_tl_mode_{result_id}",
                                use_pills=True,
                            )
                        else:
                            tl_mode = tl_modes[0]
                    with tl_c2:
                        fps_opts = [1.0, 1.5, 2.5]
                        fps_lbls = {
                            1.0: t("ai_mf_fps_slow", lang=lang),
                            1.5: t("ai_mf_fps_norm", lang=lang),
                            2.5: t("ai_mf_fps_fast", lang=lang),
                        }
                        sel_fps = _touch_selector(
                            t("ai_mf_fps_label", lang=lang),
                            options=fps_opts,
                            default=1.5,
                            format_func=lambda v: fps_lbls[v],
                            key=f"ai_mf_tl_fps_{result_id}",
                            use_pills=True,
                        )

                    gif_cache = ai_res_show.setdefault("_gif_cache", {})
                    if abs(float(sel_fps) - 1.5) < 1e-3:
                        if tl_mode == "rgb" and ai_res_show.get("timelapse_rgb_gif_bytes"):
                            gif_cache[("rgb", 1.5)] = ai_res_show["timelapse_rgb_gif_bytes"]
                        elif tl_mode == "index" and ai_res_show.get("timelapse_index_gif_bytes"):
                            gif_cache[("index", 1.5)] = ai_res_show["timelapse_index_gif_bytes"]

                    cache_k = (tl_mode, float(sel_fps))
                    gif_bytes_show = gif_cache.get(cache_k)
                    if gif_bytes_show is None:
                        src_pngs = rgb_pngs_all if tl_mode == "rgb" else idx_pngs_all
                        gif_bytes_show = compile_timelapse_gif_fps(src_pngs, fps=float(sel_fps))
                        gif_cache[cache_k] = gif_bytes_show

                    if gif_bytes_show:
                        st.image(gif_bytes_show, width="stretch")
                        gif_tag = "RGB" if tl_mode == "rgb" else idx_lbl_ts
                        gif_fname = f"Timelapse_{gif_tag}_{pl['place_name']}_{frame_dates[0]}_to_{frame_dates[-1]}.gif"
                        st.download_button(
                            label=t("ai_mf_dl_gif", lang=lang),
                            data=gif_bytes_show,
                            file_name=gif_fname,
                            mime="image/gif",
                            width="stretch",
                            key=f"ai_mf_dl_gif_{result_id}_{tl_mode}_{sel_fps}",
                        )

            # --- TAB 4: All Intermediate Scenes (Visual Strip Gallery with Multi-Index Pills) ---
            with t_strip:
                st.caption(t("ai_mf_strip_caption", lang=lang, n=n_fr))
                strip_modes: list[str] = []
                if any(f.get("rgb_png_bytes") for f in frames_list):
                    strip_modes.append("rgb")
                if any(f.get("index_png_bytes") for f in frames_list):
                    strip_modes.append("index")
                for ek in extra_codes_avail:
                    if any((f.get("extra_pngs") or {}).get(ek) for f in frames_list):
                        strip_modes.append(ek)
                if len(strip_modes) > 1:
                    st_mode = _touch_selector(
                        t("ai_mf_layer_label", lang=lang),
                        options=strip_modes,
                        default=strip_modes[0],
                        format_func=_layer_pill_label,
                        key=f"ai_mf_strip_mode_{result_id}",
                        use_pills=True,
                    )
                else:
                    st_mode = strip_modes[0] if strip_modes else "rgb"

                for row_start in range(0, n_fr, 3):
                    row_frames = frames_list[row_start : row_start + 3]
                    g_cols = st.columns(3, gap="small")
                    for gc_i, fr_item in enumerate(row_frames):
                        with g_cols[gc_i]:
                            if st_mode == "rgb":
                                raw_img = fr_item.get("rgb_png_bytes") or fr_item.get("index_png_bytes")
                                mean_val = fr_item.get("index_mean")
                                mean_lbl = idx_lbl_ts
                            elif st_mode == "index":
                                raw_img = fr_item.get("index_png_bytes") or fr_item.get("rgb_png_bytes")
                                mean_val = fr_item.get("index_mean")
                                mean_lbl = idx_lbl_ts
                            else:
                                raw_img = (fr_item.get("extra_pngs") or {}).get(st_mode) or fr_item.get("rgb_png_bytes")
                                mean_val = (fr_item.get("extra_means") or {}).get(st_mode)
                                mean_lbl = str(st_mode)
                            if raw_img:
                                cap_mean = f" · {mean_lbl}: {mean_val:.3f}" if mean_val is not None else ""
                                st.image(
                                    raw_img,
                                    caption=f"{fr_item['date']}{cap_mean}",
                                    width="stretch",
                                )

            # --- TAB 5: Time Series (Chart, Table & Single CSV Button) ---
            with t_ts:
                _draw_ts_plotly_and_table()

        elif has_legacy_swipe:
            t_sw, t_ts = st.tabs(
                [
                    t("ai_mf_tab_swipe", lang=lang),
                    t("ai_mf_tab_ts", lang=lang),
                ]
            )
            with t_sw:
                active_comp = comp_rgb if comp_rgb is not None else comp_idx
                c_dates = (active_comp or {}).get("dates") or [
                    str(ts_ai["date_str"].iloc[0]),
                    str(ts_ai["date_str"].iloc[-1]),
                ]
                c_uris = (active_comp or {}).get("overlay_uris") or []
                c_bnds = (active_comp or {}).get("overlay_bounds") or []
                ai_op_ts_sw = st.slider(
                    t("opacity_layer_label"),
                    0,
                    100,
                    95,
                    5,
                    key=f"ai_chat_ts_sw_op_{result_id}",
                )
                if len(c_uris) >= 2 and len(c_bnds) >= 2:
                    swipe_html = build_zoomable_swipe_map(
                        overlay_uri_t1=c_uris[0],
                        bounds_t1=c_bnds[0],
                        overlay_uri_t2=c_uris[-1],
                        bounds_t2=c_bnds[-1],
                        label_t1=f"t1: {c_dates[0]}",
                        label_t2=f"t2: {c_dates[-1]}",
                        center_lat=float(ai_res_show["lat"]),
                        center_lon=float(ai_res_show["lon"]),
                        basemap_mode=basemap_now,
                        aoi_bbox_4326=ai_res_show.get("bbox_4326"),
                        aoi_bbox_3794=ai_res_show.get("bbox_3794"),
                        aoi_outline_label=t("slippy_aoi_outline"),
                        height=470,
                        opacity=float(ai_op_ts_sw) / 100.0,
                        lang=lang,
                    )
                    components.html(swipe_html, height=480, scrolling=False)
                    st.caption(t("swipe_zoom_hint"))
            with t_ts:
                _draw_ts_plotly_and_table()
        else:
            _draw_ts_plotly_and_table()

    # 3. Export & Cartography Handoff Bar (CSV is already inside Tab 5 for time_series -> zero duplicate CSV buttons!)
    if ai_kind_show == "gnss_plan":
        if ai_res_show.get("csv_bytes"):
            st.download_button(
                label=t("ai_btn_dl_csv"),
                data=ai_res_show["csv_bytes"],
                file_name=ai_res_show["csv_filename"],
                mime="text/csv",
                width="stretch",
                key=f"ai_chat_dl_csv_{result_id}",
            )
    elif ai_kind_show in ("time_series", "time_series_and_comparison"):
        if ai_res_show.get("tif_bytes"):
            _crs_btn_lbl = format_crs_label(ai_crs_show, short=True)
            _send_carto_lbl = (
                f"🗺️ Pošlji v Izdelavo karte ({_crs_btn_lbl})"
                if lang == "sl"
                else f"🗺️ Send to Map Composer ({_crs_btn_lbl})"
            )
            ac1, ac2 = st.columns(2)
            with ac1:
                st.download_button(
                    label=t("ai_btn_dl_tif"),
                    data=ai_res_show["tif_bytes"],
                    file_name=ai_res_show["tif_filename"],
                    mime="image/tiff",
                    width="stretch",
                    key=f"ai_chat_dl_tif_{result_id}",
                )
            with ac2:
                if st.button(
                    _send_carto_lbl,
                    type="primary",
                    width="stretch",
                    key=f"ai_chat_send_carto_{result_id}",
                ):
                    st.session_state["tab6_ai_result"] = ai_res_show
                    st.session_state["tab4_pending_source_select"] = "tab6_ai"
                    st.session_state["tab6_sent_to_carto"] = True
                    st.session_state["pending_workspace"] = "ws_geo"
                    st.rerun(scope="app")
    else:
        has_tif = bool(ai_res_show.get("tif_bytes"))
        if has_tif:
            _crs_btn_lbl = format_crs_label(ai_crs_show, short=True)
            _send_carto_lbl = (
                f"🗺️ Pošlji v Izdelavo karte ({_crs_btn_lbl})"
                if lang == "sl"
                else f"🗺️ Send to Map Composer ({_crs_btn_lbl})"
            )
            ac1, ac2 = st.columns(2)
            with ac1:
                st.download_button(
                    label=t("ai_btn_dl_tif"),
                    data=ai_res_show["tif_bytes"],
                    file_name=ai_res_show["tif_filename"],
                    mime="image/tiff",
                    width="stretch",
                    key=f"ai_chat_dl_tif_{result_id}",
                )
            with ac2:
                if st.button(
                    _send_carto_lbl,
                    type="primary",
                    width="stretch",
                    key=f"ai_chat_send_carto_{result_id}",
                ):
                    st.session_state["tab6_ai_result"] = ai_res_show
                    st.session_state["tab4_pending_source_select"] = "tab6_ai"
                    st.session_state["tab6_sent_to_carto"] = True
                    st.session_state["pending_workspace"] = "ws_geo"
                    st.rerun(scope="app")

    with st.expander(t("ai_copy_expander"), expanded=False):
        st.code(ai_res_show.get("technical_summary", ""), language="text")


def _apply_user_profile_prefs_to_session(profile: dict[str, Any]) -> None:
    """Apply saved user profile preferences into Streamlit session state before widgets instantiate."""
    pref_lang = str(profile.get("preferred_lang") or "").strip().lower()
    if pref_lang in ("sl", "en"):
        st.session_state["lang"] = pref_lang
        st.session_state["lang_selector"] = pref_lang

    pref_crs = str(profile.get("preferred_crs_mode") or "").strip().lower()
    if pref_crs in CRS_MODE_KEYS:
        st.session_state["sidebar_crs_mode"] = pref_crs

    pref_res = profile.get("preferred_resolution")
    if pref_res is not None:
        try:
            res_int = int(pref_res)
            if res_int > 0:
                st.session_state["selected_resolution"] = res_int
                cur_s = str(st.session_state.get("selected_sensor", SENSOR_S2))
                if res_int in SENSOR_META.get(cur_s, {}).get("res_options", []):
                    st.session_state[f"sidebar_res_select_{cur_s}"] = res_int
        except (TypeError, ValueError):
            pass

    pref_model = str(profile.get("preferred_gemini_model") or "").strip()
    if pref_model:
        st.session_state["gemini_model"] = pref_model

    auto_save = bool(profile.get("auto_save_session", True))
    st.session_state["auth_auto_save_session"] = auto_save

    st.session_state["_auth_last_saved_prefs"] = {
        "preferred_lang": str(st.session_state.get("lang", "sl")),
        "preferred_crs_mode": str(st.session_state.get("sidebar_crs_mode", "auto")),
        "preferred_resolution": int(st.session_state.get("selected_resolution", 10)),
        "preferred_gemini_model": str(st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL)),
        "auto_save_session": auto_save,
    }


def _sync_logged_in_profile_prefs() -> None:
    """Auto-save preference changes to ``profile.json`` (and GitHub Cloud) for logged-in users."""
    auth_user = st.session_state.get("auth_user")
    if not isinstance(auth_user, dict) or not auth_user.get("username"):
        return

    cur_prefs = {
        "preferred_lang": str(st.session_state.get("lang", "sl")),
        "preferred_crs_mode": str(st.session_state.get("sidebar_crs_mode", "auto")),
        "preferred_resolution": int(st.session_state.get("selected_resolution", 10)),
        "preferred_gemini_model": str(st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL)),
        "auto_save_session": bool(st.session_state.get("auth_auto_save_session", True)),
    }
    last_saved = st.session_state.get("_auth_last_saved_prefs")
    if last_saved == cur_prefs:
        return

    try:
        updated = save_user_profile(
            str(auth_user["username"]),
            cur_prefs,
            sync=True,
            show_toast=True,
        )
        st.session_state["auth_user"] = updated
        st.session_state["_auth_last_saved_prefs"] = dict(cur_prefs)
    except Exception:
        pass


def _collect_current_workspace_payload(
    *,
    center_lat: float | None = None,
    center_lon: float | None = None,
    bbox_4326: Sequence[float] | None = None,
    aoi_polygon_4326: dict[str, Any] | None = None,
    loc_label: str | None = None,
) -> dict[str, Any]:
    """Build a lightweight, strictly JSON-safe snapshot of the current user workspace."""
    c_lat = float(center_lat if center_lat is not None else st.session_state.get("map_lat", SLOVENIA_CENTER_LAT))
    c_lon = float(center_lon if center_lon is not None else st.session_state.get("map_lon", SLOVENIA_CENTER_LON))
    rad_m = int(st.session_state.get("map_radius_m", 2000))
    l_name = str(loc_label or st.session_state.get("active_location_label") or "Ljubljana")
    poly_geo = aoi_polygon_4326 if aoi_polygon_4326 is not None else (
        st.session_state.get("drawn_polygon_4326") or st.session_state.get("uploaded_geojson_aoi")
    )
    bbox_list = [float(v) for v in bbox_4326] if (bbox_4326 is not None and len(bbox_4326) == 4) else (
        [float(v) for v in st.session_state["search_bbox_4326"]]
        if (isinstance(st.session_state.get("search_bbox_4326"), (list, tuple)) and len(st.session_state["search_bbox_4326"]) == 4)
        else None
    )

    s_date = st.session_state.get("search_start_date")
    e_date = st.session_state.get("search_end_date")
    start_str = s_date.isoformat() if isinstance(s_date, (date, datetime)) else str(s_date or "")
    end_str = e_date.isoformat() if isinstance(e_date, (date, datetime)) else str(e_date or "")

    # Compact analysis summary (scalars/metadata only, zero rasters)
    analysis_summary: dict[str, Any] = {}
    t2_meta = st.session_state.get("tab2_index_meta")
    if isinstance(t2_meta, dict):
        analysis_summary["active_index"] = t2_meta.get("index_code") or t2_meta.get("name")
        analysis_summary["index_date"] = t2_meta.get("date_str")
    t2_stats = st.session_state.get("tab2_index_stats")
    if isinstance(t2_stats, dict):
        analysis_summary["index_mean"] = t2_stats.get("mean")
        analysis_summary["index_median"] = t2_stats.get("median")
    t2_snow = st.session_state.get("tab2_snow_stats")
    if isinstance(t2_snow, dict):
        analysis_summary["snow_area_km2"] = t2_snow.get("snow_area_km2")
        analysis_summary["snowline_m"] = t2_snow.get("snowline_elevation_m")
    t5_dis = st.session_state.get("tab5_disaster_result")
    if isinstance(t5_dis, dict):
        analysis_summary["disaster_type"] = t5_dis.get("disaster_type")
        analysis_summary["affected_area_ha"] = t5_dis.get("affected_area_ha")
        analysis_summary["affected_area_km2"] = t5_dis.get("affected_area_km2")
    t7_gnss = st.session_state.get("tab7_gnss_result")
    if isinstance(t7_gnss, dict):
        analysis_summary["gnss_date"] = t7_gnss.get("date_str")
        analysis_summary["gnss_best_pdop"] = t7_gnss.get("min_pdop")

    return {
        "aoi": {
            "location_name": l_name,
            "center_lat": round(c_lat, 6),
            "center_lon": round(c_lon, 6),
            "radius_m": rad_m,
            "bbox_4326": bbox_list,
            "custom_polygon_geojson": poly_geo if isinstance(poly_geo, dict) else None,
        },
        "settings": {
            "sensor": str(st.session_state.get("selected_sensor", SENSOR_S2)),
            "start_date": start_str,
            "end_date": end_str,
            "interval": str(st.session_state.get("temporal_interval", INTERVAL_ALL)),
            "target_season": str(st.session_state.get("yearly_target_season", SEASON_SUMMER)),
            "max_cloud_pct": int(st.session_state.get("search_max_cloud", 20)),
            "resolution_m": int(st.session_state.get("selected_resolution", 10)),
            "crs_mode": str(st.session_state.get("sidebar_crs_mode", "auto")),
            "basemap": str(st.session_state.get("map_basemap", "satellite")),
            "exclude_l7_slc": bool(st.session_state.get("exclude_l7_slc_off", True)),
            "min_swath_coverage": 95 if bool(st.session_state.get("exclude_partial_swaths", True)) else 0,
        },
        "cartography_layout": dict(st.session_state.get("tab4_layout_config") or DEFAULT_LAYOUT_CONFIG),
        "analysis_summary": analysis_summary,
        "ai_chat_messages": list(st.session_state.get("ai_chat_messages") or []),
    }


def _apply_project_payload_to_session(payload: dict[str, Any]) -> None:
    """Restore a saved project or last-session snapshot into Streamlit session state before widgets render."""
    if not isinstance(payload, dict):
        return

    # 1. Restore AOI
    aoi = payload.get("aoi")
    if isinstance(aoi, dict):
        if aoi.get("location_name"):
            st.session_state["active_location_label"] = str(aoi["location_name"])
        if aoi.get("center_lat") is not None and aoi.get("center_lon") is not None:
            try:
                st.session_state["map_lat"] = float(aoi["center_lat"])
                st.session_state["map_lon"] = float(aoi["center_lon"])
                st.session_state["map_jump_version"] = int(st.session_state.get("map_jump_version", 0)) + 1
            except (TypeError, ValueError):
                pass
        if aoi.get("radius_m") is not None:
            try:
                _max_r = int(get_ram_limits()["max_radius_m"])
                r_val = max(250, min(int(aoi["radius_m"]), _max_r))
                st.session_state["map_radius_m"] = r_val
                st.session_state["ws1_map_radius_slider"] = r_val
            except (TypeError, ValueError):
                pass
        poly = aoi.get("custom_polygon_geojson")
        if isinstance(poly, dict) and poly.get("type"):
            st.session_state["drawn_polygon_4326"] = poly
            st.session_state["sidebar_aoi_mode"] = "draw"
            st.session_state["draw_map_version"] = int(st.session_state.get("draw_map_version", 0)) + 1
        else:
            st.session_state["drawn_polygon_4326"] = None
            st.session_state["uploaded_geojson_aoi"] = None
            st.session_state["sidebar_aoi_mode"] = "map"

    # 2. Restore Settings
    settings = payload.get("settings")
    if isinstance(settings, dict):
        sens = str(settings.get("sensor") or "")
        if sens in SENSOR_META:
            st.session_state["selected_sensor"] = sens
            st.session_state["active_sensor"] = sens
            st.session_state["sidebar_sensor_select"] = sens
            st.session_state["main_bar_sensor_select"] = sens
            st.session_state["tab3_sensor_select"] = sens

        for s_key, targets in (
            ("start_date", ("search_start_date", "sidebar_start_date", "tab3_start_date")),
            ("end_date", ("search_end_date", "sidebar_end_date", "tab3_end_date")),
        ):
            raw_d = settings.get(s_key)
            if raw_d:
                try:
                    parsed_d = date.fromisoformat(str(raw_d)[:10])
                    for tk in targets:
                        st.session_state[tk] = parsed_d
                except ValueError:
                    pass

        if settings.get("interval") in INTERVAL_KEYS:
            iv = str(settings["interval"])
            st.session_state["temporal_interval"] = iv
            st.session_state["sidebar_interval_select"] = iv
            st.session_state["tab3_interval_select"] = iv

        if settings.get("target_season") in SEASON_KEYS:
            ts = str(settings["target_season"])
            st.session_state["yearly_target_season"] = ts
            st.session_state["sidebar_yearly_season"] = ts
            st.session_state["tab3_yearly_season"] = ts

        if settings.get("max_cloud_pct") is not None:
            try:
                mc = max(0, min(100, int(settings["max_cloud_pct"])))
                st.session_state["search_max_cloud"] = mc
                st.session_state["sidebar_max_cloud"] = mc
                st.session_state["tab3_max_cloud"] = mc
            except (TypeError, ValueError):
                pass

        if settings.get("resolution_m") is not None:
            try:
                res_m = int(settings["resolution_m"])
                cur_s = str(st.session_state.get("selected_sensor", SENSOR_S2))
                if res_m in SENSOR_META.get(cur_s, {}).get("res_options", []):
                    st.session_state["selected_resolution"] = res_m
                    st.session_state[f"sidebar_res_select_{cur_s}"] = res_m
                    st.session_state["last_rec_res_key"] = f"{cur_s}_{res_m}"
            except (TypeError, ValueError):
                pass

        if settings.get("crs_mode") in CRS_MODE_KEYS:
            st.session_state["sidebar_crs_mode"] = str(settings["crs_mode"])

        if settings.get("basemap") in BASEMAP_KEYS:
            st.session_state["map_basemap"] = str(settings["basemap"])

        if "exclude_l7_slc" in settings:
            v_l7 = bool(settings["exclude_l7_slc"])
            st.session_state["exclude_l7_slc_off"] = v_l7
            st.session_state["sidebar_l7_slc_exclude"] = v_l7
            st.session_state["tab3_l7_slc_exclude"] = v_l7

        if "min_swath_coverage" in settings:
            v_ex = float(settings.get("min_swath_coverage") or 0) >= 50.0
            st.session_state["exclude_partial_swaths"] = v_ex
            st.session_state["sidebar_exclude_partial"] = v_ex
            st.session_state["tab3_exclude_partial"] = v_ex

    # 3. Restore QGIS-style Cartography LayoutConfig
    layout = payload.get("cartography_layout")
    if isinstance(layout, dict) and layout:
        merged_l = merge_layout_config(layout)
        st.session_state["tab4_layout_config"] = merged_l
        st.session_state["tab4_pending_layout_overrides"] = dict(merged_l)

    # 4. Restore text AI chat history into Workspace 5
    chats = payload.get("ai_chat_messages")
    if isinstance(chats, list):
        st.session_state["ai_chat_messages"] = [
            {
                "role": "user" if m.get("role") == "user" else "assistant",
                "content": str(m.get("content") or ""),
                "action": m.get("action"),
                "plan": m.get("plan"),
                "result_id": None,
                "fallback_note": str(m.get("fallback_note") or ""),
            }
            for m in chats
            if isinstance(m, dict) and str(m.get("content") or "").strip()
        ]
        st.session_state["ai_chat_results"] = {}


def _maybe_auto_save_session() -> None:
    """Auto-save lightweight active session to ``last_session.json`` + GitHub Cloud if enabled."""
    auth_user = st.session_state.get("auth_user")
    if not isinstance(auth_user, dict) or not auth_user.get("username"):
        return
    if not bool(st.session_state.get("auth_auto_save_session", True)):
        return
    try:
        payload = _collect_current_workspace_payload()
        save_last_session(str(auth_user["username"]), payload, sync=True, show_toast=False)
    except Exception:
        pass


def _render_sidebar_user_profile_card(lang: str) -> None:
    """Render the compact, strictly optional User Profile card at the top of the Sidebar."""
    auth_user = st.session_state.get("auth_user")
    is_logged_in = isinstance(auth_user, dict) and bool(auth_user.get("username"))
    gh_status = get_github_sync_status(lang=lang)

    with st.container(border=True):
        st.markdown(f"**{t('auth_card_title', lang=lang)}**")

        flash_msg = st.session_state.pop("_auth_flash_msg", None)
        if flash_msg:
            st.success(str(flash_msg))

        if is_logged_in:
            uname = str(auth_user.get("username", ""))
            dname = str(auth_user.get("display_name") or uname)
            safe_dname = dname.replace("<", "&lt;").replace(">", "&gt;")
            safe_uname = uname.replace("<", "&lt;").replace(">", "&gt;")
            badge_cls = "sb-env-badge sb-env-badge-hi" if gh_status["configured"] else "sb-env-badge"
            st.markdown(
                f'<div style="font-size:0.86rem;font-weight:700;color:#0F172A;margin-bottom:4px;">'
                f"👤 {safe_dname} "
                f'<span style="font-weight:500;color:#475569;">(@{safe_uname})</span>'
                f"</div>"
                f'<div class="{badge_cls}" title="{gh_status["detail"]}">{gh_status["badge_text"]}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.setdefault(
                "auth_auto_save_session", bool(auth_user.get("auto_save_session", True))
            )
            st.toggle(
                t("auth_auto_save_toggle", lang=lang),
                key="auth_auto_save_session",
            )
            col_sync, col_out = st.columns([1.25, 0.85], gap="small")
            with col_sync:
                if st.button(
                    t("auth_btn_sync", lang=lang),
                    key="auth_sync_github_btn",
                    width="stretch",
                ):
                    if is_github_sync_configured():
                        ok_sync = sync_user_data_to_github(uname)
                        if ok_sync:
                            st.success(t("auth_sync_done", lang=lang))
                        else:
                            st.info(t("auth_sync_local_only", lang=lang))
                    else:
                        st.info(t("auth_sync_local_only", lang=lang))
            with col_out:
                if st.button(
                    t("auth_btn_logout", lang=lang),
                    key="auth_logout_btn",
                    width="stretch",
                ):
                    st.session_state.pop("auth_user", None)
                    st.session_state.pop("_auth_last_saved_prefs", None)
                    st.session_state.pop("_pending_restore_last_session", None)
                    st.session_state["_auth_flash_msg"] = t("auth_logout_ok", lang=lang)
                    st.rerun()
        else:
            st.caption(t("auth_guest_badge", lang=lang))
            st.caption(t("auth_guest_cloud_hint", lang=lang))
            with st.expander(t("auth_expander_label", lang=lang), expanded=False):
                tab_login, tab_reg = st.tabs(
                    [t("auth_tab_login", lang=lang), t("auth_tab_register", lang=lang)]
                )
                with tab_login:
                    login_u = st.text_input(
                        t("auth_username_label", lang=lang),
                        key="auth_login_username",
                        placeholder="jan.novak",
                    )
                    login_p = st.text_input(
                        t("auth_password_label", lang=lang),
                        type="password",
                        key="auth_login_password",
                    )
                    if st.button(
                        t("auth_btn_login", lang=lang),
                        type="primary",
                        width="stretch",
                        key="auth_login_submit",
                    ):
                        ok, msg_key, profile = authenticate_user(login_u, login_p)
                        if ok and profile is not None:
                            st.session_state["auth_user"] = profile
                            st.session_state["_auth_pending_profile"] = profile
                            last_sess = load_last_session(str(profile.get("username", "")))
                            if last_sess is not None:
                                st.session_state["_pending_restore_last_session"] = last_sess
                            st.session_state["_auth_flash_msg"] = t(msg_key, lang=lang)
                            st.rerun()
                        else:
                            st.error(t(msg_key, lang=lang))

                with tab_reg:
                    reg_u = st.text_input(
                        t("auth_username_label", lang=lang),
                        key="auth_reg_username",
                        placeholder="jan.novak",
                    )
                    reg_dn = st.text_input(
                        t("auth_display_name_label", lang=lang),
                        key="auth_reg_display_name",
                        placeholder="Jan Novak",
                    )
                    reg_p1 = st.text_input(
                        t("auth_password_label", lang=lang),
                        type="password",
                        key="auth_reg_password",
                    )
                    reg_p2 = st.text_input(
                        t("auth_password_confirm_label", lang=lang),
                        type="password",
                        key="auth_reg_password_confirm",
                    )
                    if st.button(
                        t("auth_btn_register", lang=lang),
                        type="primary",
                        width="stretch",
                        key="auth_reg_submit",
                    ):
                        if reg_p1 != reg_p2:
                            st.error(t("auth_err_password_mismatch", lang=lang))
                        else:
                            init_prefs = {
                                "preferred_lang": str(st.session_state.get("lang", "sl")),
                                "preferred_crs_mode": str(st.session_state.get("sidebar_crs_mode", "auto")),
                                "preferred_resolution": int(st.session_state.get("selected_resolution", 10)),
                                "preferred_gemini_model": str(
                                    st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL)
                                ),
                                "auto_save_session": True,
                            }
                            ok, msg_key, profile = register_user(
                                reg_u,
                                reg_p1,
                                display_name=reg_dn,
                                initial_prefs=init_prefs,
                                sync=True,
                            )
                            if ok and profile is not None:
                                st.session_state["auth_user"] = profile
                                st.session_state["_auth_pending_profile"] = profile
                                st.session_state["_auth_flash_msg"] = t(msg_key, lang=lang)
                                st.rerun()
                            else:
                                st.error(t(msg_key, lang=lang))

        # Compact Gemini API & Model settings (accessible to both Guests and Logged-in users)
        _render_sidebar_gemini_settings(lang)


def _render_sidebar_gemini_settings(lang: str) -> tuple[str | None, str, str]:
    """Render the compact collapsible '⚙️ Gemini API in model' panel inside the Sidebar Profile card."""
    preset_ids: list[str] = [m["id"] for m in GEMINI_MODEL_OPTIONS]
    mod_lbl_key = "label_en" if lang == "en" else "label_sl"
    preset_lbl_map = {m["id"]: m[mod_lbl_key] for m in GEMINI_MODEL_OPTIONS}

    if st.session_state.get("gemini_model") not in preset_ids:
        st.session_state["gemini_model"] = DEFAULT_GEMINI_MODEL

    with st.expander(t("ai_sidebar_gemini_expander", lang=lang), expanded=False):
        st.text_input(
            t("ai_key_input_label", lang=lang),
            type="password",
            key="tab6_gemini_key",
        )

        selected_mod_choice = _touch_selector(
            t("ai_model_select_label", lang=lang),
            options=preset_ids,
            default=DEFAULT_GEMINI_MODEL,
            format_func=lambda mid: preset_lbl_map.get(mid, mid),
            key="gemini_model",
            use_pills=True,
        )

        custom_mod_id = ""
        if selected_mod_choice == "custom":
            custom_mod_id = st.text_input(
                t("ai_model_custom_label", lang=lang),
                placeholder=t("ai_model_custom_placeholder", lang=lang),
                key="gemini_custom_model",
            )

        effective_gemini_model = resolve_gemini_model_id(
            str(selected_mod_choice or DEFAULT_GEMINI_MODEL),
            custom_id=custom_mod_id,
        )
        cur_api_key, key_src = get_active_gemini_api_key(st.session_state.get("tab6_gemini_key"))

        if cur_api_key:
            st.caption(t("ai_sidebar_status_active", lang=lang, model=effective_gemini_model))
        else:
            st.caption(t("ai_sidebar_status_local", lang=lang))

        if st.button(
            t("ai_btn_verify_short", lang=lang),
            width="stretch",
            key="sb_btn_verify_gemini_key",
        ):
            if not cur_api_key:
                st.session_state["ai_api_verify_result"] = {"status": "need_key"}
            else:
                try:
                    v_res = verify_gemini_api_connection(cur_api_key, model=effective_gemini_model)
                    st.session_state["ai_api_verify_result"] = {"status": "ok", **v_res}
                except Exception as exc:
                    st.session_state["ai_api_verify_result"] = {"status": "err", "error": str(exc)}

        ver_state = st.session_state.get("ai_api_verify_result")
        if isinstance(ver_state, dict):
            v_st = ver_state.get("status")
            if v_st == "need_key":
                st.warning(t("ai_models_need_key", lang=lang))
            elif v_st == "ok":
                if ver_state.get("model_found"):
                    st.success(
                        t(
                            "ai_verify_ok",
                            lang=lang,
                            model=ver_state.get("model", effective_gemini_model),
                            n=int(ver_state.get("total_models", 0)),
                        )
                    )
                else:
                    st.warning(
                        t(
                            "ai_verify_model_missing",
                            lang=lang,
                            model=ver_state.get("model", effective_gemini_model),
                            n=int(ver_state.get("total_models", 0)),
                            fallback=ver_state.get("fallback", DEFAULT_GEMINI_MODEL),
                        )
                    )
            elif v_st == "err":
                st.error(t("ai_verify_err", lang=lang, err=ver_state.get("error", "")))

    return cur_api_key, key_src, effective_gemini_model


def _render_sidebar_projects_manager(
    *,
    lang: str,
    center_lat: float,
    center_lon: float,
    bbox_4326: Sequence[float] | None,
    aoi_polygon_4326: dict[str, Any] | None,
    loc_label: str,
) -> None:
    """Render the collapsible '💾 Moji projekti (N)' manager in the Sidebar for logged-in users."""
    auth_user = st.session_state.get("auth_user")
    if not isinstance(auth_user, dict) or not auth_user.get("username"):
        return
    uname = str(auth_user["username"])
    projects = list_user_projects(uname)

    with st.expander(t("proj_expander_title", lang=lang, n=len(projects)), expanded=False):
        p_name = st.text_input(
            t("proj_name_label", lang=lang),
            placeholder=t("proj_name_placeholder", lang=lang),
            key="sb_proj_name_input",
        )
        p_note = st.text_input(
            t("proj_note_label", lang=lang),
            placeholder=t("proj_note_placeholder", lang=lang),
            key="sb_proj_note_input",
        )
        if st.button(
            t("proj_btn_save", lang=lang),
            type="primary",
            width="stretch",
            key="sb_proj_save_btn",
        ):
            clean_pn = str(p_name or "").strip()
            if not clean_pn:
                st.warning(t("proj_save_err_empty", lang=lang))
            else:
                payload = _collect_current_workspace_payload(
                    center_lat=center_lat,
                    center_lon=center_lon,
                    bbox_4326=bbox_4326,
                    aoi_polygon_4326=aoi_polygon_4326,
                    loc_label=loc_label,
                )
                save_user_project(uname, clean_pn, str(p_note or ""), payload, sync=True)
                _maybe_auto_save_session()
                st.success(t("proj_saved_ok", lang=lang, name=clean_pn))
                st.rerun()

        st.divider()
        if not projects:
            st.caption(t("proj_list_empty", lang=lang))
        else:
            proj_ids = [str(p["id"]) for p in projects]
            proj_map = {str(p["id"]): p for p in projects}

            def _fmt_proj(pid: str) -> str:
                pr = proj_map[pid]
                loc = str((pr.get("aoi") or {}).get("location_name") or "—")
                sens = str((pr.get("settings") or {}).get("sensor") or "S2")
                s_short = SENSOR_META.get(sens, {}).get("short_name", sens)
                upd = str(pr.get("updated_at") or "")[:10]
                return f"{pr['name']} · {loc} · {s_short} · {upd}"

            sel_pid = st.selectbox(
                t("proj_select_label", lang=lang),
                options=proj_ids,
                format_func=_fmt_proj,
                key="sb_proj_selected_id",
            )
            sel_proj = proj_map.get(sel_pid)
            if sel_proj and sel_proj.get("note"):
                st.caption(f"📝 {sel_proj['note']}")

            c_ld, c_up, c_dl = st.columns(3, gap="small")
            with c_ld:
                if st.button(t("proj_btn_load", lang=lang), width="stretch", key="sb_proj_load_btn"):
                    if sel_proj is not None:
                        st.session_state["_pending_project_load"] = sel_proj
                        st.session_state["_pending_toast_msg"] = t(
                            "proj_loaded_toast", lang=lang, name=sel_proj["name"]
                        )
                        st.rerun()
            with c_up:
                if st.button(t("proj_btn_update", lang=lang), width="stretch", key="sb_proj_upd_btn"):
                    if sel_proj is not None:
                        payload = _collect_current_workspace_payload(
                            center_lat=center_lat,
                            center_lon=center_lon,
                            bbox_4326=bbox_4326,
                            aoi_polygon_4326=aoi_polygon_4326,
                            loc_label=loc_label,
                        )
                        save_user_project(
                            uname,
                            str(sel_proj["name"]),
                            str(p_note or sel_proj.get("note") or ""),
                            payload,
                            overwrite_id=str(sel_proj["id"]),
                            sync=True,
                        )
                        st.success(t("proj_updated_ok", lang=lang, name=sel_proj["name"]))
                        st.rerun()
            with c_dl:
                if st.button(t("proj_btn_delete", lang=lang), width="stretch", key="sb_proj_del_btn"):
                    if sel_proj is not None:
                        delete_user_project(uname, str(sel_proj["id"]), sync=True)
                        st.success(t("proj_deleted_ok", lang=lang, name=sel_proj["name"]))
                        st.rerun()

        st.divider()
        st.caption(t("proj_backup_header", lang=lang))
        bundle_bytes = export_user_bundle_json(uname)
        st.download_button(
            t("proj_btn_export_json", lang=lang),
            data=bundle_bytes,
            file_name=f"eo_backup_{uname}_{date.today().isoformat()}.json",
            mime="application/json",
            width="stretch",
            key="sb_proj_export_json_btn",
        )
        up_json = st.file_uploader(
            t("proj_import_label", lang=lang),
            type=["json"],
            key="sb_proj_import_json",
        )
        if up_json is not None:
            import_sig = f"{up_json.name}:{up_json.size}"
            if st.session_state.get("_last_imported_bundle_sig") != import_sig:
                try:
                    counts = import_user_bundle_json(uname, up_json.getvalue(), sync=True)
                    st.session_state["_last_imported_bundle_sig"] = import_sig
                    st.success(
                        t(
                            "proj_import_ok",
                            lang=lang,
                            p=counts["projects"],
                            l=counts["map_layouts"],
                        )
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(t("proj_import_err", lang=lang, err=exc))


def _inject_tablet_readonly_dropdowns() -> None:
    """Prevent tablet virtual keyboard popups & text editing on all Streamlit ``st.selectbox`` dropdowns."""
    components.html(
        """
        <script>
        (function () {
          try {
            const doc = window.parent && window.parent.document;
            if (!doc) return;
            function lockSelectInputs(root) {
              const inputs = (root || doc).querySelectorAll('[data-baseweb="select"] input');
              inputs.forEach(function (el) {
                if (el.getAttribute('readonly') !== 'true') {
                  el.setAttribute('readonly', 'true');
                }
                if (el.getAttribute('inputmode') !== 'none') {
                  el.setAttribute('inputmode', 'none');
                }
                el.style.caretColor = 'transparent';
                el.style.cursor = 'pointer';
              });
            }
            lockSelectInputs(doc);
            if (!window.parent.__dzSelectReadonlyObserver) {
              const obs = new MutationObserver(function () {
                lockSelectInputs(doc);
              });
              obs.observe(doc.body, { childList: true, subtree: true });
              window.parent.__dzSelectReadonlyObserver = obs;
            }
          } catch (_err) {}
        })();
        </script>
        """,
        height=0,
    )


def main() -> None:
    _init_session_state()
    _inject_tablet_readonly_dropdowns()
    github_restore_all_on_startup()

    # Apply any queued widget-bound state updates strictly BEFORE any sidebar/workspace widgets instantiate
    pending_w_state = st.session_state.pop("_pending_widget_state", None)
    if isinstance(pending_w_state, dict):
        for _wk, _wv in pending_w_state.items():
            st.session_state[_wk] = _wv

    pending_prof = st.session_state.pop("_auth_pending_profile", None)
    if isinstance(pending_prof, dict):
        _apply_user_profile_prefs_to_session(pending_prof)

    pending_proj = st.session_state.pop("_pending_project_load", None)
    if isinstance(pending_proj, dict):
        _apply_project_payload_to_session(pending_proj)

    pending_toast = st.session_state.pop("_pending_toast_msg", None)
    if pending_toast:
        st.toast(str(pending_toast))

    pending_ws = st.session_state.pop("pending_workspace", None)
    if pending_ws in WORKSPACE_KEYS:
        st.session_state["active_workspace"] = pending_ws
        st.session_state["top_workspace_nav"] = pending_ws

    pending_mode = st.session_state.pop("pending_aoi_mode", None)
    if pending_mode in WS1_AOI_MODE_KEYS:
        st.session_state["sidebar_aoi_mode"] = pending_mode
    elif pending_mode == "manual":
        st.session_state["sidebar_aoi_mode"] = "map"

    if st.session_state.pop("pending_reset_scene_widgets", False):
        reset_prefixes = (
            "tab1_scene_select_",
            "tab1_scene_table_",
            "tab2_scene_select_",
            "tab2_comp_n_",
            "tab3_max_epochs_",
            "tab3_t1_select_",
            "tab3_t2_select_",
            "tab3_comp_dates_",
            "tab3_tl_frames_",
            "__scene_page_",
            "__scene_last_idx_",
            "__scene_pill_",
            "__pg_sel_",
            "__multi_pg_",
            "__multi_pills_",
        )
        for k in list(st.session_state.keys()):
            if k == "tab2_sar_scene_select" or k.startswith(reset_prefixes):
                st.session_state.pop(k, None)

    sensor = str(st.session_state["selected_sensor"])
    sensor_cfg = SENSOR_META[sensor]
    _clamp_dates_for_sensor(sensor)

    cur_lang_pre = str(st.session_state.get("lang", "sl"))
    # Enforce hard RAM radius cap before resolving AOI so searches are never blocked
    _ram_lims_main = get_ram_limits()
    _max_rad_main = int(_ram_lims_main["max_radius_m"])
    if int(st.session_state.get("map_radius_m", 2000)) > _max_rad_main:
        st.session_state["map_radius_m"] = _max_rad_main
    if int(st.session_state.get("ws1_map_radius_slider", 2000)) > _max_rad_main:
        st.session_state["ws1_map_radius_slider"] = _max_rad_main

    (
        aoi_mode_key,
        bbox_4326,
        bbox_3794,
        aoi_polygon_4326,
        center_lat,
        center_lon,
        aoi_error,
        loc_label,
    ) = _resolve_active_aoi_state(cur_lang_pre)

    area_km2: float = 0.0
    aoi_valid: bool = False
    rec_res: int = sensor_cfg["default_res"]
    if bbox_4326 is not None and aoi_error is None:
        if aoi_polygon_4326 is not None:
            area_km2, aoi_valid, base_rec_res = validate_aoi_geometry(aoi_polygon_4326, sensor=sensor)
            _, bbox_ok, bbox_rec_res = validate_aoi_size(bbox_4326, sensor=sensor)
            aoi_valid = aoi_valid and bbox_ok
            base_rec_res = max(base_rec_res, bbox_rec_res)
        else:
            area_km2, aoi_valid, base_rec_res = validate_aoi_size(bbox_4326, sensor=sensor)
        rec_res = base_rec_res
        res_key = f"{sensor}_{rec_res}"
        if st.session_state["last_rec_res_key"] != res_key:
            st.session_state["last_rec_res_key"] = res_key
            st.session_state["selected_resolution"] = rec_res
            st.session_state[f"sidebar_res_select_{sensor}"] = rec_res

    # =======================================================================
    # MINIMALNA, PREGLEDNA STRANSKA VRSTICA (SAMO GLOBALNE NASTAVITVE SEJE)
    # =======================================================================
    with st.sidebar:
        lang = st.radio(
            t("lang_label"),
            options=["sl", "en"],
            index=0 if st.session_state["lang"] == "sl" else 1,
            horizontal=True,
            format_func=lambda code: "SL" if code == "sl" else "EN",
            key="lang_selector",
        )
        st.session_state["lang"] = lang

        # --- 0. Opcijski uporabniški profil (Gost / Prijava / GitHub Cloud) ---
        _render_sidebar_user_profile_card(lang)
        _render_sidebar_projects_manager(
            lang=lang,
            center_lat=float(center_lat),
            center_lon=float(center_lon),
            bbox_4326=bbox_4326,
            aoi_polygon_4326=aoi_polygon_4326,
            loc_label=str(loc_label),
        )

        crs_mode_cur = str(st.session_state.get("sidebar_crs_mode", "auto"))
        if crs_mode_cur not in CRS_MODE_KEYS:
            crs_mode_cur = "auto"
        target_crs_pre, _ = get_best_metric_crs(float(center_lat), float(center_lon), mode=crs_mode_cur)
        sel_res_pre = int(st.session_state.get("selected_resolution", sensor_cfg["default_res"]))

        # --- 1. Kompakten Aktivni kontekst (Status Card) ---
        runtime_env = get_runtime_environment()
        env_mode = str(runtime_env["mode"])
        if is_codespaces():
            env_badge_text = t("sb_env_codespaces")
            env_badge_cls = "sb-env-badge sb-env-badge-hi"
        elif env_mode == "hf_spaces":
            env_badge_text = t("sb_env_hf_spaces")
            env_badge_cls = "sb-env-badge sb-env-badge-hi"
        elif env_mode == "streamlit_cloud":
            env_badge_text = t("sb_env_streamlit_cloud")
            env_badge_cls = "sb-env-badge"
        else:
            env_badge_text = t("sb_env_local")
            env_badge_cls = "sb-env-badge"

        area_ha = area_km2 * 100.0
        area_ctx_str = f"{area_km2:,.1f} km² ({area_ha:,.0f} ha)" if bbox_4326 is not None else "—"
        safe_loc = str(loc_label).replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div class="sb-status-card">'
            f'<div class="sb-status-title">{t("sb_context_title")}</div>'
            f'<div class="sb-status-grid">'
            f'<span class="sb-status-key">{t("sb_ctx_location")}</span>'
            f'<span class="sb-status-val" title="{safe_loc}">{safe_loc}</span>'
            f'<span class="sb-status-key">{t("sb_ctx_area")}</span>'
            f'<span class="sb-status-val">{area_ctx_str}</span>'
            f'<span class="sb-status-key">{t("sb_ctx_sensor")}</span>'
            f'<span class="sb-status-val">{sensor_cfg["short_name"]}</span>'
            f'<span class="sb-status-key">{t("sb_ctx_crs")}</span>'
            f'<span class="sb-status-val">{format_crs_label(target_crs_pre, short=True)}</span>'
            f'<span class="sb-status-key">{t("sb_ctx_res")}</span>'
            f'<span class="sb-status-val">{sel_res_pre} m</span>'
            f'</div>'
            f'<div class="{env_badge_cls}">{env_badge_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if bbox_4326 is not None and aoi_error is None:
            if not aoi_valid:
                st.markdown(
                    f'<div class="ram-pill ram-pill-err">{t("ram_status_err", area=area_km2, max_area=float(runtime_env["max_area_km2"]))}</div>',
                    unsafe_allow_html=True,
                )
            elif sel_res_pre < rec_res:
                st.markdown(
                    f'<div class="ram-pill ram-pill-warn">'
                    f'{t("ram_status_finer", sel=sel_res_pre, rec=rec_res, area=area_km2)}</div>',
                    unsafe_allow_html=True,
                )
            elif area_km2 > float(runtime_env["res_10m_max_km2"]):
                st.markdown(
                    f'<div class="ram-pill ram-pill-warn">'
                    f'{t("ram_status_warn", area=area_km2, rec=rec_res)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="ram-pill ram-pill-ok">'
                    f'{t("ram_status_ok", area=area_km2, res=sel_res_pre)}</div>',
                    unsafe_allow_html=True,
                )

        # --- 2. Globalne nastavitve ---
        st.divider()
        st.subheader(t("sb_global_settings"))

        res_options: list[int] = list(sensor_cfg["res_options"])
        cur_sel_res = int(st.session_state["selected_resolution"])
        default_res_val = cur_sel_res if cur_sel_res in res_options else res_options[0]

        selected_resolution = _touch_selector(
            t("res_label"),
            options=res_options,
            default=default_res_val,
            format_func=lambda r: (
                f"{r} m ★" if (bbox_4326 is not None and r == rec_res) else f"{r} m"
            ),
            help=t("res_help"),
            key=f"sidebar_res_select_{sensor}",
            use_pills=True,
        )
        st.session_state["selected_resolution"] = int(selected_resolution)
        eff_res_safe, res_auto_adj = get_safe_processing_resolution(
            int(selected_resolution), area_km2, sensor=sensor
        )
        if res_auto_adj:
            selected_resolution = int(eff_res_safe)
            st.caption(t("ram_auto_res_note", res=selected_resolution))

        crs_mode = _touch_selector(
            t("output_crs_label"),
            options=CRS_MODE_KEYS,
            default="auto",
            format_func=lambda m, _lang=lang: t(CRS_MODE_I18N[m], lang=_lang),
            key="sidebar_crs_mode",
            use_pills=True,
        )
        target_crs, target_crs_label = get_best_metric_crs(
            float(center_lat), float(center_lon), mode=str(crs_mode)
        )
        _sync_logged_in_profile_prefs()

        today = date.today()
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input(
                t("start_date_label"),
                min_value=HISTORICAL_MIN_DATE,
                max_value=today,
                key="sidebar_start_date",
                on_change=_sync_start_date_state,
                args=("sidebar_start_date",),
            )
        with col_d2:
            end_date = st.date_input(
                t("end_date_label"),
                min_value=HISTORICAL_MIN_DATE,
                max_value=today,
                key="sidebar_end_date",
                on_change=_sync_end_date_state,
                args=("sidebar_end_date",),
            )

        dates_valid = start_date <= end_date
        if not dates_valid:
            st.error(t("date_order_error"))
        elif start_date < sensor_cfg["min_date"]:
            st.caption(
                f"ℹ️ {t('sensor_min_date_hint', sensor=sensor_cfg['short_name'], year=sensor_cfg['min_date'].year)}"
            )

        _touch_selector(
            t("interval_label"),
            options=INTERVAL_KEYS,
            default=str(st.session_state.get("temporal_interval", INTERVAL_ALL)),
            format_func=lambda k, _lang=lang: t(INTERVAL_I18N[k], lang=_lang),
            help=t("interval_help"),
            key="sidebar_interval_select",
            on_change=_sync_interval_state,
            args=("sidebar_interval_select",),
            use_pills=True,
        )
        active_interval = str(st.session_state.get("temporal_interval", INTERVAL_ALL))
        if active_interval == INTERVAL_YEARLY:
            _touch_selector(
                t("yearly_season_label"),
                options=list(YEARLY_SEASON_KEYS),
                default=str(st.session_state.get("yearly_target_season", SEASON_SUMMER)),
                format_func=lambda k, _lang=lang: t(YEARLY_SEASON_I18N[k], lang=_lang),
                help=t("yearly_season_help"),
                key="sidebar_yearly_season",
                on_change=_sync_yearly_season_state,
                args=("sidebar_yearly_season",),
                use_pills=True,
            )

        if sensor == SENSOR_S1:
            max_cloud_cover = 100
        else:
            max_cloud_cover = st.slider(
                t("max_cloud_label"),
                min_value=0,
                max_value=100,
                step=1,
                key="sidebar_max_cloud",
                on_change=_sync_max_cloud_state,
                args=("sidebar_max_cloud",),
            )

        # --- 3. Kompakten zložljiv razdelek: Napredni filtri ---
        with st.expander(t("sb_adv_filters"), expanded=False):
            exclude_l7_slc_off = st.checkbox(
                t("l7_exclude_label"),
                help=t("l7_exclude_help"),
                key="sidebar_l7_slc_exclude",
                on_change=_sync_l7_slc_state,
                args=("sidebar_l7_slc_exclude",),
            )
            st.session_state["exclude_l7_slc_off"] = bool(exclude_l7_slc_off)

            exclude_partial_swaths = st.checkbox(
                t("exclude_partial_label"),
                help=t("exclude_partial_help"),
                key="sidebar_exclude_partial",
                on_change=_sync_partial_filter_state,
                args=("sidebar_exclude_partial",),
            )

            expand_pan_sb = st.checkbox(
                t("expand_pan_label"),
                help=t("expand_pan_help"),
                key="sidebar_expand_pan_buffer",
            )
            st.session_state["expand_pan_buffer"] = bool(expand_pan_sb)

            if bbox_3794 is not None and bbox_4326 is not None:
                box_crs, _ = get_best_metric_crs(float(center_lat), float(center_lon), mode="auto")
                st.markdown(
                    f"<div class='coord-box'>"
                    f"<strong>{box_crs} [m]</strong><br>"
                    f"E: {bbox_3794[0]:,.1f} → {bbox_3794[2]:,.1f}<br>"
                    f"N: {bbox_3794[1]:,.1f} → {bbox_3794[3]:,.1f}<br>"
                    f"<strong>EPSG:4326 [°]</strong><br>"
                    f"lon: {bbox_4326[0]:.5f} → {bbox_4326[2]:.5f}<br>"
                    f"lat: {bbox_4326[1]:.5f} → {bbox_4326[3]:.5f}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    can_search = (bbox_4326 is not None) and aoi_valid and dates_valid and (aoi_error is None)

    # =======================================================================
    # GLAVNI DEL APLIKACIJE: KOMPAKTEN NASLOVNI PAS + 5-DELOVNA NAVIGACIJA
    # =======================================================================
    active_polygon_badge = (
        f'<span class="geo-badge">✂️ {t("polygon_clip_badge")}</span>'
        if (aoi_polygon_4326 is not None or st.session_state.get("search_polygon_4326") is not None)
        else ""
    )
    area_pill_val = f"{area_km2:,.1f} km²" if bbox_4326 is not None else "—"
    header_html = (
        f'<div class="geo-header">'
        f'<div class="geo-header-left">'
        f'<div class="geo-header-title">{t("app_title")}</div>'
        f'<div class="geo-header-sub">{t("app_subtitle_short")}</div>'
        f"</div>"
        f'<div class="geo-badges">'
        f'<span class="geo-badge">🛰️ {sensor_cfg["short_name"]}</span>'
        f'<span class="geo-badge geo-badge-slate">📐 {format_crs_label(target_crs, short=True)}</span>'
        f'<span class="geo-badge geo-badge-slate">{selected_resolution} m</span>'
        f'<span class="geo-badge geo-badge-slate">{area_pill_val}</span>'
        f"{active_polygon_badge}"
        f"</div>"
        f"</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

    _pend_last_sess = st.session_state.get("_pending_restore_last_session")
    if isinstance(_pend_last_sess, dict) and isinstance(st.session_state.get("auth_user"), dict):
        saved_at_str = str(_pend_last_sess.get("saved_at") or "")[:16].replace("T", " ")
        with st.container(border=True):
            rc1, rc2, rc3 = st.columns([3.2, 1.3, 0.7], vertical_alignment="center")
            with rc1:
                st.markdown(t("proj_restore_last_banner", lang=lang, saved_at=saved_at_str))
            with rc2:
                if st.button(
                    t("proj_restore_last_btn", lang=lang),
                    type="primary",
                    width="stretch",
                    key="banner_restore_last_sess_btn",
                ):
                    st.session_state["_pending_project_load"] = st.session_state.pop(
                        "_pending_restore_last_session", None
                    )
                    st.session_state["_pending_toast_msg"] = t("proj_restored_session_toast", lang=lang)
                    st.rerun()
            with rc3:
                if st.button(
                    t("proj_restore_last_dismiss", lang=lang),
                    width="stretch",
                    key="banner_dismiss_last_sess_btn",
                ):
                    st.session_state.pop("_pending_restore_last_session", None)
                    st.rerun()

    # 5-Workspace Top Navigation (izvede se SAMO aktivni delovni prostor -> 3–5× hitrejše delovanje!)
    active_ws = _touch_selector(
        t("ws_nav_label"),
        options=WORKSPACE_KEYS,
        default=str(st.session_state.get("active_workspace", "ws_explore")),
        format_func=lambda k, _lang=lang: t(k, lang=_lang),
        key="top_workspace_nav",
        label_visibility="collapsed",
        use_pills=False,
    )
    st.session_state["active_workspace"] = active_ws

    items: list[Any] = st.session_state["stac_items"]
    summary_df: pd.DataFrame = st.session_state["stac_summary_df"]
    search_performed: bool = st.session_state["search_performed"]
    active_sensor: str = st.session_state.get("search_sensor", sensor)
    active_sensor_cfg = SENSOR_META[active_sensor]
    active_bbox = st.session_state["search_bbox_4326"] or bbox_4326
    active_polygon = (
        st.session_state["search_polygon_4326"] if search_performed else aoi_polygon_4326
    )

    # Pripravi oznake posnetkov za vse 3 zavihke (polne + kompaktne za žetone)
    scene_labels: list[str] = []
    scene_pill_labels: list[str] = []
    default_scene_idx: int = 0
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            tile_lbl = _format_tile_label(str(row["mgrs_tile"]), sensor=active_sensor)
            d_str = str(row["date_str"])
            cov_v = float(row["coverage_pct"]) if ("coverage_pct" in row and pd.notna(row["coverage_pct"])) else 100.0
            cov_suffix = f" · {cov_v:.0f}% AOI" if cov_v < 99.5 else ""
            if active_sensor == SENSOR_S1 or pd.isna(row["cloud_cover"]):
                scene_labels.append(
                    t("scene_option_sar_fmt", date=d_str, tile=tile_lbl) + cov_suffix
                )
                scene_pill_labels.append(f"{d_str} · SAR{cov_suffix}")
            else:
                cld_v = float(row["cloud_cover"])
                scene_labels.append(
                    t(
                        "scene_option_fmt",
                        date=d_str,
                        cloud=cld_v,
                        tile=tile_lbl,
                    )
                    + cov_suffix
                )
                scene_pill_labels.append(f"{d_str} · {cld_v:.0f}%{cov_suffix}")
        if active_sensor != SENSOR_S1 and summary_df["cloud_cover"].notna().any():
            cand_idx = int(summary_df["cloud_cover"].reset_index(drop=True).idxmin())
            if 0 <= cand_idx < len(items):
                default_scene_idx = cand_idx

    # =======================================================================
    # WORKSPACE 1: ZEMLJEVID IN POSNETKI (EXPLORE & SCENES)
    # =======================================================================
    if active_ws == "ws_explore":
        map_expanded = (not search_performed) or (
            aoi_mode_key == "draw" and st.session_state.get("drawn_polygon_4326") is None
        )
        if not search_performed:
            st.info(t("onboarding_info"))
            _render_interactive_aoi_section(
                aoi_mode_key=aoi_mode_key,
                center_lat=center_lat,
                center_lon=center_lon,
                bbox_3794=bbox_3794,
                bbox_4326=bbox_4326,
                aoi_polygon_4326=aoi_polygon_4326,
                lang=lang,
            )
        else:
            with st.expander(t("main_map_expander"), expanded=map_expanded):
                _render_interactive_aoi_section(
                    aoi_mode_key=aoi_mode_key,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    bbox_3794=bbox_3794,
                    bbox_4326=bbox_4326,
                    aoi_polygon_4326=aoi_polygon_4326,
                    lang=lang,
                )

        # 1-klik gumb za iskanje posnetkov neposredno pod zemljevidom AOI
        search_clicked = st.button(
            t("btn_search"),
            type="primary",
            width="stretch",
            disabled=not can_search,
            key="ws1_search_btn",
        )
        if search_clicked and bbox_4326 is not None:
            with st.spinner(t("spinner_search", sensor=sensor_cfg["short_name"])):
                try:
                    _execute_stac_search(
                        sensor=sensor,
                        bbox_4326=bbox_4326,
                        bbox_3794=bbox_3794,
                        aoi_polygon_4326=aoi_polygon_4326,
                        area_km2=area_km2,
                        start_date=start_date,
                        end_date=end_date,
                        max_cloud_cover=float(max_cloud_cover),
                        exclude_l7_slc_off=bool(exclude_l7_slc_off),
                        exclude_partial_swaths=bool(exclude_partial_swaths),
                        interval_mode=active_interval,
                        limit=None,
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(t("search_error", err=exc))

        if search_performed:
            if not items or summary_df.empty:
                st.warning(t("no_scenes_warn"))
            else:
                st.subheader(
                    t("t1_found_header", sensor=active_sensor_cfg["short_name"], n=len(items))
                )
                st.caption(_format_interval_summary_caption(summary_df, lang=lang))

                if active_sensor == SENSOR_S1:
                    cloud_col = pd.Series(["— (SAR)"] * len(summary_df))
                else:
                    cloud_col = summary_df["cloud_cover"].round(2)

                cov_col = (
                    summary_df["coverage_pct"].round(1)
                    if "coverage_pct" in summary_df.columns
                    else pd.Series([100.0] * len(summary_df))
                )

                display_df = pd.DataFrame(
                    {
                        t("col_date"): summary_df["date_str"],
                        t("col_coverage"): cov_col,
                        t("col_cloud"): cloud_col,
                        t("col_tile"): summary_df["mgrs_tile"].apply(
                            lambda v: _format_tile_label(str(v), sensor=active_sensor)
                        ),
                        t("col_platform"): summary_df["platform"],
                        t("col_id"): summary_df["id"],
                    }
                )
                st.caption(t("scene_table_tap_hint"))
                tbl_event = st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True,
                    height=min(310, 42 + 35 * len(display_df)),
                    on_select="rerun",
                    selection_mode="single-row",
                    key=f"tab1_scene_table_{active_sensor}",
                )
                sel_tbl_rows = (
                    list(tbl_event.selection.rows)
                    if (tbl_event is not None and hasattr(tbl_event, "selection") and tbl_event.selection)
                    else []
                )
                tbl_sync_key = f"__tab1_tbl_last_row_{active_sensor}"
                if sel_tbl_rows:
                    row_idx = int(sel_tbl_rows[0])
                    if 0 <= row_idx < len(items) and st.session_state.get(tbl_sync_key) != row_idx:
                        st.session_state[f"tab1_scene_select_{active_sensor}"] = row_idx
                        st.session_state[tbl_sync_key] = row_idx
                elif tbl_sync_key in st.session_state:
                    st.session_state[tbl_sync_key] = None

                st.subheader(t("t1_vis_header"))

                is_sar_active = active_sensor == SENSOR_S1
                vis_specs = SAR_VIS_SPECS if is_sar_active else OPTICAL_COMPOSITE_SPECS

                with st.container(border=True):
                    col_sel, col_vis = st.columns([1.35, 0.9])
                    with col_sel:
                        selected_idx = _scene_touch_picker(
                            t("scene_select_label"),
                            num_scenes=len(items),
                            default_idx=default_scene_idx,
                            scene_labels=scene_labels,
                            pill_labels=scene_pill_labels,
                            key=f"tab1_scene_select_{active_sensor}",
                            lang=lang,
                        )
                    with col_vis:
                        vis_code = _touch_selector(
                            t("composite_select_label"),
                            options=list(vis_specs.keys()),
                            default=list(vis_specs.keys())[0],
                            format_func=lambda code, _lang=lang: t(vis_specs[code]["pill_key"], lang=_lang),
                            key=f"tab1_vis_select_{active_sensor}",
                        )

                    vis_spec = vis_specs[vis_code]
                    st.caption(t(vis_spec["desc_key"]))

                    if 0 <= selected_idx < len(summary_df) and "coverage_pct" in summary_df.columns:
                        _sel_cov_val = summary_df.iloc[selected_idx]["coverage_pct"]
                        if pd.notna(_sel_cov_val) and float(_sel_cov_val) < 95.0:
                            st.info(t("swath_partial_notice", cov=float(_sel_cov_val)))

                    load_clicked = st.button(
                        t("btn_load_scene"),
                        type="primary",
                        width="stretch",
                        key="tab1_load_btn",
                    )

                if load_clicked and active_bbox is not None:
                    chosen_item = items[selected_idx]
                    chosen_row = summary_df.iloc[selected_idx]
                    req_bands: list[str] = list(vis_spec["bands"])

                    with st.spinner(
                        t(
                            "spinner_load_scene",
                            bands=", ".join(req_bands),
                            date=chosen_row["date_str"],
                            crs=target_crs,
                            res=selected_resolution,
                        )
                    ):
                        try:
                            st.session_state["loaded_scene_ds"] = None
                            st.session_state["loaded_scene_display_ds"] = None
                            st.session_state["loaded_scene_overlay_cache"] = None
                            st.session_state["loaded_tif_bytes"] = None
                            gc.collect()

                            use_pan_buf = bool(st.session_state.get("expand_pan_buffer", True))
                            if use_pan_buf:
                                load_bbox = expand_bbox_for_panning(
                                    active_bbox, buffer_factor=0.35, max_area_km2=None
                                )
                                ds_display = load_scene(
                                    sensor=active_sensor,
                                    item=chosen_item,
                                    bbox_4326=load_bbox,
                                    bands=req_bands,
                                    target_epsg=target_crs,
                                    resolution=float(selected_resolution),
                                    aoi_polygon_4326=None,
                                )
                                ds_loaded = _clip_raster_to_exact_aoi(
                                    ds_display, active_bbox, active_polygon
                                )
                            else:
                                ds_loaded = load_scene(
                                    sensor=active_sensor,
                                    item=chosen_item,
                                    bbox_4326=active_bbox,
                                    bands=req_bands,
                                    target_epsg=target_crs,
                                    resolution=float(selected_resolution),
                                    aoi_polygon_4326=active_polygon,
                                )
                                ds_display = ds_loaded

                            var_names = (
                                list(vis_spec["var_names"]) if is_sar_active else list(req_bands)
                            )
                            _check_vars = [v for v in var_names if v in ds_loaded.data_vars] or list(
                                ds_loaded.data_vars
                            )
                            if _check_vars:
                                _v_mask = np.ones(ds_loaded[_check_vars[0]].shape, dtype=bool)
                                for _vn in _check_vars:
                                    _arr_v = np.asarray(ds_loaded[_vn].values)
                                    _v_mask &= np.isfinite(_arr_v)
                                loaded_valid_pct = float(
                                    100.0 * np.count_nonzero(_v_mask) / max(_v_mask.size, 1)
                                )
                            else:
                                loaded_valid_pct = 0.0

                            tif_bytes = export_to_geotiff_bytes(ds_loaded, target_epsg=target_crs)
                            epsg_num = target_crs.replace("EPSG:", "")
                            prefix = active_sensor_cfg["file_prefix"]
                            tif_filename = (
                                f"{prefix}_{chosen_row['date_str']}_{vis_code}_"
                                f"EPSG{epsg_num}_{selected_resolution}m.tif"
                            )

                            st.session_state["loaded_scene_ds"] = ds_loaded
                            st.session_state["loaded_scene_display_ds"] = ds_display
                            st.session_state["loaded_scene_meta"] = {
                                "sensor": active_sensor,
                                "sensor_name": active_sensor_cfg["short_name"],
                                "date_str": str(chosen_row["date_str"]),
                                "cloud_cover": (
                                    None
                                    if pd.isna(chosen_row["cloud_cover"])
                                    else float(chosen_row["cloud_cover"])
                                ),
                                "mgrs_tile": _format_tile_label(
                                    str(chosen_row["mgrs_tile"]), sensor=active_sensor
                                ),
                                "platform": str(chosen_row["platform"]),
                                "scene_id": str(chosen_row["id"]),
                                "crs": target_crs,
                                "resolution_m": int(selected_resolution),
                                "bands": var_names,
                                "vis_code": vis_code,
                                "area_km2": float(st.session_state["search_area_km2"] or area_km2),
                                "expanded_pan": use_pan_buf,
                                "valid_pct": round(loaded_valid_pct, 1),
                            }
                            st.session_state["loaded_tif_bytes"] = tif_bytes
                            st.session_state["loaded_tif_filename"] = tif_filename
                            if loaded_valid_pct < 5.0:
                                st.warning(
                                    t(
                                        "t1_empty_nodata_warn",
                                        date=str(chosen_row["date_str"]),
                                        pct=loaded_valid_pct,
                                    )
                                )
                            gc.collect()
                        except Exception as exc:
                            st.error(t("load_scene_error", err=exc))

                _render_tab1_scene_viewer_fragment(
                    active_sensor=active_sensor,
                    active_bbox=active_bbox,
                    active_polygon=active_polygon,
                    bbox_3794=bbox_3794,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    lang=lang,
                )

    # =======================================================================
    # WORKSPACE 2: ANALIZE IN RELIEF (INDICES, LST, SNOW, COMPOSITE & DEM)
    # =======================================================================
    elif active_ws == "ws_analysis":
        ws2_idx_tab, ws2_comp_tab, ws2_rel_tab = st.tabs(
            [t("ws2_sub_indices"), t("ws2_sub_composite"), t("ws2_sub_relief")]
        )

        # -------------------------------------------------------------------
        # PODZAVIHEK 2.1: SPEKTRALNI INDEKSI, NDSI SNEG, LST °C, SAR & RABA TAL
        # -------------------------------------------------------------------
        with ws2_idx_tab:
            if not search_performed or not items or summary_df.empty:
                st.subheader(t("t2_sar_header") if active_sensor == SENSOR_S1 else t("t2_header"))
                st.info(t("need_search_first"))
                with st.container(border=True):
                    st.markdown(f"**{t('snow_presets_header')}**")
                    st.caption(t("snow_presets_hint"))
                    sp_cols = st.columns(3)
                    for sp_col, (sp_key, sp_spec) in zip(sp_cols, SNOW_GLACIER_PRESETS.items()):
                        with sp_col:
                            st.button(
                                t(sp_spec["label_key"]),
                                width="stretch",
                                key=f"tab2_empty_snow_preset_{sp_key}",
                                on_click=_apply_snow_glacier_preset,
                                args=(sp_key,),
                            )

            # --- Način 2A: Sentinel-1 SAR Water / Flood Mask ---
            elif active_sensor == SENSOR_S1:
                st.subheader(t("t2_sar_header"))
                st.caption(t("t2_sar_caption"))

                with st.container(border=True):
                    col_s1_sc, col_s1_btn = st.columns([1.45, 0.85], vertical_alignment="bottom")
                    with col_s1_sc:
                        t2_sar_idx = _scene_touch_picker(
                            t("scene_select_label"),
                            num_scenes=len(items),
                            default_idx=default_scene_idx,
                            scene_labels=scene_labels,
                            pill_labels=scene_pill_labels,
                            key="tab2_sar_scene_select",
                            lang=lang,
                        )
                    with col_s1_btn:
                        t2_sar_clicked = st.button(
                            t("btn_compute_sar_water"),
                            type="primary",
                            width="stretch",
                            key="tab2_sar_compute_btn",
                        )

                if t2_sar_clicked and active_bbox is not None:
                    chosen_item = items[t2_sar_idx]
                    chosen_row = summary_df.iloc[t2_sar_idx]
                    with st.spinner(
                        t("spinner_index", bands="VV [dB]", idx="SAR Water Mask", date=chosen_row["date_str"])
                    ):
                        try:
                            st.session_state["tab2_sar_ds"] = None
                            st.session_state["tab2_sar_display_ds"] = None
                            gc.collect()

                            use_pan_buf_s1 = bool(st.session_state.get("expand_pan_buffer", True))
                            if use_pan_buf_s1:
                                s1_load_bbox = expand_bbox_for_panning(
                                    active_bbox, buffer_factor=0.35, max_area_km2=None
                                )
                                ds_vv_disp = load_scene(
                                    sensor=SENSOR_S1,
                                    item=chosen_item,
                                    bbox_4326=s1_load_bbox,
                                    bands=["vv"],
                                    target_epsg=target_crs,
                                    resolution=float(selected_resolution),
                                    aoi_polygon_4326=None,
                                )
                                ds_vv = _clip_raster_to_exact_aoi(
                                    ds_vv_disp, active_bbox, active_polygon
                                )
                            else:
                                ds_vv = load_scene(
                                    sensor=SENSOR_S1,
                                    item=chosen_item,
                                    bbox_4326=active_bbox,
                                    bands=["vv"],
                                    target_epsg=target_crs,
                                    resolution=float(selected_resolution),
                                    aoi_polygon_4326=active_polygon,
                                )
                                ds_vv_disp = ds_vv

                            st.session_state["tab2_sar_ds"] = ds_vv
                            st.session_state["tab2_sar_display_ds"] = ds_vv_disp
                            st.session_state["tab2_sar_meta"] = {
                                "date_str": str(chosen_row["date_str"]),
                                "mgrs_tile": _format_tile_label(str(chosen_row["mgrs_tile"]), sensor=SENSOR_S1),
                                "crs": target_crs,
                                "resolution_m": int(selected_resolution),
                                "scene_id": str(chosen_row["id"]),
                            }
                            gc.collect()
                        except Exception as exc:
                            st.error(t("index_error", err=exc))

                _render_tab2_sar_viewer_fragment(
                    active_bbox=active_bbox,
                    active_polygon=active_polygon,
                    bbox_3794=bbox_3794,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    lang=lang,
                )

            # --- Način 2B: Optični spektralni indeksi, NDSI sneg, LST °C + Spektralni podpis & Raba tal ---
            else:
                t2_opt_indices = _get_optical_index_keys(active_sensor)
                with st.container(border=True):
                    col_s2, col_i2 = st.columns([1.35, 0.9])
                    with col_s2:
                        t2_scene_idx = _scene_touch_picker(
                            t("scene_select_label"),
                            num_scenes=len(items),
                            default_idx=default_scene_idx,
                            scene_labels=scene_labels,
                            pill_labels=scene_pill_labels,
                            key=f"tab2_scene_select_{active_sensor}",
                            lang=lang,
                        )
                    with col_i2:
                        t2_index_key = _touch_selector(
                            t("index_select_label"),
                            options=t2_opt_indices,
                            default=t2_opt_indices[0],
                            format_func=lambda k, _l=lang: _fmt_index_pill(k, _l),
                            key="tab2_index_select",
                            use_pills=True,
                        )

                    st.caption(
                        f"{t(f'idx_{t2_index_key}_desc')} — `{SPECTRAL_INDICES[t2_index_key]['formula']}`"
                    )

                    # Hitri 1-klik ledeniški / snežni primeri (Triglav/Kanin, Pasterze, Aletsch)
                    with st.expander(t("snow_presets_header"), expanded=(t2_index_key == "NDSI")):
                        st.caption(t("snow_presets_hint"))
                        sp_c1, sp_c2, sp_c3 = st.columns(3)
                        for sp_col, (sp_key, sp_spec) in zip(
                            (sp_c1, sp_c2, sp_c3), SNOW_GLACIER_PRESETS.items()
                        ):
                            with sp_col:
                                st.button(
                                    t(sp_spec["label_key"]),
                                    width="stretch",
                                    key=f"tab2_idx_snow_preset_{sp_key}",
                                    on_click=_apply_snow_glacier_preset,
                                    args=(sp_key,),
                                )

                    t2_ndsi_thr = 0.40
                    t2_use_dem_snow = True
                    if t2_index_key == "NDSI":
                        sn_c1, sn_c2 = st.columns([1.15, 1.0], vertical_alignment="center")
                        with sn_c1:
                            t2_ndsi_thr = st.slider(
                                t("snow_ndsi_thr_label"),
                                min_value=0.20,
                                max_value=0.65,
                                value=0.40,
                                step=0.05,
                                help=t("snow_ndsi_thr_help"),
                                key="tab2_ndsi_thresh",
                            )
                        with sn_c2:
                            t2_use_dem_snow = st.checkbox(
                                t("snow_use_dem_label"),
                                value=True,
                                help=t("snow_use_dem_help"),
                                key="tab2_ndsi_use_dem",
                            )

                    col_scl2, col_btn2 = st.columns([1.15, 1.0], vertical_alignment="center")
                    with col_scl2:
                        t2_mask_clouds = st.checkbox(
                            t("scl_toggle_label"),
                            value=True,
                            help=t("scl_toggle_help"),
                            key="tab2_scl_mask",
                        )
                    with col_btn2:
                        t2_compute_clicked = st.button(
                            t("btn_compute_index"),
                            type="primary",
                            width="stretch",
                            key="tab2_compute_btn",
                        )

                if t2_compute_clicked and active_bbox is not None:
                    chosen_item = items[t2_scene_idx]
                    chosen_row = summary_df.iloc[t2_scene_idx]
                    req_bands = get_required_bands_for_sensor(
                        active_sensor, t2_index_key, include_cloud_mask=t2_mask_clouds
                    )
                    if t2_index_key == "NDSI" and "B08" not in req_bands:
                        req_bands.append("B08")

                    with st.spinner(
                        t("spinner_index", bands=", ".join(req_bands), idx=t2_index_key, date=chosen_row["date_str"])
                    ):
                        try:
                            st.session_state["tab2_index_da"] = None
                            st.session_state["tab2_index_display_da"] = None
                            st.session_state["tab2_index_overlay_cache"] = None
                            st.session_state["tab2_tif_bytes"] = None
                            st.session_state["tab2_snow_mask_da"] = None
                            st.session_state["tab2_snow_display_da"] = None
                            st.session_state["tab2_snow_stats"] = None
                            st.session_state["tab2_snow_tif_bytes"] = None
                            st.session_state["tab2_snow_tif_filename"] = None
                            gc.collect()

                            use_pan_buf_t2 = bool(st.session_state.get("expand_pan_buffer", True))
                            if use_pan_buf_t2:
                                t2_load_bbox = expand_bbox_for_panning(
                                    active_bbox, buffer_factor=0.35, max_area_km2=None
                                )
                                ds_raw = load_scene(
                                    sensor=active_sensor,
                                    item=chosen_item,
                                    bbox_4326=t2_load_bbox,
                                    bands=req_bands,
                                    target_epsg=target_crs,
                                    resolution=float(selected_resolution),
                                    aoi_polygon_4326=None,
                                )
                                ds_masked, _ = apply_cloud_mask(
                                    ds_raw, mask_clouds=t2_mask_clouds, sensor=active_sensor
                                )
                                index_display_da = compute_index(ds_masked, t2_index_key)
                                index_da = _clip_raster_to_exact_aoi(
                                    index_display_da, active_bbox, active_polygon
                                )
                            else:
                                ds_raw = load_scene(
                                    sensor=active_sensor,
                                    item=chosen_item,
                                    bbox_4326=active_bbox,
                                    bands=req_bands,
                                    target_epsg=target_crs,
                                    resolution=float(selected_resolution),
                                    aoi_polygon_4326=active_polygon,
                                )
                                ds_masked, _ = apply_cloud_mask(
                                    ds_raw, mask_clouds=t2_mask_clouds, sensor=active_sensor
                                )
                                index_da = compute_index(ds_masked, t2_index_key)
                                index_display_da = index_da

                            snow_mask_da_res = None
                            snow_display_da_res = None
                            snow_stats_res = None
                            snow_tif_bytes_res = None
                            snow_tif_fname_res = None
                            if t2_index_key == "NDSI":
                                dem_da_snow = (
                                    load_copernicus_dem(
                                        bbox_4326=active_bbox,
                                        target_epsg=target_crs,
                                        resolution=30.0,
                                        aoi_polygon_4326=active_polygon,
                                    )
                                    if t2_use_dem_snow
                                    else None
                                )
                                if use_pan_buf_t2:
                                    snow_display_da_res, _ = compute_snow_glacier_stats(
                                        ds_masked,
                                        dem_da=None,
                                        ndsi_thresh=float(t2_ndsi_thr),
                                        resolution_m=float(selected_resolution),
                                    )
                                    ds_aoi_snow = _clip_raster_to_exact_aoi(
                                        ds_masked, active_bbox, active_polygon
                                    )
                                    snow_mask_da_res, snow_stats_res = compute_snow_glacier_stats(
                                        ds_aoi_snow,
                                        dem_da=dem_da_snow,
                                        ndsi_thresh=float(t2_ndsi_thr),
                                        resolution_m=float(selected_resolution),
                                    )
                                else:
                                    snow_mask_da_res, snow_stats_res = compute_snow_glacier_stats(
                                        ds_masked,
                                        dem_da=dem_da_snow,
                                        ndsi_thresh=float(t2_ndsi_thr),
                                        resolution_m=float(selected_resolution),
                                    )
                                    snow_display_da_res = snow_mask_da_res
                                snow_tif_bytes_res = export_to_geotiff_bytes(
                                    snow_mask_da_res, target_epsg=target_crs
                                )
                                epsg_num_sn = target_crs.replace("EPSG:", "")
                                prefix_sn = active_sensor_cfg["file_prefix"]
                                snow_tif_fname_res = (
                                    f"{prefix_sn}_{chosen_row['date_str']}_SNOW_MASK_NDSI_"
                                    f"EPSG{epsg_num_sn}_{selected_resolution}m.tif"
                                )
                                del dem_da_snow

                            stats = compute_index_stats(index_da, resolution_m=float(selected_resolution))
                            del ds_raw, ds_masked
                            gc.collect()

                            tif_bytes = export_to_geotiff_bytes(index_da, target_epsg=target_crs)
                            epsg_num = target_crs.replace("EPSG:", "")
                            prefix = active_sensor_cfg["file_prefix"]
                            tif_filename = (
                                f"{prefix}_{chosen_row['date_str']}_{t2_index_key}_"
                                f"EPSG{epsg_num}_{selected_resolution}m.tif"
                            )

                            st.session_state["tab2_index_da"] = index_da
                            st.session_state["tab2_index_display_da"] = index_display_da
                            st.session_state["tab2_index_stats"] = stats
                            st.session_state["tab2_index_meta"] = {
                                "sensor": active_sensor,
                                "sensor_name": active_sensor_cfg["short_name"],
                                "index_key": t2_index_key,
                                "date_str": str(chosen_row["date_str"]),
                                "scene_id": str(chosen_row["id"]),
                                "mgrs_tile": _format_tile_label(
                                    str(chosen_row["mgrs_tile"]), sensor=active_sensor
                                ),
                                "crs": target_crs,
                                "resolution_m": int(selected_resolution),
                                "mask_clouds": bool(t2_mask_clouds),
                                "expanded_pan": use_pan_buf_t2,
                            }
                            st.session_state["tab2_tif_bytes"] = tif_bytes
                            st.session_state["tab2_tif_filename"] = tif_filename
                            st.session_state["tab2_snow_mask_da"] = snow_mask_da_res
                            st.session_state["tab2_snow_display_da"] = snow_display_da_res
                            st.session_state["tab2_snow_stats"] = snow_stats_res
                            st.session_state["tab2_snow_tif_bytes"] = snow_tif_bytes_res
                            st.session_state["tab2_snow_tif_filename"] = snow_tif_fname_res
                            gc.collect()
                        except Exception as exc:
                            st.error(t("index_error", err=exc))

                _render_tab2_index_viewer_fragment(
                    active_sensor=active_sensor,
                    active_bbox=active_bbox,
                    active_polygon=active_polygon,
                    bbox_3794=bbox_3794,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    lang=lang,
                )

                # --- Dodatni orodji za posamezen posnetek: Spektralni podpis & Raba tal [ha] ---
                st.divider()
                st.subheader(t("ws2_tools_divider"))

                # 1) Spektralni podpis točke (Spectral Signature Profiler)
                with st.container(border=True):
                    st.markdown(f"#### {t('sig_expander')}")
                    st.caption(t("sig_caption"))
                    sig_chosen_item = items[t2_scene_idx]
                    sig_chosen_row = summary_df.iloc[t2_scene_idx]

                    if st.button(
                        t("btn_load_sig"),
                        type="primary",
                        width="stretch",
                        key="tab2_load_sig_btn",
                    ):
                        with st.spinner(t("spinner_sig", date=sig_chosen_row["date_str"])):
                            try:
                                st.session_state["tab2_sig_ds"] = None
                                gc.collect()
                                sig_bands = [b for b, _, _ in OPTICAL_SIGNATURE_BANDS]
                                if t2_mask_clouds:
                                    sig_bands.append("QA_PIXEL" if active_sensor == SENSOR_LANDSAT else "SCL")

                                ds_sig_raw = load_scene(
                                    sensor=active_sensor,
                                    item=sig_chosen_item,
                                    bbox_4326=active_bbox,
                                    bands=sig_bands,
                                    target_epsg=target_crs,
                                    resolution=float(selected_resolution),
                                    aoi_polygon_4326=active_polygon,
                                )
                                ds_sig_clean, _ = apply_cloud_mask(ds_sig_raw, mask_clouds=t2_mask_clouds)
                                del ds_sig_raw
                                gc.collect()

                                st.session_state["tab2_sig_ds"] = ds_sig_clean
                                st.session_state["tab2_sig_meta"] = {
                                    "sensor": active_sensor,
                                    "sensor_name": active_sensor_cfg["short_name"],
                                    "date_str": str(sig_chosen_row["date_str"]),
                                    "crs": target_crs,
                                    "scene_id": str(sig_chosen_row["id"]),
                                    "resolution_m": int(selected_resolution),
                                }
                            except Exception as exc:
                                st.error(t("sig_error", err=exc))

                    sig_ds: xr.Dataset | None = st.session_state["tab2_sig_ds"]
                    sig_meta: dict[str, Any] | None = st.session_state["tab2_sig_meta"]
                    if (
                        sig_ds is not None
                        and sig_meta is not None
                        and sig_meta.get("sensor") == active_sensor
                    ):
                        x_arr = np.asarray(sig_ds.coords["x"].values, dtype=np.float64)
                        y_arr = np.asarray(sig_ds.coords["y"].values, dtype=np.float64)
                        x_min, x_max = float(x_arr.min()), float(x_arr.max())
                        y_min, y_max = float(y_arr.min()), float(y_arr.max())
                        x_mid = 0.5 * (x_min + x_max)
                        y_mid = 0.5 * (y_min + y_max)
                        step_m = max(1.0, float(sig_meta["resolution_m"]))

                        col_sx, col_sy = st.columns(2)
                        with col_sx:
                            sample_x = st.slider(
                                t("sig_x_label"),
                                min_value=x_min,
                                max_value=x_max if x_max > x_min else x_min + step_m,
                                value=x_mid,
                                step=step_m,
                                format="%.0f",
                                key=f"tab2_sig_x_{sig_meta['scene_id']}",
                            )
                        with col_sy:
                            sample_y = st.slider(
                                t("sig_y_label"),
                                min_value=y_min,
                                max_value=y_max if y_max > y_min else y_min + step_m,
                                value=y_mid,
                                step=step_m,
                                format="%.0f",
                                key=f"tab2_sig_y_{sig_meta['scene_id']}",
                            )

                        sig_df = extract_spectral_signature(sig_ds, sample_x, sample_y)
                        st.caption(
                            t(
                                "sig_snapped_caption",
                                crs=sig_meta["crs"],
                                x=sig_df.attrs["actual_x"],
                                y=sig_df.attrs["actual_y"],
                                date=sig_meta["date_str"],
                            )
                        )
                        if sig_df.empty or sig_df["reflectance"].isna().all():
                            st.warning(t("sig_nodata_warn"))
                        else:
                            sig_fig = _build_spectral_signature_chart(
                                sig_df=sig_df,
                                sensor_name=sig_meta["sensor_name"],
                                date_str=sig_meta["date_str"],
                            )
                            st.plotly_chart(sig_fig, width="stretch", key="tab2_sig_plotly_chart")

                # 2) Hitro razvrščanje rabe tal in površine v hektarjih (Quick Land-Cover Breakdown)
                with st.container(border=True):
                    st.markdown(f"#### {t('lc_expander')}")
                    st.caption(t("lc_caption"))
                    lc_chosen_item = items[t2_scene_idx]
                    lc_chosen_row = summary_df.iloc[t2_scene_idx]

                    if st.button(
                        t("btn_compute_lc"),
                        type="primary",
                        width="stretch",
                        key="tab2_compute_lc_btn",
                    ):
                        with st.spinner(t("spinner_lc", date=lc_chosen_row["date_str"])):
                            try:
                                st.session_state["tab2_lc_df"] = None
                                st.session_state["tab2_lc_tif_bytes"] = None
                                gc.collect()

                                reuse_ds = (
                                    st.session_state["tab2_sig_ds"]
                                    if (
                                        st.session_state["tab2_sig_ds"] is not None
                                        and st.session_state["tab2_sig_meta"] is not None
                                        and st.session_state["tab2_sig_meta"].get("scene_id")
                                        == str(lc_chosen_row["id"])
                                        and st.session_state["tab2_sig_meta"].get("crs") == target_crs
                                        and st.session_state["tab2_sig_meta"].get("resolution_m")
                                        == int(selected_resolution)
                                    )
                                    else None
                                )
                                if reuse_ds is not None:
                                    ds_lc_clean = reuse_ds
                                else:
                                    lc_bands = ["B03", "B04", "B08"]
                                    if t2_mask_clouds:
                                        lc_bands.append(
                                            "QA_PIXEL" if active_sensor == SENSOR_LANDSAT else "SCL"
                                        )
                                    ds_lc_raw = load_scene(
                                        sensor=active_sensor,
                                        item=lc_chosen_item,
                                        bbox_4326=active_bbox,
                                        bands=lc_bands,
                                        target_epsg=target_crs,
                                        resolution=float(selected_resolution),
                                        aoi_polygon_4326=active_polygon,
                                    )
                                    ds_lc_clean, _ = apply_cloud_mask(ds_lc_raw, mask_clouds=t2_mask_clouds)
                                    del ds_lc_raw

                                lc_da, lc_df = compute_quick_land_cover(
                                    ds_lc_clean, resolution_m=float(selected_resolution)
                                )
                                if reuse_ds is None:
                                    del ds_lc_clean
                                lc_tif = export_to_geotiff_bytes(lc_da, target_epsg=target_crs)
                                del lc_da
                                gc.collect()

                                epsg_num = target_crs.replace("EPSG:", "")
                                prefix = active_sensor_cfg["file_prefix"]
                                st.session_state["tab2_lc_df"] = lc_df
                                st.session_state["tab2_lc_meta"] = {
                                    "sensor": active_sensor,
                                    "date_str": str(lc_chosen_row["date_str"]),
                                    "crs": target_crs,
                                    "resolution_m": int(selected_resolution),
                                    "valid_area_ha": float(lc_df.attrs.get("valid_area_ha", 0.0)),
                                }
                                st.session_state["tab2_lc_tif_bytes"] = lc_tif
                                st.session_state["tab2_lc_tif_filename"] = (
                                    f"{prefix}_{lc_chosen_row['date_str']}_LandCover_"
                                    f"EPSG{epsg_num}_{selected_resolution}m.tif"
                                )
                            except Exception as exc:
                                st.error(t("lc_error", err=exc))

                    lc_df_show: pd.DataFrame | None = st.session_state["tab2_lc_df"]
                    lc_meta: dict[str, Any] | None = st.session_state["tab2_lc_meta"]
                    if (
                        lc_df_show is not None
                        and lc_meta is not None
                        and lc_meta.get("sensor") == active_sensor
                    ):
                        if lc_meta["valid_area_ha"] <= 0.0:
                            st.warning(t("t2_no_valid_pixels"))
                        else:
                            lc_table = pd.DataFrame(
                                {
                                    t("lc_col_class"): [
                                        t(f"lc_{ck}", lang=lang) for ck in lc_df_show["class_key"]
                                    ],
                                    t("lc_col_rule"): lc_df_show["rule"],
                                    t("lc_col_area"): lc_df_show["area_ha"].map(lambda v: f"{v:,.2f}"),
                                    t("lc_col_share"): lc_df_show["share_pct"].map(lambda v: f"{v:.1f} %"),
                                }
                            )
                            st.dataframe(lc_table, width="stretch", hide_index=True)

                            lc_bar = _build_land_cover_bar_chart(
                                lc_df=lc_df_show,
                                date_str=lc_meta["date_str"],
                                total_valid_ha=lc_meta["valid_area_ha"],
                                lang=lang,
                            )
                            st.plotly_chart(lc_bar, width="stretch", key="tab2_lc_bar_chart")

                            lc_tif_bytes = st.session_state["tab2_lc_tif_bytes"]
                            if lc_tif_bytes:
                                st.download_button(
                                    label=t("btn_download_lc_tif"),
                                    data=lc_tif_bytes,
                                    file_name=st.session_state["tab2_lc_tif_filename"] or "land_cover.tif",
                                    mime="image/tiff",
                                    width="stretch",
                                )

        # -------------------------------------------------------------------
        # PODZAVIHEK 2.2: BREZOBLAČNI MEDIANI MOZAIK OBDOBJA
        # -------------------------------------------------------------------
        with ws2_comp_tab:
            st.markdown(f"#### {t('comp_header')}")
            st.caption(t("comp_caption"))

            if active_sensor == SENSOR_S1:
                st.info(t("ws2_comp_sar_info"))
            elif not search_performed or not items or summary_df.empty:
                st.info(t("need_search_first"))
            else:
                comp_mode_options = ["RGB_NIR", *INDEX_KEYS]

                def _fmt_comp_mode(code: str, _lang: str = lang) -> str:
                    if code == "RGB_NIR":
                        return t("comp_mode_rgb_nir_pill", lang=_lang)
                    return code

                with st.container(border=True):
                    col_cm, col_cn, col_cbtn = st.columns([1.65, 1.0, 0.95], vertical_alignment="bottom")
                    with col_cm:
                        comp_mode = _touch_selector(
                            t("comp_mode_label"),
                            options=comp_mode_options,
                            default=comp_mode_options[0],
                            format_func=_fmt_comp_mode,
                            key=f"tab2_comp_mode_{active_sensor}",
                            use_pills=True,
                        )
                    with col_cn:
                        avail_comp = len(items)
                        if avail_comp <= 3:
                            comp_n_scenes = avail_comp
                            st.caption(f"{t('comp_scenes_label')} **{avail_comp}**")
                        else:
                            comp_max = min(12, avail_comp)
                            comp_default = min(6, comp_max)
                            comp_n_scenes = st.slider(
                                t("comp_scenes_label"),
                                min_value=3,
                                max_value=comp_max,
                                value=comp_default,
                                step=1,
                                help=t("comp_scenes_help"),
                                key=f"tab2_comp_n_{active_sensor}",
                            )
                    with col_cbtn:
                        comp_clicked = st.button(
                            t("comp_btn_compute"),
                            type="primary",
                            width="stretch",
                            key="tab2_comp_btn",
                        )

                if comp_clicked and active_bbox is not None:
                    comp_prog = st.progress(0.0)
                    comp_status = st.empty()

                    def _on_comp_progress(step: int, total: int, label: str) -> None:
                        frac = min(max(step / max(total, 1), 0.0), 1.0)
                        comp_prog.progress(frac)
                        if step < total:
                            comp_status.caption(t("comp_progress", cur=step + 1, tot=total, date=label))

                    try:
                        st.session_state["tab2_comp_obj"] = None
                        st.session_state["tab2_comp_meta"] = None
                        st.session_state["tab2_comp_tif_bytes"] = None
                        gc.collect()

                        comp_xr, comp_meta = compute_cloud_free_composite(
                            items=items,
                            bbox_4326=active_bbox,
                            mode=comp_mode,
                            target_epsg=target_crs,
                            resolution=float(selected_resolution),
                            sensor=active_sensor,
                            aoi_polygon_4326=active_polygon,
                            max_scenes=int(comp_n_scenes),
                            progress_callback=_on_comp_progress,
                        )
                        comp_prog.empty()
                        comp_status.empty()

                        if comp_xr is None or comp_meta["scenes_used"] == 0:
                            st.warning(t("comp_empty_warn"))
                        else:
                            comp_tif = export_to_geotiff_bytes(comp_xr, target_epsg=target_crs)
                            epsg_num = target_crs.replace("EPSG:", "")
                            prefix = active_sensor_cfg["file_prefix"]
                            comp_fname = (
                                f"{prefix}_CloudFreeMedian_{comp_meta['mode']}_"
                                f"{comp_meta['date_start']}_to_{comp_meta['date_end']}_"
                                f"EPSG{epsg_num}_{selected_resolution}m.tif"
                            )
                            comp_meta["sensor_name"] = active_sensor_cfg["short_name"]
                            comp_meta["crs"] = target_crs
                            comp_meta["resolution_m"] = int(selected_resolution)

                            st.session_state["tab2_comp_obj"] = comp_xr
                            st.session_state["tab2_comp_meta"] = comp_meta
                            st.session_state["tab2_comp_tif_bytes"] = comp_tif
                            st.session_state["tab2_comp_tif_filename"] = comp_fname
                            gc.collect()
                    except Exception as exc:
                        comp_prog.empty()
                        comp_status.empty()
                        st.error(t("comp_error", err=exc))

                comp_obj = st.session_state["tab2_comp_obj"]
                comp_meta_show: dict[str, Any] | None = st.session_state["tab2_comp_meta"]
                if (
                    comp_obj is not None
                    and comp_meta_show is not None
                    and comp_meta_show.get("sensor") == active_sensor
                ):
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric(
                        t("comp_m_scenes"),
                        f"{comp_meta_show['scenes_used']} / {comp_meta_show['scenes_requested']}",
                    )
                    cm2.metric(t("comp_m_obs"), f"{comp_meta_show['median_obs_per_px']:.1f}")
                    cm3.metric(t("m_valid_pct"), f"{comp_meta_show['valid_pct']:.1f} %")
                    cm4.metric(t("m_valid_ha"), f"{comp_meta_show['valid_area_ha']:,.2f} ha")

                    col_vm_c, col_op_c = st.columns([1.65, 1.0], vertical_alignment="bottom")
                    with col_vm_c:
                        comp_view_mode = _touch_selector(
                            t("viewer_mode_label"),
                            options=VIEWER_MODE_KEYS,
                            default="slippy",
                            format_func=lambda k, _lang=lang: t(VIEWER_MODE_I18N[k], lang=_lang),
                            key="tab2_comp_viewer_mode",
                        )
                    with col_op_c:
                        comp_default_op = 100 if comp_meta_show["mode"] == "RGB_NIR" else 85
                        comp_opacity_pct = st.slider(
                            t("opacity_layer_label") if comp_meta_show["mode"] == "RGB_NIR" else t("opacity_index_label"),
                            min_value=0,
                            max_value=100,
                            value=comp_default_op,
                            step=5,
                            disabled=(comp_view_mode != "slippy"),
                            key=f"tab2_comp_opacity_{comp_meta_show['mode']}",
                        )

                    if comp_meta_show["mode"] == "RGB_NIR" and isinstance(comp_obj, xr.Dataset):
                        col_cd, col_cs, col_cg = st.columns([1.3, 0.85, 0.95], vertical_alignment="center")
                        with col_cd:
                            comp_disp = st.radio(
                                t("comp_display_toggle"),
                                options=["RGB", "NIR"],
                                index=0,
                                horizontal=True,
                                format_func=lambda k, _lang=lang: (
                                    t("comp_display_rgb", lang=_lang)
                                    if k == "RGB"
                                    else t("comp_display_nir", lang=_lang)
                                ),
                                key="tab2_comp_rgb_nir_toggle",
                            )
                        with col_cs:
                            comp_stretch = st.checkbox(
                                t("stretch_label"),
                                value=True,
                                key="tab2_comp_stretch",
                            )
                        with col_cg:
                            comp_gamma = st.slider(
                                t("gamma_label"),
                                min_value=0.5,
                                max_value=2.2,
                                value=1.0,
                                step=0.05,
                                key="tab2_comp_gamma",
                            )

                        comp_bands = ["B04", "B03", "B02"] if comp_disp == "RGB" else ["B08", "B04", "B03"]
                        if comp_view_mode == "slippy":
                            st.caption(t("slippy_hint"))
                            c_uri, c_bounds = raster_to_map_overlay_png(
                                comp_obj,
                                src_epsg=comp_meta_show["crs"],
                                is_rgb=True,
                                bands=comp_bands,
                                use_percentile_stretch=comp_stretch,
                                gamma=comp_gamma,
                            )
                            c_lat_c = (
                                0.5 * (active_bbox[1] + active_bbox[3])
                                if active_bbox is not None
                                else center_lat
                            )
                            c_lon_c = (
                                0.5 * (active_bbox[0] + active_bbox[2])
                                if active_bbox is not None
                                else center_lon
                            )
                            slippy_comp = build_satellite_slippy_map(
                                center_lat=c_lat_c,
                                center_lon=c_lon_c,
                                zoom=None,
                                basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
                                overlay_layers=[
                                    {
                                        "name": f"☁️ {comp_disp} Median ({comp_meta_show['date_start']}–{comp_meta_show['date_end']})",
                                        "image": c_uri,
                                        "bounds": c_bounds,
                                        "opacity": float(comp_opacity_pct) / 100.0,
                                        "show": True,
                                        "zindex": 18,
                                    }
                                ],
                                aoi_bbox_4326=active_bbox,
                                aoi_polygon_geojson=active_polygon,
                                aoi_bbox_3794=st.session_state.get("search_bbox_3794") or bbox_3794,
                                aoi_outline_label=t("slippy_aoi_outline"),
                                lang=lang,
                            )
                            st_folium(
                                slippy_comp,
                                height=470,
                                use_container_width=True,
                                returned_objects=[],
                                key=(
                                    f"tab2_comp_rgb_slippy_{comp_disp}_{comp_opacity_pct}_"
                                    f"{comp_stretch}_{comp_gamma:.2f}_{comp_meta_show['date_end']}_"
                                    f"{st.session_state.get('map_basemap', 'satellite')}"
                                ),
                            )
                            st.caption(t("geodetic_separation_caption", crs=comp_meta_show["crs"]))
                        else:
                            rgb_u8 = _compose_rgb_uint8(
                                comp_obj, comp_bands, use_percentile_stretch=comp_stretch, gamma=comp_gamma
                            )
                            title_key = "comp_title_rgb" if comp_disp == "RGB" else "comp_title_nir"
                            comp_title = (
                                f"<b>{comp_meta_show['sensor_name']} — "
                                f"{t(title_key, d1=comp_meta_show['date_start'], d2=comp_meta_show['date_end'], n=comp_meta_show['scenes_used'])}</b><br>"
                                f"<span style='font-size:12px;color:#475569'>"
                                f"{t('fig_res_bands', crs=comp_meta_show['crs'], res=float(comp_meta_show['resolution_m']), bands='-'.join(comp_bands))}</span>"
                            )
                            st.caption(t("viewer_zoom_hint"))
                            fig_comp = render_interactive_raster_plotly(
                                mode="rgb",
                                x_coords=comp_obj.coords["x"].values,
                                y_coords=comp_obj.coords["y"].values,
                                rgb_uint8=rgb_u8,
                                title=comp_title,
                                target_crs=comp_meta_show["crs"],
                            )
                            st.plotly_chart(
                                fig_comp,
                                width="stretch",
                                config=PLOTLY_VIEWER_CONFIG,
                                key="tab2_comp_rgb_plotly_viewer",
                            )
                    elif isinstance(comp_obj, xr.DataArray):
                        c_idx_key = str(comp_meta_show["mode"])
                        c_spec = SPECTRAL_INDICES[c_idx_key]
                        c_stats = compute_index_stats(comp_obj, resolution_m=float(comp_meta_show["resolution_m"]))
                        if comp_view_mode == "slippy":
                            st.caption(t("slippy_hint"))
                            st.markdown(
                                build_horizontal_colorbar_html(
                                    cmap_name=str(c_spec["cmap"]),
                                    vmin=float(c_spec["vmin"]),
                                    vmax=float(c_spec["vmax"]),
                                    label=f"{comp_meta_show['sensor_name']} — {c_idx_key} Median ({comp_meta_show['date_start']}–{comp_meta_show['date_end']})",
                                ),
                                unsafe_allow_html=True,
                            )
                            ci_uri, ci_bounds = raster_to_map_overlay_png(
                                comp_obj,
                                src_epsg=comp_meta_show["crs"],
                                is_rgb=False,
                                cmap_name=str(c_spec["cmap"]),
                                vmin=float(c_spec["vmin"]),
                                vmax=float(c_spec["vmax"]),
                            )
                            c_lat_c = (
                                0.5 * (active_bbox[1] + active_bbox[3])
                                if active_bbox is not None
                                else center_lat
                            )
                            c_lon_c = (
                                0.5 * (active_bbox[0] + active_bbox[2])
                                if active_bbox is not None
                                else center_lon
                            )
                            slippy_ci = build_satellite_slippy_map(
                                center_lat=c_lat_c,
                                center_lon=c_lon_c,
                                zoom=None,
                                basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
                                overlay_layers=[
                                    {
                                        "name": f"☁️ {c_idx_key} Median ({comp_meta_show['date_start']}–{comp_meta_show['date_end']})",
                                        "image": ci_uri,
                                        "bounds": ci_bounds,
                                        "opacity": float(comp_opacity_pct) / 100.0,
                                        "show": True,
                                        "zindex": 18,
                                    }
                                ],
                                aoi_bbox_4326=active_bbox,
                                aoi_polygon_geojson=active_polygon,
                                aoi_bbox_3794=st.session_state.get("search_bbox_3794") or bbox_3794,
                                aoi_outline_label=t("slippy_aoi_outline"),
                                lang=lang,
                            )
                            st_folium(
                                slippy_ci,
                                height=470,
                                use_container_width=True,
                                returned_objects=[],
                                key=(
                                    f"tab2_comp_idx_slippy_{c_idx_key}_{comp_opacity_pct}_"
                                    f"{comp_meta_show['date_end']}_{st.session_state.get('map_basemap', 'satellite')}"
                                ),
                            )
                            st.caption(t("geodetic_separation_caption", crs=comp_meta_show["crs"]))
                        else:
                            comp_title = (
                                f"<b>{comp_meta_show['sensor_name']} — "
                                f"{t('comp_title_idx', idx=c_idx_key, d1=comp_meta_show['date_start'], d2=comp_meta_show['date_end'], n=comp_meta_show['scenes_used'])}</b><br>"
                                f"<span style='font-size:12px;color:#475569'>"
                                f"{comp_meta_show['crs']} | {comp_meta_show['resolution_m']} m | "
                                f"{t('m_mean')}: {c_stats['mean']:.3f} · {t('m_median')}: {c_stats['median']:.3f}</span>"
                            )
                            st.caption(t("viewer_zoom_hint"))
                            fig_comp = render_interactive_raster_plotly(
                                mode="heatmap",
                                x_coords=comp_obj.coords["x"].values,
                                y_coords=comp_obj.coords["y"].values,
                                z_data=comp_obj.values,
                                cmap_name=str(c_spec["cmap"]),
                                zmin=float(c_spec["vmin"]),
                                zmax=float(c_spec["vmax"]),
                                cbar_title=c_idx_key,
                                title=comp_title,
                                target_crs=comp_meta_show["crs"],
                            )
                            st.plotly_chart(
                                fig_comp,
                                width="stretch",
                                config=PLOTLY_VIEWER_CONFIG,
                                key="tab2_comp_idx_plotly_viewer",
                            )

                    st.caption(
                        t(
                            "comp_dates_caption",
                            n=comp_meta_show["scenes_used"],
                            dates=", ".join(comp_meta_show["dates_used"]),
                        )
                    )

                    comp_tif_bytes = st.session_state["tab2_comp_tif_bytes"]
                    if comp_tif_bytes:
                        st.download_button(
                            label=t("comp_btn_download_tif"),
                            data=comp_tif_bytes,
                            file_name=st.session_state["tab2_comp_tif_filename"] or "cloud_free_composite.tif",
                            mime="image/tiff",
                            width="stretch",
                        )

        # -------------------------------------------------------------------
        # PODZAVIHEK 2.3: RELIEF (COPERNICUS DEM 30 m) & VZDOLŽNI PREREZ A→B
        # -------------------------------------------------------------------
        with ws2_rel_tab:
            _render_relief_and_profile_section(
                target_bbox=active_bbox or bbox_4326,
                target_polygon=active_polygon,
                target_crs=target_crs,
                bbox_metric=st.session_state.get("search_bbox_3794") or bbox_3794,
                center_lat=center_lat,
                center_lon=center_lon,
                aoi_valid=aoi_valid or (active_bbox is not None),
                lang=lang,
                key_prefix="tab2_rel",
            )

    # =======================================================================
    # WORKSPACE 3: ČASOVNE VRSTE, SPREMEMBE, TIMELAPSE & NARAVNE NESREČE
    # =======================================================================
    elif active_ws == "ws_temporal":
        # --- Vgrajena nadzorna kartica za senzor, datume, hitre gumbe in osvežitev ---
        with st.container(border=True):
            st.markdown(f"**{t('t3_ctrl_header')}**")
            t3_today = date.today()
            t3_sensor_cur = str(st.session_state["selected_sensor"])

            _touch_selector(
                t("sensor_select_label"),
                options=SENSOR_KEYS,
                default=t3_sensor_cur,
                format_func=lambda s, _lang=lang: t(SENSOR_META[s]["pill_key"], lang=_lang),
                key="tab3_sensor_select",
                on_change=_sync_sensor_state,
                args=("tab3_sensor_select",),
            )

            c_t3_d1, c_t3_d2, c_t3_cld = st.columns(
                [1.0, 1.0, 1.25], vertical_alignment="bottom"
            )
            with c_t3_d1:
                t3_start = st.date_input(
                    t("start_date_label"),
                    min_value=HISTORICAL_MIN_DATE,
                    max_value=t3_today,
                    key="tab3_start_date",
                    on_change=_sync_start_date_state,
                    args=("tab3_start_date",),
                )
            with c_t3_d2:
                t3_end = st.date_input(
                    t("end_date_label"),
                    min_value=HISTORICAL_MIN_DATE,
                    max_value=t3_today,
                    key="tab3_end_date",
                    on_change=_sync_end_date_state,
                    args=("tab3_end_date",),
                )
            with c_t3_cld:
                if t3_sensor_cur == SENSOR_S1:
                    t3_max_cloud = 100
                    st.caption(f"📡 {t('sar_allweather_note')}")
                else:
                    t3_max_cloud = st.slider(
                        t("max_cloud_label"),
                        min_value=0,
                        max_value=100,
                        step=1,
                        key="tab3_max_cloud",
                        on_change=_sync_max_cloud_state,
                        args=("tab3_max_cloud",),
                    )

            # Hitri prednastavljeni časovni gumbi (6m, 2y, 5y, 10y, 1984–danes Landsat)
            p1, p2, p3, p4, p5 = st.columns([0.9, 0.9, 0.9, 0.85, 1.45])
            with p1:
                st.button(
                    t("t3_preset_6m"),
                    width="stretch",
                    key="t3_btn_preset_6m",
                    on_click=_apply_tab3_date_preset,
                    args=("6m",),
                )
            with p2:
                st.button(
                    t("t3_preset_2y"),
                    width="stretch",
                    key="t3_btn_preset_2y",
                    on_click=_apply_tab3_date_preset,
                    args=("2y",),
                )
            with p3:
                st.button(
                    t("t3_preset_5y"),
                    width="stretch",
                    key="t3_btn_preset_5y",
                    on_click=_apply_tab3_date_preset,
                    args=("5y",),
                )
            with p4:
                st.button(
                    t("t3_preset_10y"),
                    width="stretch",
                    key="t3_btn_preset_10y",
                    on_click=_apply_tab3_date_preset,
                    args=("10y",),
                )
            with p5:
                st.button(
                    t("t3_preset_1984"),
                    width="stretch",
                    key="t3_btn_preset_1984",
                    on_click=_apply_tab3_date_preset,
                    args=("1984_landsat",),
                )

            with st.expander(t("snow_presets_header"), expanded=False):
                st.caption(t("snow_presets_hint"))
                t3_sp1, t3_sp2, t3_sp3 = st.columns(3)
                for sp_col, (sp_key, sp_spec) in zip(
                    (t3_sp1, t3_sp2, t3_sp3), SNOW_GLACIER_PRESETS.items()
                ):
                    with sp_col:
                        st.button(
                            t(sp_spec["label_key"]),
                            width="stretch",
                            key=f"tab3_snow_preset_{sp_key}",
                            on_click=_apply_snow_glacier_preset,
                            args=(sp_key,),
                        )

            # Časovni korak vzorčenja + filter delnih preletov (< 90 % AOI)
            c_t3_int, c_t3_cov = st.columns([1.6, 1.0], vertical_alignment="bottom")
            with c_t3_int:
                _touch_selector(
                    t("interval_label"),
                    options=INTERVAL_KEYS,
                    default=str(st.session_state.get("temporal_interval", INTERVAL_ALL)),
                    format_func=lambda k, _lang=lang: t(INTERVAL_I18N[k], lang=_lang),
                    help=t("interval_help"),
                    key="tab3_interval_select",
                    on_change=_sync_interval_state,
                    args=("tab3_interval_select",),
                    use_pills=True,
                )
            with c_t3_cov:
                t3_exclude_partial = st.checkbox(
                    t("exclude_partial_label"),
                    help=t("exclude_partial_help"),
                    key="tab3_exclude_partial",
                    on_change=_sync_partial_filter_state,
                    args=("tab3_exclude_partial",),
                )
                if t3_sensor_cur == SENSOR_LANDSAT:
                    st.checkbox(
                        t("l7_exclude_label"),
                        help=t("l7_exclude_help"),
                        key="tab3_l7_slc_exclude",
                        on_change=_sync_l7_slc_state,
                        args=("tab3_l7_slc_exclude",),
                    )
            t3_interval_cur = str(st.session_state.get("temporal_interval", INTERVAL_ALL))
            if t3_interval_cur == INTERVAL_YEARLY:
                _touch_selector(
                    t("yearly_season_label"),
                    options=list(YEARLY_SEASON_KEYS),
                    default=str(st.session_state.get("yearly_target_season", SEASON_SUMMER)),
                    format_func=lambda k, _lang=lang: t(YEARLY_SEASON_I18N[k], lang=_lang),
                    help=t("yearly_season_help"),
                    key="tab3_yearly_season",
                    on_change=_sync_yearly_season_state,
                    args=("tab3_yearly_season",),
                    use_pills=True,
                )

            t3_dates_ok = t3_start <= t3_end
            if not t3_dates_ok:
                st.error(t("date_order_error"))
            elif t3_start < SENSOR_META[t3_sensor_cur]["min_date"]:
                st.caption(
                    f"ℹ️ {t('sensor_min_date_hint', sensor=SENSOR_META[t3_sensor_cur]['short_name'], year=SENSOR_META[t3_sensor_cur]['min_date'].year)}"
                )

            t3_target_bbox = bbox_4326 if (bbox_4326 is not None and aoi_valid and aoi_error is None) else active_bbox
            t3_can_refresh = (t3_target_bbox is not None) and aoi_valid and t3_dates_ok and (aoi_error is None)

            col_ref_btn, col_ref_stat = st.columns([1.2, 1.8], vertical_alignment="center")
            with col_ref_btn:
                t3_refresh_clicked = st.button(
                    t("t3_btn_refresh"),
                    type="primary",
                    width="stretch",
                    disabled=not t3_can_refresh,
                    key="tab3_refresh_scenes_btn",
                )
            with col_ref_stat:
                if not aoi_valid or t3_target_bbox is None:
                    st.caption(f"⚠️ {t('t3_no_aoi_warn')}")
                elif search_performed and not summary_df.empty:
                    d_min = str(summary_df["date_str"].min())
                    d_max = str(summary_df["date_str"].max())
                    st.caption(
                        f"{t('t3_loaded_status', sensor=active_sensor_cfg['short_name'], n=len(items), d1=d_min, d2=d_max)}  \n"
                        f"{_format_interval_summary_caption(summary_df, lang=lang)}"
                    )
                    exp_yrs = summary_df.attrs.get("expanded_season_years") or []
                    if exp_yrs:
                        st.caption(
                            t(
                                "yearly_season_expanded_note",
                                years=", ".join(str(y) for y in exp_yrs),
                            )
                        )
                else:
                    st.caption(t("need_search_first"))

        if t3_refresh_clicked and t3_target_bbox is not None:
            t3_sensor_cfg = SENSOR_META[t3_sensor_cur]
            with st.spinner(t("spinner_search", sensor=t3_sensor_cfg["short_name"])):
                try:
                    _execute_stac_search(
                        sensor=t3_sensor_cur,
                        bbox_4326=list(t3_target_bbox),
                        bbox_3794=bbox_3794,
                        aoi_polygon_4326=aoi_polygon_4326,
                        area_km2=area_km2,
                        start_date=t3_start,
                        end_date=t3_end,
                        max_cloud_cover=float(t3_max_cloud),
                        exclude_l7_slc_off=bool(st.session_state.get("exclude_l7_slc_off", True)),
                        exclude_partial_swaths=bool(t3_exclude_partial),
                        interval_mode=t3_interval_cur,
                        target_season=str(
                            st.session_state.get("yearly_target_season", SEASON_SUMMER)
                        ),
                        limit=None,
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(t("search_error", err=exc))

        is_sar_t3 = active_sensor == SENSOR_S1
        min_swath_guard = 85.0 if bool(st.session_state.get("exclude_partial_swaths", True)) else 0.0
        t3_opt_indices = _get_optical_index_keys(active_sensor)
        metric_options = SAR_METRIC_KEYS if is_sar_t3 else t3_opt_indices
        metric_label_title = (
            t("sar_metric_select_label") if is_sar_t3 else t("index_select_label")
        )
        has_t3_scenes = bool(search_performed and items and not summary_df.empty)

        def _fmt_metric_pill(code: str, _lang: str = lang) -> str:
            return _fmt_index_pill(code, _lang)

        sub_ts, sub_change, sub_tl, sub_dis = st.tabs(
            [
                t("ws3_sub_ts"),
                t("ws3_sub_change"),
                t("ws3_sub_tl"),
                t("ws3_sub_disasters"),
            ]
        )

        # -------------------------------------------------------------------
        # PODZAVIHEK 3.1: ČASOVNA VRSTA (GRAF)
        # -------------------------------------------------------------------
        with sub_ts:
            if not has_t3_scenes:
                st.info(t("no_scenes_warn") if search_performed else t("need_search_first"))
            else:
                st.caption(t("t3_sec_a_caption"))
                with st.container(border=True):
                    col_ts_idx, col_ts_ep, col_ts_scl = st.columns(
                        [1.35, 0.95, 0.8], vertical_alignment="bottom"
                    )
                    with col_ts_idx:
                        ts_metric_key = _touch_selector(
                            metric_label_title,
                            options=metric_options,
                            default=metric_options[0],
                            format_func=_fmt_metric_pill,
                            key=f"tab3_ts_metric_{active_sensor}",
                            use_pills=True,
                        )
                    with col_ts_ep:
                        total_available = len(items)
                        if total_available <= 5:
                            max_epochs = total_available
                            st.caption(f"{t('t3_epochs_slider')} **{total_available}**")
                        else:
                            upper_ep = min(50, total_available)
                            default_ep = (
                                upper_ep
                                if (t3_interval_cur != INTERVAL_ALL and upper_ep <= 45)
                                else min(20, upper_ep)
                            )
                            max_epochs = st.slider(
                                t("t3_epochs_slider"),
                                min_value=5,
                                max_value=upper_ep,
                                value=default_ep,
                                step=1,
                                help=t("t3_epochs_help"),
                                key=f"tab3_max_epochs_{active_sensor}",
                            )
                    with col_ts_scl:
                        if is_sar_t3:
                            ts_mask_clouds = False
                            ts_clear_thr = 10
                            st.caption(t("sar_allweather_note"))
                        else:
                            ts_mask_clouds = st.checkbox(
                                t("scl_toggle_label"),
                                value=True,
                                help=t("scl_toggle_help"),
                                key="tab3_ts_scl",
                            )
                            ts_clear_thr = st.slider(
                                t("aoi_clear_thr_label"),
                                min_value=20,
                                max_value=95,
                                value=75,
                                step=5,
                                help=t("aoi_clear_thr_help"),
                                key=f"tab3_ts_clear_thr_{active_sensor}",
                            )

                    ts_clicked = st.button(
                        t("btn_compute_ts"),
                        type="primary",
                        width="stretch",
                        key="tab3_ts_btn",
                    )

                if ts_clicked and active_bbox is not None:
                    selected_items = _select_evenly_spaced_items(items, int(max_epochs))
                    prog_bar = st.progress(0.0)
                    status_slot = st.empty()

                    def _on_progress(step: int, total: int, label: str) -> None:
                        frac = min(max(step / max(total, 1), 0.0), 1.0)
                        prog_bar.progress(frac)
                        if step < total:
                            status_slot.caption(t("ts_progress", cur=step + 1, tot=total, date=label))

                    try:
                        ts_df = extract_time_series(
                            items=selected_items,
                            bbox_4326=active_bbox,
                            index_name=ts_metric_key,
                            target_epsg=target_crs,
                            resolution=float(selected_resolution),
                            mask_clouds=ts_mask_clouds,
                            min_valid_pct=float(ts_clear_thr),
                            progress_callback=_on_progress,
                            sensor=active_sensor,
                            aoi_polygon_4326=active_polygon,
                            min_swath_coverage_pct=min_swath_guard,
                            candidates_by_date=summary_df.attrs.get("candidates_by_bucket"),
                        )
                        prog_bar.empty()
                        status_slot.empty()

                        st.session_state["tab3_ts_df"] = ts_df
                        st.session_state["tab3_ts_meta"] = {
                            "sensor": active_sensor,
                            "prefix": active_sensor_cfg["file_prefix"],
                            "index_key": ts_metric_key,
                            "crs": target_crs,
                            "resolution_m": int(selected_resolution),
                            "min_valid_pct": int(ts_clear_thr),
                            "replaced": ts_df.attrs.get("replaced_epochs", []),
                            "skipped": ts_df.attrs.get("skipped_epochs", []),
                        }
                        gc.collect()
                    except Exception as exc:
                        prog_bar.empty()
                        status_slot.empty()
                        st.error(t("ts_error", err=exc))

                ts_df_show: pd.DataFrame | None = st.session_state["tab3_ts_df"]
                ts_meta: dict[str, Any] | None = st.session_state["tab3_ts_meta"]

                if ts_df_show is not None and ts_meta is not None and ts_meta.get("sensor") == active_sensor:
                    replaced_ts = ts_meta.get("replaced", [])
                    skipped = ts_meta.get("skipped", [])
                    thr_used = int(ts_meta.get("min_valid_pct", 75))
                    if replaced_ts:
                        st.success(t("ts_replaced_info", n=len(replaced_ts)))
                        with st.expander(t("ts_replaced_expander"), expanded=False):
                            st.dataframe(pd.DataFrame(replaced_ts), width="stretch", hide_index=True)
                    if skipped:
                        st.info(t("ts_skipped_info_thr", n=len(skipped), thr=thr_used))
                        with st.expander(t("ts_skipped_expander"), expanded=False):
                            st.dataframe(pd.DataFrame(skipped), width="stretch", hide_index=True)

                    if ts_df_show.empty:
                        st.warning(t("ts_empty_warn"))
                    else:
                        ts_unit = (
                            " dB"
                            if is_sar_t3
                            else (" °C" if ts_meta.get("index_key") == "LST" else "")
                        )
                        ts_fig = _build_time_series_chart(
                            ts_df=ts_df_show,
                            metric_label=ts_meta["index_key"],
                            target_crs=ts_meta["crs"],
                            resolution_m=ts_meta["resolution_m"],
                            unit_suffix=ts_unit,
                        )
                        st.plotly_chart(ts_fig, width="stretch", key="tab3_ts_plotly_chart")

                        csv_bytes = ts_df_show.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=t("btn_download_ts_csv"),
                            data=csv_bytes,
                            file_name=(
                                f"{ts_meta['prefix']}_timeseries_{ts_meta['index_key']}_"
                                f"{ts_meta['resolution_m']}m.csv"
                            ),
                            mime="text/csv",
                            width="stretch",
                        )

        # -------------------------------------------------------------------
        # PODZAVIHEK 3.2: PRIMERJAVA 2 ALI 3 DATUMOV (ZAVESA & Δ SPREMEMBE)
        # -------------------------------------------------------------------
        with sub_change:
            st.markdown(f"#### {t('comp3_header')}")
            st.caption(t("comp3_caption"))

            if not has_t3_scenes:
                st.info(t("no_scenes_warn") if search_performed else t("need_search_first"))
            elif len(items) < 2:
                st.info(t("t3_need_two_scenes"))
            else:
                comp_layer_options = (
                    list(SAR_METRIC_KEYS) if is_sar_t3 else ["RGB", "NIR", *t3_opt_indices]
                )

                def _fmt_comp_layer_pill(code: str, _lang: str = lang) -> str:
                    return _fmt_index_pill(code, _lang)

                with st.container(border=True):
                    col_c_idx, col_c_thr, col_c_scl = st.columns(
                        [1.5, 0.9, 0.8], vertical_alignment="bottom"
                    )
                    with col_c_idx:
                        comp_layer_key = _touch_selector(
                            t("comp3_layer_label"),
                            options=comp_layer_options,
                            default=comp_layer_options[0],
                            format_func=_fmt_comp_layer_pill,
                            key=f"tab3_ch_metric_{active_sensor}",
                            use_pills=True,
                        )
                    is_comp_rgb = (not is_sar_t3) and comp_layer_key in ("RGB", "NIR")
                    is_comp_lst = (not is_sar_t3) and comp_layer_key == "LST"
                    with col_c_thr:
                        if is_sar_t3:
                            ch_threshold = st.slider(
                                f"{t('t3_threshold_label')} [dB]",
                                min_value=0.5,
                                max_value=8.0,
                                value=2.0,
                                step=0.25,
                                key="tab3_ch_threshold_sar",
                            )
                        elif is_comp_lst:
                            ch_threshold = st.slider(
                                f"{t('t3_threshold_label')} [°C]",
                                min_value=0.5,
                                max_value=12.0,
                                value=2.0,
                                step=0.5,
                                key="tab3_ch_threshold_lst",
                            )
                        else:
                            ch_threshold = st.slider(
                                t("t3_threshold_label"),
                                min_value=0.05,
                                max_value=0.40,
                                value=0.10,
                                step=0.01,
                                disabled=is_comp_rgb,
                                key="tab3_ch_threshold_opt",
                            )
                    with col_c_scl:
                        if is_sar_t3:
                            ch_mask_clouds = False
                            comp_auto_fallback = False
                            st.caption(t("sar_allweather_note"))
                        else:
                            ch_mask_clouds = st.checkbox(
                                t("scl_toggle_label"),
                                value=True,
                                help=t("scl_toggle_help"),
                                disabled=is_comp_rgb,
                                key="tab3_ch_scl",
                            )
                            comp_auto_fallback = st.checkbox(
                                t("comp3_auto_fallback_label"),
                                value=True,
                                help=t("comp3_auto_fallback_help"),
                                key="tab3_comp_auto_fallback",
                            )

                    picked_raw = _multi_date_touch_picker(
                        t("comp3_dates_label"),
                        num_scenes=len(items),
                        pill_labels=scene_pill_labels,
                        key=f"tab3_comp_dates_{active_sensor}",
                        lang=lang,
                    )
                    # Kronološka razvrstitev izbranih datumov (najstarejši -> najnovejši)
                    picked_sorted = sorted(
                        set(picked_raw),
                        key=lambda i: str(summary_df.iloc[i]["date_str"]),
                    )
                    valid_pick_count = len(picked_sorted) in (2, 3)
                    if not valid_pick_count:
                        st.warning(t("comp3_need_2_3_warn", n=len(picked_sorted)))

                    ch_clicked = st.button(
                        t("comp3_btn_run", n=len(picked_sorted)),
                        type="primary",
                        width="stretch",
                        disabled=(not valid_pick_count),
                        key="tab3_ch_btn",
                    )

                if ch_clicked and active_bbox is not None and valid_pick_count:
                    chosen_groups = [items[i] for i in picked_sorted]
                    chosen_dates = [str(summary_df.iloc[i]["date_str"]) for i in picked_sorted]
                    raw_by_date = st.session_state.get("stac_raw_items_by_date") or {}
                    all_cand_groups = (
                        list(raw_by_date.values())
                        if isinstance(raw_by_date, dict) and raw_by_date
                        else list(items)
                    )
                    cands_by_bucket = summary_df.attrs.get("candidates_by_bucket") or None
                    _cur_season_key = str(
                        st.session_state.get("yearly_target_season", SEASON_SUMMER)
                    )
                    _season_months: tuple[int, ...] | None = None
                    if (
                        str(st.session_state.get("temporal_interval", INTERVAL_ALL))
                        == INTERVAL_YEARLY
                        and _cur_season_key in YEARLY_SEASON_SPECS
                    ):
                        _sm = YEARLY_SEASON_SPECS[_cur_season_key].get("months")
                        if _sm:
                            _season_months = tuple(_sm)

                    with st.spinner(
                        t("comp3_spinner", n=len(chosen_dates), layer=comp_layer_key)
                    ):
                        try:
                            st.session_state["tab3_comp_result"] = None
                            st.session_state["tab3_change_t1"] = None
                            st.session_state["tab3_change_t2"] = None
                            st.session_state["tab3_change_diff"] = None
                            st.session_state["tab3_change_tif_bytes"] = None
                            gc.collect()

                            comp_res = render_multi_date_comparison(
                                item_groups=chosen_groups,
                                date_labels=chosen_dates,
                                bbox_4326=active_bbox,
                                layer_mode=comp_layer_key,
                                target_epsg=target_crs,
                                resolution=float(selected_resolution),
                                sensor=active_sensor,
                                aoi_polygon_4326=active_polygon,
                                mask_clouds=ch_mask_clouds,
                                all_candidates=all_cand_groups,
                                candidates_by_date=cands_by_bucket,
                                auto_cloud_fallback=bool(comp_auto_fallback),
                                min_clear_pct=75.0,
                                same_season_months=_season_months,
                            )
                            resolved_dates = list(comp_res.get("dates") or chosen_dates)
                            diff_da = comp_res.get("diff_da")
                            da_first = comp_res.get("da_first")
                            da_last = comp_res.get("da_last")
                            diff_tif = None
                            diff_fname = None
                            if diff_da is not None and da_first is not None and da_last is not None:
                                diff_tif = export_to_geotiff_bytes(diff_da, target_epsg=target_crs)
                                epsg_num = target_crs.replace("EPSG:", "")
                                prefix = active_sensor_cfg["file_prefix"]
                                diff_fname = (
                                    f"{prefix}_delta_{comp_layer_key}_{resolved_dates[0]}_to_{resolved_dates[-1]}_"
                                    f"EPSG{epsg_num}_{selected_resolution}m.tif"
                                )
                                # Sinhronizacija za Kartografijo (Workspace 4)
                                st.session_state["tab3_change_t1"] = da_first
                                st.session_state["tab3_change_t2"] = da_last
                                st.session_state["tab3_change_diff"] = diff_da
                                st.session_state["tab3_change_meta"] = {
                                    "sensor": active_sensor,
                                    "index_key": comp_layer_key,
                                    "date_t1": resolved_dates[0],
                                    "date_t2": resolved_dates[-1],
                                    "crs": target_crs,
                                    "resolution_m": int(selected_resolution),
                                }
                                st.session_state["tab3_change_tif_bytes"] = diff_tif
                                st.session_state["tab3_change_tif_filename"] = diff_fname

                            comp_res["sensor"] = active_sensor
                            comp_res["crs"] = target_crs
                            comp_res["resolution_m"] = int(selected_resolution)
                            comp_res["tif_bytes"] = diff_tif
                            comp_res["tif_filename"] = diff_fname
                            st.session_state["tab3_comp_result"] = comp_res
                            st.session_state["tab3_swipe_png_t1"] = comp_res["swipe_pngs"][0]
                            st.session_state["tab3_swipe_png_t2"] = comp_res["swipe_pngs"][-1]
                            gc.collect()
                        except Exception as exc:
                            st.error(t("change_error", err=exc))

                _render_comp3_swipe_fragment(
                    active_sensor=active_sensor,
                    ch_threshold=float(ch_threshold),
                    lang=lang,
                    active_bbox=active_bbox,
                    active_polygon=active_polygon,
                    bbox_3794=bbox_3794,
                    center_lat=center_lat,
                    center_lon=center_lon,
                )

        # -------------------------------------------------------------------
        # PODZAVIHEK 3.3: VIZUALNI TIMELAPSE (GIF + DRSNIK DATUMOV)
        # -------------------------------------------------------------------
        with sub_tl:
            st.caption(t("t3_sec_c_caption"))

            if not has_t3_scenes:
                st.info(t("no_scenes_warn") if search_performed else t("need_search_first"))
            elif len(items) < 2:
                st.info(t("tl_need_two_scenes"))
            else:
                if is_sar_t3:
                    tl_layer_options = ["VV", "VH", "VV_VH"]
                else:
                    tl_layer_options = ["RGB", "NIR", *t3_opt_indices]

                def _fmt_tl_layer_pill(code: str, _lang: str = lang) -> str:
                    return _fmt_index_pill(code, _lang)

                with st.container(border=True):
                    col_tl_layer, col_tl_n, col_tl_spd = st.columns(
                        [1.65, 0.9, 0.9], vertical_alignment="bottom"
                    )
                    with col_tl_layer:
                        tl_layer_code = _touch_selector(
                            t("tl_layer_label"),
                            options=tl_layer_options,
                            default=tl_layer_options[0],
                            format_func=_fmt_tl_layer_pill,
                            key=f"tab3_tl_layer_{active_sensor}",
                            use_pills=True,
                        )
                    with col_tl_n:
                        avail_tl = len(items)
                        if avail_tl <= 3:
                            tl_num_frames = avail_tl
                            st.caption(f"{t('tl_frames_label')} **{avail_tl}**")
                        else:
                            tl_max = min(25, avail_tl)
                            tl_default = (
                                min(12, tl_max)
                                if t3_interval_cur == INTERVAL_YEARLY
                                else min(8, tl_max)
                            )
                            tl_num_frames = st.slider(
                                t("tl_frames_label"),
                                min_value=3,
                                max_value=tl_max,
                                value=tl_default,
                                step=1,
                                help=t("tl_frames_help"),
                                key=f"tab3_tl_frames_{active_sensor}",
                            )
                    with col_tl_spd:
                        tl_speed_sec = _touch_selector(
                            t("tl_speed_label"),
                            options=[0.4, 0.8, 1.2],
                            default=0.8,
                            format_func=lambda s: f"{s:.1f} s",
                            key="tab3_tl_speed",
                        )

                    if is_sar_t3:
                        tl_clear_thr = 10
                    else:
                        tl_clear_thr = st.slider(
                            t("aoi_clear_thr_label"),
                            min_value=20,
                            max_value=95,
                            value=75,
                            step=5,
                            help=t("aoi_clear_thr_help"),
                            key=f"tab3_tl_clear_thr_{active_sensor}",
                        )

                    tl_clicked = st.button(
                        t("btn_generate_tl"),
                        type="primary",
                        width="stretch",
                        key="tab3_tl_btn",
                    )

                if tl_clicked and active_bbox is not None:
                    tl_items = _select_evenly_spaced_items(items, int(tl_num_frames))
                    tl_prog = st.progress(0.0)
                    tl_status = st.empty()

                    def _on_tl_progress(step: int, total: int, label: str) -> None:
                        frac = min(max(step / max(total, 1), 0.0), 1.0)
                        tl_prog.progress(frac)
                        if step < total:
                            tl_status.caption(t("tl_progress", cur=step + 1, tot=total, date=label))

                    try:
                        st.session_state["tab3_timelapse_data"] = None
                        gc.collect()

                        tl_result = generate_timelapse_frames(
                            items=tl_items,
                            bbox_4326=active_bbox,
                            layer_mode=tl_layer_code,
                            target_epsg=target_crs,
                            resolution=float(selected_resolution),
                            sensor=active_sensor,
                            aoi_polygon_4326=active_polygon,
                            mask_clouds=(not is_sar_t3),
                            min_valid_pct=float(tl_clear_thr),
                            duration_ms=int(round(tl_speed_sec * 1000)),
                            max_width=700,
                            progress_callback=_on_tl_progress,
                            min_swath_coverage_pct=min_swath_guard,
                            candidates_by_date=summary_df.attrs.get("candidates_by_bucket"),
                        )
                        tl_prog.empty()
                        tl_status.empty()

                        tl_result["duration_ms"] = int(round(tl_speed_sec * 1000))
                        tl_result["prefix"] = active_sensor_cfg["file_prefix"]
                        tl_result["resolution_m"] = int(selected_resolution)
                        tl_result["min_valid_pct"] = int(tl_clear_thr)
                        st.session_state["tab3_timelapse_data"] = tl_result
                        gc.collect()
                    except Exception as exc:
                        tl_prog.empty()
                        tl_status.empty()
                        st.error(t("tl_error", err=exc))

                tl_data: dict[str, Any] | None = st.session_state["tab3_timelapse_data"]
                if tl_data is not None and tl_data.get("sensor") == active_sensor:
                    tl_frames: list[dict[str, Any]] = tl_data.get("frames", [])
                    tl_replaced = tl_data.get("replaced", [])
                    tl_skipped = tl_data.get("skipped", [])
                    tl_thr_used = int(tl_data.get("min_valid_pct", 75))

                    if tl_replaced:
                        st.success(t("ts_replaced_info", n=len(tl_replaced)))
                        with st.expander(t("ts_replaced_expander"), expanded=False):
                            st.dataframe(pd.DataFrame(tl_replaced), width="stretch", hide_index=True)
                    if tl_skipped:
                        st.info(t("ts_skipped_info_thr", n=len(tl_skipped), thr=tl_thr_used))
                        with st.expander(t("ts_skipped_expander"), expanded=False):
                            st.dataframe(pd.DataFrame(tl_skipped), width="stretch", hide_index=True)

                    if not tl_frames:
                        st.warning(t("tl_empty_warn"))
                    else:
                        # Ob spremembi hitrosti ponovno sestavimo GIF iz predpomnjenih PNG sličic
                        desired_ms = int(round(tl_speed_sec * 1000))
                        if tl_data.get("duration_ms") != desired_ms or not tl_data.get("gif_bytes"):
                            tl_data["gif_bytes"] = compile_timelapse_gif(
                                [f["png_bytes"] for f in tl_frames],
                                duration_ms=desired_ms,
                            )
                            tl_data["duration_ms"] = desired_ms

                        col_gif_l, col_gif_c, col_gif_r = st.columns([0.12, 0.76, 0.12])
                        with col_gif_c:
                            st.image(tl_data["gif_bytes"], width="stretch")

                        tl_fname = (
                            f"{tl_data['prefix']}_timelapse_{tl_data['layer_mode']}_"
                            f"{tl_frames[0]['date_str']}_to_{tl_frames[-1]['date_str']}_"
                            f"{tl_data['resolution_m']}m.gif"
                        )
                        st.download_button(
                            label=t("btn_download_tl_gif"),
                            data=tl_data["gif_bytes"],
                            file_name=tl_fname,
                            mime="image/gif",
                            width="stretch",
                        )

                        # Interaktivni drsnik po datumih (frame-by-frame scrubber)
                        st.markdown(f"#### {t('tl_scrubber_header')}")
                        if len(tl_frames) == 1:
                            chosen_frame = tl_frames[0]
                        else:
                            frame_indices = list(range(len(tl_frames)))
                            scrub_idx = st.select_slider(
                                t("tl_scrubber_label"),
                                options=frame_indices,
                                value=len(tl_frames) - 1,
                                format_func=lambda idx: f"{tl_frames[idx]['date_str']} ({tl_frames[idx]['platform']})",
                                key=f"tab3_tl_scrubber_{active_sensor}_{tl_data['layer_mode']}",
                            )
                            chosen_frame = tl_frames[int(scrub_idx)]

                        col_sc_l, col_sc_c, col_sc_r = st.columns([0.12, 0.76, 0.12])
                        with col_sc_c:
                            st.image(chosen_frame["png_bytes"], width="stretch")
                            st.caption(
                                t(
                                    "tl_frame_caption",
                                    date=chosen_frame["date_str"],
                                    platform=chosen_frame["platform"],
                                    valid=chosen_frame["valid_pct"],
                                    scene_id=chosen_frame["scene_id"],
                                )
                            )

        # -------------------------------------------------------------------
        # PODZAVIHEK 3.4: HITRA OCENA NARAVNIH NESREČ (PRED / MED / PO)
        # -------------------------------------------------------------------
        with sub_dis:
            st.subheader(t("dis_header"))
            st.caption(t("dis_intro"))

            with st.container(border=True):
                # 1. Hitri primeri resničnih dogodkov v Sloveniji
                preset_name_key = "name_en" if lang == "en" else "name_sl"
                dis_preset_id = _touch_selector(
                    t("dis_preset_label"),
                    options=DISASTER_PRESET_KEYS,
                    default="custom",
                    format_func=lambda pid, _nk=preset_name_key: SLOVENIA_CASE_STUDIES[pid][_nk],
                    key="tab5_preset_selector",
                )
                preset_cfg = SLOVENIA_CASE_STUDIES[dis_preset_id]

                # Ob izbiri slovenskega primera samodejno nastavi vrsto nesreče in datum dogodka
                if st.session_state.get("tab5_last_preset") != dis_preset_id:
                    st.session_state["tab5_last_preset"] = dis_preset_id
                    if dis_preset_id != "custom":
                        st.session_state["tab5_disaster_type_selector"] = preset_cfg["disaster_type"]
                        st.session_state["tab5_event_date_input"] = preset_cfg["event_date"]

                if "tab5_event_date_input" not in st.session_state:
                    st.session_state["tab5_event_date_input"] = date(2023, 8, 4)

                if dis_preset_id != "custom":
                    full_t_key = "full_title_en" if lang == "en" else "full_title_sl"
                    desc_key = "desc_en" if lang == "en" else "desc_sl"
                    st.info(
                        t(
                            "dis_active_preset_note",
                            title=preset_cfg[full_t_key],
                            desc=preset_cfg[desc_key],
                            lat=float(preset_cfg["lat"]),
                            lon=float(preset_cfg["lon"]),
                            rad=int(preset_cfg["radius_m"]),
                        )
                    )

                # 2. Vrsta nesreče (touch pills)
                dis_type = _touch_selector(
                    t("dis_type_label"),
                    options=DISASTER_TYPE_KEYS,
                    default=DISASTER_FLOOD,
                    format_func=lambda k, _lang=lang: t(DISASTER_TYPE_I18N[k], lang=_lang),
                    key="tab5_disaster_type_selector",
                )

                # 3. Datum dogodka, iskalno okno in prag za poplave
                if dis_type == DISASTER_FLOOD:
                    dc1, dc2, dc3 = st.columns([1.0, 1.0, 1.0])
                else:
                    dc1, dc2 = st.columns([1.0, 1.0])
                    dc3 = None

                with dc1:
                    dis_event_date = st.date_input(
                        t("dis_event_date_label"),
                        min_value=HISTORICAL_MIN_DATE,
                        max_value=date.today(),
                        key="tab5_event_date_input",
                    )
                with dc2:
                    dis_window_days = st.slider(
                        t("dis_window_label"),
                        min_value=30,
                        max_value=120,
                        value=60,
                        step=5,
                        key="tab5_window_days",
                    )
                dis_flood_thr = -16.0
                if dc3 is not None:
                    with dc3:
                        dis_flood_thr = st.slider(
                            t("dis_flood_thr_label"),
                            min_value=-22.0,
                            max_value=-10.0,
                            value=-16.0,
                            step=0.5,
                            key="tab5_flood_thr_db",
                        )

                btn_run_disaster = st.button(
                    t("dis_btn_run"),
                    type="primary",
                    width="stretch",
                    key="tab5_btn_run",
                )

            if btn_run_disaster:
                if dis_preset_id != "custom" and preset_cfg["lat"] is not None:
                    dis_bbox_4326, _dis_bbox_3794 = bbox_from_point_wgs84(
                        float(preset_cfg["lat"]),
                        float(preset_cfg["lon"]),
                        float(preset_cfg["radius_m"]),
                    )
                    dis_poly = None
                    dis_res_m = max(int(selected_resolution), 10)
                else:
                    dis_bbox_4326 = active_bbox or bbox_4326
                    dis_poly = active_polygon
                    dis_res_m = max(int(selected_resolution), 10)

                dis_prog = st.progress(0.0)
                dis_status = st.empty()

                def _on_dis_prog(frac: float, msg: str) -> None:
                    dis_prog.progress(min(max(float(frac), 0.0), 1.0))
                    dis_status.caption(msg)

                try:
                    dis_out = run_disaster_assessment(
                        disaster_type=str(dis_type),
                        bbox_4326=dis_bbox_4326,
                        event_date=dis_event_date,
                        target_crs=target_crs,
                        resolution=int(dis_res_m),
                        search_window_days=int(dis_window_days),
                        flood_threshold_db=float(dis_flood_thr),
                        aoi_polygon_4326=dis_poly,
                        lang=lang,
                        progress_cb=_on_dis_prog,
                    )
                    st.session_state["tab5_disaster_result"] = dis_out
                    st.session_state["tab5_sent_to_carto"] = False
                    dis_prog.empty()
                    dis_status.empty()
                    gc.collect()
                except Exception as exc:
                    dis_prog.empty()
                    dis_status.empty()
                    st.error(t("dis_error", err=exc))

            dis_res_show: dict[str, Any] | None = st.session_state.get("tab5_disaster_result")
            if isinstance(dis_res_show, dict) and dis_res_show.get("class_da") is not None:
                st.divider()
                d_sum = dis_res_show["summary"]
                d_type_show = str(dis_res_show["disaster_type"])
                d_crs_show = str(dis_res_show["crs"])
                d_sensor_cfg = SENSOR_META.get(dis_res_show["sensor"], SENSOR_META[SENSOR_S1])

                # 1. KPI Metric Row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric(
                    t("dis_kpi_dates"),
                    f"{dis_res_show['date_pre']} → {dis_res_show['date_event']}",
                )
                k2.metric(
                    t("dis_kpi_affected_ha"),
                    f"{d_sum['affected_area_ha']:,.2f} ha",
                    delta=f"{d_sum['affected_area_km2']:.3f} km²",
                    delta_color="off",
                )
                k3.metric(
                    t("dis_kpi_affected_pct"),
                    f"{d_sum['affected_pct']:.2f} %",
                )
                k4.metric(
                    t("dis_kpi_crs"),
                    f"{d_crs_show} · {d_sensor_cfg['short_name']}",
                )

                if d_type_show == DISASTER_FLOOD:
                    st.caption(
                        t(
                            "dis_perm_water_note",
                            perm=d_sum.get("permanent_water_ha", 0.0),
                            flood=d_sum.get("newly_flooded_ha", d_sum["affected_area_ha"]),
                            km2=d_sum["affected_area_km2"],
                            thr=d_sum.get("threshold_db", -16.0),
                        )
                    )

                # 2. Interactive Slippy Map Overlay
                st.markdown(f"#### {t('dis_map_header')}")
                dis_opacity_pct = st.slider(
                    t("dis_opacity_label"),
                    min_value=0,
                    max_value=100,
                    value=85,
                    step=5,
                    key="tab5_dis_opacity",
                )
                specs_show = get_class_specs(d_type_show)
                lbl_k = "en" if lang == "en" else "sl"
                swatch_items = [
                    (str(sp[lbl_k]), str(sp["hex"]))
                    for sp in specs_show
                    if sp["affected"] or (d_type_show == DISASTER_FLOOD and sp["code"] == 1)
                ]
                st.markdown(
                    build_horizontal_colorbar_html(
                        cmap_name="Greys",
                        vmin=0.0,
                        vmax=1.0,
                        label=t(DISASTER_TYPE_I18N[d_type_show]),
                        discrete_items=swatch_items,
                    ),
                    unsafe_allow_html=True,
                )
                dis_uri, dis_bounds = raster_to_map_overlay_png(
                    dis_res_show["class_da"],
                    src_epsg=d_crs_show,
                    is_rgb=False,
                    discrete_rgba_map=build_slippy_rgba_map(d_type_show),
                )
                d_bbox = dis_res_show.get("bbox_4326") or active_bbox
                c_lat_d = 0.5 * (d_bbox[1] + d_bbox[3]) if d_bbox is not None else center_lat
                c_lon_d = 0.5 * (d_bbox[0] + d_bbox[2]) if d_bbox is not None else center_lon
                slippy_dis = build_satellite_slippy_map(
                    center_lat=c_lat_d,
                    center_lon=c_lon_d,
                    zoom=None,
                    basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
                    overlay_layers=[
                        {
                            "name": f"🚨 {t(DISASTER_TYPE_I18N[d_type_show])} ({dis_res_show['date_event']})",
                            "image": dis_uri,
                            "bounds": dis_bounds,
                            "opacity": float(dis_opacity_pct) / 100.0,
                            "show": True,
                            "zindex": 18,
                        }
                    ],
                    aoi_bbox_4326=d_bbox,
                    aoi_polygon_geojson=dis_res_show.get("polygon_4326"),
                    aoi_outline_label=t("slippy_aoi_outline"),
                    lang=lang,
                )
                st_folium(
                    slippy_dis,
                    height=470,
                    use_container_width=True,
                    returned_objects=[],
                    key=(
                        f"tab5_dis_slippy_{d_type_show}_{dis_res_show['date_pre']}_"
                        f"{dis_res_show['date_event']}_{dis_opacity_pct}_"
                        f"{st.session_state.get('map_basemap', 'satellite')}"
                    ),
                )
                st.caption(t("geodetic_separation_caption", crs=d_crs_show))

                # 3. Before / After Zoomable Leaflet Curtain Swipe Comparison
                has_dis_zoom_swipe = bool(
                    dis_res_show.get("overlay_uri_pre")
                    and dis_res_show.get("overlay_uri_post")
                    and dis_res_show.get("overlay_bounds_pre")
                    and dis_res_show.get("overlay_bounds_post")
                )
                if has_dis_zoom_swipe or (
                    dis_res_show.get("swipe_png_pre") and dis_res_show.get("swipe_png_post")
                ):
                    st.markdown(f"#### {t('dis_swipe_header')}")
                    if has_dis_zoom_swipe:
                        st.caption(t("swipe_zoom_hint"))
                        dis_swipe_html = build_zoomable_swipe_map(
                            overlay_uri_t1=dis_res_show["overlay_uri_pre"],
                            bounds_t1=dis_res_show["overlay_bounds_pre"],
                            overlay_uri_t2=dis_res_show["overlay_uri_post"],
                            bounds_t2=dis_res_show["overlay_bounds_post"],
                            label_t1=t("dis_swipe_pre_lbl", date=dis_res_show["date_pre"]),
                            label_t2=t("dis_swipe_post_lbl", date=dis_res_show["date_event"]),
                            center_lat=c_lat_d,
                            center_lon=c_lon_d,
                            basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
                            aoi_bbox_4326=d_bbox,
                            aoi_polygon_geojson=dis_res_show.get("polygon_4326"),
                            aoi_outline_label=t("slippy_aoi_outline"),
                            height=470,
                            opacity=1.0,
                            lang=lang,
                        )
                        components.html(dis_swipe_html, height=480, scrolling=False)
                    else:
                        _render_before_after_swipe_html(
                            png_t1_bytes=dis_res_show["swipe_png_pre"],
                            png_t2_bytes=dis_res_show["swipe_png_post"],
                            label_t1=t("dis_swipe_pre_lbl", date=dis_res_show["date_pre"]),
                            label_t2=t("dis_swipe_post_lbl", date=dis_res_show["date_event"]),
                            height=500,
                        )

                # 4. Severity / Area Breakdown Table + Plotly Bar Chart
                st.markdown(f"#### {t('dis_breakdown_header')}")
                bdf: pd.DataFrame = dis_res_show["breakdown_df"]
                col_tbl, col_bar = st.columns([1.05, 1.15], gap="medium")
                with col_tbl:
                    show_tbl = pd.DataFrame(
                        {
                            t("dis_col_class"): bdf["class_label"],
                            t("dis_col_ha"): bdf["area_ha"],
                            t("dis_col_km2"): bdf["area_km2"],
                            t("dis_col_pct"): bdf["pct"],
                        }
                    )
                    st.dataframe(show_tbl, width="stretch", hide_index=True)
                with col_bar:
                    fig_bar = go.Figure(
                        data=[
                            go.Bar(
                                x=bdf["area_ha"],
                                y=bdf["class_label"],
                                orientation="h",
                                marker=dict(color=bdf["hex"].tolist(), line=dict(color="#334155", width=0.8)),
                                text=[f"{v:,.1f} ha ({p:.1f}%)" for v, p in zip(bdf["area_ha"], bdf["pct"])],
                                textposition="auto",
                            )
                        ]
                    )
                    fig_bar.update_layout(
                        title=dict(text=t("dis_chart_title", crs=d_crs_show), font=dict(size=13, color="#0F172A")),
                        xaxis_title=t("dis_col_ha"),
                        yaxis=dict(autorange="reversed"),
                        height=max(240, 56 * len(bdf) + 90),
                        margin=dict(l=12, r=24, t=42, b=36),
                        plot_bgcolor="#FFFFFF",
                        paper_bgcolor="#FFFFFF",
                        font=dict(color="#0F172A", size=11),
                    )
                    st.plotly_chart(fig_bar, width="stretch", key="tab5_dis_breakdown_bar")

                # 5. Export GeoTIFF + 1-Click Send to Cartography (Workspace 4)
                act_c1, act_c2 = st.columns(2)
                with act_c1:
                    st.download_button(
                        label=t("dis_btn_dl_tif"),
                        data=dis_res_show["tif_bytes"],
                        file_name=dis_res_show["tif_filename"],
                        mime="image/tiff",
                        width="stretch",
                        key="tab5_dl_tif",
                    )
                with act_c2:
                    if st.button(
                        t("dis_btn_send_carto"),
                        type="primary",
                        width="stretch",
                        key="tab5_btn_send_carto",
                    ):
                        st.session_state["tab4_pending_source_select"] = "tab5_disaster"
                        st.session_state["tab5_sent_to_carto"] = True
                        st.session_state["pending_workspace"] = "ws_geo"
                        st.rerun()

                if st.session_state.get("tab5_sent_to_carto"):
                    st.success(t("dis_sent_carto_ok"))

    # =======================================================================
    # WORKSPACE 5: KONVERZACIJSKI AI AGENT ZA DALJINSKO ZAZNAVANJE (GEMINI)
    # =======================================================================
    elif active_ws == "ws_ai":
        _auth_u_ws5 = st.session_state.get("auth_user")
        _ws5_logged_in = isinstance(_auth_u_ws5, dict) and bool(_auth_u_ws5.get("username"))

        # Razreši aktivni Gemini model in ključ iz stranske vrstice (brez podvojenih widgetov)
        effective_gemini_model = resolve_gemini_model_id(
            str(st.session_state.get("gemini_model") or DEFAULT_GEMINI_MODEL),
            custom_id=str(st.session_state.get("gemini_custom_model") or ""),
        )
        cur_api_key, _key_src_ws5 = get_active_gemini_api_key(st.session_state.get("tab6_gemini_key"))
        model_status_pill = (
            t("ai_ws5_pill_model", lang=lang, model=effective_gemini_model)
            if cur_api_key
            else t("ai_ws5_pill_local", lang=lang)
        )

        hdr_col, new_col = st.columns([3.2, 1.8], vertical_alignment="bottom")
        with hdr_col:
            st.subheader(t("ai_header"))
            st.caption(t("ai_intro"))
        with new_col:
            pills_html_parts = [f'<span class="geo-badge">{model_status_pill}</span>']
            if _ws5_logged_in:
                pills_html_parts.append(
                    f'<span class="geo-badge">{t("ai_chat_cloud_pill", lang=lang)}</span>'
                )
            st.markdown(
                '<div style="display:flex;flex-wrap:wrap;justify-content:flex-end;gap:4px;margin-bottom:6px;">'
                + "".join(pills_html_parts)
                + "</div>",
                unsafe_allow_html=True,
            )
            if st.button(
                t("ai_chat_new_btn"),
                width="stretch",
                key="tab6_btn_new_chat",
            ):
                st.session_state["ai_chat_messages"] = []
                st.session_state["ai_chat_results"] = {}
                st.session_state["ai_last_analysis_context"] = None
                st.session_state["ai_last_map_layout"] = None
                st.session_state["tab6_ai_result"] = None
                st.session_state["tab6_sent_to_carto"] = False
                st.session_state["ai_pending_chat_prompt"] = None
                st.session_state["ai_pending_rerun_turn"] = None
                st.session_state["ai_editing_turn_idx"] = None
                st.session_state["ai_show_all_turns"] = False
                gc.collect()
                st.rerun()
            if _ws5_logged_in and st.session_state.get("ai_chat_messages"):
                if st.button(
                    t("ai_chat_save_proj_btn", lang=lang),
                    width="stretch",
                    key="tab6_btn_save_chat_proj",
                ):
                    _u5 = str(_auth_u_ws5["username"])
                    _proj_auto_name = f"AI · {loc_label} ({date.today().isoformat()})"
                    _p_pay = _collect_current_workspace_payload(
                        center_lat=float(center_lat),
                        center_lon=float(center_lon),
                        bbox_4326=bbox_4326,
                        aoi_polygon_4326=aoi_polygon_4326,
                        loc_label=str(loc_label),
                    )
                    save_user_project(_u5, _proj_auto_name, "AI Assistant session", _p_pay, sync=True)
                    _maybe_auto_save_session()
                    st.toast(t("proj_saved_ok", lang=lang, name=_proj_auto_name))

        # --- 1. Pozdravna kartica + klikabilni začetni primeri (ko je klepet prazen) ---
        chat_messages: list[dict[str, Any]] = list(st.session_state.get("ai_chat_messages") or [])
        if not chat_messages:
            welcome_html = (
                f'<div style="background:#F0FDFA;border:1px solid #99F6E4;border-left:4px solid #0F766E;'
                f'border-radius:10px;padding:16px 18px;margin:6px 0 14px 0;">'
                f'<div style="font-size:16px;font-weight:700;color:#0F172A;margin-bottom:6px;">'
                f'🛰️ {t("ai_chat_welcome_title")}</div>'
                f'<div style="font-size:13.5px;color:#334155;line-height:1.5;">'
                f'{t("ai_chat_welcome_body")}</div>'
                f'</div>'
            )
            st.markdown(welcome_html, unsafe_allow_html=True)
            st.caption(t("ai_examples_label"))
            ex_cols = st.columns(2)
            ex_lang_k = "en" if lang == "en" else "sl"
            for ex_idx, ex_item in enumerate(AI_EXAMPLE_PROMPTS):
                ex_text = ex_item[ex_lang_k]
                with ex_cols[ex_idx % 2]:
                    if st.button(
                        ex_text,
                        width="stretch",
                        key=f"tab6_ex_{ex_item['id']}",
                    ):
                        clean_ex = ex_text.split(" ", 1)[1] if " " in ex_text else ex_text
                        st.session_state["ai_pending_chat_prompt"] = clean_ex
                        st.rerun()

        # --- 2. Izris zgodovine večkrožnega pogovora (starejši krogi samodejno strnjeni, zadnja 2 odprta) ---
        chat_turns = group_ai_chat_turns(chat_messages)
        total_turns = len(chat_turns)
        older_count = max(0, total_turns - KEEP_EXPANDED_TURNS)
        show_all_turns = False
        if older_count > 0:
            show_all_turns = st.toggle(
                t("ai_chat_show_all_turns", lang=lang, n=older_count),
                key="ai_show_all_turns",
            )

        editing_turn_idx = st.session_state.get("ai_editing_turn_idx")
        chat_results_map = st.session_state.get("ai_chat_results") or {}
        for turn in chat_turns:
            t_idx = int(turn["turn_idx"])
            if t_idx < total_turns - KEEP_EXPANDED_TURNS:
                exp_lbl = build_collapsed_turn_label(
                    turn,
                    chat_results=chat_results_map,
                    lang=lang,
                )
                with st.expander(exp_lbl, expanded=bool(show_all_turns or editing_turn_idx == t_idx)):
                    _render_ai_chat_turn(turn, target_crs=target_crs, lang=lang)
            else:
                _render_ai_chat_turn(turn, target_crs=target_crs, lang=lang)

        # --- 3. Spodnje pripeto vnosno polje za klepet (st.chat_input) + obdelava Uredi/Ponovi ---
        typed_prompt = st.chat_input(
            t("ai_chat_input_placeholder"),
            key="ws5_ai_chat_input",
        )
        pending_rerun = st.session_state.pop("ai_pending_rerun_turn", None)
        pending_starter = st.session_state.pop("ai_pending_chat_prompt", None)

        if isinstance(pending_rerun, dict) and str(pending_rerun.get("prompt") or "").strip():
            incoming_prompt = str(pending_rerun["prompt"]).strip()
            cut_idx = max(0, int(pending_rerun.get("user_msg_idx", len(chat_messages))))
            prior_history = list(st.session_state.get("ai_chat_messages") or [])[:cut_idx]
            restored_ctx = None
            res_store_cur = st.session_state.get("ai_chat_results") or {}
            for prev_m in reversed(prior_history):
                p_rid = prev_m.get("result_id")
                if p_rid and p_rid in res_store_cur:
                    p_obj = res_store_cur[p_rid]
                    if isinstance(p_obj, dict) and isinstance(p_obj.get("analysis_context"), dict):
                        restored_ctx = p_obj["analysis_context"]
                        break
            st.session_state["ai_last_analysis_context"] = restored_ctx
        else:
            incoming_prompt = str(pending_starter or typed_prompt or "").strip()
            prior_history = list(st.session_state.get("ai_chat_messages") or [])

        if incoming_prompt:
            # Zapiši uporabnikovo sporočilo v zgodovino in ga takoj prikaži
            st.session_state["ai_chat_messages"] = prior_history + [
                {"role": "user", "content": incoming_prompt}
            ]
            with st.chat_message("user"):
                st.markdown(incoming_prompt)

            active_key_now, _ = get_active_gemini_api_key(st.session_state.get("tab6_gemini_key"))
            last_ctx = st.session_state.get("ai_last_analysis_context")

            with st.chat_message("assistant"):
                ai_prog = st.progress(0.0)
                ai_status = st.empty()

                def _on_ai_prog(frac: float, msg_txt: str) -> None:
                    ai_prog.progress(min(max(float(frac), 0.0), 1.0))
                    ai_status.caption(msg_txt)

                try:
                    _on_ai_prog(0.05, t("ai_chat_thinking_route"))
                    route_res = route_user_message(
                        incoming_prompt,
                        chat_history=prior_history,
                        last_context=last_ctx,
                        last_map_layout=st.session_state.get("ai_last_map_layout"),
                        api_key=active_key_now,
                        model=effective_gemini_model,
                        lang=lang,
                    )
                    action_now = str(route_res.get("action") or route_res.get("intent") or "chat_qa")

                    if action_now == "generate_map":
                        t_layer_req = str(route_res.get("target_layer") or "latest_ai")
                        target_src = _resolve_ai_target_carto_layer(t_layer_req, lang=lang)
                        fb_note_str = (
                            route_res.get("fallback_note_en" if lang == "en" else "fallback_note_sl")
                            or ""
                        )
                        if target_src is None:
                            st.session_state["ai_chat_messages"].append(
                                {
                                    "role": "assistant",
                                    "content": t("ai_map_no_layer_clarify"),
                                    "action": "ask_clarification",
                                    "plan": None,
                                    "result_id": None,
                                    "fallback_note": fb_note_str,
                                }
                            )
                        else:
                            _on_ai_prog(0.25, t("ai_map_thinking"))
                            prev_layout = st.session_state.get("ai_last_map_layout")
                            merged_cfg = apply_incremental_layout_overrides(
                                prev_layout,
                                route_res.get("layout_overrides"),
                                reset_layout=bool(route_res.get("reset_layout", False)),
                            )
                            if not merged_cfg.get("title"):
                                merged_cfg["title"] = str(target_src.get("default_title") or "")
                            if not merged_cfg.get("subtitle"):
                                merged_cfg["subtitle"] = str(target_src.get("default_subtitle") or "")
                            if not merged_cfg.get("author"):
                                merged_cfg["author"] = t("carto_author_default")
                            if not merged_cfg.get("legend_title"):
                                merged_cfg["legend_title"] = str(target_src.get("legend_title") or "")

                            map_res = render_geodetic_map(
                                target_src,
                                layout_config=merged_cfg,
                                dpi=150,
                                aoi_polygon_geojson=(
                                    target_src["aoi_polygon_4326"]
                                    if "aoi_polygon_4326" in target_src
                                    else active_polygon
                                ),
                                aoi_bbox_4326=target_src.get("aoi_bbox_4326") or active_bbox,
                                aoi_legend_label=t("slippy_aoi_outline"),
                                lang=lang,
                            )
                            st.session_state["ai_last_map_layout"] = merged_cfg
                            _on_ai_prog(0.82, t("ai_chat_thinking_reply"))

                            m_crs = str(target_src.get("crs", target_crs))
                            epsg_tag = m_crs.replace(":", "")
                            m_stem = f"{target_src.get('file_stem', 'AI_Map')}_{epsg_tag}_{str(merged_cfg['paper_format']).lower()}"
                            map_ctx = {
                                "kind": "carto_map",
                                "crs": m_crs,
                                "layer_title": str(target_src.get("default_title") or ""),
                                "scale_denom": int(map_res.get("scale_denom", 10000)),
                                "grid_spacing_m": int(map_res.get("grid_spacing_m", 1000)),
                                "layout_config": merged_cfg,
                            }
                            reply_res = generate_conversational_reply(
                                incoming_prompt,
                                chat_history=prior_history,
                                analysis_context=map_ctx,
                                api_key=active_key_now,
                                model=effective_gemini_model,
                                lang=lang,
                                is_post_analysis=True,
                            )
                            res_store = dict(st.session_state.get("ai_chat_results") or {})
                            res_key = f"ai_map_{len(st.session_state['ai_chat_messages'])}_{datetime.now().strftime('%H%M%S')}"
                            res_store[res_key] = {
                                "kind": "carto_map",
                                "crs": m_crs,
                                "stem": m_stem,
                                "layer_source_id": str(target_src.get("id") or "tab6_ai"),
                                "layout_config": merged_cfg,
                                "map_result": map_res,
                                "tif_bytes": target_src.get("tif_bytes"),
                                "tif_filename": target_src.get("tif_filename") or f"{m_stem}.tif",
                            }
                            if len(res_store) > 6:
                                oldest_k = next(iter(res_store))
                                res_store.pop(oldest_k, None)
                            st.session_state["ai_chat_results"] = res_store

                            fb_map_str = (
                                reply_res.get("fallback_note_en" if lang == "en" else "fallback_note_sl")
                                or fb_note_str
                                or ""
                            )
                            st.session_state["ai_chat_messages"].append(
                                {
                                    "role": "assistant",
                                    "content": str(reply_res.get("text") or ""),
                                    "action": "generate_map",
                                    "plan": None,
                                    "result_id": res_key,
                                    "fallback_note": fb_map_str,
                                }
                            )
                    elif action_now == "run_analysis" and isinstance(route_res.get("plan"), dict):
                        parsed_plan = route_res["plan"]
                        ai_out = execute_ai_plan(
                            parsed_plan,
                            target_epsg=None,
                            geocode_fn=lambda name, _l=lang: geocode_place(name, lang=_l),
                            lang=lang,
                            progress_cb=lambda f, m: _on_ai_prog(0.08 + 0.78 * f, m),
                        )
                        new_ctx = ai_out.get("analysis_context") or build_analysis_context_dict(
                            parsed_plan, ai_out
                        )
                        _on_ai_prog(0.90, t("ai_chat_thinking_reply"))
                        reply_res = generate_conversational_reply(
                            incoming_prompt,
                            chat_history=prior_history,
                            analysis_context=new_ctx,
                            api_key=active_key_now,
                            model=effective_gemini_model,
                            lang=lang,
                            is_post_analysis=True,
                        )

                        # Shrani rezultat v spomin pogovora (največ 6 zadnjih analiz zaradi RAM-a)
                        res_store = dict(st.session_state.get("ai_chat_results") or {})
                        res_key = f"ai_res_{len(st.session_state['ai_chat_messages'])}_{datetime.now().strftime('%H%M%S')}"
                        res_store[res_key] = ai_out
                        if len(res_store) > 6:
                            oldest_k = next(iter(res_store))
                            res_store.pop(oldest_k, None)
                        st.session_state["ai_chat_results"] = res_store
                        st.session_state["ai_last_analysis_context"] = new_ctx
                        st.session_state["ai_last_map_layout"] = None
                        st.session_state["tab6_ai_result"] = ai_out
                        st.session_state["tab6_sent_to_carto"] = False

                        # Sinhroniziraj geokodiran AOI & lokalni auto-UTM CRS (brez pisanja v že ustvarjene widget ključe!)
                        st.session_state["map_lat"] = round(float(ai_out["lat"]), 6)
                        st.session_state["map_lon"] = round(float(ai_out["lon"]), 6)
                        st.session_state["map_radius_m"] = int(parsed_plan["radius_m"])
                        st.session_state["active_location_label"] = str(
                            ai_out.get("resolved_place") or parsed_plan.get("place_name") or ""
                        )
                        st.session_state["active_epsg"] = str(ai_out.get("crs") or "")
                        st.session_state["active_crs_label"] = format_crs_label(
                            str(ai_out.get("crs") or CRS_D96TM), short=True
                        )
                        pending_w = dict(st.session_state.get("_pending_widget_state") or {})
                        pending_w["ws1_map_radius_slider"] = int(parsed_plan["radius_m"])
                        if (
                            not is_in_slovenia(float(ai_out["lat"]), float(ai_out["lon"]))
                            and str(st.session_state.get("sidebar_crs_mode", "auto")) != "auto"
                        ):
                            pending_w["sidebar_crs_mode"] = "auto"
                        st.session_state["_pending_widget_state"] = pending_w

                        fb_note_str = (
                            reply_res.get("fallback_note_en" if lang == "en" else "fallback_note_sl")
                            or route_res.get("fallback_note_en" if lang == "en" else "fallback_note_sl")
                            or ""
                        )
                        st.session_state["ai_chat_messages"].append(
                            {
                                "role": "assistant",
                                "content": str(reply_res.get("text") or ""),
                                "action": "run_analysis",
                                "plan": parsed_plan,
                                "result_id": res_key,
                                "fallback_note": fb_note_str,
                            }
                        )
                    elif action_now == "ask_clarification" and str(route_res.get("clarification_question") or "").strip():
                        fb_note_str = (
                            route_res.get("fallback_note_en" if lang == "en" else "fallback_note_sl")
                            or ""
                        )
                        st.session_state["ai_chat_messages"].append(
                            {
                                "role": "assistant",
                                "content": str(route_res.get("clarification_question") or "").strip(),
                                "action": "ask_clarification",
                                "plan": None,
                                "result_id": None,
                                "fallback_note": fb_note_str,
                            }
                        )
                    else:
                        _on_ai_prog(0.45, t("ai_chat_thinking_reply"))
                        reply_res = generate_conversational_reply(
                            incoming_prompt,
                            chat_history=prior_history,
                            analysis_context=last_ctx,
                            api_key=active_key_now,
                            model=effective_gemini_model,
                            lang=lang,
                            is_post_analysis=False,
                        )
                        fb_note_str = (
                            reply_res.get("fallback_note_en" if lang == "en" else "fallback_note_sl")
                            or route_res.get("fallback_note_en" if lang == "en" else "fallback_note_sl")
                            or ""
                        )
                        st.session_state["ai_chat_messages"].append(
                            {
                                "role": "assistant",
                                "content": str(reply_res.get("text") or ""),
                                "action": "chat_qa",
                                "plan": None,
                                "result_id": None,
                                "fallback_note": fb_note_str,
                            }
                        )

                    ai_prog.empty()
                    ai_status.empty()
                    _maybe_auto_save_session()
                    gc.collect()
                    st.rerun()
                except Exception as exc:
                    ai_prog.empty()
                    ai_status.empty()
                    err_msg = t("ai_error", err=exc)
                    st.session_state["ai_chat_messages"].append(
                        {
                            "role": "assistant",
                            "content": f"⚠️ {err_msg}",
                            "action": "error",
                            "plan": None,
                            "result_id": None,
                            "fallback_note": "",
                        }
                    )
                    st.error(err_msg)
                    st.rerun()

    # =======================================================================
    # WORKSPACE 4: KARTOGRAFIJA (300 DPI) IN NAČRTOVANJE GNSS IZMERE
    # =======================================================================
    elif active_ws == "ws_geo":
        ws4_carto_tab, ws4_gnss_tab = st.tabs(
            [t("ws4_sub_carto"), t("ws4_sub_gnss")]
        )

        # -------------------------------------------------------------------
        # PODZAVIHEK 4.1: GEODETSKA KARTOGRAFIJA (PDF / PNG 300 DPI)
        # -------------------------------------------------------------------
        with ws4_carto_tab:
            _render_cartography_fragment(
                active_sensor=active_sensor,
                active_sensor_cfg=active_sensor_cfg,
                active_bbox=active_bbox,
                active_polygon=active_polygon,
                target_crs=target_crs,
                selected_resolution=int(selected_resolution),
                area_km2=float(area_km2),
                search_performed=search_performed,
                items=items,
                summary_df=summary_df,
                default_scene_idx=int(default_scene_idx),
                lang=lang,
            )

        # -------------------------------------------------------------------
        # PODZAVIHEK 4.2: NAČRTOVANJE GNSS IZMERE (SKYPLOT, DOP & MASKA OVIR)
        # -------------------------------------------------------------------
        with ws4_gnss_tab:
            st.subheader(t("gnss_header"))
            st.caption(t("gnss_intro"))

            # Sinhroniziraj privzeto stojišče z aktivnim središčem AOI
            aoi_c_lat = float(center_lat)
            aoi_c_lon = float(center_lon)
            aoi_sync_sig = (round(aoi_c_lat, 5), round(aoi_c_lon, 5))
            if st.session_state.get("tab7_last_aoi_sig") != aoi_sync_sig:
                st.session_state["tab7_lat_input"] = round(aoi_c_lat, 6)
                st.session_state["tab7_lon_input"] = round(aoi_c_lon, 6)
                st.session_state["tab7_last_aoi_sig"] = aoi_sync_sig

            with st.container(border=True):
                g_lat_cur = float(st.session_state.get("tab7_lat_input", aoi_c_lat))
                g_lon_cur = float(st.session_state.get("tab7_lon_input", aoi_c_lon))
                g_crs_cur, g_crs_lbl = get_best_metric_crs(g_lat_cur, g_lon_cur, mode=str(crs_mode))
                g_e_cur, g_n_cur = transform_coords(g_lon_cur, g_lat_cur, EPSG_WGS84, g_crs_cur)
                st.markdown(
                    t(
                        "gnss_station_banner_crs",
                        crs_label=g_crs_lbl,
                        e=g_e_cur,
                        n=g_n_cur,
                        lat=g_lat_cur,
                        lon=g_lon_cur,
                    )
                )

                gc1, gc2, gc3, gc4 = st.columns([1.0, 1.0, 1.0, 1.0])
                with gc1:
                    gnss_lat = st.number_input(
                        t("gnss_lat_label"),
                        min_value=-85.0,
                        max_value=85.0,
                        step=0.0005,
                        format="%.5f",
                        key="tab7_lat_input",
                    )
                with gc2:
                    gnss_lon = st.number_input(
                        t("gnss_lon_label"),
                        min_value=-180.0,
                        max_value=180.0,
                        step=0.0005,
                        format="%.5f",
                        key="tab7_lon_input",
                    )
                with gc3:
                    gnss_date = st.date_input(
                        t("gnss_date_label"),
                        value=date.today(),
                        min_value=HISTORICAL_MIN_DATE,
                        max_value=date.today() + timedelta(days=365),
                        key="tab7_plan_date",
                    )
                with gc4:
                    gnss_tz = _touch_selector(
                        t("gnss_tz_label"),
                        options=["Europe/Ljubljana", "UTC"],
                        default="Europe/Ljubljana",
                        format_func=lambda z: "CET/CEST" if z == "Europe/Ljubljana" else "UTC",
                        key="tab7_tz_select",
                    )

                gp1, gp2, gp3 = st.columns([1.4, 1.0, 0.9])
                with gp1:
                    gnss_consts = st.multiselect(
                        t("gnss_const_label"),
                        options=list(GNSS_CONSTELLATIONS),
                        default=list(GNSS_CONSTELLATIONS),
                        key="tab7_constellations",
                    )
                with gp2:
                    gnss_cutoff = st.slider(
                        t("gnss_cutoff_label"),
                        min_value=5,
                        max_value=25,
                        value=10,
                        step=1,
                        key="tab7_cutoff_deg",
                    )
                with gp3:
                    gnss_ant_h = st.number_input(
                        t("gnss_ant_label"),
                        min_value=0.0,
                        max_value=20.0,
                        value=2.0,
                        step=0.2,
                        format="%.1f",
                        key="tab7_ant_height",
                    )

                with st.expander(t("gnss_obs_expander"), expanded=False):
                    oe1, oe2 = st.columns(2)
                    with oe1:
                        gnss_use_dem = st.checkbox(t("gnss_use_dem_label"), value=True, key="tab7_use_dem")
                        gnss_canopy_h = st.slider(
                            t("gnss_canopy_label"),
                            min_value=5,
                            max_value=35,
                            value=18,
                            step=1,
                            key="tab7_canopy_h",
                        )
                    with oe2:
                        gnss_use_s2 = st.checkbox(t("gnss_use_s2_label"), value=True, key="tab7_use_s2")
                        gnss_bldg_h = st.slider(
                            t("gnss_building_label"),
                            min_value=4,
                            max_value=40,
                            value=12,
                            step=1,
                            key="tab7_bldg_h",
                        )

                btn_run_gnss = st.button(
                    t("gnss_btn_run"),
                    type="primary",
                    width="stretch",
                    key="tab7_btn_run",
                )

            if btn_run_gnss:
                gnss_prog = st.progress(0.0)
                gnss_status = st.empty()

                def _on_gnss_prog(frac: float, msg: str) -> None:
                    gnss_prog.progress(min(max(float(frac), 0.0), 1.0))
                    gnss_status.caption(msg)

                try:
                    gnss_out = run_gnss_mission_plan(
                        lat_deg=float(gnss_lat),
                        lon_deg=float(gnss_lon),
                        plan_date=gnss_date,
                        constellations=gnss_consts or list(GNSS_CONSTELLATIONS),
                        cutoff_deg=float(gnss_cutoff),
                        ant_height_m=float(gnss_ant_h),
                        canopy_h_m=float(gnss_canopy_h),
                        building_h_m=float(gnss_bldg_h),
                        use_dem=bool(gnss_use_dem),
                        use_s2_obstacles=bool(gnss_use_s2),
                        tz_name=str(gnss_tz),
                        step_minutes=15,
                        target_epsg=target_crs,
                        progress_cb=_on_gnss_prog,
                    )
                    st.session_state["tab7_gnss_result"] = gnss_out
                    gnss_prog.empty()
                    gnss_status.empty()
                    gc.collect()
                except Exception as exc:
                    gnss_prog.empty()
                    gnss_status.empty()
                    st.error(t("gnss_error", err=exc))

            gnss_res: dict[str, Any] | None = st.session_state.get("tab7_gnss_result")
            if isinstance(gnss_res, dict) and gnss_res.get("timeline_df") is not None:
                st.divider()
                g_sum = gnss_res["summary"]
                tl_df: pd.DataFrame = gnss_res["timeline_df"]
                t_labels = tl_df["time_str"].tolist()

                # Ephemeris source badge
                if gnss_res["orbit_source"] == "celestrak_tle":
                    st.success(t("gnss_badge_tle"))
                else:
                    st.info(t("gnss_badge_walker"))

                # 1. KPI Summary Row
                gk1, gk2, gk3, gk4, gk5 = st.columns(5)
                gk1.metric(
                    t("gnss_kpi_vis"),
                    f"{g_sum['avg_vis_real']:.1f} / {g_sum['avg_vis_open']:.1f}",
                )
                gk2.metric(
                    t("gnss_kpi_min_pdop"),
                    f"{g_sum['min_pdop_real']:.2f}" if not np.isnan(g_sum["min_pdop_real"]) else "—",
                )
                gk3.metric(
                    t("gnss_kpi_mean_pdop"),
                    f"{g_sum['mean_pdop_real']:.2f}" if not np.isnan(g_sum["mean_pdop_real"]) else "—",
                )
                gk4.metric(
                    t("gnss_kpi_blocked"),
                    f"{g_sum['sky_blocked_pct']:.1f} %",
                )
                opt_wins = g_sum.get("optimal_windows") or []
                best_win_txt = opt_wins[0] if opt_wins else t("gnss_no_window")
                gk5.metric(
                    t("gnss_kpi_window"),
                    best_win_txt,
                    delta=(f"+{len(opt_wins) - 1} more" if len(opt_wins) > 1 else None),
                    delta_color="off",
                )

                # 2. Chart 1 — Interactive Polar Skyplot (@st.fragment za takojšen drsnik ure)
                _render_gnss_skyplot_fragment(gnss_res=gnss_res, lang=lang)

                # 3. Chart 2 — 24-Hour Visibility & PDOP / HDOP / VDOP Dual-Panel Chart
                st.markdown(f"#### {t('gnss_dop_header')}")
                fig_dop = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.09,
                    subplot_titles=(
                        "Število vidnih satelitov (Odprto nebo vs. Realno z ovirami)"
                        if lang == "sl"
                        else "Visible Satellites (Open Sky vs. Real Obstructed Sky)",
                        "Geometrija DOP (PDOP / HDOP / VDOP — zeleno = optimalno okno PDOP ≤ 2,5)"
                        if lang == "sl"
                        else "DOP Geometry (PDOP / HDOP / VDOP — green = optimal window PDOP ≤ 2.5)",
                    ),
                )
                fig_dop.add_trace(
                    go.Scatter(
                        x=tl_df["time_str"],
                        y=tl_df["visible_open"],
                        mode="lines",
                        name="Odprto nebo" if lang == "sl" else "Open Sky",
                        line=dict(color="#94A3B8", width=1.8, dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
                fig_dop.add_trace(
                    go.Scatter(
                        x=tl_df["time_str"],
                        y=tl_df["visible_real"],
                        mode="lines",
                        fill="tozeroy",
                        fillcolor="rgba(15, 118, 110, 0.14)",
                        name="Realno (z ovirami)" if lang == "sl" else "Real (Obstructed)",
                        line=dict(color="#0F766E", width=2.4),
                    ),
                    row=1,
                    col=1,
                )

                fig_dop.add_trace(
                    go.Scatter(
                        x=tl_df["time_str"],
                        y=tl_df["pdop_real"],
                        mode="lines",
                        name="PDOP (Real)",
                        line=dict(color="#DC2626", width=2.4),
                    ),
                    row=2,
                    col=1,
                )
                fig_dop.add_trace(
                    go.Scatter(
                        x=tl_df["time_str"],
                        y=tl_df["hdop_real"],
                        mode="lines",
                        name="HDOP (Real)",
                        line=dict(color="#0284C7", width=1.8),
                    ),
                    row=2,
                    col=1,
                )
                fig_dop.add_trace(
                    go.Scatter(
                        x=tl_df["time_str"],
                        y=tl_df["vdop_real"],
                        mode="lines",
                        name="VDOP (Real)",
                        line=dict(color="#D97706", width=1.8),
                    ),
                    row=2,
                    col=1,
                )
                fig_dop.add_trace(
                    go.Scatter(
                        x=tl_df["time_str"],
                        y=tl_df["pdop_open"],
                        mode="lines",
                        name="PDOP (Open Sky)",
                        line=dict(color="#94A3B8", width=1.4, dash="dot"),
                    ),
                    row=2,
                    col=1,
                )

                # Highlight optimal measurement windows (PDOP <= 2.5 & visible >= 6)
                opt_flags = (
                    tl_df["pdop_real"].notna() & (tl_df["pdop_real"] <= 2.5) & (tl_df["visible_real"] >= 6)
                ).to_numpy()
                seg_s: int | None = None
                for i_op, flg in enumerate(opt_flags):
                    if flg and seg_s is None:
                        seg_s = i_op
                    elif not flg and seg_s is not None:
                        fig_dop.add_vrect(
                            x0=t_labels[seg_s],
                            x1=t_labels[i_op - 1],
                            fillcolor="rgba(22, 163, 74, 0.14)",
                            line_width=0,
                            row=2,
                            col=1,
                        )
                        seg_s = None
                if seg_s is not None:
                    fig_dop.add_vrect(
                        x0=t_labels[seg_s],
                        x1=t_labels[-1],
                        fillcolor="rgba(22, 163, 74, 0.14)",
                        line_width=0,
                        row=2,
                        col=1,
                    )

                fig_dop.add_hline(y=2.5, line_dash="dash", line_color="#16A34A", line_width=1.2, row=2, col=1)
                fig_dop.update_yaxes(title_text="Sat.", row=1, col=1, gridcolor="#E2E8F0")
                fig_dop.update_yaxes(title_text="DOP", range=[0, 8], row=2, col=1, gridcolor="#E2E8F0")
                fig_dop.update_xaxes(nticks=12, gridcolor="#E2E8F0")
                fig_dop.update_layout(
                    height=520,
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    font=dict(color="#0F172A"),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_dop, width="stretch", key="tab7_dop_plotly")

                # 4. Chart 3 — Local Obstruction Overview (Sentinel-2 10 m + 50 m / 100 m rings)
                st.markdown(f"#### {t('gnss_local_header')}")
                s2_inf = gnss_res.get("s2_info") or {}
                if s2_inf.get("ok") and s2_inf.get("rgb_u8") is not None:
                    rgb_loc = s2_inf["rgb_u8"]
                    obs_loc = s2_inf["obs_cls"]
                    xv_loc = np.asarray(s2_inf["x_coords"], dtype=np.float64)
                    yv_loc = np.asarray(s2_inf["y_coords"], dtype=np.float64)
                    e0_loc = float(s2_inf["station_e"])
                    n0_loc = float(s2_inf["station_n"])

                    # Overlay detected forest (teal) & buildings (amber) subtly onto 10 m RGB
                    rgb_vis = rgb_loc.astype(np.float32).copy()
                    f_mask = obs_loc == 1
                    b_mask = obs_loc == 2
                    rgb_vis[f_mask] = 0.55 * rgb_vis[f_mask] + 0.45 * np.array([15.0, 118.0, 110.0])
                    rgb_vis[b_mask] = 0.55 * rgb_vis[b_mask] + 0.45 * np.array([217.0, 119.0, 6.0])
                    rgb_vis_u8 = np.clip(np.round(rgb_vis), 0, 255).astype(np.uint8)

                    fig_loc = px.imshow(
                        rgb_vis_u8,
                        x=xv_loc,
                        y=yv_loc,
                        origin="upper" if (yv_loc.size > 1 and yv_loc[0] > yv_loc[-1]) else "lower",
                    )
                    theta_ring = np.linspace(0.0, 2.0 * np.pi, 121)
                    for r_ring, r_dash in ((50.0, "dash"), (100.0, "dot")):
                        fig_loc.add_trace(
                            go.Scatter(
                                x=e0_loc + r_ring * np.cos(theta_ring),
                                y=n0_loc + r_ring * np.sin(theta_ring),
                                mode="lines",
                                line=dict(color="#FFFFFF", width=2.0, dash=r_dash),
                                name=f"{int(r_ring)} m",
                            )
                        )
                    fig_loc.add_trace(
                        go.Scatter(
                            x=[e0_loc],
                            y=[n0_loc],
                            mode="markers",
                            marker=dict(size=12, color="#DC2626", symbol="cross-thin", line=dict(width=3, color="#DC2626")),
                            name="GNSS",
                        )
                    )
                    fig_loc.update_layout(
                        title=f"Sentinel-2 10 m ({s2_inf.get('scene_date', '')}) · Teal = Forest · Amber = Buildings",
                        xaxis_title="e [m] (D96/TM)",
                        yaxis_title="n [m] (D96/TM)",
                        yaxis=dict(scaleanchor="x", scaleratio=1.0),
                        height=460,
                        paper_bgcolor="#FFFFFF",
                        font=dict(color="#0F172A"),
                    )
                    st.plotly_chart(fig_loc, width="stretch", key="tab7_local_obs_plotly")
                else:
                    s2_err = s2_inf.get("error")
                    if s2_err:
                        st.warning(f"{t('gnss_local_unavail')} ({s2_err})")
                    else:
                        st.info(t("gnss_local_unavail"))

                # 5. Export GNSS Mission Plan CSV
                st.download_button(
                    label=t("gnss_btn_dl_csv"),
                    data=gnss_res["csv_bytes"],
                    file_name=gnss_res["csv_filename"],
                    mime="text/csv",
                    width="stretch",
                    key="tab7_dl_csv",
                )

    # Prepreči odpiranje virtualne tipkovnice na preostalih dinamičnih selectboxih (tablice/telefoni)
    _inject_tablet_no_keyboard_js()


if __name__ == "__main__":
    main()
