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
from datetime import date, timedelta
from typing import Any, Callable, Sequence

import folium
from folium.plugins import Draw
import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    "geodezija_crs",
    "map_overlay",
    "kartografija",
    "stac_loader",
    "analize",
    "casovne_vrste",
    "nesrece",
    "ai_pomocnik",
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

from ai_pomocnik import (
    AI_EXAMPLE_PROMPTS,
    execute_ai_plan,
    get_active_gemini_api_key,
    parse_eo_request,
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
    extract_sar_metric,
    extract_spectral_signature,
    get_required_bands_for_sensor,
)
from casovne_vrste import (
    compile_timelapse_gif,
    compute_change_detection,
    compute_change_summary,
    compute_cloud_free_composite,
    extract_time_series,
    generate_timelapse_frames,
    render_change_swipe_pair,
)
from geodezija_crs import (
    CRS_D96TM,
    CRS_UTM33N,
    EPSG_D96TM,
    EPSG_WGS84,
    bbox_from_point_d96tm,
    bbox_from_point_wgs84,
    clip_to_aoi_polygon,
    export_to_geotiff_bytes,
    transform_coords,
    validate_aoi_geometry,
    validate_aoi_size,
)
from i18n import t
from kartografija import format_metric_number, render_geodetic_map
from map_overlay import (
    build_horizontal_colorbar_html,
    build_satellite_slippy_map,
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
    SENSOR_LANDSAT,
    SENSOR_S1,
    SENSOR_S2,
    TEMPORAL_INTERVAL_KEYS,
    filter_by_temporal_interval,
    load_scene,
    search_scenes,
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

INDEX_KEYS: list[str] = ["NDVI", "NDWI", "MNDWI", "NDBI", "NBR"]
SAR_METRIC_KEYS: list[str] = ["VV", "VH", "VV_VH"]

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

CARTO_FORMAT_KEYS: list[str] = ["a4_landscape", "a4_portrait", "square"]
CARTO_FORMAT_I18N: dict[str, str] = {
    "a4_landscape": "carto_fmt_a4_land",
    "a4_portrait": "carto_fmt_a4_port",
    "square": "carto_fmt_square",
}
CARTO_GRID_OPTIONS: list[int] = [0, 500, 1000, 2000, 5000]  # 0 = Auto
CARTO_GRID_STYLE_KEYS: list[str] = ["light", "dark"]
CARTO_GRID_STYLE_I18N: dict[str, str] = {
    "light": "carto_grid_light",
    "dark": "carto_grid_dark",
}

DISASTER_TYPE_KEYS: list[str] = list(DISASTER_TYPES)
DISASTER_TYPE_I18N: dict[str, str] = {
    DISASTER_FLOOD: "dis_type_flood",
    DISASTER_WILDFIRE: "dis_type_wildfire",
    DISASTER_FOREST: "dis_type_forest",
}
DISASTER_PRESET_KEYS: list[str] = list(SLOVENIA_CASE_STUDIES.keys())


# ---------------------------------------------------------------------------
# Pomožne funkcije: stanje seje, sinhronizacija senzorja/datumov, iskanje STAC
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Inicializira spremenljivke seje."""
    today = date.today()
    default_start = today - timedelta(days=183)
    defaults: dict[str, Any] = {
        "lang": "sl",
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
        "exclude_l7_slc_off": False,
        "exclude_partial_swaths": True,
        "sidebar_exclude_partial": True,
        "tab3_exclude_partial": True,
        "temporal_interval": INTERVAL_ALL,
        "sidebar_interval_select": INTERVAL_ALL,
        "tab3_interval_select": INTERVAL_ALL,
        "expand_pan_buffer": True,
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
        "tab3_timelapse_data": None,
        # Tab 4 Cartography / Print Layout state
        "tab4_carto_result": None,
        "tab4_carto_meta": None,
        # Tab 5 Natural Disaster Assessment state
        "tab5_disaster_result": None,
        "tab5_last_preset": "custom",
        "tab5_sent_to_carto": False,
        # Tab 6 AI EO Assistant state
        "tab6_ai_result": None,
        "tab6_sent_to_carto": False,
        "tab6_gemini_key": "",
        "tab6_query_input": "",
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
                "legend_units": "",
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

    # 7. Tab 6: AI EO Assistant Raster Output (Disaster / Spectral Index / RGB Scene)
    ai_res = st.session_state.get("tab6_ai_result")
    if isinstance(ai_res, dict) and ai_res.get("kind") in ("disaster", "spectral_index", "scene_rgb"):
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

    raw_df = st.session_state.get("stac_raw_summary_df")
    raw_by_date = st.session_state.get("stac_raw_items_by_date")
    if (
        st.session_state.get("search_performed", False)
        and isinstance(raw_df, pd.DataFrame)
        and not raw_df.empty
        and isinstance(raw_by_date, dict)
        and raw_by_date
    ):
        grouped_f, df_f = filter_by_temporal_interval(raw_df, raw_by_date, interval_mode=new_int)
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


def _format_interval_summary_caption(summary_df: pd.DataFrame, lang: str) -> str:
    """Sestavi jasen povzetek števila najdenih vs. izbranih reprezentativnih datumov in izločenih delnih preletov."""
    sel_n = int(len(summary_df))
    total_n = int(summary_df.attrs.get("total_dates_found", sel_n))
    partial_n = int(summary_df.attrs.get("filtered_partial_dates", 0))
    int_mode = str(summary_df.attrs.get("interval_mode", st.session_state.get("temporal_interval", INTERVAL_ALL)))
    int_label = t(INTERVAL_I18N.get(int_mode, "interval_all"), lang=lang)
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
    limit: int | None = None,
) -> None:
    """Izvede celotno zgodovinsko STAC iskanje s filtrom pokritosti AOI in časovnim vzorčenjem."""
    min_cov = DEFAULT_MIN_AOI_COVERAGE_PCT if exclude_partial_swaths else 0.0
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


def _d96tm_corners_to_wgs84_polygon(bbox_3794: list[float]) -> list[list[float]]:
    min_e, min_n, max_e, max_n = bbox_3794
    corners_3794 = [
        (min_e, min_n),
        (max_e, min_n),
        (max_e, max_n),
        (min_e, max_n),
        (min_e, min_n),
    ]
    poly_latlon: list[list[float]] = []
    for e_val, n_val in corners_3794:
        lon, lat = transform_coords(e_val, n_val, EPSG_D96TM, EPSG_WGS84)
        poly_latlon.append([lat, lon])
    return poly_latlon


def _polygon_bounds_to_bboxes(
    poly_geom_4326: dict[str, Any],
) -> tuple[list[float], list[float], float, float]:
    """Izračuna WGS84/D96TM okvir in središče za poljuben GeoJSON poligon."""
    geom = shape(poly_geom_4326)
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    bbox_4326 = [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]
    e1, n1 = transform_coords(min_lon, min_lat, EPSG_WGS84, EPSG_D96TM)
    e2, n2 = transform_coords(max_lon, max_lat, EPSG_WGS84, EPSG_D96TM)
    bbox_3794 = [min(e1, e2), min(n1, n2), max(e1, e2), max(n1, n2)]
    centroid = geom.centroid
    return bbox_4326, bbox_3794, float(centroid.y), float(centroid.x)


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
        poly_coords = _d96tm_corners_to_wgs84_polygon(bbox_3794)
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
    return t("axis_e_utm"), t("axis_n_utm")


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
    """Izriše enotno vrstico (Senzor | Podlaga | Kraj / D96/TM / WGS84) in 600 px zemljevid AOI."""
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
                f"D96/TM ({float(e_jump):,.1f} m, {float(n_jump):,.1f} m)",
                zoom=12,
            )
        except Exception as exc:
            st.error(t("place_search_error", err=exc))

    if wgs_jump_clicked:
        _jump_to_location(
            float(lat_jump),
            float(lon_jump),
            f"WGS84 ({float(lat_jump):.5f}°, {float(lon_jump):.5f}°)",
            zoom=12,
        )

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
            height=600,
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
            prev_poly = st.session_state.get("drawn_polygon_4326")
            if json.dumps(extracted_poly, sort_keys=True) != json.dumps(prev_poly, sort_keys=True):
                if isinstance(draw_out, dict) and draw_out.get("zoom") is not None:
                    try:
                        st.session_state["map_zoom"] = int(draw_out["zoom"])
                    except (TypeError, ValueError):
                        pass
                st.session_state["drawn_polygon_4326"] = extracted_poly
                st.rerun()

    # Način 2: Velik 600 px interaktivni zemljevid (klik posodobi središče AOI)
    else:
        is_click_mode = aoi_mode_key == "map"
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
            height=600,
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
def main() -> None:
    _init_session_state()
    pending_mode = st.session_state.pop("pending_aoi_mode", None)
    if pending_mode in AOI_MODE_KEYS:
        st.session_state["sidebar_aoi_mode"] = pending_mode
    if st.session_state.pop("pending_reset_scene_widgets", False):
        reset_prefixes = (
            "tab1_scene_select_",
            "tab2_scene_select_",
            "tab2_comp_n_",
            "tab3_max_epochs_",
            "tab3_t1_select_",
            "tab3_t2_select_",
            "tab3_tl_frames_",
        )
        for k in list(st.session_state.keys()):
            if k == "tab2_sar_scene_select" or k.startswith(reset_prefixes):
                st.session_state.pop(k, None)
    _clamp_dates_for_sensor(str(st.session_state["selected_sensor"]))

    # =======================================================================
    # STRANSKA VRSTICA (SIDEBAR): JEZIK, SENZOR, AOI, GEODETSKE NASTAVITVE
    # =======================================================================
    with st.sidebar:
        lang = st.radio(
            t("lang_label"),
            options=["sl", "en"],
            index=0 if st.session_state["lang"] == "sl" else 1,
            horizontal=True,
            format_func=lambda code: "SL — Slovenščina" if code == "sl" else "EN — English",
            key="lang_selector",
        )
        st.session_state["lang"] = lang

        # --- 1. Satelitski senzor (sinhroniziran z glavno vrstico in Zavihkom 3) ---
        st.divider()
        st.subheader(t("sec_sensor"))

        _touch_selector(
            t("sensor_select_label"),
            options=SENSOR_KEYS,
            default=str(st.session_state.get("selected_sensor", SENSOR_S2)),
            format_func=lambda s, _lang=lang: t(SENSOR_META[s]["pill_key"], lang=_lang),
            key="sidebar_sensor_select",
            on_change=_sync_sensor_state,
            args=("sidebar_sensor_select",),
            use_pills=True,
        )
        sensor = str(st.session_state["selected_sensor"])
        sensor_cfg = SENSOR_META[sensor]

        exclude_l7_slc_off = False
        if sensor == SENSOR_LANDSAT:
            exclude_l7_slc_off = st.checkbox(
                t("l7_exclude_label"),
                value=bool(st.session_state.get("exclude_l7_slc_off", False)),
                help=t("l7_exclude_help"),
                key="sidebar_l7_slc_exclude",
            )
            st.session_state["exclude_l7_slc_off"] = exclude_l7_slc_off
        elif sensor == SENSOR_S1:
            st.caption(f"📡 {t('sar_allweather_note')}")

        # --- 2. Območje zanimanja (AOI) ---
        st.divider()
        st.subheader(t("sec_aoi"))

        aoi_mode_key = st.radio(
            t("aoi_mode_label"),
            options=AOI_MODE_KEYS,
            format_func=lambda k, _lang=lang: t(AOI_MODE_I18N[k], lang=_lang),
            label_visibility="collapsed",
            key="sidebar_aoi_mode",
        )

        bbox_4326: list[float] | None = None
        bbox_3794: list[float] | None = None
        aoi_polygon_4326: dict[str, Any] | None = None
        center_lat: float = float(st.session_state["map_lat"])
        center_lon: float = float(st.session_state["map_lon"])
        aoi_error: str | None = None

        # --- Način 1: Interaktivni zemljevid (klik na velikem zemljevidu + radij) ---
        if aoi_mode_key == "map":
            radius_m = st.slider(
                t("radius_label"),
                min_value=500,
                max_value=15000,
                value=int(st.session_state["map_radius_m"]),
                step=250,
                key="sidebar_map_radius_slider",
                help=t("radius_help"),
            )
            st.session_state["map_radius_m"] = radius_m

            center_lat = float(st.session_state["map_lat"])
            center_lon = float(st.session_state["map_lon"])
            bbox_4326, bbox_3794 = bbox_from_point_wgs84(center_lat, center_lon, float(radius_m))

            e_c, n_c = transform_coords(center_lon, center_lat, EPSG_WGS84, EPSG_D96TM)
            st.markdown(
                f"<div class='coord-box'>"
                f"<strong>D96/TM:</strong> e = {e_c:,.1f} m · n = {n_c:,.1f} m<br>"
                f"<strong>WGS84:</strong> {center_lat:.5f}°, {center_lon:.5f}°"
                f"</div>",
                unsafe_allow_html=True,
            )

        # --- Način 2: Interaktivno risanje poligona / pravokotnika (Draw) ---
        elif aoi_mode_key == "draw":
            st.caption(t("draw_sidebar_hint"))
            drawn_poly = st.session_state.get("drawn_polygon_4326")
            if drawn_poly is not None:
                aoi_polygon_4326 = drawn_poly
                bbox_4326, bbox_3794, center_lat, center_lon = _polygon_bounds_to_bboxes(drawn_poly)
                st.success(t("draw_active_status"))
                if st.button(t("btn_clear_polygon"), width="stretch", key="btn_clear_drawn_poly"):
                    st.session_state["drawn_polygon_4326"] = None
                    st.session_state["draw_map_version"] = int(st.session_state["draw_map_version"]) + 1
                    st.rerun()
            else:
                st.info(t("draw_waiting_status"))

        # --- Način 3: Ročni vnos koordinat (D96/TM ali WGS84) ---
        elif aoi_mode_key == "manual":
            coord_sys = st.radio(
                t("input_crs_label"),
                options=["D96/TM (EPSG:3794)", "WGS84 (EPSG:4326)"],
                horizontal=True,
            )
            radius_m = st.slider(
                t("radius_label"),
                min_value=500,
                max_value=15000,
                value=2000,
                step=250,
                key="manual_radius_slider",
                help=t("radius_help"),
            )

            if coord_sys == "D96/TM (EPSG:3794)":
                col_e, col_n = st.columns(2)
                with col_e:
                    e_in = st.number_input(t("e_input_label"), value=461500.0, step=100.0, format="%.2f")
                with col_n:
                    n_in = st.number_input(t("n_input_label"), value=101500.0, step=100.0, format="%.2f")
                center_lon, center_lat = transform_coords(e_in, n_in, EPSG_D96TM, EPSG_WGS84)
                e_disp, n_disp = float(e_in), float(n_in)
                bbox_4326, bbox_3794 = bbox_from_point_d96tm(e_disp, n_disp, float(radius_m))
            else:
                col_lat, col_lon = st.columns(2)
                with col_lat:
                    lat_in = st.number_input(
                        t("lat_input_label"),
                        min_value=-90.0,
                        max_value=90.0,
                        value=46.056900,
                        step=0.005,
                        format="%.6f",
                    )
                with col_lon:
                    lon_in = st.number_input(
                        t("lon_input_label"),
                        min_value=-180.0,
                        max_value=180.0,
                        value=14.505800,
                        step=0.005,
                        format="%.6f",
                    )
                center_lat, center_lon = float(lat_in), float(lon_in)
                e_disp, n_disp = transform_coords(center_lon, center_lat, EPSG_WGS84, EPSG_D96TM)
                bbox_4326, bbox_3794 = bbox_from_point_wgs84(center_lat, center_lon, float(radius_m))

            st.markdown(
                f"<div class='coord-box'>"
                f"<strong>{t('live_transform_title')}</strong><br>"
                f"D96/TM: e = {e_disp:,.2f} m · n = {n_disp:,.2f} m<br>"
                f"WGS84 : lat = {center_lat:.6f}° · lon = {center_lon:.6f}°"
                f"</div>",
                unsafe_allow_html=True,
            )

        # --- Način 4: Prednastavljene lokacije v Sloveniji ---
        elif aoi_mode_key == "preset":
            preset_name = st.selectbox(
                t("preset_select_label"),
                options=list(SLOVENIA_PRESETS.keys()),
                index=0,
            )
            preset = SLOVENIA_PRESETS[preset_name]
            st.caption(t(preset["desc_key"]))

            radius_m = st.slider(
                t("radius_label"),
                min_value=500,
                max_value=15000,
                value=int(preset["radius_m"]),
                step=250,
                key=f"preset_radius_{preset_name}",
                help=t("radius_help"),
            )
            center_lat = float(preset["lat"])
            center_lon = float(preset["lon"])
            e_p, n_p = transform_coords(center_lon, center_lat, EPSG_WGS84, EPSG_D96TM)
            bbox_4326, bbox_3794 = bbox_from_point_wgs84(center_lat, center_lon, float(radius_m))
            st.markdown(
                f"<div class='coord-box'>"
                f"<strong>D96/TM:</strong> e = {e_p:,.1f} m · n = {n_p:,.1f} m<br>"
                f"<strong>WGS84:</strong> {center_lat:.5f}°, {center_lon:.5f}°"
                f"</div>",
                unsafe_allow_html=True,
            )

        # --- Način 5: Nalaganje GeoJSON poligona ---
        else:
            uploaded_geojson = st.file_uploader(
                t("geojson_upload_label"),
                type=["geojson", "json"],
                help=t("geojson_upload_help"),
            )
            if uploaded_geojson is not None:
                try:
                    bbox_4326, bbox_3794, aoi_polygon_4326 = _parse_geojson_aoi(uploaded_geojson)
                    center_lon = 0.5 * (bbox_4326[0] + bbox_4326[2])
                    center_lat = 0.5 * (bbox_4326[1] + bbox_4326[3])
                    st.success(t("geojson_loaded", name=uploaded_geojson.name))
                except Exception as exc:
                    aoi_error = t("geojson_error", err=exc)
                    st.error(aoi_error)
            else:
                st.info(t("geojson_prompt"))

        # --- 3. Geodetske in RAM nastavitve (kompaktno za tablice) ---
        st.divider()
        st.subheader(t("sec_geodetic"))

        area_km2: float = 0.0
        aoi_valid: bool = False
        rec_res: int = sensor_cfg["default_res"]

        if bbox_4326 is not None and aoi_error is None:
            if aoi_polygon_4326 is not None:
                area_km2, aoi_valid, base_rec_res = validate_aoi_geometry(aoi_polygon_4326)
                _, bbox_ok, bbox_rec_res = validate_aoi_size(bbox_4326)
                aoi_valid = aoi_valid and bbox_ok
                base_rec_res = max(base_rec_res, bbox_rec_res)
            else:
                area_km2, aoi_valid, base_rec_res = validate_aoi_size(bbox_4326)

            if sensor == SENSOR_LANDSAT:
                rec_res = 60 if base_rec_res >= 60 else 30
            else:
                rec_res = base_rec_res

            area_ha = area_km2 * 100.0
            c_km, c_ha = st.columns(2)
            c_km.metric(t("metric_area_km2"), f"{area_km2:,.2f}")
            c_ha.metric(t("metric_area_ha"), f"{area_ha:,.1f}")

            res_key = f"{sensor}_{rec_res}"
            if st.session_state["last_rec_res_key"] != res_key:
                st.session_state["last_rec_res_key"] = res_key
                st.session_state["selected_resolution"] = rec_res
                st.session_state[f"sidebar_res_select_{sensor}"] = rec_res

        res_options: list[int] = list(sensor_cfg["res_options"])
        cur_sel_res = st.session_state["selected_resolution"]
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
        )
        st.session_state["selected_resolution"] = int(selected_resolution)

        target_crs = _touch_selector(
            t("output_crs_label"),
            options=CRS_KEYS,
            default=CRS_D96TM,
            format_func=lambda c, _lang=lang: t(CRS_PILL_I18N[c], lang=_lang),
            key="sidebar_crs_select",
            use_pills=True,
        )

        # En sam subtilen inline RAM statusni indikator namesto ponavljajočih se opozoril
        if bbox_4326 is not None and aoi_error is None:
            if not aoi_valid:
                st.markdown(
                    f'<div class="ram-pill ram-pill-err">{t("ram_status_err", area=area_km2)}</div>',
                    unsafe_allow_html=True,
                )
            elif selected_resolution < rec_res:
                st.markdown(
                    f'<div class="ram-pill ram-pill-warn">'
                    f'{t("ram_status_finer", sel=selected_resolution, rec=rec_res, area=area_km2)}</div>',
                    unsafe_allow_html=True,
                )
            elif area_km2 > 100.0:
                st.markdown(
                    f'<div class="ram-pill ram-pill-warn">'
                    f'{t("ram_status_warn", area=area_km2, rec=rec_res)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="ram-pill ram-pill-ok">'
                    f'{t("ram_status_ok", area=area_km2, res=selected_resolution)}</div>',
                    unsafe_allow_html=True,
                )

            with st.expander(t("sidebar_geo_details_expander"), expanded=False):
                st.caption(f"**CRS:** {t(CRS_I18N[target_crs])}")
                if not aoi_valid:
                    st.caption(t("aoi_too_large", area=area_km2))
                elif selected_resolution < rec_res:
                    st.caption(t("res_finer_warn", sel=selected_resolution, rec=rec_res, area=area_km2))
                elif area_km2 > 100.0:
                    st.caption(t("aoi_large_warn", area=area_km2, res=rec_res))
                if bbox_3794 is not None:
                    st.markdown(
                        f"<div class='coord-box'>"
                        f"<strong>EPSG:3794 [m]</strong><br>"
                        f"e: {bbox_3794[0]:,.1f} → {bbox_3794[2]:,.1f}<br>"
                        f"n: {bbox_3794[1]:,.1f} → {bbox_3794[3]:,.1f}<br>"
                        f"<strong>EPSG:4326 [°]</strong><br>"
                        f"lon: {bbox_4326[0]:.5f} → {bbox_4326[2]:.5f}<br>"
                        f"lat: {bbox_4326[1]:.5f} → {bbox_4326[3]:.5f}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # --- 4. Časovno obdobje, časovni korak in oblačnost (sinhronizirano z Zavihkom 3) ---
        st.divider()
        st.subheader(t("sec_time"))

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

        exclude_partial_swaths = st.checkbox(
            t("exclude_partial_label"),
            help=t("exclude_partial_help"),
            key="sidebar_exclude_partial",
            on_change=_sync_partial_filter_state,
            args=("sidebar_exclude_partial",),
        )

        can_search = (bbox_4326 is not None) and aoi_valid and dates_valid and (aoi_error is None)
        search_clicked = st.button(
            t("btn_search"),
            type="primary",
            width="stretch",
            disabled=not can_search,
            key="sidebar_search_btn",
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
                        exclude_l7_slc_off=exclude_l7_slc_off,
                        exclude_partial_swaths=bool(exclude_partial_swaths),
                        interval_mode=active_interval,
                        limit=None,
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(t("search_error", err=exc))

    # =======================================================================
    # GLAVNI DEL APLIKACIJE: KOMPAKTEN HORIZONTALNI PAS Z AKTIVNIMI ZNAČKAMI
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
        f'<span class="geo-badge geo-badge-slate">{t(CRS_SHORT_I18N[target_crs])}</span>'
        f'<span class="geo-badge geo-badge-slate">{selected_resolution} m</span>'
        f'<span class="geo-badge geo-badge-slate">{area_pill_val}</span>'
        f"{active_polygon_badge}"
        f"</div>"
        f"</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [t("tab1"), t("tab2"), t("tab3"), t("tab4"), t("tab5"), t("tab6")]
    )

    items: list[Any] = st.session_state["stac_items"]
    summary_df: pd.DataFrame = st.session_state["stac_summary_df"]
    search_performed: bool = st.session_state["search_performed"]
    active_sensor: str = st.session_state.get("search_sensor", sensor)
    active_sensor_cfg = SENSOR_META[active_sensor]
    active_bbox = st.session_state["search_bbox_4326"] or bbox_4326
    active_polygon = (
        st.session_state["search_polygon_4326"] if search_performed else aoi_polygon_4326
    )

    # Pripravi oznake posnetkov za vse 3 zavihke
    scene_labels: list[str] = []
    default_scene_idx: int = 0
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            tile_lbl = _format_tile_label(str(row["mgrs_tile"]), sensor=active_sensor)
            if active_sensor == SENSOR_S1 or pd.isna(row["cloud_cover"]):
                scene_labels.append(
                    t("scene_option_sar_fmt", date=row["date_str"], tile=tile_lbl)
                )
            else:
                scene_labels.append(
                    t(
                        "scene_option_fmt",
                        date=row["date_str"],
                        cloud=float(row["cloud_cover"]),
                        tile=tile_lbl,
                    )
                )
        if active_sensor != SENSOR_S1 and summary_df["cloud_cover"].notna().any():
            cand_idx = int(summary_df["cloud_cover"].reset_index(drop=True).idxmin())
            if 0 <= cand_idx < len(items):
                default_scene_idx = cand_idx

    # =======================================================================
    # ZAVIHEK 1: ISKANJE POSNETKOV, PREGLED (OPTIKA / SAR) IN GEOTIFF IZVOZ
    # =======================================================================
    with tab1:
        # Velik 600 px interaktivni zemljevid + iskalnik krajev (Nominatim)
        map_expanded = (not search_performed) or (
            aoi_mode_key == "draw" and st.session_state.get("drawn_polygon_4326") is None
        )
        if not search_performed:
            st.info(t("onboarding_info"))
            st.subheader(
                t("draw_main_header") if aoi_mode_key == "draw" else t("preview_aoi_header")
            )
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
                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True,
                    height=min(310, 42 + 35 * len(display_df)),
                )

                st.subheader(t("t1_vis_header"))

                is_sar_active = active_sensor == SENSOR_S1
                vis_specs = SAR_VIS_SPECS if is_sar_active else OPTICAL_COMPOSITE_SPECS

                with st.container(border=True):
                    col_sel, col_vis = st.columns([1.15, 1.0])
                    with col_sel:
                        selected_idx = st.selectbox(
                            t("scene_select_label"),
                            options=list(range(len(items))),
                            index=default_scene_idx,
                            format_func=lambda idx: scene_labels[idx],
                            key=f"tab1_scene_select_{active_sensor}",
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

                    col_btn, col_stretch, col_gamma = st.columns([1.2, 0.9, 1.0], vertical_alignment="center")
                    with col_btn:
                        load_clicked = st.button(
                            t("btn_load_scene"),
                            type="primary",
                            width="stretch",
                            key="tab1_load_btn",
                        )
                    with col_stretch:
                        use_percentile = st.checkbox(
                            t("stretch_label"),
                            value=True,
                            help=t("stretch_help"),
                            key="tab1_stretch",
                        )
                    with col_gamma:
                        gamma_disabled = is_sar_active and vis_code in ("VV", "VH")
                        gamma_val = st.slider(
                            t("gamma_label"),
                            min_value=0.5,
                            max_value=2.2,
                            value=1.0,
                            step=0.05,
                            disabled=gamma_disabled,
                            key="tab1_gamma",
                        )

                    expand_pan_buffer = st.checkbox(
                        t("expand_pan_label"),
                        value=bool(st.session_state.get("expand_pan_buffer", True)),
                        help=t("expand_pan_help"),
                        key="tab1_expand_pan_chk",
                    )
                    st.session_state["expand_pan_buffer"] = bool(expand_pan_buffer)

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
                                    active_bbox, buffer_factor=0.35, max_area_km2=900.0
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

                            tif_bytes = export_to_geotiff_bytes(ds_loaded, target_epsg=target_crs)
                            epsg_num = target_crs.replace("EPSG:", "")
                            prefix = active_sensor_cfg["file_prefix"]
                            tif_filename = (
                                f"{prefix}_{chosen_row['date_str']}_{vis_code}_"
                                f"EPSG{epsg_num}_{selected_resolution}m.tif"
                            )

                            var_names = (
                                list(vis_spec["var_names"]) if is_sar_active else list(req_bands)
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
                            }
                            st.session_state["loaded_tif_bytes"] = tif_bytes
                            st.session_state["loaded_tif_filename"] = tif_filename
                            gc.collect()
                        except Exception as exc:
                            st.error(t("load_scene_error", err=exc))

                ds_show: xr.Dataset | None = st.session_state["loaded_scene_ds"]
                _raw_disp = st.session_state.get("loaded_scene_display_ds")
                ds_disp_show: xr.Dataset | None = _raw_disp if _raw_disp is not None else ds_show
                meta: dict[str, Any] | None = st.session_state["loaded_scene_meta"]

                if ds_show is not None and meta is not None and meta.get("sensor") == active_sensor:
                    st.divider()
                    first_band = meta["bands"][0]
                    h_px, w_px = ds_show[first_band].shape

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric(t("m_date"), meta["date_str"])
                    m2.metric(t("m_crs"), meta["crs"])
                    m3.metric(t("m_res"), f"{meta['resolution_m']} m")
                    m4.metric(t("m_dims"), f"{w_px} × {h_px} px")
                    m5.metric(t("m_area"), f"{meta['area_km2']:.2f} km²")

                    col_vm1, col_op1 = st.columns([1.65, 1.0], vertical_alignment="bottom")
                    with col_vm1:
                        t1_view_mode = _touch_selector(
                            t("viewer_mode_label"),
                            options=VIEWER_MODE_KEYS,
                            default="slippy",
                            format_func=lambda k, _lang=lang: t(VIEWER_MODE_I18N[k], lang=_lang),
                            key="tab1_viewer_mode",
                        )
                    with col_op1:
                        t1_opacity_pct = st.slider(
                            t("opacity_layer_label"),
                            min_value=0,
                            max_value=100,
                            value=100,
                            step=5,
                            disabled=(t1_view_mode != "slippy"),
                            key="tab1_overlay_opacity",
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
                        c_lat_t1 = (
                            0.5 * (active_bbox[1] + active_bbox[3])
                            if active_bbox is not None
                            else center_lat
                        )
                        c_lon_t1 = (
                            0.5 * (active_bbox[0] + active_bbox[2])
                            if active_bbox is not None
                            else center_lon
                        )
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
                            height=620,
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

                    cloud_disp = (
                        "— (SAR)"
                        if meta["cloud_cover"] is None
                        else f"{meta['cloud_cover']:.1f} %"
                    )
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

                    tif_bytes = st.session_state["loaded_tif_bytes"]
                    if tif_bytes:
                        st.download_button(
                            label=t("btn_download_rgb_tif"),
                            data=tif_bytes,
                            file_name=st.session_state["loaded_tif_filename"] or "scene_clip.tif",
                            mime="image/tiff",
                            width="stretch",
                            help=t("help_download_rgb_tif", bands=", ".join(meta["bands"]), crs=meta["crs"]),
                        )

    # =======================================================================
    # ZAVIHEK 2: SPEKTRALNI INDEKSI (OPTIKA) ALI SAR MASKA VODA/POPLAV (S1)
    # =======================================================================
    with tab2:
        if not search_performed or not items or summary_df.empty:
            st.subheader(t("t2_sar_header") if active_sensor == SENSOR_S1 else t("t2_header"))
            st.info(t("need_search_first"))

        # --- Način 2A: Sentinel-1 SAR Water / Flood Mask ---
        elif active_sensor == SENSOR_S1:
            st.subheader(t("t2_sar_header"))
            st.caption(t("t2_sar_caption"))

            col_s1_sc, col_s1_thr = st.columns([1.2, 1.0])
            with col_s1_sc:
                t2_sar_idx = st.selectbox(
                    t("scene_select_label"),
                    options=list(range(len(items))),
                    index=default_scene_idx,
                    format_func=lambda idx: scene_labels[idx],
                    key="tab2_sar_scene_select",
                )
            with col_s1_thr:
                sar_thr_db = st.slider(
                    t("sar_water_thr_label"),
                    min_value=-25.0,
                    max_value=-8.0,
                    value=-16.0,
                    step=0.5,
                    help=t("sar_water_thr_help"),
                    key="tab2_sar_water_thr",
                )

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
                                active_bbox, buffer_factor=0.35, max_area_km2=900.0
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

            ds_sar_loaded: xr.Dataset | None = st.session_state["tab2_sar_ds"]
            _raw_sar_disp = st.session_state.get("tab2_sar_display_ds")
            ds_sar_disp: xr.Dataset | None = (
                _raw_sar_disp if _raw_sar_disp is not None else ds_sar_loaded
            )
            sar_meta: dict[str, Any] | None = st.session_state["tab2_sar_meta"]

            if ds_sar_loaded is not None and sar_meta is not None:
                st.divider()
                mask_da, water_stats = compute_sar_water_mask(
                    ds_sar_loaded,
                    threshold_db=float(sar_thr_db),
                    polarization="VV",
                    resolution_m=float(sar_meta["resolution_m"]),
                )
                vv_stats = compute_index_stats(
                    ds_sar_loaded["VV"], resolution_m=float(sar_meta["resolution_m"])
                )

                if water_stats["valid_pixels"] == 0:
                    st.warning(t("t2_no_valid_pixels"))
                else:
                    w1, w2, w3, w4 = st.columns(4)
                    w1.metric(t("m_water_ha"), f"{water_stats['water_area_ha']:,.2f} ha")
                    w2.metric(t("m_water_pct"), f"{water_stats['water_pct']:.2f} %")
                    w3.metric(t("m_sar_vv_mean"), f"{vv_stats['mean']:.2f} dB")
                    w4.metric(t("m_valid_ha"), f"{water_stats['valid_area_ha']:,.2f} ha")

                    col_vm_s1, col_op_s1 = st.columns([1.65, 1.0], vertical_alignment="bottom")
                    with col_vm_s1:
                        t2_sar_view_mode = _touch_selector(
                            t("viewer_mode_label"),
                            options=VIEWER_MODE_KEYS,
                            default="slippy",
                            format_func=lambda k, _lang=lang: t(VIEWER_MODE_I18N[k], lang=_lang),
                            key="tab2_sar_viewer_mode",
                        )
                    with col_op_s1:
                        t2_sar_opacity_pct = st.slider(
                            t("opacity_index_label"),
                            min_value=0,
                            max_value=100,
                            value=85,
                            step=5,
                            disabled=(t2_sar_view_mode != "slippy"),
                            key="tab2_sar_overlay_opacity",
                        )

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
                                label=t(
                                    "sar_water_map_title",
                                    thr=sar_thr_db,
                                    date=sar_meta["date_str"],
                                ),
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
                        c_lat_s1 = (
                            0.5 * (active_bbox[1] + active_bbox[3])
                            if active_bbox is not None
                            else center_lat
                        )
                        c_lon_s1 = (
                            0.5 * (active_bbox[0] + active_bbox[2])
                            if active_bbox is not None
                            else center_lon
                        )
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
                            height=620,
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

        # --- Način 2B: Optični spektralni indeksi, Brezoblačni mozaik & Orodja ---
        else:
            t2_sub_idx, t2_sub_comp, t2_sub_tools = st.tabs(
                [t("t2_subtab_index"), t("t2_subtab_composite"), t("t2_subtab_tools")]
            )

            # ---------------------------------------------------------------
            # PODZAVIHEK 2.1: SPEKTRALNI INDEKSI (POSAMEZEN DATUM)
            # ---------------------------------------------------------------
            with t2_sub_idx:
                with st.container(border=True):
                    col_s2, col_i2 = st.columns([1.15, 1.0])
                    with col_s2:
                        t2_scene_idx = st.selectbox(
                            t("scene_select_label"),
                            options=list(range(len(items))),
                            index=default_scene_idx,
                            format_func=lambda idx: scene_labels[idx],
                            key=f"tab2_scene_select_{active_sensor}",
                        )
                    with col_i2:
                        t2_index_key = _touch_selector(
                            t("index_select_label"),
                            options=INDEX_KEYS,
                            default=INDEX_KEYS[0],
                            format_func=lambda k: k,
                            key="tab2_index_select",
                            use_pills=True,
                        )

                    st.caption(
                        f"{t(f'idx_{t2_index_key}_desc')} — `{SPECTRAL_INDICES[t2_index_key]['formula']}`"
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

                    with st.spinner(
                        t("spinner_index", bands=", ".join(req_bands), idx=t2_index_key, date=chosen_row["date_str"])
                    ):
                        try:
                            st.session_state["tab2_index_da"] = None
                            st.session_state["tab2_index_display_da"] = None
                            st.session_state["tab2_index_overlay_cache"] = None
                            st.session_state["tab2_tif_bytes"] = None
                            gc.collect()

                            use_pan_buf_t2 = bool(st.session_state.get("expand_pan_buffer", True))
                            if use_pan_buf_t2:
                                t2_load_bbox = expand_bbox_for_panning(
                                    active_bbox, buffer_factor=0.35, max_area_km2=900.0
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
                                ds_masked, _ = apply_cloud_mask(ds_raw, mask_clouds=t2_mask_clouds)
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
                                ds_masked, _ = apply_cloud_mask(ds_raw, mask_clouds=t2_mask_clouds)
                                index_da = compute_index(ds_masked, t2_index_key)
                                index_display_da = index_da

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
                            gc.collect()
                        except Exception as exc:
                            st.error(t("index_error", err=exc))

                idx_da: xr.DataArray | None = st.session_state["tab2_index_da"]
                _raw_idx_disp = st.session_state.get("tab2_index_display_da")
                idx_disp_da: xr.DataArray | None = (
                    _raw_idx_disp if _raw_idx_disp is not None else idx_da
                )
                idx_stats: dict[str, Any] | None = st.session_state["tab2_index_stats"]
                idx_meta: dict[str, Any] | None = st.session_state["tab2_index_meta"]

                if (
                    idx_da is not None
                    and idx_stats is not None
                    and idx_meta is not None
                    and idx_meta.get("sensor") == active_sensor
                ):
                    if idx_stats["valid_pixels"] == 0:
                        st.warning(t("t2_no_valid_pixels"))
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
                            # Če je v Zavihku 1 že naložen satelitski posnetek istega senzorja, ga ponudimo kot spodnji sloj v LayerControl
                            _raw_t1_disp = st.session_state.get("loaded_scene_display_ds")
                            t1_disp = (
                                _raw_t1_disp
                                if _raw_t1_disp is not None
                                else st.session_state.get("loaded_scene_ds")
                            )
                            t1_meta = st.session_state.get("loaded_scene_meta")
                            if (
                                t1_disp is not None
                                and t1_meta is not None
                                and t1_meta.get("sensor") == active_sensor
                            ):
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
                                    "show": True,
                                    "zindex": 20,
                                }
                            )

                            c_lat_t2 = (
                                0.5 * (active_bbox[1] + active_bbox[3])
                                if active_bbox is not None
                                else center_lat
                            )
                            c_lon_t2 = (
                                0.5 * (active_bbox[0] + active_bbox[2])
                                if active_bbox is not None
                                else center_lon
                            )
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
                                height=620,
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

                        st.caption(
                            t(
                                "t2_stats_caption",
                                formula=SPECTRAL_INDICES[idx_meta["index_key"]]["formula"],
                                vmin=idx_stats["min"],
                                vmax=idx_stats["max"],
                                p10=idx_stats["p10"],
                                p90=idx_stats["p90"],
                                scl=t("yes") if idx_meta["mask_clouds"] else t("no"),
                            )
                        )

                        hist_fig = _build_index_histogram(
                            index_da=idx_da,
                            stats=idx_stats,
                            index_key=idx_meta["index_key"],
                            date_str=idx_meta["date_str"],
                        )
                        st.plotly_chart(hist_fig, width="stretch", key="tab2_index_hist")

                        idx_tif = st.session_state["tab2_tif_bytes"]
                        if idx_tif:
                            st.download_button(
                                label=t("btn_download_index_tif"),
                                data=idx_tif,
                                file_name=st.session_state["tab2_tif_filename"] or "index.tif",
                                mime="image/tiff",
                                width="stretch",
                                help=t("help_download_index_tif", idx=idx_meta["index_key"], crs=idx_meta["crs"]),
                            )

            # ---------------------------------------------------------------
            # PODZAVIHEK 2.2: BREZOBLAČNI MEDIANI MOZAIK OBDOBJA
            # ---------------------------------------------------------------
            with t2_sub_comp:
                st.markdown(f"#### {t('comp_header')}")
                st.caption(t("comp_caption"))

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
                                height=620,
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
                                height=620,
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

            # ---------------------------------------------------------------
            # PODZAVIHEK 2.3: SPEKTRALNI PODPIS TOČKE IN RABA TAL [HA]
            # ---------------------------------------------------------------
            with t2_sub_tools:
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

    # =======================================================================
    # ZAVIHEK 3: ČASOVNE VRSTE, ZAZNAVA SPREMEMB IN TIMELAPSE
    # =======================================================================
    with tab3:
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
            t3_interval_cur = str(st.session_state.get("temporal_interval", INTERVAL_ALL))

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
                        exclude_l7_slc_off=bool(st.session_state.get("exclude_l7_slc_off", False)),
                        exclude_partial_swaths=bool(t3_exclude_partial),
                        interval_mode=t3_interval_cur,
                        limit=None,
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(t("search_error", err=exc))

        if not search_performed or not items or summary_df.empty:
            st.info(t("no_scenes_warn") if search_performed else t("need_search_first"))
        else:
            is_sar_t3 = active_sensor == SENSOR_S1
            min_swath_guard = 85.0 if bool(st.session_state.get("exclude_partial_swaths", True)) else 0.0
            metric_options = SAR_METRIC_KEYS if is_sar_t3 else INDEX_KEYS
            metric_label_title = (
                t("sar_metric_select_label") if is_sar_t3 else t("index_select_label")
            )

            def _fmt_metric_pill(code: str, _lang: str = lang) -> str:
                if is_sar_t3 and code == "VV_VH":
                    return t("sar_VV_VH_pill", lang=_lang)
                return code

            sub_ts, sub_change, sub_tl = st.tabs(
                [t("t3_subtab_ts"), t("t3_subtab_change"), t("t3_subtab_tl")]
            )

            # ---------------------------------------------------------------
            # PODZAVIHEK 3.1: ČASOVNA VRSTA (GRAF)
            # ---------------------------------------------------------------
            with sub_ts:
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
                            st.caption(t("sar_allweather_note"))
                        else:
                            ts_mask_clouds = st.checkbox(
                                t("scl_toggle_label"),
                                value=True,
                                help=t("scl_toggle_help"),
                                key="tab3_ts_scl",
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
                            min_valid_pct=10.0,
                            progress_callback=_on_progress,
                            sensor=active_sensor,
                            aoi_polygon_4326=active_polygon,
                            min_swath_coverage_pct=min_swath_guard,
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
                    skipped = ts_meta.get("skipped", [])
                    if skipped:
                        st.info(t("ts_skipped_info", n=len(skipped)))
                        with st.expander(t("ts_skipped_expander"), expanded=False):
                            st.dataframe(pd.DataFrame(skipped), width="stretch", hide_index=True)

                    if ts_df_show.empty:
                        st.warning(t("ts_empty_warn"))
                    else:
                        ts_unit = " dB" if is_sar_t3 else ""
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

            # ---------------------------------------------------------------
            # PODZAVIHEK 3.2: ZAZNAVA SPREMEMB (Δ = t2 - t1 & PREJ / POTEM)
            # ---------------------------------------------------------------
            with sub_change:
                st.caption(t("t3_sec_b_caption"))

                if len(items) < 2:
                    st.info(t("t3_need_two_scenes"))
                else:
                    with st.container(border=True):
                        col_c_idx, col_c_thr, col_c_scl = st.columns(
                            [1.35, 0.95, 0.8], vertical_alignment="bottom"
                        )
                        with col_c_idx:
                            ch_metric_key = _touch_selector(
                                metric_label_title,
                                options=metric_options,
                                default=metric_options[0],
                                format_func=_fmt_metric_pill,
                                key=f"tab3_ch_metric_{active_sensor}",
                                use_pills=True,
                            )
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
                            else:
                                ch_threshold = st.slider(
                                    t("t3_threshold_label"),
                                    min_value=0.05,
                                    max_value=0.40,
                                    value=0.10,
                                    step=0.01,
                                    key="tab3_ch_threshold_opt",
                                )
                        with col_c_scl:
                            if is_sar_t3:
                                ch_mask_clouds = False
                                st.caption(t("sar_allweather_note"))
                            else:
                                ch_mask_clouds = st.checkbox(
                                    t("scl_toggle_label"),
                                    value=True,
                                    help=t("scl_toggle_help"),
                                    key="tab3_ch_scl",
                                )

                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            t1_idx = st.selectbox(
                                t("t3_select_t1"),
                                options=list(range(len(items))),
                                index=len(items) - 1,
                                format_func=lambda idx: scene_labels[idx],
                                key=f"tab3_t1_select_{active_sensor}",
                            )
                        with col_t2:
                            t2_idx = st.selectbox(
                                t("t3_select_t2"),
                                options=list(range(len(items))),
                                index=0,
                                format_func=lambda idx: scene_labels[idx],
                                key=f"tab3_t2_select_{active_sensor}",
                            )

                        if t1_idx == t2_idx:
                            st.warning(t("t3_same_scene_warn"))

                        ch_clicked = st.button(
                            t("btn_compute_change"),
                            type="primary",
                            width="stretch",
                            disabled=(t1_idx == t2_idx),
                            key="tab3_ch_btn",
                        )

                    if ch_clicked and active_bbox is not None and t1_idx != t2_idx:
                        row_t1 = summary_df.iloc[t1_idx]
                        row_t2 = summary_df.iloc[t2_idx]
                        with st.spinner(
                            t("spinner_change", idx=ch_metric_key, d1=row_t1["date_str"], d2=row_t2["date_str"])
                        ):
                            try:
                                st.session_state["tab3_change_t1"] = None
                                st.session_state["tab3_change_t2"] = None
                                st.session_state["tab3_change_diff"] = None
                                st.session_state["tab3_change_tif_bytes"] = None
                                gc.collect()

                                da_t1, da_t2, da_diff = compute_change_detection(
                                    item_t1=items[t1_idx],
                                    item_t2=items[t2_idx],
                                    bbox_4326=active_bbox,
                                    index_name=ch_metric_key,
                                    target_epsg=target_crs,
                                    resolution=float(selected_resolution),
                                    mask_clouds=ch_mask_clouds,
                                    sensor=active_sensor,
                                    aoi_polygon_4326=active_polygon,
                                )
                                diff_tif = export_to_geotiff_bytes(da_diff, target_epsg=target_crs)
                                swipe_t1_png, swipe_t2_png = render_change_swipe_pair(
                                    da_t1=da_t1,
                                    da_t2=da_t2,
                                    index_key=ch_metric_key,
                                    is_sar=is_sar_t3,
                                )
                                epsg_num = target_crs.replace("EPSG:", "")
                                prefix = active_sensor_cfg["file_prefix"]
                                diff_fname = (
                                    f"{prefix}_delta_{ch_metric_key}_{row_t1['date_str']}_to_{row_t2['date_str']}_"
                                    f"EPSG{epsg_num}_{selected_resolution}m.tif"
                                )

                                st.session_state["tab3_change_t1"] = da_t1
                                st.session_state["tab3_change_t2"] = da_t2
                                st.session_state["tab3_change_diff"] = da_diff
                                st.session_state["tab3_change_meta"] = {
                                    "sensor": active_sensor,
                                    "index_key": ch_metric_key,
                                    "date_t1": str(row_t1["date_str"]),
                                    "date_t2": str(row_t2["date_str"]),
                                    "crs": target_crs,
                                    "resolution_m": int(selected_resolution),
                                }
                                st.session_state["tab3_change_tif_bytes"] = diff_tif
                                st.session_state["tab3_change_tif_filename"] = diff_fname
                                st.session_state["tab3_swipe_png_t1"] = swipe_t1_png
                                st.session_state["tab3_swipe_png_t2"] = swipe_t2_png
                                gc.collect()
                            except Exception as exc:
                                st.error(t("change_error", err=exc))

                    da_t1_s: xr.DataArray | None = st.session_state["tab3_change_t1"]
                    da_t2_s: xr.DataArray | None = st.session_state["tab3_change_t2"]
                    da_diff_s: xr.DataArray | None = st.session_state["tab3_change_diff"]
                    ch_meta: dict[str, Any] | None = st.session_state["tab3_change_meta"]

                    if (
                        da_t1_s is not None
                        and da_t2_s is not None
                        and da_diff_s is not None
                        and ch_meta is not None
                        and ch_meta.get("sensor") == active_sensor
                    ):
                        summary = compute_change_summary(
                            da_diff_s,
                            resolution_m=float(ch_meta["resolution_m"]),
                            threshold=float(ch_threshold),
                        )

                        if summary["valid_pixels"] == 0:
                            st.warning(t("t2_no_valid_pixels"))
                        else:
                            mean_str = (
                                f"{summary['mean']:+.2f} dB"
                                if is_sar_t3
                                else f"{summary['mean']:+.3f}"
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

                            unit_b = " [dB]" if is_sar_t3 else ""
                            idx_lbl = f"{ch_meta['index_key']}{unit_b}"
                            layer_labels = {
                                "diff": t("change_panel_diff", idx=idx_lbl),
                                "t1": t("change_panel_t1", idx=idx_lbl, d1=ch_meta["date_t1"]),
                                "t2": t("change_panel_t2", idx=idx_lbl, d2=ch_meta["date_t2"]),
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
                                index_key=ch_meta["index_key"],
                                target_crs=ch_meta["crs"],
                                date_t1=ch_meta["date_t1"],
                                date_t2=ch_meta["date_t2"],
                                resolution_m=float(ch_meta["resolution_m"]),
                                is_sar=is_sar_t3,
                                view_layer=view_layer,
                            )
                            st.plotly_chart(
                                fig_ch,
                                width="stretch",
                                config=PLOTLY_VIEWER_CONFIG,
                                key="tab3_change_plotly_viewer",
                            )

                            # Interaktivna zavesa Prej / Potem (t₁ ↔ t₂)
                            swipe_t1 = st.session_state.get("tab3_swipe_png_t1")
                            swipe_t2 = st.session_state.get("tab3_swipe_png_t2")
                            if swipe_t1 and swipe_t2:
                                st.markdown(f"#### {t('swipe_header')}")
                                st.caption(t("swipe_hint"))
                                _render_before_after_swipe_html(
                                    png_t1_bytes=swipe_t1,
                                    png_t2_bytes=swipe_t2,
                                    label_t1=f"t₁: {ch_meta['date_t1']} ({idx_lbl})",
                                    label_t2=f"t₂: {ch_meta['date_t2']} ({idx_lbl})",
                                    height=520,
                                )

                            diff_bytes = st.session_state["tab3_change_tif_bytes"]
                            if diff_bytes:
                                st.download_button(
                                    label=t("btn_download_change_tif"),
                                    data=diff_bytes,
                                    file_name=st.session_state["tab3_change_tif_filename"] or "change_map.tif",
                                    mime="image/tiff",
                                    width="stretch",
                                )

            # ---------------------------------------------------------------
            # PODZAVIHEK 3.3: VIZUALNI TIMELAPSE (GIF + DRSNIK DATUMOV)
            # ---------------------------------------------------------------
            with sub_tl:
                st.caption(t("t3_sec_c_caption"))

                if len(items) < 2:
                    st.info(t("tl_need_two_scenes"))
                else:
                    if is_sar_t3:
                        tl_layer_options = ["VV", "VH", "VV_VH"]
                    else:
                        tl_layer_options = ["RGB", "NIR", "NDVI", "NDWI", "MNDWI", "NDBI", "NBR"]

                    def _fmt_tl_layer_pill(code: str, _lang: str = lang) -> str:
                        if code == "VV_VH":
                            return t("sar_VV_VH_pill", lang=_lang)
                        return code

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
                                min_valid_pct=10.0,
                                duration_ms=int(round(tl_speed_sec * 1000)),
                                max_width=700,
                                progress_callback=_on_tl_progress,
                                min_swath_coverage_pct=min_swath_guard,
                            )
                            tl_prog.empty()
                            tl_status.empty()

                            tl_result["duration_ms"] = int(round(tl_speed_sec * 1000))
                            tl_result["prefix"] = active_sensor_cfg["file_prefix"]
                            tl_result["resolution_m"] = int(selected_resolution)
                            st.session_state["tab3_timelapse_data"] = tl_result
                            gc.collect()
                        except Exception as exc:
                            tl_prog.empty()
                            tl_status.empty()
                            st.error(t("tl_error", err=exc))

                    tl_data: dict[str, Any] | None = st.session_state["tab3_timelapse_data"]
                    if tl_data is not None and tl_data.get("sensor") == active_sensor:
                        tl_frames: list[dict[str, Any]] = tl_data.get("frames", [])
                        tl_skipped = tl_data.get("skipped", [])

                        if tl_skipped:
                            st.info(t("ts_skipped_info", n=len(tl_skipped)))

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

    # =======================================================================
    # ZAVIHEK 4: GEODETSKA KARTOGRAFIJA / IZDELAVA KARTE ZA TISK (D96/TM)
    # =======================================================================
    with tab4:
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
                            ql_tif = export_to_geotiff_bytes(
                                ds_ql[ql_vars], target_epsg=target_crs
                            )
                            epsg_num_ql = target_crs.split(":")[-1]
                            ql_fname = (
                                f"{active_sensor_cfg['prefix']}_{ql_vis}_"
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
                                    None
                                    if pd.isna(ql_row["cloud_cover"])
                                    else float(ql_row["cloud_cover"])
                                ),
                                "mgrs_tile": _format_tile_label(
                                    str(ql_row["mgrs_tile"]), sensor=active_sensor
                                ),
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
        else:
            col_cfg, col_prev = st.columns([0.35, 0.65], gap="large")

            with col_cfg:
                with st.container(border=True):
                    st.markdown(f"#### {t('carto_settings_header')}")

                    src_ids = [s["id"] for s in carto_sources]
                    src_map = {s["id"]: s for s in carto_sources}
                    pending_src = st.session_state.pop("tab4_pending_source_select", None)
                    if pending_src in src_ids:
                        st.session_state["tab4_source_select"] = pending_src
                        st.session_state["tab4_last_src_sig"] = None
                    selected_src_id = st.selectbox(
                        t("carto_layer_label"),
                        options=src_ids,
                        index=0,
                        format_func=lambda sid: src_map[sid]["label"],
                        key="tab4_source_select",
                    )
                    src = src_map[selected_src_id]

                    # Posodobi privzeti naslov/podnaslov ob zamenjavi sloja
                    if st.session_state.get("tab4_last_src_sig") != src["label"]:
                        st.session_state["tab4_title_input"] = src["default_title"]
                        st.session_state["tab4_subtitle_input"] = src["default_subtitle"]
                        st.session_state["tab4_last_src_sig"] = src["label"]

                    if "tab4_author_input" not in st.session_state:
                        st.session_state["tab4_author_input"] = t("carto_author_default")

                    carto_title = st.text_input(
                        t("carto_title_label"),
                        key="tab4_title_input",
                    )
                    carto_subtitle = st.text_input(
                        t("carto_subtitle_label"),
                        key="tab4_subtitle_input",
                    )
                    carto_author = st.text_input(
                        t("carto_author_label"),
                        key="tab4_author_input",
                    )

                    carto_fmt = _touch_selector(
                        t("carto_format_label"),
                        options=CARTO_FORMAT_KEYS,
                        default="a4_landscape",
                        format_func=lambda k, _lang=lang: t(CARTO_FORMAT_I18N[k], lang=_lang),
                        key="tab4_page_format",
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
                    )

                    carto_grid_style = _touch_selector(
                        t("carto_grid_style_label"),
                        options=CARTO_GRID_STYLE_KEYS,
                        default="light",
                        format_func=lambda k, _lang=lang: t(CARTO_GRID_STYLE_I18N[k], lang=_lang),
                        key="tab4_grid_style",
                    )

                    st.caption(t("carto_elements_label"))
                    chk_c1, chk_c2 = st.columns(2)
                    with chk_c1:
                        chk_grid = st.checkbox(t("carto_chk_grid"), value=True, key="tab4_chk_grid")
                        chk_north = st.checkbox(t("carto_chk_north"), value=True, key="tab4_chk_north")
                        chk_legend = st.checkbox(t("carto_chk_legend"), value=True, key="tab4_chk_legend")
                    with chk_c2:
                        chk_scale = st.checkbox(t("carto_chk_scalebar"), value=True, key="tab4_chk_scalebar")
                        chk_aoi = st.checkbox(t("carto_chk_aoi"), value=True, key="tab4_chk_aoi")

                    btn_gen_map = st.button(
                        t("carto_btn_generate"),
                        type="primary",
                        width="stretch",
                        key="tab4_btn_generate",
                    )

                if btn_gen_map:
                    with st.spinner(t("carto_spinner")):
                        try:
                            res_map = render_geodetic_map(
                                src["raster"],
                                crs=src["crs"],
                                is_rgb=bool(src["is_rgb"]),
                                bands=src["bands"],
                                use_percentile_stretch=bool(src["use_percentile_stretch"]),
                                gamma=float(src["gamma"]),
                                cmap_name=str(src["cmap_name"]),
                                vmin=float(src["vmin"]),
                                vmax=float(src["vmax"]),
                                discrete_items=src["discrete_items"],
                                legend_title=str(src["legend_title"]),
                                legend_units=str(src["legend_units"]),
                                stats_summary=src["stats_summary"],
                                title=carto_title,
                                subtitle=carto_subtitle,
                                author=carto_author,
                                sensor_label=str(src["sensor_label"]),
                                date_label=str(src["date_label"]),
                                resolution_m=float(src["resolution_m"]),
                                page_format=str(carto_fmt),
                                show_grid=bool(chk_grid),
                                grid_spacing_m=(None if int(carto_grid_val) == 0 else int(carto_grid_val)),
                                grid_style=str(carto_grid_style),
                                show_scalebar=bool(chk_scale),
                                show_north_arrow=bool(chk_north),
                                show_legend=bool(chk_legend),
                                show_aoi_outline=bool(chk_aoi),
                                aoi_polygon_geojson=(
                                    src["aoi_polygon_4326"]
                                    if "aoi_polygon_4326" in src
                                    else active_polygon
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
                                tif_out = export_to_geotiff_bytes(
                                    tif_src,
                                    target_epsg=src["crs"],
                                )
                            epsg_tag = str(src["crs"]).replace(":", "")
                            stem = f"{src['file_stem']}_{epsg_tag}_{carto_fmt}"
                            st.session_state["tab4_carto_result"] = res_map
                            st.session_state["tab4_carto_meta"] = {
                                "crs": str(src["crs"]),
                                "stem": stem,
                                "tif_bytes": tif_out,
                                "tif_filename": src["tif_filename"],
                            }
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

    # =======================================================================
    # ZAVIHEK 5: HITRA OCENA NARAVNIH NESREČ (PRED / MED / PO)
    # =======================================================================
    with tab5:
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
                # Za 6 km radij (144 km²) je 20 m varno in hitro za Streamlit Cloud RAM
                dis_res_m = max(int(selected_resolution), 20)
            else:
                dis_bbox_4326 = active_bbox
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

            # 2. Interactive Slippy Map Overlay (Phase 1 engine)
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
                height=580,
                use_container_width=True,
                returned_objects=[],
                key=(
                    f"tab5_dis_slippy_{d_type_show}_{dis_res_show['date_pre']}_"
                    f"{dis_res_show['date_event']}_{dis_opacity_pct}_"
                    f"{st.session_state.get('map_basemap', 'satellite')}"
                ),
            )
            st.caption(t("geodetic_separation_caption", crs=d_crs_show))

            # 3. Before / After Curtain Swipe Comparison
            if dis_res_show.get("swipe_png_pre") and dis_res_show.get("swipe_png_post"):
                st.markdown(f"#### {t('dis_swipe_header')}")
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

            # 5. Export GeoTIFF + 1-Click Send to Cartography (Phase 2)
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
                    st.rerun()

            if st.session_state.get("tab5_sent_to_carto"):
                st.success(t("dis_sent_carto_ok"))

    # =======================================================================
    # ZAVIHEK 6: AI POMOČNIK ZA DALJINSKO ZAZNAVANJE
    # =======================================================================
    with tab6:
        st.subheader(t("ai_header"))
        st.caption(t("ai_intro"))

        # Predhodni prenos izbranega hitrega primera v tekstovno polje pred ustvarjanjem gradnika
        pending_ai_q = st.session_state.pop("tab6_pending_query", None)
        if pending_ai_q is not None:
            st.session_state["tab6_query_input"] = str(pending_ai_q)
        auto_run_ai = bool(st.session_state.pop("tab6_auto_run", False))

        with st.container(border=True):
            # 1. Hitri primeri ukazov (4 gumbi za 1 klik)
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
                        # Odstrani uvodni emoji iz ukaza ob vstavitvi
                        clean_ex = ex_text.split(" ", 1)[1] if " " in ex_text else ex_text
                        st.session_state["tab6_pending_query"] = clean_ex
                        st.session_state["tab6_auto_run"] = True
                        st.rerun()

            # 2. Vnos ukaza + primarni gumb
            col_q, col_go = st.columns([3.4, 1.1], vertical_alignment="bottom")
            with col_q:
                ai_query_text = st.text_input(
                    t("ai_input_label"),
                    placeholder=t("ai_input_placeholder"),
                    key="tab6_query_input",
                )
            with col_go:
                btn_ai_run = st.button(
                    t("ai_btn_run"),
                    type="primary",
                    width="stretch",
                    key="tab6_btn_run",
                )

            # 3. Opcijske nastavitve Gemini API (z jasnim statusom)
            cur_api_key, key_src = get_active_gemini_api_key(st.session_state.get("tab6_gemini_key"))
            with st.expander(t("ai_api_expander"), expanded=False):
                if cur_api_key:
                    st.success(t("ai_status_gemini", source=key_src))
                else:
                    st.info(t("ai_status_local"))
                st.text_input(
                    t("ai_key_input_label"),
                    type="password",
                    key="tab6_gemini_key",
                )

        if btn_ai_run or auto_run_ai:
            q_clean = str(ai_query_text or "").strip()
            if not q_clean:
                st.warning(t("ai_empty_query"))
            else:
                ai_prog = st.progress(0.0)
                ai_status = st.empty()

                def _on_ai_prog(frac: float, msg: str) -> None:
                    ai_prog.progress(min(max(float(frac), 0.0), 1.0))
                    ai_status.caption(msg)

                try:
                    _on_ai_prog(0.04, "Razčlenjujem ukaz..." if lang == "sl" else "Parsing request...")
                    active_key_now, _ = get_active_gemini_api_key(st.session_state.get("tab6_gemini_key"))
                    parsed_plan = parse_eo_request(q_clean, api_key=active_key_now)
                    ai_out = execute_ai_plan(
                        parsed_plan,
                        target_epsg=target_crs,
                        geocode_fn=lambda name, _l=lang: geocode_place(name, lang=_l),
                        lang=lang,
                        progress_cb=_on_ai_prog,
                    )
                    st.session_state["tab6_ai_result"] = ai_out
                    st.session_state["tab6_sent_to_carto"] = False
                    # Sinhroniziraj geokodiran AOI v sejo (da ga vidijo tudi ostali zavihki)
                    st.session_state["map_lat"] = round(float(ai_out["lat"]), 6)
                    st.session_state["map_lon"] = round(float(ai_out["lon"]), 6)
                    st.session_state["radius_m"] = int(parsed_plan["radius_m"])
                    ai_prog.empty()
                    ai_status.empty()
                    gc.collect()
                except Exception as exc:
                    ai_prog.empty()
                    ai_status.empty()
                    st.error(t("ai_error", err=exc))

        ai_res_show: dict[str, Any] | None = st.session_state.get("tab6_ai_result")
        if isinstance(ai_res_show, dict) and ai_res_show.get("plan"):
            st.divider()
            pl = ai_res_show["plan"]
            ai_crs_show = str(ai_res_show.get("crs", target_crs))
            ai_kind_show = str(ai_res_show.get("kind", ""))
            eng_lbl = (
                t("ai_plan_engine_gemini")
                if pl.get("parser_engine") == "gemini"
                else t("ai_plan_engine_local")
            )
            expl_txt = pl["explanation_en"] if lang == "en" else pl["explanation_sl"]
            period_txt = (
                pl["event_date"]
                if pl["task_type"].startswith("disaster_")
                else f"{pl['start_date']} → {pl['end_date']}"
            )
            ai_sensor_short = SENSOR_META.get(pl["sensor"], SENSOR_META[SENSOR_S2])["short_name"]

            # Transparent Geodetic Plan Card (crisp teal/slate cartographic pills, no purple)
            st.markdown(f"#### {t('ai_plan_header')}")
            plan_pills_html = (
                f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;">'
                f'<span style="background:#F0FDFA;color:#0F766E;border:1px solid #99F6E4;border-radius:999px;padding:4px 12px;font-size:12px;font-weight:600;">📍 {t("ai_plan_place")}: {ai_res_show.get("resolved_place", pl["place_name"])} ({pl["radius_m"]} m)</span>'
                f'<span style="background:#F8FAFC;color:#0F172A;border:1px solid #CBD5E1;border-radius:999px;padding:4px 12px;font-size:12px;font-weight:600;">🛰️ {t("ai_plan_sensor")}: {ai_sensor_short}</span>'
                f'<span style="background:#F8FAFC;color:#0F172A;border:1px solid #CBD5E1;border-radius:999px;padding:4px 12px;font-size:12px;font-weight:600;">📅 {t("ai_plan_period")}: {period_txt}</span>'
                f'<span style="background:#F8FAFC;color:#0F172A;border:1px solid #CBD5E1;border-radius:999px;padding:4px 12px;font-size:12px;font-weight:600;">⚙️ {t("ai_plan_task")}: {pl["task_type"]} ({pl["index_name"]})</span>'
                f'<span style="background:#F8FAFC;color:#0F172A;border:1px solid #CBD5E1;border-radius:999px;padding:4px 12px;font-size:12px;font-weight:600;">📐 {t("ai_plan_crs")}: {ai_crs_show} · {pl["resolution_m"]} m</span>'
                f'<span style="background:#FFFFFF;color:#475569;border:1px solid #CBD5E1;border-radius:999px;padding:4px 12px;font-size:12px;">{eng_lbl}</span>'
                f'</div>'
            )
            st.markdown(plan_pills_html, unsafe_allow_html=True)
            st.info(expl_txt)

            # Display Output based on task kind
            if ai_kind_show == "disaster":
                d_obj = ai_res_show["disaster_result"]
                d_sum = d_obj["summary"]
                dt_k = str(d_obj["disaster_type"])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(t("dis_kpi_dates"), f"{d_obj['date_pre']} → {d_obj['date_event']}")
                m2.metric(t("dis_kpi_affected_ha"), f"{d_sum['affected_area_ha']:,.2f} ha", delta=f"{d_sum['affected_area_km2']:.3f} km²", delta_color="off")
                m3.metric(t("dis_kpi_affected_pct"), f"{d_sum['affected_pct']:.2f} %")
                m4.metric(t("ai_plan_crs"), ai_crs_show)

                ai_op_d = st.slider(t("opacity_layer_label"), 0, 100, 85, 5, key="tab6_ai_dis_opacity")
                lbl_k_ai = "en" if lang == "en" else "sl"
                sw_ai = [
                    (str(sp[lbl_k_ai]), str(sp["hex"]))
                    for sp in get_class_specs(dt_k)
                    if sp["affected"] or (dt_k == DISASTER_FLOOD and sp["code"] == 1)
                ]
                st.markdown(
                    build_horizontal_colorbar_html("Greys", 0.0, 1.0, label=t(DISASTER_TYPE_I18N[dt_k]), discrete_items=sw_ai),
                    unsafe_allow_html=True,
                )
                ai_uri, ai_bnds = raster_to_map_overlay_png(
                    d_obj["class_da"], src_epsg=ai_crs_show, is_rgb=False, discrete_rgba_map=build_slippy_rgba_map(dt_k)
                )
                slippy_ai = build_satellite_slippy_map(
                    center_lat=float(ai_res_show["lat"]),
                    center_lon=float(ai_res_show["lon"]),
                    zoom=None,
                    basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
                    overlay_layers=[{"name": f"🤖 {t(DISASTER_TYPE_I18N[dt_k])}", "image": ai_uri, "bounds": ai_bnds, "opacity": ai_op_d / 100.0, "show": True, "zindex": 18}],
                    aoi_bbox_4326=ai_res_show["bbox_4326"],
                    aoi_outline_label=t("slippy_aoi_outline"),
                    lang=lang,
                )
                st_folium(slippy_ai, height=560, use_container_width=True, returned_objects=[], key=f"tab6_dis_map_{ai_op_d}_{d_obj['date_event']}_{st.session_state.get('map_basemap', 'satellite')}")

            elif ai_kind_show == "spectral_index":
                st_ai = ai_res_show["stats"]
                idx_n = str(pl["index_name"])
                is_sar_ai = pl["sensor"] == SENSOR_S1
                spec_ai = SAR_METRICS.get(idx_n, SAR_METRICS["VV"]) if is_sar_ai else SPECTRAL_INDICES.get(idx_n, SPECTRAL_INDICES["NDVI"])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(t("col_date"), str(ai_res_show["date_str"]))
                m2.metric("Povprečje" if lang == "sl" else "Mean", f"{st_ai['mean']:.3f}")
                m3.metric(t("m_valid_ha"), f"{st_ai['valid_area_ha']:,.2f} ha")
                m4.metric(t("ai_plan_crs"), ai_crs_show)

                ai_op_i = st.slider(t("opacity_index_label"), 0, 100, 85, 5, key="tab6_ai_idx_opacity")
                st.markdown(
                    build_horizontal_colorbar_html(spec_ai["cmap"], float(spec_ai["vmin"]), float(spec_ai["vmax"]), label=idx_n, units=spec_ai.get("units", "")),
                    unsafe_allow_html=True,
                )
                ai_uri, ai_bnds = raster_to_map_overlay_png(
                    ai_res_show["index_da"], src_epsg=ai_crs_show, is_rgb=False, cmap_name=spec_ai["cmap"], vmin=float(spec_ai["vmin"]), vmax=float(spec_ai["vmax"])
                )
                slippy_ai = build_satellite_slippy_map(
                    center_lat=float(ai_res_show["lat"]),
                    center_lon=float(ai_res_show["lon"]),
                    zoom=None,
                    basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
                    overlay_layers=[{"name": f"🤖 {idx_n} ({ai_res_show['date_str']})", "image": ai_uri, "bounds": ai_bnds, "opacity": ai_op_i / 100.0, "show": True, "zindex": 18}],
                    aoi_bbox_4326=ai_res_show["bbox_4326"],
                    aoi_outline_label=t("slippy_aoi_outline"),
                    lang=lang,
                )
                st_folium(slippy_ai, height=560, use_container_width=True, returned_objects=[], key=f"tab6_idx_map_{ai_op_i}_{ai_res_show['date_str']}_{st.session_state.get('map_basemap', 'satellite')}")

            elif ai_kind_show == "scene_rgb":
                m1, m2, m3 = st.columns(3)
                m1.metric(t("col_date"), str(ai_res_show["date_str"]))
                m2.metric(t("col_tile"), str(ai_res_show.get("tile", "")))
                m3.metric(t("ai_plan_crs"), ai_crs_show)

                ai_op_r = st.slider(t("opacity_layer_label"), 0, 100, 100, 5, key="tab6_ai_rgb_opacity")
                ai_uri, ai_bnds = raster_to_map_overlay_png(
                    ai_res_show["scene_ds"], src_epsg=ai_crs_show, is_rgb=True, bands=ai_res_show.get("bands", ["B04", "B03", "B02"])
                )
                slippy_ai = build_satellite_slippy_map(
                    center_lat=float(ai_res_show["lat"]),
                    center_lon=float(ai_res_show["lon"]),
                    zoom=None,
                    basemap_mode=str(st.session_state.get("map_basemap", "satellite")),
                    overlay_layers=[{"name": f"🤖 RGB ({ai_res_show['date_str']})", "image": ai_uri, "bounds": ai_bnds, "opacity": ai_op_r / 100.0, "show": True, "zindex": 18}],
                    aoi_bbox_4326=ai_res_show["bbox_4326"],
                    aoi_outline_label=t("slippy_aoi_outline"),
                    lang=lang,
                )
                st_folium(slippy_ai, height=560, use_container_width=True, returned_objects=[], key=f"tab6_rgb_map_{ai_op_r}_{ai_res_show['date_str']}_{st.session_state.get('map_basemap', 'satellite')}")

            elif ai_kind_show == "time_series":
                ts_ai: pd.DataFrame = ai_res_show["ts_df"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Št. datumov" if lang == "sl" else "Valid Dates", f"{len(ts_ai)}")
                m2.metric("Povprečje" if lang == "sl" else "Mean", f"{ts_ai['mean'].mean():.3f}")
                m3.metric(t("ai_plan_period"), f"{ts_ai['date_str'].iloc[0]} → {ts_ai['date_str'].iloc[-1]}")
                m4.metric(t("ai_plan_crs"), ai_crs_show)

                fig_ai_ts = go.Figure()
                fig_ai_ts.add_trace(
                    go.Scatter(
                        x=ts_ai["date_str"],
                        y=ts_ai["mean"],
                        mode="lines+markers",
                        name=pl["index_name"],
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
                        idx=pl["index_name"],
                        place=pl["place_name"],
                        sensor=ai_sensor_short,
                        crs=ai_crs_show,
                    ),
                    xaxis_title=t("col_date"),
                    yaxis_title=pl["index_name"],
                    height=430,
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    font=dict(color="#0F172A"),
                )
                st.plotly_chart(fig_ai_ts, width="stretch", key="tab6_ai_ts_plotly")

            # Action & Export Bar
            if ai_kind_show == "time_series":
                st.download_button(
                    label=t("ai_btn_dl_csv"),
                    data=ai_res_show["csv_bytes"],
                    file_name=ai_res_show["csv_filename"],
                    mime="text/csv",
                    width="stretch",
                    key="tab6_dl_csv",
                )
            else:
                ac1, ac2 = st.columns(2)
                with ac1:
                    st.download_button(
                        label=t("ai_btn_dl_tif"),
                        data=ai_res_show["tif_bytes"],
                        file_name=ai_res_show["tif_filename"],
                        mime="image/tiff",
                        width="stretch",
                        key="tab6_dl_tif",
                    )
                with ac2:
                    if st.button(
                        t("ai_btn_send_carto"),
                        type="primary",
                        width="stretch",
                        key="tab6_btn_send_carto",
                    ):
                        st.session_state["tab4_pending_source_select"] = "tab6_ai"
                        st.session_state["tab6_sent_to_carto"] = True
                        st.rerun()

                if st.session_state.get("tab6_sent_to_carto"):
                    st.success(t("ai_sent_carto_ok"))

            # Copy-ready Technical Report Summary
            st.markdown(f"#### {t('ai_copy_header')}")
            st.code(ai_res_show.get("technical_summary", ""), language="text")

    # Prepreči odpiranje virtualne tipkovnice na preostalih dinamičnih selectboxih (tablice/telefoni)
    _inject_tablet_no_keyboard_js()


if __name__ == "__main__":
    main()
