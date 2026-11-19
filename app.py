"""Glavna spletna aplikacija za daljinsko zaznavanje (Multi-Sensor EO — Slovenija).

Geodetsko natančna obdelava posnetkov Sentinel-2 L2A (10 m, 2015–danes),
Landsat Collection 2 Level-2 (30 m, 1984–danes) in Sentinel-1 SAR RTC (10 m,
radarski posnetki skozi oblake) v D96/TM (EPSG:3794) in UTM 33N (EPSG:32633)
z neposrednim pretočnim nalaganjem COG iz Microsoft Planetary Computer STAC API.
Podpira interaktivno risanje poligonov (`rio.clip`) ter dvojezični vmesnik (SL | EN).
"""

from __future__ import annotations

import gc
import json
from datetime import date, timedelta
from typing import Any

import folium
from folium.plugins import Draw
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from streamlit_folium import st_folium

from analize import (
    SAR_METRICS,
    SPECTRAL_INDICES,
    apply_cloud_mask,
    compute_index,
    compute_index_stats,
    compute_sar_water_mask,
    extract_sar_metric,
    get_required_bands_for_sensor,
)
from casovne_vrste import (
    compute_change_detection,
    compute_change_summary,
    extract_time_series,
)
from geodezija_crs import (
    CRS_D96TM,
    CRS_UTM33N,
    EPSG_D96TM,
    EPSG_WGS84,
    bbox_from_point_d96tm,
    bbox_from_point_wgs84,
    export_to_geotiff_bytes,
    transform_coords,
    validate_aoi_geometry,
    validate_aoi_size,
)
from i18n import t
from stac_loader import (
    SENSOR_LANDSAT,
    SENSOR_S1,
    SENSOR_S2,
    load_scene,
    search_scenes,
)

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
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2.25rem;
        max-width: 1380px;
    }

    /* Čist tehnični naslovni pas */
    .geo-header {
        background-color: var(--geo-surface);
        border: 1px solid var(--geo-border);
        border-left: 4px solid var(--geo-teal);
        border-radius: 6px;
        padding: 0.95rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .geo-header-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--geo-slate);
        letter-spacing: -0.015em;
        margin: 0 0 0.25rem 0;
        line-height: 1.25;
    }
    .geo-header-sub {
        font-size: 0.9rem;
        color: var(--geo-muted);
        margin: 0 0 0.55rem 0;
        line-height: 1.4;
    }
    .geo-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .geo-badge {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--geo-teal-dark);
        background-color: var(--geo-teal-soft);
        border: 1px solid rgba(15, 118, 110, 0.22);
        border-radius: 4px;
        padding: 0.14rem 0.5rem;
    }

    /* Monospace kartica za geodetske koordinate */
    .coord-box {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.78rem;
        line-height: 1.5;
        color: var(--geo-slate);
        background-color: var(--geo-surface);
        border: 1px solid var(--geo-border);
        border-left: 3px solid var(--geo-teal);
        border-radius: 5px;
        padding: 0.5rem 0.65rem;
        margin: 0.35rem 0 0.5rem 0;
    }
    .coord-box strong {
        color: var(--geo-teal-dark);
    }

    /* Kartice metrik */
    div[data-testid="stMetric"] {
        background-color: var(--geo-surface);
        border: 1px solid var(--geo-border);
        border-radius: 6px;
        padding: 0.65rem 0.85rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    div[data-testid="stMetric"] label {
        color: var(--geo-muted);
        font-size: 0.8rem;
        font-weight: 600;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--geo-slate);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 1.25rem;
        font-weight: 700;
    }

    /* Stranska vrstica z jasno ločnico */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #CBD5E1;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--geo-slate);
        letter-spacing: -0.01em;
    }

    /* Gumbi prilagojeni tudi za tablice (touchscreen) */
    .stButton > button, .stDownloadButton > button {
        min-height: 2.65rem;
        font-weight: 600;
        border-radius: 6px;
        border: 1px solid var(--geo-border);
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--geo-teal);
        border-color: var(--geo-teal);
        color: #FFFFFF;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--geo-teal-dark);
        border-color: var(--geo-teal-dark);
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

    /* Zavihki */
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: var(--geo-muted);
        padding: 0.55rem 0.9rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--geo-teal-dark);
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

SENSOR_KEYS: list[str] = [SENSOR_S2, SENSOR_LANDSAT, SENSOR_S1]
SENSOR_META: dict[str, dict[str, Any]] = {
    SENSOR_S2: {
        "label_key": "sensor_s2_label",
        "short_name": "Sentinel-2 L2A",
        "file_prefix": "S2L2A",
        "min_date": date(2015, 6, 23),
        "res_options": [10, 20, 60],
        "default_res": 10,
    },
    SENSOR_LANDSAT: {
        "label_key": "sensor_landsat_label",
        "short_name": "Landsat C2 L2",
        "file_prefix": "LandsatC2L2",
        "min_date": date(1984, 1, 1),
        "res_options": [30, 60],
        "default_res": 30,
    },
    SENSOR_S1: {
        "label_key": "sensor_s1_label",
        "short_name": "Sentinel-1 SAR RTC",
        "file_prefix": "S1RTC",
        "min_date": date(2014, 10, 1),
        "res_options": [10, 20, 60],
        "default_res": 10,
    },
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

OPTICAL_COMPOSITE_SPECS: dict[str, dict[str, Any]] = {
    "RGB": {
        "bands": ["B04", "B03", "B02"],
        "label_key": "vis_rgb_label",
        "desc_key": "vis_rgb_desc",
    },
    "NIR": {
        "bands": ["B08", "B04", "B03"],
        "label_key": "vis_nir_label",
        "desc_key": "vis_nir_desc",
    },
    "SWIR": {
        "bands": ["B12", "B08", "B04"],
        "label_key": "vis_swir_label",
        "desc_key": "vis_swir_desc",
    },
}

SAR_VIS_SPECS: dict[str, dict[str, Any]] = {
    "VV": {
        "bands": ["vv"],
        "var_names": ["VV"],
        "label_key": "sar_vis_vv_label",
        "desc_key": "sar_vis_vv_desc",
    },
    "VH": {
        "bands": ["vh"],
        "var_names": ["VH"],
        "label_key": "sar_vis_vh_label",
        "desc_key": "sar_vis_vh_desc",
    },
    "SAR_FC": {
        "bands": ["vv", "vh"],
        "var_names": ["VV", "VH"],
        "label_key": "sar_vis_fc_label",
        "desc_key": "sar_vis_fc_desc",
    },
}

INDEX_KEYS: list[str] = ["NDVI", "NDWI", "MNDWI", "NDBI", "NBR"]
SAR_METRIC_KEYS: list[str] = ["VV", "VH", "VV_VH"]


# ---------------------------------------------------------------------------
# Pomožne funkcije: stanje seje, zemljevid, risanje poligonov, GeoJSON, izrisi
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Inicializira spremenljivke seje."""
    defaults: dict[str, Any] = {
        "lang": "sl",
        "active_sensor": SENSOR_S2,
        "map_lat": SLOVENIA_CENTER_LAT,
        "map_lon": SLOVENIA_CENTER_LON,
        "map_radius_m": 2000,
        "drawn_polygon_4326": None,
        "draw_map_version": 0,
        "stac_items": [],
        "stac_summary_df": pd.DataFrame(),
        "search_performed": False,
        "search_sensor": SENSOR_S2,
        "search_bbox_4326": None,
        "search_bbox_3794": None,
        "search_polygon_4326": None,
        "search_area_km2": None,
        # Tab 1 state
        "loaded_scene_ds": None,
        "loaded_scene_meta": None,
        "loaded_tif_bytes": None,
        "loaded_tif_filename": None,
        # Tab 2 state (optical & SAR)
        "tab2_index_da": None,
        "tab2_index_stats": None,
        "tab2_index_meta": None,
        "tab2_tif_bytes": None,
        "tab2_tif_filename": None,
        "tab2_sar_ds": None,
        "tab2_sar_meta": None,
        # Tab 3 state
        "tab3_ts_df": None,
        "tab3_ts_meta": None,
        "tab3_change_t1": None,
        "tab3_change_t2": None,
        "tab3_change_diff": None,
        "tab3_change_meta": None,
        "tab3_change_tif_bytes": None,
        "tab3_change_tif_filename": None,
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
        "loaded_scene_meta",
        "loaded_tif_bytes",
        "loaded_tif_filename",
        "tab2_index_da",
        "tab2_index_stats",
        "tab2_index_meta",
        "tab2_tif_bytes",
        "tab2_tif_filename",
        "tab2_sar_ds",
        "tab2_sar_meta",
        "tab3_ts_df",
        "tab3_ts_meta",
        "tab3_change_t1",
        "tab3_change_t2",
        "tab3_change_diff",
        "tab3_change_meta",
        "tab3_change_tif_bytes",
        "tab3_change_tif_filename",
    ):
        st.session_state[key] = None
    gc.collect()


def _format_tile_label(tile: str, sensor: str = SENSOR_S2) -> str:
    clean = str(tile or "").strip()
    if not clean:
        return "—"
    if sensor == SENSOR_S2:
        return clean if clean.startswith("T") else f"T{clean}"
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


def _polygon_bounds_to_bboxes(poly_geom_4326: dict[str, Any]) -> tuple[list[float], list[float], float, float]:
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


def _build_aoi_folium_map(
    center_lat: float,
    center_lon: float,
    bbox_3794: list[float] | None = None,
    bbox_4326: list[float] | None = None,
    polygon_geojson_4326: dict[str, Any] | None = None,
    zoom_start: int = SLOVENIA_DEFAULT_ZOOM,
    interactive_hint: bool = True,
) -> folium.Map:
    """Ustvari Folium zemljevid s kartografsko oznako AOI v temno petrol-zeleni barvi."""
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name=t("map_layer_sat"),
        overlay=False,
        control=True,
    ).add_to(fmap)

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
        radius=5,
        color="#0F172A",
        weight=2,
        fill=True,
        fill_color="#0F766E",
        fill_opacity=0.95,
        tooltip=tooltip_txt,
    ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def _build_draw_folium_map(
    center_lat: float,
    center_lon: float,
    existing_polygon_4326: dict[str, Any] | None = None,
    zoom_start: int = 9,
) -> folium.Map:
    """Ustvari interaktivni Folium zemljevid z orodjem Draw (poligon in pravokotnik)."""
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name=t("map_layer_sat"),
        overlay=False,
        control=True,
    ).add_to(fmap)

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

    folium.LayerControl(collapsed=False).add_to(fmap)
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


def _raster_extent_and_origin(
    obj: xr.DataArray | xr.Dataset,
    res_fallback: float = 10.0,
) -> tuple[list[float], str]:
    x_coords = obj.coords["x"].values
    y_coords = obj.coords["y"].values
    dx = abs(float(x_coords[1] - x_coords[0])) / 2.0 if x_coords.size > 1 else res_fallback / 2.0
    dy = abs(float(y_coords[1] - y_coords[0])) / 2.0 if y_coords.size > 1 else res_fallback / 2.0
    extent = [
        float(x_coords.min()) - dx,
        float(x_coords.max()) + dx,
        float(y_coords.min()) - dy,
        float(y_coords.max()) + dy,
    ]
    origin = "upper" if (y_coords.size < 2 or y_coords[0] >= y_coords[-1]) else "lower"
    return extent, origin


def _style_geodetic_axes(ax: plt.Axes, target_crs: str, grid_color: str = "#FFFFFF") -> None:
    if target_crs == CRS_D96TM:
        ax.set_xlabel(t("axis_e_d96"), fontsize=9.5, fontweight="bold", color="#0F172A")
        ax.set_ylabel(t("axis_n_d96"), fontsize=9.5, fontweight="bold", color="#0F172A")
    else:
        ax.set_xlabel(t("axis_e_utm"), fontsize=9.5, fontweight="bold", color="#0F172A")
        ax.set_ylabel(t("axis_n_utm"), fontsize=9.5, fontweight="bold", color="#0F172A")

    ax.ticklabel_format(style="plain", useOffset=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
    ax.tick_params(axis="x", labelrotation=15, labelsize=8.5, colors="#334155")
    ax.tick_params(axis="y", labelsize=8.5, colors="#334155")
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.55, color=grid_color)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(0.9)


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


def _compose_rgb_image(
    ds: xr.Dataset,
    bands: list[str],
    use_percentile_stretch: bool = True,
    gamma: float = 1.0,
) -> np.ndarray:
    channels: list[np.ndarray] = []
    valid_union = np.ones(ds[bands[0]].shape, dtype=bool)
    for b in bands:
        arr = ds[b].values.astype(np.float32, copy=False)
        valid_union &= np.isfinite(arr)
        channels.append(arr)

    stretched_channels = [
        np.where(
            valid_union,
            _stretch_channel(arr, use_percentile_stretch=use_percentile_stretch, fixed_range=(0.0, 0.35), gamma=gamma),
            np.nan,
        )
        for arr in channels
    ]
    rgb = np.stack(stretched_channels, axis=-1)
    alpha = np.where(valid_union, 1.0, 0.0).astype(np.float32)
    rgba = np.concatenate([np.nan_to_num(rgb, nan=0.0), alpha[..., np.newaxis]], axis=-1)
    return np.clip(rgba, 0.0, 1.0)


def _render_rgb_figure(
    ds: xr.Dataset,
    bands: list[str],
    composite_label: str,
    target_crs: str,
    use_percentile_stretch: bool,
    gamma: float,
    sensor_name: str = "Sentinel-2 L2A",
    tile_lbl: str = "—",
) -> plt.Figure:
    rgba = _compose_rgb_image(ds, bands, use_percentile_stretch, gamma)
    res = float(ds.attrs.get("resolution_m", 10.0))
    extent, origin = _raster_extent_and_origin(ds, res_fallback=res)

    fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=130, facecolor="#FFFFFF")
    ax.set_facecolor("#F1F5F9")
    ax.imshow(rgba, extent=extent, origin=origin, interpolation="nearest")
    _style_geodetic_axes(ax, target_crs, grid_color="#FFFFFF")

    date_str = str(ds.attrs.get("datetime", ""))[:10]
    ax.set_title(
        f"{sensor_name} ({date_str} | {tile_lbl}) — {composite_label}\n"
        + t("fig_res_bands", crs=target_crs, res=res, bands="-".join(bands)),
        fontsize=10,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    fig.tight_layout()
    return fig


def _render_sar_figure(
    ds: xr.Dataset,
    vis_code: str,
    vis_label: str,
    target_crs: str,
    use_percentile_stretch: bool,
    gamma: float,
    tile_lbl: str = "—",
) -> plt.Figure:
    """Izriše Sentinel-1 SAR RTC sliko (VV dB, VH dB ali SAR False Color R:VV G:VH B:VV/VH)."""
    res = float(ds.attrs.get("resolution_m", 10.0))
    extent, origin = _raster_extent_and_origin(ds, res_fallback=res)
    date_str = str(ds.attrs.get("datetime", ""))[:10]

    fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=130, facecolor="#FFFFFF")
    ax.set_facecolor("#E2E8F0")

    if vis_code in ("VV", "VH"):
        arr = ds[vis_code].values.astype(np.float32, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0 and use_percentile_stretch:
            vmin, vmax = np.percentile(finite, [2.0, 98.0])
            if vmax <= vmin:
                vmin, vmax = float(finite.min()), float(finite.max())
        else:
            vmin, vmax = (-22.0, -2.0) if vis_code == "VV" else (-28.0, -8.0)

        cmap = matplotlib.colormaps["Greys_r"].copy()
        cmap.set_bad(color="#E2E8F0")
        im = ax.imshow(
            arr,
            extent=extent,
            origin=origin,
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
        cbar.set_label(f"{vis_code} [dB]", fontsize=9.5, fontweight="bold", color="#0F172A")
        cbar.ax.tick_params(labelsize=8.5, colors="#334155")
        _style_geodetic_axes(ax, target_crs, grid_color="#0F766E")
        bands_txt = f"{vis_code} [dB]"
    else:
        # SAR False Color: R = VV [dB], G = VH [dB], B = VV - VH [dB] (VV/VH ratio)
        vv = ds["VV"].values.astype(np.float32, copy=False)
        vh = ds["VH"].values.astype(np.float32, copy=False)
        valid = np.isfinite(vv) & np.isfinite(vh)
        ratio = np.where(valid, vv - vh, np.nan)

        r_ch = _stretch_channel(vv, use_percentile_stretch=use_percentile_stretch, fixed_range=(-20.0, 0.0), gamma=gamma)
        g_ch = _stretch_channel(vh, use_percentile_stretch=use_percentile_stretch, fixed_range=(-26.0, -6.0), gamma=gamma)
        b_ch = _stretch_channel(ratio, use_percentile_stretch=use_percentile_stretch, fixed_range=(2.0, 14.0), gamma=gamma)

        rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
        alpha = np.where(valid, 1.0, 0.0).astype(np.float32)
        rgba = np.concatenate([np.nan_to_num(rgb, nan=0.0), alpha[..., np.newaxis]], axis=-1)
        ax.imshow(np.clip(rgba, 0.0, 1.0), extent=extent, origin=origin, interpolation="nearest")
        _style_geodetic_axes(ax, target_crs, grid_color="#FFFFFF")
        bands_txt = "R:VV · G:VH · B:VV/VH [dB]"

    ax.set_title(
        f"Sentinel-1 SAR RTC ({date_str} | {tile_lbl}) — {vis_label}\n"
        + t("fig_res_bands", crs=target_crs, res=res, bands=bands_txt),
        fontsize=10,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    fig.tight_layout()
    return fig


def _render_sar_water_figure(
    mask_da: xr.DataArray,
    threshold_db: float,
    target_crs: str,
    date_str: str,
    tile_lbl: str,
    resolution_m: float,
) -> plt.Figure:
    """Izriše dvoklasno kartografsko masko voda/poplav iz SAR VV [dB] (brez vijolične)."""
    extent, origin = _raster_extent_and_origin(mask_da, res_fallback=resolution_m)
    raw = mask_da.values
    plot_arr = np.where(raw == 255, np.nan, raw.astype(np.float32))

    # 0 = Kopno (#E2E8F0), 1 = Voda / Poplava (#0369A1)
    cmap = ListedColormap(["#E2E8F0", "#0369A1"])
    cmap.set_bad(color="#FFFFFF")

    fig, ax = plt.subplots(figsize=(8.6, 7.1), dpi=130, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.imshow(
        plot_arr,
        extent=extent,
        origin=origin,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    _style_geodetic_axes(ax, target_crs, grid_color="#475569")

    legend_handles = [
        Patch(facecolor="#0369A1", edgecolor="#0F172A", linewidth=0.7, label=t("sar_water_legend_water")),
        Patch(facecolor="#E2E8F0", edgecolor="#0F172A", linewidth=0.7, label=t("sar_water_legend_land")),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9,
    )

    ax.set_title(
        f"{t('sar_water_map_title', thr=threshold_db, date=date_str)} | {tile_lbl}\n"
        f"{target_crs} ({resolution_m:.0f} m)",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    fig.tight_layout()
    return fig


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


def _render_index_figure(
    index_da: xr.DataArray,
    index_key: str,
    target_crs: str,
    date_str: str,
    tile_lbl: str,
    resolution_m: float,
    sensor_name: str = "",
) -> plt.Figure:
    spec = SPECTRAL_INDICES[index_key]
    extent, origin = _raster_extent_and_origin(index_da, res_fallback=resolution_m)

    cmap = matplotlib.colormaps[spec["cmap"]].copy()
    cmap.set_bad(color="#E2E8F0")

    fig, ax = plt.subplots(figsize=(8.6, 7.1), dpi=130, facecolor="#FFFFFF")
    ax.set_facecolor("#E2E8F0")
    im = ax.imshow(
        index_da.values,
        extent=extent,
        origin=origin,
        cmap=cmap,
        vmin=spec["vmin"],
        vmax=spec["vmax"],
        interpolation="nearest",
    )
    _style_geodetic_axes(ax, target_crs, grid_color="#0F172A")

    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label(index_key, fontsize=9.5, fontweight="bold", color="#0F172A")
    cbar.ax.tick_params(labelsize=8.5, colors="#334155")

    prefix = f"{sensor_name} · " if sensor_name else ""
    ax.set_title(
        f"{prefix}{index_key} ({date_str} | {tile_lbl}) — {target_crs} ({resolution_m:.0f} m)",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    fig.tight_layout()
    return fig


def _render_change_figure(
    da_t1: xr.DataArray,
    da_t2: xr.DataArray,
    da_diff: xr.DataArray,
    index_key: str,
    target_crs: str,
    date_t1: str,
    date_t2: str,
    resolution_m: float,
    is_sar: bool = False,
) -> plt.Figure:
    spec = SAR_METRICS[index_key] if is_sar else SPECTRAL_INDICES[index_key]
    extent, origin = _raster_extent_and_origin(da_diff, res_fallback=resolution_m)

    idx_cmap = matplotlib.colormaps[spec["cmap"]].copy()
    idx_cmap.set_bad(color="#E2E8F0")
    diff_cmap = matplotlib.colormaps[spec["diff_cmap"]].copy()
    diff_cmap.set_bad(color="#E2E8F0")

    valid_diff = da_diff.values[np.isfinite(da_diff.values)]
    if is_sar:
        if valid_diff.size > 0:
            abs_lim = float(np.percentile(np.abs(valid_diff), 98.0))
            abs_lim = max(min(abs_lim, 12.0), 3.0)
        else:
            abs_lim = 6.0
        unit_suffix = " [dB]"
    else:
        if valid_diff.size > 0:
            abs_lim = float(np.percentile(np.abs(valid_diff), 98.0))
            abs_lim = max(min(abs_lim, 1.0), 0.2)
        else:
            abs_lim = 0.5
        unit_suffix = ""

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), dpi=125, facecolor="#FFFFFF")

    im1 = axes[0].imshow(
        da_t1.values,
        extent=extent,
        origin=origin,
        cmap=idx_cmap,
        vmin=spec["vmin"],
        vmax=spec["vmax"],
        interpolation="nearest",
    )
    axes[0].set_title(
        t("change_panel_t1", idx=f"{index_key}{unit_suffix}", d1=date_t1),
        fontsize=10,
        fontweight="bold",
        color="#0F172A",
    )
    _style_geodetic_axes(axes[0], target_crs, grid_color="#0F172A")
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.03)
    cbar1.ax.tick_params(labelsize=8)

    im2 = axes[1].imshow(
        da_t2.values,
        extent=extent,
        origin=origin,
        cmap=idx_cmap,
        vmin=spec["vmin"],
        vmax=spec["vmax"],
        interpolation="nearest",
    )
    axes[1].set_title(
        t("change_panel_t2", idx=f"{index_key}{unit_suffix}", d2=date_t2),
        fontsize=10,
        fontweight="bold",
        color="#0F172A",
    )
    _style_geodetic_axes(axes[1], target_crs, grid_color="#0F172A")
    axes[1].set_ylabel("")
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.03)
    cbar2.ax.tick_params(labelsize=8)

    im3 = axes[2].imshow(
        da_diff.values,
        extent=extent,
        origin=origin,
        cmap=diff_cmap,
        vmin=-abs_lim,
        vmax=abs_lim,
        interpolation="nearest",
    )
    axes[2].set_title(
        t("change_panel_diff", idx=f"{index_key}{unit_suffix}"),
        fontsize=10,
        fontweight="bold",
        color="#0F172A",
    )
    _style_geodetic_axes(axes[2], target_crs, grid_color="#0F172A")
    axes[2].set_ylabel("")
    cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.03)
    cbar3.set_label(f"Δ{index_key}{unit_suffix}", fontsize=8.5, fontweight="bold", color="#0F172A")
    cbar3.ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


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


def _select_evenly_spaced_items(items: list[Any], max_epochs: int) -> list[Any]:
    """Kronološko izbere največ ``max_epochs`` enakomerno razporejenih posnetkov."""
    if len(items) <= max_epochs:
        return list(items)
    ordered = sorted(items, key=lambda it: it.datetime or it.properties.get("datetime", ""))
    indices = np.linspace(0, len(ordered) - 1, num=max_epochs, dtype=int)
    seen: set[int] = set()
    selected = []
    for idx in indices:
        i = int(idx)
        if i not in seen:
            seen.add(i)
            selected.append(ordered[i])
    return selected


# ---------------------------------------------------------------------------
# Glavni tok aplikacije
# ---------------------------------------------------------------------------
def main() -> None:
    _init_session_state()

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

        # --- 1. Satelitski senzor ---
        st.divider()
        st.subheader(t("sec_sensor"))

        sensor = st.selectbox(
            t("sensor_select_label"),
            options=SENSOR_KEYS,
            index=SENSOR_KEYS.index(st.session_state["active_sensor"]),
            format_func=lambda s, _lang=lang: t(SENSOR_META[s]["label_key"], lang=_lang),
            key="sidebar_sensor_select",
        )

        # Ob menjavi senzorja počistimo stare rezultate iskanja in naložene rastre
        if sensor != st.session_state["active_sensor"]:
            st.session_state["active_sensor"] = sensor
            st.session_state["stac_items"] = []
            st.session_state["stac_summary_df"] = pd.DataFrame()
            st.session_state["search_performed"] = False
            st.session_state["search_sensor"] = sensor
            st.session_state["last_rec_res_key"] = None
            st.session_state["selected_resolution"] = SENSOR_META[sensor]["default_res"]
            _clear_loaded_products()

        sensor_cfg = SENSOR_META[sensor]
        exclude_l7_slc_off = False
        if sensor == SENSOR_LANDSAT:
            exclude_l7_slc_off = st.checkbox(
                t("l7_exclude_label"),
                value=False,
                help=t("l7_exclude_help"),
                key="sidebar_l7_slc_exclude",
            )
        elif sensor == SENSOR_S1:
            st.info(t("sar_allweather_note"))

        # --- 2. Območje zanimanja (AOI) ---
        st.divider()
        st.subheader(t("sec_aoi"))

        aoi_mode_key = st.radio(
            t("aoi_mode_label"),
            options=AOI_MODE_KEYS,
            index=0,
            format_func=lambda k, _lang=lang: t(AOI_MODE_I18N[k], lang=_lang),
            label_visibility="collapsed",
            key="sidebar_aoi_mode",
        )

        bbox_4326: list[float] | None = None
        bbox_3794: list[float] | None = None
        aoi_polygon_4326: dict[str, Any] | None = None
        center_lat: float = SLOVENIA_CENTER_LAT
        center_lon: float = SLOVENIA_CENTER_LON
        aoi_error: str | None = None

        # --- Način 1: Interaktivni zemljevid (klik + radij) ---
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

            st.caption(t("map_click_hint"))
            sidebar_map = _build_aoi_folium_map(
                center_lat=center_lat,
                center_lon=center_lon,
                bbox_3794=bbox_3794,
                zoom_start=SLOVENIA_DEFAULT_ZOOM,
                interactive_hint=True,
            )
            map_out = st_folium(
                sidebar_map,
                height=250,
                use_container_width=True,
                returned_objects=["last_clicked"],
                key="aoi_picker_map",
            )
            clicked = map_out.get("last_clicked") if isinstance(map_out, dict) else None
            if clicked and "lat" in clicked and "lng" in clicked:
                new_lat = round(float(clicked["lat"]), 6)
                new_lon = round(float(clicked["lng"]), 6)
                if abs(new_lat - center_lat) > 1e-6 or abs(new_lon - center_lon) > 1e-6:
                    st.session_state["map_lat"] = new_lat
                    st.session_state["map_lon"] = new_lon
                    st.rerun()

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

        # --- 3. Geodetske in RAM nastavitve ---
        st.divider()
        st.subheader(t("sec_geodetic"))

        area_km2: float = 0.0
        aoi_valid: bool = False
        rec_res: int = sensor_cfg["default_res"]

        if bbox_4326 is not None and aoi_error is None:
            if aoi_polygon_4326 is not None:
                area_km2, aoi_valid, base_rec_res = validate_aoi_geometry(aoi_polygon_4326)
                # Preverimo tudi obseg okvirja (zaradi RAM pri nalaganju pred rio.clip)
                bbox_area_km2, bbox_ok, bbox_rec_res = validate_aoi_size(bbox_4326)
                aoi_valid = aoi_valid and bbox_ok
                base_rec_res = max(base_rec_res, bbox_rec_res)
            else:
                area_km2, aoi_valid, base_rec_res = validate_aoi_size(bbox_4326)

            # Prilagodi priporočeno ločljivost senzorju (Landsat min 30 m)
            if sensor == SENSOR_LANDSAT:
                rec_res = 60 if base_rec_res >= 60 else 30
            else:
                rec_res = base_rec_res

            area_ha = area_km2 * 100.0

            c_km, c_ha = st.columns(2)
            c_km.metric(t("metric_area_km2"), f"{area_km2:,.2f}")
            c_ha.metric(t("metric_area_ha"), f"{area_ha:,.1f}")

            res_key = f"{sensor}_{round(area_km2, 1)}_{rec_res}"
            if st.session_state["last_rec_res_key"] != res_key:
                st.session_state["last_rec_res_key"] = res_key
                st.session_state["selected_resolution"] = rec_res

            if not aoi_valid:
                st.error(t("aoi_too_large", area=area_km2))
            elif area_km2 > 100.0:
                st.warning(t("aoi_large_warn", area=area_km2, res=rec_res))

            if bbox_3794 is not None:
                with st.expander(t("bbox_expander"), expanded=False):
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

        res_options: list[int] = list(sensor_cfg["res_options"])
        cur_sel_res = st.session_state["selected_resolution"]
        default_res_idx = res_options.index(cur_sel_res) if cur_sel_res in res_options else 0
        selected_resolution = st.selectbox(
            t("res_label"),
            options=res_options,
            index=default_res_idx,
            format_func=lambda r, _lang=lang: (
                t("res_recommended", lang=_lang, r=r)
                if (bbox_4326 is not None and r == rec_res)
                else t("res_plain", lang=_lang, r=r)
            ),
            help=t("res_help"),
            key=f"sidebar_res_select_{sensor}",
        )
        st.session_state["selected_resolution"] = selected_resolution

        if bbox_4326 is not None and aoi_valid and selected_resolution < rec_res:
            st.warning(t("res_finer_warn", sel=selected_resolution, rec=rec_res, area=area_km2))

        target_crs = st.selectbox(
            t("output_crs_label"),
            options=CRS_KEYS,
            index=0,
            format_func=lambda c, _lang=lang: t(CRS_I18N[c], lang=_lang),
            key="sidebar_crs_select",
        )

        # --- 4. Časovno obdobje in oblačnost ---
        st.divider()
        st.subheader(t("sec_time"))

        today = date.today()
        min_sensor_date: date = sensor_cfg["min_date"]
        default_start = today - timedelta(days=365 if sensor == SENSOR_LANDSAT else 183)
        if default_start < min_sensor_date:
            default_start = min_sensor_date

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input(
                t("start_date_label"),
                value=default_start,
                min_value=min_sensor_date,
                max_value=today,
                key=f"start_date_{sensor}",
            )
        with col_d2:
            end_date = st.date_input(
                t("end_date_label"),
                value=today,
                min_value=min_sensor_date,
                max_value=today,
                key=f"end_date_{sensor}",
            )

        dates_valid = start_date <= end_date
        if not dates_valid:
            st.error(t("date_order_error"))

        if sensor == SENSOR_S1:
            max_cloud_cover = 100
        else:
            max_cloud_cover = st.slider(
                t("max_cloud_label"),
                min_value=0,
                max_value=100,
                value=20,
                step=1,
                key="sidebar_max_cloud",
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
                    items, summary_df = search_scenes(
                        sensor=sensor,
                        bbox_4326=bbox_4326,
                        start_date=start_date,
                        end_date=end_date,
                        max_cloud_cover=float(max_cloud_cover),
                        limit=50,
                        exclude_l7_slc_off=exclude_l7_slc_off,
                    )
                    st.session_state["stac_items"] = items
                    st.session_state["stac_summary_df"] = summary_df
                    st.session_state["search_performed"] = True
                    st.session_state["search_sensor"] = sensor
                    st.session_state["search_bbox_4326"] = list(bbox_4326)
                    st.session_state["search_bbox_3794"] = list(bbox_3794) if bbox_3794 else None
                    st.session_state["search_polygon_4326"] = aoi_polygon_4326
                    st.session_state["search_area_km2"] = area_km2
                    _clear_loaded_products()
                except Exception as exc:
                    st.error(t("search_error", err=exc))

    # =======================================================================
    # GLAVNI DEL APLIKACIJE: TEHNIČNA GLAVA, INTERAKTIVNO RISANJE IN ZAVIHKI
    # =======================================================================
    active_polygon_badge = ""
    if aoi_polygon_4326 is not None or st.session_state.get("search_polygon_4326") is not None:
        active_polygon_badge = f'<span class="geo-badge">✂️ {t("polygon_clip_badge")}</span>'

    st.markdown(
        f"""
        <div class="geo-header">
            <div class="geo-header-title">{t('app_title')}</div>
            <div class="geo-header-sub">{t('app_subtitle')}</div>
            <div class="geo-badges">
                <span class="geo-badge">{t('badge_crs')}</span>
                <span class="geo-badge">{t('badge_sensor')}</span>
                <span class="geo-badge">{t('badge_radiometry')}</span>
                {active_polygon_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prostorno platno za interaktivno risanje poligona (ko je izbran način "draw")
    if aoi_mode_key == "draw":
        draw_expanded = (st.session_state.get("drawn_polygon_4326") is None) or (
            not st.session_state["search_performed"]
        )
        with st.expander(t("draw_main_header"), expanded=draw_expanded):
            st.caption(t("draw_main_hint"))
            draw_map = _build_draw_folium_map(
                center_lat=center_lat,
                center_lon=center_lon,
                existing_polygon_4326=st.session_state.get("drawn_polygon_4326"),
                zoom_start=12 if st.session_state.get("drawn_polygon_4326") is not None else 9,
            )
            draw_ver = int(st.session_state.get("draw_map_version", 0))
            draw_out = st_folium(
                draw_map,
                height=440,
                use_container_width=True,
                returned_objects=["last_active_drawing", "all_drawings"],
                key=f"aoi_draw_map_{draw_ver}",
            )
            extracted_poly = _extract_drawn_polygon(draw_out)
            if extracted_poly is not None:
                prev_poly = st.session_state.get("drawn_polygon_4326")
                if json.dumps(extracted_poly, sort_keys=True) != json.dumps(prev_poly, sort_keys=True):
                    st.session_state["drawn_polygon_4326"] = extracted_poly
                    st.rerun()

    tab1, tab2, tab3 = st.tabs([t("tab1"), t("tab2"), t("tab3")])

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
            default_scene_idx = int(summary_df["cloud_cover"].idxmin())

    # =======================================================================
    # ZAVIHEK 1: ISKANJE POSNETKOV, PREGLED (OPTIKA / SAR) IN GEOTIFF IZVOZ
    # =======================================================================
    with tab1:
        if not search_performed:
            st.info(t("onboarding_info"))
            if bbox_4326 is not None and aoi_valid and aoi_mode_key != "draw":
                st.subheader(t("preview_aoi_header"))
                preview_map = _build_aoi_folium_map(
                    center_lat=center_lat,
                    center_lon=center_lon,
                    bbox_3794=bbox_3794,
                    bbox_4326=bbox_4326,
                    polygon_geojson_4326=aoi_polygon_4326,
                    zoom_start=12 if area_km2 <= 50 else 10,
                    interactive_hint=False,
                )
                st_folium(
                    preview_map,
                    height=380,
                    use_container_width=True,
                    returned_objects=[],
                    key="main_aoi_preview_map",
                )
        elif not items or summary_df.empty:
            st.warning(t("no_scenes_warn"))
        else:
            st.subheader(
                t("t1_found_header", sensor=active_sensor_cfg["short_name"], n=len(items))
            )

            if active_sensor == SENSOR_S1:
                cloud_col = pd.Series(["— (SAR)"] * len(summary_df))
            else:
                cloud_col = summary_df["cloud_cover"].round(2)

            display_df = pd.DataFrame(
                {
                    t("col_date"): summary_df["date_str"],
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

            st.divider()
            st.subheader(t("t1_vis_header"))

            is_sar_active = active_sensor == SENSOR_S1
            vis_specs = SAR_VIS_SPECS if is_sar_active else OPTICAL_COMPOSITE_SPECS

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
                vis_code = st.selectbox(
                    t("composite_select_label"),
                    options=list(vis_specs.keys()),
                    index=0,
                    format_func=lambda code, _lang=lang: t(vis_specs[code]["label_key"], lang=_lang),
                    key=f"tab1_vis_select_{active_sensor}",
                )

            vis_spec = vis_specs[vis_code]
            st.caption(t(vis_spec["desc_key"]))

            col_btn, col_stretch, col_gamma = st.columns([1.2, 0.9, 1.0])
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
                        st.session_state["loaded_tif_bytes"] = None
                        gc.collect()

                        ds_loaded = load_scene(
                            sensor=active_sensor,
                            item=chosen_item,
                            bbox_4326=active_bbox,
                            bands=req_bands,
                            target_epsg=target_crs,
                            resolution=float(selected_resolution),
                            aoi_polygon_4326=active_polygon,
                        )
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
                        }
                        st.session_state["loaded_tif_bytes"] = tif_bytes
                        st.session_state["loaded_tif_filename"] = tif_filename
                        gc.collect()
                    except Exception as exc:
                        st.error(t("load_scene_error", err=exc))

            ds_show: xr.Dataset | None = st.session_state["loaded_scene_ds"]
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

                if active_sensor == SENSOR_S1:
                    vis_lbl = t(SAR_VIS_SPECS[meta["vis_code"]]["label_key"])
                    fig = _render_sar_figure(
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
                    fig = _render_rgb_figure(
                        ds=ds_show,
                        bands=meta["bands"],
                        composite_label=vis_lbl,
                        target_crs=meta["crs"],
                        use_percentile_stretch=use_percentile,
                        gamma=gamma_val,
                        sensor_name=meta["sensor_name"],
                        tile_lbl=meta["mgrs_tile"],
                    )

                st.pyplot(fig, width="stretch")
                plt.close(fig)

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
                        gc.collect()

                        ds_vv = load_scene(
                            sensor=SENSOR_S1,
                            item=chosen_item,
                            bbox_4326=active_bbox,
                            bands=["vv"],
                            target_epsg=target_crs,
                            resolution=float(selected_resolution),
                            aoi_polygon_4326=active_polygon,
                        )
                        st.session_state["tab2_sar_ds"] = ds_vv
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
            sar_meta: dict[str, Any] | None = st.session_state["tab2_sar_meta"]

            if ds_sar_loaded is not None and sar_meta is not None:
                st.divider()
                # Dinamični ponovni izračun ob premiku drsnika praga (brez ponovnega prenosa!)
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

                    fig_water = _render_sar_water_figure(
                        mask_da=mask_da,
                        threshold_db=float(sar_thr_db),
                        target_crs=sar_meta["crs"],
                        date_str=sar_meta["date_str"],
                        tile_lbl=sar_meta["mgrs_tile"],
                        resolution_m=float(sar_meta["resolution_m"]),
                    )
                    st.pyplot(fig_water, width="stretch")
                    plt.close(fig_water)

                    hist_vv = _build_sar_vv_histogram(
                        vv_da=ds_sar_loaded["VV"],
                        threshold_db=float(sar_thr_db),
                        date_str=sar_meta["date_str"],
                    )
                    st.plotly_chart(hist_vv, width="stretch")

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

        # --- Način 2B: Optični spektralni indeksi (Sentinel-2 & Landsat C2 L2) ---
        else:
            st.subheader(t("t2_header"))

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
                t2_index_key = st.selectbox(
                    t("index_select_label"),
                    options=INDEX_KEYS,
                    index=0,
                    format_func=lambda k, _lang=lang: t(f"idx_{k}_label", lang=_lang),
                    key="tab2_index_select",
                )

            st.caption(
                f"{t(f'idx_{t2_index_key}_desc')} — `{SPECTRAL_INDICES[t2_index_key]['formula']}`"
            )

            col_scl2, col_btn2 = st.columns([1.15, 1.0])
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
                        st.session_state["tab2_tif_bytes"] = None
                        gc.collect()

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
                        st.session_state["tab2_index_stats"] = stats
                        st.session_state["tab2_index_meta"] = {
                            "sensor": active_sensor,
                            "sensor_name": active_sensor_cfg["short_name"],
                            "index_key": t2_index_key,
                            "date_str": str(chosen_row["date_str"]),
                            "mgrs_tile": _format_tile_label(
                                str(chosen_row["mgrs_tile"]), sensor=active_sensor
                            ),
                            "crs": target_crs,
                            "resolution_m": int(selected_resolution),
                            "mask_clouds": bool(t2_mask_clouds),
                        }
                        st.session_state["tab2_tif_bytes"] = tif_bytes
                        st.session_state["tab2_tif_filename"] = tif_filename
                        gc.collect()
                    except Exception as exc:
                        st.error(t("index_error", err=exc))

            idx_da: xr.DataArray | None = st.session_state["tab2_index_da"]
            idx_stats: dict[str, Any] | None = st.session_state["tab2_index_stats"]
            idx_meta: dict[str, Any] | None = st.session_state["tab2_index_meta"]

            if (
                idx_da is not None
                and idx_stats is not None
                and idx_meta is not None
                and idx_meta.get("sensor") == active_sensor
            ):
                st.divider()
                if idx_stats["valid_pixels"] == 0:
                    st.warning(t("t2_no_valid_pixels"))
                else:
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric(t("m_mean"), f"{idx_stats['mean']:.3f}")
                    k2.metric(t("m_median"), f"{idx_stats['median']:.3f}")
                    k3.metric(t("m_std"), f"{idx_stats['std']:.3f}")
                    k4.metric(t("m_valid_pct"), f"{idx_stats['valid_pct']:.1f} %")
                    k5.metric(t("m_valid_ha"), f"{idx_stats['valid_area_ha']:,.2f} ha")

                    fig_idx = _render_index_figure(
                        index_da=idx_da,
                        index_key=idx_meta["index_key"],
                        target_crs=idx_meta["crs"],
                        date_str=idx_meta["date_str"],
                        tile_lbl=idx_meta["mgrs_tile"],
                        resolution_m=float(idx_meta["resolution_m"]),
                        sensor_name=idx_meta["sensor_name"],
                    )
                    st.pyplot(fig_idx, width="stretch")
                    plt.close(fig_idx)

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
                    st.plotly_chart(hist_fig, width="stretch")

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

    # =======================================================================
    # ZAVIHEK 3: ČASOVNE VRSTE IN ZAZNAVA SPREMEMB (Δ = t2 - t1)
    # =======================================================================
    with tab3:
        if not search_performed or not items or summary_df.empty:
            st.info(t("need_search_first"))
        else:
            is_sar_t3 = active_sensor == SENSOR_S1
            metric_options = SAR_METRIC_KEYS if is_sar_t3 else INDEX_KEYS
            metric_label_title = (
                t("sar_metric_select_label") if is_sar_t3 else t("index_select_label")
            )

            def _fmt_metric(code: str, _lang: str = lang) -> str:
                return (
                    t(f"sar_{code}_label", lang=_lang)
                    if is_sar_t3
                    else t(f"idx_{code}_label", lang=_lang)
                )

            # ---------------------------------------------------------------
            # SEKCIJA A: ČASOVNA VRSTA
            # ---------------------------------------------------------------
            st.subheader(t("t3_sec_a_header"))
            st.caption(t("t3_sec_a_caption"))

            col_ts_idx, col_ts_ep, col_ts_scl = st.columns([1.1, 1.0, 0.9])
            with col_ts_idx:
                ts_metric_key = st.selectbox(
                    metric_label_title,
                    options=metric_options,
                    index=0,
                    format_func=_fmt_metric,
                    key=f"tab3_ts_metric_{active_sensor}",
                )
            with col_ts_ep:
                total_available = len(items)
                if total_available <= 5:
                    max_epochs = total_available
                    st.caption(f"{t('t3_epochs_slider')} **{total_available}**")
                else:
                    upper_ep = min(40, total_available)
                    default_ep = min(15, upper_ep)
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
                    st.plotly_chart(ts_fig, width="stretch")

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
            # SEKCIJA B: ZAZNAVA SPREMEMB (Δ = t2 - t1)
            # ---------------------------------------------------------------
            st.divider()
            st.subheader(t("t3_sec_b_header"))
            st.caption(t("t3_sec_b_caption"))

            if len(items) < 2:
                st.info(t("t3_need_two_scenes"))
            else:
                col_c_idx, col_c_thr, col_c_scl = st.columns([1.15, 1.0, 0.85])
                with col_c_idx:
                    ch_metric_key = st.selectbox(
                        metric_label_title,
                        options=metric_options,
                        index=0,
                        format_func=_fmt_metric,
                        key=f"tab3_ch_metric_{active_sensor}",
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

                # Privzeto: t1 najstarejši posnetek (zadnji v newest-first seznamu), t2 najnovejši (0)
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
                    st.divider()
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

                        fig_ch = _render_change_figure(
                            da_t1=da_t1_s,
                            da_t2=da_t2_s,
                            da_diff=da_diff_s,
                            index_key=ch_meta["index_key"],
                            target_crs=ch_meta["crs"],
                            date_t1=ch_meta["date_t1"],
                            date_t2=ch_meta["date_t2"],
                            resolution_m=float(ch_meta["resolution_m"]),
                            is_sar=is_sar_t3,
                        )
                        st.pyplot(fig_ch, width="stretch")
                        plt.close(fig_ch)

                        diff_bytes = st.session_state["tab3_change_tif_bytes"]
                        if diff_bytes:
                            st.download_button(
                                label=t("btn_download_change_tif"),
                                data=diff_bytes,
                                file_name=st.session_state["tab3_change_tif_filename"] or "change_map.tif",
                                mime="image/tiff",
                                width="stretch",
                            )


if __name__ == "__main__":
    main()
