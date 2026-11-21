"""Geodetic GNSS Mission Planning Engine (Orbits + Copernicus DEM 30 m + Sentinel-2 10 m Obstacles).

Computes 24-hour multi-constellation GNSS satellite visibility (GPS, Galileo, GLONASS)
and strict topocentric ENU Dilution of Precision (HDOP, VDOP, PDOP, GDOP) combining:
1. Live CelesTrak TLE ephemerides via ``skyfield`` (cached 6 hours) with an automatic
   built-in Walker nominal constellation propagator fallback when offline.
2. 360° (72 × 5° azimuth sectors) terrain horizon obstruction mask derived from
   Copernicus DEM 30 m (``cop-dem-glo-30`` on Planetary Computer STAC) in ``EPSG:3794``.
3. Local vegetation canopy & building obstruction mask derived from the latest clear
   10 m Sentinel-2 L2A scene (NDVI ≥ 0.50 forest; NDBI > 0.05 & NDVI < 0.20 built-up)
   within 150 m of the GNSS station in ``EPSG:3794``.
"""

from __future__ import annotations

import gc
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import requests
import rioxarray  # noqa: F401
import xarray as xr

from analize import apply_cloud_mask, compute_index
from geodezija_crs import (
    CRS_D96TM,
    EPSG_D96TM,
    EPSG_WGS84,
    bbox_from_point_wgs84,
    transform_coords,
)
from stac_loader import (
    STAC_API_URL,
    SENSOR_S2,
    load_scene,
    search_scenes,
)

# ---------------------------------------------------------------------------
# Constants: Constellations, CelesTrak Endpoints & Colors (No Purple)
# ---------------------------------------------------------------------------
CONSTELLATIONS: tuple[str, ...] = ("GPS", "Galileo", "GLONASS")

CELESTRAK_TLE_URLS: dict[str, str] = {
    "GPS": "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",
    "Galileo": "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle",
    "GLONASS": "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle",
}

CONSTELLATION_COLORS: dict[str, str] = {
    "GPS": "#0F766E",      # Cartographic Teal
    "Galileo": "#0284C7",  # Ocean Blue
    "GLONASS": "#D97706",  # Warm Amber
}

CONSTELLATION_PREFIX: dict[str, str] = {
    "GPS": "G",
    "Galileo": "E",
    "GLONASS": "R",
}

N_AZ_SECTORS: int = 72
SECTOR_WIDTH_DEG: float = 5.0
AZ_SECTOR_CENTERS_DEG: np.ndarray = (np.arange(N_AZ_SECTORS, dtype=np.float64) + 0.5) * SECTOR_WIDTH_DEG

# WGS84 ellipsoid parameters for exact ECEF / ENU transforms
_WGS84_A: float = 6_378_137.0
_WGS84_F: float = 1.0 / 298.257223563
_WGS84_E2: float = _WGS84_F * (2.0 - _WGS84_F)


# ---------------------------------------------------------------------------
# 1. CelesTrak TLE Fetcher (Cached 6 h) + Walker Nominal Fallback
# ---------------------------------------------------------------------------
def _parse_tle_text(text: str, constellation: str) -> list[tuple[str, str, str]]:
    """Parse 3-line TLE text into ``[(short_label, line1, line2), ...]``."""
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    entries: list[tuple[str, str, str]] = []
    prefix = CONSTELLATION_PREFIX.get(constellation, "S")
    idx = 0
    sat_num = 1
    while idx + 2 < len(lines):
        name_ln = lines[idx]
        l1 = lines[idx + 1]
        l2 = lines[idx + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            # Extract compact PRN label if present, e.g. "(PRN 14)" or "GSAT0201"
            short = f"{prefix}{sat_num:02d}"
            upper_name = name_ln.upper()
            prn_pos = upper_name.find("PRN ")
            if prn_pos >= 0:
                digits = "".join(ch for ch in upper_name[prn_pos + 4 : prn_pos + 7] if ch.isdigit())
                if digits:
                    short = f"{prefix}{int(digits):02d}"
            entries.append((short, l1, l2))
            sat_num += 1
            idx += 3
        else:
            idx += 1
    return entries


def _fetch_celestrak_tle_uncached(constellations: tuple[str, ...]) -> dict[str, list[tuple[str, str, str]]]:
    result: dict[str, list[tuple[str, str, str]]] = {}
    headers = {"User-Agent": "DaljinskoZaznavanje-GNSS-Planner/1.0"}
    for const in constellations:
        url = CELESTRAK_TLE_URLS.get(const)
        if not url:
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            resp.raise_for_status()
            parsed = _parse_tle_text(resp.text, const)
            if len(parsed) >= 6:
                result[const] = parsed
        except Exception:
            continue
    return result


try:
    import streamlit as _st

    @_st.cache_data(ttl=21600, show_spinner=False)
    def fetch_celestrak_tles(constellations: tuple[str, ...]) -> dict[str, list[tuple[str, str, str]]]:
        """Fetch and cache (6 h) live CelesTrak TLE ephemerides for GPS, Galileo, and GLONASS."""
        return _fetch_celestrak_tle_uncached(constellations)

except Exception:

    def fetch_celestrak_tles(constellations: tuple[str, ...]) -> dict[str, list[tuple[str, str, str]]]:
        return _fetch_celestrak_tle_uncached(constellations)


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    """Convert WGS84 geodetic ``(lat, lon, h)`` to ECEF ``[X, Y, Z]`` in meters."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n_val = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (n_val + h_m) * cos_lat * math.cos(lon)
    y = (n_val + h_m) * cos_lat * math.sin(lon)
    z = (n_val * (1.0 - _WGS84_E2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def _ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    s_lat, c_lat = math.sin(lat), math.cos(lat)
    s_lon, c_lon = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-s_lon, c_lon, 0.0],
            [-s_lat * c_lon, -s_lat * s_lon, c_lat],
            [c_lat * c_lon, c_lat * s_lon, s_lat],
        ],
        dtype=np.float64,
    )


def _gmst_rad(dt_utc: datetime) -> float:
    """Compute Greenwich Mean Sidereal Time [radians] from a UTC datetime."""
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    d_days = (dt_utc - j2000).total_seconds() / 86400.0
    gmst_deg = (280.46061837 + 360.98564736629 * d_days) % 360.0
    return math.radians(gmst_deg)


# Nominal Walker constellation orbital parameters (fallback when CelesTrak is unreachable)
_WALKER_SPECS: dict[str, dict[str, Any]] = {
    "GPS": {
        "n_planes": 6,
        "sats_per_plane": 4,
        "a_m": 26_560_000.0,
        "inc_deg": 55.0,
        "period_s": 43_082.0,
        "raan0_deg": 15.0,
        "phase_f": 2,
    },
    "Galileo": {
        "n_planes": 3,
        "sats_per_plane": 8,
        "a_m": 29_600_000.0,
        "inc_deg": 56.0,
        "period_s": 50_681.0,
        "raan0_deg": 77.0,
        "phase_f": 1,
    },
    "GLONASS": {
        "n_planes": 3,
        "sats_per_plane": 8,
        "a_m": 25_510_000.0,
        "inc_deg": 64.8,
        "period_s": 40_544.0,
        "raan0_deg": 135.0,
        "phase_f": 1,
    },
}


def _propagate_nominal_walker(
    constellations: Sequence[str],
    lat_deg: float,
    lon_deg: float,
    h_m: float,
    epochs_utc: Sequence[datetime],
) -> list[list[dict[str, Any]]]:
    """Propagate nominal 24-satellite Walker constellations to exact topocentric (Az, El)."""
    sta_ecef = _geodetic_to_ecef(lat_deg, lon_deg, h_m)
    rot_enu = _ecef_to_enu_matrix(lat_deg, lon_deg)
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Pre-build satellite orbital elements
    sats: list[tuple[str, str, float, float, float, float, float]] = []
    for const in constellations:
        spec = _WALKER_SPECS.get(const)
        if not spec:
            continue
        prefix = CONSTELLATION_PREFIX[const]
        n_planes = int(spec["n_planes"])
        n_per = int(spec["sats_per_plane"])
        total_sats = n_planes * n_per
        a_m = float(spec["a_m"])
        inc = math.radians(float(spec["inc_deg"]))
        n_rad_s = 2.0 * math.pi / float(spec["period_s"])
        raan0 = float(spec["raan0_deg"])
        phase_f = int(spec["phase_f"])

        prn_idx = 1
        for p in range(n_planes):
            raan = math.radians((raan0 + p * (360.0 / n_planes)) % 360.0)
            for s in range(n_per):
                u0 = math.radians((s * (360.0 / n_per) + p * phase_f * (360.0 / total_sats)) % 360.0)
                sats.append((f"{prefix}{prn_idx:02d}", const, a_m, inc, raan, u0, n_rad_s))
                prn_idx += 1

    epochs_sats: list[list[dict[str, Any]]] = []
    for dt_utc in epochs_utc:
        t_sec = (dt_utc - j2000).total_seconds()
        gmst = _gmst_rad(dt_utc)
        c_g, s_g = math.cos(gmst), math.sin(gmst)
        epoch_list: list[dict[str, Any]] = []

        for prn, const, a_m, inc, raan, u0, n_rad_s in sats:
            u = u0 + n_rad_s * t_sec
            c_u, s_u = math.cos(u), math.sin(u)
            c_o, s_o = math.cos(raan), math.sin(raan)
            c_i, s_i = math.cos(inc), math.sin(inc)

            # ECI position
            x_eci = a_m * (c_o * c_u - s_o * s_u * c_i)
            y_eci = a_m * (s_o * c_u + c_o * s_u * c_i)
            z_eci = a_m * (s_u * s_i)

            # Rotate ECI -> ECEF via GMST
            x_ecef = c_g * x_eci + s_g * y_eci
            y_ecef = -s_g * x_eci + c_g * y_eci
            z_ecef = z_eci

            dx = np.array([x_ecef, y_ecef, z_ecef], dtype=np.float64) - sta_ecef
            e_val, n_val, u_val = rot_enu @ dx
            horiz = math.hypot(e_val, n_val)
            el_deg = math.degrees(math.atan2(u_val, horiz))
            if el_deg > 0.0:
                az_deg = (math.degrees(math.atan2(e_val, n_val)) + 360.0) % 360.0
                epoch_list.append(
                    {
                        "prn": prn,
                        "constellation": const,
                        "az_deg": round(az_deg, 2),
                        "el_deg": round(el_deg, 2),
                    }
                )
        epochs_sats.append(epoch_list)
    return epochs_sats


def propagate_gnss_constellations(
    constellations: Sequence[str],
    lat_deg: float,
    lon_deg: float,
    h_m: float,
    epochs_utc: Sequence[datetime],
    *,
    use_live_tle: bool = True,
) -> tuple[list[list[dict[str, Any]]], str]:
    """Propagate GNSS orbits across ``epochs_utc`` using live CelesTrak TLEs (Skyfield) or Walker fallback.

    Returns ``(epochs_sats, orbit_source)`` where ``orbit_source`` is ``"celestrak_tle"``
    or ``"nominal_walker"``.
    """
    const_tuple = tuple(c for c in constellations if c in CONSTELLATIONS)
    if not const_tuple:
        const_tuple = ("GPS", "Galileo")

    if use_live_tle:
        try:
            from skyfield.api import EarthSatellite, load, wgs84

            tle_dict = fetch_celestrak_tles(const_tuple)
            if tle_dict:
                ts = load.timescale(builtin=True)
                t_sf = ts.from_datetimes(list(epochs_utc))
                observer = wgs84.latlon(float(lat_deg), float(lon_deg), elevation_m=float(h_m))

                epochs_sats: list[list[dict[str, Any]]] = [[] for _ in epochs_utc]
                used_any = False

                for const in const_tuple:
                    sat_entries = tle_dict.get(const, [])
                    if not sat_entries:
                        # Fill missing individual constellation with nominal Walker if needed
                        nom_part = _propagate_nominal_walker([const], lat_deg, lon_deg, h_m, epochs_utc)
                        for idx, part in enumerate(nom_part):
                            epochs_sats[idx].extend(part)
                        continue

                    for prn, l1, l2 in sat_entries:
                        try:
                            sat = EarthSatellite(l1, l2, prn, ts)
                            diff = sat - observer
                            topocentric = diff.at(t_sf)
                            alt, az, _dist = topocentric.altaz()
                            el_arr = np.asarray(alt.degrees, dtype=np.float64)
                            az_arr = np.asarray(az.degrees, dtype=np.float64)
                            used_any = True
                            for idx in range(len(epochs_utc)):
                                el_v = float(el_arr[idx])
                                if math.isfinite(el_v) and el_v > 0.0:
                                    az_v = float(az_arr[idx]) % 360.0
                                    epochs_sats[idx].append(
                                        {
                                            "prn": prn,
                                            "constellation": const,
                                            "az_deg": round(az_v, 2),
                                            "el_deg": round(el_v, 2),
                                        }
                                    )
                        except Exception:
                            continue

                if used_any:
                    return epochs_sats, "celestrak_tle"
        except Exception:
            pass

    return _propagate_nominal_walker(const_tuple, lat_deg, lon_deg, h_m, epochs_utc), "nominal_walker"


# ---------------------------------------------------------------------------
# 2. 360° Geodetic Horizon Obstruction Mask (Copernicus DEM 30 m + Sentinel-2 10 m)
# ---------------------------------------------------------------------------
def compute_horizon_mask_from_dem_array(
    dem_arr: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    station_e: float,
    station_n: float,
    *,
    ant_height_m: float = 2.0,
    station_elev_m: float | None = None,
    min_dist_m: float = 30.0,
    max_dist_m: float = 2500.0,
) -> tuple[np.ndarray, float]:
    """Compute a 72-sector (5° azimuth) terrain obstruction mask from a metric DEM grid."""
    mask_deg = np.zeros(N_AZ_SECTORS, dtype=np.float64)
    vals = np.asarray(dem_arr, dtype=np.float64)
    xv = np.asarray(x_coords, dtype=np.float64)
    yv = np.asarray(y_coords, dtype=np.float64)

    if vals.size == 0 or xv.size == 0 or yv.size == 0:
        return mask_deg, float(station_elev_m or 300.0)

    # Station ground elevation from nearest finite pixel
    if station_elev_m is None:
        ix = int(np.argmin(np.abs(xv - station_e)))
        iy = int(np.argmin(np.abs(yv - station_n)))
        h0 = float(vals[iy, ix])
        if not math.isfinite(h0):
            finite_vals = vals[np.isfinite(vals)]
            h0 = float(np.median(finite_vals)) if finite_vals.size else 300.0
    else:
        h0 = float(station_elev_m)

    xx, yy = np.meshgrid(xv, yv)
    de = xx - float(station_e)
    dn = yy - float(station_n)
    dist = np.hypot(de, dn)

    valid = np.isfinite(vals) & (dist >= float(min_dist_m)) & (dist <= float(max_dist_m))
    if not np.any(valid):
        return mask_deg, h0

    dh = vals[valid] - (h0 + float(ant_height_m))
    el_deg = np.degrees(np.arctan2(dh, dist[valid]))
    az_deg = (np.degrees(np.arctan2(de[valid], dn[valid])) + 360.0) % 360.0
    sectors = (az_deg // SECTOR_WIDTH_DEG).astype(np.int32) % N_AZ_SECTORS

    pos = el_deg > 0.0
    if np.any(pos):
        np.maximum.at(mask_deg, sectors[pos], el_deg[pos])
    return np.clip(mask_deg, 0.0, 89.0), h0


def compute_obstacle_mask_from_s2_indices(
    ndvi_arr: np.ndarray,
    ndbi_arr: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    station_e: float,
    station_n: float,
    *,
    ant_height_m: float = 2.0,
    canopy_h_m: float = 18.0,
    building_h_m: float = 12.0,
    min_dist_m: float = 10.0,
    max_dist_m: float = 150.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify local 10 m Sentinel-2 obstacles (Forest / Built-up) and compute 72-sector mask.

    Returns ``(obstacle_mask_72, obstacle_class_map)`` where class codes are:
    ``0`` = Open ground, ``1`` = Forest / Tall vegetation (``NDVI >= 0.50``),
    ``2`` = Buildings / Built-up (``NDBI > 0.05`` & ``NDVI < 0.20``).
    """
    mask_deg = np.zeros(N_AZ_SECTORS, dtype=np.float64)
    ndvi = np.asarray(ndvi_arr, dtype=np.float32)
    ndbi = np.asarray(ndbi_arr, dtype=np.float32)
    xv = np.asarray(x_coords, dtype=np.float64)
    yv = np.asarray(y_coords, dtype=np.float64)

    obs_cls = np.zeros(ndvi.shape, dtype=np.uint8)
    valid_px = np.isfinite(ndvi) & np.isfinite(ndbi)
    is_forest = valid_px & (ndvi >= 0.50)
    is_building = valid_px & (ndbi > 0.05) & (ndvi < 0.20)
    obs_cls[is_forest] = 1
    obs_cls[is_building] = 2

    obs_h = np.zeros(ndvi.shape, dtype=np.float64)
    obs_h[is_forest] = float(canopy_h_m)
    obs_h[is_building] = float(building_h_m)

    xx, yy = np.meshgrid(xv, yv)
    de = xx - float(station_e)
    dn = yy - float(station_n)
    dist = np.hypot(de, dn)

    active = (obs_h > float(ant_height_m)) & (dist >= float(min_dist_m)) & (dist <= float(max_dist_m))
    if np.any(active):
        dh = obs_h[active] - float(ant_height_m)
        el_deg = np.degrees(np.arctan2(dh, dist[active]))
        az_deg = (np.degrees(np.arctan2(de[active], dn[active])) + 360.0) % 360.0
        sectors = (az_deg // SECTOR_WIDTH_DEG).astype(np.int32) % N_AZ_SECTORS
        pos = el_deg > 0.0
        if np.any(pos):
            np.maximum.at(mask_deg, sectors[pos], el_deg[pos])

    return np.clip(mask_deg, 0.0, 89.0), obs_cls


def fetch_copernicus_dem_mask(
    lat_deg: float,
    lon_deg: float,
    *,
    ant_height_m: float = 2.0,
    radius_m: float = 2500.0,
    target_epsg: str = CRS_D96TM,
) -> tuple[np.ndarray, float, bool]:
    """Stream Copernicus DEM 30 m (`cop-dem-glo-30`) around station and compute 360° terrain mask."""
    zeros = np.zeros(N_AZ_SECTORS, dtype=np.float64)
    try:
        bbox_4326, _ = bbox_from_point_wgs84(lat_deg, lon_deg, radius_m)
        sta_e, sta_n = transform_coords(lon_deg, lat_deg, EPSG_WGS84, target_epsg)

        catalog = pystac_client.Client.open(
            STAC_API_URL,
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox_4326,
            max_items=8,
        )
        items = list(search.items())
        if not items:
            return zeros, 300.0, False

        ds = odc.stac.load(
            items,
            bands=["data"],
            bbox=bbox_4326,
            crs=target_epsg,
            resolution=30,
            groupby="solar_day",
        )
        da = ds["data"].isel(time=0).compute().astype(np.float32)
        da = da.where(da > -500.0)
        mask_deg, h_sta = compute_horizon_mask_from_dem_array(
            da.values,
            da.coords["x"].values,
            da.coords["y"].values,
            sta_e,
            sta_n,
            ant_height_m=ant_height_m,
            min_dist_m=30.0,
            max_dist_m=radius_m,
        )
        del ds, da
        gc.collect()
        return mask_deg, h_sta, True
    except Exception:
        return zeros, 300.0, False


def fetch_sentinel2_obstacle_mask(
    lat_deg: float,
    lon_deg: float,
    *,
    ant_height_m: float = 2.0,
    canopy_h_m: float = 18.0,
    building_h_m: float = 12.0,
    radius_m: float = 200.0,
    target_epsg: str = CRS_D96TM,
) -> dict[str, Any]:
    """Stream the clearest 10 m Sentinel-2 L2A scene (last 12 months, <=35% cloud, 200 m radius).

    Runs completely independently of any sidebar cloud/date filters and logs exact
    exceptions instead of silently hiding them.
    """
    import logging

    logger = logging.getLogger(__name__)
    zeros = np.zeros(N_AZ_SECTORS, dtype=np.float64)
    sta_e, sta_n = transform_coords(lon_deg, lat_deg, EPSG_WGS84, target_epsg)
    fallback: dict[str, Any] = {
        "obstacle_mask_deg": zeros,
        "ok": False,
        "scene_date": None,
        "rgb_u8": None,
        "obs_cls": None,
        "x_coords": None,
        "y_coords": None,
        "station_e": sta_e,
        "station_n": sta_n,
        "error": None,
    }
    try:
        bbox_4326, _ = bbox_from_point_wgs84(lat_deg, lon_deg, float(radius_m))
        today = date.today()
        # Independent 12-month search, max_cloud=35%, no strict large-AOI swath rejection
        items, df = search_scenes(
            sensor=SENSOR_S2,
            bbox_4326=bbox_4326,
            start_date=today - timedelta(days=365),
            end_date=today,
            max_cloud_cover=35.0,
            min_coverage_pct=0.0,
        )
        if not items or df.empty:
            fallback["error"] = "No Sentinel-2 L2A scenes found in the last 12 months (<=35% cloud)."
            logger.warning("GNSS Sentinel-2 obstacle search returned no scenes for (%.5f, %.5f)", lat_deg, lon_deg)
            return fallback

        # Order candidates by lowest cloud cover first (clearest scene over the 200 m station clip)
        cc_vals = df["cloud_cover"].fillna(50.0).to_numpy(dtype=np.float64)
        cand_order = np.argsort(cc_vals)[:3]

        last_err: str | None = None
        for idx in cand_order:
            best_group = items[int(idx)]
            scene_date = str(df.iloc[int(idx)]["date_str"])
            try:
                ds = load_scene(
                    sensor=SENSOR_S2,
                    item=best_group,
                    bbox_4326=bbox_4326,
                    bands=["B04", "B03", "B02", "B08", "B11", "SCL"],
                    target_epsg=target_epsg,
                    resolution=10,
                )
                ds_m, clear_pct = apply_cloud_mask(ds, mask_clouds=True, sensor=SENSOR_S2)
                # If this scene is locally clouded right on top of the 200 m station, try next clearest
                if clear_pct < 15.0 and len(cand_order) > 1 and int(idx) != int(cand_order[-1]):
                    del ds, ds_m
                    continue

                ndvi_da = compute_index(ds_m, "NDVI")
                ndbi_da = compute_index(ds_m, "NDBI")

                x_vals = np.asarray(ndvi_da.coords["x"].values, dtype=np.float64)
                y_vals = np.asarray(ndvi_da.coords["y"].values, dtype=np.float64)
                obs_mask, obs_cls = compute_obstacle_mask_from_s2_indices(
                    ndvi_da.values,
                    ndbi_da.values,
                    x_vals,
                    y_vals,
                    sta_e,
                    sta_n,
                    ant_height_m=ant_height_m,
                    canopy_h_m=canopy_h_m,
                    building_h_m=building_h_m,
                    min_dist_m=10.0,
                    max_dist_m=float(radius_m),
                )

                # Build crisp RGB preview array for Chart 3 (Local Obstacles Overview)
                ch_list: list[np.ndarray] = []
                for b_name in ("B04", "B03", "B02"):
                    arr = np.asarray(ds[b_name].values, dtype=np.float32)
                    fin = arr[np.isfinite(arr)]
                    lo = float(np.percentile(fin, 2.0)) if fin.size else 0.0
                    hi = float(np.percentile(fin, 98.0)) if fin.size else 0.3
                    if hi <= lo + 1e-6:
                        hi = lo + 0.3
                    norm = np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
                    ch_list.append(np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8))
                rgb_u8 = np.stack(ch_list, axis=-1)

                del ds, ds_m, ndvi_da, ndbi_da
                gc.collect()

                return {
                    "obstacle_mask_deg": obs_mask,
                    "ok": True,
                    "scene_date": scene_date,
                    "rgb_u8": rgb_u8,
                    "obs_cls": obs_cls,
                    "x_coords": x_vals,
                    "y_coords": y_vals,
                    "station_e": sta_e,
                    "station_n": sta_n,
                    "error": None,
                }
            except Exception as inner_exc:
                last_err = str(inner_exc)
                logger.warning("GNSS Sentinel-2 load failed for %s: %s", scene_date, inner_exc)
                continue

        fallback["error"] = last_err or "Failed to load Sentinel-2 L2A clip."
        return fallback
    except Exception as exc:
        logger.exception("GNSS Sentinel-2 obstacle loader failed: %s", exc)
        fallback["error"] = str(exc)
        return fallback


# ---------------------------------------------------------------------------
# 3. Strict Geodetic ENU DOP Computation & Optimal Measurement Windows
# ---------------------------------------------------------------------------
def compute_dop_from_az_el(
    az_deg_list: Sequence[float],
    el_deg_list: Sequence[float],
) -> tuple[float, float, float, float]:
    """Compute exact ``(HDOP, VDOP, PDOP, GDOP)`` from topocentric ENU design matrix ``H``."""
    n = len(az_deg_list)
    if n < 4:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    az = np.radians(np.asarray(az_deg_list, dtype=np.float64))
    el = np.radians(np.asarray(el_deg_list, dtype=np.float64))
    cos_el = np.cos(el)

    h_mat = np.column_stack(
        [
            cos_el * np.sin(az),  # East
            cos_el * np.cos(az),  # North
            np.sin(el),           # Up
            np.ones(n, dtype=np.float64),  # Receiver clock bias
        ]
    )
    normal = h_mat.T @ h_mat
    try:
        if np.linalg.cond(normal) > 1e10:
            return (float("nan"), float("nan"), float("nan"), float("nan"))
        q_mat = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    q_ee, q_nn, q_uu, q_tt = float(q_mat[0, 0]), float(q_mat[1, 1]), float(q_mat[2, 2]), float(q_mat[3, 3])
    if q_ee < 0.0 or q_nn < 0.0 or q_uu < 0.0 or q_tt < 0.0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    hdop = math.sqrt(q_ee + q_nn)
    vdop = math.sqrt(q_uu)
    pdop = math.sqrt(q_ee + q_nn + q_uu)
    gdop = math.sqrt(q_ee + q_nn + q_uu + q_tt)

    if pdop > 50.0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    return (round(hdop, 2), round(vdop, 2), round(pdop, 2), round(gdop, 2))


def find_optimal_windows(
    timeline_df: pd.DataFrame,
    *,
    max_pdop: float = 2.5,
    min_sats: int = 6,
    step_minutes: int = 15,
) -> list[str]:
    """Identify contiguous optimal GNSS measurement time windows (PDOP <= 2.5 & visible >= 6)."""
    if timeline_df.empty:
        return []

    ok_mask = (
        timeline_df["pdop_real"].notna()
        & (timeline_df["pdop_real"] <= float(max_pdop))
        & (timeline_df["visible_real"] >= int(min_sats))
    ).to_numpy()

    times = timeline_df["time_str"].tolist()
    windows: list[str] = []
    start_i: int | None = None

    def _end_time_str(idx: int) -> str:
        t_parts = times[idx].split(":")
        mins = int(t_parts[0]) * 60 + int(t_parts[1]) + int(step_minutes)
        if mins >= 1440:
            return "24:00"
        return f"{mins // 60:02d}:{mins % 60:02d}"

    for i, flag in enumerate(ok_mask):
        if flag and start_i is None:
            start_i = i
        elif not flag and start_i is not None:
            windows.append(f"{times[start_i]}–{_end_time_str(i - 1)}")
            start_i = None
    if start_i is not None:
        windows.append(f"{times[start_i]}–{_end_time_str(len(ok_mask) - 1)}")

    return windows


# ---------------------------------------------------------------------------
# 4. Complete 24-Hour GNSS Mission Planning Pipeline
# ---------------------------------------------------------------------------
def run_gnss_mission_plan(
    *,
    lat_deg: float,
    lon_deg: float,
    plan_date: date,
    constellations: Sequence[str] = ("GPS", "Galileo", "GLONASS"),
    cutoff_deg: float = 10.0,
    ant_height_m: float = 2.0,
    canopy_h_m: float = 18.0,
    building_h_m: float = 12.0,
    use_dem: bool = True,
    use_s2_obstacles: bool = True,
    use_live_tle: bool = True,
    tz_name: str = "Europe/Ljubljana",
    step_minutes: int = 15,
    target_epsg: str = CRS_D96TM,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run complete 24-hour geodetic GNSS mission planning with DEM + Sentinel-2 horizon masking."""

    def _prog(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(frac, msg)

    sel_const = [c for c in constellations if c in CONSTELLATIONS] or ["GPS", "Galileo"]
    sta_e, sta_n = transform_coords(lon_deg, lat_deg, EPSG_WGS84, EPSG_D96TM)

    # 1. Horizon Obstruction Mask (Terrain DEM + Local Sentinel-2 Obstacles)
    terrain_mask = np.zeros(N_AZ_SECTORS, dtype=np.float64)
    h_station_m = 300.0
    dem_ok = False
    if use_dem:
        _prog(0.18, "Prenašam Copernicus DEM 30 m (2,5 km okolica) za terenske ovire...")
        terrain_mask, h_station_m, dem_ok = fetch_copernicus_dem_mask(
            lat_deg,
            lon_deg,
            ant_height_m=ant_height_m,
            radius_m=2500.0,
            target_epsg=target_epsg,
        )

    s2_info: dict[str, Any] = {
        "obstacle_mask_deg": np.zeros(N_AZ_SECTORS, dtype=np.float64),
        "ok": False,
        "scene_date": None,
        "rgb_u8": None,
        "obs_cls": None,
        "x_coords": None,
        "y_coords": None,
        "station_e": sta_e,
        "station_n": sta_n,
    }
    if use_s2_obstacles:
        _prog(0.42, "Analiziram zadnji Sentinel-2 (10 m) za gozd in stavbe (200 m okolica)...")
        s2_info = fetch_sentinel2_obstacle_mask(
            lat_deg,
            lon_deg,
            ant_height_m=ant_height_m,
            canopy_h_m=canopy_h_m,
            building_h_m=building_h_m,
            radius_m=200.0,
            target_epsg=target_epsg,
        )

    obstacle_mask = np.asarray(s2_info["obstacle_mask_deg"], dtype=np.float64)
    cutoff_arr = np.full(N_AZ_SECTORS, float(cutoff_deg), dtype=np.float64)
    total_mask = np.maximum(cutoff_arr, np.maximum(terrain_mask, obstacle_mask))

    # Solid-angle sky blockage fraction above 0° horizon (exact sin(el) hemisphere integral)
    sky_blocked_pct = float(100.0 * np.mean(np.sin(np.radians(total_mask))))
    obst_only_pct = float(
        100.0 * np.mean(np.sin(np.radians(np.maximum(terrain_mask, obstacle_mask))))
    )

    # 2. Build 24-hour epoch grid in local CET/CEST or UTC
    _prog(0.65, "Računam 24-urne orbite GNSS satelitov (Azimut / Elevacija)...")
    try:
        tz = timezone.utc if tz_name == "UTC" else ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    n_steps = max(24, int(1440 // max(int(step_minutes), 5)))
    start_local = datetime(plan_date.year, plan_date.month, plan_date.day, 0, 0, tzinfo=tz)
    epochs_local = [start_local + timedelta(minutes=i * int(step_minutes)) for i in range(n_steps)]
    epochs_utc = [dt.astimezone(timezone.utc) for dt in epochs_local]
    time_labels = [dt.strftime("%H:%M") for dt in epochs_local]

    epochs_sats, orbit_source = propagate_gnss_constellations(
        sel_const,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        h_m=h_station_m + float(ant_height_m),
        epochs_utc=epochs_utc,
        use_live_tle=use_live_tle,
    )

    # 3. Evaluate Real Obstructed Sky vs. Open Sky & Exact ENU DOP for every epoch
    _prog(0.86, "Računam matriko zasnove ENU in vrednosti PDOP / HDOP / VDOP...")
    rows: list[dict[str, Any]] = []
    epoch_sat_states: list[list[dict[str, Any]]] = []

    for idx, sat_list in enumerate(epochs_sats):
        open_az: list[float] = []
        open_el: list[float] = []
        real_az: list[float] = []
        real_el: list[float] = []
        sat_states: list[dict[str, Any]] = []

        for sat in sat_list:
            az_v = float(sat["az_deg"])
            el_v = float(sat["el_deg"])
            sec = int(az_v // SECTOR_WIDTH_DEG) % N_AZ_SECTORS
            mask_el = float(total_mask[sec])

            is_above_cutoff = el_v >= float(cutoff_deg)
            is_visible_real = el_v >= mask_el

            if is_above_cutoff:
                open_az.append(az_v)
                open_el.append(el_v)
            if is_visible_real:
                real_az.append(az_v)
                real_el.append(el_v)

            sat_states.append(
                {
                    "prn": sat["prn"],
                    "constellation": sat["constellation"],
                    "az_deg": az_v,
                    "el_deg": el_v,
                    "visible_real": is_visible_real,
                    "blocked_by_obstacle": is_above_cutoff and (not is_visible_real),
                }
            )

        hdop_o, vdop_o, pdop_o, gdop_o = compute_dop_from_az_el(open_az, open_el)
        hdop_r, vdop_r, pdop_r, gdop_r = compute_dop_from_az_el(real_az, real_el)

        rows.append(
            {
                "time_str": time_labels[idx],
                "utc_str": epochs_utc[idx].strftime("%H:%M"),
                "visible_open": len(open_az),
                "visible_real": len(real_az),
                "blocked_count": max(0, len(open_az) - len(real_az)),
                "hdop_open": hdop_o,
                "vdop_open": vdop_o,
                "pdop_open": pdop_o,
                "gdop_open": gdop_o,
                "hdop_real": hdop_r,
                "vdop_real": vdop_r,
                "pdop_real": pdop_r,
                "gdop_real": gdop_r,
            }
        )
        epoch_sat_states.append(sat_states)

    timeline_df = pd.DataFrame(rows)
    optimal_windows = find_optimal_windows(
        timeline_df, max_pdop=2.5, min_sats=6, step_minutes=step_minutes
    )

    valid_pdop = timeline_df["pdop_real"].dropna()
    summary = {
        "avg_vis_real": round(float(timeline_df["visible_real"].mean()), 1),
        "avg_vis_open": round(float(timeline_df["visible_open"].mean()), 1),
        "min_pdop_real": round(float(valid_pdop.min()), 2) if not valid_pdop.empty else float("nan"),
        "mean_pdop_real": round(float(valid_pdop.mean()), 2) if not valid_pdop.empty else float("nan"),
        "sky_blocked_pct": round(sky_blocked_pct, 1),
        "obstacle_blocked_pct": round(obst_only_pct, 1),
        "optimal_windows": optimal_windows,
        "station_elev_m": round(float(h_station_m), 1),
    }

    csv_bytes = timeline_df.to_csv(index=False).encode("utf-8")
    _prog(1.0, "GNSS načrt izmere pripravljen.")

    return {
        "lat": float(lat_deg),
        "lon": float(lon_deg),
        "e_3794": round(float(sta_e), 1),
        "n_3794": round(float(sta_n), 1),
        "plan_date": plan_date.isoformat(),
        "tz_name": tz_name,
        "constellations": sel_const,
        "cutoff_deg": float(cutoff_deg),
        "ant_height_m": float(ant_height_m),
        "orbit_source": orbit_source,
        "dem_ok": dem_ok,
        "s2_info": s2_info,
        "az_centers_deg": AZ_SECTOR_CENTERS_DEG,
        "terrain_mask_deg": terrain_mask,
        "obstacle_mask_deg": obstacle_mask,
        "total_mask_deg": total_mask,
        "timeline_df": timeline_df,
        "epoch_sat_states": epoch_sat_states,
        "summary": summary,
        "csv_bytes": csv_bytes,
        "csv_filename": f"GNSS_Plan_{plan_date.isoformat()}_E{int(round(sta_e))}_N{int(round(sta_n))}.csv",
    }
