"""Geodetically correct satellite slippy-map overlay engine (Web Visualization Layer).

Strictly separates geodetic computation/export (EPSG:3794 D96/TM & EPSG:32633 UTM 33N)
from web slippy-map visualization (EPSG:4326 / EPSG:3857):
- ``expand_bbox_for_panning`` expands the download bounding box by a RAM-safe buffer
  (default +35% on each side) so the user can smoothly pan around their AOI while
  keeping all statistics, histograms, and GeoTIFF exports locked to the exact original AOI.
- ``raster_to_map_overlay_png`` reprojects ONLY a lightweight visualization copy to
  EPSG:4326 with transparent alpha (`alpha = 0`) for NaN / cloud-masked / collar pixels
  and returns a compact PNG data URI plus exact Leaflet corner bounds.
- ``build_satellite_slippy_map`` builds a full-width Folium slippy map with persistent
  basemaps, QGIS-style LayerControl, MousePosition readout, and exact AOI boundary overlay.
"""

from __future__ import annotations

import base64
import html as _html
import io
import json
import math
from typing import Any, Sequence

import folium
from folium.plugins import MeasureControl, MousePosition
import matplotlib
import numpy as np
from PIL import Image
from rasterio.enums import Resampling
import rioxarray  # noqa: F401 - registers .rio accessor on xarray objects
import xarray as xr

from geodezija_crs import EPSG_D96TM, EPSG_WGS84, get_best_metric_crs, get_ram_limits, transform_coords, validate_aoi_size

# Persistent cartographic basemap definitions (mirrors app.py)
SLIPPY_BASEMAP_SPECS: dict[str, dict[str, Any]] = {
    "satellite": {
        "name_sl": "🛰️ Satelitska podlaga (Esri)",
        "name_en": "🛰️ Satellite Imagery (Esri)",
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri World Imagery",
        "max_zoom": 19,
    },
    "topo": {
        "name_sl": "🗺️ Topografska (OpenTopoMap)",
        "name_en": "🗺️ Topographic (OpenTopoMap)",
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)",
        "max_zoom": 17,
    },
    "osm": {
        "name_sl": "🛣️ Osnovna karta (OpenStreetMap)",
        "name_en": "🛣️ Street Map (OpenStreetMap)",
        "tiles": "OpenStreetMap",
        "attr": None,
        "max_zoom": 19,
    },
}


def expand_bbox_for_panning(
    bbox_4326: Sequence[float],
    buffer_factor: float = 0.35,
    max_area_km2: float | None = None,
) -> list[float]:
    """Expand an EPSG:4326 AOI bounding box on each side for smooth slippy-map panning.

    Caps the effective buffer automatically against the runtime's RAM limit and skips
    expansion when the core AOI already occupies >= 75% of the RAM cap so expanded
    buffers never pull in half-empty swath edges over large AOIs.
    """
    if len(bbox_4326) != 4:
        raise ValueError(f"bbox_4326 must contain 4 values, got {len(bbox_4326)}.")

    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox_4326)
    if min_lon >= max_lon or min_lat >= max_lat:
        return [min_lon, min_lat, max_lon, max_lat]

    factor = max(0.0, float(buffer_factor))
    if factor == 0.0:
        return [min_lon, min_lat, max_lon, max_lat]

    runtime_max = float(get_ram_limits()["max_area_km2"])
    eff_max = min(float(max_area_km2), runtime_max) if max_area_km2 is not None else runtime_max

    cur_area, _, _ = validate_aoi_size([min_lon, min_lat, max_lon, max_lat], max_area_km2=eff_max)
    safe_cap = eff_max * 0.95
    # Avoid pulling in distant partial-swath edges when core AOI is already large
    if cur_area <= 0.0 or cur_area >= eff_max * 0.75:
        return [min_lon, min_lat, max_lon, max_lat]

    # Expanding each side by f scales width & height by (1 + 2f) and area by (1 + 2f)^2
    max_linear_scale = math.sqrt(safe_cap / cur_area)
    max_safe_factor = max(0.0, 0.5 * (max_linear_scale - 1.0))
    eff_factor = min(factor, max_safe_factor)
    if eff_factor <= 0.01:
        return [min_lon, min_lat, max_lon, max_lat]

    width = max_lon - min_lon
    height = max_lat - min_lat

    for _ in range(4):
        cand = [
            max(-180.0, min_lon - width * eff_factor),
            max(-85.0, min_lat - height * eff_factor),
            min(180.0, max_lon + width * eff_factor),
            min(85.0, max_lat + height * eff_factor),
        ]
        cand_area, _, _ = validate_aoi_size(cand, max_area_km2=max_area_km2)
        if cand_area <= float(max_area_km2):
            return cand
        eff_factor *= 0.8

    return [min_lon, min_lat, max_lon, max_lat]


def _estimate_zoom_for_bbox(bbox_4326: Sequence[float]) -> int:
    """Estimate a comfortable Leaflet zoom level fitting an EPSG:4326 bounding box."""
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox_4326)
    span = max(abs(max_lon - min_lon), abs(max_lat - min_lat), 1e-4)
    if span <= 0.015:
        return 14
    if span <= 0.04:
        return 13
    if span <= 0.09:
        return 12
    if span <= 0.22:
        return 11
    if span <= 0.45:
        return 10
    return 9


def _stretch_channel_01(
    arr: np.ndarray,
    use_percentile_stretch: bool = True,
    fixed_range: tuple[float, float] = (0.0, 0.35),
    gamma: float = 1.0,
) -> np.ndarray:
    """Normalize a float32 channel into [0, 1] using 2–98% stretch or fixed bounds + gamma."""
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
    if gamma != 1.0 and gamma > 0.0:
        stretched = np.power(np.clip(stretched, 0.0, 1.0), np.float32(1.0 / gamma))
    return stretched.astype(np.float32, copy=False)


def _prepare_vis_dataarray(
    da: xr.DataArray,
    src_epsg: int | str,
    resampling: Resampling,
    nodata: float | int,
) -> xr.DataArray:
    """Ensure CRS/spatial dims and reproject ONLY a visualization copy to EPSG:4326."""
    work = da
    if "time" in work.dims and work.sizes["time"] == 1:
        work = work.isel(time=0, drop=True)

    dims = set(work.dims)
    if "x" in dims and "y" in dims:
        work = work.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    elif "longitude" in dims and "latitude" in dims:
        work = work.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=False)

    if work.rio.crs is None:
        crs_str = f"EPSG:{src_epsg}" if isinstance(src_epsg, int) else str(src_epsg)
        work = work.rio.write_crs(crs_str, inplace=False)

    work = work.rio.write_nodata(nodata, inplace=False)
    reproj = work.rio.reproject("EPSG:4326", resampling=resampling, nodata=nodata)
    return reproj


def _orient_north_up(arr: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """Ensure row 0 is North (max latitude) so ImageOverlay matches Leaflet bounds."""
    if y_coords.size > 1 and float(y_coords[0]) < float(y_coords[-1]):
        return arr[::-1, ...]
    return arr


def _encode_rgba_to_data_uri(
    rgba: np.ndarray,
    max_width: int = 1800,
    discrete: bool = False,
) -> str:
    """Encode an ``(H, W, 4) uint8`` RGBA array to a compact PNG base64 data URI."""
    img = Image.fromarray(rgba, mode="RGBA")
    if max_width > 0 and img.width > int(max_width):
        scale = float(max_width) / float(img.width)
        new_h = max(1, int(round(img.height * scale)))
        resamp = Image.Resampling.NEAREST if discrete else Image.Resampling.LANCZOS
        img = img.resize((int(max_width), new_h), resample=resamp)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def raster_to_map_overlay_png(
    data_array_or_rgb: xr.DataArray | xr.Dataset,
    src_epsg: int | str = "EPSG:3794",
    is_rgb: bool = True,
    cmap_name: str = "RdYlGn",
    vmin: float = -1.0,
    vmax: float = 1.0,
    *,
    bands: Sequence[str] | None = None,
    use_percentile_stretch: bool = True,
    gamma: float = 1.0,
    fixed_ranges: Sequence[tuple[float, float]] | None = None,
    max_width: int = 1800,
    discrete_rgba_map: dict[int, tuple[int, int, int, int]] | None = None,
) -> tuple[str, list[list[float]]]:
    """Reproject a visualization copy of a geodetic raster to EPSG:4326 and encode as RGBA PNG.

    NaN / cloud-masked / reprojection-collar pixels receive ``alpha = 0`` (fully transparent)
    so underlying basemaps shine cleanly around irregular edges or masked clouds.

    Returns
    -------
    tuple[str, list[list[float]]]
        ``(png_data_uri, [[south, west], [north, east]])`` ready for ``folium.raster_layers.ImageOverlay``.
    """
    if is_rgb:
        is_sar_fc = False
        # Build a 3-band float32 DataArray (band, y, x) in source CRS before a single clean reprojection
        if isinstance(data_array_or_rgb, xr.Dataset):
            ds = data_array_or_rgb
            req_bands = list(bands) if bands else list(ds.data_vars)[:3]

            # Special case: Sentinel-1 2-pol false color composite (VV, VH -> R=VV, G=VH, B=VV-VH)
            if len(req_bands) == 2 and set(req_bands) == {"VV", "VH"}:
                is_sar_fc = True
                vv = ds["VV"].astype(np.float32, copy=False)
                vh = ds["VH"].astype(np.float32, copy=False)
                diff = (vv - vh).astype(np.float32, copy=False)
                stacked = xr.concat([vv, vh, diff], dim="band")
                if fixed_ranges is None:
                    fixed_ranges = [(-22.0, -2.0), (-28.0, -8.0), (2.0, 14.0)]
            elif len(req_bands) >= 3:
                ch_list = [ds[b].astype(np.float32, copy=False) for b in req_bands[:3]]
                stacked = xr.concat(ch_list, dim="band")
            else:
                raise ValueError(f"RGB overlay requires 3 bands (or SAR VV+VH), got {req_bands!r}.")

            if ds.rio.crs is not None:
                stacked = stacked.rio.write_crs(ds.rio.crs, inplace=False)
        elif isinstance(data_array_or_rgb, xr.DataArray):
            stacked = data_array_or_rgb.astype(np.float32, copy=False)
            if stacked.ndim != 3:
                raise ValueError(f"RGB DataArray must be 3D (band, y, x), got shape {stacked.shape}.")
            if stacked.shape[0] != 3 and stacked.shape[-1] == 3:
                dims_3 = list(stacked.dims)
                stacked = stacked.transpose(dims_3[2], dims_3[0], dims_3[1])
        else:
            raise TypeError(f"Expected xarray DataArray or Dataset, got {type(data_array_or_rgb).__name__}.")

        # Mask optical all-zero NoData pixels to NaN BEFORE reprojection so bilinear interpolation
        # never bleeds black (0,0,0) swath borders into adjacent valid pixels.
        if not is_sar_fc:
            src_vals = stacked.values.astype(np.float32, copy=True)
            src_valid = (
                np.isfinite(src_vals[0])
                & np.isfinite(src_vals[1])
                & np.isfinite(src_vals[2])
                & ~((src_vals[0] <= 0.0) & (src_vals[1] <= 0.0) & (src_vals[2] <= 0.0))
            )
            src_vals[:, ~src_valid] = np.nan
            src_crs_saved = stacked.rio.crs
            stacked = xr.DataArray(src_vals, dims=stacked.dims, coords=stacked.coords, attrs=dict(stacked.attrs))
            if src_crs_saved is not None:
                stacked = stacked.rio.write_crs(src_crs_saved, inplace=False)

        reproj = _prepare_vis_dataarray(stacked, src_epsg=src_epsg, resampling=Resampling.bilinear, nodata=np.nan)
        west, south, east, north = reproj.rio.bounds()
        y_coords = np.asarray(reproj.coords["y"].values, dtype=np.float64)

        arr3 = reproj.values.astype(np.float32, copy=False)  # (3, H, W)
        valid_mask = np.isfinite(arr3[0]) & np.isfinite(arr3[1]) & np.isfinite(arr3[2])
        if not is_sar_fc:
            valid_mask &= ~((arr3[0] <= 0.0) & (arr3[1] <= 0.0) & (arr3[2] <= 0.0))

        h, w = arr3.shape[1], arr3.shape[2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for idx in range(3):
            f_range = (
                fixed_ranges[idx]
                if (fixed_ranges is not None and idx < len(fixed_ranges))
                else (0.0, 0.35)
            )
            ch_valid_only = np.where(valid_mask, arr3[idx], np.nan)
            stretched = _stretch_channel_01(
                ch_valid_only,
                use_percentile_stretch=use_percentile_stretch,
                fixed_range=f_range,
                gamma=gamma,
            )
            rgba[..., idx] = np.where(
                valid_mask,
                np.clip(np.round(stretched * 255.0), 0, 255),
                0,
            ).astype(np.uint8)
        rgba[..., 3] = np.where(valid_mask, 255, 0).astype(np.uint8)
        rgba = _orient_north_up(rgba, y_coords)
        data_uri = _encode_rgba_to_data_uri(rgba, max_width=max_width, discrete=False)
        return data_uri, [[float(south), float(west)], [float(north), float(east)]]

    # Single-band visualization (Spectral Index, SAR dB, or Discrete SAR Water Mask)
    if isinstance(data_array_or_rgb, xr.Dataset):
        var_name = bands[0] if bands else next(iter(data_array_or_rgb.data_vars))
        da_in = data_array_or_rgb[var_name]
        if da_in.rio.crs is None and data_array_or_rgb.rio.crs is not None:
            da_in = da_in.rio.write_crs(data_array_or_rgb.rio.crs, inplace=False)
    elif isinstance(data_array_or_rgb, xr.DataArray):
        da_in = data_array_or_rgb
    else:
        raise TypeError(f"Expected xarray DataArray or Dataset, got {type(data_array_or_rgb).__name__}.")

    if discrete_rgba_map is not None:
        da_u8 = da_in.astype(np.uint8, copy=False)
        reproj = _prepare_vis_dataarray(da_u8, src_epsg=src_epsg, resampling=Resampling.nearest, nodata=255)
        west, south, east, north = reproj.rio.bounds()
        y_coords = np.asarray(reproj.coords["y"].values, dtype=np.float64)
        vals = reproj.values.astype(np.uint8, copy=False)

        rgba = np.zeros((vals.shape[0], vals.shape[1], 4), dtype=np.uint8)
        for cls_val, col_rgba in discrete_rgba_map.items():
            mask = vals == np.uint8(cls_val)
            if np.any(mask):
                rgba[mask] = np.array(col_rgba, dtype=np.uint8)
        rgba = _orient_north_up(rgba, y_coords)
        data_uri = _encode_rgba_to_data_uri(rgba, max_width=max_width, discrete=True)
        return data_uri, [[float(south), float(west)], [float(north), float(east)]]

    da_f32 = da_in.astype(np.float32, copy=False)
    reproj = _prepare_vis_dataarray(da_f32, src_epsg=src_epsg, resampling=Resampling.bilinear, nodata=np.nan)
    west, south, east, north = reproj.rio.bounds()
    y_coords = np.asarray(reproj.coords["y"].values, dtype=np.float64)
    vals = reproj.values.astype(np.float32, copy=False)

    finite = np.isfinite(vals)
    denom = max(float(vmax - vmin), 1e-6)
    norm = np.clip((np.nan_to_num(vals, nan=vmin) - np.float32(vmin)) / np.float32(denom), 0.0, 1.0)

    cmap = matplotlib.colormaps[cmap_name]
    rgba_f = cmap(norm)  # (H, W, 4) in [0, 1]
    rgba = np.clip(np.round(rgba_f * 255.0), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
    rgba = _orient_north_up(rgba, y_coords)
    data_uri = _encode_rgba_to_data_uri(rgba, max_width=max_width, discrete=False)
    return data_uri, [[float(south), float(west)], [float(north), float(east)]]


def build_horizontal_colorbar_html(
    cmap_name: str,
    vmin: float,
    vmax: float,
    *,
    label: str = "",
    units: str = "",
    discrete_items: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Build a crisp cartographic horizontal legend HTML bar for the slippy map."""
    safe_label = str(label).replace("<", "&lt;").replace(">", "&gt;")
    unit_str = f" [{units}]" if units else ""

    if discrete_items:
        swatches = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;'
            f'font-size:12px;color:#0F172A;font-weight:500;">'
            f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;'
            f'background:{color};border:1px solid #94A3B8;"></span>{text}</span>'
            for text, color in discrete_items
        )
        return (
            f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;'
            f'gap:8px;padding:8px 14px;margin:4px 0 8px 0;background:#FFFFFF;border:1px solid #CBD5E1;'
            f'border-radius:8px;border-left:4px solid #0F766E;">'
            f'<span style="font-size:12px;font-weight:700;color:#0F172A;">{safe_label}</span>'
            f'<div style="display:flex;align-items:center;flex-wrap:wrap;">{swatches}</div>'
            f'</div>'
        )

    cmap = matplotlib.colormaps[cmap_name]
    stops: list[str] = []
    n_stops = 16
    for i in range(n_stops):
        frac = float(i) / float(n_stops - 1)
        r, g, b, _ = cmap(frac)
        stops.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)}) {frac * 100:.1f}%")
    gradient_css = f"linear-gradient(90deg, {', '.join(stops)})"
    vmid = 0.5 * (float(vmin) + float(vmax))
    fmt = "{:.1f}" if abs(vmax - vmin) >= 5.0 else "{:.2f}"

    return (
        f'<div style="padding:8px 14px;margin:4px 0 8px 0;background:#FFFFFF;border:1px solid #CBD5E1;'
        f'border-radius:8px;border-left:4px solid #0F766E;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">'
        f'<span style="font-size:12px;font-weight:700;color:#0F172A;">{safe_label}{unit_str}</span>'
        f'<span style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
        f'font-size:11px;color:#475569;">{fmt.format(vmin)} → {fmt.format(vmax)}{unit_str}</span>'
        f'</div>'
        f'<div style="height:12px;width:100%;border-radius:4px;background:{gradient_css};'
        f'border:1px solid #CBD5E1;"></div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:3px;'
        f'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;color:#334155;">'
        f'<span>{fmt.format(vmin)}</span>'
        f'<span>{fmt.format(vmid)}</span>'
        f'<span>{fmt.format(vmax)}</span>'
        f'</div>'
        f'</div>'
    )


def _metric_bbox_to_wgs84_ring(
    bbox_metric: Sequence[float],
    metric_crs: int | str = EPSG_D96TM,
) -> list[list[float]]:
    min_e, min_n, max_e, max_n = (float(v) for v in bbox_metric)
    corners = [
        (min_e, min_n),
        (max_e, min_n),
        (max_e, max_n),
        (min_e, max_n),
        (min_e, min_n),
    ]
    ring: list[list[float]] = []
    for e_val, n_val in corners:
        lon, lat = transform_coords(e_val, n_val, metric_crs, EPSG_WGS84)
        ring.append([lat, lon])
    return ring


def _d96tm_bbox_to_wgs84_ring(bbox_3794: Sequence[float]) -> list[list[float]]:
    return _metric_bbox_to_wgs84_ring(bbox_3794, metric_crs=EPSG_D96TM)


def build_satellite_slippy_map(
    center_lat: float,
    center_lon: float,
    zoom: int | None = None,
    basemap_mode: str = "satellite",
    overlay_layers: Sequence[dict[str, Any]] = (),
    aoi_bbox_4326: Sequence[float] | None = None,
    aoi_polygon_geojson: dict[str, Any] | None = None,
    *,
    aoi_bbox_3794: Sequence[float] | None = None,
    aoi_metric_crs: int | str | None = None,
    aoi_outline_label: str = "AOI (meja analize)",
    transect_latlon: tuple[tuple[float, float], tuple[float, float]] | None = None,
    lang: str = "sl",
) -> folium.Map:
    """Build a full-width interactive Folium slippy map with satellite/index overlays & QGIS LayerControl.

    Parameters
    ----------
    overlay_layers : Sequence[dict[str, Any]]
        Each dict contains ``name``, ``image`` (PNG data URI), ``bounds`` (``[[S, W], [N, E]]``),
        and optional ``opacity`` (0..1), ``show`` (bool), ``zindex`` (int).
    """
    active_bm = basemap_mode if basemap_mode in SLIPPY_BASEMAP_SPECS else "satellite"
    name_key = "name_en" if lang == "en" else "name_sl"

    if zoom is None:
        if aoi_bbox_4326 is not None and len(aoi_bbox_4326) == 4:
            zoom_val = _estimate_zoom_for_bbox(aoi_bbox_4326)
        else:
            zoom_val = 12
    else:
        zoom_val = int(zoom)

    fmap = folium.Map(
        location=[float(center_lat), float(center_lon)],
        zoom_start=zoom_val,
        tiles=None,
        control_scale=True,
        max_zoom=19,
    )

    # Add persistent basemap as active base layer plus alternatives inside LayerControl
    ordered_bm_keys = [active_bm] + [k for k in ("satellite", "topo", "osm") if k != active_bm]
    for bm_key in ordered_bm_keys:
        spec = SLIPPY_BASEMAP_SPECS[bm_key]
        tile_kw: dict[str, Any] = {
            "tiles": spec["tiles"],
            "name": spec[name_key],
            "overlay": False,
            "control": True,
            "show": (bm_key == active_bm),
            "max_zoom": int(spec["max_zoom"]),
        }
        if spec["attr"]:
            tile_kw["attr"] = spec["attr"]
        folium.TileLayer(**tile_kw).add_to(fmap)

    # Add satellite / spectral index / SAR ImageOverlay layers
    for idx, layer in enumerate(overlay_layers):
        img_uri = layer.get("image")
        bounds = layer.get("bounds")
        if not img_uri or not bounds:
            continue
        folium.raster_layers.ImageOverlay(
            name=str(layer.get("name", f"Layer {idx + 1}")),
            image=img_uri,
            bounds=bounds,
            opacity=float(np.clip(float(layer.get("opacity", 1.0)), 0.0, 1.0)),
            interactive=True,
            cross_origin=False,
            zindex=int(layer.get("zindex", 10 + idx)),
            show=bool(layer.get("show", True)),
        ).add_to(fmap)

    # Draw exact original AOI analysis outline on top (crisp teal dashed border, no fill)
    if aoi_polygon_geojson is not None or aoi_bbox_3794 is not None or aoi_bbox_4326 is not None:
        aoi_fg = folium.FeatureGroup(name=aoi_outline_label, show=True)
        if aoi_polygon_geojson is not None:
            folium.GeoJson(
                data={"type": "Feature", "geometry": aoi_polygon_geojson, "properties": {}},
                style_function=lambda _f: {
                    "color": "#0F766E",
                    "weight": 2.8,
                    "dashArray": "6, 4",
                    "fill": False,
                    "fillOpacity": 0.0,
                },
                tooltip=aoi_outline_label,
            ).add_to(aoi_fg)
        elif aoi_bbox_3794 is not None and len(aoi_bbox_3794) == 4:
            if aoi_metric_crs is not None:
                eff_ring_crs: int | str = aoi_metric_crs
            else:
                eff_ring_crs, _ = get_best_metric_crs(float(center_lat), float(center_lon), mode="auto")
            ring = _metric_bbox_to_wgs84_ring(aoi_bbox_3794, metric_crs=eff_ring_crs)
            folium.Polygon(
                locations=ring,
                color="#0F766E",
                weight=2.8,
                dash_array="6, 4",
                fill=False,
                tooltip=aoi_outline_label,
            ).add_to(aoi_fg)
        elif aoi_bbox_4326 is not None and len(aoi_bbox_4326) == 4:
            min_lon, min_lat, max_lon, max_lat = (float(v) for v in aoi_bbox_4326)
            folium.Rectangle(
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                color="#0F766E",
                weight=2.8,
                dash_array="6, 4",
                fill=False,
                tooltip=aoi_outline_label,
            ).add_to(aoi_fg)
        aoi_fg.add_to(fmap)

    if transect_latlon is not None and len(transect_latlon) == 2:
        (lat_a, lon_a), (lat_b, lon_b) = transect_latlon
        tr_fg = folium.FeatureGroup(name="📐 Profil A → B" if lang == "sl" else "📐 Profile A → B", show=True)
        folium.PolyLine(
            locations=[[float(lat_a), float(lon_a)], [float(lat_b), float(lon_b)]],
            color="#EA580C",
            weight=3.2,
            opacity=0.95,
            tooltip="A → B",
        ).add_to(tr_fg)
        for lbl_pt, (p_lat, p_lon) in (("A", (lat_a, lon_a)), ("B", (lat_b, lon_b))):
            folium.CircleMarker(
                location=[float(p_lat), float(p_lon)],
                radius=6,
                color="#0F172A",
                weight=2,
                fill=True,
                fill_color="#EA580C",
                fill_opacity=1.0,
                tooltip=f"{'Točka' if lang == 'sl' else 'Point'} {lbl_pt}",
            ).add_to(tr_fg)
        tr_fg.add_to(fmap)

    # Live cursor coordinate readout (WGS84 lat, lon)
    MousePosition(
        position="bottomleft",
        separator=" , ",
        empty_string="",
        lng_first=False,
        num_digits=5,
        prefix="WGS84 (φ, λ):",
    ).add_to(fmap)

    # Interactive distance (m / km) & area (ha / km²) measurement tool
    MeasureControl(
        position="topleft",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
        primary_area_unit="hectares",
        secondary_area_unit="sqkilometers",
    ).add_to(fmap)

    # QGIS-style always-visible layer control
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def build_zoomable_swipe_map(
    *,
    overlay_uri_t1: str,
    bounds_t1: list[list[float]],
    overlay_uri_t2: str,
    bounds_t2: list[list[float]],
    label_t1: str,
    label_t2: str,
    center_lat: float,
    center_lon: float,
    basemap_mode: str = "satellite",
    aoi_bbox_4326: Sequence[float] | None = None,
    aoi_polygon_geojson: dict[str, Any] | None = None,
    aoi_bbox_3794: Sequence[float] | None = None,
    aoi_metric_crs: int | str | None = None,
    aoi_outline_label: str = "AOI",
    height: int = 620,
    opacity: float = 1.0,
    lang: str = "sl",
) -> str:
    """Build a bulletproof, self-contained HTML5 + Leaflet 1.9.4 zoomable curtain swipe map (`t1 ↔ t2`).

    Does NOT rely on Folium ``MacroElement`` or auto-generated ``map_xxxx`` variable injection.
    Renders both ``L.imageOverlay`` rasters in dedicated panes (``t1Pane`` zIndex 450, ``t2Pane``
    zIndex 460) clipped via ``map.containerPointToLayerPoint`` on every pan/zoom/resize event,
    paired with static HTML date badges and a 44×44 px touch-friendly draggable ``↔`` handle.
    """
    t1_uri = str(overlay_uri_t1 or "")
    t2_uri = str(overlay_uri_t2 or "")
    if not t1_uri.startswith("data:image/"):
        raise ValueError("overlay_uri_t1 must be a valid base64 image data URI.")
    if not t2_uri.startswith("data:image/"):
        raise ValueError("overlay_uri_t2 must be a valid base64 image data URI.")

    def _norm_bounds(b: Sequence[Sequence[float]]) -> list[list[float]]:
        if len(b) != 2 or len(b[0]) != 2 or len(b[1]) != 2:
            raise ValueError(f"Overlay bounds must be [[south, west], [north, east]], got {b!r}.")
        return [
            [float(b[0][0]), float(b[0][1])],
            [float(b[1][0]), float(b[1][1])],
        ]

    b1 = _norm_bounds(bounds_t1)
    b2 = _norm_bounds(bounds_t2)
    op_val = float(np.clip(float(opacity), 0.0, 1.0))
    map_h = max(320, int(height))

    bm_key = basemap_mode if basemap_mode in SLIPPY_BASEMAP_SPECS else "satellite"
    name_key = "name_en" if lang == "en" else "name_sl"

    # Build real tile URLs for standalone Leaflet 1.9.4 (resolve Folium's 'OpenStreetMap' shortcut)
    basemap_defs: list[dict[str, Any]] = []
    ordered_keys = [bm_key] + [k for k in ("satellite", "topo", "osm") if k != bm_key]
    for k in ordered_keys:
        spec = SLIPPY_BASEMAP_SPECS[k]
        tile_url = str(spec["tiles"])
        attr = spec["attr"]
        if tile_url == "OpenStreetMap":
            tile_url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            attr = "© OpenStreetMap contributors"
        basemap_defs.append(
            {
                "key": k,
                "name": str(spec[name_key]),
                "url": tile_url,
                "attr": str(attr or ""),
                "maxZoom": int(spec["max_zoom"]),
                "active": (k == bm_key),
            }
        )

    # Prepare exact AOI geometry payload for Leaflet
    aoi_payload: dict[str, Any] | None = None
    aoi_bounds_rect: list[list[float]] | None = None
    if aoi_polygon_geojson is not None:
        aoi_payload = {"kind": "geojson", "data": aoi_polygon_geojson}
    elif aoi_bbox_3794 is not None and len(aoi_bbox_3794) == 4:
        if aoi_metric_crs is not None:
            eff_ring_crs: int | str = aoi_metric_crs
        else:
            eff_ring_crs, _ = get_best_metric_crs(float(center_lat), float(center_lon), mode="auto")
        ring = _metric_bbox_to_wgs84_ring(aoi_bbox_3794, metric_crs=eff_ring_crs)
        aoi_payload = {"kind": "polygon", "coords": ring}
    elif aoi_bbox_4326 is not None and len(aoi_bbox_4326) == 4:
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in aoi_bbox_4326)
        aoi_bounds_rect = [[min_lat, min_lon], [max_lat, max_lon]]
        aoi_payload = {"kind": "rectangle", "bounds": aoi_bounds_rect}

    # Initial fitBounds target: exact union of t1/t2 overlays or AOI bbox
    fit_bounds = [
        [min(b1[0][0], b2[0][0]), min(b1[0][1], b2[0][1])],
        [max(b1[1][0], b2[1][0]), max(b1[1][1], b2[1][1])],
    ]
    if aoi_bbox_4326 is not None and len(aoi_bbox_4326) == 4:
        fit_bounds = [
            [float(aoi_bbox_4326[1]), float(aoi_bbox_4326[0])],
            [float(aoi_bbox_4326[3]), float(aoi_bbox_4326[2])],
        ]

    safe_t1 = _html.escape(str(label_t1))
    safe_t2 = _html.escape(str(label_t2))
    safe_aoi_lbl = _html.escape(str(aoi_outline_label))

    # Safe JSON serialization preventing accidental </script> termination
    def _js_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")

    html_doc = f"""<!DOCTYPE html>
<html lang="{_html.escape(lang)}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
  <style>
    * {{
      box-sizing: border-box;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: {map_h}px;
      overflow: hidden;
      background: #F8FAFC;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    #swipe-wrapper {{
      position: relative;
      width: 100%;
      height: {map_h}px;
      overflow: hidden;
      border-radius: 10px;
      border: 1px solid #CBD5E1;
      background: #0F172A;
    }}
    #swipe-map {{
      width: 100%;
      height: 100%;
    }}
    .swipe-badge {{
      position: absolute;
      top: 12px;
      z-index: 1000;
      padding: 6px 12px;
      border-radius: 8px;
      color: #FFFFFF;
      font-size: 12.5px;
      font-weight: 600;
      letter-spacing: 0.01em;
      box-shadow: 0 2px 10px rgba(15, 23, 42, 0.40);
      pointer-events: none;
      max-width: 42%;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      line-height: 1.3;
    }}
    #swipe-badge-t1 {{
      left: 54px;
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid rgba(255, 255, 255, 0.25);
      border-left: 3.5px solid #14B8A6;
    }}
    #swipe-badge-t2 {{
      right: 12px;
      background: rgba(15, 118, 110, 0.94);
      border: 1px solid rgba(255, 255, 255, 0.30);
    }}
    #swipe-divider {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 3px;
      margin-left: -1.5px;
      background: #FFFFFF;
      box-shadow: 0 0 8px rgba(0, 0, 0, 0.65);
      z-index: 1001;
      cursor: ew-resize;
      touch-action: none;
      user-select: none;
      -webkit-user-select: none;
    }}
    #swipe-hitarea {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: -22px;
      width: 47px;
      cursor: ew-resize;
      touch-action: none;
      user-select: none;
      -webkit-user-select: none;
    }}
    #swipe-handle {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 44px;
      height: 44px;
      border-radius: 50%;
      background: #0F766E;
      color: #FFFFFF;
      border: 2.5px solid #FFFFFF;
      box-shadow: 0 3px 14px rgba(15, 23, 42, 0.55);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      font-weight: 700;
      line-height: 1;
      cursor: ew-resize;
      touch-action: none;
      user-select: none;
      -webkit-user-select: none;
      transition: background 0.15s ease, transform 0.15s ease;
    }}
    #swipe-handle:hover, #swipe-divider.dragging #swipe-handle {{
      background: #115E59;
      transform: translate(-50%, -50%) scale(1.06);
    }}
    #swipe-coords {{
      position: absolute;
      left: 12px;
      bottom: 34px;
      z-index: 999;
      padding: 3px 8px;
      border-radius: 5px;
      background: rgba(255, 255, 255, 0.88);
      color: #0F172A;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      border: 1px solid #CBD5E1;
      pointer-events: none;
    }}
  </style>
</head>
<body>
  <div id="swipe-wrapper">
    <div id="swipe-map"></div>
    <div id="swipe-badge-t1" class="swipe-badge">◀ {safe_t1}</div>
    <div id="swipe-badge-t2" class="swipe-badge">{safe_t2} ▶</div>
    <div id="swipe-divider" title="↔">
      <div id="swipe-hitarea"></div>
      <div id="swipe-handle">↔</div>
    </div>
    <div id="swipe-coords">WGS84 (φ, λ): {float(center_lat):.5f}, {float(center_lon):.5f}</div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  <script>
    (function () {{
      const t1Uri = {_js_json(t1_uri)};
      const t2Uri = {_js_json(t2_uri)};
      const bounds1 = {_js_json(b1)};
      const bounds2 = {_js_json(b2)};
      const fitBounds = {_js_json(fit_bounds)};
      const opacity = {op_val:.3f};
      const basemapDefs = {_js_json(basemap_defs)};
      const aoiPayload = {_js_json(aoi_payload)};
      const aoiLabel = {_js_json(safe_aoi_lbl)};
      const initCenter = [{float(center_lat):.6f}, {float(center_lon):.6f}];

      const map = L.map('swipe-map', {{
        center: initCenter,
        zoom: 12,
        zoomControl: true,
        preferCanvas: true,
        maxZoom: 19
      }});

      const baseLayers = {{}};
      basemapDefs.forEach(function (bm) {{
        const layer = L.tileLayer(bm.url, {{
          attribution: bm.attr,
          maxZoom: bm.maxZoom
        }});
        baseLayers[bm.name] = layer;
        if (bm.active) {{
          layer.addTo(map);
        }}
      }});

      const t1Pane = map.createPane('t1Pane');
      t1Pane.style.zIndex = 450;
      t1Pane.style.pointerEvents = 'none';
      t1Pane.style.position = 'absolute';

      const t2Pane = map.createPane('t2Pane');
      t2Pane.style.zIndex = 460;
      t2Pane.style.pointerEvents = 'none';
      t2Pane.style.position = 'absolute';

      const aoiPane = map.createPane('aoiPane');
      aoiPane.style.zIndex = 470;
      aoiPane.style.pointerEvents = 'none';

      L.imageOverlay(t1Uri, bounds1, {{
        pane: 't1Pane',
        opacity: opacity,
        interactive: false
      }}).addTo(map);

      L.imageOverlay(t2Uri, bounds2, {{
        pane: 't2Pane',
        opacity: opacity,
        interactive: false
      }}).addTo(map);

      const overlayLayers = {{}};
      if (aoiPayload) {{
        const aoiStyle = {{
          color: '#0F766E',
          weight: 2.5,
          dashArray: '6, 6',
          fill: false,
          fillOpacity: 0.0,
          pane: 'aoiPane',
          interactive: false
        }};
        let aoiLayer = null;
        if (aoiPayload.kind === 'geojson') {{
          aoiLayer = L.geoJSON(aoiPayload.data, {{ style: function () {{ return aoiStyle; }}, pane: 'aoiPane' }});
        }} else if (aoiPayload.kind === 'polygon') {{
          aoiLayer = L.polygon(aoiPayload.coords, aoiStyle);
        }} else if (aoiPayload.kind === 'rectangle') {{
          aoiLayer = L.rectangle(aoiPayload.bounds, aoiStyle);
        }}
        if (aoiLayer) {{
          aoiLayer.addTo(map);
          overlayLayers['📐 ' + aoiLabel] = aoiLayer;
        }}
      }}

      L.control.scale({{ imperial: false, position: 'bottomleft' }}).addTo(map);
      L.control.layers(baseLayers, overlayLayers, {{ position: 'bottomright', collapsed: true }}).addTo(map);

      map.fitBounds(fitBounds, {{ padding: [15, 15] }});

      const wrapper = document.getElementById('swipe-wrapper');
      const divider = document.getElementById('swipe-divider');
      const coordsEl = document.getElementById('swipe-coords');

      map.on('mousemove', function (ev) {{
        if (ev && ev.latlng && coordsEl) {{
          coordsEl.textContent = 'WGS84 (φ, λ): ' + ev.latlng.lat.toFixed(5) + ', ' + ev.latlng.lng.toFixed(5);
        }}
      }});

      let splitRatio = 0.5;
      let isDragging = false;

      function updateClip() {{
        const size = map.getSize();
        const w = size.x;
        const h = size.y;
        if (!w || !h) return;
        const splitX = Math.max(0, Math.min(w, Math.round(w * splitRatio)));
        divider.style.left = splitX + 'px';

        const nw = map.containerPointToLayerPoint([0, 0]);
        const se = map.containerPointToLayerPoint([w, h]);
        const clipX = map.containerPointToLayerPoint([splitX, 0]).x;

        t1Pane.style.clip = 'rect(' + nw.y + 'px, ' + clipX + 'px, ' + se.y + 'px, ' + nw.x + 'px)';
        t2Pane.style.clip = 'rect(' + nw.y + 'px, ' + se.x + 'px, ' + se.y + 'px, ' + clipX + 'px)';
      }}

      function setFromClientX(clientX) {{
        const rect = wrapper.getBoundingClientRect();
        if (rect.width <= 0) return;
        const ratio = (clientX - rect.left) / rect.width;
        splitRatio = Math.max(0.02, Math.min(0.98, ratio));
        updateClip();
      }}

      function startDrag(e) {{
        isDragging = true;
        divider.classList.add('dragging');
        if (e.preventDefault) e.preventDefault();
        if (e.stopPropagation) e.stopPropagation();
        if (map.dragging && map.dragging.enabled()) {{
          map.dragging.disable();
        }}
        if (e.pointerId !== undefined && divider.setPointerCapture) {{
          try {{ divider.setPointerCapture(e.pointerId); }} catch (_err) {{}}
        }}
        if (e.touches && e.touches.length > 0) {{
          setFromClientX(e.touches[0].clientX);
        }} else if (typeof e.clientX === 'number') {{
          setFromClientX(e.clientX);
        }}
      }}

      function moveDrag(e) {{
        if (!isDragging) return;
        if (e.preventDefault) e.preventDefault();
        if (e.stopPropagation) e.stopPropagation();
        if (e.touches && e.touches.length > 0) {{
          setFromClientX(e.touches[0].clientX);
        }} else if (typeof e.clientX === 'number') {{
          setFromClientX(e.clientX);
        }}
      }}

      function endDrag(e) {{
        if (!isDragging) return;
        isDragging = false;
        divider.classList.remove('dragging');
        if (e && e.pointerId !== undefined && divider.releasePointerCapture) {{
          try {{ divider.releasePointerCapture(e.pointerId); }} catch (_err) {{}}
        }}
        if (map.dragging && !map.dragging.enabled()) {{
          map.dragging.enable();
        }}
      }}

      L.DomEvent.disableClickPropagation(divider);
      L.DomEvent.disableScrollPropagation(divider);

      if (window.PointerEvent) {{
        divider.addEventListener('pointerdown', startDrag);
        window.addEventListener('pointermove', moveDrag);
        window.addEventListener('pointerup', endDrag);
        window.addEventListener('pointercancel', endDrag);
      }} else {{
        divider.addEventListener('mousedown', startDrag);
        divider.addEventListener('touchstart', startDrag, {{ passive: false }});
        window.addEventListener('mousemove', moveDrag);
        window.addEventListener('touchmove', moveDrag, {{ passive: false }});
        window.addEventListener('mouseup', endDrag);
        window.addEventListener('touchend', endDrag);
        window.addEventListener('touchcancel', endDrag);
      }}

      map.on('move zoom moveend zoomend viewreset resize', updateClip);
      map.whenReady(function () {{
        updateClip();
        setTimeout(updateClip, 150);
        setTimeout(updateClip, 400);
      }});
      window.addEventListener('resize', function () {{
        map.invalidateSize();
        updateClip();
      }});
    }})();
  </script>
</body>
</html>
"""
    return html_doc

