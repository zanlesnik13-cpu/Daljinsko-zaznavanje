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
import io
import math
from typing import Any, Sequence

import folium
from folium.plugins import MousePosition
import matplotlib
import numpy as np
from PIL import Image
from rasterio.enums import Resampling
import rioxarray  # noqa: F401 - registers .rio accessor on xarray objects
import xarray as xr

from geodezija_crs import EPSG_D96TM, EPSG_WGS84, transform_coords, validate_aoi_size

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
    max_area_km2: float = 900.0,
) -> list[float]:
    """Expand an EPSG:4326 AOI bounding box on each side for smooth slippy-map panning.

    Caps the effective buffer automatically so the expanded download area never
    exceeds ``max_area_km2`` (Streamlit Cloud 1 GB RAM safety guard).
    """
    if len(bbox_4326) != 4:
        raise ValueError(f"bbox_4326 must contain 4 values, got {len(bbox_4326)}.")

    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox_4326)
    if min_lon >= max_lon or min_lat >= max_lat:
        return [min_lon, min_lat, max_lon, max_lat]

    factor = max(0.0, float(buffer_factor))
    if factor == 0.0:
        return [min_lon, min_lat, max_lon, max_lat]

    cur_area, _, _ = validate_aoi_size([min_lon, min_lat, max_lon, max_lat], max_area_km2=max_area_km2)
    safe_cap = float(max_area_km2) * 0.96
    if cur_area <= 0.0 or cur_area >= safe_cap:
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
    max_width: int = 1200,
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
    max_width: int = 1200,
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
        # Build a 3-band float32 DataArray (band, y, x) in source CRS before a single clean reprojection
        if isinstance(data_array_or_rgb, xr.Dataset):
            ds = data_array_or_rgb
            req_bands = list(bands) if bands else list(ds.data_vars)[:3]

            # Special case: Sentinel-1 2-pol false color composite (VV, VH -> R=VV, G=VH, B=VV-VH)
            if len(req_bands) == 2 and set(req_bands) == {"VV", "VH"}:
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
        else:
            raise TypeError(f"Expected xarray DataArray or Dataset, got {type(data_array_or_rgb).__name__}.")

        reproj = _prepare_vis_dataarray(stacked, src_epsg=src_epsg, resampling=Resampling.bilinear, nodata=np.nan)
        west, south, east, north = reproj.rio.bounds()
        y_coords = np.asarray(reproj.coords["y"].values, dtype=np.float64)

        arr3 = reproj.values.astype(np.float32, copy=False)  # (3, H, W)
        valid = np.isfinite(arr3[0]) & np.isfinite(arr3[1]) & np.isfinite(arr3[2])

        h, w = arr3.shape[1], arr3.shape[2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for idx in range(3):
            f_range = (
                fixed_ranges[idx]
                if (fixed_ranges is not None and idx < len(fixed_ranges))
                else (0.0, 0.35)
            )
            stretched = _stretch_channel_01(
                arr3[idx],
                use_percentile_stretch=use_percentile_stretch,
                fixed_range=f_range,
                gamma=gamma,
            )
            rgba[..., idx] = np.where(
                valid,
                np.clip(np.round(stretched * 255.0), 0, 255),
                0,
            ).astype(np.uint8)
        rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
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


def _d96tm_bbox_to_wgs84_ring(bbox_3794: Sequence[float]) -> list[list[float]]:
    min_e, min_n, max_e, max_n = (float(v) for v in bbox_3794)
    corners = [
        (min_e, min_n),
        (max_e, min_n),
        (max_e, max_n),
        (min_e, max_n),
        (min_e, min_n),
    ]
    ring: list[list[float]] = []
    for e_val, n_val in corners:
        lon, lat = transform_coords(e_val, n_val, EPSG_D96TM, EPSG_WGS84)
        ring.append([lat, lon])
    return ring


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
    aoi_outline_label: str = "AOI (meja analize)",
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
            ring = _d96tm_bbox_to_wgs84_ring(aoi_bbox_3794)
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

    # Live cursor coordinate readout (WGS84 lat, lon)
    MousePosition(
        position="bottomleft",
        separator=" , ",
        empty_string="",
        lng_first=False,
        num_digits=5,
        prefix="WGS84 (φ, λ):",
    ).add_to(fmap)

    # QGIS-style always-visible layer control
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap
