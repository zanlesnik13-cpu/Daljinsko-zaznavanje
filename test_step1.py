"""Automated Step 1 verification script for geodezija_crs.py and stac_loader.py.

Verifies:
1. Geodetic CRS transformations and 2 km x 2 km AOI generation over Ljubljana (D96/TM & WGS84).
2. AOI area calculation and resolution step-down validation.
3. Planetary Computer Sentinel-2 L2A STAC search (last 6 months, < 20% cloud cover).
4. Clipped COG loading of the clearest scene (B04, B08, SCL) onto EPSG:3794 at 10 m.
5. Physical surface reflectance range [0, 1] and uint8 SCL integrity.
6. In-memory LZW-compressed GeoTIFF export (EPSG:3794 and EPSG:32633).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np

from geodezija_crs import (
    CRS_D96TM,
    CRS_UTM33N,
    EPSG_D96TM,
    EPSG_UTM33N,
    EPSG_WGS84,
    bbox_from_point_d96tm,
    bbox_from_point_wgs84,
    export_to_geotiff_bytes,
    transform_coords,
    validate_aoi_size,
)
from rasterio.io import MemoryFile
from stac_loader import load_sentinel2_scene, search_sentinel2


def main() -> int:
    print("=" * 72)
    print("STEP 1 VERIFICATION — GEODETIC ENGINE & SENTINEL-2 STAC LOADER")
    print("=" * 72)

    # 1. Define 2 km x 2 km AOI around Ljubljana (radius = 1000 m)
    lj_lat, lj_lon = 46.0569, 14.5058
    radius_m = 1000.0

    e_lj, n_lj = transform_coords(lj_lon, lj_lat, EPSG_WGS84, EPSG_D96TM)
    lon_back, lat_back = transform_coords(e_lj, n_lj, EPSG_D96TM, EPSG_WGS84)
    print(f"[1] Ljubljana WGS84 : lat={lj_lat:.6f}, lon={lj_lon:.6f}")
    print(f"    Ljubljana D96/TM: E={e_lj:.2f} m, N={n_lj:.2f} m")
    assert abs(lon_back - lj_lon) < 1e-7 and abs(lat_back - lj_lat) < 1e-7, (
        "Round-trip WGS84 <-> D96/TM transformation exceeded tolerance!"
    )
    # Sanity check D96/TM range for Ljubljana (~E=461790, N=101850)
    assert 455_000 < e_lj < 470_000 and 95_000 < n_lj < 110_000, (
        f"Unexpected D96/TM coordinates for Ljubljana: E={e_lj}, N={n_lj}"
    )

    bbox_4326, bbox_3794 = bbox_from_point_wgs84(lj_lat, lj_lon, radius_m)
    bbox_4326_d96, bbox_3794_d96 = bbox_from_point_d96tm(e_lj, n_lj, radius_m)
    assert np.allclose(bbox_3794, bbox_3794_d96, atol=1e-6), "D96/TM bbox mismatch!"
    assert np.allclose(bbox_4326, bbox_4326_d96, atol=1e-9), "WGS84 bbox mismatch!"

    width_m = bbox_3794[2] - bbox_3794[0]
    height_m = bbox_3794[3] - bbox_3794[1]
    print(f"    AOI EPSG:3794   : {[round(v, 2) for v in bbox_3794]} ({width_m:.0f} m x {height_m:.0f} m)")
    print(f"    AOI EPSG:4326   : {[round(v, 6) for v in bbox_4326]}")
    assert abs(width_m - 2000.0) < 1e-6 and abs(height_m - 2000.0) < 1e-6

    area_km2, is_valid, rec_res = validate_aoi_size(bbox_4326)
    print(f"    AOI validation  : area={area_km2:.4f} km^2, valid={is_valid}, rec_res={rec_res} m")
    assert is_valid is True
    assert 3.9 <= area_km2 <= 4.3, f"Expected ~4 km^2 enclosing bbox, got {area_km2}"
    assert rec_res == 10

    # Verify large-AOI resolution step-down guard
    large_bbox_4326, _ = bbox_from_point_wgs84(lj_lat, lj_lon, 18_000.0)  # 36x36 km = ~1296 km^2
    large_area, large_valid, large_res = validate_aoi_size(large_bbox_4326, max_area_km2=900.0)
    assert large_valid is False and large_res == 60, (
        f"Large AOI guard failed: area={large_area}, valid={large_valid}, res={large_res}"
    )
    print(f"    Large AOI guard : area={large_area:.1f} km^2, valid={large_valid}, rec_res={large_res} m (OK)")

    # 2. Search Sentinel-2 L2A scenes from the last 6 months with < 20% cloud cover
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=183)
    print(f"\n[2] Searching Sentinel-2 L2A ({start_dt} to {end_dt}, cloud < 20%)...")
    items, summary_df = search_sentinel2(
        bbox_4326=bbox_4326,
        start_date=start_dt,
        end_date=end_dt,
        max_cloud_cover=20,
        limit=50,
    )
    print(f"    Scenes found    : {len(items)}")
    assert len(items) > 0, "No Sentinel-2 L2A scenes found over Ljubljana in the last 6 months!"
    assert len(items) == len(summary_df)
    expected_cols = ["id", "datetime", "date_str", "cloud_cover", "mgrs_tile", "platform"]
    assert list(summary_df.columns) == expected_cols, f"Unexpected columns: {list(summary_df.columns)}"

    # Pick the clearest scene
    best_idx = int(summary_df["cloud_cover"].idxmin())
    best_item = items[best_idx]
    best_row = summary_df.iloc[best_idx]
    print(
        f"    Clearest scene  : {best_row['id']} "
        f"(date={best_row['date_str']}, cloud={best_row['cloud_cover']:.2f}%, "
        f"tile={best_row['mgrs_tile']}, platform={best_row['platform']})"
    )

    # 3. Load clearest scene (B04, B08, SCL) in EPSG:3794 at 10 m
    bands = ["B04", "B08", "SCL"]
    print(f"\n[3] Loading clipped scene bands={bands} in {CRS_D96TM} @ 10 m...")
    ds = load_sentinel2_scene(
        item=best_item,
        bbox_4326=bbox_4326,
        bands=bands,
        target_epsg=CRS_D96TM,
        resolution=10,
    )

    crs_loaded = ds.rio.crs
    print(f"    Loaded CRS      : {crs_loaded}")
    print(f"    Radiometric PB  : {ds.attrs.get('processing_baseline')} "
          f"(offset -1000 applied: {ds.attrs.get('radiometric_offset_applied')})")
    assert crs_loaded is not None and crs_loaded.to_epsg() == EPSG_D96TM

    for band in bands:
        da = ds[band]
        print(f"    Band {band:<3}        : shape={da.shape}, dtype={da.dtype}", end="")
        if band in ("B04", "B08"):
            valid_vals = da.values[np.isfinite(da.values)]
            assert valid_vals.size > 0, f"Band {band} has no valid pixels!"
            b_min = float(valid_vals.min())
            b_max = float(valid_vals.max())
            b_mean = float(valid_vals.mean())
            print(f", min={b_min:.4f}, max={b_max:.4f}, mean={b_mean:.4f}")
            assert 0.0 <= b_min <= b_max <= 1.0, (
                f"Band {band} reflectance out of physical [0, 1] bounds: [{b_min}, {b_max}]"
            )
            assert da.dtype == np.float32
        else:
            unique_classes = np.unique(da.values).tolist()
            print(f", unique_classes={unique_classes}")
            assert da.dtype == np.uint8, f"Expected SCL dtype uint8, got {da.dtype}"
            assert all(0 <= c <= 11 for c in unique_classes), f"Invalid SCL classes: {unique_classes}"

    # Verify approximate 200 x 200 pixel grid (~2 km at 10 m)
    h, w = ds["B04"].shape
    assert 180 <= h <= 240 and 180 <= w <= 240, f"Unexpected grid shape for 2x2 km @ 10m: {(h, w)}"

    # 4. Verify GeoTIFF export (single DataArray in D96/TM & Dataset reprojected to UTM 33N)
    print("\n[4] Testing in-memory GeoTIFF export...")
    b04_tif_bytes = export_to_geotiff_bytes(ds["B04"], target_epsg=EPSG_D96TM)
    print(f"    B04 GeoTIFF (EPSG:3794)     : {len(b04_tif_bytes):,} bytes")
    assert len(b04_tif_bytes) > 0

    with MemoryFile(b04_tif_bytes) as mf, mf.open() as src:
        print(
            f"      verified -> CRS={src.crs}, shape=({src.height}, {src.width}), "
            f"bands={src.count}, compression={src.compression}, desc={src.descriptions}"
        )
        assert src.crs.to_epsg() == EPSG_D96TM
        assert src.count == 1
        assert src.compression is not None and src.compression.name.upper() == "LZW"

    ds_tif_bytes = export_to_geotiff_bytes(ds[["B04", "B08"]], target_epsg=CRS_UTM33N)
    print(f"    B04+B08 GeoTIFF (EPSG:32633): {len(ds_tif_bytes):,} bytes")
    assert len(ds_tif_bytes) > 0

    with MemoryFile(ds_tif_bytes) as mf, mf.open() as src:
        print(
            f"      verified -> CRS={src.crs}, shape=({src.height}, {src.width}), "
            f"bands={src.count}, compression={src.compression}, desc={src.descriptions}"
        )
        assert src.crs.to_epsg() == EPSG_UTM33N
        assert src.count == 2
        assert src.descriptions == ("B04", "B08")

    print("\n" + "=" * 72)
    print("ALL STEP 1 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
