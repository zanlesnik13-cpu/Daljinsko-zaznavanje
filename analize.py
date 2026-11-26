"""Multi-sensor cloud/shadow masking, optical spectral indices, SAR analytics, and statistics.

Supports:
- Sentinel-2 L2A SCL cloud & shadow masking (`classes 0, 1, 3, 8, 9, 10`).
- Landsat Collection 2 Level-2 `QA_PIXEL` bitmasking (`(qa & 0b11110) != 0`).
- Harmonized optical spectral indices: NDVI, NDWI, MNDWI, NDBI, NBR.
- Sentinel-1 SAR RTC decibel analytics (`VV`, `VH`, `VV/VH` dB ratio, and
  threshold-based water/flood mapping `VV < threshold_db`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers .rio accessor on xarray objects
import xarray as xr

# Sentinel-2 L2A Scene Classification Layer (SCL) classes masked out as invalid/cloudy:
# 0: No Data, 1: Saturated/Defective, 3: Cloud Shadows,
# 8: Cloud Medium Prob., 9: Cloud High Prob., 10: Thin Cirrus
SCL_INVALID_CLASSES: tuple[int, ...] = (0, 1, 3, 8, 9, 10)

# Landsat C2 L2 QA_PIXEL cloud/shadow bitmask:
# bit 1: Dilated Cloud, bit 2: Cirrus, bit 3: Cloud, bit 4: Cloud Shadow
LANDSAT_QA_CLOUD_MASK: int = 0b11110  # 30
LANDSAT_QA_FILL_MASK: int = 0b00001   # 1

SCL_CLASS_LABELS: dict[int, str] = {
    0: "No Data",
    1: "Saturated / Defective",
    2: "Dark Area / Cast Shadows",
    3: "Cloud Shadows",
    4: "Vegetation",
    5: "Not Vegetated",
    6: "Water",
    7: "Unclassified",
    8: "Cloud Medium Probability",
    9: "Cloud High Probability",
    10: "Thin Cirrus",
    11: "Snow / Ice",
}

INDEX_CATEGORIES: tuple[dict[str, str], ...] = (
    {"key": "Vegetacija", "label_sl": "🌿 Vegetacija", "label_en": "🌿 Vegetation"},
    {"key": "Voda in sneg", "label_sl": "💧 Voda in sneg", "label_en": "💧 Water & Snow"},
    {"key": "Tla in pozidanost", "label_sl": "🏗️ Tla in pozidanost", "label_en": "🏗️ Soil & Built-Up"},
    {"key": "Požari in toplota", "label_sl": "🔥 Požari in toplota", "label_en": "🔥 Fire & Thermal"},
)

SPECTRAL_INDICES_META: dict[str, dict[str, Any]] = {
    # --- 1. VEGETACIJA ---
    "NDVI": {
        "code": "NDVI",
        "name_sl": "Normalizirani diferenčni indeks vegetacije",
        "name_en": "Normalized Difference Vegetation Index",
        "category": "Vegetacija",
        "bands": ("B08", "B04"),
        "formula": "(NIR − Red) / (NIR + Red)",
        "short_desc_sl": "Meri vitalnost rastlinja, fotosintetsko aktivnost in gostoto biomase.",
        "short_desc_en": "Measures vegetation vigor, photosynthetic activity, and biomass density.",
        "interpretation_sl": "< 0: voda/sneg · 0–0.2: gola tla/pozidava · 0.2–0.5: redko rastlinje · > 0.6: gost zdrav gozd",
        "interpretation_en": "< 0: water/snow · 0–0.2: bare soil/urban · 0.2–0.5: sparse vegetation · > 0.6: dense healthy forest",
        "best_for_sl": "Gozdovi, kmetijstvo, fenologija in spremljanje krčenja vegetacije.",
        "best_for_en": "Forests, agriculture, phenology, and vegetation loss monitoring.",
        "colormap": "RdYlGn",
        "cmap": "RdYlGn",
        "vmin": -0.2,
        "vmax": 0.9,
        "diff_cmap": "RdYlGn",
        "supported_sensors": ("sentinel-2-l2a", "landsat-c2-l2"),
    },
    "EVI": {
        "code": "EVI",
        "name_sl": "Izboljšan indeks vegetacije",
        "name_en": "Enhanced Vegetation Index",
        "category": "Vegetacija",
        "bands": ("B08", "B04", "B02"),
        "formula": "2.5 × (NIR − Red) / (NIR + 6×Red − 7.5×Blue + 1)",
        "short_desc_sl": "Ne zasiči se v gostih gozdovih in zmanjšuje vpliv atmosfere ter ozadja tal.",
        "short_desc_en": "Resists saturation in dense canopies while correcting for atmosphere and soil background.",
        "interpretation_sl": "< 0.1: gola tla/voda · 0.2–0.4: travniki/poljščine · > 0.5: zelo gosta biomasa",
        "interpretation_en": "< 0.1: bare/water · 0.2–0.4: grasslands/crops · > 0.5: very dense canopy biomass",
        "best_for_sl": "Gosti gozdovi, tropska/bujna biomasa in primerjava med letnimi časi.",
        "best_for_en": "Dense forests, high-biomass canopies, and multi-season vigor tracking.",
        "colormap": "RdYlGn",
        "cmap": "RdYlGn",
        "vmin": -0.2,
        "vmax": 0.9,
        "diff_cmap": "RdYlGn",
        "supported_sensors": ("sentinel-2-l2a", "landsat-c2-l2"),
    },
    "SAVI": {
        "code": "SAVI",
        "name_sl": "Talno prilagojen indeks vegetacije",
        "name_en": "Soil-Adjusted Vegetation Index",
        "category": "Vegetacija",
        "bands": ("B08", "B04"),
        "formula": "1.5 × (NIR − Red) / (NIR + Red + 0.5)",
        "short_desc_sl": "Zmanjšuje vpliv svetlosti golih tal (faktor L = 0.5) pri redkejši vegetaciji.",
        "short_desc_en": "Minimizes bare soil brightness effects (L = 0.5) over sparse vegetation cover.",
        "interpretation_sl": "< 0.1: gola tla · 0.15–0.4: mlade poljščine/kras · > 0.5: sklenjena vegetacija",
        "interpretation_en": "< 0.1: bare soil · 0.15–0.4: young crops/karst · > 0.5: closed canopy",
        "best_for_sl": "Kmetijske površine v zgodnji rasti, sušna in kraška območja.",
        "best_for_en": "Early-stage cropland, arid regions, and sparse karst vegetation.",
        "colormap": "RdYlGn",
        "cmap": "RdYlGn",
        "vmin": -0.2,
        "vmax": 0.9,
        "diff_cmap": "RdYlGn",
        "supported_sensors": ("sentinel-2-l2a", "landsat-c2-l2"),
    },
    "NDMI": {
        "code": "NDMI",
        "name_sl": "Indeks vlažnosti vegetacije",
        "name_en": "Normalized Difference Moisture Index",
        "category": "Vegetacija",
        "bands": ("B08", "B11"),
        "formula": "(NIR − SWIR1) / (NIR + SWIR1)",
        "short_desc_sl": "Meri vsebnost vode v listih — zgodnje odkrivanje sušnega stresa in lubadarja.",
        "short_desc_en": "Measures canopy leaf water content for early drought and bark-beetle stress detection.",
        "interpretation_sl": "< −0.2: močan vodni stres/gola tla · 0–0.2: zmerna vlaga · > 0.4: dobro preskrbljen gozd",
        "interpretation_en": "< −0.2: severe water stress/bare · 0–0.2: moderate moisture · > 0.4: well-hydrated canopy",
        "best_for_sl": "Sušni stres gozdov, poškodbe zaradi lubadarja in namakanje v kmetijstvu.",
        "best_for_en": "Forest drought stress, bark-beetle damage, and crop irrigation monitoring.",
        "colormap": "BrBG",
        "cmap": "BrBG",
        "vmin": -0.6,
        "vmax": 0.7,
        "diff_cmap": "BrBG",
        "supported_sensors": ("sentinel-2-l2a", "landsat-c2-l2"),
    },
    "NDRE": {
        "code": "NDRE",
        "name_sl": "Indeks rdečega roba (Sentinel-2)",
        "name_en": "Normalized Difference Red Edge (Sentinel-2)",
        "category": "Vegetacija",
        "bands": ("B08", "B05"),
        "formula": "(NIR − RedEdge1) / (NIR + RedEdge1)",
        "short_desc_sl": "Zelo občutljiv na vsebnost klorofila — odkriva stres rastlin prej kot klasični NDVI.",
        "short_desc_en": "Highly sensitive to chlorophyll content — detects plant stress earlier than NDVI.",
        "interpretation_sl": "< 0.1: gola tla/brez klorofila · 0.2–0.4: zmeren klorofil · > 0.5: visoka vitalnost",
        "interpretation_en": "< 0.1: bare/no chlorophyll · 0.2–0.4: moderate chlorophyll · > 0.5: peak canopy vigor",
        "best_for_sl": "Precizno kmetijstvo, gnojenje in zgodnji stres v gostih krošnjah (samo Sentinel-2).",
        "best_for_en": "Precision agriculture, nitrogen status, and early canopy stress (Sentinel-2 only).",
        "colormap": "RdYlGn",
        "cmap": "RdYlGn",
        "vmin": -0.2,
        "vmax": 0.8,
        "diff_cmap": "RdYlGn",
        "s2_only": True,
        "supported_sensors": ("sentinel-2-l2a",),
    },
    # --- 2. VODA IN SNEG ---
    "NDWI": {
        "code": "NDWI",
        "name_sl": "Normalizirani indeks vode",
        "name_en": "Normalized Difference Water Index",
        "category": "Voda in sneg",
        "bands": ("B03", "B08"),
        "formula": "(Green − NIR) / (Green + NIR)",
        "short_desc_sl": "Poudarja odprte vodne površine: reke, jezera, mokrišča in ledeniška jezera.",
        "short_desc_en": "Highlights open surface water: rivers, lakes, wetlands, and proglacial lakes.",
        "interpretation_sl": "< 0: kopno/vegetacija · 0.05–0.2: motna/plitva voda · > 0.3: čista globoka voda",
        "interpretation_en": "< 0: land/vegetation · 0.05–0.2: turbid/shallow water · > 0.3: clear deep water",
        "best_for_sl": "Kartiranje jezer, poplav, obalnih linij in novih ledeniških jezer.",
        "best_for_en": "Mapping lakes, floods, shorelines, and newly formed proglacial lakes.",
        "colormap": "RdBu",
        "cmap": "RdBu",
        "vmin": -0.6,
        "vmax": 0.6,
        "diff_cmap": "RdBu",
        "supported_sensors": ("sentinel-2-l2a", "landsat-c2-l2"),
    },
    "MNDWI": {
        "code": "MNDWI",
        "name_sl": "Modificirani indeks vode",
        "name_en": "Modified Normalized Difference Water Index",
        "category": "Voda in sneg",
        "bands": ("B03", "B11"),
        "formula": "(Green − SWIR1) / (Green + SWIR1)",
        "short_desc_sl": "Natančno zaznava vodo v urbanem okolju in učinkovito izloča sence stavb.",
        "short_desc_en": "Accurately isolates water in urban areas while suppressing building shadows.",
        "interpretation_sl": "< 0: pozidava/tla · > 0.1: vodne površine (tudi mestni kanali in poplave)",
        "interpretation_en": "< 0: built-up/land · > 0.1: open water (including urban canals and floods)",
        "best_for_sl": "Mestne poplave, pristanišča, kanali in vodotoki ob pozidanih območjih.",
        "best_for_en": "Urban flooding, harbors, canals, and water bodies near dense structures.",
        "colormap": "RdBu",
        "cmap": "RdBu",
        "vmin": -0.6,
        "vmax": 0.6,
        "diff_cmap": "RdBu",
        "supported_sensors": ("sentinel-2-l2a", "landsat-c2-l2"),
    },
    "NDSI": {
        "code": "NDSI",
        "name_sl": "Indeks snega in ledu",
        "name_en": "Normalized Difference Snow Index",
        "category": "Voda in sneg",
        "bands": ("B03", "B11"),
        "formula": "(Green − SWIR1) / (Green + SWIR1)",
        "short_desc_sl": "Loči sneg in ledeniški led od oblakov, skal in tal (z vodno masko za jezera).",
        "short_desc_en": "Separates snow and glacier ice from clouds, rock, and soil (water-masked for lakes).",
        "interpretation_sl": "< 0.2: brez snega · 0.2–0.4: mešan sneg/grušč · ≥ 0.4: čisti sneg in ledeniški led",
        "interpretation_en": "< 0.2: snow-free · 0.2–0.4: mixed snow/debris · ≥ 0.4: clean snow & glacier ice",
        "best_for_sl": "Umik ledenikov, obseg snežne odeje in nadmorska višina snežne meje.",
        "best_for_en": "Glacier retreat, seasonal snow cover extent, and snowline altitude.",
        "colormap": "PuBu",
        "cmap": "PuBu",
        "vmin": -0.2,
        "vmax": 0.9,
        "diff_cmap": "RdBu",
        "supported_sensors": ("sentinel-2-l2a", "landsat-c2-l2"),
    },
    # --- 3. TLA IN POZIDANOST ---
    "NDBI": {
        "code": "NDBI",
        "name_sl": "Indeks pozidanosti",
        "name_en": "Normalized Difference Built-Up Index",
        "category": "Tla in pozidanost",
        "bands": ("B11", "B08"),
        "formula": "(SWIR1 − NIR) / (SWIR1 + NIR)",
        "short_desc_sl": "Poudarja urbane površine, beton, asfalt, industrijske cone in širjenje naselij.",
        "short_desc_en": "Highlights urban surfaces, concrete, asphalt, industrial zones, and sprawl.",
        "interpretation_sl": "< −0.1: vegetacija/voda · 0–0.15: mešana pozidava · > 0.2: gosta urbana/industrijska raba",
        "interpretation_en": "< −0.1: vegetation/water · 0–0.15: mixed urban · > 0.2: dense built-up/industrial",
        "best_for_sl": "Širjenje mest, gradnja infrastrukture in zaznava neprepustnih površin.",
        "best_for_en": "Urban expansion, infrastructure growth, and impervious surface mapping.",
        "colormap": "RdYlBu_r",
        "cmap": "RdYlBu_r",
        "vmin": -0.5,
        "vmax": 0.5,
        "diff_cmap": "RdBu_r",
        "supported_sensors": ("sentinel-2-l2a", "landsat-c2-l2"),
    },
    "BSI": {
        "code": "BSI",
        "name_sl": "Indeks golih tal",
        "name_en": "Bare Soil Index",
        "category": "Tla in pozidanost",
        "bands": ("B11", "B04", "B08", "B02"),
        "formula": "((SWIR1 + Red) − (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))",
        "short_desc_sl": "Izpostavlja gola tla, gradbišča, kamnolome, erozijo in sveže gozdne poseke.",
        "short_desc_en": "Highlights bare soil, construction sites, quarries, erosion, and clearcuts.",
        "interpretation_sl": "< −0.1: gosta vegetacija/voda · 0–0.15: redka tla · > 0.2: izpostavljena gola tla/gradbišča",
        "interpretation_en": "< −0.1: dense vegetation/water · 0–0.15: sparse soil · > 0.2: exposed bare soil/quarries",
        "best_for_sl": "Kartiranje gradbišč, kamnolomov, erozije in razgaljenih kmetijskih tal.",
        "best_for_en": "Construction sites, quarries, soil erosion, and tilled bare fields.",
        "colormap": "YlOrBr",
        "cmap": "YlOrBr",
        "vmin": -0.6,
        "vmax": 0.6,
        "diff_cmap": "RdBu_r",
        "supported_sensors": ("sentinel-2-l2a", "landsat-c2-l2"),
    },
    # --- 4. POŽARI IN TOPLOTA ---
    "NBR": {
        "code": "NBR",
        "name_sl": "Normalizirano razmerje požarišč",
        "name_en": "Normalized Burn Ratio",
        "category": "Požari in toplota",
        "bands": ("B08", "B12"),
        "formula": "(NIR − SWIR2) / (NIR + SWIR2)",
        "short_desc_sl": "Ocenjuje zdravje gozda pred požarom in jakost poškodb (dNBR) po požarih.",
        "short_desc_en": "Assesses pre-fire forest health and post-fire burn severity (dNBR).",
        "interpretation_sl": "< −0.2: hudo požgana tla · 0–0.2: gola tla/zmerne poškodbe · > 0.4: zdrav nepoškodovan gozd",
        "interpretation_en": "< −0.2: severely burned · 0–0.2: bare/moderate burn · > 0.4: healthy unburned forest",
        "best_for_sl": "Obseg in jakost gozdnih požarov ter spremljanje obnove vegetacije.",
        "best_for_en": "Wildfire extent, burn severity mapping, and post-fire regrowth.",
        "colormap": "BrBG",
        "cmap": "BrBG",
        "vmin": -0.5,
        "vmax": 0.8,
        "diff_cmap": "BrBG",
        "supported_sensors": ("sentinel-2-l2a", "landsat-c2-l2"),
    },
    "BAI": {
        "code": "BAI",
        "name_sl": "Indeks požganih površin",
        "name_en": "Burned Area Index",
        "category": "Požari in toplota",
        "bands": ("B04", "B08"),
        "formula": "1 / ((0.1 − Red)² + (0.06 − NIR)²)",
        "short_desc_sl": "Močno poudari sveža požarišča in zoglenela tla takoj po gozdnem požaru.",
        "short_desc_en": "Strongly highlights freshly burned scars and charcoal immediately after wildfire.",
        "interpretation_sl": "< 25: nepožgano · 25–60: delno ožgano · > 60: sveže požarišče / oglje",
        "interpretation_en": "< 25: unburned · 25–60: partially scorched · > 60: fresh burn scar / charcoal",
        "best_for_sl": "Hitro izločanje svežih požarišč z vidnim in NIR pasom.",
        "best_for_en": "Rapid detection of fresh wildfire scars using Red and NIR bands.",
        "colormap": "inferno",
        "cmap": "inferno",
        "vmin": 0.0,
        "vmax": 120.0,
        "diff_cmap": "RdBu_r",
        "supported_sensors": ("sentinel-2-l2a", "landsat-c2-l2"),
    },
    "LST": {
        "code": "LST",
        "name_sl": "Temperatura površja tal [°C] (Landsat)",
        "name_en": "Land Surface Temperature [°C] (Landsat)",
        "category": "Požari in toplota",
        "bands": ("LST",),
        "formula": "DN × 0.00341802 + 149.0 − 273.15 [°C]",
        "short_desc_sl": "Fizična temperatura površja tal v °C iz termalnega kanala Landsat ST_B10.",
        "short_desc_en": "Physical land surface temperature in °C from the Landsat ST_B10 thermal band.",
        "interpretation_sl": "< 15 °C: voda/gore · 20–30 °C: parki in gozdovi · > 35 °C: mestni toplotni otoki",
        "interpretation_en": "< 15 °C: water/alpine · 20–30 °C: parks & forests · > 35 °C: urban heat islands",
        "best_for_sl": "Mestni toplotni otoki (UHI), vročinski valovi in hladilni učinek parkov (Landsat).",
        "best_for_en": "Urban heat islands (UHI), heatwaves, and urban vegetation cooling (Landsat).",
        "colormap": "inferno",
        "cmap": "inferno",
        "vmin": -5.0,
        "vmax": 45.0,
        "diff_cmap": "RdBu_r",
        "units": "°C",
        "landsat_only": True,
        "supported_sensors": ("landsat-c2-l2",),
    },
}

# Backward-compatible alias used across all modules
SPECTRAL_INDICES: dict[str, dict[str, Any]] = SPECTRAL_INDICES_META

SAR_METRICS: dict[str, dict[str, Any]] = {
    "VV": {
        "pols": ("vv",),
        "formula": "10 * log10(VV)",
        "cmap": "Greys_r",
        "vmin": -22.0,
        "vmax": -2.0,
        "diff_cmap": "RdBu",
        "units": "dB",
    },
    "VH": {
        "pols": ("vh",),
        "formula": "10 * log10(VH)",
        "cmap": "Greys_r",
        "vmin": -28.0,
        "vmax": -8.0,
        "diff_cmap": "RdBu",
        "units": "dB",
    },
    "VV_VH": {
        "pols": ("vv", "vh"),
        "formula": "VV[dB] - VH[dB]",
        "cmap": "BrBG",
        "vmin": 2.0,
        "vmax": 14.0,
        "diff_cmap": "RdBu",
        "units": "dB",
    },
}

# Optical bands & nominal central wavelengths [nm] for Spectral Signature Profiler
OPTICAL_SIGNATURE_BANDS: tuple[tuple[str, int, str], ...] = (
    ("B02", 490, "Blue"),
    ("B03", 560, "Green"),
    ("B04", 665, "Red"),
    ("B08", 842, "NIR"),
    ("B11", 1610, "SWIR-1"),
    ("B12", 2190, "SWIR-2"),
)

# 4-class physical land-cover definitions (clean cartographic palette)
LAND_COVER_CLASSES: tuple[dict[str, Any], ...] = (
    {
        "class_key": "water",
        "class_id": 1,
        "rule": "NDWI > 0.1",
        "color": "#0369A1",
    },
    {
        "class_key": "built_bare",
        "class_id": 2,
        "rule": "NDWI <= 0.1 & NDVI < 0.2",
        "color": "#94A3B8",
    },
    {
        "class_key": "sparse_veg",
        "class_id": 3,
        "rule": "NDWI <= 0.1 & 0.2 <= NDVI < 0.5",
        "color": "#84CC16",
    },
    {
        "class_key": "dense_veg",
        "class_id": 4,
        "rule": "NDWI <= 0.1 & NDVI >= 0.5",
        "color": "#0F766E",
    },
)


def get_required_bands(index_name: str, include_scl: bool = True) -> list[str]:
    """Return the minimal list of Sentinel-2 bands needed for an index calculation."""
    key = index_name.strip().upper()
    if key not in SPECTRAL_INDICES:
        raise ValueError(
            f"Unsupported spectral index {index_name!r}. "
            f"Supported indices: {', '.join(SPECTRAL_INDICES.keys())}."
        )
    bands = list(SPECTRAL_INDICES[key]["bands"])
    if include_scl and "SCL" not in bands:
        bands.append("SCL")
    return bands


def get_required_bands_for_sensor(
    sensor: str,
    metric_name: str,
    include_cloud_mask: bool = True,
) -> list[str]:
    """Return the minimal band/polarization list for a given sensor and index/metric."""
    key = metric_name.strip().upper()
    s_norm = str(sensor or "").strip().lower()
    if s_norm == "sentinel-1-rtc":
        if key not in SAR_METRICS:
            raise ValueError(
                f"Unsupported SAR metric {metric_name!r}. "
                f"Supported SAR metrics: {', '.join(SAR_METRICS.keys())}."
            )
        return list(SAR_METRICS[key]["pols"])

    if key not in SPECTRAL_INDICES:
        raise ValueError(
            f"Unsupported spectral index {metric_name!r}. "
            f"Supported indices: {', '.join(SPECTRAL_INDICES.keys())}."
        )
    bands = list(SPECTRAL_INDICES[key]["bands"])
    if include_cloud_mask and not s_norm.startswith("modis"):
        mask_band = "QA_PIXEL" if s_norm == "landsat-c2-l2" else "SCL"
        if mask_band not in bands:
            bands.append(mask_band)
    return bands


def apply_scl_mask(
    ds: xr.Dataset,
    mask_clouds: bool = True,
) -> tuple[xr.Dataset, float]:
    """Mask cloudy, shadowed, and defective pixels using Sentinel-2 L2A SCL band.

    Masks SCL classes ``{0, 1, 3, 8, 9, 10}`` to ``np.nan`` across all optical
    data variables while keeping ``SCL`` intact as ``uint8``.
    """
    optical_vars = [name for name in ds.data_vars if str(name) != "SCL"]
    if not optical_vars:
        return ds, 0.0

    ref_shape = ds[optical_vars[0]].shape
    total_pixels = int(np.prod(ref_shape))
    if total_pixels == 0:
        return ds, 0.0

    optical_finite = np.ones(ref_shape, dtype=bool)
    for var in optical_vars:
        optical_finite &= np.isfinite(ds[var].values)

    if not mask_clouds or "SCL" not in ds.data_vars:
        valid_pct = float(100.0 * np.count_nonzero(optical_finite) / total_pixels)
        return ds, round(valid_pct, 2)

    scl_vals = ds["SCL"].values
    scl_clear = ~np.isin(scl_vals, SCL_INVALID_CLASSES)
    valid_mask = optical_finite & scl_clear
    valid_pct = float(100.0 * np.count_nonzero(valid_mask) / total_pixels)

    src_crs = ds.rio.crs
    masked_vars: dict[str, xr.DataArray] = {}
    for var in ds.data_vars:
        da = ds[var]
        if str(var) == "SCL":
            masked_vars[str(var)] = da
        else:
            arr = da.values.astype(np.float32, copy=True)
            arr[~scl_clear] = np.nan
            out_da = xr.DataArray(
                arr,
                dims=da.dims,
                coords=da.coords,
                name=da.name,
                attrs=dict(da.attrs),
            )
            if src_crs is not None:
                out_da = out_da.rio.write_crs(src_crs, inplace=False)
            out_da = out_da.rio.write_nodata(np.nan, inplace=False)
            masked_vars[str(var)] = out_da

    masked_ds = xr.Dataset(
        data_vars=masked_vars,
        coords=ds.coords,
        attrs={**dict(ds.attrs), "scl_mask_applied": True, "valid_pixel_pct": round(valid_pct, 2)},
    )
    if src_crs is not None:
        masked_ds = masked_ds.rio.write_crs(src_crs, inplace=False)

    return masked_ds, round(valid_pct, 2)


def apply_landsat_qa_mask(
    ds: xr.Dataset,
    mask_clouds: bool = True,
) -> tuple[xr.Dataset, float]:
    """Mask cloudy, cirrus, shadow, and fill pixels using Landsat C2 L2 ``QA_PIXEL``.

    Masks pixels where bit 1 (Dilated Cloud), bit 2 (Cirrus), bit 3 (Cloud),
    or bit 4 (Cloud Shadow) is set (``(qa & 0b11110) != 0``), as well as Fill
    bit 0 (``(qa & 0b00001) != 0``), replacing optical reflectance with ``np.nan``.
    """
    qa_key = "QA_PIXEL" if "QA_PIXEL" in ds.data_vars else ("qa_pixel" if "qa_pixel" in ds.data_vars else None)
    optical_vars = [name for name in ds.data_vars if str(name) != qa_key]
    if not optical_vars:
        return ds, 0.0

    ref_shape = ds[optical_vars[0]].shape
    total_pixels = int(np.prod(ref_shape))
    if total_pixels == 0:
        return ds, 0.0

    optical_finite = np.ones(ref_shape, dtype=bool)
    for var in optical_vars:
        optical_finite &= np.isfinite(ds[var].values)

    if not mask_clouds or qa_key is None:
        valid_pct = float(100.0 * np.count_nonzero(optical_finite) / total_pixels)
        return ds, round(valid_pct, 2)

    qa_vals = ds[qa_key].values.astype(np.uint16, copy=False)
    is_cloudy_or_fill = ((qa_vals & np.uint16(LANDSAT_QA_CLOUD_MASK)) != 0) | (
        (qa_vals & np.uint16(LANDSAT_QA_FILL_MASK)) != 0
    )
    qa_clear = ~is_cloudy_or_fill
    valid_mask = optical_finite & qa_clear
    valid_pct = float(100.0 * np.count_nonzero(valid_mask) / total_pixels)

    src_crs = ds.rio.crs
    masked_vars: dict[str, xr.DataArray] = {}
    for var in ds.data_vars:
        da = ds[var]
        if str(var) == qa_key:
            masked_vars[str(var)] = da
        else:
            arr = da.values.astype(np.float32, copy=True)
            arr[~qa_clear] = np.nan
            out_da = xr.DataArray(
                arr,
                dims=da.dims,
                coords=da.coords,
                name=da.name,
                attrs=dict(da.attrs),
            )
            if src_crs is not None:
                out_da = out_da.rio.write_crs(src_crs, inplace=False)
            out_da = out_da.rio.write_nodata(np.nan, inplace=False)
            masked_vars[str(var)] = out_da

    masked_ds = xr.Dataset(
        data_vars=masked_vars,
        coords=ds.coords,
        attrs={**dict(ds.attrs), "qa_mask_applied": True, "valid_pixel_pct": round(valid_pct, 2)},
    )
    if src_crs is not None:
        masked_ds = masked_ds.rio.write_crs(src_crs, inplace=False)

    return masked_ds, round(valid_pct, 2)


def apply_cloud_mask(
    ds: xr.Dataset,
    mask_clouds: bool = True,
    sensor: str | None = None,
    sensor_id: str | None = None,
    **_kwargs: Any,
) -> tuple[xr.Dataset, float]:
    """Unified multi-sensor cloud mask dispatcher (Sentinel-2 SCL, Landsat QA_PIXEL, or SAR).

    Accepts optional ``sensor`` / ``sensor_id`` arguments for full compatibility across
    all modules (``nesrece.py``, ``ai_pomocnik.py``, ``gnss_plan.py``, ``casovne_vrste.py``, ``app.py``).
    """
    eff_sensor = str(sensor or sensor_id or "").strip().lower()
    if eff_sensor == "sentinel-1-rtc":
        mask_clouds = False

    if "SCL" in ds.data_vars and eff_sensor != "landsat-c2-l2":
        return apply_scl_mask(ds, mask_clouds=mask_clouds)
    if "QA_PIXEL" in ds.data_vars or "qa_pixel" in ds.data_vars:
        return apply_landsat_qa_mask(ds, mask_clouds=mask_clouds)

    # Radar (Sentinel-1) or dataset without QA band: report finite coverage
    data_vars = list(ds.data_vars)
    if not data_vars:
        return ds, 0.0
    ref_shape = ds[data_vars[0]].shape
    total_pixels = int(np.prod(ref_shape))
    if total_pixels == 0:
        return ds, 0.0
    finite_mask = np.ones(ref_shape, dtype=bool)
    for var in data_vars:
        finite_mask &= np.isfinite(ds[var].values)
    valid_pct = float(100.0 * np.count_nonzero(finite_mask) / total_pixels)
    return ds, round(valid_pct, 2)


def compute_index(ds: xr.Dataset, index_name: str) -> xr.DataArray:
    """Compute a normalized-difference spectral index ``[-1, 1]`` or Land Surface Temperature ``LST [°C]``.

    Works identically for Sentinel-2 L2A and harmonized Landsat C2 L2 Datasets (with ``LST``
    supported on Landsat C2 L2 thermal band ``lwir11`` / ``ST_B10`` / ``ST_B6``).
    """
    key = index_name.strip().upper()
    if key not in SPECTRAL_INDICES:
        raise ValueError(
            f"Unsupported spectral index {index_name!r}. "
            f"Choose one of: {', '.join(SPECTRAL_INDICES.keys())}."
        )

    spec = SPECTRAL_INDICES[key]

    # Single-band thermal Land Surface Temperature (LST in °C from Landsat C2 L2 lwir11 / ST_B10)
    if key == "LST":
        var_found = next((v for v in ("LST", "lwir11", "ST_B10", "lwir", "ST_B6") if v in ds.data_vars), None)
        if var_found is None:
            raise KeyError(
                f"Dataset is missing thermal band 'LST' / 'lwir11'. Available: {list(ds.data_vars)}."
            )
        src_da = ds[var_found]
        vals = src_da.values.astype(np.float32, copy=True)
        # If raw DN (> 1000) was passed directly instead of pre-calibrated °C, apply USGS C2 L2 formula
        fin_vals = vals[np.isfinite(vals)]
        if var_found != "LST" or (fin_vals.size > 0 and float(np.nanmedian(fin_vals)) > 200.0):
            valid_dn = np.isfinite(vals) & (vals > 0.0) & (vals < 65535.0)
            out_c = np.full(vals.shape, np.nan, dtype=np.float32)
            if np.any(valid_dn):
                scaled = vals[valid_dn].astype(np.float64) * 0.00341802 + 149.0 - 273.15
                out_c[valid_dn] = np.clip(scaled, -50.0, 80.0).astype(np.float32)
            vals = out_c
        else:
            valid_c = np.isfinite(vals)
            vals = np.where(valid_c, np.clip(vals, np.float32(-50.0), np.float32(80.0)), np.nan).astype(np.float32)

        lst_da = xr.DataArray(
            vals,
            dims=src_da.dims,
            coords=src_da.coords,
            name="LST",
            attrs={
                "index_name": "LST",
                "formula": spec["formula"],
                "bands": "lwir11 (ST_B10)",
                "units": "°C",
                "long_name": "Land Surface Temperature [°C]",
                "sensor": ds.attrs.get("sensor", "landsat-c2-l2"),
                "scene_id": ds.attrs.get("scene_id", ""),
                "datetime": ds.attrs.get("datetime", ""),
                "mgrs_tile": ds.attrs.get("mgrs_tile", ""),
                "resolution_m": ds.attrs.get("resolution_m", 30.0),
            },
        )
        crs = ds.rio.crs or src_da.rio.crs
        if crs is not None:
            lst_da = lst_da.rio.write_crs(crs, inplace=False)
        lst_da = lst_da.rio.write_nodata(np.nan, inplace=False)
        return lst_da

    req_bands = tuple(spec["bands"])
    missing = [b for b in req_bands if b not in ds.data_vars]
    if missing:
        raise KeyError(
            f"Dataset is missing required bands {missing} for index {key}. "
            f"Available bands: {list(ds.data_vars)}."
        )

    ref_da = ds[req_bands[0]]
    out = np.full(ref_da.shape, np.nan, dtype=np.float32)

    if key == "EVI":
        nir = ds["B08"].values.astype(np.float32, copy=False)
        red = ds["B04"].values.astype(np.float32, copy=False)
        blue = ds["B02"].values.astype(np.float32, copy=False)
        num = np.float32(2.5) * (nir - red)
        denom = nir + np.float32(6.0) * red - np.float32(7.5) * blue + np.float32(1.0)
        valid = np.isfinite(num) & np.isfinite(denom) & (np.abs(denom) > np.float32(1e-6))
        np.divide(num, denom, out=out, where=valid)
        np.clip(out, np.float32(-1.0), np.float32(1.0), out=out)
    elif key == "SAVI":
        nir = ds["B08"].values.astype(np.float32, copy=False)
        red = ds["B04"].values.astype(np.float32, copy=False)
        num = np.float32(1.5) * (nir - red)
        denom = nir + red + np.float32(0.5)
        valid = np.isfinite(num) & np.isfinite(denom) & (np.abs(denom) > np.float32(1e-6))
        np.divide(num, denom, out=out, where=valid)
        np.clip(out, np.float32(-1.0), np.float32(1.0), out=out)
    elif key == "BSI":
        swir1 = ds["B11"].values.astype(np.float32, copy=False)
        red = ds["B04"].values.astype(np.float32, copy=False)
        nir = ds["B08"].values.astype(np.float32, copy=False)
        blue = ds["B02"].values.astype(np.float32, copy=False)
        num = (swir1 + red) - (nir + blue)
        denom = (swir1 + red) + (nir + blue)
        valid = np.isfinite(num) & np.isfinite(denom) & (np.abs(denom) > np.float32(1e-6))
        np.divide(num, denom, out=out, where=valid)
        np.clip(out, np.float32(-1.0), np.float32(1.0), out=out)
    elif key == "BAI":
        red = ds["B04"].values.astype(np.float32, copy=False)
        nir = ds["B08"].values.astype(np.float32, copy=False)
        denom = (np.float32(0.1) - red) ** 2 + (np.float32(0.06) - nir) ** 2
        valid = np.isfinite(denom) & (denom > np.float32(1e-6))
        np.divide(np.float32(1.0), denom, out=out, where=valid)
        np.clip(out, np.float32(0.0), np.float32(200.0), out=out)
    else:
        b1, b2 = req_bands[0], req_bands[1]
        arr1 = ds[b1].values.astype(np.float32, copy=False)
        arr2 = ds[b2].values.astype(np.float32, copy=False)
        num = arr1 - arr2
        denom = arr1 + arr2
        valid = np.isfinite(num) & np.isfinite(denom) & (denom != 0.0)
        np.divide(num, denom, out=out, where=valid)
        np.clip(out, np.float32(-1.0), np.float32(1.0), out=out)

    index_da = xr.DataArray(
        out,
        dims=ref_da.dims,
        coords=ref_da.coords,
        name=key,
        attrs={
            "index_name": key,
            "formula": spec["formula"],
            "bands": ", ".join(req_bands),
            "units": str(spec.get("units", "dimensionless")),
            "long_name": f"{key}",
            "sensor": ds.attrs.get("sensor", ""),
            "scene_id": ds.attrs.get("scene_id", ""),
            "datetime": ds.attrs.get("datetime", ""),
            "mgrs_tile": ds.attrs.get("mgrs_tile", ""),
            "resolution_m": ds.attrs.get("resolution_m", 10.0),
        },
    )

    crs = ds.rio.crs or ref_da.rio.crs
    if crs is not None:
        index_da = index_da.rio.write_crs(crs, inplace=False)
    index_da = index_da.rio.write_nodata(np.nan, inplace=False)
    return index_da


compute_spectral_index = compute_index


def get_supported_indices_for_sensor(sensor: str) -> list[str]:
    """Return ordered list of spectral index codes supported by ``sensor``."""
    s_norm = str(sensor or "sentinel-2-l2a").strip().lower()
    if s_norm == "modis-11a2-061":
        return ["LST"]
    if s_norm == "modis-09a1-061":
        return [
            code
            for code, meta in SPECTRAL_INDICES_META.items()
            if not meta.get("s2_only") and not meta.get("landsat_only")
        ]
    return [
        code
        for code, meta in SPECTRAL_INDICES_META.items()
        if s_norm in tuple(meta.get("supported_sensors") or ("sentinel-2-l2a", "landsat-c2-l2"))
    ]


def extract_sar_metric(ds: xr.Dataset, metric_name: str) -> xr.DataArray:
    """Extract a Sentinel-1 SAR decibel layer: ``VV``, ``VH``, or ``VV_VH`` (VV - VH in dB)."""
    key = metric_name.strip().upper()
    if key not in SAR_METRICS:
        raise ValueError(
            f"Unsupported SAR metric {metric_name!r}. "
            f"Choose one of: {', '.join(SAR_METRICS.keys())}."
        )

    crs = ds.rio.crs
    if key in ("VV", "VH"):
        if key not in ds.data_vars:
            raise KeyError(f"SAR dataset missing polarization {key}. Available: {list(ds.data_vars)}")
        da = ds[key].astype(np.float32, copy=False)
        da = da.assign_attrs(
            {
                **dict(da.attrs),
                "index_name": key,
                "units": "dB",
                "sensor": ds.attrs.get("sensor", "sentinel-1-rtc"),
                "scene_id": ds.attrs.get("scene_id", ""),
                "datetime": ds.attrs.get("datetime", ""),
                "mgrs_tile": ds.attrs.get("mgrs_tile", ""),
                "resolution_m": ds.attrs.get("resolution_m", 10.0),
            }
        )
        if crs is not None:
            da = da.rio.write_crs(crs, inplace=False)
        da = da.rio.write_nodata(np.nan, inplace=False)
        return da

    # VV / VH linear ratio in dB equals VV_dB - VH_dB
    if "VV" not in ds.data_vars or "VH" not in ds.data_vars:
        raise KeyError(f"VV_VH metric requires both VV and VH. Available: {list(ds.data_vars)}")

    vv = ds["VV"].values.astype(np.float32, copy=False)
    vh = ds["VH"].values.astype(np.float32, copy=False)
    valid = np.isfinite(vv) & np.isfinite(vh)
    ratio_db = np.full(vv.shape, np.nan, dtype=np.float32)
    np.subtract(vv, vh, out=ratio_db, where=valid)

    ratio_da = xr.DataArray(
        ratio_db,
        dims=ds["VV"].dims,
        coords=ds["VV"].coords,
        name="VV_VH",
        attrs={
            "index_name": "VV_VH",
            "formula": "VV[dB] - VH[dB]",
            "units": "dB",
            "long_name": "Sentinel-1 VV/VH Backscatter Ratio [dB]",
            "sensor": ds.attrs.get("sensor", "sentinel-1-rtc"),
            "scene_id": ds.attrs.get("scene_id", ""),
            "datetime": ds.attrs.get("datetime", ""),
            "mgrs_tile": ds.attrs.get("mgrs_tile", ""),
            "resolution_m": ds.attrs.get("resolution_m", 10.0),
        },
    )
    if crs is not None:
        ratio_da = ratio_da.rio.write_crs(crs, inplace=False)
    ratio_da = ratio_da.rio.write_nodata(np.nan, inplace=False)
    return ratio_da


def compute_sar_water_mask(
    ds: xr.Dataset,
    threshold_db: float = -16.0,
    polarization: str = "VV",
    resolution_m: float = 10.0,
) -> tuple[xr.DataArray, dict[str, float | int]]:
    """Classify smooth water / flood surfaces from SAR backscatter (``pol < threshold_db``).

    Returns ``(mask_da, stats)`` where ``mask_da`` is a ``uint8`` georeferenced
    DataArray (`1` = Water / Flood, `0` = Land / Other, `255` = NoData) ready
    for visualization and GeoTIFF export, and ``stats`` includes exact water
    area in hectares (``water_area_ha``).
    """
    pol_key = polarization.strip().upper()
    if pol_key not in ds.data_vars:
        raise KeyError(f"Polarization {pol_key!r} not in dataset. Available: {list(ds.data_vars)}")
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be strictly positive, got {resolution_m}.")

    pol_da = ds[pol_key]
    vals = pol_da.values.astype(np.float32, copy=False)
    valid = np.isfinite(vals)
    is_water = valid & (vals < np.float32(threshold_db))

    mask_arr = np.full(vals.shape, 255, dtype=np.uint8)
    mask_arr[valid] = 0
    mask_arr[is_water] = 1

    total_pixels = int(vals.size)
    valid_pixels = int(np.count_nonzero(valid))
    water_pixels = int(np.count_nonzero(is_water))
    px_area_ha = (float(resolution_m) ** 2) / 10_000.0

    valid_area_ha = valid_pixels * px_area_ha
    water_area_ha = water_pixels * px_area_ha
    valid_pct = (100.0 * valid_pixels / total_pixels) if total_pixels > 0 else 0.0
    water_pct = (100.0 * water_pixels / valid_pixels) if valid_pixels > 0 else 0.0

    mask_da = xr.DataArray(
        mask_arr,
        dims=pol_da.dims,
        coords=pol_da.coords,
        name=f"SAR_WATER_{pol_key}",
        attrs={
            "long_name": f"Sentinel-1 Water/Flood Mask ({pol_key} < {threshold_db:.1f} dB)",
            "threshold_db": float(threshold_db),
            "polarization": pol_key,
            "classes": "0=Land, 1=Water/Flood, 255=NoData",
            "resolution_m": float(resolution_m),
            "scene_id": ds.attrs.get("scene_id", ""),
            "datetime": ds.attrs.get("datetime", ""),
        },
    )
    crs = ds.rio.crs or pol_da.rio.crs
    if crs is not None:
        mask_da = mask_da.rio.write_crs(crs, inplace=False)
    mask_da = mask_da.rio.write_nodata(255, inplace=False)

    stats: dict[str, float | int] = {
        "threshold_db": float(threshold_db),
        "water_pixels": water_pixels,
        "water_area_ha": round(water_area_ha, 3),
        "water_pct": round(water_pct, 2),
        "valid_pixels": valid_pixels,
        "valid_area_ha": round(valid_area_ha, 3),
        "valid_pct": round(valid_pct, 2),
    }
    return mask_da, stats


def compute_index_stats(
    index_da: xr.DataArray,
    resolution_m: float,
) -> dict[str, float | int]:
    """Compute descriptive statistics and exact valid area in hectares for an index or SAR raster."""
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be strictly positive, got {resolution_m}.")

    vals = index_da.values
    total_pixels = int(vals.size)
    valid_vals = vals[np.isfinite(vals)]
    valid_pixels = int(valid_vals.size)
    valid_pct = float(100.0 * valid_pixels / total_pixels) if total_pixels > 0 else 0.0
    valid_area_ha = float(valid_pixels * (float(resolution_m) ** 2) / 10_000.0)

    if valid_pixels == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "valid_pixels": 0,
            "valid_pct": 0.0,
            "valid_area_ha": 0.0,
        }

    p10, median, p90 = np.percentile(valid_vals, [10.0, 50.0, 90.0])
    return {
        "min": float(np.min(valid_vals)),
        "max": float(np.max(valid_vals)),
        "mean": float(np.mean(valid_vals)),
        "median": float(median),
        "std": float(np.std(valid_vals)),
        "p10": float(p10),
        "p90": float(p90),
        "valid_pixels": valid_pixels,
        "valid_pct": round(valid_pct, 2),
        "valid_area_ha": round(valid_area_ha, 3),
    }


def extract_spectral_signature(
    ds: xr.Dataset,
    x_coord: float,
    y_coord: float,
) -> pd.DataFrame:
    """Extract surface reflectance ``[0, 1]`` at the nearest pixel to ``(x_coord, y_coord)``.

    Works for both Sentinel-2 L2A and harmonized Landsat C2 L2 Datasets across
    ``B02 (490 nm)``, ``B03 (560 nm)``, ``B04 (665 nm)``, ``B08 (842 nm)``,
    ``B11 (1610 nm)``, and ``B12 (2190 nm)``.
    """
    x_vals = np.asarray(ds.coords["x"].values, dtype=np.float64)
    y_vals = np.asarray(ds.coords["y"].values, dtype=np.float64)
    if x_vals.size == 0 or y_vals.size == 0:
        raise ValueError("Dataset has empty spatial coordinates.")

    ix = int(np.argmin(np.abs(x_vals - float(x_coord))))
    iy = int(np.argmin(np.abs(y_vals - float(y_coord))))
    actual_x = float(x_vals[ix])
    actual_y = float(y_vals[iy])

    rows: list[dict[str, Any]] = []
    for band_code, wl_nm, band_name in OPTICAL_SIGNATURE_BANDS:
        if band_code not in ds.data_vars:
            continue
        raw_val = float(ds[band_code].isel(y=iy, x=ix).values)
        rows.append(
            {
                "band": band_code,
                "wavelength_nm": int(wl_nm),
                "band_name": band_name,
                "reflectance": raw_val if np.isfinite(raw_val) else np.nan,
            }
        )

    df = pd.DataFrame(rows, columns=["band", "wavelength_nm", "band_name", "reflectance"])
    df.attrs["actual_x"] = actual_x
    df.attrs["actual_y"] = actual_y
    df.attrs["ix"] = ix
    df.attrs["iy"] = iy
    return df


def compute_quick_land_cover(
    ds: xr.Dataset,
    resolution_m: float,
) -> tuple[xr.DataArray, pd.DataFrame]:
    """Classify an optical (cloud-masked) scene into 4 intuitive physical land-cover classes.

    Classes:
    - ``1`` (``water``): ``NDWI > 0.1``
    - ``2`` (``built_bare``): ``NDWI <= 0.1`` and ``NDVI < 0.2``
    - ``3`` (``sparse_veg``): ``NDWI <= 0.1`` and ``0.2 <= NDVI < 0.5``
    - ``4`` (``dense_veg``): ``NDWI <= 0.1`` and ``NDVI >= 0.5``
    - ``255``: NoData / Cloud-masked pixels
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be strictly positive, got {resolution_m}.")

    ndvi_da = compute_index(ds, "NDVI")
    ndwi_da = compute_index(ds, "NDWI")

    ndvi = ndvi_da.values
    ndwi = ndwi_da.values
    valid = np.isfinite(ndvi) & np.isfinite(ndwi)

    lc_arr = np.full(ndvi.shape, 255, dtype=np.uint8)
    is_water = valid & (ndwi > 0.1)
    not_water = valid & (~is_water)
    is_built = not_water & (ndvi < 0.2)
    is_sparse = not_water & (ndvi >= 0.2) & (ndvi < 0.5)
    is_dense = not_water & (ndvi >= 0.5)

    lc_arr[is_water] = 1
    lc_arr[is_built] = 2
    lc_arr[is_sparse] = 3
    lc_arr[is_dense] = 4

    valid_pixels = int(np.count_nonzero(valid))
    px_area_ha = (float(resolution_m) ** 2) / 10_000.0

    counts = {
        "water": int(np.count_nonzero(is_water)),
        "built_bare": int(np.count_nonzero(is_built)),
        "sparse_veg": int(np.count_nonzero(is_sparse)),
        "dense_veg": int(np.count_nonzero(is_dense)),
    }

    rows: list[dict[str, Any]] = []
    for spec in LAND_COVER_CLASSES:
        ckey = spec["class_key"]
        px = counts[ckey]
        area_ha = px * px_area_ha
        share_pct = (100.0 * px / valid_pixels) if valid_pixels > 0 else 0.0
        rows.append(
            {
                "class_key": ckey,
                "class_id": spec["class_id"],
                "rule": spec["rule"],
                "color": spec["color"],
                "pixels": px,
                "area_ha": round(area_ha, 3),
                "share_pct": round(share_pct, 2),
            }
        )

    lc_da = xr.DataArray(
        lc_arr,
        dims=ndvi_da.dims,
        coords=ndvi_da.coords,
        name="LAND_COVER",
        attrs={
            "long_name": "Quick Physical Land-Cover Classification",
            "classes": "1=Water, 2=Built-up/Bare, 3=Sparse Veg, 4=Dense Veg, 255=NoData",
            "resolution_m": float(resolution_m),
            "scene_id": ds.attrs.get("scene_id", ""),
            "datetime": ds.attrs.get("datetime", ""),
        },
    )
    crs = ds.rio.crs or ndvi_da.rio.crs
    if crs is not None:
        lc_da = lc_da.rio.write_crs(crs, inplace=False)
    lc_da = lc_da.rio.write_nodata(255, inplace=False)

    summary_df = pd.DataFrame(rows)
    summary_df.attrs["valid_pixels"] = valid_pixels
    summary_df.attrs["valid_area_ha"] = round(valid_pixels * px_area_ha, 3)
    return lc_da, summary_df


def compute_snow_glacier_stats(
    ds: xr.Dataset,
    dem_da: xr.DataArray | None = None,
    ndsi_thresh: float = 0.40,
    nir_thresh: float = 0.11,
    resolution_m: float | None = None,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Classify Snow / Ice / Glacier extent via ESA/NASA NDSI + NIR mask and optional DEM snowline.

    Uses the standard ESA/NASA optical cryosphere rule:
    ``is_snow = (NDSI >= ndsi_thresh) & (B08 > nir_thresh)`` (when ``B08`` is present),
    where the ``NIR > 0.11`` condition cleanly separates bright snow/ice from dark water bodies.

    When ``dem_da`` (e.g., Copernicus DEM 30 m) is supplied, computes the glaciological
    snowline altitude (10th percentile elevation of snow/ice pixels), plus median and max
    snow elevations [m a.s.l.].

    Returns
    -------
    tuple[xr.DataArray, dict[str, Any]]
        ``(snow_mask_da, stats)`` where ``snow_mask_da`` is a ``uint8`` georeferenced raster
        (``1`` = Snow/Ice/Glacier, ``0`` = Snow-Free Land/Water, ``255`` = NoData/Cloud)
        and ``stats`` contains exact ``snow_area_ha``, ``snow_area_km2``, ``snow_pct``,
        and DEM elevation statistics (``snowline_p10_m``, ``snow_median_elev_m``, ``snow_max_elev_m``).
    """
    ndsi_da = compute_index(ds, "NDSI")
    ndsi_vals = ndsi_da.values.astype(np.float32, copy=False)

    # Infer exact metric pixel size from argument, dataset attrs, or coordinates
    if resolution_m is not None and float(resolution_m) > 0:
        res_m = float(resolution_m)
    else:
        attr_res = ds.attrs.get("resolution_m")
        if attr_res is not None and float(attr_res) > 0:
            res_m = float(attr_res)
        else:
            x_c = np.asarray(ndsi_da.coords["x"].values, dtype=np.float64)
            res_m = float(abs(np.median(np.diff(x_c)))) if x_c.size > 1 else 10.0

    has_nir = "B08" in ds.data_vars
    if has_nir:
        nir_vals = ds["B08"].values.astype(np.float32, copy=False)
        valid = np.isfinite(ndsi_vals) & np.isfinite(nir_vals)
        is_snow = valid & (ndsi_vals >= np.float32(ndsi_thresh)) & (nir_vals > np.float32(nir_thresh))
    else:
        valid = np.isfinite(ndsi_vals)
        is_snow = valid & (ndsi_vals >= np.float32(ndsi_thresh))

    mask_arr = np.full(ndsi_vals.shape, 255, dtype=np.uint8)
    mask_arr[valid] = 0
    mask_arr[is_snow] = 1

    total_pixels = int(ndsi_vals.size)
    valid_pixels = int(np.count_nonzero(valid))
    snow_pixels = int(np.count_nonzero(is_snow))
    px_area_ha = (res_m ** 2) / 10_000.0
    snow_area_ha = snow_pixels * px_area_ha
    snow_area_km2 = snow_area_ha / 100.0
    valid_area_ha = valid_pixels * px_area_ha
    valid_pct = (100.0 * valid_pixels / total_pixels) if total_pixels > 0 else 0.0
    snow_pct = (100.0 * snow_pixels / valid_pixels) if valid_pixels > 0 else 0.0

    # Snowline elevation via Copernicus DEM 30 m (if provided)
    dem_available = False
    snowline_p10_m: float | None = None
    snow_median_elev_m: float | None = None
    snow_max_elev_m: float | None = None
    snow_min_elev_m: float | None = None

    if dem_da is not None and snow_pixels > 0:
        try:
            from rasterio.enums import Resampling

            dem_aligned = dem_da
            if dem_aligned.shape != ndsi_da.shape or (
                dem_aligned.rio.crs is not None
                and ndsi_da.rio.crs is not None
                and dem_aligned.rio.crs != ndsi_da.rio.crs
            ):
                dem_aligned = dem_aligned.rio.reproject_match(
                    ndsi_da,
                    resampling=Resampling.bilinear,
                    nodata=np.nan,
                )
            dem_vals = dem_aligned.values.astype(np.float32, copy=False)
            snow_elevs = dem_vals[is_snow & np.isfinite(dem_vals) & (dem_vals > -500.0)]
            if snow_elevs.size > 0:
                p10, med, mx, mn = (
                    float(np.percentile(snow_elevs, 10.0)),
                    float(np.median(snow_elevs)),
                    float(np.max(snow_elevs)),
                    float(np.min(snow_elevs)),
                )
                dem_available = True
                snowline_p10_m = round(p10, 1)
                snow_median_elev_m = round(med, 1)
                snow_max_elev_m = round(mx, 1)
                snow_min_elev_m = round(mn, 1)
        except Exception:
            pass

    rule_str = (
        f"NDSI >= {ndsi_thresh:.2f} & B08 > {nir_thresh:.2f}"
        if has_nir
        else f"NDSI >= {ndsi_thresh:.2f}"
    )
    snow_mask_da = xr.DataArray(
        mask_arr,
        dims=ndsi_da.dims,
        coords=ndsi_da.coords,
        name="SNOW_ICE_MASK",
        attrs={
            "long_name": f"Snow / Ice / Glacier Mask ({rule_str})",
            "ndsi_thresh": float(ndsi_thresh),
            "nir_thresh": float(nir_thresh),
            "nir_water_filter": bool(has_nir),
            "classes": "0=Snow-Free, 1=Snow/Ice/Glacier, 255=NoData",
            "resolution_m": float(res_m),
            "scene_id": ds.attrs.get("scene_id", ""),
            "datetime": ds.attrs.get("datetime", ""),
        },
    )
    crs = ds.rio.crs or ndsi_da.rio.crs
    if crs is not None:
        snow_mask_da = snow_mask_da.rio.write_crs(crs, inplace=False)
    snow_mask_da = snow_mask_da.rio.write_nodata(255, inplace=False)

    stats: dict[str, Any] = {
        "ndsi_thresh": float(ndsi_thresh),
        "nir_thresh": float(nir_thresh),
        "nir_water_filter": bool(has_nir),
        "resolution_m": float(res_m),
        "snow_pixels": snow_pixels,
        "snow_area_ha": round(snow_area_ha, 3),
        "snow_area_km2": round(snow_area_km2, 4),
        "snow_pct": round(snow_pct, 2),
        "valid_pixels": valid_pixels,
        "valid_area_ha": round(valid_area_ha, 3),
        "valid_pct": round(valid_pct, 2),
        "dem_available": dem_available,
        "snowline_p10_m": snowline_p10_m,
        "snow_median_elev_m": snow_median_elev_m,
        "snow_max_elev_m": snow_max_elev_m,
        "snow_min_elev_m": snow_min_elev_m,
    }
    return snow_mask_da, stats


# ---------------------------------------------------------------------------
# Water-Aware GLIMS Glacier Change Detection (Terminus -> Proglacial Lake)
# ---------------------------------------------------------------------------
GLACIER_CHANGE_RGBA_MAP: dict[int, tuple[int, int, int, int]] = {
    1: (220, 38, 38, 225),   # Crimson #DC2626 — Ice -> Rock/Moraine (Retreat)
    2: (147, 51, 234, 235),  # Vivid Purple #9333EA — Ice -> New Proglacial Lake
    3: (37, 99, 235, 220),   # Royal Blue #2563EB — Snow/Ice Gain
}


def _detect_glims_water_mask(
    ndsi_arr: np.ndarray,
    ndwi_arr: np.ndarray | None = None,
    nir_arr: np.ndarray | None = None,
) -> np.ndarray:
    """Identify water / proglacial lake pixels that mimic snow/ice in raw NDSI.

    Water strongly absorbs NIR (``NIR < 0.11``) and exhibits positive ``NDWI > 0.10``,
    whereas bright snow/ice reflects NIR strongly.
    """
    fin_ndsi = np.isfinite(ndsi_arr)
    water = np.zeros(ndsi_arr.shape, dtype=bool)
    if ndwi_arr is not None and nir_arr is not None:
        fin_w = np.isfinite(ndwi_arr)
        fin_n = np.isfinite(nir_arr)
        water = fin_ndsi & (
            (fin_w & (ndwi_arr > 0.15))
            | (fin_w & fin_n & (ndsi_arr > 0.20) & (ndwi_arr > 0.10) & (nir_arr < 0.15))
            | (fin_n & (ndsi_arr > 0.20) & (nir_arr < 0.11))
        )
    elif ndwi_arr is not None:
        water = fin_ndsi & np.isfinite(ndwi_arr) & (ndwi_arr > 0.15)
    elif nir_arr is not None:
        water = fin_ndsi & np.isfinite(nir_arr) & (ndsi_arr > 0.20) & (nir_arr < 0.11)
    return water


def compute_water_cleaned_ndsi(
    ndsi: np.ndarray | xr.DataArray,
    ndwi: np.ndarray | xr.DataArray | None = None,
    nir: np.ndarray | xr.DataArray | None = None,
) -> np.ndarray:
    """Return a float32 water-cleaned NDSI ice score where proglacial lake water is set to ``-0.5``.

    Eliminates the classic glacier-terminus blind spot where Ice (``+0.65``) melting into a
    proglacial lake (``+0.55`` raw NDSI) previously produced ``Δ NDSI ≈ 0``.
    """
    ndsi_arr = np.asarray(ndsi.values if isinstance(ndsi, xr.DataArray) else ndsi, dtype=np.float32)
    ndwi_arr = (
        np.asarray(ndwi.values if isinstance(ndwi, xr.DataArray) else ndwi, dtype=np.float32)
        if ndwi is not None
        else None
    )
    nir_arr = (
        np.asarray(nir.values if isinstance(nir, xr.DataArray) else nir, dtype=np.float32)
        if nir is not None
        else None
    )
    is_water = _detect_glims_water_mask(ndsi_arr, ndwi_arr=ndwi_arr, nir_arr=nir_arr)
    return np.where(is_water, np.float32(-0.5), ndsi_arr).astype(np.float32)


def detect_new_proglacial_lake_mask(
    ndwi_t1: np.ndarray | xr.DataArray | None,
    ndwi_t2: np.ndarray | xr.DataArray | None,
) -> np.ndarray | None:
    """Detect newly formed proglacial lakes directly via NDWI water gain.

    Catches turbid glacial-milk lakes and ``Debris-Covered Ice -> Proglacial Lake``
    transitions without requiring ultra-low NIR reflectance.
    """
    if ndwi_t1 is None or ndwi_t2 is None:
        return None
    w1 = np.asarray(ndwi_t1.values if isinstance(ndwi_t1, xr.DataArray) else ndwi_t1, dtype=np.float32)
    w2 = np.asarray(ndwi_t2.values if isinstance(ndwi_t2, xr.DataArray) else ndwi_t2, dtype=np.float32)
    valid = np.isfinite(w1) & np.isfinite(w2)
    return valid & (w2 > np.float32(0.02)) & ((w2 - w1) > np.float32(0.12)) & (w1 < np.float32(0.15))


def classify_glacier_change(
    ndsi_t1: np.ndarray | xr.DataArray,
    ndsi_t2: np.ndarray | xr.DataArray,
    *,
    ndwi_t1: np.ndarray | xr.DataArray | None = None,
    ndwi_t2: np.ndarray | xr.DataArray | None = None,
    nir_t1: np.ndarray | xr.DataArray | None = None,
    nir_t2: np.ndarray | xr.DataArray | None = None,
    resolution_m: float = 30.0,
    ice_thresh: float = 0.40,
) -> tuple[np.ndarray, dict[str, float]]:
    """Classify pixel-wise glacier retreat into categorical classes and exact metric areas.

    Class codes (``uint8``):
    - ``0``: Unchanged / NoData (transparent)
    - ``1``: Ice -> Rock / Moraine (Retreat on land)
    - ``2``: Ice / Debris-Ice -> New Proglacial Lake (GLIMS water OR ΔNDWI water gain)
    - ``3``: Snow / Ice Gain
    """
    n1 = np.asarray(ndsi_t1.values if isinstance(ndsi_t1, xr.DataArray) else ndsi_t1, dtype=np.float32)
    n2 = np.asarray(ndsi_t2.values if isinstance(ndsi_t2, xr.DataArray) else ndsi_t2, dtype=np.float32)
    w1 = np.asarray(ndwi_t1.values if isinstance(ndwi_t1, xr.DataArray) else ndwi_t1, dtype=np.float32) if ndwi_t1 is not None else None
    w2 = np.asarray(ndwi_t2.values if isinstance(ndwi_t2, xr.DataArray) else ndwi_t2, dtype=np.float32) if ndwi_t2 is not None else None
    r1 = np.asarray(nir_t1.values if isinstance(nir_t1, xr.DataArray) else nir_t1, dtype=np.float32) if nir_t1 is not None else None
    r2 = np.asarray(nir_t2.values if isinstance(nir_t2, xr.DataArray) else nir_t2, dtype=np.float32) if nir_t2 is not None else None

    valid = np.isfinite(n1) & np.isfinite(n2)
    water_1 = _detect_glims_water_mask(n1, ndwi_arr=w1, nir_arr=r1)
    water_2 = _detect_glims_water_mask(n2, ndwi_arr=w2, nir_arr=r2)
    new_lake_ndwi = detect_new_proglacial_lake_mask(w1, w2)

    ice_1 = valid & (n1 >= np.float32(ice_thresh)) & (~water_1)
    ice_2 = valid & (n2 >= np.float32(ice_thresh)) & (~water_2)

    cls = np.zeros(n1.shape, dtype=np.uint8)
    loss_lake = ice_1 & water_2
    if new_lake_ndwi is not None:
        loss_lake = loss_lake | (valid & new_lake_ndwi)
    loss_land = ice_1 & (~ice_2) & (~water_2) & (~loss_lake)
    gain_ice = valid & (~ice_1) & ice_2 & (~loss_lake)

    cls[loss_land] = 1
    cls[loss_lake] = 2
    cls[gain_ice] = 3

    px_ha = (float(resolution_m) ** 2) / 10_000.0
    land_ha = round(int(np.count_nonzero(loss_land)) * px_ha, 2)
    lake_ha = round(int(np.count_nonzero(loss_lake)) * px_ha, 2)
    gain_ha = round(int(np.count_nonzero(gain_ice)) * px_ha, 2)
    stats = {
        "ice_loss_land_ha": land_ha,
        "new_proglacial_lake_ha": lake_ha,
        "total_ice_loss_ha": round(land_ha + lake_ha, 2),
        "ice_gain_ha": gain_ha,
    }
    return cls, stats



