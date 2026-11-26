"""Optional Google Earth Engine (GEE) server-side backend for continental-scale EO analyses.

CRITICAL DESIGN RULE:
    Google Earth Engine is 100% OPTIONAL. This module safely guards ``import ee`` and all
    authentication calls so users without a GEE account or credentials continue using
    Microsoft Planetary Computer STAC / MODIS seamlessly with zero errors or crashes.

Supported Authentication Paths (checked in order):
    1. Explicit / Session-Level Service Account JSON + Project ID (Sidebar expander,
       stored ONLY in ``st.session_state``, never written to disk or GitHub).
    2. Streamlit Secrets / Environment Service Account (``GEE_PROJECT_ID`` +
       ``GEE_SERVICE_ACCOUNT_JSON``, or ``GEE_SERVICE_ACCOUNT_EMAIL`` + ``GEE_PRIVATE_KEY``).
    3. Local Desktop Auth (existing ``~/.config/earthengine/credentials`` via ``ee.Initialize``).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import requests
from PIL import Image

try:
    import ee  # type: ignore[import-untyped]

    _EE_IMPORT_OK = True
    _EE_IMPORT_ERR = ""
except Exception as _exc:  # pragma: no cover - optional dependency
    ee = None  # type: ignore[assignment]
    _EE_IMPORT_OK = False
    _EE_IMPORT_ERR = str(_exc)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level connection state (safe, lightweight, never persists secrets)
# ---------------------------------------------------------------------------
_GEE_AUTO_ATTEMPTED: bool = False
_GEE_STATE: dict[str, Any] = {
    "connected": False,
    "project_id": "",
    "mode": "none",
    "error": "",
}

# ---------------------------------------------------------------------------
# Supported Server-Side Spectral Indices & Visualization Palettes
# ---------------------------------------------------------------------------
GEE_SUPPORTED_INDICES: tuple[str, ...] = (
    "NDVI",
    "EVI",
    "SAVI",
    "NDMI",
    "NDWI",
    "MNDWI",
    "NDSI",
    "NDBI",
    "BSI",
    "NBR",
    "BAI",
)

GEE_INDEX_VIS: dict[str, dict[str, Any]] = {
    "NDVI": {
        "min": -0.2,
        "max": 0.9,
        "palette": ["a50026", "d73027", "f46d43", "fdae61", "fee08b", "d9ef8b", "a6d96a", "66bd63", "1a9850", "006837"],
        "cmap": "RdYlGn",
        "delta_cmap": "RdYlGn",
        "delta_lim": 0.5,
    },
    "EVI": {
        "min": -0.2,
        "max": 0.9,
        "palette": ["a50026", "f46d43", "fee08b", "a6d96a", "1a9850", "006837"],
        "cmap": "RdYlGn",
        "delta_cmap": "RdYlGn",
        "delta_lim": 0.5,
    },
    "SAVI": {
        "min": -0.2,
        "max": 0.85,
        "palette": ["a50026", "f46d43", "fee08b", "a6d96a", "1a9850", "006837"],
        "cmap": "RdYlGn",
        "delta_cmap": "RdYlGn",
        "delta_lim": 0.5,
    },
    "NDMI": {
        "min": -0.6,
        "max": 0.6,
        "palette": ["8c510a", "d8b365", "f6e8c3", "c7eae5", "5ab4ac", "01665e"],
        "cmap": "BrBG",
        "delta_cmap": "BrBG",
        "delta_lim": 0.5,
    },
    "NDWI": {
        "min": -0.6,
        "max": 0.6,
        "palette": ["8c510a", "d8b365", "f5f5f5", "92c5de", "2166ac", "053061"],
        "cmap": "RdBu",
        "delta_cmap": "RdBu",
        "delta_lim": 0.5,
    },
    "MNDWI": {
        "min": -0.6,
        "max": 0.6,
        "palette": ["8c510a", "d8b365", "f5f5f5", "92c5de", "2166ac", "053061"],
        "cmap": "RdBu",
        "delta_cmap": "RdBu",
        "delta_lim": 0.5,
    },
    "NDSI": {
        "min": -0.4,
        "max": 0.9,
        "palette": ["7f3b08", "d8b365", "f7f7f7", "9ecae1", "4292c6", "084594"],
        "cmap": "Blues",
        "delta_cmap": "RdBu",
        "delta_lim": 0.6,
    },
    "NDBI": {
        "min": -0.5,
        "max": 0.5,
        "palette": ["1a9850", "a6d96a", "ffffbf", "fdae61", "d73027", "7f0000"],
        "cmap": "RdYlBu_r",
        "delta_cmap": "RdBu_r",
        "delta_lim": 0.5,
    },
    "BSI": {
        "min": -0.5,
        "max": 0.5,
        "palette": ["1a9850", "a6d96a", "f5f5f5", "dfc27d", "a6611a", "543005"],
        "cmap": "BrBG_r",
        "delta_cmap": "BrBG_r",
        "delta_lim": 0.5,
    },
    "NBR": {
        "min": -0.5,
        "max": 0.8,
        "palette": ["7f0000", "d73027", "fc8d59", "fee08b", "d9ef8b", "1a9850"],
        "cmap": "RdYlGn",
        "delta_cmap": "RdYlGn",
        "delta_lim": 0.5,
    },
    "BAI": {
        "min": 0.0,
        "max": 200.0,
        "palette": ["0d0887", "6a00a8", "b12a90", "e16462", "fca636", "f0f921"],
        "cmap": "inferno",
        "delta_cmap": "RdBu_r",
        "delta_lim": 80.0,
    },
}

GEE_DELTA_PALETTES: dict[str, list[str]] = {
    "RdYlGn": ["b2182b", "ef8a62", "fddbc7", "f7f7f7", "d9f0d3", "7fbf7b", "1b7837"],
    "RdBu": ["b2182b", "ef8a62", "fddbc7", "f7f7f7", "d1e5f0", "67a9cf", "2166ac"],
    "RdBu_r": ["2166ac", "67a9cf", "d1e5f0", "f7f7f7", "fddbc7", "ef8a62", "b2182b"],
    "BrBG": ["8c510a", "d8b365", "f6e8c3", "f5f5f5", "c7eae5", "5ab4ac", "01665e"],
    "BrBG_r": ["01665e", "5ab4ac", "c7eae5", "f5f5f5", "f6e8c3", "d8b365", "8c510a"],
}


def _read_secret_or_env(key: str) -> Any:
    """Safely read a secret from Streamlit ``st.secrets`` or environment variables."""
    try:
        import streamlit as st

        val = st.secrets.get(key, None)
        if val is not None:
            return val
    except Exception:
        try:
            import streamlit as st

            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    return os.environ.get(key)


def _normalize_sa_dict(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """Convert a Mapping/AttrDict into a clean dict with normalized PEM newlines in ``private_key``."""
    out = {str(k): v for k, v in data.items()}
    email = str(out.get("client_email") or "").strip()
    raw_pk = out.get("private_key")
    if not email or not raw_pk:
        return None
    pk_norm = str(raw_pk).replace("\\n", "\n").strip()
    if not pk_norm.endswith("\n"):
        pk_norm += "\n"
    out["client_email"] = email
    out["private_key"] = pk_norm
    if "type" not in out:
        out["type"] = "service_account"
    return out


def _parse_service_account_payload(
    raw_json: str | Mapping[str, Any] | None,
) -> tuple[str | None, str | None, str | None]:
    """Extract ``(client_email, json_str, project_id)`` from a service-account JSON string or Mapping/AttrDict."""
    if raw_json is None:
        return None, None, None
    data: dict[str, Any] | None = None
    if isinstance(raw_json, Mapping):
        data = _normalize_sa_dict(raw_json)
    elif isinstance(raw_json, str):
        txt = raw_json.strip()
        # Strip accidental triple-quote wrappers pasted from TOML examples
        for q in ("'''", '"""'):
            if txt.startswith(q) and txt.endswith(q) and len(txt) >= 6:
                txt = txt[3:-3].strip()
        if not txt:
            return None, None, None
        try:
            parsed = json.loads(txt)
        except json.JSONDecodeError:
            parsed = json.loads(txt, strict=False)
        if isinstance(parsed, Mapping):
            data = _normalize_sa_dict(parsed)
    if not isinstance(data, dict):
        return None, None, None
    email = str(data.get("client_email") or "").strip()
    if not email or not data.get("private_key"):
        return None, None, None
    proj = str(data.get("project_id") or "").strip() or None
    return email, json.dumps(data), proj


def _sanitize_project_id(raw_id: Any, fallback_id: Any = "") -> str:
    """Extract strictly the first clean GCP project identifier token; never leak JSON or private keys."""
    for candidate in (raw_id, fallback_id):
        if candidate is None:
            continue
        txt = str(candidate).strip()
        if not txt:
            continue
        # Take only the first line and first non-whitespace/non-quote token
        first_line = txt.splitlines()[0].strip()
        if "{" in first_line or "private_key" in first_line.lower() or "begin private" in first_line.lower():
            continue
        tokens = [tok for tok in re.split(r"""[\s"'`=,;{}]+""", first_line) if tok]
        if not tokens:
            continue
        tok = tokens[0].strip()
        if (
            "private_key" in tok.lower()
            or "{" in tok
            or "}" in tok
            or not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,62}$", tok)
        ):
            continue
        return tok[:48]
    return ""


def _load_secrets_credentials() -> tuple[dict[str, Any] | None, str]:
    """Read and normalize GEE service-account credentials and project ID from ``st.secrets`` / env.

    Supports:
        - ``GEE_SERVICE_ACCOUNT_JSON`` as a raw JSON string (including TOML ``'''...'''`` with ``\\n``)
          or as a parsed TOML table / Streamlit ``AttrDict`` (``Mapping``).
        - ``GEE_SERVICE_ACCOUNT_EMAIL`` + ``GEE_PRIVATE_KEY``.
        - ``GEE_PROJECT_ID`` (falling back to ``project_id`` inside the JSON itself).

    Returns ``(creds_dict_or_None, project_id_str)``.
    """
    raw_secret_proj = _read_secret_or_env("GEE_PROJECT_ID")
    raw_sa = _read_secret_or_env("GEE_SERVICE_ACCOUNT_JSON")
    if raw_sa is not None and raw_sa != "":
        try:
            email, json_str, sa_proj = _parse_service_account_payload(raw_sa)
            if email and json_str:
                creds_dict = json.loads(json_str)
                eff_proj = _sanitize_project_id(raw_secret_proj, fallback_id=sa_proj or creds_dict.get("project_id", ""))
                return creds_dict, eff_proj
        except Exception as exc:
            logger.warning("Failed to parse GEE_SERVICE_ACCOUNT_JSON from secrets: %s", exc)

    secret_proj = _sanitize_project_id(raw_secret_proj)
    sa_email = str(_read_secret_or_env("GEE_SERVICE_ACCOUNT_EMAIL") or "").strip()
    sa_key = str(_read_secret_or_env("GEE_PRIVATE_KEY") or "").strip()
    if sa_email and sa_key:
        creds_dict = _normalize_sa_dict(
            {
                "type": "service_account",
                "client_email": sa_email,
                "private_key": sa_key,
                "project_id": secret_proj,
            }
        )
        if creds_dict is not None:
            return creds_dict, secret_proj

    return None, secret_proj


def _has_local_ee_credentials() -> bool:
    """Return True if local ``earthengine authenticate`` credentials exist on disk."""
    cred_path = Path.home() / ".config" / "earthengine" / "credentials"
    return cred_path.is_file() and cred_path.stat().st_size > 10


def _configure_ee_deadline(timeout_ms: int = 90000) -> None:
    """Increase the Earth Engine server HTTP deadline (default 90 s) when supported by the SDK."""
    try:
        if ee is not None and hasattr(ee, "data") and hasattr(ee.data, "setDeadline"):
            ee.data.setDeadline(int(timeout_ms))
    except Exception:
        pass


def init_gee(
    project_id: str | None = None,
    service_account_json: str | Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Initialize Google Earth Engine using 3 optional authentication paths.

    Order of precedence:
        1. Explicit / session-level ``service_account_json`` + ``project_id``
        2. Streamlit secrets / env vars (``GEE_SERVICE_ACCOUNT_JSON`` or
           ``GEE_SERVICE_ACCOUNT_EMAIL`` + ``GEE_PRIVATE_KEY`` + ``GEE_PROJECT_ID``)
        3. Existing local desktop credentials (``~/.config/earthengine/credentials``)

    Never raises unhandled exceptions; returns a status dict with keys:
    ``connected``, ``project_id``, ``mode``, ``error``.
    """
    global _GEE_STATE, _GEE_AUTO_ATTEMPTED
    _GEE_AUTO_ATTEMPTED = True

    if _GEE_STATE["connected"] and not force and not service_account_json and not project_id:
        return dict(_GEE_STATE)

    if not _EE_IMPORT_OK or ee is None:
        _GEE_STATE = {
            "connected": False,
            "project_id": "",
            "mode": "unavailable",
            "error": f"earthengine-api not available: {_EE_IMPORT_ERR}",
        }
        return dict(_GEE_STATE)

    req_proj = _sanitize_project_id(project_id)
    sec_creds, secret_proj = _load_secrets_credentials()

    # Path 1: Explicit / session-level Service Account JSON
    if service_account_json:
        try:
            email, json_str, sa_proj = _parse_service_account_payload(service_account_json)
            if not email or not json_str:
                raise ValueError("Invalid Service Account JSON (missing client_email or private_key).")
            eff_proj = _sanitize_project_id(req_proj or sa_proj or secret_proj)
            creds = ee.ServiceAccountCredentials(email, key_data=json_str)
            if eff_proj:
                ee.Initialize(credentials=creds, project=eff_proj)
            else:
                ee.Initialize(credentials=creds)
            _configure_ee_deadline(90000)
            _GEE_STATE = {
                "connected": True,
                "project_id": eff_proj or "service-account",
                "mode": "session_service_account",
                "error": "",
            }
            return dict(_GEE_STATE)
        except Exception as exc:
            _GEE_STATE = {
                "connected": False,
                "project_id": req_proj,
                "mode": "none",
                "error": str(exc),
            }
            return dict(_GEE_STATE)

    # Path 2: Streamlit Secrets / Env Service Account
    if sec_creds is not None:
        try:
            email = str(sec_creds["client_email"])
            json_str = json.dumps(sec_creds)
            eff_proj = _sanitize_project_id(req_proj or secret_proj, fallback_id=sec_creds.get("project_id", ""))
            creds = ee.ServiceAccountCredentials(email, key_data=json_str)
            if eff_proj:
                ee.Initialize(credentials=creds, project=eff_proj)
            else:
                ee.Initialize(credentials=creds)
            _configure_ee_deadline(90000)
            _GEE_STATE = {
                "connected": True,
                "project_id": eff_proj or "secrets-service-account",
                "mode": "secrets_service_account",
                "error": "",
            }
            return dict(_GEE_STATE)
        except Exception as exc:
            logger.warning("GEE secrets service-account auth failed: %s", exc)
            _GEE_STATE = {
                "connected": False,
                "project_id": _sanitize_project_id(req_proj or secret_proj),
                "mode": "none",
                "error": str(exc),
            }
            return dict(_GEE_STATE)

    # Path 3: Local Desktop Auth (only if local credentials file actually exists)
    if _has_local_ee_credentials():
        try:
            eff_proj = _sanitize_project_id(req_proj or secret_proj)
            if eff_proj:
                ee.Initialize(project=eff_proj)
            else:
                ee.Initialize()
            _configure_ee_deadline(90000)
            _GEE_STATE = {
                "connected": True,
                "project_id": eff_proj or "local-desktop",
                "mode": "local_desktop",
                "error": "",
            }
            return dict(_GEE_STATE)
        except Exception as exc:
            _GEE_STATE = {
                "connected": False,
                "project_id": _sanitize_project_id(req_proj or secret_proj),
                "mode": "none",
                "error": str(exc),
            }
            return dict(_GEE_STATE)

    _GEE_STATE = {
        "connected": False,
        "project_id": _sanitize_project_id(req_proj or secret_proj),
        "mode": "none",
        "error": "",
    }
    return dict(_GEE_STATE)


def ensure_gee_initialized() -> bool:
    """Auto-initialize GEE once from ``st.secrets`` / env / local creds if not yet attempted."""
    global _GEE_AUTO_ATTEMPTED
    if _GEE_STATE.get("connected"):
        return True
    if _GEE_AUTO_ATTEMPTED:
        return False
    res = init_gee()
    return bool(res.get("connected"))


def disconnect_gee() -> dict[str, Any]:
    """Reset the active GEE session state (returns to Planetary Computer default)."""
    global _GEE_STATE, _GEE_AUTO_ATTEMPTED
    _GEE_AUTO_ATTEMPTED = True
    _GEE_STATE = {
        "connected": False,
        "project_id": "",
        "mode": "none",
        "error": "",
    }
    return dict(_GEE_STATE)


def is_gee_ready() -> bool:
    """Return True if Google Earth Engine is initialized and ready for server-side runs."""
    if not _EE_IMPORT_OK or ee is None:
        return False
    if not _GEE_STATE.get("connected") and not _GEE_AUTO_ATTEMPTED:
        ensure_gee_initialized()
    return bool(_GEE_STATE.get("connected"))


def get_gee_status() -> dict[str, Any]:
    """Return a copy of the current GEE connection status dict (with sanitized project_id)."""
    if _EE_IMPORT_OK and ee is not None and not _GEE_STATE.get("connected") and not _GEE_AUTO_ATTEMPTED:
        ensure_gee_initialized()
    safe_proj = _sanitize_project_id(_GEE_STATE.get("project_id"))
    raw_err = str(_GEE_STATE.get("error") or "")
    if "private_key" in raw_err.lower() or "begin private" in raw_err.lower():
        raw_err = "GEE authentication error (check credentials format)."
    return {
        "connected": bool(_GEE_STATE.get("connected")),
        "project_id": safe_proj,
        "mode": str(_GEE_STATE.get("mode") or "none"),
        "error": raw_err,
        "api_installed": bool(_EE_IMPORT_OK),
    }


def get_gee_index_vis(index_code: str) -> dict[str, Any]:
    """Return visualization metadata for a supported GEE spectral index."""
    code = str(index_code or "NDVI").strip().upper()
    if code not in GEE_INDEX_VIS:
        raise KeyError(f"Unsupported GEE index: {code!r}. Supported: {GEE_SUPPORTED_INDICES}")
    return dict(GEE_INDEX_VIS[code])


# ---------------------------------------------------------------------------
# Server-Side Spectral Index Computation
# ---------------------------------------------------------------------------
def gee_compute_index(image: Any, index_code: str) -> Any:
    """Compute a spectral index on a harmonized GEE ``ee.Image`` (bands: blue, green, red, nir, swir16, swir22).

    Supports all 11 optical indices:
    ``NDVI, EVI, SAVI, NDMI, NDWI, MNDWI, NDSI, NDBI, BSI, NBR, BAI``.
    For ``NDSI``, also attaches ``NDWI`` so proglacial water can be separated from ice/snow.
    """
    if not _EE_IMPORT_OK or ee is None:
        raise RuntimeError("earthengine-api is not installed.")
    code = str(index_code or "NDVI").strip().upper()
    if code not in GEE_SUPPORTED_INDICES:
        raise KeyError(f"Unsupported GEE index: {code!r}. Supported: {GEE_SUPPORTED_INDICES}")

    if code == "NDVI":
        return image.normalizedDifference(["nir", "red"]).rename("NDVI")
    if code == "EVI":
        return image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))",
            {"NIR": image.select("nir"), "RED": image.select("red"), "BLUE": image.select("blue")},
        ).clamp(-1.0, 1.0).rename("EVI")
    if code == "SAVI":
        return image.expression(
            "1.5 * ((NIR - RED) / (NIR + RED + 0.5))",
            {"NIR": image.select("nir"), "RED": image.select("red")},
        ).clamp(-1.0, 1.0).rename("SAVI")
    if code == "NDMI":
        return image.normalizedDifference(["nir", "swir16"]).rename("NDMI")
    if code == "NDWI":
        return image.normalizedDifference(["green", "nir"]).rename("NDWI")
    if code == "MNDWI":
        return image.normalizedDifference(["green", "swir16"]).rename("MNDWI")
    if code == "NDSI":
        ndsi = image.normalizedDifference(["green", "swir16"]).rename("NDSI")
        ndwi = image.normalizedDifference(["green", "nir"]).rename("NDWI")
        # Water-aware NDSI: depress open water (NDWI > 0.15 & NIR < 0.12) so lakes aren't confused with glacier ice
        water_mask = ndwi.gt(0.15).And(image.select("nir").lt(0.12))
        clean_ndsi = ndsi.where(water_mask, -0.35).rename("NDSI")
        return clean_ndsi
    if code == "NDBI":
        return image.normalizedDifference(["swir16", "nir"]).rename("NDBI")
    if code == "BSI":
        return image.expression(
            "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE) + 1e-6)",
            {
                "SWIR1": image.select("swir16"),
                "RED": image.select("red"),
                "NIR": image.select("nir"),
                "BLUE": image.select("blue"),
            },
        ).clamp(-1.0, 1.0).rename("BSI")
    if code == "NBR":
        return image.normalizedDifference(["nir", "swir22"]).rename("NBR")
    if code == "BAI":
        return image.expression(
            "1.0 / (pow(0.1 - RED, 2) + pow(0.06 - NIR, 2) + 1e-4)",
            {"RED": image.select("red"), "NIR": image.select("nir")},
        ).clamp(0.0, 500.0).rename("BAI")
    raise KeyError(f"Unsupported GEE index: {code!r}")


# ---------------------------------------------------------------------------
# Server-Side Harmonized Collection Builders (Gap-Free Multi-Tile Composites)
# ---------------------------------------------------------------------------
def _build_gap_free_median(
    col: Any,
    cloud_prop: str,
    mask_fn: Callable[[Any], Any],
    raw_prep_fn: Callable[[Any], Any] | None = None,
) -> Any:
    """Build a two-tier cloud-filtered gap-free ``.median()`` composite across ALL tiles.

    NEVER calls global ``.sort().limit(...)`` across the AOI, which would select all scenes
    from a single corner MGRS/WRS-2 tile and leave the rest of a multi-tile AOI black.

    - Tier 1: Cloud-masked median of scenes with ``cloud_prop < 30%`` across all tiles.
    - Tier 2: Cloud-masked median of scenes with ``cloud_prop < 75%`` across all tiles.
    - Tier 3 (optional): Unmasked median of scenes with ``cloud_prop < 40%`` to eliminate
      any rare residual single-pixel cloud holes in tough tropical years.
    """
    tier1 = col.filter(ee.Filter.lt(cloud_prop, 30)).map(mask_fn).median()
    tier2 = col.filter(ee.Filter.lt(cloud_prop, 75)).map(mask_fn).median()
    composite = tier1.unmask(tier2)
    if raw_prep_fn is not None:
        raw_fallback = col.filter(ee.Filter.lt(cloud_prop, 40)).map(raw_prep_fn).median()
        composite = composite.unmask(raw_fallback)
    return composite


def _gee_s2_composite(
    region: Any,
    start_iso: str,
    end_iso: str,
) -> tuple[Any, int]:
    """Build gap-free cloud-masked Sentinel-2 SR Harmonized seasonal median composite."""
    img, col_sz = _gee_s2_composite_lazy(region, start_iso, end_iso)
    cnt = int(col_sz.getInfo())
    if cnt == 0:
        return None, 0
    return img, cnt


def _gee_s2_composite_lazy(
    region: Any,
    start_iso: str,
    end_iso: str,
) -> tuple[Any, Any]:
    """Pure server-side lazy S2 composite (NO .getInfo(), NO global .limit(); two-tier gap-free median)."""
    def _scale_s2(img: Any) -> Any:
        return (
            img.select(["B2", "B3", "B4", "B8", "B11", "B12"], ["blue", "green", "red", "nir", "swir16", "swir22"])
            .multiply(0.0001)
            .clamp(0.0, 1.0)
        )

    def _prep_s2(img: Any) -> Any:
        scl = img.select("SCL")
        mask = (
            scl.neq(1)
            .And(scl.neq(3))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
        )
        return _scale_s2(img).updateMask(mask)

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_iso, end_iso)
    )
    filtered_cnt = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 75)).size()
    composite = _build_gap_free_median(col, "CLOUDY_PIXEL_PERCENTAGE", _prep_s2, raw_prep_fn=_scale_s2)
    return composite.clip(region), filtered_cnt


def _gee_landsat_composite_lazy(
    region: Any,
    start_iso: str,
    end_iso: str,
) -> tuple[Any, Any, str]:
    """Pure server-side lazy Landsat C2 L2 composite (NO .getInfo(), NO global .limit(); two-tier gap-free median)."""
    def _mask_qa(img: Any) -> Any:
        qa = img.select("QA_PIXEL")
        return (
            qa.bitwiseAnd(1 << 1).eq(0)
            .And(qa.bitwiseAnd(1 << 3).eq(0))
            .And(qa.bitwiseAnd(1 << 4).eq(0))
        )

    def _scale_l57(img: Any) -> Any:
        return (
            img.select(
                ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
                ["blue", "green", "red", "nir", "swir16", "swir22"],
            )
            .multiply(0.0000275)
            .add(-0.2)
            .clamp(0.0, 1.0)
        )

    def _prep_l57(img: Any) -> Any:
        return _scale_l57(img).updateMask(_mask_qa(img))

    def _scale_l89(img: Any) -> Any:
        return (
            img.select(
                ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
                ["blue", "green", "red", "nir", "swir16", "swir22"],
            )
            .multiply(0.0000275)
            .add(-0.2)
            .clamp(0.0, 1.0)
        )

    def _prep_l89(img: Any) -> Any:
        return _scale_l89(img).updateMask(_mask_qa(img))

    yr = int(start_iso[:4])
    if yr >= 2013:
        raw_col = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_iso, end_iso)
            .merge(
                ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                .filterBounds(region)
                .filterDate(start_iso, end_iso)
            )
        )
        prep_fn = _prep_l89
        raw_fn = _scale_l89
        plat = "Landsat-8/9"
    else:
        raw_col = (
            ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_iso, end_iso)
            .merge(
                ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                .filterBounds(region)
                .filterDate(start_iso, end_iso)
            )
        )
        prep_fn = _prep_l57
        raw_fn = _scale_l57
        plat = "Landsat-5/7"

    filtered_cnt = raw_col.filter(ee.Filter.lt("CLOUD_COVER", 75)).size()
    composite = _build_gap_free_median(raw_col, "CLOUD_COVER", prep_fn, raw_prep_fn=raw_fn)
    return composite.clip(region), filtered_cnt, plat


def _gee_landsat_composite(
    region: Any,
    start_iso: str,
    end_iso: str,
) -> tuple[Any, int, str]:
    """Build gap-free cloud-masked Landsat C2 L2 seasonal median composite."""
    img, col_sz, plat = _gee_landsat_composite_lazy(region, start_iso, end_iso)
    cnt = int(col_sz.getInfo())
    if cnt == 0:
        return None, 0, plat
    return img, cnt, plat


def _gee_modis_composite_lazy(
    region: Any,
    start_iso: str,
    end_iso: str,
) -> tuple[Any, Any]:
    """Pure server-side lazy MODIS MOD09A1 composite (NO .getInfo() round-trip, NO global .limit())."""
    def _prep_modis(img: Any) -> Any:
        return (
            img.select(
                ["sur_refl_b03", "sur_refl_b04", "sur_refl_b01", "sur_refl_b02", "sur_refl_b06", "sur_refl_b07"],
                ["blue", "green", "red", "nir", "swir16", "swir22"],
            )
            .multiply(0.0001)
            .clamp(0.0, 1.0)
        )

    col = (
        ee.ImageCollection("MODIS/061/MOD09A1")
        .filterBounds(region)
        .filterDate(start_iso, end_iso)
    )
    return col.map(_prep_modis).median().clip(region), col.size()


def _gee_modis_composite(
    region: Any,
    start_iso: str,
    end_iso: str,
) -> tuple[Any, int]:
    """Build Terra MODIS MOD09A1 500 m 8-day SR seasonal median composite."""
    img, col_sz = _gee_modis_composite_lazy(region, start_iso, end_iso)
    cnt = int(col_sz.getInfo())
    if cnt == 0:
        return None, 0
    return img, cnt


def _build_batched_yearly_stats_fc(
    year_specs: Sequence[dict[str, Any]],
    region: Any,
    idx_key: str,
    reduce_scale: int,
    tile_scale: int = 4,
) -> Any:
    """Build ONE server-side ``ee.FeatureCollection`` evaluating all years in parallel in a single ``.getInfo()`` call."""
    reducer = ee.Reducer.mean().combine(ee.Reducer.median(), sharedInputs=True)
    feats: list[Any] = []
    for sp in year_specs:
        idx_img = sp["idx_img"]
        stats_d = ee.Algorithms.If(
            sp["col_size_ee"].gt(0),
            idx_img.reduceRegion(
                reducer=reducer,
                geometry=region,
                scale=int(reduce_scale),
                bestEffort=True,
                maxPixels=int(1e9),
                tileScale=int(tile_scale),
            ),
            ee.Dictionary({}),
        )
        stats_dict = ee.Dictionary(stats_d)
        feats.append(
            ee.Feature(
                None,
                {
                    "year": int(sp["year"]),
                    "scene_cnt": sp["col_size_ee"],
                    "mean": stats_dict.get(f"{idx_key}_mean", None),
                    "median": stats_dict.get(f"{idx_key}_median", None),
                },
            )
        )
    return ee.FeatureCollection(feats)


def _season_date_window(year: int, target_season: str, center_lat: float) -> tuple[str, str]:
    """Return ``(start_iso, end_iso)`` for the target season (hemisphere, tropical dry-season & current-year aware)."""
    s = str(target_season or "summer").strip().lower()
    if s == "glacier_ablation":
        if center_lat < -20.0:
            start_iso, end_iso = f"{year}-01-10", f"{year}-03-15"
        else:
            start_iso, end_iso = f"{year}-07-15", f"{year}-09-15"
    elif s == "winter":
        start_iso, end_iso = f"{year}-01-01", f"{year}-03-31"
    elif s == "spring":
        start_iso, end_iso = f"{year}-04-01", f"{year}-06-15"
    elif s == "autumn":
        start_iso, end_iso = f"{year}-09-15", f"{year}-11-30"
    elif center_lat <= -55.0:
        # Antarctic austral summer
        start_iso, end_iso = f"{year}-01-01", f"{year}-03-15"
    elif s == "dry" or (-25.0 < center_lat < 5.0):
        # Southern/Equatorial tropical dry season (Amazon / Rondônia: Jun 1 – Oct 15, 4.5 months)
        start_iso, end_iso = f"{year}-06-01", f"{year}-10-15"
    elif center_lat <= -25.0:
        start_iso, end_iso = f"{year}-01-01", f"{year}-03-31"
    else:
        # Default Northern Hemisphere summer
        start_iso, end_iso = f"{year}-06-15", f"{year}-09-15"

    # Clamp current-year end_date to today UTC so GEE never queries future dates
    now_utc = datetime.now(timezone.utc)
    if int(year) >= now_utc.year:
        today_iso = now_utc.strftime("%Y-%m-%d")
        if end_iso > today_iso:
            end_iso = today_iso
        if start_iso >= end_iso:
            start_iso = f"{now_utc.year}-01-01"
    return start_iso, end_iso


def select_gee_scale_for_area(area_km2: float, sensor_mode: str = "auto_fusion") -> int:
    """Select a fast, timeout-safe GEE server-side ``scale_m`` from AOI area in km².

    - ``<= 250 km²``: ``30 m``
    - ``250 – 2,500 km²`` (e.g., Rondônia 903 km²): ``60 m``
    - ``2,500 – 25,000 km²``: ``150 m``
    - ``25,000 – 250,000 km²``: ``300 m``
    - ``250,000 – 1,500,000 km²``: ``600 m``
    - ``> 1,500,000 km²``: ``1000 m``
    """
    a = float(area_km2)
    if a > 1_500_000.0:
        scale_m = 1000
    elif a > 250_000.0:
        scale_m = 600
    elif a > 25_000.0:
        scale_m = 300
    elif a > 2_500.0:
        scale_m = 150
    elif a > 250.0:
        scale_m = 60
    else:
        scale_m = 30
    if sensor_mode == "modis":
        scale_m = max(scale_m, 500)
    return int(scale_m)


def select_gee_sensor_for_year(
    year: int,
    aoi_area_km2: float,
    sensor_mode: str = "auto_fusion",
) -> str:
    """Select the optimal GEE sensor family (``'modis'``, ``'sentinel2'``, or ``'landsat'``) for a year & AOI area.

    - Continental (``aoi_area_km2 > 1,500,000 km²``) or explicit ``'modis'``: ``'modis'`` (``>= 2000``).
    - Local/hotspot (``aoi_area_km2 <= 15,000 km²``) & ``year >= 2017`` (or explicit ``'sentinel2'``): ``'sentinel2'``.
    - Regional (``15,000 < aoi_area_km2 <= 1,500,000 km²``) or ``year < 2017``: ``'landsat'``
      (Landsat-5/7 ``< 2013``, Landsat-8/9 ``2013–2026`` — 5–10x faster than S2 on large regional AOIs,
      preventing 2020–2026 timeouts and preserving radiometric consistency).
    """
    sm = str(sensor_mode or "auto_fusion").strip().lower()
    a = float(aoi_area_km2)
    yr = int(year)
    if sm == "modis" or (sm == "auto_fusion" and a > 1_500_000.0):
        return "modis" if yr >= 2000 else "landsat"
    if sm == "sentinel2":
        return "sentinel2"
    if sm == "auto_fusion" and a <= 15_000.0 and yr >= 2017:
        return "sentinel2"
    return "landsat"


def select_gee_scale_and_mode(
    bbox_4326: Sequence[float],
    sensor_mode: str = "auto_fusion",
) -> dict[str, Any]:
    """Compute rectangular AOI area in km² and pick a timeout-safe GEE server-side scale & tileScale."""
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox_4326]
    center_lat = 0.5 * (min_lat + max_lat)
    h_km = abs(max_lat - min_lat) * 111.32
    w_km = abs(max_lon - min_lon) * 111.32 * max(0.12, float(np.cos(np.radians(center_lat))))
    area_km2 = max(1.0, h_km * w_km)

    scale_m = select_gee_scale_for_area(area_km2, sensor_mode=sensor_mode)
    tile_scale = 4
    thumb_dim = 480
    use_modis_post_2000 = bool(sensor_mode == "modis" or (sensor_mode == "auto_fusion" and area_km2 > 1_500_000.0))
    return {
        "area_km2": round(area_km2, 1),
        "scale_m": int(scale_m),
        "tile_scale": int(tile_scale),
        "thumb_dim": int(thumb_dim),
        "use_modis_post_2000": use_modis_post_2000,
    }


def is_gee_iam_permission_error(exc: Exception | str) -> bool:
    """Return True if an exception indicates a Google Cloud IAM permission error (never swallow!)."""
    low = str(exc or "").lower()
    return (
        "thumbnails.create" in low
        or "earthengine.thumbnails" in low
        or ("permission" in low and ("denied" in low or "required" in low or "does not have" in low))
        or ("403" in low and ("forbidden" in low or "earth engine" in low or "earthengine" in low))
    )


def is_transient_gee_timeout(exc: Exception | str) -> bool:
    """Return True if an exception represents a transient GEE network/read/computation timeout."""
    if is_gee_iam_permission_error(exc):
        return False
    if isinstance(
        exc,
        (
            requests.exceptions.ReadTimeout,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            TimeoutError,
        ),
    ):
        return True
    low = str(exc or "").lower()
    return any(
        tok in low
        for tok in (
            "read timed out",
            "timed out",
            "timeout",
            "deadline exceeded",
            "computation timed out",
            "connection aborted",
            "connection reset",
            "httpsconnectionpool",
        )
    )


def _gee_call_with_retry(
    primary_fn: Callable[[], Any],
    retry_fn: Callable[[], Any] | None = None,
    *,
    label: str = "",
) -> Any:
    """Execute a GEE server call with 1 automatic lighter retry on transient timeout; returns None if both time out."""
    try:
        return primary_fn()
    except Exception as exc1:
        if not is_transient_gee_timeout(exc1):
            raise
        logger.warning("GEE transient timeout on %s (%s); retrying once with lighter query...", label or "call", exc1)
        try:
            return (retry_fn or primary_fn)()
        except Exception as exc2:
            if not is_transient_gee_timeout(exc2):
                raise
            logger.warning("GEE retry also timed out on %s (%s); skipping frame.", label or "call", exc2)
            return None


def format_gee_error(exc: Exception | str, lang: str = "sl") -> str:
    """Translate raw Google Earth Engine IAM/permission errors into clear actionable guidance."""
    raw = str(exc or "").strip()
    if is_gee_iam_permission_error(raw):
        if lang == "en":
            return (
                "Google Earth Engine permission denied (IAM): Your Google Cloud Service Account needs the "
                "**Earth Engine Resource Admin** (or **Earth Engine Resource Writer**) role in Google Cloud IAM "
                "(*Viewer* is not sufficient for `thumbnails.create`). Open Google Cloud Console → IAM & Admin, "
                "grant the role to your service account, and retry."
            )
        return (
            "Google Earth Engine zavrnil dostop (IAM pravice): Vaš Google Cloud Service Account potrebuje vlogo "
            "**Earth Engine Resource Admin** (ali **Earth Engine Resource Writer**) v Google Cloud IAM "
            "(vloga *Viewer* ne zadostuje za `thumbnails.create`). Odprite Google Cloud Console → IAM & Admin, "
            "dodelite vlogo svojemu Service Accountu in poskusite znova."
        )
    return raw


def _fetch_thumb_png(
    image: Any,
    vis_params: dict[str, Any],
    region: Any,
    max_dim: int = 480,
    lang: str = "sl",
) -> bytes:
    """Download a lightweight PNG visualization from GEE ``getThumbURL`` (EPSG:3857, 480 px, 30 s timeout)."""
    params = dict(vis_params)
    params["region"] = region
    params["dimensions"] = int(max_dim)
    params["crs"] = "EPSG:3857"
    params["format"] = "png"
    try:
        url = image.getThumbURL(params)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        if is_gee_iam_permission_error(exc):
            raise RuntimeError(format_gee_error(exc, lang=lang)) from exc
        raise


def _png_bytes_to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _stamp_png_bytes(png_bytes: bytes, banner: str) -> bytes:
    """Stamp a crisp date+sensor banner onto a GEE PNG frame using our existing helper."""
    try:
        from casovne_vrste import _stamp_banner_on_pil, pil_to_png_bytes

        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        stamped = _stamp_banner_on_pil(pil_img, banner)
        return pil_to_png_bytes(stamped)
    except Exception:
        return png_bytes


# ---------------------------------------------------------------------------
# Main Server-Side Temporal Analysis Engine
# ---------------------------------------------------------------------------
def gee_run_temporal_analysis(
    bbox_4326: Sequence[float],
    start_year: int,
    end_year: int,
    index_code: str = "NDVI",
    sensor_mode: str = "auto_fusion",
    target_season: str = "summer",
    max_frames: int = 7,
    progress_cb: Callable[[float, str], None] | None = None,
    lang: str = "sl",
    max_total_seconds: float = 55.0,
) -> dict[str, Any]:
    """Run fast server-side multi-year compositing, batched stats, parallel thumbnails & Δ stats on GEE.

    Architecture (typically ~15–25 s total):
        1. Builds lazy server-side two-tier gap-free ``.median().unmask(...)`` composites across
           ALL tiles (no global ``.limit()``) for <= 7 sampled years.
        2. Evaluates ALL yearly mean/median stats + scene counts in ONE batched ``ee.FeatureCollection.getInfo()``.
        3. Fetches lightweight 480 px thumbnails concurrently via ``ThreadPoolExecutor(max_workers=4)``,
           prioritizing ``t1`` (earliest reliable) and ``t2`` (latest) first, omitting redundant
           complementary-index thumbnails on intermediate years, and enforcing a strict wall-clock
           budget (``max_total_seconds=55``).
    """
    if not is_gee_ready():
        raise RuntimeError("Google Earth Engine is not connected.")

    t_start = time.monotonic()
    idx_key = str(index_code or "NDVI").strip().upper()
    vis_spec = get_gee_index_vis(idx_key)
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox_4326]
    center_lat = 0.5 * (min_lat + max_lat)
    bounds_4326 = [[min_lat, min_lon], [max_lat, max_lon]]
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], None, False)

    scale_cfg = select_gee_scale_and_mode(bbox_4326, sensor_mode=sensor_mode)
    aoi_area_km2 = float(scale_cfg["area_km2"])
    reduce_scale = int(scale_cfg["scale_m"])
    tile_scale = int(scale_cfg["tile_scale"])
    thumb_dim = int(scale_cfg["thumb_dim"])
    use_modis_post_2000 = bool(scale_cfg["use_modis_post_2000"])

    now_utc_yr = datetime.now(timezone.utc).year
    sy = max(1984, int(start_year))
    ey = min(now_utc_yr, max(sy, int(end_year)))
    if sensor_mode == "modis" or use_modis_post_2000:
        # Terra MODIS launched Dec 1999; never query MODIS before year 2000
        sy = max(2000, sy)
        ey = max(sy, ey)
    elif sensor_mode == "sentinel2":
        sy = max(2017, sy)
        ey = max(sy, ey)

    all_years = list(range(sy, ey + 1))
    cap = max(2, min(int(max_frames), 7))
    if len(all_years) > cap:
        pick_idx = np.linspace(0, len(all_years) - 1, cap, dtype=int)
        sampled_years = sorted(set(all_years[i] for i in pick_idx))
    else:
        sampled_years = list(all_years)

    # If end_year is the current year and (end_year - 1) isn't sampled yet, include (end_year - 1)
    # as a zero-extra-round-trip fallback candidate in the single batched stats call!
    fallback_prev_yr: int | None = None
    if ey >= now_utc_yr and (ey - 1) >= sy and (ey - 1) not in sampled_years:
        fallback_prev_yr = ey - 1
        sampled_years = sorted(set(sampled_years + [fallback_prev_yr]))

    if idx_key == "NDSI":
        comp_idx = "NDWI"
    elif idx_key in ("NDBI", "NBR", "BSI"):
        comp_idx = "NDVI"
    else:
        comp_idx = "NDMI"
    comp_vis = get_gee_index_vis(comp_idx)

    def _prog(frac: float, msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(min(1.0, max(0.0, float(frac))), msg)
            except Exception:
                pass

    # 1. Build pure lazy server-side composites for all sampled years (ZERO network calls yet!)
    _prog(
        0.12,
        f"🌍 GEE pripravlja strežniške mozaike za {len(sampled_years)} let..."
        if lang == "sl"
        else f"🌍 GEE preparing server-side composites for {len(sampled_years)} years...",
    )
    year_specs: list[dict[str, Any]] = []
    for yr in sampled_years:
        w_start, w_end = _season_date_window(yr, target_season, center_lat)
        mid_date = f"{yr}-08-01" if center_lat >= -20.0 else f"{yr}-02-01"
        if mid_date > w_end:
            mid_date = w_end

        yr_sensor = select_gee_sensor_for_year(yr, aoi_area_km2, sensor_mode=sensor_mode)
        if yr_sensor == "modis":
            comp_img, col_sz = _gee_modis_composite_lazy(region, w_start, w_end)
            plat_lbl, sensor_fam = "MODIS Terra", "MODIS (500 m)"
        elif yr_sensor == "sentinel2":
            comp_img, col_sz = _gee_s2_composite_lazy(region, w_start, w_end)
            plat_lbl, sensor_fam = "Sentinel-2", "Sentinel-2 (10–20 m)"
        else:
            comp_img, col_sz, plat_lbl = _gee_landsat_composite_lazy(region, w_start, w_end)
            sensor_fam = "Landsat (30 m)"

        idx_img = gee_compute_index(comp_img, idx_key)
        comp_idx_img = gee_compute_index(comp_img, comp_idx)
        year_specs.append(
            {
                "year": yr,
                "mid_date": mid_date,
                "comp_img": comp_img,
                "idx_img": idx_img,
                "comp_idx_img": comp_idx_img,
                "col_size_ee": col_sz,
                "plat_lbl": plat_lbl,
                "sensor_fam": sensor_fam,
                "is_reliable": yr >= 1990,
            }
        )

    # 2. ONE SINGLE BATCHED .getInfo() call for ALL years' stats + scene counts!
    _prog(
        0.24,
        "🌍 GEE paketno računa statistike vseh let na Googlovih strežnikih..."
        if lang == "sl"
        else "🌍 GEE computing batched statistics for all years on Google's servers...",
    )
    stats_fc = _build_batched_yearly_stats_fc(year_specs, region, idx_key, reduce_scale, tile_scale)
    try:
        fc_info = stats_fc.getInfo() or {}
    except Exception as exc:
        if is_gee_iam_permission_error(exc):
            raise RuntimeError(format_gee_error(exc, lang=lang)) from exc
        raise

    stats_by_year: dict[int, dict[str, Any]] = {}
    for feat in fc_info.get("features") or []:
        props = feat.get("properties") or {}
        if props.get("year") is not None:
            stats_by_year[int(props["year"])] = props

    valid_specs: list[dict[str, Any]] = []
    for sp in year_specs:
        pr = stats_by_year.get(int(sp["year"])) or {}
        sc_cnt = int(pr.get("scene_cnt") or 0)
        im_val = pr.get("mean")
        if sc_cnt <= 0 or im_val is None:
            continue
        imed_val = pr.get("median")
        sp["scene_cnt"] = sc_cnt
        sp["idx_mean_f"] = round(float(im_val), 4)
        sp["idx_med_f"] = round(float(imed_val if imed_val is not None else im_val), 4)
        valid_specs.append(sp)

    # If the current year succeeded and we included a backup (ey - 1) that pushed us over cap, drop the backup
    if fallback_prev_yr is not None and len(valid_specs) > cap:
        valid_yrs_set = {int(sp["year"]) for sp in valid_specs}
        if ey in valid_yrs_set and fallback_prev_yr in valid_yrs_set:
            valid_specs = [sp for sp in valid_specs if int(sp["year"]) != fallback_prev_yr]

    if not valid_specs:
        raise RuntimeError(
            "Google Earth Engine ni našel uporabnih ne-oblačnih mozaikov za izbrano obdobje in območje."
            if lang == "sl"
            else "Google Earth Engine found no usable cloud-free composites for the selected period and region."
        )

    # Identify t1 (earliest reliable >= 1990) and t2 (latest) and prioritize t2 & t1 FIRST!
    t1_pos = 0
    for idx_i, sp in enumerate(valid_specs):
        if sp["is_reliable"]:
            t1_pos = idx_i
            break
    t2_pos = len(valid_specs) - 1
    endpoint_years = {valid_specs[t1_pos]["year"], valid_specs[t2_pos]["year"]}

    priority_order = [valid_specs[t2_pos]]
    if t1_pos != t2_pos:
        priority_order.append(valid_specs[t1_pos])
    for idx_i, sp in enumerate(valid_specs):
        if idx_i not in (t1_pos, t2_pos):
            priority_order.append(sp)

    # 3. Parallel 4-worker thumbnail fetching with strict overall wall-clock budget!
    _prog(
        0.40,
        f"🌍 GEE vzporedno generira satelitske prikaze ({len(valid_specs)} let)..."
        if lang == "sl"
        else f"🌍 GEE rendering satellite overlays in parallel ({len(valid_specs)} years)...",
    )
    rgb_vis = {"bands": ["red", "green", "blue"], "min": 0.02, "max": 0.35, "gamma": 1.25}
    idx_vis_p = {"min": vis_spec["min"], "max": vis_spec["max"], "palette": vis_spec["palette"]}
    comp_vis_p = {"min": comp_vis["min"], "max": comp_vis["max"], "palette": comp_vis["palette"]}

    def _fetch_year_bundle(sp: dict[str, Any]) -> dict[str, Any] | None:
        yr_i = int(sp["year"])
        is_endpoint = yr_i in endpoint_years
        # Respect strict wall-clock budget for non-endpoint intermediate frames
        if not is_endpoint and (time.monotonic() - t_start) > float(max_total_seconds):
            return None
        try:
            rgb_raw = _fetch_thumb_png(sp["comp_img"], rgb_vis, region, max_dim=thumb_dim, lang=lang)
            idx_raw = _fetch_thumb_png(sp["idx_img"], idx_vis_p, region, max_dim=thumb_dim, lang=lang)
            # Fetch complementary index ONLY for t1 & t2 endpoints (cuts GEE renders by ~30-60%!)
            comp_raw: bytes | None = None
            if is_endpoint and (time.monotonic() - t_start) <= float(max_total_seconds):
                try:
                    comp_raw = _fetch_thumb_png(sp["comp_idx_img"], comp_vis_p, region, max_dim=thumb_dim, lang=lang)
                except Exception as exc_c:
                    if is_gee_iam_permission_error(exc_c):
                        raise
                    comp_raw = None
            return {
                "spec": sp,
                "rgb_raw": rgb_raw,
                "idx_raw": idx_raw,
                "comp_raw": comp_raw,
            }
        except Exception as exc_f:
            if is_gee_iam_permission_error(exc_f):
                raise
            logger.warning("GEE thumbnail skipped for %s: %s", yr_i, exc_f)
            return None

    fetched_bundles: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_map = {pool.submit(_fetch_year_bundle, sp): sp for sp in priority_order}
        done_cnt = 0
        for fut in as_completed(fut_map):
            done_cnt += 1
            res_b = fut.result()
            if isinstance(res_b, dict):
                fetched_bundles.append(res_b)
            _prog(
                0.40 + 0.38 * (done_cnt / max(len(priority_order), 1)),
                f"🌍 GEE prikazi: {done_cnt}/{len(priority_order)}"
                if lang == "sl"
                else f"🌍 GEE overlays: {done_cnt}/{len(priority_order)}",
            )

    if not fetched_bundles:
        raise RuntimeError(
            "Google Earth Engine ni uspel generirati satelitskih prikazov (časovna omejitev)."
            if lang == "sl"
            else "Google Earth Engine failed to render satellite overlays (timed out)."
        )

    # Restore strict chronological order
    fetched_bundles.sort(key=lambda b: int(b["spec"]["year"]))

    rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    index_images: list[tuple[int, str, Any, bool]] = []
    excluded_archive_dates: list[str] = []

    for b in fetched_bundles:
        sp = b["spec"]
        yr = int(sp["year"])
        mid_date = str(sp["mid_date"])
        plat_lbl = str(sp["plat_lbl"])
        sensor_fam = str(sp["sensor_fam"])
        is_reliable = bool(sp["is_reliable"])
        idx_mean_f = float(sp["idx_mean_f"])
        idx_med_f = float(sp["idx_med_f"])
        scene_cnt = int(sp["scene_cnt"])
        if not is_reliable:
            excluded_archive_dates.append(mid_date)

        rgb_uri = _png_bytes_to_data_uri(b["rgb_raw"])
        idx_uri = _png_bytes_to_data_uri(b["idx_raw"])
        rgb_stamped = _stamp_png_bytes(b["rgb_raw"], f"{mid_date} | GEE {plat_lbl} | RGB")
        idx_stamped = _stamp_png_bytes(b["idx_raw"], f"{mid_date} | GEE {plat_lbl} | {idx_key}={idx_mean_f:.2f}")

        extra_overlays: dict[str, str] = {}
        extra_pngs: dict[str, bytes] = {}
        extra_means: dict[str, Any] = {}
        if b["comp_raw"]:
            extra_overlays[comp_idx] = _png_bytes_to_data_uri(b["comp_raw"])
            extra_pngs[comp_idx] = _stamp_png_bytes(b["comp_raw"], f"{mid_date} | GEE {plat_lbl} | {comp_idx}")
            extra_means[comp_idx] = None

        rows.append(
            {
                "date": pd.to_datetime(mid_date),
                "date_str": mid_date,
                "scene_id": f"GEE_{plat_lbl.replace(' ', '_')}_{yr}_n{scene_cnt}",
                "platform": plat_lbl,
                "sensor_family": sensor_fam,
                "mgrs_tile": "GEE-Composite",
                "cloud_cover": 0.0,
                "mean": idx_mean_f,
                "median": idx_med_f,
                "std": np.nan,
                "p10": np.nan,
                "p90": np.nan,
                "valid_pct": 100.0,
                "analysis_reliable": is_reliable,
                "quality_note": "" if is_reliable else "Pre-1990 archive frame",
                "engine": "gee",
            }
        )
        frames.append(
            {
                "date": mid_date,
                "requested_date": mid_date,
                "sensor": f"gee:{sensor_mode}",
                "sensor_family": sensor_fam,
                "platform": f"{plat_lbl} (GEE)",
                "resolution_m": reduce_scale,
                "clear_pct": 100.0,
                "saturated_pct": 0.0,
                "analysis_reliable": is_reliable,
                "quality_note": "" if is_reliable else "Pre-1990 archive frame",
                "index_name": idx_key,
                "index_mean": idx_mean_f,
                "bounds_4326": bounds_4326,
                "rgb_overlay_uri": rgb_uri,
                "index_overlay_uri": idx_uri,
                "extra_overlays": extra_overlays,
                "extra_pngs": extra_pngs,
                "extra_means": extra_means,
                "rgb_png_bytes": rgb_stamped,
                "index_png_bytes": idx_stamped,
                "index_grid_small": None,
                "ndwi_grid_small": None,
                "nir_grid_small": None,
            }
        )
        index_images.append((yr, mid_date, sp["idx_img"], is_reliable))

    ts_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    default_t1_idx = 0
    for idx_i, fr in enumerate(frames):
        if fr.get("analysis_reliable", True):
            default_t1_idx = idx_i
            break

    delta_uri: str | None = None
    delta_bnds: list[list[float]] | None = None
    delta_lim = float(vis_spec["delta_lim"])
    delta_cmap = str(vis_spec["delta_cmap"])
    change_summary: dict[str, Any] | None = None

    if len(index_images) >= 2 and default_t1_idx < len(index_images) - 1:
        _prog(
            0.82,
            "🌍 GEE računa prostorsko spremembo (Δ) in površine v km²..."
            if lang == "sl"
            else "🌍 GEE computing spatial Δ change map and km² area statistics...",
        )
        _, d1, img_t1, _ = index_images[default_t1_idx]
        _, d2, img_t2, _ = index_images[-1]
        diff_img = img_t2.subtract(img_t1).rename("DELTA")

        d_pal = GEE_DELTA_PALETTES.get(delta_cmap, GEE_DELTA_PALETTES["RdBu"])
        try:
            delta_png = _fetch_thumb_png(
                diff_img,
                {"min": -delta_lim, "max": delta_lim, "palette": d_pal},
                region,
                max_dim=thumb_dim,
                lang=lang,
            )
            delta_uri = _png_bytes_to_data_uri(delta_png)
            delta_bnds = bounds_4326
        except Exception as exc_d:
            if is_gee_iam_permission_error(exc_d):
                raise
            logger.warning("GEE delta thumbnail skipped: %s", exc_d)

        # Single combined pixelArea() reduction
        thr = 0.15 if idx_key != "BAI" else 25.0
        px_area = ee.Image.pixelArea()
        dec_area_img = px_area.updateMask(diff_img.lt(-thr)).rename("dec_m2")
        inc_area_img = px_area.updateMask(diff_img.gt(thr)).rename("inc_m2")
        tot_area_img = px_area.updateMask(diff_img.mask()).rename("tot_m2")

        area_sums: dict[str, Any] = {}
        try:
            area_sums = (
                dec_area_img.addBands(inc_area_img)
                .addBands(tot_area_img)
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=region,
                    scale=reduce_scale,
                    bestEffort=True,
                    maxPixels=int(1e9),
                    tileScale=tile_scale,
                )
                .getInfo()
                or {}
            )
        except Exception as exc_a:
            logger.warning("GEE pixelArea reduction skipped: %s", exc_a)

        dec_km2 = float(area_sums.get("dec_m2") or 0.0) / 1e6
        inc_km2 = float(area_sums.get("inc_m2") or 0.0) / 1e6
        tot_km2 = max(1e-6, float(area_sums.get("tot_m2") or 0.0) / 1e6)
        delta_mean_v = float(frames[-1]["index_mean"] - frames[default_t1_idx]["index_mean"])

        change_summary = {
            "t1_date": d1,
            "t2_date": d2,
            "t1_mean": frames[default_t1_idx]["index_mean"],
            "t2_mean": frames[-1]["index_mean"],
            "delta_mean": round(delta_mean_v, 4),
            "mean": round(delta_mean_v, 4),
            "threshold": thr,
            "decrease_area_km2": round(dec_km2, 2),
            "decrease_area_ha": round(dec_km2 * 100.0, 1),
            "decrease_pct": round(100.0 * dec_km2 / tot_km2, 2),
            "increase_area_km2": round(inc_km2, 2),
            "increase_area_ha": round(inc_km2 * 100.0, 1),
            "increase_pct": round(100.0 * inc_km2 / tot_km2, 2),
            "total_valid_km2": round(tot_km2, 2),
            "engine": "gee",
        }

    gif_rgb: bytes | None = None
    gif_idx: bytes | None = None
    try:
        from casovne_vrste import compile_timelapse_gif_fps

        rgb_pngs = [f["rgb_png_bytes"] for f in frames if f.get("rgb_png_bytes")]
        idx_pngs = [f["index_png_bytes"] for f in frames if f.get("index_png_bytes")]
        if len(rgb_pngs) >= 2:
            gif_rgb = compile_timelapse_gif_fps(rgb_pngs, fps=1.5)
        if len(idx_pngs) >= 2:
            gif_idx = compile_timelapse_gif_fps(idx_pngs, fps=1.5)
    except Exception as exc:
        logger.warning("GEE timelapse GIF compilation skipped: %s", exc)

    actual_sensors = sorted({str(f.get("platform") or f.get("sensor_family") or "").strip() for f in frames if f and (f.get("platform") or f.get("sensor_family"))})
    actual_start_date = str(frames[0]["date"]) if frames else ""
    actual_end_date = str(frames[-1]["date"]) if frames else ""
    actual_aoi_area_km2 = (
        round(float(change_summary["total_valid_km2"]), 1)
        if isinstance(change_summary, dict) and float(change_summary.get("total_valid_km2") or 0.0) > 0.0
        else round(float(aoi_area_km2), 1)
    )

    _prog(1.0, "GEE obdelava zaključena." if lang == "sl" else "GEE processing complete.")

    return {
        "engine": "gee",
        "gee_project_id": get_gee_status().get("project_id", ""),
        "ts_df": ts_df,
        "frames": frames,
        "default_t1_index": default_t1_idx,
        "excluded_archive_dates": excluded_archive_dates,
        "bounds_4326": bounds_4326,
        "delta_overlay_uri": delta_uri,
        "delta_overlay_bounds": delta_bnds,
        "delta_vmin": -delta_lim,
        "delta_vmax": delta_lim,
        "delta_cmap": delta_cmap,
        "change_summary": change_summary,
        "timelapse_rgb_gif_bytes": gif_rgb,
        "timelapse_index_gif_bytes": gif_idx,
        "extra_indices": [comp_idx],
        "reduce_scale_m": reduce_scale,
        "actual_sensors": actual_sensors,
        "actual_start_date": actual_start_date,
        "actual_end_date": actual_end_date,
        "actual_resolution_m": reduce_scale,
        "actual_aoi_area_km2": actual_aoi_area_km2,
    }
