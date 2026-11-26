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
from collections.abc import Mapping
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
# Server-Side Harmonized Collection Builders
# ---------------------------------------------------------------------------
def _gee_s2_composite(region: Any, start_iso: str, end_iso: str) -> tuple[Any, int]:
    """Build cloud-masked Sentinel-2 SR Harmonized seasonal median composite."""
    def _prep_s2(img: Any) -> Any:
        scl = img.select("SCL")
        # Mask cloud shadow (3), medium/high cloud (8, 9), cirrus (10), saturated/defective (1)
        mask = (
            scl.neq(1)
            .And(scl.neq(3))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
        )
        optical = (
            img.select(["B2", "B3", "B4", "B8", "B11", "B12"], ["blue", "green", "red", "nir", "swir16", "swir22"])
            .multiply(0.0001)
            .clamp(0.0, 1.0)
        )
        return optical.updateMask(mask)

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_iso, end_iso)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 45))
        .map(_prep_s2)
    )
    count = int(col.size().getInfo())
    if count == 0:
        return None, 0
    return col.median().clip(region), count


def _gee_landsat_composite(region: Any, start_iso: str, end_iso: str) -> tuple[Any, int, str]:
    """Build cloud-masked Landsat C2 L2 (L5/L7/L8/L9) seasonal median composite."""
    def _mask_qa(img: Any) -> Any:
        qa = img.select("QA_PIXEL")
        # Bit 1: dilated cloud, Bit 3: cloud, Bit 4: cloud shadow
        mask = (
            qa.bitwiseAnd(1 << 1).eq(0)
            .And(qa.bitwiseAnd(1 << 3).eq(0))
            .And(qa.bitwiseAnd(1 << 4).eq(0))
        )
        return mask

    def _prep_l57(img: Any) -> Any:
        mask = _mask_qa(img)
        opt = (
            img.select(
                ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
                ["blue", "green", "red", "nir", "swir16", "swir22"],
            )
            .multiply(0.0000275)
            .add(-0.2)
            .clamp(0.0, 1.0)
        )
        return opt.updateMask(mask)

    def _prep_l89(img: Any) -> Any:
        mask = _mask_qa(img)
        opt = (
            img.select(
                ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
                ["blue", "green", "red", "nir", "swir16", "swir22"],
            )
            .multiply(0.0000275)
            .add(-0.2)
            .clamp(0.0, 1.0)
        )
        return opt.updateMask(mask)

    yr = int(start_iso[:4])
    if yr >= 2013:
        c8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_iso, end_iso)
            .filter(ee.Filter.lt("CLOUD_COVER", 45))
            .map(_prep_l89)
        )
        c9 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_iso, end_iso)
            .filter(ee.Filter.lt("CLOUD_COVER", 45))
            .map(_prep_l89)
        )
        col = c8.merge(c9)
        plat = "Landsat-8/9"
    else:
        c5 = (
            ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_iso, end_iso)
            .filter(ee.Filter.lt("CLOUD_COVER", 45))
            .map(_prep_l57)
        )
        c7 = (
            ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_iso, end_iso)
            .filter(ee.Filter.lt("CLOUD_COVER", 45))
            .map(_prep_l57)
        )
        col = c5.merge(c7)
        plat = "Landsat-5/7"

    count = int(col.size().getInfo())
    if count == 0:
        return None, 0, plat
    return col.median().clip(region), count, plat


def _gee_modis_composite(region: Any, start_iso: str, end_iso: str) -> tuple[Any, int]:
    """Build Terra MODIS MOD09A1 500 m 8-day SR seasonal median composite."""
    def _prep_modis(img: Any) -> Any:
        opt = (
            img.select(
                ["sur_refl_b03", "sur_refl_b04", "sur_refl_b01", "sur_refl_b02", "sur_refl_b06", "sur_refl_b07"],
                ["blue", "green", "red", "nir", "swir16", "swir22"],
            )
            .multiply(0.0001)
            .clamp(0.0, 1.0)
        )
        return opt

    col = (
        ee.ImageCollection("MODIS/061/MOD09A1")
        .filterBounds(region)
        .filterDate(start_iso, end_iso)
        .map(_prep_modis)
    )
    count = int(col.size().getInfo())
    if count == 0:
        return None, 0
    return col.median().clip(region), count


def _season_date_window(year: int, target_season: str, center_lat: float) -> tuple[str, str]:
    """Return ``(start_iso, end_iso)`` for the target season (hemisphere-aware)."""
    s = str(target_season or "summer").strip().lower()
    if s == "glacier_ablation":
        if center_lat < -20.0:
            return f"{year}-01-10", f"{year}-03-15"
        return f"{year}-07-15", f"{year}-09-15"
    if s == "winter":
        return f"{year}-01-01", f"{year}-03-31"
    if s == "spring":
        return f"{year}-04-01", f"{year}-06-15"
    if s == "autumn":
        return f"{year}-09-15", f"{year}-11-30"
    if s == "dry" or center_lat < -5.0:
        # Southern hemisphere dry/summer window (Jun–Sep for Amazon dry season, Jan–Mar for Antarctica)
        if center_lat <= -55.0:
            return f"{year}-01-01", f"{year}-03-15"
        return f"{year}-06-15", f"{year}-09-30"
    # Default Northern Hemisphere summer
    return f"{year}-06-15", f"{year}-09-15"


def _fetch_thumb_png(image: Any, vis_params: dict[str, Any], region: Any, max_dim: int = 768) -> bytes:
    """Download a lightweight PNG visualization from GEE ``getThumbURL``."""
    params = dict(vis_params)
    params["region"] = region
    params["dimensions"] = int(max_dim)
    params["format"] = "png"
    url = image.getThumbURL(params)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


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
    max_frames: int = 9,
    progress_cb: Callable[[float, str], None] | None = None,
    lang: str = "sl",
) -> dict[str, Any]:
    """Run server-side multi-year compositing, spectral index analysis, Δ change & area stats on GEE.

    Returns the exact same frame / time-series / change-map bundle expected by
    Workspace 3 and Workspace 5 (AI Assistant), with zero heavy raster RAM on Streamlit Cloud.
    """
    if not is_gee_ready():
        raise RuntimeError("Google Earth Engine is not connected.")

    idx_key = str(index_code or "NDVI").strip().upper()
    vis_spec = get_gee_index_vis(idx_key)
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox_4326]
    center_lat = 0.5 * (min_lat + max_lat)
    bounds_4326 = [[min_lat, min_lon], [max_lat, max_lon]]
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], None, False)

    # Estimate AOI span to pick an efficient server-side reduction scale (30m–500m)
    span_km = max(abs(max_lat - min_lat) * 111.0, abs(max_lon - min_lon) * 111.0 * max(0.15, np.cos(np.radians(center_lat))))
    if sensor_mode == "modis" or span_km > 600.0:
        reduce_scale = 500
    elif span_km > 150.0:
        reduce_scale = 120
    elif span_km > 40.0:
        reduce_scale = 60
    else:
        reduce_scale = 30

    sy = max(1984, int(start_year))
    ey = max(sy, int(end_year))
    if sensor_mode == "modis":
        sy = max(2000, sy)
    elif sensor_mode == "sentinel2":
        sy = max(2017, sy)

    all_years = list(range(sy, ey + 1))
    cap = max(2, min(int(max_frames), 12))
    if len(all_years) > cap:
        pick_idx = np.linspace(0, len(all_years) - 1, cap, dtype=int)
        sampled_years = sorted(set(all_years[i] for i in pick_idx))
    else:
        sampled_years = all_years

    # Choose 1 complementary index for rich layer switching
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

    rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    index_images: list[tuple[int, str, Any, bool]] = []  # (year, date_str, ee.Image, reliable)
    excluded_archive_dates: list[str] = []

    total_y = len(sampled_years)
    for i, yr in enumerate(sampled_years):
        _prog(
            0.10 + 0.65 * (i / max(total_y, 1)),
            f"🌍 GEE strežniški mozaik {yr} ({i + 1}/{total_y})..."
            if lang == "sl"
            else f"🌍 GEE server-side composite {yr} ({i + 1}/{total_y})...",
        )
        w_start, w_end = _season_date_window(yr, target_season, center_lat)
        mid_date = f"{yr}-08-01" if center_lat >= -20.0 else f"{yr}-02-01"

        comp_img = None
        scene_cnt = 0
        plat_lbl = "MODIS"
        sensor_fam = "MODIS (500 m)"

        if sensor_mode == "modis" or (sensor_mode == "auto_fusion" and span_km > 800.0 and yr >= 2000):
            comp_img, scene_cnt = _gee_modis_composite(region, w_start, w_end)
            plat_lbl = "MODIS Terra"
            sensor_fam = "MODIS (500 m)"
        elif sensor_mode == "sentinel2" or (sensor_mode == "auto_fusion" and yr >= 2017):
            comp_img, scene_cnt = _gee_s2_composite(region, w_start, w_end)
            plat_lbl = "Sentinel-2"
            sensor_fam = "Sentinel-2 (10–20 m)"
            if comp_img is None and sensor_mode == "auto_fusion":
                comp_img, scene_cnt, plat_lbl = _gee_landsat_composite(region, w_start, w_end)
                sensor_fam = "Landsat (30 m)"
        else:
            comp_img, scene_cnt, plat_lbl = _gee_landsat_composite(region, w_start, w_end)
            sensor_fam = "Landsat (30 m)"

        if comp_img is None or scene_cnt == 0:
            continue

        idx_img = gee_compute_index(comp_img, idx_key)
        comp_idx_img = gee_compute_index(comp_img, comp_idx)

        # Server-side mean & median reduction
        reducer = ee.Reducer.mean().combine(ee.Reducer.median(), sharedInputs=True)
        stats = idx_img.reduceRegion(
            reducer=reducer,
            geometry=region,
            scale=reduce_scale,
            bestEffort=True,
            maxPixels=int(1e9),
        ).getInfo() or {}
        idx_mean = stats.get(f"{idx_key}_mean")
        idx_med = stats.get(f"{idx_key}_median")
        if idx_mean is None:
            continue
        idx_mean_f = round(float(idx_mean), 4)
        idx_med_f = round(float(idx_med if idx_med is not None else idx_mean), 4)

        is_reliable = yr >= 1990
        if not is_reliable:
            excluded_archive_dates.append(mid_date)

        # Lightweight PNG overlays via getThumbURL
        rgb_vis = {"bands": ["red", "green", "blue"], "min": 0.02, "max": 0.35, "gamma": 1.25}
        idx_vis_p = {"min": vis_spec["min"], "max": vis_spec["max"], "palette": vis_spec["palette"]}
        comp_vis_p = {"min": comp_vis["min"], "max": comp_vis["max"], "palette": comp_vis["palette"]}

        rgb_png_raw = _fetch_thumb_png(comp_img, rgb_vis, region, max_dim=768)
        idx_png_raw = _fetch_thumb_png(idx_img, idx_vis_p, region, max_dim=768)
        comp_png_raw = _fetch_thumb_png(comp_idx_img, comp_vis_p, region, max_dim=768)

        rgb_uri = _png_bytes_to_data_uri(rgb_png_raw)
        idx_uri = _png_bytes_to_data_uri(idx_png_raw)
        comp_uri = _png_bytes_to_data_uri(comp_png_raw)

        rgb_stamped = _stamp_png_bytes(rgb_png_raw, f"{mid_date} | GEE {plat_lbl} | RGB")
        idx_stamped = _stamp_png_bytes(idx_png_raw, f"{mid_date} | GEE {plat_lbl} | {idx_key}={idx_mean_f:.2f}")
        comp_stamped = _stamp_png_bytes(comp_png_raw, f"{mid_date} | GEE {plat_lbl} | {comp_idx}")

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
                "extra_overlays": {comp_idx: comp_uri},
                "extra_pngs": {comp_idx: comp_stamped},
                "extra_means": {comp_idx: None},
                "rgb_png_bytes": rgb_stamped,
                "index_png_bytes": idx_stamped,
                "index_grid_small": None,
                "ndwi_grid_small": None,
                "nir_grid_small": None,
            }
        )
        index_images.append((yr, mid_date, idx_img, is_reliable))

    if not frames:
        raise RuntimeError(
            "Google Earth Engine ni našel uporabnih ne-oblačnih mozaikov za izbrano obdobje in območje."
            if lang == "sl"
            else "Google Earth Engine found no usable cloud-free composites for the selected period and region."
        )

    ts_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Pick default reliable t1 (>= 1990) and latest t2 for server-side Δ Change Map + pixelArea() stats
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
            0.80,
            "🌍 GEE računa prostorsko spremembo (Δ) in površine v km²..."
            if lang == "sl"
            else "🌍 GEE computing spatial Δ change map and km² area statistics...",
        )
        _, d1, img_t1, _ = index_images[default_t1_idx]
        _, d2, img_t2, _ = index_images[-1]
        diff_img = img_t2.subtract(img_t1).rename("DELTA")

        d_pal = GEE_DELTA_PALETTES.get(delta_cmap, GEE_DELTA_PALETTES["RdBu"])
        delta_png = _fetch_thumb_png(
            diff_img,
            {"min": -delta_lim, "max": delta_lim, "palette": d_pal},
            region,
            max_dim=768,
        )
        delta_uri = _png_bytes_to_data_uri(delta_png)
        delta_bnds = bounds_4326

        # Server-side pixelArea() statistics (exact km² and ha!)
        thr = 0.15 if idx_key != "BAI" else 25.0
        px_area = ee.Image.pixelArea()
        dec_area_img = px_area.updateMask(diff_img.lt(-thr)).rename("dec_m2")
        inc_area_img = px_area.updateMask(diff_img.gt(thr)).rename("inc_m2")
        tot_area_img = px_area.updateMask(diff_img.mask()).rename("tot_m2")

        area_sums = (
            dec_area_img.addBands(inc_area_img)
            .addBands(tot_area_img)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=reduce_scale,
                bestEffort=True,
                maxPixels=int(1e9),
            )
            .getInfo()
            or {}
        )
        mean_diff_d = (
            diff_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=reduce_scale,
                bestEffort=True,
                maxPixels=int(1e9),
            ).getInfo()
            or {}
        )

        dec_km2 = float(area_sums.get("dec_m2") or 0.0) / 1e6
        inc_km2 = float(area_sums.get("inc_m2") or 0.0) / 1e6
        tot_km2 = max(1e-6, float(area_sums.get("tot_m2") or 0.0) / 1e6)
        delta_mean_v = float(mean_diff_d.get("DELTA") or (frames[-1]["index_mean"] - frames[default_t1_idx]["index_mean"]))

        change_summary = {
            "t1_date": d1,
            "t2_date": d2,
            "t1_mean": frames[default_t1_idx]["index_mean"],
            "t2_mean": frames[-1]["index_mean"],
            "delta_mean": round(delta_mean_v, 4),
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

    # Build crisp Timelapse GIFs using our existing builder
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
    }
