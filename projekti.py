"""Per-user project workspace storage, saved cartography layouts, AI chat persistence,
auto-saved last session, and 1-click JSON backup/restore with GitHub Cloud sync.

All user data is stored as lightweight JSON under ``data/users/<username>/``:
- ``projects.json`` — saved EO projects (AOI, sensor/date settings, QGIS-style
  ``MapLayoutConfig``, compact analysis summary, and text AI chat turns)
- ``map_layouts.json`` — custom QGIS-style cartography layout templates
- ``last_session.json`` — lightweight auto-saved workspace state for 1-click restore

Raw rasters, NumPy arrays, and binary PNG/PDF/GeoTIFF blobs are NEVER written into JSON.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
import uuid

from auth import (
    DEFAULT_PROFILE_PREFS,
    get_user_dir,
    load_user_profile,
    normalize_username,
    save_user_profile,
)
from github_sync import github_push_file

_MAX_CHAT_TURNS = 80
_MAX_STRING_LEN = 12_000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify_name(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    base = base[:24] or "project"
    return f"{base}-{uuid.uuid4().hex[:6]}"


def sanitize_json_payload(obj: Any, *, depth: int = 0) -> Any:
    """Recursively coerce a Python structure into lightweight, strictly JSON-safe primitives."""
    if depth > 8:
        return None
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            return None
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return None
    if isinstance(obj, str):
        if obj.startswith("data:image/"):
            return None
        return obj if len(obj) <= _MAX_STRING_LEN else obj[:_MAX_STRING_LEN] + "…"
    # Handle NumPy scalars gracefully without hard-requiring numpy import
    if hasattr(obj, "item") and callable(obj.item) and not isinstance(obj, (dict, list, tuple)):
        try:
            return sanitize_json_payload(obj.item(), depth=depth + 1)
        except Exception:
            return None
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            # Skip known binary/raster/frame/GIF keys
            if any(
                tok in ks.lower()
                for tok in (
                    "bytes",
                    "png",
                    "pdf",
                    "tif",
                    "gif",
                    "timelapse",
                    "frames",
                    "raster",
                    "dataset",
                    "dataarray",
                    "overlay_b64",
                    "overlay_uri",
                )
            ):
                continue
            sv = sanitize_json_payload(v, depth=depth + 1)
            out[ks] = sv
        return out
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_json_payload(x, depth=depth + 1) for x in list(obj)[:250]]
    return str(obj)[:400]


_REPLAY_PLAN_KEYS: tuple[str, ...] = (
    "action",
    "task_type",
    "location_name",
    "place_name",
    "lat",
    "lon",
    "radius_m",
    "bbox_4326",
    "target_epsg",
    "crs",
    "start_date",
    "end_date",
    "event_date",
    "target_season",
    "interval",
    "index",
    "index_name",
    "index_code",
    "primary_index",
    "complementary_indices",
    "sensor",
    "sensors",
    "fusion_mode",
    "resolution_m",
    "max_cloud",
    "include_comparison",
    "disaster_type",
    "layer_type",
    "engine",
    "user_prompt",
    "user_query",
)


def _extract_compact_replay_plan(raw_plan: Any) -> dict[str, Any]:
    """Build a strictly JSON-serializable, compact replay plan with zero binary/base64 blobs."""
    if not isinstance(raw_plan, dict):
        return {}
    out: dict[str, Any] = {}
    for k in _REPLAY_PLAN_KEYS:
        val = raw_plan.get(k)
        if val is not None:
            out[k] = sanitize_json_payload(val)
    # Canonical aliases so downstream replay always finds core keys cleanly
    if "location_name" not in out and out.get("place_name"):
        out["location_name"] = out["place_name"]
    if "place_name" not in out and out.get("location_name"):
        out["place_name"] = out["location_name"]
    if "index" not in out and (out.get("index_name") or out.get("primary_index") or out.get("index_code")):
        out["index"] = out.get("index_name") or out.get("primary_index") or out.get("index_code")
    if "index_name" not in out and out.get("index"):
        out["index_name"] = out["index"]
    if "action" not in out and out.get("task_type"):
        out["action"] = out["task_type"]
    if "sensors" not in out and out.get("sensor"):
        out["sensors"] = [out["sensor"]]
    return out


def sanitize_chat_messages(messages: Any) -> list[dict[str, Any]]:
    """Return a compact, JSON-safe list of AI chat turns (strips heavy visual blobs, keeps replay_plan)."""
    if not isinstance(messages, (list, tuple)):
        return []
    cleaned: list[dict[str, Any]] = []
    for raw in list(messages)[-_MAX_CHAT_TURNS:]:
        if not isinstance(raw, dict):
            continue
        role = "user" if raw.get("role") == "user" else "assistant"
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        entry: dict[str, Any] = {
            "role": role,
            "content": content[:_MAX_STRING_LEN],
        }
        if raw.get("action"):
            entry["action"] = str(raw["action"])
        if raw.get("result_id"):
            entry["result_id"] = str(raw["result_id"])[:120]
        if raw.get("fallback_note"):
            entry["fallback_note"] = str(raw["fallback_note"])[:500]

        raw_spec = raw.get("replay_spec")
        if isinstance(raw_spec, dict) and raw_spec:
            spec_clean = {
                str(k): sanitize_json_payload(v)
                for k, v in raw_spec.items()
                if v is not None
            }
            if spec_clean:
                entry["replay_spec"] = spec_clean

        raw_replay = raw.get("replay_plan") if isinstance(raw.get("replay_plan"), dict) else raw.get("plan")
        compact_plan = _extract_compact_replay_plan(raw_replay)
        if isinstance(raw.get("plan"), dict):
            plan_merged = _extract_compact_replay_plan(raw.get("plan"))
            plan_merged.update(compact_plan)
            compact_plan = plan_merged
        if compact_plan:
            entry["plan"] = dict(compact_plan)
            entry["replay_plan"] = dict(compact_plan)

        res_meta = raw.get("result_meta")
        if isinstance(res_meta, dict):
            compact_meta = {
                k: sanitize_json_payload(res_meta.get(k))
                for k in ("is_visual_analysis", "summary_metrics", "engine", "resolved_place", "crs")
                if res_meta.get(k) is not None
            }
            if compact_meta:
                entry["result_meta"] = compact_meta
        elif (
            entry.get("action") == "run_analysis"
            or bool(entry.get("replay_spec"))
            or bool(compact_plan.get("task_type") or compact_plan.get("action"))
        ):
            entry["result_meta"] = {"is_visual_analysis": True}

        cleaned.append(entry)
    return cleaned


def _projects_path(username: str) -> tuple[Path, str]:
    uname = normalize_username(username)
    return get_user_dir(uname) / "projects.json", f"data/users/{uname}/projects.json"


def _layouts_path(username: str) -> tuple[Path, str]:
    uname = normalize_username(username)
    return get_user_dir(uname) / "map_layouts.json", f"data/users/{uname}/map_layouts.json"


def _last_session_path(username: str) -> tuple[Path, str]:
    uname = normalize_username(username)
    return get_user_dir(uname) / "last_session.json", f"data/users/{uname}/last_session.json"


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1. SAVED EO PROJECTS (data/users/<username>/projects.json)
# ---------------------------------------------------------------------------
def list_user_projects(username: str) -> list[dict[str, Any]]:
    """Return all saved projects for ``username``, sorted newest-updated first."""
    uname = normalize_username(username)
    if not uname:
        return []
    path, _ = _projects_path(uname)
    raw = _read_json_file(path, {"projects": []})
    projs = raw.get("projects", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    valid = [p for p in projs if isinstance(p, dict) and p.get("id") and p.get("name")]
    valid.sort(key=lambda p: str(p.get("updated_at") or p.get("created_at") or ""), reverse=True)
    return valid


def get_user_project(username: str, project_id: str) -> dict[str, Any] | None:
    """Return a single saved project by ID, or ``None`` if not found."""
    for p in list_user_projects(username):
        if str(p.get("id")) == str(project_id):
            return p
    return None


def save_user_project(
    username: str,
    name: str,
    note: str,
    state_payload: dict[str, Any],
    *,
    overwrite_id: str | None = None,
    sync: bool = True,
    show_toast: bool = True,
) -> dict[str, Any]:
    """Save or overwrite a user's EO project and immediately sync to GitHub Cloud."""
    uname = normalize_username(username)
    if not uname:
        raise ValueError("Username must not be empty.")
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("Project name must not be empty.")

    projects = list_user_projects(uname)
    now_iso = _utc_now_iso()
    existing_idx: int | None = None
    existing_created: str = now_iso
    proj_id = str(overwrite_id or "").strip() or _slugify_name(clean_name)

    if overwrite_id:
        for idx, p in enumerate(projects):
            if str(p.get("id")) == str(overwrite_id):
                existing_idx = idx
                existing_created = str(p.get("created_at") or now_iso)
                proj_id = str(p["id"])
                break

    payload = state_payload if isinstance(state_payload, dict) else {}
    entry: dict[str, Any] = {
        "id": proj_id,
        "name": clean_name[:120],
        "note": str(note or "").strip()[:1000],
        "created_at": existing_created,
        "updated_at": now_iso,
        "aoi": sanitize_json_payload(payload.get("aoi") or {}),
        "settings": sanitize_json_payload(payload.get("settings") or {}),
        "cartography_layout": sanitize_json_payload(payload.get("cartography_layout") or {}),
        "analysis_summary": sanitize_json_payload(payload.get("analysis_summary") or {}),
        "ai_chat_messages": sanitize_chat_messages(payload.get("ai_chat_messages") or []),
    }

    if existing_idx is not None:
        projects[existing_idx] = entry
    else:
        projects.insert(0, entry)

    path, rel_path = _projects_path(uname)
    _write_json_file(path, {"version": 1, "username": uname, "projects": projects})

    if sync:
        action = "update" if existing_idx is not None else "save"
        github_push_file(
            rel_path,
            commit_message=f"chore(projects): {action} project '{clean_name[:40]}' for {uname}",
            show_toast=show_toast,
        )
    return entry


def delete_user_project(
    username: str,
    project_id: str,
    *,
    sync: bool = True,
    show_toast: bool = True,
) -> bool:
    """Delete a saved project by ID and immediately sync ``projects.json`` to GitHub Cloud."""
    uname = normalize_username(username)
    if not uname or not project_id:
        return False

    projects = list_user_projects(uname)
    filtered = [p for p in projects if str(p.get("id")) != str(project_id)]
    if len(filtered) == len(projects):
        return False

    path, rel_path = _projects_path(uname)
    _write_json_file(path, {"version": 1, "username": uname, "projects": filtered})

    if sync:
        github_push_file(
            rel_path,
            commit_message=f"chore(projects): delete project {project_id} for {uname}",
            show_toast=show_toast,
        )
    return True


# ---------------------------------------------------------------------------
# 2. SAVED CARTOGRAPHY LAYOUT TEMPLATES (data/users/<username>/map_layouts.json)
# ---------------------------------------------------------------------------
def list_map_layouts(username: str) -> dict[str, dict[str, Any]]:
    """Return ``{layout_name: {"name": ..., "updated_at": ..., "config": {...}}}``."""
    uname = normalize_username(username)
    if not uname:
        return {}
    path, _ = _layouts_path(uname)
    raw = _read_json_file(path, {"layouts": {}})
    layouts = raw.get("layouts", {}) if isinstance(raw, dict) else {}
    if not isinstance(layouts, dict):
        return {}
    return {
        str(k): v
        for k, v in layouts.items()
        if isinstance(v, dict) and isinstance(v.get("config"), dict)
    }


def save_map_layout(
    username: str,
    layout_name: str,
    layout_config: dict[str, Any],
    *,
    sync: bool = True,
    show_toast: bool = True,
) -> dict[str, Any]:
    """Save or overwrite a custom QGIS-style ``MapLayoutConfig`` and sync to GitHub Cloud."""
    uname = normalize_username(username)
    if not uname:
        raise ValueError("Username must not be empty.")
    clean_name = str(layout_name or "").strip()[:80]
    if not clean_name:
        raise ValueError("Layout template name must not be empty.")

    layouts = list_map_layouts(uname)
    entry = {
        "name": clean_name,
        "updated_at": _utc_now_iso(),
        "config": sanitize_json_payload(layout_config or {}),
    }
    layouts[clean_name] = entry

    path, rel_path = _layouts_path(uname)
    _write_json_file(path, {"version": 1, "username": uname, "layouts": layouts})

    if sync:
        github_push_file(
            rel_path,
            commit_message=f"chore(layouts): save map layout '{clean_name[:40]}' for {uname}",
            show_toast=show_toast,
        )
    return entry


def delete_map_layout(
    username: str,
    layout_name: str,
    *,
    sync: bool = True,
    show_toast: bool = True,
) -> bool:
    """Delete a saved custom cartography layout template and sync to GitHub Cloud."""
    uname = normalize_username(username)
    clean_name = str(layout_name or "").strip()
    if not uname or not clean_name:
        return False

    layouts = list_map_layouts(uname)
    if clean_name not in layouts:
        return False
    del layouts[clean_name]

    path, rel_path = _layouts_path(uname)
    _write_json_file(path, {"version": 1, "username": uname, "layouts": layouts})

    if sync:
        github_push_file(
            rel_path,
            commit_message=f"chore(layouts): delete map layout '{clean_name[:40]}' for {uname}",
            show_toast=show_toast,
        )
    return True


# ---------------------------------------------------------------------------
# 3. AUTO-SAVE LAST SESSION (data/users/<username>/last_session.json)
# ---------------------------------------------------------------------------
def save_last_session(
    username: str,
    session_payload: dict[str, Any],
    *,
    sync: bool = True,
    show_toast: bool = False,
) -> dict[str, Any]:
    """Save lightweight active session state for 1-click restore on next login."""
    uname = normalize_username(username)
    if not uname:
        raise ValueError("Username must not be empty.")
    payload = session_payload if isinstance(session_payload, dict) else {}
    record = {
        "version": 1,
        "username": uname,
        "saved_at": _utc_now_iso(),
        "aoi": sanitize_json_payload(payload.get("aoi") or {}),
        "settings": sanitize_json_payload(payload.get("settings") or {}),
        "cartography_layout": sanitize_json_payload(payload.get("cartography_layout") or {}),
        "analysis_summary": sanitize_json_payload(payload.get("analysis_summary") or {}),
        "ai_chat_messages": sanitize_chat_messages(payload.get("ai_chat_messages") or []),
    }
    path, rel_path = _last_session_path(uname)
    _write_json_file(path, record)

    if sync:
        github_push_file(
            rel_path,
            commit_message=f"chore(session): auto-save session for {uname}",
            show_toast=show_toast,
        )
    return record


def load_last_session(username: str) -> dict[str, Any] | None:
    """Load the user's auto-saved last session, or ``None`` if unavailable."""
    uname = normalize_username(username)
    if not uname:
        return None
    path, _ = _last_session_path(uname)
    raw = _read_json_file(path, None)
    if isinstance(raw, dict) and (raw.get("aoi") or raw.get("settings") or raw.get("ai_chat_messages")):
        return raw
    return None


# ---------------------------------------------------------------------------
# 3B. MULTI-CONVERSATION AI CHAT THREADS (data/users/<username>/ai_chats.json)
# ---------------------------------------------------------------------------
_MAX_CHAT_THREADS = 60

_TITLE_FILLER_PREFIXES: tuple[str, ...] = (
    "prosim izvedi analizo ",
    "prosim naredi analizo ",
    "izvedi analizo ",
    "naredi analizo ",
    "prosim analiziraj ",
    "analiziraj ",
    "prosim pokaži mi ",
    "prosim pokazi mi ",
    "pokaži mi ",
    "pokazi mi ",
    "pokaži ",
    "pokazi ",
    "izračunaj mi ",
    "izracunaj mi ",
    "izračunaj ",
    "izracunaj ",
    "please run an analysis of ",
    "run an analysis of ",
    "please analyze ",
    "analyze ",
    "show me ",
    "show ",
    "calculate ",
)


def _ai_chats_path(username: str) -> tuple[Path, str]:
    uname = normalize_username(username)
    return get_user_dir(uname) / "ai_chats.json", f"data/users/{uname}/ai_chats.json"


def auto_title_from_prompt(prompt: str, max_len: int = 44) -> str:
    """Derive a crisp, readable ~44-char thread title from the first user prompt."""
    clean = " ".join(str(prompt or "").strip().split())
    if not clean:
        return "Nov pogovor"
    low = clean.lower()
    for pref in _TITLE_FILLER_PREFIXES:
        if low.startswith(pref) and len(clean) > len(pref) + 3:
            clean = clean[len(pref) :].strip()
            break
    clean = clean.strip(" .,;:!?\"'«»")
    if not clean:
        return "Nov pogovor"
    clean = clean[0].upper() + clean[1:]
    if len(clean) <= max_len:
        return clean
    cut = clean[: max_len - 1].rsplit(" ", 1)[0].strip()
    if len(cut) < 12:
        cut = clean[: max_len - 1].strip()
    return cut + "…"


def create_chat_thread(first_prompt: str = "", *, title: str | None = None) -> dict[str, Any]:
    """Create a fresh AI chat thread dictionary."""
    now_iso = _utc_now_iso()
    clean_title = str(title).strip()[:60] if (title and str(title).strip()) else auto_title_from_prompt(first_prompt)
    return {
        "id": uuid.uuid4().hex[:8],
        "title": clean_title,
        "created_at": now_iso,
        "updated_at": now_iso,
        "pinned": False,
        "messages": [],
        "last_analysis_context": None,
        "last_map_layout": None,
    }


def sanitize_chat_thread(thread: dict[str, Any]) -> dict[str, Any]:
    """Strip all binary/base64 raster/GIF payloads from a thread dict while preserving text & KPIs."""
    now_iso = _utc_now_iso()
    tid = str(thread.get("id") or uuid.uuid4().hex[:8]).strip()[:32]
    msgs = sanitize_chat_messages(thread.get("messages") or [])
    raw_title = str(thread.get("title") or "").strip()
    if not raw_title:
        first_u = next((m["content"] for m in msgs if m.get("role") == "user" and m.get("content")), "")
        raw_title = auto_title_from_prompt(first_u)
    ctx_raw = thread.get("last_analysis_context")
    layout_raw = thread.get("last_map_layout")
    return {
        "id": tid,
        "title": raw_title[:60],
        "created_at": str(thread.get("created_at") or now_iso),
        "updated_at": str(thread.get("updated_at") or now_iso),
        "pinned": bool(thread.get("pinned", False)),
        "messages": msgs,
        "last_analysis_context": sanitize_json_payload(ctx_raw) if isinstance(ctx_raw, dict) else None,
        "last_map_layout": sanitize_json_payload(layout_raw) if isinstance(layout_raw, dict) else None,
    }


def sort_chat_threads(threads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort chat threads by ``(pinned DESC, updated_at DESC)``."""
    return sorted(
        threads,
        key=lambda th: (
            1 if bool(th.get("pinned", False)) else 0,
            str(th.get("updated_at") or th.get("created_at") or ""),
        ),
        reverse=True,
    )


def load_user_chat_threads(username: str) -> list[dict[str, Any]]:
    """Load all persisted AI chat threads for ``username``, sorted by ``(pinned DESC, updated_at DESC)``."""
    uname = normalize_username(username)
    if not uname:
        return []
    path, _ = _ai_chats_path(uname)
    raw = _read_json_file(path, {"threads": []})
    raw_threads = raw.get("threads", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    cleaned = [sanitize_chat_thread(th) for th in raw_threads if isinstance(th, dict) and th.get("id")]
    return sort_chat_threads(cleaned)[:_MAX_CHAT_THREADS]


def save_user_chat_threads(
    username: str,
    threads: list[dict[str, Any]],
    *,
    sync: bool = True,
    show_toast: bool = False,
) -> list[dict[str, Any]]:
    """Persist lightweight sanitized AI chat threads to ``data/users/<username>/ai_chats.json`` and GitHub Cloud."""
    uname = normalize_username(username)
    if not uname:
        raise ValueError("Username must not be empty.")
    dedup: dict[str, dict[str, Any]] = {}
    for th in threads or []:
        if not isinstance(th, dict):
            continue
        clean_th = sanitize_chat_thread(th)
        dedup[clean_th["id"]] = clean_th
    ordered = sort_chat_threads(list(dedup.values()))[:_MAX_CHAT_THREADS]
    path, rel_path = _ai_chats_path(uname)
    _write_json_file(path, {"version": 1, "username": uname, "threads": ordered})
    if sync:
        github_push_file(
            rel_path,
            commit_message=f"chore(chats): update AI chat history ({len(ordered)} threads) for {uname}",
            show_toast=show_toast,
        )
    return ordered


def rename_chat_thread(
    username: str | None,
    thread_id: str,
    new_title: str,
    *,
    threads: list[dict[str, Any]] | None = None,
    sync: bool = True,
) -> list[dict[str, Any]]:
    """Rename a chat thread by ``thread_id`` and persist if ``username`` is provided."""
    clean_title = str(new_title or "").strip()[:60]
    cur = list(threads) if threads is not None else (load_user_chat_threads(username) if username else [])
    if not clean_title:
        return sort_chat_threads(cur)
    for th in cur:
        if str(th.get("id")) == str(thread_id):
            th["title"] = clean_title
            th["updated_at"] = _utc_now_iso()
            break
    if username:
        return save_user_chat_threads(username, cur, sync=sync, show_toast=False)
    return sort_chat_threads(cur)


def toggle_pin_chat_thread(
    username: str | None,
    thread_id: str,
    *,
    threads: list[dict[str, Any]] | None = None,
    sync: bool = True,
) -> list[dict[str, Any]]:
    """Toggle ``pinned`` status of a chat thread and persist if ``username`` is provided."""
    cur = list(threads) if threads is not None else (load_user_chat_threads(username) if username else [])
    for th in cur:
        if str(th.get("id")) == str(thread_id):
            th["pinned"] = not bool(th.get("pinned", False))
            break
    if username:
        return save_user_chat_threads(username, cur, sync=sync, show_toast=False)
    return sort_chat_threads(cur)


def delete_chat_thread(
    username: str | None,
    thread_id: str,
    *,
    threads: list[dict[str, Any]] | None = None,
    sync: bool = True,
) -> list[dict[str, Any]]:
    """Delete a chat thread by ``thread_id`` and persist if ``username`` is provided."""
    cur = list(threads) if threads is not None else (load_user_chat_threads(username) if username else [])
    remaining = [th for th in cur if str(th.get("id")) != str(thread_id)]
    if username:
        return save_user_chat_threads(username, remaining, sync=sync, show_toast=False)
    return sort_chat_threads(remaining)


# ---------------------------------------------------------------------------
# 4. 1-CLICK LOCAL JSON BACKUP & RESTORE BUNDLE
# ---------------------------------------------------------------------------
def export_user_bundle_json(username: str) -> bytes:
    """Export a user's profile, projects, map layouts, AI chats, and last session as UTF-8 JSON bytes."""
    uname = normalize_username(username)
    bundle = {
        "bundle_version": 2,
        "app": "Daljinsko-zaznavanje",
        "exported_at": _utc_now_iso(),
        "username": uname,
        "profile": load_user_profile(uname),
        "projects": list_user_projects(uname),
        "map_layouts": list_map_layouts(uname),
        "ai_chats": load_user_chat_threads(uname),
        "last_session": load_last_session(uname),
    }
    return (json.dumps(bundle, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def import_user_bundle_json(
    username: str,
    json_bytes: bytes | str,
    *,
    sync: bool = True,
) -> dict[str, int]:
    """Import a previously exported JSON backup bundle for ``username`` and sync to GitHub Cloud.

    Merges projects by ``id``, map layouts by ``name``, and AI chat threads by ``id``,
    updates profile preferences, and restores ``last_session`` if present.
    """
    uname = normalize_username(username)
    if not uname:
        raise ValueError("Username must not be empty.")

    raw_text = json_bytes.decode("utf-8") if isinstance(json_bytes, (bytes, bytearray)) else str(json_bytes)
    bundle = json.loads(raw_text)
    if not isinstance(bundle, dict):
        raise ValueError("Invalid JSON backup bundle format.")

    imported_projects = 0
    imported_layouts = 0
    imported_chats = 0
    restored_session = 0

    # 1. Profile preferences (keep current username intact)
    prof_in = bundle.get("profile")
    if isinstance(prof_in, dict):
        allowed_pref_keys = set(DEFAULT_PROFILE_PREFS.keys()) | {"display_name", "auto_save_session"}
        pref_updates = {k: prof_in[k] for k in allowed_pref_keys if k in prof_in and prof_in[k] is not None}
        if pref_updates:
            save_user_profile(uname, pref_updates, sync=sync, show_toast=False)

    # 2. Projects (merge by id)
    projs_in = bundle.get("projects")
    if isinstance(projs_in, list) and projs_in:
        existing = {str(p["id"]): p for p in list_user_projects(uname)}
        for p in projs_in:
            if not isinstance(p, dict) or not p.get("name"):
                continue
            pid = str(p.get("id") or _slugify_name(str(p["name"])))
            existing[pid] = {
                "id": pid,
                "name": str(p["name"])[:120],
                "note": str(p.get("note") or "")[:1000],
                "created_at": str(p.get("created_at") or _utc_now_iso()),
                "updated_at": str(p.get("updated_at") or _utc_now_iso()),
                "aoi": sanitize_json_payload(p.get("aoi") or {}),
                "settings": sanitize_json_payload(p.get("settings") or {}),
                "cartography_layout": sanitize_json_payload(p.get("cartography_layout") or {}),
                "analysis_summary": sanitize_json_payload(p.get("analysis_summary") or {}),
                "ai_chat_messages": sanitize_chat_messages(p.get("ai_chat_messages") or []),
            }
            imported_projects += 1
        merged_list = sorted(
            existing.values(),
            key=lambda x: str(x.get("updated_at") or x.get("created_at") or ""),
            reverse=True,
        )
        p_path, p_rel = _projects_path(uname)
        _write_json_file(p_path, {"version": 1, "username": uname, "projects": merged_list})
        if sync:
            github_push_file(
                p_rel,
                commit_message=f"chore(projects): import {imported_projects} projects for {uname}",
                show_toast=False,
            )

    # 3. Map layouts (merge by name)
    layouts_in = bundle.get("map_layouts")
    if isinstance(layouts_in, dict) and layouts_in:
        existing_l = list_map_layouts(uname)
        for lname, lval in layouts_in.items():
            if not isinstance(lval, dict) or not isinstance(lval.get("config"), dict):
                continue
            clean_ln = str(lval.get("name") or lname).strip()[:80]
            if not clean_ln:
                continue
            existing_l[clean_ln] = {
                "name": clean_ln,
                "updated_at": str(lval.get("updated_at") or _utc_now_iso()),
                "config": sanitize_json_payload(lval["config"]),
            }
            imported_layouts += 1
        l_path, l_rel = _layouts_path(uname)
        _write_json_file(l_path, {"version": 1, "username": uname, "layouts": existing_l})
        if sync:
            github_push_file(
                l_rel,
                commit_message=f"chore(layouts): import {imported_layouts} map layouts for {uname}",
                show_toast=False,
            )

    # 4. AI chat threads (merge by id)
    chats_in = bundle.get("ai_chats")
    if isinstance(chats_in, list) and chats_in:
        existing_c = {str(th["id"]): th for th in load_user_chat_threads(uname)}
        for th in chats_in:
            if not isinstance(th, dict):
                continue
            clean_th = sanitize_chat_thread(th)
            existing_c[clean_th["id"]] = clean_th
            imported_chats += 1
        save_user_chat_threads(uname, list(existing_c.values()), sync=sync, show_toast=False)

    # 5. Last session
    ls_in = bundle.get("last_session")
    if isinstance(ls_in, dict) and (ls_in.get("aoi") or ls_in.get("settings") or ls_in.get("ai_chat_messages")):
        save_last_session(uname, ls_in, sync=sync, show_toast=False)
        restored_session = 1

    return {
        "projects": imported_projects,
        "map_layouts": imported_layouts,
        "ai_chats": imported_chats,
        "restored_session": restored_session,
    }
