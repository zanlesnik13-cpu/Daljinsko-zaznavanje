"""GitHub Cloud persistence module for optional user profiles and saved data.

Synchronizes JSON files under ``data/`` (such as ``data/auth_users.json`` and
``data/users/<username>/profile.json``) with a GitHub repository using the
GitHub Contents REST API (``https://api.github.com/repos/{owner}/{repo}/contents/{path}``).

All operations are strictly fault-tolerant: when ``GITHUB_TOKEN`` is not configured
or the VM/cloud container is offline, functions degrade gracefully to local disk
storage without raising unhandled exceptions.
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any

import requests

_REPO_ROOT = Path(__file__).resolve().parent
_API_BASE = "https://api.github.com"
_TIMEOUT_SEC = 12


def _read_secret_or_env(key: str, default: str = "") -> str:
    """Read a configuration value from ``st.secrets`` first, then ``os.environ``."""
    try:
        import streamlit as st

        if key in st.secrets:
            val = str(st.secrets[key] or "").strip()
            if val:
                return val
    except Exception:
        pass
    val = str(os.environ.get(key, "") or "").strip()
    return val if val else default


def detect_github_repo_from_git(repo_root: Path | None = None) -> str:
    """Auto-detect ``owner/repo`` from ``.git/config`` origin URL if available."""
    root = repo_root or _REPO_ROOT
    git_cfg = root / ".git" / "config"
    if not git_cfg.is_file():
        return ""
    try:
        text = git_cfg.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""

    # Match both HTTPS (https://github.com/owner/repo.git) and SSH (git@github.com:owner/repo.git)
    patterns = (
        r"github\.com[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?\s*$",
        r"github\.com[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
    )
    for line in text.splitlines():
        if "github.com" not in line:
            continue
        for pat in patterns:
            m = re.search(pat, line.strip())
            if m:
                repo = m.group(1).strip()
                if repo.endswith(".git"):
                    repo = repo[:-4]
                if "/" in repo:
                    return repo
    return ""


def get_github_config() -> dict[str, str]:
    """Return active GitHub sync configuration ``{token, repo, branch}``."""
    token = _read_secret_or_env("GITHUB_TOKEN", "")
    repo = _read_secret_or_env("GITHUB_REPO", "") or detect_github_repo_from_git()
    branch = _read_secret_or_env("GITHUB_BRANCH", "main") or "main"
    return {"token": token, "repo": repo, "branch": branch}


def is_github_sync_configured() -> bool:
    """Return ``True`` when both ``GITHUB_TOKEN`` and ``owner/repo`` are available."""
    cfg = get_github_config()
    return bool(cfg["token"] and cfg["repo"] and "/" in cfg["repo"])


def get_github_sync_status(lang: str = "sl") -> dict[str, Any]:
    """Return a structured status dict suitable for UI badges and diagnostics."""
    cfg = get_github_config()
    configured = bool(cfg["token"] and cfg["repo"] and "/" in cfg["repo"])
    if lang == "en":
        badge_text = "☁️ GitHub Cloud" if configured else "💾 Local"
        detail = (
            f"Connected to {cfg['repo']} ({cfg['branch']})"
            if configured
            else "Local storage only (set GITHUB_TOKEN in secrets for cloud sync)"
        )
    else:
        badge_text = "☁️ GitHub Cloud" if configured else "💾 Lokalno"
        detail = (
            f"Povezano z {cfg['repo']} ({cfg['branch']})"
            if configured
            else "Samo lokalna hramba (dodaj GITHUB_TOKEN v secrets za oblačno sinhronizacijo)"
        )
    return {
        "configured": configured,
        "repo": cfg["repo"],
        "branch": cfg["branch"],
        "badge_text": badge_text,
        "detail": detail,
    }


def _build_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Daljinsko-Zaznavanje-App",
    }


def _normalize_rel_path(rel_path: str) -> str:
    """Normalize a repo-relative POSIX path without leading slashes or traversal."""
    clean = str(rel_path).replace("\\", "/").lstrip("/")
    parts = [p for p in clean.split("/") if p and p not in (".", "..")]
    return "/".join(parts)


def _notify_cloud_toast() -> None:
    """Display a subtle non-blocking toast on successful GitHub cloud push."""
    try:
        import streamlit as st

        lang = str(st.session_state.get("lang", "sl"))
        msg = "☁️ Saved to GitHub Cloud!" if lang == "en" else "☁️ Shranjeno v GitHub oblak!"
        st.toast(msg)
    except Exception:
        pass


def _set_last_sync_error(err: str | None) -> None:
    """Record or clear the latest GitHub sync diagnostic message in Streamlit session state."""
    try:
        import streamlit as st

        if err:
            st.session_state["_gh_last_error"] = str(err)
        else:
            st.session_state.pop("_gh_last_error", None)
    except Exception:
        pass


def _fetch_remote_sha(url: str, headers: dict[str, str], branch: str) -> str | None:
    """Fetch the current blob SHA for a file on GitHub, returning None if absent."""
    get_resp = requests.get(
        url,
        headers=headers,
        params={"ref": branch},
        timeout=_TIMEOUT_SEC,
    )
    if get_resp.status_code == 200:
        payload_meta = get_resp.json()
        if isinstance(payload_meta, dict):
            return payload_meta.get("sha")
    return None


def github_push_file(
    local_rel_path: str,
    commit_message: str | None = None,
    *,
    show_toast: bool = True,
) -> bool:
    """Push a local file (relative to repository root) to GitHub via Contents API.

    Automatically retries once on HTTP ``409 Conflict`` (stale SHA) and stores any
    failure reason in ``st.session_state["_gh_last_error"]`` without raising.
    """
    cfg = get_github_config()
    if not (cfg["token"] and cfg["repo"] and "/" in cfg["repo"]):
        return False

    rel_posix = _normalize_rel_path(local_rel_path)
    if not rel_posix:
        return False

    local_path = _REPO_ROOT / rel_posix
    if not local_path.is_file():
        return False

    try:
        raw_bytes = local_path.read_bytes()
    except OSError as exc:
        _set_last_sync_error(f"Local read error ({rel_posix}): {exc}")
        return False

    url = f"{_API_BASE}/repos/{cfg['repo']}/contents/{rel_posix}"
    headers = _build_headers(cfg["token"])
    content_b64 = base64.b64encode(raw_bytes).decode("ascii")
    msg = commit_message or f"chore(sync): update {rel_posix}"

    last_err = ""
    for attempt in range(2):
        try:
            existing_sha = _fetch_remote_sha(url, headers, cfg["branch"])
        except Exception as exc:
            _set_last_sync_error(f"GitHub GET error ({rel_posix}): {exc}")
            return False

        body: dict[str, Any] = {
            "message": msg,
            "content": content_b64,
            "branch": cfg["branch"],
        }
        if existing_sha:
            body["sha"] = existing_sha

        try:
            put_resp = requests.put(url, headers=headers, json=body, timeout=_TIMEOUT_SEC)
            if put_resp.status_code in (200, 201):
                _set_last_sync_error(None)
                if show_toast:
                    _notify_cloud_toast()
                return True
            if put_resp.status_code == 409 and attempt == 0:
                # Stale SHA conflict -> re-fetch fresh SHA and retry once
                continue
            try:
                err_detail = str(put_resp.json().get("message") or "").strip()
            except Exception:
                err_detail = (put_resp.text or "")[:160].strip()
            last_err = f"HTTP {put_resp.status_code} ({rel_posix}): {err_detail}"
            break
        except Exception as exc:
            last_err = f"Network error ({rel_posix}): {exc}"
            break

    if last_err:
        _set_last_sync_error(last_err)
    return False


def pull_user_data_from_github(username: str) -> dict[str, bool]:
    """Pull a user's ``profile.json``, ``projects.json``, ``map_layouts.json``, ``last_session.json``, and ``ai_chats.json``."""
    uname = str(username or "").strip().lower()
    if not uname or not is_github_sync_configured():
        return {}
    files = ("profile.json", "projects.json", "map_layouts.json", "last_session.json", "ai_chats.json")
    return {fn: github_pull_file(f"data/users/{uname}/{fn}") for fn in files}


def github_pull_file(remote_rel_path: str) -> bool:
    """Download a single file from GitHub Contents API and write it locally.

    Returns ``True`` if written locally, ``False`` otherwise. Never raises.
    """
    cfg = get_github_config()
    if not (cfg["token"] and cfg["repo"] and "/" in cfg["repo"]):
        return False

    rel_posix = _normalize_rel_path(remote_rel_path)
    if not rel_posix:
        return False

    url = f"{_API_BASE}/repos/{cfg['repo']}/contents/{rel_posix}"
    headers = _build_headers(cfg["token"])

    try:
        resp = requests.get(
            url,
            headers=headers,
            params={"ref": cfg["branch"]},
            timeout=_TIMEOUT_SEC,
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        if not isinstance(data, dict) or data.get("type") != "file":
            return False

        content_b64 = data.get("content")
        if content_b64:
            raw_bytes = base64.b64decode(content_b64)
        elif data.get("download_url"):
            dl_resp = requests.get(data["download_url"], headers=headers, timeout=_TIMEOUT_SEC)
            if dl_resp.status_code != 200:
                return False
            raw_bytes = dl_resp.content
        else:
            return False

        local_path = _REPO_ROOT / rel_posix
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(raw_bytes)
        return True
    except Exception:
        return False


def _restore_directory_recursive(
    remote_dir_rel: str,
    cfg: dict[str, str],
    headers: dict[str, str],
    restored: list[str],
    depth: int = 0,
) -> None:
    """Recursively restore JSON/text files inside a GitHub directory."""
    if depth > 4:
        return
    rel_posix = _normalize_rel_path(remote_dir_rel)
    url = f"{_API_BASE}/repos/{cfg['repo']}/contents/{rel_posix}"
    try:
        resp = requests.get(
            url,
            headers=headers,
            params={"ref": cfg["branch"]},
            timeout=_TIMEOUT_SEC,
        )
        if resp.status_code != 200:
            return
        entries = resp.json()
        if not isinstance(entries, list):
            return
    except Exception:
        return

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        etype = entry.get("type")
        epath = str(entry.get("path") or "")
        if not epath:
            continue
        if etype == "file":
            if github_pull_file(epath):
                restored.append(epath)
        elif etype == "dir":
            _restore_directory_recursive(epath, cfg, headers, restored, depth=depth + 1)


def github_restore_all_on_startup(force: bool = False) -> dict[str, Any]:
    """Restore ``data/auth_users.json`` and ``data/users/**`` from GitHub once per session.

    Uses ``st.session_state["_gh_restored"]`` to avoid redundant API calls on reruns.
    Returns a summary dict ``{"configured": bool, "restored": list[str], "skipped": bool}``.
    """
    try:
        import streamlit as st

        if not force and st.session_state.get("_gh_restored"):
            return {"configured": is_github_sync_configured(), "restored": [], "skipped": True}
    except Exception:
        st = None  # type: ignore[assignment]

    cfg = get_github_config()
    if not (cfg["token"] and cfg["repo"] and "/" in cfg["repo"]):
        if st is not None:
            try:
                st.session_state["_gh_restored"] = True
            except Exception:
                pass
        return {"configured": False, "restored": [], "skipped": False}

    restored: list[str] = []
    if github_pull_file("data/auth_users.json"):
        restored.append("data/auth_users.json")

    headers = _build_headers(cfg["token"])
    _restore_directory_recursive("data/users", cfg, headers, restored)

    if st is not None:
        try:
            st.session_state["_gh_restored"] = True
        except Exception:
            pass

    return {"configured": True, "restored": restored, "skipped": False}
