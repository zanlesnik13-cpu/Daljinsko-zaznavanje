"""Optional user registration, authentication, and profile persistence.

Registration and login are strictly optional — guest users can use the entire
application without creating an account. Registered users get persistent
per-user preferences (``data/users/<username>/profile.json``) secured with
PBKDF2-SHA256 password hashing (120,000 iterations) and automatic GitHub Cloud
synchronization via ``github_sync.py`` when configured.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
from typing import Any

from github_sync import github_push_file

_REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = _REPO_ROOT / "data"
AUTH_USERS_REL = "data/auth_users.json"
AUTH_USERS_PATH = _REPO_ROOT / AUTH_USERS_REL
USERS_DIR = DATA_DIR / "users"

PBKDF2_ITERATIONS = 120_000
_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")

DEFAULT_PROFILE_PREFS: dict[str, Any] = {
    "preferred_lang": "sl",
    "preferred_crs_mode": "auto",
    "preferred_resolution": 10,
    "preferred_gemini_model": "gemini-3.8-flash",
    "auto_save_session": True,
}


def normalize_username(raw_username: str) -> str:
    """Normalize a username to trimmed lowercase."""
    return str(raw_username or "").strip().lower()


def validate_username(username: str) -> tuple[bool, str]:
    """Validate normalized username (3–32 chars, letters/digits/._-)."""
    uname = normalize_username(username)
    if not uname or not _USERNAME_RE.match(uname):
        return False, "auth_err_username"
    return True, ""


def hash_password(password: str, *, salt_hex: str | None = None, iterations: int = PBKDF2_ITERATIONS) -> str:
    """Hash a plaintext password using PBKDF2-SHA256 (120,000 iterations)."""
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    return f"pbkdf2_sha256${int(iterations)}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2-SHA256 hash string."""
    if not password or not stored_hash:
        return False
    try:
        algo, iter_str, salt_hex, expected_hex = stored_hash.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iter_str)
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(dk.hex(), expected_hex)
    except Exception:
        return False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_users_store() -> dict[str, Any]:
    """Load ``data/auth_users.json`` safely, returning ``{"users": {}}`` if absent."""
    if not AUTH_USERS_PATH.is_file():
        return {"users": {}}
    try:
        data = json.loads(AUTH_USERS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("users"), dict):
            return data
    except Exception:
        pass
    return {"users": {}}


def _save_users_store(store: dict[str, Any], *, sync: bool = True, commit_msg: str | None = None) -> bool:
    """Write ``data/auth_users.json`` to disk and optionally push to GitHub Cloud."""
    AUTH_USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTH_USERS_PATH.write_text(
        json.dumps(store, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if sync:
        return github_push_file(
            AUTH_USERS_REL,
            commit_message=commit_msg or "chore(auth): update auth_users.json",
            show_toast=True,
        )
    return False


def get_user_dir(username: str) -> Path:
    """Return the local directory path ``data/users/<username>``."""
    uname = normalize_username(username)
    return USERS_DIR / uname


def get_user_profile_rel_path(username: str) -> str:
    """Return the repo-relative POSIX path ``data/users/<username>/profile.json``."""
    uname = normalize_username(username)
    return f"data/users/{uname}/profile.json"


def load_user_profile(username: str) -> dict[str, Any]:
    """Load a user's ``profile.json`` merged with defaults."""
    uname = normalize_username(username)
    profile = {
        "username": uname,
        "display_name": uname,
        **DEFAULT_PROFILE_PREFS,
    }
    if not uname:
        return profile

    prof_path = get_user_dir(uname) / "profile.json"
    if prof_path.is_file():
        try:
            loaded = json.loads(prof_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                profile.update(loaded)
        except Exception:
            pass
    return profile


def save_user_profile(
    username: str,
    updates: dict[str, Any],
    *,
    sync: bool = True,
    show_toast: bool = True,
) -> dict[str, Any]:
    """Persist updated preferences to ``data/users/<username>/profile.json`` and sync to GitHub."""
    uname = normalize_username(username)
    if not uname:
        raise ValueError("Username must not be empty.")

    profile = load_user_profile(uname)
    for k, v in updates.items():
        if v is not None:
            profile[k] = v
    profile["username"] = uname
    profile["updated_at"] = _utc_now_iso()

    user_dir = get_user_dir(uname)
    user_dir.mkdir(parents=True, exist_ok=True)
    prof_path = user_dir / "profile.json"
    prof_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if sync:
        github_push_file(
            get_user_profile_rel_path(uname),
            commit_message=f"chore(profile): update profile for {uname}",
            show_toast=show_toast,
        )
    return profile


def register_user(
    username: str,
    password: str,
    display_name: str = "",
    *,
    initial_prefs: dict[str, Any] | None = None,
    sync: bool = True,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Register a new optional user account.

    Returns ``(ok, status_key, user_profile_or_none)``.
    """
    uname = normalize_username(username)
    ok_u, err_u = validate_username(uname)
    if not ok_u:
        return False, err_u, None

    if not isinstance(password, str) or len(password) < 6:
        return False, "auth_err_password_short", None

    store = _load_users_store()
    users = store.setdefault("users", {})
    if uname in users:
        return False, "auth_err_exists", None

    clean_display = str(display_name or "").strip() or uname
    now_iso = _utc_now_iso()
    users[uname] = {
        "username": uname,
        "display_name": clean_display,
        "password_hash": hash_password(password),
        "created_at": now_iso,
    }
    _save_users_store(store, sync=sync, commit_msg=f"chore(auth): register user {uname}")

    prefs = dict(DEFAULT_PROFILE_PREFS)
    if initial_prefs:
        for k, v in initial_prefs.items():
            if v is not None:
                prefs[k] = v
    prefs["display_name"] = clean_display
    prefs["created_at"] = now_iso

    profile = save_user_profile(uname, prefs, sync=sync, show_toast=False)
    return True, "auth_register_ok", profile


def authenticate_user(username: str, password: str) -> tuple[bool, str, dict[str, Any] | None]:
    """Authenticate an existing user by username and password.

    Returns ``(ok, status_key, user_profile_or_none)``.
    """
    uname = normalize_username(username)
    if not uname or not password:
        return False, "auth_err_invalid_creds", None

    store = _load_users_store()
    record = store.get("users", {}).get(uname)
    if not isinstance(record, dict):
        return False, "auth_err_invalid_creds", None

    if not verify_password(password, str(record.get("password_hash") or "")):
        return False, "auth_err_invalid_creds", None

    profile = load_user_profile(uname)
    if record.get("display_name") and not profile.get("display_name"):
        profile["display_name"] = str(record["display_name"])
    return True, "auth_login_ok", profile


def sync_user_data_to_github(username: str) -> bool:
    """Manually push ``auth_users.json`` and all user JSON files to GitHub Cloud."""
    uname = normalize_username(username)
    pushed_any = False
    if AUTH_USERS_PATH.is_file():
        if github_push_file(AUTH_USERS_REL, commit_message="chore(sync): sync auth_users.json", show_toast=False):
            pushed_any = True
    if uname:
        udir = get_user_dir(uname)
        for fname in ("profile.json", "projects.json", "map_layouts.json", "last_session.json"):
            fpath = udir / fname
            if fpath.is_file():
                rel = f"data/users/{uname}/{fname}"
                if github_push_file(
                    rel,
                    commit_message=f"chore(sync): sync {fname} for {uname}",
                    show_toast=False,
                ):
                    pushed_any = True
    return pushed_any
