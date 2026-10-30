"""Project persistence (doc §5-6): SQLite for projects, correction versions and exports; large arrays
(STFTs, renders) as .npz files in a cache folder, named by the hash of everything they depend on.

Editable project state (parameters, selection, references, events) is one JSON document per project:
small, versioned with the app, and reloaded as a whole. Per-frame curves are not stored as rows; they
are recomputed in milliseconds from the cached STFT.
"""
import datetime
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from . import __version__

SCHEMA = """
CREATE TABLE projects(
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, source_path TEXT NOT NULL, source_hash TEXT NOT NULL,
    sample_rate INTEGER, channels INTEGER, frames INTEGER, state_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT, updated_at TEXT, app_version TEXT);
CREATE TABLE corrections(
    id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    correction_hash TEXT NOT NULL, params_json TEXT, render_path TEXT, created_at TEXT, active INTEGER);
CREATE TABLE exports(
    id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    correction_id INTEGER REFERENCES corrections(id), output_path TEXT, format TEXT, sample_rate INTEGER,
    bit_depth TEXT, created_at TEXT);
"""
MIGRATIONS = [SCHEMA]  # index i upgrades user_version i -> i+1


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def cache_key(*parts):
    """Stable hash of the inputs a cached result depends on (arrays hashed by content)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(np.ascontiguousarray(p).tobytes() if isinstance(p, np.ndarray) else repr(p).encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = self.root / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "deflutter.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        for i, script in enumerate(MIGRATIONS[version:], start=version):
            self.db.executescript(f"BEGIN; {script} PRAGMA user_version = {i + 1}; COMMIT;")

    # ---- projects
    def create_project(self, name, source_path, source_hash, sample_rate, channels, frames, state):
        cur = self.db.execute(
            "INSERT INTO projects(name, source_path, source_hash, sample_rate, channels, frames, state_json,"
            " created_at, updated_at, app_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (name, str(source_path), source_hash, sample_rate, channels, frames, json.dumps(state),
             _now(), _now(), __version__))
        self.db.commit()
        return cur.lastrowid

    def project(self, pid):
        return self.db.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()

    def recent(self, limit=30):
        return self.db.execute("SELECT * FROM projects ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()

    def save_state(self, pid, state):
        self.db.execute("UPDATE projects SET state_json = ?, updated_at = ?, app_version = ? WHERE id = ?",
                        (json.dumps(state), _now(), __version__, pid))
        self.db.commit()

    def relocate(self, pid, source_path):
        self.db.execute("UPDATE projects SET source_path = ? WHERE id = ?", (str(source_path), pid))
        self.db.commit()

    def delete_project(self, pid):
        self.db.execute("DELETE FROM projects WHERE id = ?", (pid,))
        self.db.commit()

    # ---- correction versions and exports
    def add_correction(self, pid, correction_hash, params, render_path):
        self.db.execute("UPDATE corrections SET active = 0 WHERE project_id = ?", (pid,))
        cur = self.db.execute(
            "INSERT INTO corrections(project_id, correction_hash, params_json, render_path, created_at, active)"
            " VALUES (?,?,?,?,?,1)", (pid, correction_hash, json.dumps(params), str(render_path), _now()))
        self.db.commit()
        return cur.lastrowid

    def active_correction(self, pid):
        return self.db.execute("SELECT * FROM corrections WHERE project_id = ? AND active = 1", (pid,)).fetchone()

    def add_export(self, pid, correction_id, output_path, fmt, sample_rate, bit_depth):
        self.db.execute(
            "INSERT INTO exports(project_id, correction_id, output_path, format, sample_rate, bit_depth,"
            " created_at) VALUES (?,?,?,?,?,?,?)",
            (pid, correction_id, str(output_path), fmt, sample_rate, bit_depth, _now()))
        self.db.commit()

    # ---- array cache (safe to call from worker threads: files only, no SQLite)
    def cache_path(self, key):
        return self.cache / f"{key}.npz"

    def load_arrays(self, key):
        p = self.cache_path(key)
        if not p.exists():
            return None
        with np.load(p) as z:
            return {k: z[k] for k in z.files}

    def save_arrays(self, key, **arrays):
        tmp = self.cache / f"{key}.tmp.npz"
        np.savez(tmp, **arrays)
        tmp.replace(self.cache_path(key))  # atomic: never leaves a half-written cache entry
        return self.cache_path(key)
