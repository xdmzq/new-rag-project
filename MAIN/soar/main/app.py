from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin
import json
import uuid
import traceback
import threading
from typing import Dict, Any, List
import os
import subprocess
import re
from datetime import datetime

from sqlalchemy.exc import SQLAlchemyError


def _apply_project_dotenv(override: bool = False) -> None:
    """Load PROJECT_ROOT/.env into os.environ for direct main/app.py launches."""
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if not os.path.isfile(env_path):
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=override)
        return
    except Exception:
        pass

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if override or key not in os.environ:
            os.environ[key] = val


_apply_project_dotenv(override=False)

from paths import PROJECT_ROOT

from api import ACTIVE_KB_CATEGORIES, CATEGORY_LABELS, run_rag_session

_MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_sf = os.environ.get("SEMICONDUCTOR_MAIN_STATIC")
_tf = os.environ.get("SEMICONDUCTOR_MAIN_TEMPLATES")
if _sf and _tf:
    _static_folder, _template_folder = _sf, _tf
else:
    _static_folder = os.path.join(_MAIN_DIR, "static")
    _template_folder = os.path.join(_MAIN_DIR, "templates")

app = Flask(
    __name__,
    static_folder=_static_folder,
    template_folder=_template_folder,
)
CORS(app)

# =================  Config =================
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your_very_secret_key_here")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "debug"}


DEBUG_MODE = _env_flag("APP_DEBUG", _env_flag("FLASK_DEBUG", False))
app.config["TEMPLATES_AUTO_RELOAD"] = DEBUG_MODE
app.jinja_env.auto_reload = DEBUG_MODE
if DEBUG_MODE:
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

def _sqlite_uri(sqlite_path: str) -> str:
    abs_path = os.path.abspath(sqlite_path)
    return "sqlite:///" + abs_path.replace("\\", "/")

def _resolve_db_uri() -> str:
    # DB_MODE: auto | mysql | sqlite
    db_mode = os.environ.get("DB_MODE", "auto").strip().lower()
    sqlite_path = os.environ.get("SQLITE_PATH", os.path.join(str(PROJECT_ROOT), "main", "semicDatabase.sqlite3"))
    # If SQLITE_PATH is a relative path, resolve it relative to PROJECT_ROOT
    if not os.path.isabs(sqlite_path):
        sqlite_path = os.path.join(str(PROJECT_ROOT), sqlite_path)
    sqlite_uri = _sqlite_uri(sqlite_path)

    mysql_uri = os.environ.get("DATABASE_URL", "").strip()

    if db_mode == "sqlite":
        return sqlite_uri
    if db_mode == "mysql":
        if not mysql_uri:
            raise RuntimeError("DB_MODE=mysql 但未设置 DATABASE_URL")
        return mysql_uri

    return mysql_uri or sqlite_uri

app.config["SQLALCHEMY_DATABASE_URI"] = _resolve_db_uri()
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = None

# ================= Database Models=================

class User(UserMixin, db.Model):
    __tablename__ = 'semiconductor'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    nickname = db.Column(db.String(100))
    avatar = db.Column(db.String(255), default='default.png')
    
    history_records = db.relationship('ChatHistory', backref='user', lazy=True)

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('semiconductor.id'), nullable=False)
    run_id = db.Column(db.String(100), nullable=False)
    prompt = db.Column(db.Text)
    category = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


RUNS: Dict[str, Dict[str, Any]] = {}

LOGS: Dict[str, Dict[str, Any]] = {}
LOGS_LOCK = threading.Lock()
OUTPUT_DIR = os.path.join(str(PROJECT_ROOT), "output_history")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_META_SUFFIX = ".meta.json"

V2_KB_MEDIA_DIRS = [
    "SpaceRAG_v2",
    "生物科技与健康_RAG",
    "ITSoftwareRAG",
    "SmartManufacturingRAG_v2",
    "半导体_RAG",
    "军工_RAG",
    "互联网_RAG",
    "通信与电子硬件_RAG",
    "新材料_RAG",
    "能源环保_RAG",
    "消费品与现代服务_RAG",
]
ALLOWED_OPEN_ROOTS = [
    os.path.abspath(os.path.join(str(PROJECT_ROOT), "data")),
    os.path.abspath(os.path.join(str(PROJECT_ROOT), "pre_data")),
    os.path.abspath(os.path.join(str(PROJECT_ROOT), "SpaceRAG")),
]
ALLOWED_OPEN_ROOTS.extend(
    os.path.abspath(os.path.join(str(PROJECT_ROOT), directory))
    for directory in V2_KB_MEDIA_DIRS
)
MEDIA_ALLOWED_ROOTS = [
    os.path.abspath(os.path.join(str(PROJECT_ROOT), "main", "static", "img")),
    os.path.abspath(os.path.join(str(PROJECT_ROOT), "pre_data")),
    os.path.abspath(os.path.join(str(PROJECT_ROOT), "SpaceRAG")),
]
MEDIA_ALLOWED_ROOTS.extend(
    os.path.abspath(os.path.join(str(PROJECT_ROOT), directory))
    for directory in V2_KB_MEDIA_DIRS
)


def _is_within_dir(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(path), os.path.normcase(root)]) == os.path.normcase(root)
    except ValueError:
        return False

def _init_log_channel(run_id: str):
    with LOGS_LOCK:
        if run_id not in LOGS:
            LOGS[run_id] = {"items": [], "done": False}

def _append_log(run_id: str, pretty_line: str, step: Dict[str, Any]):
    try:
        filepath = os.path.join(OUTPUT_DIR, f"{run_id}.txt")
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(pretty_line + '\n')
    except IOError as e:
        print(f"Error: Could not write to log file for run_id {run_id}: {e}")

    with LOGS_LOCK:
        chan = LOGS.get(run_id)
        if not chan:
            chan = {"items": [], "done": False}
            LOGS[run_id] = chan
        seq = len(chan["items"])
        record = {"seq": seq, "pretty": pretty_line, "step": step}
        chan["items"].append(record)

def _mark_done(run_id: str):
    with LOGS_LOCK:
        if run_id in LOGS:
            LOGS[run_id]["done"] = True


def _run_meta_path(run_id: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{run_id}{RUN_META_SUFFIX}")


def _save_run_meta(run_id: str, params: Dict[str, Any]) -> None:
    payload = {
        "run_id": run_id,
        "prompt": params.get("prompt", ""),
        "category": params.get("category", "space"),
        "retrieval_mode": params.get("retrieval_mode", "v2"),
        "started_at": params.get("started_at") or datetime.now().isoformat(timespec="seconds"),
    }
    try:
        with open(_run_meta_path(run_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[WARN] Failed to save run metadata for {run_id}: {e}")


def _load_run_meta(run_id: str) -> Dict[str, Any]:
    params = dict(RUNS.get(run_id, {}))
    meta_path = _run_meta_path(run_id)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                params.update(stored)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[WARN] Failed to load run metadata for {run_id}: {e}")
    return params


def _list_recent_runs(limit: int = 18) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        names = [
            name for name in os.listdir(OUTPUT_DIR)
            if name.endswith(RUN_META_SUFFIX)
        ]
    except OSError:
        names = []

    names.sort(
        key=lambda name: os.path.getmtime(os.path.join(OUTPUT_DIR, name)),
        reverse=True,
    )
    for name in names[:limit]:
        run_id = name[:-len(RUN_META_SUFFIX)]
        meta = _load_run_meta(run_id)
        records.append(
            {
                "run_id": run_id,
                "prompt": meta.get("prompt", ""),
                "category": meta.get("category", "space"),
                "category_label": CATEGORY_LABELS.get(meta.get("category", "space"), meta.get("category", "space")),
                "retrieval_mode": meta.get("retrieval_mode", "v2"),
                "started_at": meta.get("started_at", ""),
            }
        )
    return records


def _build_category_cards() -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for category, label in CATEGORY_LABELS.items():
        cards.append(
            {
                "id": category,
                "label": label,
                "active": category in ACTIVE_KB_CATEGORIES,
            }
        )
    return cards


_RUN_LINE_RE = re.compile(
    r"^\[RUN (?P<run>[^\]]+)\] phase=(?P<phase>[^|,]+)"
    r"(?:, gen=(?P<gen>\d+))?"
    r"(?: \| (?P<message>.*))?$"
)


def _serialize_step(step: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "phase": step.get("phase"),
        "message": step.get("message", ""),
        "run_id": step.get("run_id"),
    }
    if "gen" in step:
        payload["gen"] = step.get("gen")
    if "best_plan" in step:
        payload["best_plan"] = step.get("best_plan")
    for key in ("answer", "references", "related_questions", "text"):
        if key in step:
            payload[key] = step.get(key)
    return payload


def _parse_pretty_line(seq: int, line: str) -> Dict[str, Any]:
    text = line.strip()
    payload: Dict[str, Any] = {"seq": seq, "pretty": text}

    match = _RUN_LINE_RE.match(text)
    if not match:
        if text.startswith("===== [RAG] Session started"):
            payload["phase"] = "session_start"
        elif "[OK] RAG Session completed" in text:
            payload["phase"] = "final"
        elif "[ERROR] RAG Session error" in text:
            payload["phase"] = "error"
        else:
            payload["phase"] = "raw"
        return payload

    payload["phase"] = match.group("phase")
    payload["run_id"] = match.group("run")
    message = match.group("message") or ""
    payload["message"] = message
    if match.group("gen"):
        payload["gen"] = int(match.group("gen"))

    if message and payload["phase"] in {"result", "answer_chunk"}:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            payload["data"] = data
    return payload


def _serialize_log_entry(seq: int, pretty: str, step: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = _parse_pretty_line(seq, pretty)
    if step:
        payload["step"] = _serialize_step(step)
    return payload

def _format_pretty_line(run_id: str, step: Dict[str, Any]) -> str:
    phase = step.get("phase")
    gen_idx = step.get("gen")
    msg = step.get("message", "")
    best_plan = step.get("best_plan")
    line = (f"[RUN {run_id[:8]}] phase={phase}" + (f", gen={gen_idx}" if gen_idx is not None else "") + (f" | {msg}" if msg else ""))
    if best_plan:
        plan_str = json.dumps(best_plan, ensure_ascii=False, separators=(',', ':'))
        line += f" | Plan: {plan_str}"
    return line

def start_background_run(run_id: str, params: dict):
    """Background thread: execute RAG session"""
    try:
        category = params.get("category", "space")
        retrieval_mode = params.get("retrieval_mode", "v2")
        head = f"===== [RAG] Session started (run_id={run_id}, KB={category}, mode={retrieval_mode}) ====="
        _append_log(run_id, head, {"phase": "meta", "message": f"started ({category}, {retrieval_mode})", "run_id": run_id})

        gen = run_rag_session(
            prompt=params["prompt"],
            category=category,
            retrieval_mode=retrieval_mode,
        )
        for step in gen:
            pretty = _format_pretty_line(run_id, step)
            _append_log(run_id, pretty, step)

        tail = f"===== [OK] RAG Session completed (run_id={run_id}) ====="
        _append_log(run_id, tail, {"phase": "final", "message": "completed", "run_id": run_id})
    except Exception as e:
        err = f"===== [ERROR] RAG Session error (run_id={run_id}): {e} ====="
        _append_log(run_id, err, {"phase": "error", "message": str(e), "run_id": run_id})
        traceback.print_exc()
    finally:
        _mark_done(run_id)

# ================= Routes =================

@app.route('/login', methods=['GET', 'POST'])
def login():
    return redirect(url_for('workspace'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    return redirect(url_for('workspace'))


@app.route('/logout')
def logout():
    return redirect(url_for('workspace'))


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    return redirect(url_for('workspace'))


@app.route('/', methods=['GET'])
def workspace():
    return _render_workspace()


@app.route('/workspace', methods=['GET'])
def workspace_alias():
    return redirect(url_for('workspace'))


@app.route('/session/<run_id>', methods=['GET'])
def session_page(run_id):
    return _render_workspace(run_id=run_id)


def _render_workspace(run_id: str | None = None):
    history = _list_recent_runs()
    initial_prompt = request.args.get('prompt', '')
    current_run = _load_run_meta(run_id) if run_id else {}
    if current_run:
        prompt = current_run.get("prompt") or initial_prompt
        category = current_run.get("category", "space")
        retrieval_mode = current_run.get("retrieval_mode", "v2")
    else:
        prompt = initial_prompt
        category = request.args.get('category', 'space')
        retrieval_mode = request.args.get('retrieval_mode', 'v2')

    invalid = False
    if run_id:
        log_path = os.path.join(OUTPUT_DIR, f"{run_id}.txt")
        invalid = not (os.path.exists(log_path) or current_run or run_id in RUNS)

    return render_template(
        'workspace.html',
        run_id=run_id or "",
        prompt=prompt,
        invalid=invalid,
        history=history,
        category=category,
        retrieval_mode=retrieval_mode,
        categories=_build_category_cards(),
        active_categories=sorted(ACTIVE_KB_CATEGORIES),
    )

@app.route('/solve', methods=['POST'])
def solve_rag():
    try:
        prompt = request.form.get('query', '').strip()
        if not prompt:
             prompt = request.form.get('vrp_problem', '').strip()
        if not prompt:
            return jsonify({"status": "error", "message": "query is required"}), 400
        category = request.form.get('category', 'space')
        
        retrieval_mode = "v2"

        params = {
            "prompt": prompt,
            "category": category,
            "retrieval_mode": retrieval_mode,
            "started_at": request.form.get('started_at', '') or datetime.now().isoformat(timespec="seconds"),
        }
        run_id = str(uuid.uuid4())
        RUNS[run_id] = params
        _init_log_channel(run_id)
        _save_run_meta(run_id, params)

        print(f"\n[OK] Public workspace started run: {run_id} (KB: {category}, mode: {retrieval_mode})")

        threading.Thread(
            target=start_background_run,
            args=(run_id, dict(params)),
            daemon=True
        ).start()

        return jsonify({"status": "accepted", "run_id": run_id})
    except Exception as e:
        print("[ERROR] /solve error:", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/logs/<run_id>', methods=['GET'])
def fetch_logs(run_id: str):
    try:
        cursor = int(request.args.get('cursor', 0))
    except Exception:
        cursor = 0

    with LOGS_LOCK:
        chan = LOGS.get(run_id)
        if chan:
            items = chan["items"]
            done = chan["done"]
            batch = items[cursor:cursor + 200]
            next_cursor = cursor + len(batch)
            payload = [
                _serialize_log_entry(rec["seq"], rec["pretty"], rec.get("step"))
                for rec in batch
            ]
            return jsonify({"status": "ok", "logs": payload, "next_cursor": next_cursor, "done": done})

    filepath = os.path.join(OUTPUT_DIR, f"{run_id}.txt")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
            # In multi-worker deployments, /logs may hit another worker that has no in-memory channel.
            # Infer completion from file tail markers instead of forcing done=True.
            file_done = any(
                ("[OK] RAG Session completed" in line) or ("[ERROR] RAG Session error" in line)
                for line in lines[-20:]
            )
            if cursor < len(lines):
                batch_lines = lines[cursor:cursor + 200]
                payload = [
                    _serialize_log_entry(cursor + i, line)
                    for i, line in enumerate(batch_lines)
                ]
                return jsonify({"status": "ok", "logs": payload, "next_cursor": cursor + len(batch_lines), "done": file_done})
            else:
                return jsonify({"status": "ok", "logs": [], "next_cursor": cursor, "done": file_done})
        except IOError as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "error", "message": "Invalid run_id"}), 404

@app.route('/media/<path:rel_path>', methods=['GET'])
def serve_media(rel_path: str):
    if not rel_path:
        return jsonify({'status': 'error', 'message': 'path is required'}), 400
    if os.path.isabs(rel_path):
        return jsonify({'status': 'error', 'message': 'absolute path is not allowed'}), 400
    if '..' in rel_path.replace('\\', '/').split('/'):
        return jsonify({'status': 'error', 'message': 'invalid path'}), 400

    candidate = os.path.abspath(os.path.join(str(PROJECT_ROOT), rel_path))
    allowed = any(_is_within_dir(candidate, root) for root in MEDIA_ALLOWED_ROOTS)
    if not allowed:
        return jsonify({'status': 'error', 'message': 'path is outside allowed roots'}), 403
    if not os.path.exists(candidate) or not os.path.isfile(candidate):
        return jsonify({'status': 'error', 'message': 'file not found'}), 404

    return send_file(candidate, conditional=True)


@app.route('/open-file-location', methods=['POST'])
def open_file_location():
    payload = request.get_json(silent=True) or request.form or {}
    raw_path = str(payload.get('file_path', '')).strip()

    if not raw_path:
        return jsonify({'status': 'error', 'message': 'file_path is required'}), 400
    if not os.path.isabs(raw_path):
        return jsonify({'status': 'error', 'message': 'file_path must be an absolute path'}), 400

    normalized_path = os.path.abspath(raw_path)
    if not os.path.exists(normalized_path):
        return jsonify({'status': 'error', 'message': 'file not found'}), 404
    if not os.path.isfile(normalized_path):
        return jsonify({'status': 'error', 'message': 'path is not a file'}), 400

    try:
        # Windows paths are case-insensitive; normalize case to avoid false 403.
        normalized_casefold = os.path.normcase(normalized_path)
        in_allowed_root = False
        for allowed_root in ALLOWED_OPEN_ROOTS:
            allowed_root_casefold = os.path.normcase(allowed_root)
            if os.path.commonpath([normalized_casefold, allowed_root_casefold]) == allowed_root_casefold:
                in_allowed_root = True
                break
    except ValueError:
        in_allowed_root = False
    if not in_allowed_root:
        return jsonify({'status': 'error', 'message': 'path is outside allowed root'}), 403

    if os.name != 'nt':
        return jsonify({'status': 'error', 'message': 'only supported on Windows'}), 501

    try:
        proc = subprocess.run(
            ['explorer.exe', '/select,', normalized_path],
            check=False,
            shell=False,
            timeout=10,
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'failed to open explorer: {e}'}), 500

    return jsonify({'status': 'ok', 'message': 'opened', 'file_path': normalized_path})
@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"})


@app.route('/favicon.ico', methods=['GET'])
def favicon():
    icon_path = os.path.join(app.static_folder, "logo", "logo.png")
    if os.path.exists(icon_path):
        return send_file(icon_path, mimetype='image/png')
    return ('', 204)

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
        except SQLAlchemyError as e:
            print(f"[WARN] Database init failed (db.create_all): {e}")
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=DEBUG_MODE,
        use_reloader=DEBUG_MODE,
        threaded=True,
    )
