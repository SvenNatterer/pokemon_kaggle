import os
import json
import sys
import shutil
import subprocess
import time
from urllib.parse import quote
from flask import Flask, jsonify, redirect, send_from_directory, request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.arena_control import ArenaController
from src.arena_core import ArenaStore, EVALUATION_FILE, atomic_write_json, discover_participants, rank_participants, read_json, utc_now
from src.arena_match import load_holdout_results
from src.model_paths import resolve_deck_model_base, resolve_deck_model_path

app = Flask(__name__, static_folder=BASE_DIR)

train_process = None
evaluation_process = None
arena_store = ArenaStore()
arena_controller = ArenaController(arena_store)

@app.after_request
def add_cors_headers(response):
    # Allow the static dashboard on port 8080 to talk to this API on 8050.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/")
def index():
    return redirect("/dashboard/dashboard.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(BASE_DIR, path)

@app.route("/api/status", methods=["GET"])
def status():
    participants = discover_participants()
    matches = arena_store.matches()
    board = rank_participants(participants, matches, load_holdout_results())
    evaluation = read_json(EVALUATION_FILE, {"state": "idle"})
    arena = arena_controller.status()
    return jsonify({
        "arena": arena,
        "state": arena.get("state"),
        "running": arena.get("state") == "running",
        "leaderboard": board,
        "participants": [participant.to_dict() for participant in participants],
        "current_match": arena.get("current_match"),
        "recent_matches": matches[-20:][::-1],
        "evaluation": evaluation,
        "errors": [participant.to_dict() for participant in participants if participant.load_status != "loadable"],
    })

@app.route("/api/start", methods=["POST"])
def start():
    success, message = arena_controller.start()
    return jsonify({"success": success, "message": message, "arena": arena_controller.status()}), (200 if success else 409)

@app.route("/api/start", methods=["OPTIONS"])
def start_options():
    return ("", 204)

@app.route("/api/pause", methods=["POST"])
def pause():
    success, message = arena_controller.pause()
    return jsonify({"success": success, "message": message, "arena": arena_controller.status()}), (200 if success else 409)

@app.route("/api/pause", methods=["OPTIONS"])
def pause_options():
    return ("", 204)

@app.route("/api/stop", methods=["POST"])
def stop():
    success, message = arena_controller.stop()
    return jsonify({"success": success, "message": message, "arena": arena_controller.status()})


@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    if data.get("confirmation") != "RESET ARENA":
        return jsonify({"success": False, "message": "Exact confirmation 'RESET ARENA' is required."}), 400
    include_replays = bool(data.get("include_replays", False))
    success, message = arena_controller.reset()
    if include_replays:
        replay_root = os.path.join(BASE_DIR, "replays")
        if os.path.isdir(replay_root):
            for dirpath, _, filenames in os.walk(replay_root):
                for filename in filenames:
                    if filename.endswith(".json"):
                        os.unlink(os.path.join(dirpath, filename))
        message += " Arena replays were also removed as confirmed."
    return jsonify({"success": success, "message": message, "arena": arena_controller.status()})

@app.route("/api/reset", methods=["OPTIONS"])
def reset_options():
    return ("", 204)


@app.route("/api/refresh", methods=["GET"])
def refresh():
    return status()


@app.route("/api/leaderboard", methods=["GET"])
def leaderboard():
    return jsonify({"rows": rank_participants(discover_participants(), arena_store.matches(), load_holdout_results())})


@app.route("/api/participants", methods=["GET"])
def participants():
    values = discover_participants()
    return jsonify({"participants": [value.to_dict() for value in values]})

@app.route('/api/available_decks', methods=['GET'])
def get_available_decks():
    import glob
    decks = []
    for filepath in glob.glob("decks/deck_*.csv"):
        filename = os.path.basename(filepath)
        decks.append(filename)
    return jsonify({"decks": sorted(decks)})

WATCHED_FILE = os.path.join(BASE_DIR, "decks", "watched_models.json")
replay_metadata_cache = {}

@app.route('/api/watched', methods=['GET'])
def get_watched():
    try:
        if os.path.exists(WATCHED_FILE):
            with open(WATCHED_FILE, 'r') as f:
                return jsonify(json.load(f))
    except Exception:
        pass
    return jsonify({"watched": []})

@app.route('/api/watched', methods=['POST'])
def set_watched():
    data = request.json or {}
    watched = data.get('watched', [])
    try:
        os.makedirs(os.path.dirname(WATCHED_FILE), exist_ok=True)
        with open(WATCHED_FILE, 'w') as f:
            json.dump({"watched": watched}, f, indent=2)
        return jsonify({"success": True, "watched": watched})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/replays', methods=['GET'])
def get_replays():
    replay_roots = [
        ("Arena replays", os.path.join(BASE_DIR, "replays")),
    ]
    replays = []
    seen = set()

    for group, root in replay_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith(".json"):
                    continue

                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, BASE_DIR).replace(os.sep, "/")
                if rel_path in seen:
                    continue
                seen.add(rel_path)

                stat = os.stat(abs_path)
                cache_key = (stat.st_mtime_ns, stat.st_size)
                cached = replay_metadata_cache.get(abs_path)
                if cached and cached["key"] == cache_key:
                    metadata = cached["metadata"]
                    snapshots = cached["snapshots"]
                else:
                    metadata = {}
                    snapshots = None
                    try:
                        with open(abs_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            snapshots = len(data)
                            if data and isinstance(data[0], dict):
                                metadata = data[0].get("metadata") or {}
                    except Exception:
                        pass
                    replay_metadata_cache[abs_path] = {
                        "key": cache_key,
                        "metadata": metadata,
                        "snapshots": snapshots,
                    }

                replay_url = "/" + quote(rel_path, safe="/")
                replays.append({
                    "group": group,
                    "path": rel_path,
                    "url": replay_url,
                    "name": filename,
                    "snapshots": snapshots,
                    "metadata": metadata,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })

    replays.sort(key=lambda item: item["mtime"], reverse=True)
    return jsonify({"replays": replays})


@app.route('/api/evaluation/start', methods=['POST'])
def start_evaluation():
    global evaluation_process
    if evaluation_process is not None and evaluation_process.poll() is None:
        return jsonify({"success": False, "message": "An evaluation is already running."}), 409
    data = request.get_json(silent=True) or {}
    bot_id = str(data.get("bot_id") or "")
    games = int(data.get("games", 30))
    participant = next((item for item in discover_participants() if item.bot_id == bot_id), None)
    if participant is None or not participant.enabled or participant.load_status != "loadable":
        return jsonify({"success": False, "message": "Select an enabled, loadable PPO bot."}), 400
    if participant.bot_type != "ppo" or not participant.model_path:
        return jsonify({"success": False, "message": "Holdout evaluation currently requires a PPO model."}), 400
    if arena_controller.status().get("state") == "running":
        arena_controller.pause()
    os.makedirs(os.path.join(BASE_DIR, "arena_data"), exist_ok=True)
    evaluation_process = subprocess.Popen(
        [sys.executable, "-m", "src.evaluation_worker", "--bot-id", bot_id,
         "--model", participant.model_path, "--games", str(games)],
        cwd=BASE_DIR, stdout=open(os.path.join(BASE_DIR, "arena_data", "evaluation.log"), "a"),
        stderr=subprocess.STDOUT,
    )
    return jsonify({"success": True, "message": "Holdout evaluation started.", "evaluation": read_json(EVALUATION_FILE, {"state": "starting"})})


@app.route('/api/evaluation', methods=['GET'])
def get_evaluation():
    return jsonify(read_json(EVALUATION_FILE, {"state": "idle"}))

@app.route('/api/train_custom', methods=['POST'])
def train_custom():
    data = request.json
    
    learning_deck = data.get("learning_deck")
    mode = data.get("mode", "self_play")
    timesteps = data.get("timesteps", 500000)
    reward_config = data.get("reward_config", {})
    model_title = data.get("model_title", "").strip()
    algo = data.get("algo", "RecurrentPPO")
    ent_start = data.get("ent_start", 0.02)
    ent_end = data.get("ent_end", 0.005)
    
    if not learning_deck:
        return jsonify({"error": "learning_deck is required"}), 400
        
    opponent_deck = data.get("opponent_deck") if mode == "vs_bot" else learning_deck
    
    # Extract IDs
    learn_id = learning_deck.split("_")[1].split(".")[0]
    opp_id = opponent_deck.split("_")[1].split(".")[0]
    
    if model_title:
        # Sanitize it basically
        model_title = "".join([c for c in model_title if c.isalnum() or c in ['_', '-']])
        learning_model = f"models/{model_title}"
    else:
        learning_model = resolve_deck_model_base(learn_id)
        
    opponent_model = resolve_deck_model_path(opp_id, include_variants=False) or f"models/ppo_deck_{opp_id}.zip"
    
    global train_process
    
    # Optional: we can kill the train process if it's running so it doesn't conflict
    if train_process is not None and train_process.poll() is None:
        train_process.terminate()
        
    if "models/holdout" in opponent_model.replace("\\", "/"):
        return jsonify({"error": "Holdout opponents are evaluation-only and cannot be used for training."}), 400

    cmd = [
        "venv/bin/python", "src/train.py",
        "--deck", f"decks/{learning_deck}",
        "--model-name", learning_model,
        "--opp-deck", f"decks/{opponent_deck}",
        "--opp-model", opponent_model,
        "--timesteps", str(timesteps),
        "--num-envs", "8",
        "--ent-coef", str(ent_start),
    ]
    target_zip = learning_model if learning_model.endswith(".zip") else f"{learning_model}.zip"
    if os.path.exists(target_zip):
        cmd.append("--continue-existing")
    train_log = open("server_train.log", "w")
    train_process = subprocess.Popen(cmd, stdout=train_log, stderr=subprocess.STDOUT)
    # Reset progress file
    try:
        with open("decks/training_progress.json", "w") as f:
            json.dump({"current": 0, "total": timesteps, "status": "starting"}, f)
    except:
        pass
        
    return jsonify({"status": "training started", "cmd": " ".join(cmd)})

@app.route('/api/cancel_custom', methods=['POST'])
def cancel_custom_training():
    global train_process
    try:
        # 1. Terminate the process if we have the reference
        if train_process is not None and train_process.poll() is None:
            train_process.terminate()
            train_process = None
            
        # 2. Forcefully kill any stray background training processes (e.g., if server was restarted)
        import os
        os.system("pkill -f 'src/train.py'")
            
        # 3. Update the progress file so frontend knows it stopped
        try:
            import json
            with open("decks/training_progress.json", "w") as f:
                json.dump({"current": 0, "total": 0, "status": "not_started"}, f)
        except:
            pass
            
        return jsonify({"status": "cancelled"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/training_progress', methods=['GET'])
def get_training_progress():
    try:
        if not os.path.exists("decks/training_progress.json"):
            return jsonify({"current": 0, "total": 0, "status": "not_started"})
        with open("decks/training_progress.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
