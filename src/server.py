import os
import subprocess
import signal
import json
import sys
from urllib.parse import quote
from flask import Flask, jsonify, redirect, send_from_directory, request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.model_paths import resolve_deck_model_base, resolve_deck_model_path

app = Flask(__name__, static_folder=BASE_DIR)

arena_process = None
train_process = None

def is_running():
    global arena_process, train_process
    if arena_process is None and train_process is None:
        return False
    if arena_process is not None and arena_process.poll() is None:
        return True
    if train_process is not None and train_process.poll() is None:
        return True
    return False

def kill_tourney():
    global arena_process
    if arena_process is not None and arena_process.poll() is None:
        arena_process.terminate()
        try:
            arena_process.wait(timeout=5)
        except:
            arena_process.kill()
    
    arena_process = None
    
    # Also aggressively kill any lingering evaluate.py or arena processes
    subprocess.run("pkill -f 'python src/evaluate_single.py'", shell=True)
    subprocess.run("pkill -f 'python src/auto_arena.py'", shell=True)

@app.route("/")
def index():
    return redirect("/dashboard/dashboard.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(BASE_DIR, path)

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"running": is_running()})

@app.route("/api/start", methods=["POST"])
def start():
    global arena_process, train_process
    if not is_running():
        # Make sure no old processes are running
        kill_tourney()
        
        # Start new processes
        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cmd_arena = "source venv/bin/activate && python src/auto_arena.py"
        # cmd_train = "source venv/bin/activate && python src/auto_train.py"
        
        arena_process = subprocess.Popen(cmd_arena, shell=True, cwd=cwd, executable="/bin/zsh")
        # train_process = subprocess.Popen(cmd_train, shell=True, cwd=cwd, executable="/bin/zsh")
        return jsonify({"success": True, "message": "Arena started (Training disabled)!"})
    return jsonify({"success": False, "message": "Already running!"})

@app.route("/api/pause", methods=["POST"])
def pause():
    kill_tourney()
    return jsonify({"success": True, "message": "Tournament paused!"})

@app.route("/api/reset", methods=["POST"])
def reset():
    kill_tourney()
    
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    wipe_script = """
    # 1. Nur Stats und Replays löschen
    rm -f decks/*.json
    find replays -name "*.json" -type f -delete
    
    # Keine Modelle mehr aus Backup kopieren!
    
    # 2. Initialize generation
    echo '{"generation": 1}' > decks/generation.json
    """
    
    subprocess.run(wipe_script, shell=True, cwd=cwd, executable="/bin/zsh")
    
    # Restart
    global arena_process, train_process
    cmd_arena = "source venv/bin/activate && python src/auto_arena.py"
    # cmd_train = "source venv/bin/activate && python src/auto_train.py"
    arena_process = subprocess.Popen(cmd_arena, shell=True, cwd=cwd, executable="/bin/zsh")
    # train_process = subprocess.Popen(cmd_train, shell=True, cwd=cwd, executable="/bin/zsh")
    
    return jsonify({"success": True, "message": "Factory reset complete. Tournament restarting..."})

@app.route('/api/available_decks', methods=['GET'])
def get_available_decks():
    import glob
    decks = []
    for filepath in glob.glob("decks/deck_*.csv"):
        filename = os.path.basename(filepath)
        decks.append(filename)
    return jsonify({"decks": sorted(decks)})

WATCHED_FILE = os.path.join(BASE_DIR, "decks", "watched_models.json")

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

                stat = os.stat(abs_path)
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

@app.route('/api/train_custom', methods=['POST'])
def train_custom():
    global process
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
        
    opponent_model = resolve_deck_model_path(opp_id) or f"models/ppo_deck_{opp_id}"
    
    global train_process
    
    # Optional: we can kill the train process if it's running so it doesn't conflict
    if train_process is not None and train_process.poll() is None:
        train_process.terminate()
        
    cmd = [
        "venv/bin/python", "src/train_vs.py",
        "--learning-deck", f"decks/{learning_deck}",
        "--learning-model", learning_model,
        "--opponent-deck", f"decks/{opponent_deck}",
        "--opponent-model", opponent_model,
        "--timesteps", str(timesteps),
        "--num-envs", "8",
        "--algo", algo,
        "--ent-start", str(ent_start),
        "--ent-end", str(ent_end),
        "--reward-config", json.dumps(reward_config)
    ]
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
        os.system("pkill -f 'src/train_vs.py'")
            
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
