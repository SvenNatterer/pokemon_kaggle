import os
import subprocess
import signal
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="..")

tourney_process = None

def is_running():
    global tourney_process
    if tourney_process is None:
        return False
    if tourney_process.poll() is None:
        return True
    return False

def kill_tourney():
    global tourney_process
    if is_running():
        # Kill the main process
        tourney_process.terminate()
        try:
            tourney_process.wait(timeout=5)
        except:
            tourney_process.kill()
        tourney_process = None
    
    # Also aggressively kill any lingering evaluate.py or auto_tourney processes
    subprocess.run("pkill -f 'python src/evaluate_single.py'", shell=True)
    subprocess.run("pkill -f 'python src/auto_tourney.py'", shell=True)

@app.route("/")
def index():
    return send_from_directory("..", "dashboard.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("..", path)

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"running": is_running()})

@app.route("/api/start", methods=["POST"])
def start():
    global tourney_process
    if not is_running():
        # Make sure no old processes are running
        kill_tourney()
        
        # Start new process
        cmd = "source venv/bin/activate && python src/auto_tourney.py"
        tourney_process = subprocess.Popen(cmd, shell=True, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), executable="/bin/zsh")
        return jsonify({"success": True, "message": "Tournament started!"})
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
    mv decks/ghost_pool/deck_2.csv decks/ 2>/dev/null || true
    rm -f decks/ghost_pool/deck_*.csv
    # Delete all decks except 1 through 5
    find decks -name "deck_*.csv" ! -name "deck_1.csv" ! -name "deck_2.csv" ! -name "deck_3.csv" ! -name "deck_4.csv" ! -name "deck_5.csv" -type f -delete
    find models -name "*.zip" ! -name "ppo_base_brain.zip" -type f -delete
    rm -f decks/pairwise_winrates.json decks/games_played.json decks/current_generation_winrates.json
    rm -f replays/*.json
    echo '{"generation": 1}' > decks/generation.json
    rm -f decks/status.json
    """
    
    subprocess.run(wipe_script, shell=True, cwd=cwd, executable="/bin/zsh")
    
    # Restart
    global tourney_process
    cmd = "source venv/bin/activate && python src/auto_tourney.py"
    tourney_process = subprocess.Popen(cmd, shell=True, cwd=cwd, executable="/bin/zsh")
    
    return jsonify({"success": True, "message": "Factory reset complete. Tournament restarting..."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
