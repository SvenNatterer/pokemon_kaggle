import http.server
import socketserver
import os
import time

PORT = 8080
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DASHBOARD_FILES = {
    "/dashboard.html": "/dashboard/dashboard.html",
    "/dashboard.css": "/dashboard/dashboard.css",
    "/dashboard.js": "/dashboard/dashboard.js",
}

class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

    def do_GET(self):
        self.path = ROOT_DASHBOARD_FILES.get(self.path, self.path)
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


with ReusableTCPServer(("", PORT), NoCacheHTTPRequestHandler) as httpd:
    print(f"Serving dashboard at http://localhost:{PORT}/dashboard.html")
    print(f"Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Server stopped.")
