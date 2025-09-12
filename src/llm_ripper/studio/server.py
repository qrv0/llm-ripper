from __future__ import annotations

import http.server
import socketserver
from pathlib import Path
import json
import logging
from functools import partial


HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>LLM Ripper Studio (MVP)</title>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    .col { float:left; width:32%; margin-right:2%; }
    pre { background:#f7f7f7; padding:10px; overflow:auto; max-height:60vh; }
    h2 { margin-top:0; }
    .row:after { content: ""; display: table; clear: both; }
  </style>
  <script>
    async function loadJSON(path) {
      try { const r = await fetch(path); return await r.json(); } catch(e){ return { error: String(e) } }
    }
    async function init() {
      const root = new URLSearchParams(window.location.search).get('root') || '.';
      const cat = await loadJSON(root + '/catalog/heads.json');
      const trace = await loadJSON(root + '/traces/summary.json');
      const val = await loadJSON(root + '/validation/validation_results.json');
      document.getElementById('catalog').textContent = JSON.stringify(cat, null, 2);
      document.getElementById('trace').textContent = JSON.stringify(trace, null, 2);
      document.getElementById('validate').textContent = JSON.stringify(val, null, 2);
      document.getElementById('root').textContent = root;
    }
    window.onload = init;
  </script>
  </head>
  <body>
    <h1>LLM Ripper Studio</h1>
    <p>Root: <span id="root"></span></p>
    <div class="row">
      <div class="col">
        <h2>Catalog</h2>
        <pre id="catalog"></pre>
      </div>
      <div class="col">
        <h2>Trace</h2>
        <pre id="trace"></pre>
      </div>
      <div class="col">
        <h2>Transplant + Validate</h2>
        <pre id="validate"></pre>
      </div>
    </div>
  </body>
  </html>
"""


def launch_studio(doc_root: str, port: int = 8000) -> None:
    """Serve a static Studio MVP rooted at doc_root/studio and block (Ctrl+C to stop)."""
    root = Path(doc_root)
    studio_dir = root / "studio"
    studio_dir.mkdir(parents=True, exist_ok=True)
    index = studio_dir / "index.html"
    if not index.exists():
        index.write_text(HTML)
    # Serve from root directory without changing global CWD
    handler_cls = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with socketserver.TCPServer(("", port), handler_cls) as httpd:
        url = f"http://localhost:{port}/studio/index.html?root={root}"
        logging.getLogger(__name__).info(
            "Studio running at: %s\nPress Ctrl+C to stop.", url
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
