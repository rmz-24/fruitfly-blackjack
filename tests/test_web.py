import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from flyjack.anatomy import PATH_ANNOT
from flyjack.record import RESPONSES

pytestmark = pytest.mark.skipif(not (PATH_ANNOT.exists() and RESPONSES.exists()),
                                reason="annotations / recorded responses not available")


@pytest.fixture(scope="module")
def server():
    from flyjack.web.lab import FlyLab
    from flyjack.web.server import make_handler
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FlyLab()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.status, r.read()


def test_page_and_meta(server):
    status, html = call(server, "/")
    assert status == 200 and b"Fly Lab" in html
    meta = json.loads(call(server, "/api/meta")[1])
    assert len(meta["observations"]) == 280 and len(meta["kc"]["preference"]) == meta["kc"]["n"]


def test_readout_decision_and_custom_hand(server):
    r = json.loads(call(server, "/api/sense", {"obs": [20, 10, False]})[1])
    assert r["action"] == 0 and r["basic"] == 0 and r["kc_active"]
    h = json.loads(call(server, "/api/hand/new", {"player": [10, 6], "dealer": [9, 8]})[1])
    assert h["player"] == [10, 6] and h["dealer"] == [9] and h["obs"] == [16, 9, False]
    done = json.loads(call(server, "/api/hand/act", {"id": h["id"], "action": 0})[1])
    assert done["done"] and done["dealer"] == [9, 8] and done["reward"] == -1.0


def test_tournament_and_path_traversal(server):
    t = json.loads(call(server, "/api/tournament", {"hands": 300, "seed": 1})[1])
    assert set(t["players"]) >= {"fly", "basic strategy", "random"}
    with pytest.raises(urllib.error.HTTPError):
        call(server, "/../README.md")
