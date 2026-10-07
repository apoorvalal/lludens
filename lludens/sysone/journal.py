"""Append-only choice journals with exact-request, network-free replay."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import gzip
import json
from pathlib import Path
import threading

from .choices import evaluate_choice


def read_jsonl(path):
    """Read JSONL, or its ``.jsonl.gz`` archive when the plain file is absent."""
    path = Path(path)
    if not path.exists():
        archived = path.with_suffix(path.suffix + ".gz")
        if not archived.exists():
            return []
        with gzip.open(archived, "rt") as stream:
            return [json.loads(s) for s in stream if s.strip()]
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


class DecisionJournal:
    """Cache successful calls by context, player, and exact ordered request.

    ``offline=True`` fails on a cache miss and never calls the transport.
    A successful first move is reusable after an interrupted second move.
    ``call`` can inject a test transport without either provider installed.
    """

    def __init__(self, path, offline=False, *, call=None):
        path = Path(path)
        self.path = path
        self.offline = offline
        self.call = call or evaluate_choice
        archived = path.with_suffix(path.suffix + ".gz")
        if not offline and not path.exists() and archived.exists():
            # Restore the full append-only journal before a live resume.
            path.write_bytes(gzip.decompress(archived.read_bytes()))
        if not offline:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        records = read_jsonl(path)
        self.saved = {x["key"]: x for x in records}
        if len(self.saved) != len(records):
            raise ValueError("Duplicate saved call IDs")

    def caller(self, match_id, player):
        def call(provider, prompt, options, instructions):
            payload = {"provider": provider, "prompt": prompt, "options": options, "instructions": instructions}
            digest = sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
            key = f"{match_id}/player{player}/{digest}"
            with self.lock:
                saved = self.saved.get(key)
            if saved is not None:
                return saved["result"]
            if self.offline:
                raise ValueError("Missing exact saved request; offline replay will not call the API")
            result = self.call(provider, prompt, options, instructions)
            record = {"key": key, "match_id": match_id, "player": player,
                      "utc": datetime.now(timezone.utc).isoformat(), **payload, "result": result}
            with self.lock:
                with self.path.open("a") as stream:
                    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                self.saved[key] = record
            return result
        return call
