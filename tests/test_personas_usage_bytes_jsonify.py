"""/api/personas/usage must serialise even when a persona key is bytes (PR #164).

SQLite can hand back the ``persona`` column as ``bytes`` (BLOB affinity), which
made ``jsonify`` reject the ``by_persona`` dict with
``keys must be str, int, float, bool or None, not bytes``. usage_stats now
coerces its payload through the shared ``_jsonable`` helper (PR #152 family).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.personas.persona_registry import PersonaRegistry


def _registry(tmp_path):
    return PersonaRegistry(data_path=tmp_path)


def test_usage_stats_coerces_bytes_persona_key(tmp_path):
    reg = _registry(tmp_path)
    with reg._conn() as c:
        # Insert a bytes persona value -> SQLite keeps it as a BLOB, so the row
        # comes back as bytes, reproducing the production failure.
        c.execute("INSERT INTO persona_usage (persona) VALUES (?)", (b"trader",))
        c.execute("INSERT INTO persona_usage (persona) VALUES (?)", ("analyst",))
        c.commit()

    stats = reg.usage_stats(days=7)

    # Every key must be a str so jsonify accepts it.
    assert all(isinstance(k, str) for k in stats["by_persona"])
    assert "trader" in stats["by_persona"]
    assert stats["total"] == 2


def test_usage_stats_is_jsonify_serialisable(tmp_path):
    from flask import Flask

    reg = _registry(tmp_path)
    with reg._conn() as c:
        c.execute("INSERT INTO persona_usage (persona) VALUES (?)", (b"trader",))
        c.commit()

    app = Flask(__name__)
    with app.app_context():
        from flask import jsonify
        resp = jsonify(reg.usage_stats(days=7))  # must not raise
        assert resp.status_code == 200
        assert b"trader" in resp.data
