"""SICore.add_insight must accept the ``insight_text=`` keyword form (PR #164).

stage_learner and the other learning orchestrators call
``si_core.add_insight(insight_text=..., entity_type=..., source_topic=...)``,
but the signature only took ``(domain, concept, source, confidence, metadata)``
— so every call raised
``SICore.add_insight() got an unexpected keyword argument 'insight_text'``.
The signature now normalises both conventions.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.si_core import SICore


def _bare_sicore():
    # Bypass the heavy __init__; add_insight only touches the filesystem and an
    # optional insights_store attribute.
    return SICore.__new__(SICore)


def test_add_insight_accepts_insight_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # add_insight writes data/research/insights.jsonl
    si = _bare_sicore()
    rec = si.add_insight(
        insight_text="Backprop is gradient descent through the chain rule",
        entity_type="concept",
        entities=["backprop", "gradient descent"],
        relationship="detailed_knowledge",
        source_topic="ml",
        target_topic="backprop",
        confidence=0.9,
        source_title="Baby stage",
        source_url="syllabus_Baby",
    )
    assert rec["concept"] == "Backprop is gradient descent through the chain rule"
    # extra keywords are folded into metadata, not dropped
    assert rec["metadata"]["entity_type"] == "concept"
    assert rec["metadata"]["source_topic"] == "ml"
    assert rec["confidence"] == 0.9
    # persisted to jsonl
    assert (tmp_path / "data" / "research" / "insights.jsonl").exists()


def test_add_insight_legacy_positional_still_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    si = _bare_sicore()
    rec = si.add_insight("ml", "some concept", source="internal")
    assert rec["domain"] == "ml"
    assert rec["concept"] == "some concept"
    assert rec["source"] == "internal"
