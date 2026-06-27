"""
sandbox_client.py — thin HTTP client used by DMAI core (dmai_core_complete.py)
to call the isolated ``dmai-sandbox`` container.

Design goal: graceful degradation. If the sandbox container is offline or
unreachable, the client returns a structured ``SandboxResult`` with
``status="unavailable"`` rather than raising — callers can branch on that.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

DEFAULT_SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://dmai-sandbox:8765")
_CONNECT_TIMEOUT = 5


@dataclass
class SandboxResult:
    request_id: str
    status: str  # ok | timeout | error | blocked | unavailable
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: int
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    sanitised: bool = True

    @property
    def is_safe(self) -> bool:
        return self.status == "ok" and not any(
            a.get("severity") in ("HIGH", "CRITICAL") for a in self.anomalies
        )

    @property
    def has_critical_anomaly(self) -> bool:
        return any(a.get("severity") == "CRITICAL" for a in self.anomalies)

    @property
    def anomaly_summary(self) -> str:
        if not self.anomalies:
            return f"status={self.status}, no anomalies"
        counts: Dict[str, int] = {}
        for a in self.anomalies:
            sev = a.get("severity", "NONE")
            counts[sev] = counts.get(sev, 0) + 1
        parts = ", ".join(f"{sev}:{n}" for sev, n in sorted(counts.items()))
        return f"status={self.status}, {len(self.anomalies)} anomalies ({parts})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "execution_time_ms": self.execution_time_ms,
            "anomalies": self.anomalies,
            "sanitised": self.sanitised,
            "is_safe": self.is_safe,
            "has_critical_anomaly": self.has_critical_anomaly,
            "anomaly_summary": self.anomaly_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SandboxResult":
        return cls(
            request_id=data.get("request_id", ""),
            status=data.get("status", "error"),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            exit_code=int(data.get("exit_code", -1)),
            execution_time_ms=int(data.get("execution_time_ms", 0)),
            anomalies=data.get("anomalies", []) or [],
            sanitised=bool(data.get("sanitised", True)),
        )


class SandboxClient:
    def __init__(self, sandbox_url: Optional[str] = None) -> None:
        self.sandbox_url = (sandbox_url or DEFAULT_SANDBOX_URL).rstrip("/")

    def execute(self, code: str, language: str = "python", timeout: int = 10) -> SandboxResult:
        request_id = str(uuid.uuid4())
        payload = {
            "code": code,
            "language": language,
            "timeout": timeout,
            "request_id": request_id,
        }
        try:
            resp = requests.post(
                f"{self.sandbox_url}/execute",
                json=payload,
                timeout=(_CONNECT_TIMEOUT, timeout + 10),
            )
            resp.raise_for_status()
            return SandboxResult.from_dict(resp.json())
        except requests.exceptions.ConnectionError:
            return self._unavailable(request_id)
        except Exception as exc:  # noqa: BLE001 — any other failure → structured error
            return SandboxResult(
                request_id=request_id,
                status="error",
                stdout="",
                stderr=f"Sandbox client error: {exc}",
                exit_code=-1,
                execution_time_ms=0,
                anomalies=[],
                sanitised=True,
            )

    def health(self) -> Dict[str, Any]:
        try:
            resp = requests.get(
                f"{self.sandbox_url}/health",
                timeout=(_CONNECT_TIMEOUT, _CONNECT_TIMEOUT),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            return {"status": "unavailable", "error": str(exc)}

    def is_available(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _unavailable(request_id: str) -> SandboxResult:
        return SandboxResult(
            request_id=request_id,
            status="unavailable",
            stdout="",
            stderr="Sandbox offline — start with "
                   "docker-compose -f docker-compose.sandbox.yml up -d",
            exit_code=-1,
            execution_time_ms=0,
            anomalies=[],
            sanitised=True,
        )
