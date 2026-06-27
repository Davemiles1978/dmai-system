# DMAI Execution Sandbox

A separate, hardened Docker container (`dmai-sandbox`) that safely executes
untrusted code extracted from research sources (dark-web monitoring, web
crawling, etc.) without risking the production DMAI deployment.

## Architecture

```
 ┌─────────────────────┐
 │   DMAI Core         │  dmai_core_complete.py
 │  (dmai-complete)    │  POST /api/sandbox/execute  (JWT)
 └──────────┬──────────┘  GET  /api/sandbox/health
            │
            │ SandboxClient.execute(code, language, timeout)
            │ HTTP (requests)
            ▼
 ┌─────────────────────────────────── Docker network ───────────────────────────┐
 │                                                                               │
 │   http://dmai-sandbox:8765                                                    │
 │            │                                                                  │
 │            ▼                                                                  │
 │   ┌──────────────────────┐                                                   │
 │   │  dmai-sandbox        │  network_mode: none · read_only · cap_drop ALL    │
 │   │  gunicorn            │  no-new-privileges · pids_limit 64 · mem 256m     │
 │   │  sandbox_api:app     │                                                   │
 │   └──────────┬───────────┘                                                   │
 │              │ POST /execute                                                 │
 │              ▼                                                               │
 │      ┌───────────────┐    seccomp_profile.json (kernel-level allowlist)     │
 │      │  seccomp       │ ── blocks socket/connect/ptrace/mount/...            │
 │      └──────┬─────────┘                                                      │
 │             ▼                                                                │
 │      ┌────────────────────────┐                                             │
 │      │  subprocess.run        │  python3 -I -S · node --disallow-code-gen   │
 │      │  (isolated)            │  bash --restricted · timeout 1-30s          │
 │      └──────┬─────────────────┘                                             │
 │             ▼                                                                │
 │        UNTRUSTED CODE  →  stdout/stderr → AnomalyDetector → SandboxLogger    │
 │                                                                               │
 └───────────────────────────────────────────────────────────────────────────┘
```

## Security model

| Control | Mechanism | Why |
|---|---|---|
| No network | `network_mode: none` + seccomp blocks `socket`/`connect`/... | Untrusted code cannot exfiltrate data or call home |
| Read-only FS | `read_only: true` root filesystem | Code cannot persist anything outside `/tmp` |
| No-exec scratch | `tmpfs /tmp size=64m,noexec` | Prevents dropping & executing payloads |
| No privilege escalation | `no-new-privileges:true`, `cap_drop: ALL` | setuid binaries cannot raise privileges |
| Non-root | runs as `sandbox` / `nobody` (uid 65534) | No root inside the container |
| Fork-bomb guard | `pids_limit: 64` | Caps total processes |
| Memory guard | `mem_limit: 256m` | Prevents memory exhaustion |
| CPU guard | `cpus: "0.5"` | Caps CPU usage |
| Syscall allowlist | `seccomp_profile.json` (`SCMP_ACT_ERRNO` default) | Only safe syscalls permitted at kernel level |
| Python isolation | `python3 -I -S` | Ignores `PYTHONPATH`, site-packages, user site |
| JS isolation | `node --no-experimental-fetch --disallow-code-generation-from-strings` | No network fetch, no `eval`/`Function` |
| Bash isolation | `bash --restricted` | No `cd`, no `PATH` change, no path-qualified commands |
| Input cap | code truncated at 32KB | Limits payload size |
| Output cap | stdout/stderr truncated at 64KB | Limits exfiltration channel & memory |

## Quick start

```bash
docker-compose -f docker-compose.sandbox.yml up -d
```

The container exposes **no published ports** — it is reachable only over the
internal Docker network at `http://dmai-sandbox:8765`.

## How DMAI core calls it

```python
from components.sandbox.sandbox_client import SandboxClient

sandbox = SandboxClient()  # defaults to http://dmai-sandbox:8765

if sandbox.is_available():
    result = sandbox.execute("print('hello from sandbox')", language="python", timeout=10)
    print(result.status)          # "ok"
    print(result.stdout)          # "hello from sandbox\n"
    print(result.is_safe)         # True
    print(result.anomaly_summary) # "status=ok, no anomalies"

    if result.has_critical_anomaly:
        # e.g. a leaked credential or shell-injection pattern in the code
        ...
```

Exposed by DMAI core:

* `POST /api/sandbox/execute` — JWT required. Body: `{code, language, timeout}`.
* `GET  /api/sandbox/health` — no auth. Returns sandbox health + recent log entries.

## Anomaly categories

| Category | Severity | Description |
|---|---|---|
| `CREDENTIAL_LEAK` | CRITICAL | AWS/GCP keys, private-key PEM headers, JWTs, hardcoded passwords |
| `NETWORK_ATTEMPT` | HIGH | IPv4 addresses, URLs, curl/wget/requests/urllib usage, network imports |
| `SHELL_INJECTION` | HIGH (output) / CRITICAL (code) | `eval(`, `exec(`, `__import__`, `os.system`, `subprocess`, `importlib`, `compile(` |
| `FILE_TRAVERSAL` | HIGH | `../../`, `/etc/passwd`, `/proc/<pid>`, `/sys/...`, `~root` |
| `ENCODING_BYPASS` | MEDIUM | Large base64 blobs, long hex escape strings, long unicode escapes |
| `LARGE_OUTPUT` | MEDIUM | stdout larger than 10KB (possible data exfiltration) |

Any **CRITICAL** anomaly causes the response `status` to become `blocked`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SANDBOX_URL` | `http://dmai-sandbox:8765` | URL the `SandboxClient` connects to |
| `SANDBOX_LOG_PATH` | `/tmp/sandbox_activity.jsonl` | Path of the JSONL audit log |

## Log format

`SandboxLogger` writes one JSON object per line to
`/tmp/sandbox_activity.jsonl` (rotated to `.jsonl.1` once it passes 10MB):

```json
{
  "ts": "2026-06-22T12:00:00.000000+00:00",
  "request_id": "f1e2...",
  "language": "python",
  "code_sha256": "9f86d0...",
  "code_length": 512,
  "execution_time_ms": 123,
  "exit_code": 0,
  "status": "ok",
  "stdout_length": 45,
  "stderr_length": 0,
  "anomaly_count": 0,
  "highest_severity": "NONE",
  "anomalies": []
}
```

## Inspecting logs

```bash
docker exec dmai-sandbox cat /tmp/sandbox_activity.jsonl
```
