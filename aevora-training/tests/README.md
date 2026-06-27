# DMAI Metrics Pipeline Tests

## test_metrics_pipeline.py

End-to-end diagnostic that verifies the full metrics pipeline is healthy after any redeploy.

### What it checks

| Check | Endpoint | Assertion |
|---|---|---|
| System Health | `/api/status` | HTTP 200, status = ok/healthy |
| DB Seeder | `/api/metrics` | capabilities > 0, insights > 0 |
| KPI Cache | `/api/learning/full-status` | all 8 KPIs present, ≥1 non-zero |
| Orchestrator Status | `/api/training/status` | services dict present, domain count > 0 |
| Learning Progress | `/api/learning/progress` | current_stage set, no Error state |
| Graph Schema | `/api/graph/schema` | total_neurons > 0 |
| Cross-Endpoint Consistency | metrics vs full-status | capabilities within 5%, KPIs consistent |

### Usage

```bash
# Against live Render deployment
python3 aevora-training/tests/test_metrics_pipeline.py

# Against a specific URL
python3 aevora-training/tests/test_metrics_pipeline.py --base-url https://dmai-web.onrender.com

# Local dev server
python3 aevora-training/tests/test_metrics_pipeline.py --local

# Machine-readable JSON output
python3 aevora-training/tests/test_metrics_pipeline.py --json
```

Exit code `0` = all passed. Exit code `1` = one or more failures.

### Running in CI

Add to your Render post-deploy hook or GitHub Actions:

```yaml
- name: Run metrics diagnostic
  run: python3 aevora-training/tests/test_metrics_pipeline.py
  env:
    BASE_URL: https://dmai-web.onrender.com
```
