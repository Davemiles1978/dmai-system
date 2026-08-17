# Post-Deploy Load Test & Auto-Rollback

Defined in `.github/workflows/post-deploy-load-test.yml`.

## What it does

On every push to `main` (and on manual `workflow_dispatch`):

1. **wait_for_deploy** — polls the Render API until the deploy for the
   incoming commit reaches `live`, then waits up to 3 more minutes for
   `/health` to return 200. Captures the previous live deploy's ID as the
   rollback target.
2. **load_test** — runs `ops/load_test_worker_stability.py` against
   `https://dmai-web.onrender.com` (50 concurrent, 5 min by default). Uploads
   the full run artifacts (`requests.jsonl`, `monitor.jsonl`, `summary.json`,
   `report.md`).
3. **gate** — fails the deploy gate if any of:
   - load-test verdict is not `PASS`
   - `min active_count < 8` at any monitor poll
   - any request returned HTTP 5xx or transport error
4. **rollback** — if the gate fails (and `skip_rollback` is not set), calls
   `POST /v1/services/{id}/rollback` with the previous deploy ID, then polls
   until that rollback reaches `live`.
5. **notify** — posts a single Slack message summarising verdict, key metrics,
   commit, rollback outcome, and the first 2.5 KB of `report.md`.
6. **final_status** — fails the workflow run if a rollback was triggered, so
   you see a red ✗ in the Actions tab.

## Required repository secrets

Configure under **Settings → Secrets and variables → Actions → New repository
secret**:

| Secret | Value | Where to get it |
|---|---|---|
| `RENDER_API_KEY` | `rnd_…` | [Render dashboard → Account Settings → API Keys](https://dashboard.render.com/u/settings/api-keys). Needs deploy + rollback scope. |
| `RENDER_SERVICE_ID` | `srv-d6sd3chj16oc73emdj6g` | Render dashboard → dmai-web → Settings → Service ID. |
| `DMAI_MASTER_PASSWORD` | `Talula.78` (or current value) | `docs/HANDOVER.md §3`. Used as the `X-Master-Password` header. |
| `SLACK_BOT_TOKEN` | `xoxb-…` | [api.slack.com/apps](https://api.slack.com/apps) → your app → OAuth & Permissions → Bot User OAuth Token. Requires `chat:write` scope. The bot must be invited to the target channel: `/invite @your-bot` in Slack. |
| `SLACK_CHANNEL_ID` | `C0BCKABKVDG` (the existing alerts channel) | Right-click channel → View channel details → ID at the bottom. |

## Manual trigger

From the Actions tab → "Post-Deploy Load Test & Auto-Rollback" → Run workflow:

| Input | Default | Purpose |
|---|---|---|
| `skip_rollback` | `false` | Override to `true` to run the load test without auto-rollback (useful for tuning). |
| `duration` | `300` | Test duration in seconds. |
| `concurrency` | `50` | Concurrent connection count. |

## Failure modes the workflow handles

| Situation | Behaviour |
|---|---|
| Render deploy never goes `live` within 12 min | `wait_for_deploy` fails; load test and rollback are skipped. |
| Render deploy goes `live` but `/health` never returns 200 | Same — fail in `wait_for_deploy`. No load applied to a broken instance. |
| Load test runs but verdict is FAIL | Gate flags rollback needed → rollback job runs → Slack alert sent → workflow ends red. |
| Load test PASSES | No rollback, green Slack message, workflow ends green. |
| No previous live deploy exists (first deploy ever) | Rollback step exits with `no_previous_deploy`; Slack alert still goes out. |
| Slack token missing or wrong | Final notify step fails; workflow goes red regardless of test verdict. |

## Cost note

A single run consumes:
- ~15,000 HTTP requests against `dmai-web` (lightweight `/health` only)
- ~60 monitor polls (`/api/training/status` + `/api/harvester/status`)
- ~10 min of GitHub Actions runner time

If you push frequently to `main`, consider gating this workflow with a
`paths:` filter on `post.push` so doc-only or test-only changes skip it.

## Roll-out checklist

1. Merge PR #142 (the render.yaml fix) and confirm the load test passes
   manually first: `python3 ops/load_test_worker_stability.py`.
2. Configure the five secrets above.
3. Merge this workflow PR.
4. Trigger manually once via workflow_dispatch to validate the full
   pipeline end-to-end before letting it run automatically on push.
5. Watch the Slack channel for the first auto-triggered run.
