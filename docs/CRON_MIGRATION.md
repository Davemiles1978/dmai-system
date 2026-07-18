# DMAI cron migration off Perplexity (PRs DDD-1, DDD-2, DDD-3)

Started 2026-07-18. Motivation: Perplexity scheduled crons burn user
credit on every tick and auto-pause when credit runs out. DMAI's five
scheduled tasks are simple `curl` POSTs that don't need an LLM in the
loop, so they've been moved to zero-cost hosts that Perplexity's credit
balance doesn't gate.

## Where each cron lives now

| Cron                          | Old host (Perplexity ID) | New host          | Endpoint                                    | Schedule (UTC) |
|-------------------------------|--------------------------|-------------------|---------------------------------------------|----------------|
| R2 backup                     | 1c65ac52 + 24356c1c      | Render Cron       | POST /api/cron/backup/run                   | 00:45 daily    |
| Coding curriculum study       | 3554d97b + baa2e9ab      | Render Cron       | POST /api/cron/coding-curriculum/study      | 02:15 daily    |
| Weekly promoter drift report  | ebd617e8                 | GitHub Actions    | POST /api/cron/promoter-drift/email         | 06:00 Mondays  |

## Cutover plan

1. **2026-07-18 (today)**: Ship DDD-1..3. Render cron services + GitHub
   Actions workflow start firing alongside the existing Perplexity crons.
   Both schedulers hit the same endpoints; endpoints are idempotent
   (backup rotation dedupes; curriculum `n:5` is bounded), so double
   fires are safe.
2. **Days 1-7 (until 2026-07-25)**: Watch daily runs.
   * Render dashboard -> Services -> `dmai-r2-backup` and
     `dmai-coding-curriculum` should show green rows every day.
   * GitHub Actions tab of `Davemiles1978/dmai-system` should show a
     green `promoter-drift` run on Monday 2026-07-20 06:00 UTC.
   * Any email or Slack `#dmaitalk` alert that fires from the new
     stack proves the delivery chain works.
3. **2026-07-25**: If all daily runs above have been green for a full
   week, delete the five Perplexity crons (1c65ac52, 24356c1c,
   3554d97b, baa2e9ab, ebd617e8) via the Perplexity scheduled-task UI.
   Zero further Perplexity credit spent on scheduled work.

## Required env vars

### Render dashboard (dmai-web service AND both new cron services)

* `CRON_SECRET` - already set. Applies to the new cron services too.
* `RESEND_API_KEY` - **new**. Grab a free key from
  https://resend.com/api-keys and paste it. Free tier is 3k emails/month
  which is 400x our real volume.
* `RESEND_FROM` - optional. Defaults to `onboarding@resend.dev`.
* `SLACK_WEBHOOK_URL` - already set. Used as the fallback delivery
  path in `/api/cron/promoter-drift/email` when Resend fails.

### GitHub repo secrets (Davemiles1978/dmai-system -> Settings -> Secrets)

* `DMAI_CRON_SECRET` - copy the same value as `CRON_SECRET` above.

## Failure alerting

* **Render cron failed**: run exits non-zero (2 = HTTP non-2xx,
  3 = ok:false in body). The Render dashboard shows the run in red,
  and Render sends the workspace-configured email/slack alert (if any).
* **GitHub Action failed**: run appears red in the Actions tab AND
  GitHub emails the repo owner by default.
* **Email delivery failed**: `/api/cron/promoter-drift/email` returns
  200 with `delivered_via:"none"` after trying Resend then Slack. The
  GitHub Action treats this as an alert-needed-but-not-delivered case
  and exits code 3, so the run turns red in the Actions tab.

## Rollback

If any new cron misbehaves during the cutover week, the Perplexity
crons are still active - simply pause the Render service (Manual
Suspend in dashboard) or disable the GitHub workflow (Actions ->
`promoter-drift` -> `...` -> Disable workflow). No code change needed.
