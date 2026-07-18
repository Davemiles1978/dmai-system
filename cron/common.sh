#!/usr/bin/env bash
# Shared helpers for DMAI cron containers.
set -eEuo pipefail

: "${CRON_SECRET:?CRON_SECRET env var is required}"
: "${TARGET_URL:?TARGET_URL env var is required}"

# POST_BODY defaults to empty JSON object; curriculum overrides via env.
POST_BODY="${POST_BODY:-{}}"

JOB="${JOB:-generic}"

log() {
  # ISO-8601 UTC timestamp for consistent parsing in Render's log viewer.
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[cron:${JOB} ${ts}] $*"
}

http_post() {
  # Portable POST wrapper - separates the response body from the HTTP
  # code and elapsed time so run.sh can decide success/failure without
  # re-parsing curl output. Timeout 300s matches the historical
  # Perplexity crons.
  local url="$1"
  local body="$2"
  local out_file="/tmp/cron_response.json"
  local http_time
  http_time=$(curl -sS -X POST "$url" \
    -H "X-Cron-Secret: ${CRON_SECRET}" \
    -H "Content-Type: application/json" \
    -d "$body" \
    -m 300 \
    -o "$out_file" \
    -w "%{http_code} %{time_total}")
  echo "$http_time $out_file"
}
