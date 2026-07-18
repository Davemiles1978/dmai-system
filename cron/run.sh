#!/usr/bin/env bash
# PR DDD-1: run one cron tick and exit with the appropriate code.
#
# Exit codes:
#   0 - HTTP 2xx AND JSON body has ok:true (or ok not present)
#   2 - HTTP non-2xx
#   3 - HTTP 2xx but JSON body has ok:false
#
# Render's cron scheduler treats any non-zero exit as a failed run and
# highlights it in the dashboard. That IS our "failure notification" -
# no separate alerting path required. If we later want a Slack ping on
# failure, extend this script rather than shelling out from render.yaml.
set -eEuo pipefail
source /common.sh

log "Starting ${JOB} against ${TARGET_URL}"

read -r http_code elapsed out_file < <(http_post "${TARGET_URL}" "${POST_BODY}")

body=$(cat "${out_file}")
log "http_code=${http_code} elapsed=${elapsed}s"
log "body_head=$(echo "${body}" | head -c 400)"

if [ "${http_code:0:1}" != "2" ]; then
  log "FAIL: non-2xx HTTP status"
  exit 2
fi

# ok:true / ok:false detection - jq returns "true", "false", or "null"
ok_field=$(echo "${body}" | jq -r '.ok // empty' 2>/dev/null || true)
if [ "${ok_field}" = "false" ]; then
  log "FAIL: ok:false in response body"
  exit 3
fi

log "OK"
exit 0
