#!/bin/bash
BASE="https://dmai-web.onrender.com"
# Endpoints called by static HTML, plus a few key system ones
GET_ENDPOINTS=(
  "/health" "/api/status" "/api/persona" "/api/conversations"
  "/api/learning/progress" "/api/learning/status" "/api/learning/full-status" "/api/learning/unified-status"
  "/api/training/status" "/api/training/full" "/api/dashboard"
  "/api/orchestrator/status" "/api/extended_hub/status"
  "/api/kaizen" "/api/kaizen/status" "/api/kaizen/cycle-status" "/api/kaizen/repair-stats"
  "/api/heartbeat" "/api/metrics" "/api/stage/analytics"
  "/api/graph/status" "/api/graph/schema"
  "/api/social/status" "/api/funding/status" "/api/funding/revenue-streams"
  "/api/financial/uk/status" "/api/trading/status" "/api/trading/mastery"
  "/api/research/status" "/api/research/history" "/api/research/autonomous/status"
  "/api/harvester/status" "/api/harvester/providers"
  "/api/knowledge/status" "/api/consciousness/state"
  "/api/ai/discovery/status" "/api/tutors/list" "/api/evolution" "/api/evolution/consciousness" "/api/evolution/metrics"
  "/api/optimizer/status" "/api/capabilities/list" "/api/capability-map"
  "/api/sandbox/health" "/api/master/status" "/api/admin/circuit-breakers"
  "/api/integrity/report" "/api/suggestions" "/api/system/health"
  "/api/registrar/status" "/api/registrar/pending"
  "/api/vocabulary/stats" "/api/vocabulary/sample"
  "/api/music/status" "/api/art/gallery" "/api/content/list"
  "/api/integration/free-apis" "/api/integration/repos"
  "/api/code-writer/history" "/api/github/stars" "/api/ingestor/status"
  "/api/settings" "/api/chat/debug" "/api/admin/keys"
)
fail=0
pass=0
for e in "${GET_ENDPOINTS[@]}"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$BASE$e")
  if [ "$code" = "200" ] || [ "$code" = "401" ] || [ "$code" = "503" ]; then
    pass=$((pass+1))
    printf "  OK %3s  %s\n" "$code" "$e"
  else
    fail=$((fail+1))
    printf " FAIL %3s  %s\n" "$code" "$e"
  fi
done
echo "---"
echo "PASS: $pass  FAIL: $fail"
