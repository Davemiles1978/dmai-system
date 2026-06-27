#!/bin/bash
# Full UI audit: probe every endpoint referenced by any HTML file.
BASE="https://dmai-web.onrender.com"

# Collect every /api/* endpoint referenced anywhere in static/*.html
endpoints=$(grep -hoE "/api/[a-zA-Z0-9/_-]+" /tmp/dmai-fix/static/*.html | sort -u)
# Plus the page routes themselves
pages="/ /chat /dashboard /admin /mobile /trading /health"

PASS=0; FAIL_404=0; FAIL_500=0; FAIL_OTHER=0
declare -a FAIL_LIST

probe() {
  local path="$1" method="$2"
  if [ "$method" = "POST" ]; then
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST "$BASE$path" -H 'Content-Type: application/json' -d '{}')
  else
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$BASE$path")
  fi
  case "$code" in
    200|201|204|304|401|403) PASS=$((PASS+1)); printf "  OK %3s  %-6s %s\n" "$code" "$method" "$path" ;;
    404) FAIL_404=$((FAIL_404+1)); printf " 404 %3s  %-6s %s\n" "$code" "$method" "$path"; FAIL_LIST+=("$code $method $path") ;;
    405) # Try opposite method
        if [ "$method" = "GET" ]; then probe "$path" "POST"; else PASS=$((PASS+1)); printf "  OK %3s  %-6s %s (POST-only)\n" "$code" "$method" "$path"; fi
        return
        ;;
    500|502|503) FAIL_500=$((FAIL_500+1)); printf " 5xx %3s  %-6s %s\n" "$code" "$method" "$path"; FAIL_LIST+=("$code $method $path") ;;
    *)  FAIL_OTHER=$((FAIL_OTHER+1)); printf " ??? %3s  %-6s %s\n" "$code" "$method" "$path"; FAIL_LIST+=("$code $method $path") ;;
  esac
}

echo "=== PAGES ==="
for p in $pages; do probe "$p" GET; done
echo ""
echo "=== API ENDPOINTS (GET first) ==="
for e in $endpoints; do
  # skip parametric paths and chat post route (already tested)
  case "$e" in
    */data/*|*/static/*) continue ;;
  esac
  probe "$e" GET
done

echo ""
echo "==========================================="
echo " PASS: $PASS   404: $FAIL_404   5xx: $FAIL_500   Other: $FAIL_OTHER"
echo "==========================================="
if [ ${#FAIL_LIST[@]} -gt 0 ]; then
  echo "Failures:"
  for f in "${FAIL_LIST[@]}"; do echo "  $f"; done
fi
