#!/bin/bash
# Instructions for cron-job.org

echo "========================================="
echo "📅 DMAI CRON-JOB.ORG SETUP"
echo "========================================="
echo ""
echo "Step 1: Go to https://cron-job.org"
echo "Step 2: Create free account"
echo "Step 3: Click 'Create Cronjob'"
echo ""
echo "Settings:"
echo "  • Title: DMAI Evolution"
echo "  • URL: https://your-render-app.onrender.com/evolve"
echo "  • Schedule: Every 5 minutes"
echo "  • Method: GET"
echo "  • Save responses: Yes"
echo ""
echo "Step 4: Create second cronjob"
echo "  • Title: DMAI Keep-Alive"
echo "  • URL: https://your-render-app.onrender.com/health"
echo "  • Schedule: Every 5 minutes"
echo "  • Method: GET"
echo ""
echo "========================================="
echo "Alternative: Simple curl script if you"
echo "want to self-host the cron job:"
echo "========================================="
echo ""
echo '#!/bin/bash'
echo 'while true; do'
echo '  curl https://your-render-app.onrender.com/evolve'
echo '  sleep 300  # Wait 5 minutes'
echo 'done'
