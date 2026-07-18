# PR DDD-1: R2 backup cron container.
#
# Runs once per Render schedule tick, POSTs the CRON_SECRET-protected
# backup endpoint on dmai-web, exits 0 on 2xx / ok:true and non-zero
# on anything else so Render marks the run failed and surfaces it in
# the dashboard. See render.yaml for the schedule.
FROM alpine:3.20
RUN apk add --no-cache curl jq bash ca-certificates && update-ca-certificates
COPY run.sh /run.sh
COPY common.sh /common.sh
RUN chmod +x /run.sh /common.sh
ENV JOB=backup
CMD ["/run.sh"]
