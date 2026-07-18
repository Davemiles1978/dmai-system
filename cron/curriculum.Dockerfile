# PR DDD-1: coding-curriculum study cron container.
# Same shape as backup.Dockerfile; only the target endpoint differs
# (set via TARGET_URL env var, see render.yaml).
FROM alpine:3.20
RUN apk add --no-cache curl jq bash ca-certificates && update-ca-certificates
COPY run.sh /run.sh
COPY common.sh /common.sh
RUN chmod +x /run.sh /common.sh
ENV JOB=curriculum
ENV POST_BODY='{"n": 5}'
CMD ["/run.sh"]
