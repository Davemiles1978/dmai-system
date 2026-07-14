# Adding an API key

There are three ways to add or update a provider API key. All of them persist
to the DB, inject the value into the running process, and trigger an
AutoAPIActivator rescan so the provider flips to `active` without a restart.

## Method 1 (recommended): DMAI dashboard

1. Open the DMAI dashboard → **API Keys**.
2. Paste the key into the provider's field and click **Save**.
3. Watch the toast:
   - **Green** — saved and ACTIVATED (the provider validated the key).
   - **Red** — saved but the key was **rejected** by the provider; the inline
     panel shows the provider's error. Fix the key and save again.
   - **Amber** — saved but not activated (missing env/config); check the
     provider setup.

If the dashboard shows a *"Render env not synced (RENDER_API_KEY not set)"*
footnote, the value lives only in the DB + running process — see the durability
note below.

## Method 2 (fallback): curl to `/api/admin/keys`

```bash
curl -X POST https://dmai-web.onrender.com/api/admin/keys \
  -H "X-Master-Password: $MASTER_PASSWORD" \
  -H "Content-Type: application/json" \
  -d '{"provider_id":"groq","key":"<your-key>"}'
```

The JSON response includes `sinks.activator.provider_status`
(`active` / `invalid` / `pending_api_key`) — the same signal the dashboard toast
reflects.

## Method 3 (Render env only, requires redeploy)

Set the provider's env var in the Render dashboard (e.g. `GROQ_API_KEY`). A
redeploy normally picks it up. To hot-wire it into the **running** process
without waiting for a redeploy, open the dashboard → **API Keys** and click
**Rescan All Providers** (or `POST /api/harvester/scan`). The rescan re-reads
env vars, re-validates every key, and reports the activated / invalid / pending
lists.

## Durability note

A key stored in the **DB only** (not the Render env) will be **lost on the next
container recreation / redeploy**. Always mirror the key to the Render env var
for durability — either set `RENDER_API_KEY` so the dashboard can sync it
automatically, or add the env var manually in the Render dashboard.
