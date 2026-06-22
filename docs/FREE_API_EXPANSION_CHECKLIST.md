# DMAI Free API Expansion — Integration Checklist
**Date:** June 2026  
**Providers:** Cerebras Inference · GitHub Models · Mistral AI  
**Status:** Code committed — API keys pending activation

---

## What Was Built

Three new free-tier providers have been fully wired into DMAI. No code changes are required — just set the API keys and redeploy.

| Provider | Free Tier | Speed | Best Model | Env Var |
|---|---|---|---|---|
| **Cerebras** | 1M tok/day, 30 RPM | 2,600+ tok/s | `llama-3.3-70b` | `CEREBRAS_API_KEY` |
| **GitHub Models** | 45+ models, 15 RPM | Standard | `gpt-4o-mini` | `GITHUB_MODELS_TOKEN` |
| **Mistral AI** | 1B tok/month, 2 RPM | Standard | `mistral-large-latest` | `MISTRAL_API_KEY` |

---

## Step 1 — Get Your API Keys (15 minutes total)

### 1a. Cerebras Inference
- [ ] Go to [cloud.cerebras.ai](https://cloud.cerebras.ai)
- [ ] Sign up with email — **no credit card required**
- [ ] Navigate to **API Keys** → **Create new key**
- [ ] Copy the key (starts with `csk-`)
- [ ] Note: You get **1,000,000 tokens/day** permanently for free

### 1b. GitHub Models
- [ ] Go to [github.com/marketplace/models](https://github.com/marketplace/models)
- [ ] You're already signed in to GitHub — just click **Get started**
- [ ] Generate a **Fine-grained Personal Access Token** at [github.com/settings/tokens](https://github.com/settings/tokens)
  - Token name: `DMAI-Models`
  - Expiration: No expiration (or 1 year)
  - Permissions: **Models** → Read (under "Account permissions")
- [ ] Copy the token (starts with `github_pat_`)
- [ ] Note: `GITHUB_TOKEN_MAIN` already set for repo ops — can reuse if it has Models read permission

### 1c. Mistral AI
- [ ] Go to [console.mistral.ai](https://console.mistral.ai)
- [ ] Sign up — requires **phone number verification**, no credit card
- [ ] Stay on the free **Experiment** plan
- [ ] Navigate to **API Keys** → **Create new key**
- [ ] Copy the key (starts with a random string)
- [ ] Note: Experiment plan gives access to **all models** including Large and Codestral

---

## Step 2 — Add Keys to Render Dashboard

- [ ] Log in to [dashboard.render.com](https://dashboard.render.com)
- [ ] Select the **dmai-complete** service
- [ ] Click **Environment** in the left sidebar
- [ ] Add / update these three environment variables:

| Key | Value | Notes |
|---|---|---|
| `CEREBRAS_API_KEY` | `csk-xxxx...` | From cloud.cerebras.ai |
| `GITHUB_MODELS_TOKEN` | `github_pat_xxxx...` | From github.com/settings/tokens |
| `MISTRAL_API_KEY` | `xxxx...` | From console.mistral.ai |

- [ ] Click **Save Changes**
- [ ] Click **Manual Deploy** → **Deploy latest commit**

---

## Step 3 — Verify in Admin Harvester Panel

Once redeployed (2–3 minutes):

- [ ] Open [https://dmai-complete.onrender.com/admin](https://dmai-complete.onrender.com/admin)
- [ ] Log in with JWT token
- [ ] Click **API Harvester** panel in the left sidebar
- [ ] Click **Scan & Hot-Wire APIs** button
- [ ] Verify these three appear as **ACTIVE** (green):
  - `Cerebras Inference`
  - `GitHub Models`
  - `Mistral AI`
- [ ] If any show **INVALID**: regenerate the key at the provider dashboard and repeat Step 2

---

## Step 4 — Test the New Tutors

- [ ] Open [https://dmai-complete.onrender.com/chat](https://dmai-complete.onrender.com/chat)
- [ ] Send a test message (e.g. `"What is meta-learning?"`)
- [ ] Check the response includes contributions from the new tutors
- [ ] Confirm Cerebras responds fastest (2,600 tok/s vs ~100 tok/s for others)

---

## Step 5 — Rate Limit Awareness

Important limits to keep in mind:

| Provider | RPM | RPD | Daily Token Cap |
|---|---|---|---|
| Cerebras | 30 | 14,400 | 1M tokens |
| GitHub Models (low-tier) | 15 | 150 | No hard limit |
| Mistral Experiment | 2 | ~2,880 | ~33M tokens/day (within 1B/month) |

**Mistral note:** 2 RPM is strict. DMAI's `query_all_tutors()` queries sequentially — Mistral will not be called more than once per 30 seconds under normal load. If you start bulk training loops, add a `time.sleep(31)` between Mistral calls or switch to the `mistral-small-latest` model which may have looser limits.

---

## Step 6 — Optional: Expand GitHub Models to Frontier Tier

GitHub Models has two usage tiers per model. To access the high-tier models (o4-mini, GPT-4.1, Llama 4 Maverick):

- [ ] Go to [github.com/marketplace/models](https://github.com/marketplace/models) and verify your account tier
- [ ] High-tier limits: 10 RPM / 50 RPD — lower rate but much more powerful models
- [ ] To use a different model, edit `AIIntegrationHub._query_github_models()`:
  ```python
  'model': 'gpt-4.1',  # Change from 'gpt-4o-mini'
  ```
  Or pass `model='gpt-4.1'` when calling `query_github_models()` directly.

---

## Files Changed in This Update

| File | Change |
|---|---|
| `components/integration/auto_api_activator.py` | Added `cerebras`, `github_models`, `mistral` to `PROVIDER_CATALOGUE` and `hub_key_map` |
| `components/phase11/AIIntegrationHub.py` | Added `_query_cerebras`, `_query_github_models`, `_query_mistral`; updated `_load_api_keys`, `query_methods`, `_get_active_tutors` |
| `components/integration/new_provider_connectors.py` | **New file** — standalone health check + query module |
| `render.yaml` | Added `CEREBRAS_API_KEY` and `GITHUB_MODELS_TOKEN` env var declarations |

---

## KPI Impact Estimate

| SICore KPI | Expected Improvement | Reason |
|---|---|---|
| `skill_acquisition_rate` | +15–25% | 3 extra diverse tutors = more response perspectives to synthesise from |
| `zero_shot_success_count` | +10–20% | Mistral Large and GPT-4.1 (via GitHub) are top-tier zero-shot performers |
| `agentic_capability_score` | +5–10% | Cerebras' speed enables tighter agentic loops with sub-second latency |
| `sample_efficiency_trend` | +10–15% | 1B extra free tokens/month = more training rounds without cost |
| `multi_modal_integration_score` | +5% | Pixtral (Mistral) and Llama 4 (GitHub) add vision capability |

> **Note:** These are directional estimates only. Actual KPI changes will be measured by the SICore monitor after the first 7-day training cycle with the new tutors active.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Cerebras shows `invalid` | Key typo or expired | Regenerate at cloud.cerebras.ai |
| GitHub Models shows `invalid` | Token missing Models permission | Create new fine-grained token with Models read |
| Mistral shows `quota_exceeded` | 1B token monthly cap hit | Wait for month reset or upgrade to Codestral-only |
| All three show `pending_api_key` | Env vars not saved in Render | Repeat Step 2 and trigger Manual Deploy |
| Admin panel not loading | Render free tier spun down | Wait 30–60s for cold start, refresh |
