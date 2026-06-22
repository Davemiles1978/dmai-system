# DMAI Admin Harvester Panel — Integration Checklist
**Version:** 1.0 · June 2026  
**Scope:** API key management, health monitoring, cost controls, model fallback routing  
**Admin Panel:** `https://dmai-complete.onrender.com/admin` (JWT-gated)

---

## Overview

The Admin Harvester panel is powered by three live endpoints:

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /api/harvester/status` | None | Read current provider status from `data/api_registry.json` |
| `POST /api/harvester/scan` | JWT required | Force re-scan + hot-wire newly valid keys |
| `GET /api/harvester/providers` | None | Full catalogue: signup URLs, free tiers, env vars |

The `AutoAPIActivator` runs on startup and re-validates all keys every **3,600 seconds (1 hour)** via a background daemon thread. Valid keys are hot-wired into `AIIntegrationHub` without a restart.

---

## Section 1 — API Key Management via Environment Variables

### 1.1 All Recognised Environment Variables

Set these in the Render dashboard under **Environment → Add Environment Variable**. All values are `sync: false` (secret). Never commit key values to the repo.

#### Free-Tier Providers (Zero Cost to Activate)

| Provider | Env Var | Free Allowance | Signup |
|---|---|---|---|
| Groq | `GROQ_API_KEY` | 14,400 req/day | [console.groq.com/keys](https://console.groq.com/keys) |
| OpenRouter | `OPENROUTER_API_KEY` | $0 on free-tier models | [openrouter.ai/keys](https://openrouter.ai/keys) |
| Google AI Studio | `GOOGLE_AI_STUDIO_KEY` | 1,500 req/day, 250K TPM | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` | 10,000 neurons/day | [dash.cloudflare.com](https://dash.cloudflare.com/profile/api-tokens) |
| Cerebras | `CEREBRAS_API_KEY` | 1M tokens/day, 30 RPM | [cloud.cerebras.ai](https://cloud.cerebras.ai) |
| GitHub Models | `GITHUB_MODELS_TOKEN` | 45+ models, 15 RPM/150 RPD | [github.com/marketplace/models](https://github.com/marketplace/models) |
| Mistral AI | `MISTRAL_API_KEY` | 1B tokens/month, 2 RPM | [console.mistral.ai](https://console.mistral.ai) |
| Cohere | `COHERE_API_KEY` | 1,000 req/month, 20 RPM | [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) |
| HuggingFace | `HUGGINGFACE_API_KEY` | ~$0.10/month credit | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Tavily Search | `TAVILY_API_KEY` | 1,000 searches/month | [tavily.com](https://tavily.com/#api) |

#### Paid Providers (Already Wired — Add Keys When Ready)

| Provider | Env Var | Cost |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | $0.15/1M tokens (GPT-4o mini) |
| Anthropic | `ANTHROPIC_API_KEY` | Pay-per-use |
| Perplexity Sonar | `PERPLEXITY_API_KEY` | Pay-per-use (web search grounding) |
| DeepSeek | `DEEPSEEK_API_KEY` | $0.14/1M tokens |
| xAI Grok | `XAI_API_KEY` | Pay-per-use |

#### Supporting Services

| Service | Env Var | Purpose |
|---|---|---|
| ElevenLabs | `ELEVENLABS_API_KEY` | DMAI voice synthesis (TTS) |
| Brave Search | `BRAVE_SEARCH_API_KEY` | DeepResearch search grounding |
| GitHub (repo ops) | `GITHUB_TOKEN_MAIN` | Auto PR creation, knowledge updates |
| JWT | `JWT_SECRET` | Admin panel authentication |

---

### 1.2 Step-by-Step: Adding a Key in Render

- [ ] Log in to [dashboard.render.com](https://dashboard.render.com)
- [ ] Select service **dmai-complete**
- [ ] Click **Environment** in the left sidebar
- [ ] Click **Add Environment Variable**
- [ ] Enter the exact key name from the table above (case-sensitive)
- [ ] Paste the key value — no quotes, no spaces
- [ ] Click **Save Changes**
- [ ] Click **Manual Deploy → Deploy latest commit** (keys only apply after redeploy)
- [ ] After deploy (~2 min), open Admin panel → click **Scan & Hot-Wire APIs**
- [ ] Confirm the provider now shows **ACTIVE** (green) in the Harvester grid

> **Rule:** Never set a key to a placeholder value such as `pending`, `your_key_here`, or `none`. The AutoAPIActivator treats those strings as absent and marks the provider `pending_api_key`. Only set a key when you have a real value.

---

### 1.3 Key Rotation Procedure

When rotating a key (provider regenerated it or it was compromised):

- [ ] Generate the new key at the provider's dashboard
- [ ] In Render, update the env var value (do not delete and re-add — edit in place)
- [ ] Save → Manual Deploy
- [ ] Watch logs for: `AutoAPIActivator: hot-wired [Provider] into AIIntegrationHub`
- [ ] Verify status via `GET /api/harvester/status` — old key removed, new key `active`

---

## Section 2 — Health Check Implementation

### 2.1 How the AutoAPIActivator Validates Keys

Each provider has a `validation` block in `PROVIDER_CATALOGUE`. On every scan cycle the activator:

1. Reads the env var — if absent, marks `pending_api_key` (no HTTP call made)
2. Calls the provider's validation endpoint with a **minimal test request** (3–5 tokens max)
3. Interprets the response code:

| HTTP Code | Meaning | AutoAPIActivator Action |
|---|---|---|
| `200` / `201` | Key valid, request succeeded | Status → `active`, hot-wire to AIIntegrationHub |
| `429` | Rate limited — but key IS valid | Status → `active` (noted as rate-limited) |
| `401` | Authentication failed | Status → `invalid`, key excluded from pool |
| `402` | Billing limit / quota exceeded | Status → `quota_exceeded` |
| Other | Unexpected error | Status → `invalid`, error logged |
| Timeout (>12s) | Provider unreachable | Status → `invalid`, retry next cycle |

### 2.2 Health Check Endpoints to Monitor Manually

Use these in Render's **Monitoring** tab or any uptime checker (e.g. UptimeRobot):

```
# System health (Render uses this for zero-downtime deploys)
GET https://dmai-complete.onrender.com/health

# All provider statuses (no auth)
GET https://dmai-complete.onrender.com/api/harvester/status

# Full provider catalogue with active flags
GET https://dmai-complete.onrender.com/api/harvester/providers

# SICore KPI snapshot
GET https://dmai-complete.onrender.com/api/status
```

### 2.3 Setting Up the Scan Schedule

The background daemon runs automatically every hour. To also trigger on-demand:

- [ ] Open Admin panel → **API Harvester** tab
- [ ] Click **Scan & Hot-Wire APIs** button (calls `POST /api/harvester/scan` with your JWT)
- [ ] Results appear within 15–30 seconds (14 providers × ~1s per validation)
- [ ] Check `Last Scan` timestamp — if >2 hours old, the background thread may have died → trigger manual redeploy

For external scheduled monitoring, set a cron job or UptimeRobot to `GET /api/harvester/status` every 30 minutes and alert if `active` count drops below your minimum threshold (recommended: ≥ 3 active providers).

### 2.4 Reading the Registry File Directly

The scan results are persisted at `data/api_registry.json` on the Render instance. The structure is:

```json
{
  "timestamp": "2026-06-23T00:00:00Z",
  "total_active": 7,
  "activated": ["groq", "cerebras", "google_ai_studio"],
  "pending": ["openai", "anthropic"],
  "invalid": [],
  "providers": {
    "groq": {
      "status": "active",
      "latency_ms": 312,
      "key_prefix": "gsk_abc1...",
      "validated_at": "2026-06-23T00:00:00Z"
    }
  }
}
```

> **Note:** Render's free tier does not guarantee persistent disk. The registry is rebuilt from scratch on every cold start. This is by design — keys are always re-validated against live endpoints on startup.

---

## Section 3 — Cost-Monitoring Thresholds

### 3.1 Provider Cost Tiers

| Tier | Providers | Cost | Daily Risk |
|---|---|---|---|
| **Zero cost** | Groq, Cerebras, Google AI Studio, Cloudflare, OpenRouter (free models), GitHub Models, Mistral, Tavily | $0 | None |
| **Micro cost** | DeepSeek | $0.14/1M tokens | ~$0.01/day at normal load |
| **Low cost** | OpenAI (GPT-4o mini), Cohere | $0.15–$0.30/1M tokens | ~$0.05–0.15/day |
| **Medium cost** | OpenAI (GPT-4o), Perplexity Sonar | $2.50–$5/1M tokens | $0.50–2.00/day |
| **High cost** | Anthropic Claude Sonnet, Grok | $3–$15/1M tokens | $1–5/day |

### 3.2 Recommended Spend Thresholds

Set these alerts directly at each provider's dashboard — not in Render:

| Provider | Where to Set | Recommended Alert | Hard Limit |
|---|---|---|---|
| OpenAI | [platform.openai.com/settings/organization/billing](https://platform.openai.com/settings/organization/billing) → Spending limits | Alert at $5/month | Hard stop at $20/month |
| Anthropic | [console.anthropic.com/settings/plans](https://console.anthropic.com/settings/plans) | Alert at $5/month | Hard stop at $15/month |
| Perplexity | [docs.perplexity.ai](https://docs.perplexity.ai) dashboard | Alert at $5/month | Hard stop at $10/month |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) | Alert at $2/month | Hard stop at $5/month |
| OpenRouter | [openrouter.ai/credits](https://openrouter.ai/credits) | Alert when credits fall below $2 | — |

### 3.3 Cost-Control Rules Implemented in DMAI

The following rules are already enforced by `AutoAPIActivator` and `AIIntegrationHub`:

- **402 = immediate exclusion.** A `402 Payment Required` response marks the provider `quota_exceeded` and removes it from the active pool — no further charges accumulate in that scan cycle.
- **Free-tier models are preferred by default.** The model fallback table in Section 4 always routes to free providers first. Paid providers are only reached if all free-tier options for that task type have failed or are rate-limited.
- **Minimal validation tokens.** Health checks use `max_tokens: 3–5` to keep validation cost below $0.000001 per check. 14 providers × 1 check/hour = negligible cost.
- **No training on paid providers.** The `run_training.py` script targets only providers in the `free_tier` group unless explicitly overridden with `--allow-paid`.

### 3.4 Recommended Additional Guard: Token Budget Middleware

This is not yet implemented — add it to the next development cycle:

```python
# Suggested addition to dmai_core_complete.py api_chat() handler
PAID_PROVIDERS = {"openai", "anthropic", "perplexity", "deepseek", "grok"}
DAILY_PAID_TOKEN_BUDGET = 100_000  # ~$0.015 at GPT-4o mini rates

def _check_budget(provider_id: str, estimated_tokens: int) -> bool:
    """Return False if adding estimated_tokens would exceed the daily budget."""
    # Read from data/token_ledger.json (create if absent)
    # Sum today's paid token usage, compare to budget
    # Return True = allow, False = block and route to free provider
    ...
```

Add this to the Kaizen Queue (`POST /api/kaizen`) for tracking.

---

## Section 4 — Model Fallback Priority Table

The table below defines the routing order DMAI uses when selecting a model for each core function. The AutoAPIActivator's `get_active_providers()` is consulted at query time — if a provider is not `active`, the next row is tried automatically.

**Priority 1 = cheapest/fastest. Higher numbers = fallback only.**

### 4.1 General Chat / Reasoning

| Priority | Provider | Model | Cost | Notes |
|---|---|---|---|---|
| 1 | Cerebras | `llama-3.3-70b` | Free | Fastest — 2,600 tok/s. Use first. |
| 2 | Groq | `llama-3.3-70b-versatile` | Free | 14,400 req/day. Second fastest. |
| 3 | Google AI Studio | `gemini-2.0-flash` | Free | 1,500 req/day. Good reasoning. |
| 4 | GitHub Models | `gpt-4o-mini` | Free | 150 RPD. OpenAI quality at zero cost. |
| 5 | OpenRouter | `meta-llama/llama-3.3-70b-instruct:free` | Free | No daily hard cap on free models. |
| 6 | Mistral AI | `mistral-large-latest` | Free | 2 RPM cap — use only if all above fail. |
| 7 | DeepSeek | `deepseek-chat` | $0.14/1M | Paid fallback. Excellent value. |
| 8 | OpenAI | `gpt-4o-mini` | $0.15/1M | Paid fallback. |
| 9 | Anthropic | `claude-3-haiku-20240307` | Paid | Last resort — highest reliability. |

### 4.2 Code Generation

| Priority | Provider | Model | Cost | Notes |
|---|---|---|---|---|
| 1 | Cerebras | `llama-3.3-70b` | Free | Strong code at 2,600 tok/s. |
| 2 | Groq | `qwen-qwq-32b` | Free | Excellent at structured code output. |
| 3 | GitHub Models | `o4-mini` | Free (high-tier) | Best free code model. 50 RPD. |
| 4 | Mistral AI | `codestral-latest` | Free | Purpose-built code model. 2 RPM cap. |
| 5 | DeepSeek | `deepseek-chat` | $0.14/1M | Best paid value for code. |
| 6 | OpenAI | `gpt-4o` | $2.50/1M | High-quality code fallback. |

### 4.3 DeepResearch / Web-Grounded Queries

| Priority | Provider | Model | Cost | Notes |
|---|---|---|---|---|
| 1 | Tavily | `tavily-search` | Free (1K/month) | Primary search grounding source. |
| 2 | Brave Search | `brave-search` | Free tier | Backup grounding when Tavily exhausted. |
| 3 | Perplexity Sonar | `sonar` | Paid | Real-time web + AI. Use only when search grounding is critical and free sources exhausted. |
| 4 | Google AI Studio | `gemini-2.0-flash` | Free | Use for synthesis pass after Tavily retrieval. |

### 4.4 Long-Context / Document Tasks (>8K tokens)

| Priority | Provider | Model | Cost | Notes |
|---|---|---|---|---|
| 1 | Google AI Studio | `gemini-2.0-flash` | Free | 1M context window. Best free long-context option. |
| 2 | GitHub Models | `meta/llama-4-maverick-17b-128e-instruct` | Free | 128K context. 50 RPD. |
| 3 | Mistral AI | `mistral-large-latest` | Free | 128K context. 2 RPM — serial only. |
| 4 | OpenAI | `gpt-4o` | $2.50/1M | 128K context. Paid fallback. |
| 5 | Anthropic | `claude-3-5-sonnet-20241022` | Paid | 200K context. Last resort for very long docs. |

### 4.5 Content Generation (Books, Scripts, KDP)

| Priority | Provider | Model | Cost | Notes |
|---|---|---|---|---|
| 1 | Groq | `llama-3.3-70b-versatile` | Free | Fast, high-quality prose. |
| 2 | Cerebras | `llama-3.3-70b` | Free | Speed advantage for bulk generation. |
| 3 | Mistral AI | `mistral-large-latest` | Free | Strong creative writing quality. |
| 4 | Google AI Studio | `gemini-2.0-flash` | Free | Good for structured content formats. |
| 5 | OpenRouter | `meta-llama/llama-3.3-70b-instruct:free` | Free | Overflow buffer when others rate-limited. |
| 6 | OpenAI | `gpt-4o` | $2.50/1M | Premium quality for final-draft passes only. |

### 4.6 SICore Training / Knowledge Ingestion

| Priority | Provider | Model | Cost | Notes |
|---|---|---|---|---|
| 1 | Cerebras | `llama-3.1-8b` | Free | Use small model for bulk training — speed matters more than quality here. |
| 2 | Groq | `llama-3.1-8b-instant` | Free | 14,400 req/day for high-volume training rounds. |
| 3 | Cloudflare | `@cf/meta/llama-3.1-8b-instruct` | Free | 10,000 neurons/day. Good overflow capacity. |
| 4 | HuggingFace | `mistralai/Mistral-7B-Instruct-v0.3` | Free | Large model catalogue — use for specialist domains. |
| 5 | OpenRouter | `google/gemma-3-12b-it:free` | Free | Zero cost overflow. |

> **Never use paid providers for SICore training rounds.** The volume (hundreds of prompts per session) makes costs unpredictable. The free tier providers above have sufficient combined capacity for all 8 SICore KPI domains.

### 4.7 SICore KPI Domain → Recommended Primary Provider

| KPI Domain | Primary Provider | Reason |
|---|---|---|
| `skill_acquisition_rate` | Cerebras | High throughput enables rapid iteration |
| `transfer_learning_rate` | Google AI Studio | Gemini Flash excels at cross-domain synthesis |
| `zero_shot_success_count` | GitHub Models (gpt-4o-mini) | OpenAI instruction-following for zero-shot reliability |
| `agentic_capability_score` | Groq | Sub-second latency for tight agentic loops |
| `recursive_self_improvement_rate` | Mistral (`mistral-large-latest`) | Strong meta-reasoning at zero cost |
| `sample_efficiency_trend` | Cerebras | 1M tokens/day enables large sample batches |
| `metacognition_accuracy` | Google AI Studio | Gemini Flash strong on self-evaluation tasks |
| `multi_modal_integration_score` | GitHub Models (`o4-mini`) or Mistral (`pixtral-large-latest`) | Vision + text in same free tier |

---

## Section 5 — Final Integration Checklist

Work through this top to bottom on first setup, or after any infrastructure change.

### Phase A — Environment

- [ ] All free-tier provider keys set in Render (Section 1.1 zero-cost table)
- [ ] `CLOUDFLARE_ACCOUNT_ID` set alongside `CLOUDFLARE_API_KEY`
- [ ] `JWT_SECRET` set (admin panel will not load without it)
- [ ] `TAVILY_API_KEY` or `BRAVE_SEARCH_API_KEY` set (required for DeepResearch)
- [ ] `GITHUB_TOKEN_MAIN` set (required for nightly knowledge evolution cron)
- [ ] No placeholder values in any env var (`pending`, `none`, `your_key_here`)

### Phase B — Health Verification

- [ ] Manual deploy triggered after all keys are set
- [ ] `GET /api/harvester/status` returns `total_active ≥ 4`
- [ ] `POST /api/harvester/scan` (via Admin panel) confirms hot-wiring messages in logs
- [ ] `GET /health` returns `200 OK`
- [ ] `GET /api/status` shows `si_kpis` present (not empty `{}`)

### Phase C — Cost Controls

- [ ] Spend alerts configured at OpenAI, Anthropic, and Perplexity dashboards (Section 3.2)
- [ ] Hard monthly limits set at all paid providers
- [ ] Confirmed `run_training.py` does not target paid providers (check `--allow-paid` flag is absent)
- [ ] Confirmed `402` responses are being caught by AutoAPIActivator (check logs after any quota event)

### Phase D — Fallback Routing Smoke Test

- [ ] Temporarily remove `GROQ_API_KEY` from Render env → trigger scan → confirm Groq shows `pending_api_key`
- [ ] Send a test chat message → confirm response still arrives (via Cerebras or Google AI Studio)
- [ ] Re-add `GROQ_API_KEY` → trigger scan → confirm Groq returns to `active`
- [ ] Confirm fallback routing does not escalate to paid providers when ≥ 3 free providers are active

### Phase E — Ongoing Maintenance

- [ ] Check Admin Harvester panel weekly for any providers drifting to `invalid`
- [ ] Review `data/api_registry.json` monthly — check latency trends
- [ ] Rotate provider keys every 90 days (set a calendar reminder)
- [ ] Add new providers to `PROVIDER_CATALOGUE` in `auto_api_activator.py` as they emerge
- [ ] After any DMAI model update, re-run the SICore KPI → provider mapping (Section 4.7) to ensure the best free model is still assigned per domain

---

*This document is maintained at `docs/ADMIN_HARVESTER_INTEGRATION_CHECKLIST.md` in the `Davemiles1978/dmai-system` repo. Update it whenever new providers are added or limits change.*
