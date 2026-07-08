# BuyWhere SG API

Real-time Singapore price comparison with a unified schema. Built for agents that need to reason about price, availability, and vendor reliability — not parse HTML.

---

## Install

```bash
pip install buywhere-sg
```

---

## Three lines to best price in SGD

```python
from buywhere_sg import BuyWhereSGClient

client = BuyWhereSGClient(api_key="YOUR_API_KEY")
result = await client.prices.compare("Sony WH-1000XM5")

print(result.best_available.price_sgd, result.price_dispersion_bits)
# 389.00 SGD  |  dispersion: 2.41 bits  (high spread — worth checking Courts vs Harvey Norman)
```

---

## Why not scrape BuyWhere yourself

You could. Here is what that project looks like after two weeks:

- A Playwright script that breaks when BuyWhere updates their DOM (they do, often)
- No normalization across Lazada, Shopee, Courts, Harvey Norman — each returns price in a different field, currency format, and availability string
- No history. You know the price right now. You don't know if it spiked yesterday
- Zero integration with AutoGen or LangChain — your agent still needs a custom parsing layer on top of your custom scraper
- No signal for when a price is anomalous vs. when the market genuinely moved

The BuyWhere SG API handles all of that and exposes it through a single OpenAPI contract your agent can call directly.

---

## What you get

**Unified schema across SG retailers**
Every vendor response — Lazada, Shopee, Courts, Harvey Norman, Challenger — is normalized to the same `PriceListing` object. `price_sgd`, `availability_type` (`in_store` | `online` | `both`), `vendor_id`, `last_verified_utc`. No per-source parsing.

**Price history in SGD, no conversion guesswork**
Every product carries a `price_history` array. TimescaleDB underneath. Ask for 7d, 30d, or 90d windows. Spot a Courts promotion before your competitor does.

**Shannon entropy as a first-class field**
`price_dispersion_bits` is the information-theoretic entropy of the price distribution across vendors at query time. High entropy (> 2 bits) means vendors disagree significantly — actionable signal that the market is fragmented and a deal likely exists. Low entropy means prices have converged — you're paying market rate everywhere. No scraper gives you this.

**MCP server with domain-scoped tool specs**
Drop the MCP server into any AutoGen or LangChain agent. Tools are scoped by product domain:

```
buywhere_electronics_compare_prices
buywhere_appliances_availability_by_store
buywhere_computing_price_trend_sgd
```

An agent reading the tool name alone knows what it does and when to call it. No ambiguity, no prompt engineering to coax it into the right path.

**Self-refreshing data flywheel**
Every `/prices` call checks snapshot age against `TTL_VENDOR_FRESHNESS`. If stale, a background task re-ingests that vendor's listing before returning. High-volume callers automatically fund fresher data for the entire user base — marginal cost of freshness approaches zero at scale.

---

## Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| `GET` | `/prices/compare` | Best price + full vendor breakdown + `price_dispersion_bits` |
| `GET` | `/prices/history/{product_id}` | SGD price history with configurable window |
| `GET` | `/availability/stores` | Physical store stock by vendor and region |
| `GET` | `/products/search` | Catalog search with domain filter (`electronics`, `appliances`, `computing`) |
| `POST` | `/alerts/price-anomaly` | Webhook registration for entropy-triggered price anomaly events |

---

## Agent integration (AutoGen)

```python
from autogen import AssistantAgent
from buywhere_sg.mcp import BuyWhereMCPServer

mcp = BuyWhereMCPServer(api_key="YOUR_API_KEY")

agent = AssistantAgent(
    name="price_researcher",
    tools=mcp.tools(domains=["electronics", "computing"]),
)

# Agent can now reason: "Is the price on the Samsung monitor anomalous?
# Should I recommend waiting, or is dispersion low and this is market rate?"
```

No parsing. No prompt gymnastics. The tool spec carries the semantics.

---

## Authentication

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "https://api.buywhere-sg.com/v1/prices/compare?q=LG+C3+OLED+55"
```

API keys are scoped to read (`sk_test_xxxxxxxxxxxxxxxx` for sandbox, production keys provisioned via dashboard).

---

## Requirements

- Python 3.11+
- Works with AutoGen 0.2+, LangChain 0.1+, or any HTTP client
- No credentials to Singapore retailers required — that complexity lives in the API layer, not yours

---

## Status and SLA

Endpoint health, p50/p99 latency, and vendor freshness metrics: **status.buywhere-sg.com**

---

*Built on FastAPI + asyncpg + Redis + TimescaleDB. Schema versioned. Breaking changes communicated 30 days in advance.*

---

## Pricing

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |

No base fee. No storage fee. No minimum commitment. You pay for computation, not for parking vectors you queried once.