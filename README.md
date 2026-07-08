# BuyWhere MCP — Singapore Product & Price Intelligence for LLM Agents

The only MCP-native tool that exposes BuyWhere Singapore's catalogue with semantic search, SGD price normalization, and an auditable value-score — so your agent can recommend *and justify* the best purchase, not just return the cheapest number.

---

## Install

```bash
pip install buywhere-mcp
```

---

## 30-second example

```python
from buywhere_mcp import BuyWhereClient

client = BuyWhereClient(api_key="YOUR_API_KEY")
result = client.search_ranked_by_value("Sony WH-1000XM5 Singapore")
print(result.top_pick)  # -> Product(vendor, price_sgd, value_score, justification)
```

---

## What you get back

Every response is structured JSON, ready for direct LLM consumption — no parsing, no normalization, no post-processing:

```json
{
  "query": "Sony WH-1000XM5 Singapore",
  "top_pick": {
    "vendor": "Lazada SG",
    "price_sgd": 389.00,
    "value_score": 0.847,
    "value_score_breakdown": {
      "price_entropy_H": 1.24,
      "vendor_variance_penalty": 0.08,
      "normalized_rank": 0.91
    },
    "justification": "Lowest price in a high-entropy market (H=1.24 bits across 6 vendors); vendor shows <4% intra-session price variance — reliable quote."
  },
  "all_vendors": [...]
}
```

Your agent can cite `value_score`, `price_entropy_H`, and `justification` verbatim in its reasoning chain. No black-box "best pick" — every recommendation is traceable to a formula.

---

## Why not build this yourself

| What you'd have to build | What you get here |
|---|---|
| Scraper that survives BuyWhere's layout changes | Maintained, versioned API endpoint |
| SGD normalization + multi-vendor deduplication | Built-in, updated with SGX fx rates |
| Semantic search over product titles + specs | Embedding-based retrieval, no regex hacks |
| Value metric your agent can cite in reasoning | Shannon entropy + causal variance penalty, auditable |
| MCP tool spec consumable by AutoGen / LangChain | Drop-in tool registration, zero glue code |

The hard part isn't the HTTP call. It's knowing that `$389 at Lazada` and `$389.90 at Shopee` are the same SKU, that the Lazada price has been stable for 72 hours while the Shopee price spiked 18% yesterday, and expressing that difference as a single number a language model can reason about. That logic is what this primitive contains.

---

## Value-score: the math in one paragraph

For each product, we collect prices `{p_1 ... p_n}` across all vendors. We normalize each price to a relative position within the observed range, then compute Shannon entropy `H = -sum(p_i * log2(p_i))` over that distribution. High entropy means prices are genuinely spread across vendors — real competition, real arbitrage opportunity. Low entropy means all vendors cluster together — the "cheapest" pick matters less. We then apply a causal penalty for vendors whose own price has varied more than ±5% within the current scrape session (a proxy for quote unreliability). The final `value_score` is `normalized_price_rank * (1 - variance_penalty)`, bounded `[0, 1]`. A score of `1.0` means: cheapest vendor in the most competitive market, with the most stable quote. An agent can reproduce this calculation from the returned breakdown fields.

---

## MCP tool registration (AutoGen / LangChain)

```python
from buywhere_mcp import BuyWhereMCPTool

# AutoGen
tools = [BuyWhereMCPTool(api_key="YOUR_API_KEY")]

# LangChain
from langchain.agents import initialize_agent
agent = initialize_agent(tools=[BuyWhereMCPTool(api_key="YOUR_API_KEY")], ...)
```

Tools exposed:

- `search_singapore_products_ranked` — semantic search + full ranked list
- `get_product_value_score` — value-score for a known SKU or URL
- `compare_vendors_by_entropy` — head-to-head vendor reliability report

Each tool name describes the operation. An agent reading the name without the description can infer what it does.

---

## Requirements

- Python 3.11+
- FastAPI (if self-hosting the MCP server)
- No browser, no Playwright, no Selenium — pure API

---

## Quickstart with your own MCP server

```bash
git clone https://github.com/nexus-forge/buywhere-mcp
cd buywhere-mcp
cp .env.example .env          # add YOUR_API_KEY
uvicorn buywhere_mcp.server:app --port 8000
```

---

## Auth

All requests require `X-API-Key` header or `api_key` constructor argument.

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "https://api.buywhere-mcp.io/v1/search?q=Sony+WH-1000XM5&market=sg"
```

Keys are scoped per environment. Never commit keys to source control — use environment variables.

---

## Status & support

- API status: [status.buywhere-mcp.io](https://status.buywhere-mcp.io)
- Issues: GitHub Issues
- Docs: [docs.buywhere-mcp.io](https://docs.buywhere-mcp.io)

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