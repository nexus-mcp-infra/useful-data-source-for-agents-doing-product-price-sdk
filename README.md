# BuyWhere Singapore Value Intelligence API

Semantic product search over Singapore e-commerce (BuyWhere), ranked by an auditable value-score derived from Shannon entropy across vendor price distributions - not just "cheapest first."

Available both as a plain HTTP API and as an MCP server (5 tools) for AI agents.

---

## Base URL
https://useful-data-source-for-agents-production.up.railway.app

## Authentication

All business endpoints require an `X-API-Key` header:
X-API-Key: <your key>

`/health` requires no authentication.

## Pricing

Two ways to pay, same endpoints:

- **x402 (pay-per-call, USDC on Base)** - currently on **Base Sepolia testnet**, $0.01/call. A request without payment gets `402 Payment Required` with the payment details in the `payment-required` response header.
- **Stripe (metered billing)** - for callers provisioned with an API key and a Stripe customer on the account.

---

## Endpoints

### `GET /search`
Semantic product search over the BuyWhere Singapore catalog, ranked by composite value-score.

Query params:
- `query` (required) - natural-language product description
- `limit` (default 10, 1-50)
- `min_value_score` (default 0.0, range 0.0-1.0) - filter out products below this score
GET /search?query=noise+cancelling+headphones&limit=5&min_value_score=0.5

Response (`ProductSearchResponse`):
```json
{
  "query": "noise cancelling headphones",
  "result_count": 5,
  "currency": "SGD",
  "products": [
    {
      "product_id": "abc123",
      "title": "Sony WH-1000XM5",
      "category": "electronics",
      "image_url": "...",
      "vendor_offers": ["..."],
      "price_min_sgd": 389.00,
      "price_max_sgd": 429.00,
      "price_mean_sgd": 405.30,
      "shannon_entropy_bits": 1.24,
      "causal_reliability_score": 0.92,
      "value_score": 0.847,
      "value_score_explanation": "..."
    }
  ],
  "computation_ms": 43.2
}
```

### `GET /product/{product_id}/value_breakdown`
Full auditable decomposition of the value-score formula for one product - entropy component, reliability component, price-rank component, the formula itself, and the recommended vendor/price.

### `GET /product/{product_id}/price_distribution`
Vendor price distribution for one product: entropy (raw and normalized), price spread, coefficient of variation, and a plain-language interpretation of market fragmentation.

### `GET /health`
Liveness probe. Returns `{ status, upstream_reachable, api_version, timestamp }`. No auth required. Not billed (excluded from both Stripe and x402).

---

## MCP tools

Connect an MCP-compatible client (Claude, Cursor, etc.) to the streamable HTTP endpoint at:
https://useful-data-source-for-agents-production.up.railway.app/mcp

Exposes 5 tools:
- `rank_buywhere_products_by_value_score` - ranked search with `max_price_sgd` and `category_filter`
- `fetch_buywhere_vendor_price_distribution` - vendor spread for a known `product_id`
- `resolve_buywhere_product_identity` - disambiguate a partial/fuzzy product name into candidate product IDs
- `compare_buywhere_products_causal_rank` - head-to-head comparison of 2-10 products with confounder adjustment
- `extract_buywhere_deal_anomalies` - z-score based detection of mispriced or outlier listings

---

## Value-score: the method

For each product, prices across vendors are treated as a distribution. The composite `value_score` blends:

- **Entropy component** - Shannon entropy of the price distribution across vendors (`H`, in bits) normalized against the maximum possible entropy for that vendor count. Higher entropy = genuine price competition, not vendors clustered at the same number.
- **Reliability component** - a causal-reliability score per vendor.
- **Price-rank component** - where this vendor's price sits in the observed range.

Every response includes the raw components (`shannon_entropy_bits`, `entropy_component`, `reliability_component`, `price_rank_component`) and, on `/product/{id}/value_breakdown`, the `formula` string itself - so an agent can cite the reasoning, not just the number.

---

## Limits

- `/search`: `limit` up to 50 results per call
- MCP `compare_buywhere_products_causal_rank`: 2-10 product IDs per call
