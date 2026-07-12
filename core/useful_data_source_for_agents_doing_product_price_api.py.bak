from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import httpx
import numpy as np
from scipy.stats import entropy as scipy_entropy
import os
import time
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("buywhere_singapore_api")

app = FastAPI(
    title="BuyWhere Singapore Value Intelligence API",
    description="Semantic product search over Singapore e-commerce with auditable value-scores derived from Shannon entropy across vendor price distributions.",
    version="1.0.0",
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
VALID_API_KEYS = set(filter(None, os.environ.get("BUYWHERE_API_KEYS", "").split(",")))
BUYWHERE_BASE_URL = os.environ.get("BUYWHERE_BASE_URL", "https://www.buywhere.com.sg")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "12.0"))
MAX_VENDORS_PER_PRODUCT = 20
SGD_FLOOR = 0.01
SGD_CEILING = 1_000_000.0


class VendorOffer(BaseModel):
    vendor_name: str
    price_sgd: float
    listing_url: str
    in_stock: bool
    intra_session_variance_proxy: float
    reliability_penalty: float


class RankedProduct(BaseModel):
    product_id: str
    title: str
    category: str
    image_url: Optional[str]
    vendor_offers: list[VendorOffer]
    price_min_sgd: float
    price_max_sgd: float
    price_mean_sgd: float
    shannon_entropy_bits: float
    causal_reliability_score: float
    value_score: float
    value_score_explanation: str


class ProductSearchResponse(BaseModel):
    query: str
    result_count: int
    currency: str
    products: list[RankedProduct]
    computation_ms: float


class PriceDistributionResponse(BaseModel):
    product_id: str
    title: str
    vendor_count: int
    entropy_bits: float
    entropy_normalized: float
    price_spread_sgd: float
    coefficient_of_variation: float
    interpretation: str


class ValueScoreBreakdownResponse(BaseModel):
    product_id: str
    title: str
    shannon_entropy_bits: float
    max_possible_entropy_bits: float
    entropy_component: float
    reliability_component: float
    price_rank_component: float
    composite_value_score: float
    formula: str
    recommended_vendor: str
    recommended_price_sgd: float


class HealthResponse(BaseModel):
    status: str
    upstream_reachable: bool
    api_version: str
    timestamp: float


def _require_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if not VALID_API_KEYS:
        logger.warning("No API keys configured — running in open mode (not for production)")
        return api_key
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return api_key


def _validate_query(query: str) -> str:
    if not query or not isinstance(query, str):
        raise HTTPException(status_code=422, detail="Query must be a non-empty string.")
    query = query.strip()
    if len(query) < 2:
        raise HTTPException(status_code=422, detail="Query must be at least 2 characters.")
    if len(query) > 256:
        raise HTTPException(status_code=422, detail="Query must not exceed 256 characters.")
    return query


def _validate_product_id(product_id: str) -> str:
    if not product_id or not isinstance(product_id, str):
        raise HTTPException(status_code=422, detail="product_id must be a non-empty string.")
    product_id = product_id.strip()
    if len(product_id) < 1 or len(product_id) > 128:
        raise HTTPException(status_code=422, detail="product_id length must be between 1 and 128 characters.")
    return product_id


def _compute_intra_session_variance_proxy(prices: list[float], vendor_index: int) -> float:
    if len(prices) < 2:
        return 0.0
    arr = np.array(prices, dtype=np.float64)
    global_std = float(np.std(arr))
    global_mean = float(np.mean(arr))
    if global_mean == 0:
        return 0.0
    cv = global_std / global_mean
    price = prices[vendor_index]
    z_score = abs(price - global_mean) / (global_std + 1e-9)
    return float(cv * z_score)


def _compute_reliability_penalty(variance_proxy: float) -> float:
    return float(1.0 / (1.0 + np.exp(-variance_proxy + 1.5)))


def _compute_shannon_entropy_bits(prices: list[float]) -> float:
    if len(prices) < 2:
        return 0.0
    arr = np.array(prices, dtype=np.float64)
    price_min = arr.min()
    price_max = arr.max()
    price_range = price_max - price_min
    if price_range < 1e-9:
        return 0.0
    relative_positions = (arr - price_min) / price_range
    bin_count = min(len(arr), 10)
    counts, _ = np.histogram(relative_positions, bins=bin_count, range=(0.0, 1.0))
    counts = counts[counts > 0].astype(np.float64)
    probabilities = counts / counts.sum()
    h = float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))
    return round(h, 6)


def _compute_causal_reliability_score(vendor_offers_raw: list[dict]) -> float:
    if not vendor_offers_raw:
        return 0.5
    prices = [v["price_sgd"] for v in vendor_offers_raw]
    variance_proxies = [
        _compute_intra_session_variance_proxy(prices, i)
        for i in range(len(prices))
    ]
    penalties = [_compute_reliability_penalty(vp) for vp in variance_proxies]
    mean_penalty = float(np.mean(penalties))
    causal_score = 1.0 - mean_penalty
    return round(max(0.0, min(1.0, causal_score)), 6)


def _compute_value_score(
    price_min_sgd: float,
    price_mean_sgd: float,
    shannon_entropy_bits: float,
    causal_reliability_score: float,
    vendor_count: int,
) -> tuple[float, str]:
    max_possible_entropy = float(np.log2(max(vendor_count, 2)))
    entropy_normalized = shannon_entropy_bits / (max_possible_entropy + 1e-9)
    price_efficiency = price_min_sgd / (price_mean_sgd + 1e-9)
    price_rank_component = 1.0 - price_efficiency
    w_entropy = 0.35
    w_reliability = 0.40
    w_price_rank = 0.25
    value_score = (
        w_entropy * entropy_normalized
        + w_reliability * causal_reliability_score
        + w_price_rank * price_rank_component
    )
    value_score = round(max(0.0, min(1.0, value_score)), 6)
    explanation = (
        f"value_score={value_score:.4f} = "
        f"0.35 * H_norm({entropy_normalized:.4f}) + "
        f"0.40 * reliability({causal_reliability_score:.4f}) + "
        f"0.25 * price_rank({price_rank_component:.4f}); "
        f"H={shannon_entropy_bits:.4f} bits over {vendor_count} vendors; "
        f"price_min={price_min_sgd:.2f} SGD, price_mean={price_mean_sgd:.2f} SGD"
    )
    return value_score, explanation


def _deterministic_product_id(title: str, category: str) -> str:
    raw = f"{title.lower().strip()}::{category.lower().strip()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


async def _fetch_buywhere_search(query: str, limit: int) -> list[dict]:
    params = {"q": query, "limit": limit, "country": "SG", "currency": "SGD"}
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(f"{BUYWHERE_BASE_URL}/api/search", params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("products", [])
    except httpx.HTTPStatusError as exc:
        logger.error("BuyWhere upstream HTTP error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"BuyWhere upstream returned HTTP {exc.response.status_code}.",
        )
    except httpx.RequestError as exc:
        logger.error("BuyWhere upstream unreachable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="BuyWhere upstream unreachable. Retry after backoff.",
        )
    except Exception as exc:
        logger.error("Unexpected error fetching BuyWhere data: %s", exc)
        raise HTTPException(status_code=500, detail="Internal error fetching product data.")


async def _fetch_buywhere_product(product_id: str) -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(f"{BUYWHERE_BASE_URL}/api/product/{product_id}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"BuyWhere upstream returned HTTP {exc.response.status_code} for product {product_id}.",
        )
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="BuyWhere upstream unreachable.")


def _normalize_vendor_offers(raw_vendors: list[dict], prices: list[float]) -> list[VendorOffer]:
    offers = []
    for i, v in enumerate(raw_vendors):
        price_sgd = float(v.get("price", 0.0))
        if not (SGD_FLOOR <= price_sgd <= SGD_CEILING):
            continue
        variance_proxy = _compute_intra_session_variance_proxy(prices, i)
        reliability_penalty = _compute_reliability_penalty(variance_proxy)
        offers.append(VendorOffer(
            vendor_name=str(v.get("vendor", "unknown")),
            price_sgd=round(price_sgd, 2),
            listing_url=str(v.get("url", "")),
            in_stock=bool(v.get("in_stock", True)),
            intra_session_variance_proxy=round(variance_proxy, 6),
            reliability_penalty=round(reliability_penalty, 6),
        ))
    return offers


def _build_ranked_product(raw: dict) -> RankedProduct:
    title = str(raw.get("title", ""))
    category = str(raw.get("category", ""))
    product_id = str(raw.get("id", _deterministic_product_id(title, category)))
    raw_vendors = raw.get("vendors", [])[:MAX_VENDORS_PER_PRODUCT]
    prices = [
        float(v.get("price", 0.0))
        for v in raw_vendors
        if SGD_FLOOR <= float(v.get("price", 0.0)) <= SGD_CEILING
    ]
    if not prices:
        prices = [0.0]
    arr = np.array(prices, dtype=np.float64)
    price_min = float(arr.min())
    price_max = float(arr.max())
    price_mean = float(arr.mean())
    shannon_entropy = _compute_shannon_entropy_bits(prices)
    causal_reliability = _compute_causal_reliability_score(
        [{"price_sgd": p} for p in prices]
    )
    value_score, explanation = _compute_value_score(
        price_min, price_mean, shannon_entropy, causal_reliability, len(prices)
    )
    vendor_offers = _normalize_vendor_offers(raw_vendors, prices)
    return RankedProduct(
        product_id=product_id,
        title=title,
        category=category,
        image_url=raw.get("image_url"),
        vendor_offers=vendor_offers,
        price_min_sgd=round(price_min, 2),
        price_max_sgd=round(price_max, 2),
        price_mean_sgd=round(price_mean, 2),
        shannon_entropy_bits=shannon_entropy,
        causal_reliability_score=causal_reliability,
        value_score=value_score,
        value_score_explanation=explanation,
    )


@app.get("/search", response_model=ProductSearchResponse, tags=["core"])
async def search_singapore_products_by_value_score(
    query: str,
    limit: int = 10,
    min_value_score: float = 0.0,
    _api_key: str = Depends(_require_api_key),
):
    """
    Semantic product search over BuyWhere Singapore catalog, ranked by composite value-score
    (Shannon entropy + causal reliability + price rank). Returns structured output ready for
    direct LLM agent consumption without postprocessing.

    Use this when: an agent needs ranked product recommendations for a Singapore market query
    with auditable justification per product.

    Do NOT use this for: real-time stock ticker data, non-SGD markets, or exact product ID lookup
    (use /product/{product_id}/value_breakdown for that).
    """
    query = _validate_query(query)
    if not isinstance(limit, int) or not (1 <= limit <= 50):
        raise HTTPException(status_code=422, detail="limit must be an integer between 1 and 50.")
    if not isinstance(min_value_score, float) or not (0.0 <= min_value_score <= 1.0):
        raise HTTPException(status_code=422, detail="min_value_score must be a float in [0.0, 1.0].")

    t0 = time.perf_counter()
    raw_products = await _fetch_buywhere_search(query, limit)
    ranked = [_build_ranked_product(r) for r in raw_products]
    ranked = [p for p in ranked if p.value_score >= min_value_score]
    ranked.sort(key=lambda p: p.value_score, reverse=True)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return ProductSearchResponse(
        query=query,
        result_count=len(ranked),
        currency="SGD",
        products=ranked,
        computation_ms=elapsed_ms,
    )


@app.get("/product/{product_id}/value_breakdown", response_model=ValueScoreBreakdownResponse, tags=["core"])
async def get_product_value_score_breakdown(
    product_id: str,
    _api_key: str = Depends(_require_api_key),
):
    """
    Returns the full auditable decomposition of the value-score formula for a specific product:
    entropy component, reliability component, price rank component, and the recommended vendor.

    Use this when: an LLM agent needs to cite the mathematical justification for a product
    recommendation (e.g., 'recommended because value_score=0.82 driven by high vendor diversity').

    Do NOT use this for: bulk catalog exploration (use /search for that).
    """
    product_id = _validate_product_id(product_id)
    raw = await _fetch_buywhere_product(product_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found in BuyWhere catalog.")

    ranked = _build_ranked_product(raw)
    prices = [o.price_sgd for o in ranked.vendor_offers]
    arr = np.array(prices, dtype=np.float64) if prices else np.array([0.0])
    price_min = float(arr.min())
    price_mean = float(arr.mean())
    vendor_count = len(prices)
    max_possible_entropy = float(np.log2(max(vendor_count, 2)))
    entropy_normalized = ranked.shannon_entropy_bits / (max_possible_entropy + 1e-9)
    price_efficiency = price_min / (price_mean + 1e-9)
    price_rank_component = round(1.0 - price_efficiency, 6)

    best_vendor = min(ranked.vendor_offers, key=lambda o: o.price_sgd, default=None)
    recommended_vendor = best_vendor.vendor_name if best_vendor else "N/A"
    recommended_price = best_vendor.price_sgd if best_vendor else 0.0

    return ValueScoreBreakdownResponse(
        product_id=product_id,
        title=ranked.title,
        shannon_entropy_bits=ranked.shannon_entropy_bits,
        max_possible_entropy_bits=round(max_possible_entropy, 6),
        entropy_component=round(entropy_normalized, 6),
        reliability_component=ranked.causal_reliability_score,
        price_rank_component=price_rank_component,
        composite_value_score=ranked.value_score,
        formula=(
            "value_score = 0.35 * (H / log2(N)) + 0.40 * causal_reliability + 0.25 * (1 - price_min/price_mean); "
            "H = Shannon entropy in bits over N vendor price distribution; "
            "causal_reliability = 1 - mean(sigmoid(z_score * CV - 1.5)) per vendor"
        ),
        recommended_vendor=recommended_vendor,
        recommended_price_sgd=recommended_price,
    )


@app.get("/product/{product_id}/price_distribution", response_model=PriceDistributionResponse, tags=["core"])
async def get_product_vendor_price_distribution(
    product_id: str,
    _api_key: str = Depends(_require_api_key),
):
    """
    Returns the full price-entropy analysis for a product: spread, coefficient of variation,
    Shannon entropy, and a human-readable interpretation for LLM reasoning chains.

    Use this when: an agent needs to determine whether a product's price is stable across
    vendors (low entropy) or volatile/competitive (high entropy) before making a recommendation.

    Do NOT use this for: value-score ranking across multiple products (use /search for that).
    """
    product_id = _validate_product_id(product_id)
    raw = await _fetch_buywhere_product(product_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found in BuyWhere catalog.")

    ranked = _build_ranked_product(raw)
    prices = [o.price_sgd for o in ranked.vendor_offers]
    if not prices:
        raise HTTPException(status_code=422, detail="No valid vendor offers with SGD prices found for this product.")

    arr = np.array(prices, dtype=np.float64)
    price_spread = round(float(arr.max() - arr.min()), 2)
    mean = float(arr.mean())
    std = float(arr.std())
    cv = round(std / (mean + 1e-9), 6)
    vendor_count = len(prices)
    max_possible_entropy = float(np.log2(max(vendor_count, 2)))
    entropy_normalized = round(ranked.shannon_entropy_bits / (max_possible_entropy + 1e-9), 6)

    if entropy_normalized < 0.25:
        interpretation = (
            f"Price is highly stable across {vendor_count} vendors (H={ranked.shannon_entropy_bits:.3f} bits, "
            f"normalized={entropy_normalized:.3f}). Low vendor competition — less negotiation leverage for buyer."
        )
    elif entropy_normalized < 0.65:
        interpretation = (
            f"Moderate price dispersion across {vendor_count} vendors (H={ranked.shannon_entropy_bits:.3f} bits, "
            f"normalized={entropy_normalized:.3f}). Some vendor differentiation — recommend comparing top 3 offers."
        )
    else:
        interpretation = (
            f"High price volatility across {vendor_count} vendors (H={ranked.shannon_entropy_bits:.3f} bits, "
            f"normalized={entropy_normalized:.3f}). Strong market fragmentation — buyer has significant arbitrage opportunity."
        )

    return PriceDistributionResponse(
        product_id=product_id,
        title=ranked.title,
        vendor_count=vendor_count,
        entropy_bits=ranked.shannon_entropy_bits,
        entropy_normalized=entropy_normalized,
        price_spread_sgd=price_spread,
        coefficient_of_variation=cv,
        interpretation=interpretation,
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def check_api_and_upstream_health():
    """
    Returns liveness status and whether BuyWhere upstream is reachable.

    Use this for: health checks, uptime monitors, deployment readiness probes.

    Do NOT use this for: product queries or value-score computations.
    """
    upstream_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{BUYWHERE_BASE_URL}/api/ping")
            upstream_ok = r.status_code < 500
    except Exception:
        upstream_ok = False

    return HealthResponse(
        status="ok",
        upstream_reachable=upstream_ok,
        api_version="1.0.0",
        timestamp=time.time(),
    )

# --- NEXUS: servidor MCP real montado en el mismo proceso (inyectado por forge_agent) ---
# Reemplaza el wrapper Node/TypeScript separado -- un solo deploy, sin
# segundo servicio, sin salto de red interno. Ver mcp_wrapper_generator.py
# (v2.0) para el razonamiento completo, incluido el gotcha de
# session_manager que explica el patron startup/shutdown de abajo.

from typing import Annotated, Any, Literal
from contextlib import AsyncExitStack as _NexusMcpExitStack

import httpx
from pydantic import Field
from mcp.server.fastmcp import FastMCP as _NexusFastMCP

_nexus_mcp = _NexusFastMCP('nexus-useful-data-source-for-agents-doing-prod', stateless_http=True)


async def _nexus_mcp_call_core(method: str, path: str, params: dict) -> Any:
    """
    Llama al endpoint real del core -- via ASGI in-process (sin red
    real, sin segundo proceso), no un HTTP call externo. `app` ya
    existe en este mismo modulo (es el FastAPI que FORGE genero).
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nexus-internal") as client:
        if method == "GET":
            resp = await client.get(path, params=params)
        else:
            resp = await client.post(path, json=params)
        resp.raise_for_status()
        return resp.json()


@_nexus_mcp.tool(name='nexus_useful_data_source_for_agents_doing_prod_rank_buywhere_products_by_value_score', description="Searches the BuyWhere Singapore catalogue for products matching a natural-language query and returns them ranked by an auditable value-score derived from information-theoretic entropy of cross-vendor price variance. Use this when an agent needs to compare multiple products across vendors and justify a recommendation beyond lowest-price. Do NOT use for exact SKU or model-number lookups where identity matters more than value ranking, and do NOT use when the agent needs only a single vendor's listing.")
async def rank_buywhere_products_by_value_score(query: Annotated[str, Field(..., description="Natural-language product description, e.g. 'noise-cancelling wireless headphones under SGD 300'. Must be in English.", min_length=3, max_length=300)], top_k: Annotated[float, Field(10, description='Maximum number of ranked products to return. Higher values increase latency. Recommended 5-20 for agent consumption.', ge=1, le=50)], max_price_sgd: Annotated[float, Field(None, description='Hard upper bound on product price in SGD. Products strictly above this value are excluded before ranking. Set to null to disable.', ge=0.01, le=99999.99)], category_filter: Annotated[str, Field(None, description="BuyWhere top-level category slug to restrict the search scope, e.g. 'electronics', 'home-appliances', 'health-beauty'. Leave null for cross-category search.", min_length=2, max_length=60)]) -> dict[str, Any]:
    """Rank BuyWhere Products by Value Score"""
    params = {"query": query, "top_k": top_k, "max_price_sgd": max_price_sgd, "category_filter": category_filter}
    return await _nexus_mcp_call_core('POST', '/buywhere/rank-by-value-score', params)

@_nexus_mcp.tool(name='nexus_useful_data_source_for_agents_doing_prod_fetch_buywhere_vendor_price_distribution', description='Given a BuyWhere product identifier, returns the full cross-vendor price distribution in SGD along with the Shannon entropy of that distribution and a causal price-spread coefficient that quantifies whether price differences are driven by vendor margin or by product condition/variant. Use this when an agent already knows the product and needs to audit price fairness or detect outlier vendor pricing. Do NOT use as a discovery tool — it requires a known BuyWhere product_id. Do NOT use when only the cheapest vendor matters and no justification is needed.')
async def fetch_buywhere_vendor_price_distribution(product_id: Annotated[str, Field(..., description='BuyWhere internal product identifier as returned by rank_buywhere_products_by_value_score or resolve_buywhere_product_identity. Format: alphanumeric string.', min_length=4, max_length=64)], include_out_of_stock: Annotated[bool, Field(False, description='If true, vendors with current stock_status=out_of_stock are included in the distribution for historical completeness. If false, only in-stock vendors are used. Default false to reflect actionable prices.')]) -> dict[str, Any]:
    """Fetch Vendor Price Distribution for SKU"""
    params = {"product_id": product_id, "include_out_of_stock": include_out_of_stock}
    return await _nexus_mcp_call_core('POST', '/buywhere/vendor-price-distribution', params)

@_nexus_mcp.tool(name='nexus_useful_data_source_for_agents_doing_prod_resolve_buywhere_product_identity', description='Takes an ambiguous product name or partial model number and returns the canonical BuyWhere product_id with a semantic similarity score and disambiguation metadata. Use this before calling fetch_buywhere_vendor_price_distribution when the agent has a user-supplied name that may match multiple listings. Do NOT use when the agent already holds a validated product_id from a previous tool call — redundant resolution adds latency. Do NOT use for open-ended exploratory queries; use rank_buywhere_products_by_value_score instead.')
async def resolve_buywhere_product_identity(raw_product_name: Annotated[str, Field(..., description="Partial or ambiguous product name, brand + model fragment, or user-typed string. E.g. 'Sony WH-1000' or 'dyson v11 vacuum'.", min_length=2, max_length=200)], candidate_limit: Annotated[float, Field(3, description='Number of candidate product identities to return when the match is ambiguous. The agent should surface these to the user if score < 0.90.', ge=1, le=10)]) -> dict[str, Any]:
    """Resolve Product Identity from Partial Description"""
    params = {"raw_product_name": raw_product_name, "candidate_limit": candidate_limit}
    return await _nexus_mcp_call_core('POST', '/buywhere/resolve-product-identity', params)

@_nexus_mcp.tool(name='nexus_useful_data_source_for_agents_doing_prod_compare_buywhere_products_causal_rank', description='Accepts a list of BuyWhere product_ids and returns a causal ranking that controls for confounders (brand premium, recency bias, vendor count asymmetry) using do-calculus-inspired adjustment on the value-score. Use this when an agent has a shortlist of 2-10 products and needs a defensible recommendation that is not contaminated by spurious correlations. Do NOT use with more than 10 products — use rank_buywhere_products_by_value_score for larger candidate sets. Do NOT use when the comparison must be within a single vendor.')
async def compare_buywhere_products_causal_rank(product_ids: Annotated[list[str], Field(..., description='List of BuyWhere product identifiers to compare. Must contain between 2 and 10 items. Order does not affect output ranking.', min_length=2, max_length=10)], adjustment_covariates: Annotated[list[str], Field(None, description="Confounders to adjust for in the causal model. Accepted values: 'brand_premium', 'vendor_count_bias', 'recency_bias', 'condition_variant'. Defaults to all four if omitted.")]) -> dict[str, Any]:
    """Causal Rank Comparison Across Product List"""
    params = {"product_ids": product_ids, "adjustment_covariates": adjustment_covariates}
    return await _nexus_mcp_call_core('POST', '/buywhere/causal-rank-comparison', params)

@_nexus_mcp.tool(name='nexus_useful_data_source_for_agents_doing_prod_extract_buywhere_deal_anomalies', description='Scans a BuyWhere category or query result for listings whose price deviates significantly from the expected distribution (z-score threshold configurable) and flags them as anomalies — potential flash sales, data errors, or grey-market listings. Returns anomaly type classification and confidence. Use this when an agent is doing price-monitoring or deal-hunting on behalf of a user. Do NOT use as a general ranking tool — anomalies are not always good deals (errors and grey-market items are also flagged). Do NOT call on product_ids already confirmed as anomalies in the same session.')
async def extract_buywhere_deal_anomalies(query_or_category: Annotated[str, Field(..., description='Either a natural-language product query or a BuyWhere category slug. The tool resolves which interpretation applies based on format.', min_length=2, max_length=200)], z_score_threshold: Annotated[float, Field(2.0, description='Minimum absolute z-score (relative to category price distribution) for a listing to be classified as anomalous. Lower values return more candidates with lower confidence; higher values return fewer with higher confidence. Typical useful range 1.5-3.0.', ge=0.5, le=4.0)], anomaly_types: Annotated[list[str], Field(None, description="Filter output to specific anomaly classifications. Accepted values: 'price_too_low', 'price_too_high', 'vendor_outlier', 'possible_data_error'. Defaults to all types if omitted.")]) -> dict[str, Any]:
    """Extract Statistically Anomalous Deals"""
    params = {"query_or_category": query_or_category, "z_score_threshold": z_score_threshold, "anomaly_types": anomaly_types}
    return await _nexus_mcp_call_core('POST', '/buywhere/deal-anomaly-detection', params)


# Crea el sub-app ASGI de streamable HTTP -- DEBE llamarse antes de
# poder acceder a _nexus_mcp.session_manager (se crea de forma
# perezosa, ver docstring del modulo).
# Se monta en "/" (no en "/mcp"): streamable_http_app() YA expone su
# propia ruta interna en "/mcp" -- montarlo de nuevo en "/mcp" duplica
# el path a "/mcp/mcp" y da 404 (bug real encontrado probando esto en
# runtime con un cliente MCP de verdad, no algo teorico).
_nexus_mcp_asgi_app = _nexus_mcp.streamable_http_app()
_nexus_mcp_stack = _NexusMcpExitStack()


@app.on_event("startup")
async def _nexus_mcp_startup():
    await _nexus_mcp_stack.enter_async_context(_nexus_mcp.session_manager.run())


@app.on_event("shutdown")
async def _nexus_mcp_shutdown():
    await _nexus_mcp_stack.aclose()


app.mount("/", _nexus_mcp_asgi_app)

# --- NEXUS: reporte de uso real a Stripe (inyectado por forge_output_saver_v6) ---
@app.middleware("http")
async def _nexus_usage_middleware(request, call_next):
    response = await call_next(request)
    try:
        if response.status_code < 400:
            import os as _nexus_os
            import stripe as _nexus_stripe
            _customer_id = _nexus_os.environ.get("STRIPE_CUSTOMER_ID")
            _event_name = _nexus_os.environ.get("STRIPE_EVENT_NAME")
            _secret_key = _nexus_os.environ.get("STRIPE_SECRET_KEY")
            if _customer_id and _event_name and _secret_key:
                _nexus_stripe.api_key = _secret_key
                _nexus_stripe.billing.MeterEvent.create(
                    event_name=_event_name,
                    payload={
                        "stripe_customer_id": _customer_id,
                        "value": "1",
                    },
                )
    except Exception:
        pass  # nunca romper la response real por un fallo de billing
    return response
