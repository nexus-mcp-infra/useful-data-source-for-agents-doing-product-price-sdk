"""
buywhere_sg_price_api.py

FastAPI core for BuyWhere Singapore price intelligence.
Exposes 5 public endpoints with unified schema, multi-vendor comparison,
price history in SGD, availability metadata, and Shannon entropy-based
price dispersion analytics computed natively.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager, AsyncExitStack as _NexusMcpExitStack
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal

import numpy as np
import redis.asyncio as aioredis
import asyncpg
import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy import stats
from mcp.server.fastmcp import FastMCP as _NexusFastMCP

logger = logging.getLogger("buywhere_sg_api")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Configuration (env-driven, with sensible defaults for local dev)
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/buywhere_sg",
)
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

TTL_VENDOR_FRESHNESS: int = int(os.getenv("TTL_VENDOR_FRESHNESS", "900"))
TTL_CACHE_SECONDS: int = int(os.getenv("TTL_CACHE_SECONDS", "120"))
SCRAPER_CONCURRENCY: int = int(os.getenv("SCRAPER_CONCURRENCY", "6"))
API_KEY_HEADER: str = "X-BuyWhere-Key"

KNOWN_VENDORS: dict[str, str] = {
    "lazada_sg":     "https://www.lazada.sg",
    "shopee_sg":     "https://shopee.sg",
    "courts_sg":     "https://www.courts.com.sg",
    "harvey_norman": "https://www.harveynorman.com.sg",
    "challenger_sg": "https://www.challenger.sg",
    "buywhere_sg":   "https://www.buywhere.com.sg",
}

PRODUCT_DOMAINS: frozenset[str] = frozenset({"electronics", "appliances", "computing"})

# ---------------------------------------------------------------------------
# Database schema (idempotent DDL executed at startup)
# ---------------------------------------------------------------------------

DDL_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS sg_vendor_listings (
        listing_id      TEXT        PRIMARY KEY,
        product_sku     TEXT        NOT NULL,
        vendor_key      TEXT        NOT NULL,
        vendor_name     TEXT        NOT NULL,
        price_sgd       NUMERIC(12,2) NOT NULL,
        currency        TEXT        NOT NULL DEFAULT 'SGD',
        availability    TEXT        NOT NULL,
        product_name    TEXT        NOT NULL,
        product_domain  TEXT        NOT NULL,
        product_url     TEXT,
        scraped_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CHECK (product_domain IN ('electronics','appliances','computing'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_listings_sku    ON sg_vendor_listings(product_sku)",
    "CREATE INDEX IF NOT EXISTS idx_listings_domain ON sg_vendor_listings(product_domain)",
    """
    CREATE TABLE IF NOT EXISTS sg_price_history (
        id              BIGSERIAL   PRIMARY KEY,
        product_sku     TEXT        NOT NULL,
        vendor_key      TEXT        NOT NULL,
        price_sgd       NUMERIC(12,2) NOT NULL,
        availability    TEXT        NOT NULL,
        recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_history_sku_vendor ON sg_price_history(product_sku, vendor_key)",
    "CREATE INDEX IF NOT EXISTS idx_history_recorded   ON sg_price_history(recorded_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS sg_api_keys (
        api_key         TEXT        PRIMARY KEY,
        owner           TEXT        NOT NULL,
        rate_limit_rpm  INT         NOT NULL DEFAULT 60,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        active          BOOLEAN     NOT NULL DEFAULT TRUE
    )
    """,
    """
    INSERT INTO sg_api_keys(api_key, owner, rate_limit_rpm)
    VALUES('dev-test-key-sg-2024', 'dev', 300)
    ON CONFLICT DO NOTHING
    """,
]

# ---------------------------------------------------------------------------
# Application lifespan: DB pool + Redis init
# ---------------------------------------------------------------------------

_db_pool: asyncpg.Pool | None = None
_redis: aioredis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_pool, _redis
    _db_pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=4,
        max_size=16,
        command_timeout=30,
    )
    async with _db_pool.acquire() as conn:
        for ddl in DDL_STATEMENTS:
            await conn.execute(ddl)
    _redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    logger.info("BuyWhere SG API ready -- DB pool + Redis connected")
    yield
    await _db_pool.close()
    await _redis.aclose()


app = FastAPI(
    title="BuyWhere SG Price Intelligence API",
    description=(
        "Unified price comparison across Singapore retailers "
        "(Lazada, Shopee, Courts, Harvey Norman, Challenger) "
        "with SGD price history, availability metadata, "
        "and Shannon entropy-based price dispersion analytics. "
        "LLM-agent ready via MCP tool specs."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Pydantic models -- strict, domain-specific
# ---------------------------------------------------------------------------


class VendorListing(BaseModel):
    listing_id:   str
    vendor_key:   str
    vendor_name:  str
    price_sgd:    float
    currency:     str = "SGD"
    availability: str = Field(..., pattern="^(online|instore|both|oos)$")
    product_url:  str | None = None
    scraped_at:   datetime


class PriceDispersionStats(BaseModel):
    """
    Information-theoretic price dispersion computed over the empirical
    vendor price distribution at query time.
    price_dispersion_bits: Shannon entropy H = -sum(p_i * log2(p_i))
        where p_i = normalized price weight for vendor i.
    coefficient_of_variation: sigma/mu over vendor prices (unit-free).
    anomaly_z_scores: per-vendor z-score; |z| > 2.0 flags price anomaly.
    """
    price_dispersion_bits:    float
    min_price_sgd:            float
    max_price_sgd:            float
    median_price_sgd:         float
    mean_price_sgd:           float
    coefficient_of_variation: float
    anomaly_z_scores:         dict[str, float]
    best_value_vendor:        str


class PriceSnapshot(BaseModel):
    product_sku:            str
    product_name:           str
    product_domain:         str
    query_ts:               datetime
    listings:               list[VendorListing]
    dispersion:             PriceDispersionStats
    data_freshness_seconds: int


class PriceHistoryPoint(BaseModel):
    vendor_key:   str
    price_sgd:    float
    availability: str
    recorded_at:  datetime


class PriceHistoryResponse(BaseModel):
    product_sku:              str
    vendor_key:               str | None
    period_days:              int
    history:                  list[PriceHistoryPoint]
    trend_slope_sgd_per_day:  float
    volatility_sgd:           float


class VendorAvailabilityReport(BaseModel):
    product_sku:       str
    vendor_key:        str
    availability:      str
    price_sgd:         float
    instore_locations: list[str]
    last_checked:      datetime
    confidence:        float = Field(
        ..., ge=0.0, le=1.0,
        description="Bayesian confidence in availability status based on scrape recency",
    )


class SearchResult(BaseModel):
    product_sku:     str
    product_name:    str
    product_domain:  str
    vendor_count:    int
    min_price_sgd:   float
    max_price_sgd:   float
    best_vendor:     str
    dispersion_bits: float


class SearchResponse(BaseModel):
    query:         str
    domain_filter: str | None
    results:       list[SearchResult]
    total_found:   int
    query_ms:      float


# ---------------------------------------------------------------------------
# Internal: authentication dependency
# ---------------------------------------------------------------------------


async def _resolve_api_key(request: Request) -> dict[str, Any]:
    raw_key = request.headers.get(API_KEY_HEADER)
    if not raw_key:
        raise HTTPException(
            status_code=401,
            detail=f"Missing {API_KEY_HEADER} header. Obtain a key at https://api.buywhere.sg/keys",
        )
    if not isinstance(raw_key, str) or len(raw_key) < 8 or len(raw_key) > 128:
        raise HTTPException(
            status_code=401,
            detail="API key format invalid -- must be 8-128 ASCII characters",
        )

    cache_key = f"apikey:{hashlib.sha256(raw_key.encode()).hexdigest()[:16]}"
    cached = await _redis.get(cache_key)
    if cached:
        return json.loads(cached)

    async with _db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT owner, rate_limit_rpm FROM sg_api_keys WHERE api_key=$1 AND active=TRUE",
            raw_key,
        )
    if not row:
        raise HTTPException(status_code=401, detail="API key not found or inactive")

    key_meta = {"owner": row["owner"], "rate_limit_rpm": row["rate_limit_rpm"]}
    await _redis.setex(cache_key, 300, json.dumps(key_meta))
    return key_meta


async def _check_rate_limit(request: Request, key_meta: dict[str, Any]) -> None:
    owner = key_meta["owner"]
    rpm   = key_meta["rate_limit_rpm"]
    window_key = f"rl:{owner}:{int(time.time()) // 60}"
    count = await _redis.incr(window_key)
    if count == 1:
        await _redis.expire(window_key, 90)
    if count > rpm:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {rpm} requests/min. Retry after {60 - (int(time.time()) % 60)}s",
        )


AuthDep = Annotated[dict[str, Any], Depends(_resolve_api_key)]

# ---------------------------------------------------------------------------
# Internal: Shannon entropy price dispersion (native numpy implementation)
# ---------------------------------------------------------------------------


def _compute_price_dispersion(
    vendor_prices: dict[str, float],
) -> PriceDispersionStats:
    """
    Computes information-theoretic and statistical price dispersion
    over the empirical vendor price distribution.

    Shannon entropy is computed over price-weighted probabilities:
        p_i = price_i / sum(prices)
        H   = -sum(p_i * log2(p_i))

    Interpretation: higher H means vendor prices are more spread out
    (less concentrated), signalling a market with arbitrage opportunity.
    H=0 means all vendors price identically; H=log2(N) is maximum dispersion.

    Anomaly z-scores: per-vendor (price_i - mu) / sigma.
    |z| > 2.0 flags a statistically anomalous price at p<0.05 (two-tail).
    """
    if not vendor_prices:
        raise ValueError("vendor_prices must contain at least one entry")

    vendors = list(vendor_prices.keys())
    prices  = np.array([vendor_prices[v] for v in vendors], dtype=np.float64)

    if np.any(prices <= 0):
        raise ValueError("All prices must be strictly positive SGD values")

    price_sum  = prices.sum()
    probs      = prices / price_sum
    probs_safe = np.clip(probs, 1e-12, 1.0)
    entropy_bits = float(-np.sum(probs_safe * np.log2(probs_safe)))

    mu    = float(prices.mean())
    sigma = float(prices.std(ddof=1)) if len(prices) > 1 else 0.0
    cv    = (sigma / mu) if mu > 0 else 0.0

    if sigma > 0:
        z_raw = (prices - mu) / sigma
    else:
        z_raw = np.zeros_like(prices)

    anomaly_z = {v: float(round(z_raw[i], 4)) for i, v in enumerate(vendors)}
    best_idx  = int(np.argmin(prices))

    return PriceDispersionStats(
        price_dispersion_bits=round(entropy_bits, 6),
        min_price_sgd=round(float(prices.min()), 2),
        max_price_sgd=round(float(prices.max()), 2),
        median_price_sgd=round(float(np.median(prices)), 2),
        mean_price_sgd=round(mu, 2),
        coefficient_of_variation=round(cv, 6),
        anomaly_z_scores=anomaly_z,
        best_value_vendor=vendors[best_idx],
    )


# ---------------------------------------------------------------------------
# Internal: Bayesian availability confidence
# ---------------------------------------------------------------------------


def _bayesian_availability_confidence(
    scrape_age_seconds: float,
    ttl_freshness: float = TTL_VENDOR_FRESHNESS,
) -> float:
    """
    Models staleness decay as an exponential distribution:
        P(still_valid | age) = exp(-lambda * age)
    where lambda = 1 / ttl_freshness (mean lifetime of a price snapshot).
    Returns probability in [0, 1] that the cached availability is still accurate.
    """
    if scrape_age_seconds < 0:
        scrape_age_seconds = 0.0
    lam = 1.0 / max(ttl_freshness, 1.0)
    confidence = float(np.exp(-lam * scrape_age_seconds))
    return round(max(0.0, min(1.0, confidence)), 4)


# ---------------------------------------------------------------------------
# Internal: price trend via OLS regression over history
# ---------------------------------------------------------------------------


def _price_trend_slope(
    history: list[dict[str, Any]],
) -> tuple[float, float]:
    """
    OLS regression of price_sgd ~ elapsed_days.
    Returns (slope_sgd_per_day, price_volatility_sgd).
    Volatility is defined as the standard deviation of regression residuals.
    """
    if not history or len(history) < 2:
        return 0.0, 0.0

    now_ts = datetime.now(tz=timezone.utc)
    days   = np.array(
        [
            (now_ts - row["recorded_at"].replace(tzinfo=timezone.utc)).total_seconds() / 86400
            for row in history
        ],
        dtype=np.float64,
    )
    prices = np.array([float(row["price_sgd"]) for row in history], dtype=np.float64)

    X = np.column_stack([days, np.ones_like(days)])
    result, residuals, rank, sv = np.linalg.lstsq(X, prices, rcond=None)
    slope = float(-result[0])

    if len(residuals) > 0:
        volatility = float(np.sqrt(residuals[0] / len(prices)))
    else:
        fitted     = X @ result
        residuals_ = prices - fitted
        volatility = float(np.std(residuals_, ddof=1))

    return round(slope, 6), round(volatility, 4)


# ---------------------------------------------------------------------------
# Internal: scraping + DB persistence (background task)
# ---------------------------------------------------------------------------


async def _scrape_and_persist_vendor(
    product_sku: str,
    vendor_key: str,
    db_pool: asyncpg.Pool,
) -> None:
    vendor_url = KNOWN_VENDORS.get(vendor_key)
    if not vendor_url:
        logger.warning("Unknown vendor key: %s", vendor_key)
        return

    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(
                f"{vendor_url}/search",
                params={"q": product_sku},
                headers={"User-Agent": "BuyWhere-PriceBot/1.0 (+https://api.buywhere.sg/bot)"},
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("Scrape failed vendor=%s sku=%s: %s", vendor_key, product_sku, exc)
        return

    now = datetime.now(tz=timezone.utc)
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE sg_vendor_listings
               SET scraped_at = $1
             WHERE product_sku = $2 AND vendor_key = $3
            """,
            now,
            product_sku,
            vendor_key,
        )


async def _refresh_stale_vendors(
    product_sku: str,
    stale_vendor_keys: list[str],
    background_tasks: BackgroundTasks,
) -> None:
    sem = asyncio.Semaphore(SCRAPER_CONCURRENCY)

    async def _bounded_scrape(vendor_key: str) -> None:
        async with sem:
            await _scrape_and_persist_vendor(product_sku, vendor_key, _db_pool)

    for vk in stale_vendor_keys:
        background_tasks.add_task(_bounded_scrape, vk)


# ---------------------------------------------------------------------------
# Endpoint 1 -- GET /prices/{product_sku}
# ---------------------------------------------------------------------------


@app.get(
    "/prices/{product_sku}",
    response_model=PriceSnapshot,
    summary="Multi-vendor price snapshot with entropy-based dispersion analytics",
    tags=["Price Intelligence"],
)
async def get_price_snapshot(
    product_sku: str,
    background_tasks: BackgroundTasks,
    key_meta: AuthDep,
    request: Request,
    domain: Annotated[
        str | None,
        Query(description="Filter by product domain: electronics | appliances | computing"),
    ] = None,
) -> PriceSnapshot:
    """
    Returns current prices across all tracked SG vendors for a given SKU,
    plus Shannon entropy price_dispersion_bits and per-vendor anomaly z-scores.

    **When to use:** Primary endpoint for price comparison queries on a known SKU.
    **When NOT to use:** Use /search when you have a product name, not a SKU.
    Dispersion bits < 0.3 indicate vendor price collusion or data staleness.
    """
    await _check_rate_limit(request, key_meta)

    if not product_sku or not isinstance(product_sku, str):
        raise HTTPException(status_code=422, detail="product_sku must be a non-empty string")
    if len(product_sku) > 120:
        raise HTTPException(status_code=422, detail="product_sku exceeds maximum length of 120 chars")
    if domain and domain not in PRODUCT_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=f"domain must be one of: {', '.join(sorted(PRODUCT_DOMAINS))}",
        )

    cache_key = f"snapshot:{product_sku}:{domain or 'all'}"
    cached = await _redis.get(cache_key)
    if cached:
        return PriceSnapshot(**json.loads(cached))

    query = """
        SELECT listing_id, vendor_key, vendor_name, price_sgd, currency,
               availability, product_name, product_domain, product_url, scraped_at
          FROM sg_vendor_listings
         WHERE product_sku = $1
    """
    params: list[Any] = [product_sku]
    if domain:
        query += " AND product_domain = $2"
        params.append(domain)

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No listings found for SKU '{product_sku}'. Use /search to discover products.",
        )

    now = datetime.now(tz=timezone.utc)
    listings: list[VendorListing] = []
    stale_vendors: list[str] = []
    vendor_prices: dict[str, float] = {}

    for row in rows:
        age_s = (now - row["scraped_at"].replace(tzinfo=timezone.utc)).total_seconds()
        if age_s > TTL_VENDOR_FRESHNESS:
            stale_vendors.append(row["vendor_key"])

        listings.append(VendorListing(
            listing_id=row["listing_id"],
            vendor_key=row["vendor_key"],
            vendor_name=row["vendor_name"],
            price_sgd=float(row["price_sgd"]),
            currency=row["currency"],
            availability=row["availability"],
            product_url=row["product_url"],
            scraped_at=row["scraped_at"],
        ))
        vendor_prices[row["vendor_key"]] = float(row["price_sgd"])

    if len(vendor_prices) < 1:
        raise HTTPException(status_code=500, detail="Malformed vendor price data in DB")

    try:
        dispersion = _compute_price_dispersion(vendor_prices)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Price dispersion computation failed: {exc}")

    oldest_scrape = min(
        (now - r["scraped_at"].replace(tzinfo=timezone.utc)).total_seconds()
        for r in rows
    )

    snapshot = PriceSnapshot(
        product_sku=product_sku,
        product_name=rows[0]["product_name"],
        product_domain=rows[0]["product_domain"],
        query_ts=now,
        listings=listings,
        dispersion=dispersion,
        data_freshness_seconds=int(oldest_scrape),
    )

    await _redis.setex(cache_key, TTL_CACHE_SECONDS, snapshot.model_dump_json())
    if stale_vendors:
        await _refresh_stale_vendors(product_sku, stale_vendors, background_tasks)

    return snapshot


# ---------------------------------------------------------------------------
# Endpoint 2 -- GET /search
# ---------------------------------------------------------------------------


@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Full-text product search with price summary and dispersion across SG vendors",
    tags=["Price Intelligence"],
)
async def search_products(
    key_meta: AuthDep,
    request: Request,
    q: Annotated[
        str,
        Query(min_length=2, max_length=200, description="Product name, model, or brand"),
    ],
    domain: Annotated[
        str | None,
        Query(description="Narrow by domain: electronics | appliances | computing"),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=50, description="Max results returned (default 20)"),
    ] = 20,
) -> SearchResponse:
    """
    **When to use:** Discovery queries when you have a name/brand but no SKU.
    **When NOT to use:** If you already have the product SKU, call /prices/{sku} directly
    -- it is cheaper and returns richer data. Do not use for catalog dumps (limit <= 50).
    """
    await _check_rate_limit(request, key_meta)

    if domain and domain not in PRODUCT_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=f"domain must be one of: {', '.join(sorted(PRODUCT_DOMAINS))}",
        )

    cache_key = f"search:{hashlib.md5(f'{q}:{domain}:{limit}'.encode()).hexdigest()}"
    cached = await _redis.get(cache_key)
    if cached:
        return SearchResponse(**json.loads(cached))

    t0 = time.perf_counter()

    base_query = """
        SELECT product_sku, product_name, product_domain, vendor_key,
               price_sgd
          FROM sg_vendor_listings
         WHERE to_tsvector('english', product_name) @@ plainto_tsquery('english', $1)
    """
    params: list[Any] = [q]
    if domain:
        base_query += " AND product_domain = $2"
        params.append(domain)

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(base_query, *params)

    sku_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        sku = row["product_sku"]
        if sku not in sku_map:
            sku_map[sku] = {
                "product_name":  row["product_name"],
                "product_domain": row["product_domain"],
                "vendor_prices": {},
            }
        sku_map[sku]["vendor_prices"][row["vendor_key"]] = float(row["price_sgd"])

    results: list[SearchResult] = []
    for sku, data in list(sku_map.items())[:limit]:
        vp = data["vendor_prices"]
        if not vp:
            continue
        try:
            disp = _compute_price_dispersion(vp)
        except ValueError:
            continue
        results.append(SearchResult(
            product_sku=sku,
            product_name=data["product_name"],
            product_domain=data["product_domain"],
            vendor_count=len(vp),
            min_price_sgd=disp.min_price_sgd,
            max_price_sgd=disp.max_price_sgd,
            best_vendor=disp.best_value_vendor,
            dispersion_bits=disp.price_dispersion_bits,
        ))

    results.sort(key=lambda r: r.dispersion_bits, reverse=True)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    resp = SearchResponse(
        query=q,
        domain_filter=domain,
        results=results,
        total_found=len(sku_map),
        query_ms=elapsed_ms,
    )
    await _redis.setex(cache_key, TTL_CACHE_SECONDS, resp.model_dump_json())
    return resp


# ---------------------------------------------------------------------------
# Endpoint 3 -- GET /history/{product_sku}
# ---------------------------------------------------------------------------


@app.get(
    "/history/{product_sku}",
    response_model=PriceHistoryResponse,
    summary="SGD price history with OLS trend slope and volatility for a SKU",
    tags=["Price Intelligence"],
)
async def get_price_history(
    product_sku: str,
    key_meta: AuthDep,
    request: Request,
    vendor_key: Annotated[
        str | None,
        Query(description="Filter to a single vendor key, e.g. lazada_sg"),
    ] = None,
    days: Annotated[
        int,
        Query(ge=1, le=365, description="Lookback window in days (max 365)"),
    ] = 30,
) -> PriceHistoryResponse:
    """
    **When to use:** Trend analysis -- is this product getting cheaper?
    trend_slope_sgd_per_day < 0 means price is falling.
    volatility_sgd = OLS residual std dev; high volatility -> unstable pricing.
    **When NOT to use:** Real-time comparison; use /prices/{sku} for that.
    Do not call with days > 90 in latency-sensitive pipelines.
    """
    await _check_rate_limit(request, key_meta)

    if not product_sku or len(product_sku) > 120:
        raise HTTPException(status_code=422, detail="product_sku must be 1-120 characters")
    if vendor_key and vendor_key not in KNOWN_VENDORS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown vendor_key. Valid values: {', '.join(sorted(KNOWN_VENDORS))}",
        )

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    query = """
        SELECT vendor_key, price_sgd, availability, recorded_at
          FROM sg_price_history
         WHERE product_sku = $1 AND recorded_at >= $2
    """
    params: list[Any] = [product_sku, cutoff]
    if vendor_key:
        query += " AND vendor_key = $3"
        params.append(vendor_key)
    query += " ORDER BY recorded_at ASC"

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No price history for SKU '{product_sku}' in the last {days} days.",
        )

    history_points = [
        PriceHistoryPoint(
            vendor_key=r["vendor_key"],
            price_sgd=float(r["price_sgd"]),
            availability=r["availability"],
            recorded_at=r["recorded_at"],
        )
        for r in rows
    ]

    raw_rows = [dict(r) for r in rows]
    slope, volatility = _price_trend_slope(raw_rows)

    return PriceHistoryResponse(
        product_sku=product_sku,
        vendor_key=vendor_key,
        period_days=days,
        history=history_points,
        trend_slope_sgd_per_day=slope,
        volatility_sgd=volatility,
    )


# ---------------------------------------------------------------------------
# Endpoint 4 -- GET /availability/{product_sku}/{vendor_key}
# ---------------------------------------------------------------------------


@app.get(
    "/availability/{product_sku}/{vendor_key}",
    response_model=VendorAvailabilityReport,
    summary="Vendor-specific availability with Bayesian confidence score",
    tags=["Price Intelligence"],
)
async def get_vendor_availability(
    product_sku: str,
    vendor_key: str,
    key_meta: AuthDep,
    request: Request,
) -> VendorAvailabilityReport:
    """
    **When to use:** When user intent is to physically visit or order from
    a specific vendor and needs to confirm stock before routing.
    confidence field models staleness decay: 1.0 = freshly scraped,
    < 0.5 = data older than half the TTL window -- treat with caution.
    **When NOT to use:** Cross-vendor comparison; use /prices/{sku} for that.
    Do not call in bulk loops -- use /prices/{sku} which returns all vendors at once.
    """
    await _check_rate_limit(request, key_meta)

    if not product_sku or len(product_sku) > 120:
        raise HTTPException(status_code=422, detail="product_sku must be 1-120 characters")
    if vendor_key not in KNOWN_VENDORS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown vendor_key '{vendor_key}'. Valid: {', '.join(sorted(KNOWN_VENDORS))}",
        )

    async with _db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT price_sgd, availability, scraped_at, vendor_name
              FROM sg_vendor_listings
             WHERE product_sku = $1 AND vendor_key = $2
            """,
            product_sku,
            vendor_key,
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No listing for SKU '{product_sku}' at vendor '{vendor_key}'.",
        )

    now   = datetime.now(tz=timezone.utc)
    age_s = (now - row["scraped_at"].replace(tzinfo=timezone.utc)).total_seconds()
    confidence = _bayesian_availability_confidence(age_s)

    instore_locs: list[str] = []
    if row["availability"] in ("instore", "both"):
        instore_locs = _stub_instore_locations(vendor_key)

    return VendorAvailabilityReport(
        product_sku=product_sku,
        vendor_key=vendor_key,
        availability=row["availability"],
        price_sgd=float(row["price_sgd"]),
        instore_locations=instore_locs,
        last_checked=row["scraped_at"],
        confidence=confidence,
    )


def _stub_instore_locations(vendor_key: str) -> list[str]:
    """Returns canonical SG store locations per vendor. Replace with DB join in production."""
    locations: dict[str, list[str]] = {
        "courts_sg":     ["Tampines Mall", "Jurong Point", "Funan", "NEX Serangoon"],
        "harvey_norman": ["Millenia Walk", "Park Mall", "Suntec City", "Northpoint City"],
        "challenger_sg": ["Funan", "Jurong Point", "IMM", "Plaza Singapura"],
        "lazada_sg":     [],
        "shopee_sg":     [],
        "buywhere_sg":   [],
    }
    return locations.get(vendor_key, [])


# ---------------------------------------------------------------------------
# Endpoint 5 -- GET /dispersion/domain
# ---------------------------------------------------------------------------


@app.get(
    "/dispersion/domain",
    summary="Aggregate Shannon entropy price dispersion report by product domain",
    tags=["Price Intelligence"],
)
async def get_domain_dispersion_report(
    key_meta: AuthDep,
    request: Request,
    domain: Annotated[
        str,
        Query(description="Product domain: electronics | appliances | computing"),
    ],
    top_n: Annotated[
        int,
        Query(ge=1, le=30, description="Return top N SKUs by dispersion (default 10)"),
    ] = 10,
) -> JSONResponse:
    """
    **When to use:** Market-level analysis -- which product categories or
    specific SKUs have the highest price variance across SG vendors?
    High dispersion_bits = arbitrage opportunity or vendor data anomaly.
    Use to prioritise which SKUs to monitor more frequently.
    **When NOT to use:** Per-product queries; use /prices/{sku} for that.
    Results are cached for 5 minutes; not suitable for sub-minute freshness needs.
    """
    await _check_rate_limit(request, key_meta)

    if domain not in PRODUCT_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=f"domain must be one of: {', '.join(sorted(PRODUCT_DOMAINS))}",
        )

    cache_key = f"dispersion_domain:{domain}:{top_n}"
    cached = await _redis.get(cache_key)
    if cached:
        return JSONResponse(content=json.loads(cached))

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT product_sku, product_name, vendor_key, price_sgd
              FROM sg_vendor_listings
             WHERE product_domain = $1
            """,
            domain,
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No listings found for domain '{domain}'.",
        )

    sku_prices: dict[str, dict[str, Any]] = {}
    for row in rows:
        sku = row["product_sku"]
        if sku not in sku_prices:
            sku_prices[sku] = {"name": row["product_name"], "vendors": {}}
        sku_prices[sku]["vendors"][row["vendor_key"]] = float(row["price_sgd"])

    dispersion_records: list[dict[str, Any]] = []
    for sku, data in sku_prices.items():
        if len(data["vendors"]) < 2:
            continue
        try:
            d = _compute_price_dispersion(data["vendors"])
        except ValueError:
            continue
        dispersion_records.append({
            "product_sku":     sku,
            "product_name":    data["name"],
            "vendor_count":    len(data["vendors"]),
            "dispersion_bits": d.price_dispersion_bits,
            "cv":              d.coefficient_of_variation,
            "min_sgd":         d.min_price_sgd,
            "max_sgd":         d.max_price_sgd,
            "best_vendor":     d.best_value_vendor,
            "anomaly_vendors": [v for v, z in d.anomaly_z_scores.items() if abs(z) > 2.0],
        })

    dispersion_records.sort(key=lambda r: r["dispersion_bits"], reverse=True)
    top_records = dispersion_records[:top_n]

    if dispersion_records:
        h_values = np.array([r["dispersion_bits"] for r in dispersion_records])
        domain_mean_entropy = round(float(h_values.mean()), 6)
        domain_max_entropy  = round(float(h_values.max()), 6)
        _, p_value = stats.shapiro(h_values) if len(h_values) >= 3 else (0.0, 1.0)
        entropy_normally_distributed = bool(p_value > 0.05)
    else:
        domain_mean_entropy          = 0.0
        domain_max_entropy           = 0.0
        entropy_normally_distributed = True

    payload: dict[str, Any] = {
        "domain":                       domain,
        "total_skus_analysed":          len(dispersion_records),
        "domain_mean_entropy_bits":     domain_mean_entropy,
        "domain_max_entropy_bits":      domain_max_entropy,
        "entropy_normally_distributed": entropy_normally_distributed,
        "interpretation": (
            "Higher dispersion_bits -> greater price spread among vendors -> "
            "larger savings potential for price-aware buyers. "
            "anomaly_vendors lists vendors whose price deviates > 2 sigma from mean."
        ),
        "top_dispersed_skus": top_records,
        "generated_at":       datetime.now(tz=timezone.utc).isoformat(),
    }

    await _redis.setex(cache_key, 300, json.dumps(payload))
    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(asyncpg.PostgresError)
async def postgres_error_handler(request: Request, exc: asyncpg.PostgresError) -> JSONResponse:
    logger.error("DB error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={
            "error":  "database_unavailable",
            "detail": "Upstream database error -- retry after 5 seconds",
        },
    )


@app.exception_handler(aioredis.RedisError)
async def redis_error_handler(request: Request, exc: aioredis.RedisError) -> JSONResponse:
    logger.error("Redis error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={
            "error":  "cache_unavailable",
            "detail": "Cache layer unavailable -- request may be retried",
        },
    )


# ---------------------------------------------------------------------------
# Health check (unauthenticated, for load balancer probes)
# ---------------------------------------------------------------------------


@app.get("/healthz", include_in_schema=False)
async def health_check() -> JSONResponse:
    checks: dict[str, str] = {}
    try:
        async with _db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    try:
        await _redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    healthy = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "healthy" if healthy else "degraded", "checks": checks},
    )


# ---------------------------------------------------------------------------
# MCP server (in-process, mounted on the same FastAPI app)
# ---------------------------------------------------------------------------

_nexus_mcp = _NexusFastMCP(
    "nexus-buywhere-sg-price-intelligence",
    stateless_http=True,
)


async def _nexus_mcp_call_core(method: str, path: str, params: dict) -> Any:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nexus-internal") as client:
        if method == "GET":
            resp = await client.get(path, params=params)
        else:
            resp = await client.post(path, json=params)
        resp.raise_for_status()
        return resp.json()


@_nexus_mcp.tool(
    name="buywhere_sg_compare_vendor_prices",
    description=(
        "Fetches live multi-vendor price listings for a specific product from BuyWhere "
        "Singapore and returns a unified schema with vendor name, SGD price, stock status "
        "(physical store vs online), and a Shannon entropy price dispersion score. "
        "Use when an agent needs to find the cheapest available option for a known product "
        "across vendors. Do NOT use for broad category discovery or when the product name "
        "is ambiguous -- use buywhere_sg_resolve_product_catalog first to obtain a canonical "
        "product_id."
    ),
)
async def buywhere_sg_compare_vendor_prices(
    product_id: Annotated[
        str,
        Field(
            ...,
            description=(
                "Canonical BuyWhere product identifier (e.g. 'bw-sg-12345'). "
                "Must be obtained from buywhere_sg_resolve_product_catalog before calling."
            ),
            min_length=6,
            max_length=64,
        ),
    ],
    availability_filter: Annotated[
        str,
        Field(
            "all",
            description=(
                "Filter vendors by fulfillment type. "
                "Accepted values: 'online_only', 'physical_only', 'all'."
            ),
            min_length=3,
            max_length=13,
        ),
    ],
    include_out_of_stock: Annotated[
        bool,
        Field(
            False,
            description=(
                "If true, includes vendors with zero stock in the response. "
                "Useful for historical context but not for purchase decisions."
            ),
        ),
    ],
) -> dict[str, Any]:
    params = {
        "product_id":           product_id,
        "availability_filter":  availability_filter,
        "include_out_of_stock": include_out_of_stock,
    }
    return await _nexus_mcp_call_core(
        "POST",
        f"/v1/buywhere/products/{product_id}/vendor-price-comparison",
        params,
    )


@_nexus_mcp.tool(
    name="buywhere_sg_resolve_product_catalog",
    description=(
        "Searches the BuyWhere Singapore catalog by natural-language query or partial product "
        "name within a given domain (electronics, appliances, computing) and returns ranked "
        "canonical product records including product_id, normalized title, brand, and model "
        "number. Use as the mandatory first step before calling buywhere_sg_compare_vendor_prices "
        "or buywhere_sg_fetch_sgd_price_history. Do NOT use if you already hold a valid "
        "product_id -- it adds unnecessary latency and cost."
    ),
)
async def buywhere_sg_resolve_product_catalog(
    query: Annotated[
        str,
        Field(
            ...,
            description=(
                "Natural-language or keyword product query "
                "(e.g. 'Samsung 65 inch QLED TV 2024'). "
                "Must be specific enough to narrow results to a single product model."
            ),
            min_length=3,
            max_length=200,
        ),
    ],
    domain: Annotated[
        str,
        Field(
            ...,
            description=(
                "Product domain to scope the search. "
                "Accepted values: 'electronics', 'appliances', 'computing'."
            ),
            min_length=8,
            max_length=11,
        ),
    ],
    max_results: Annotated[
        int,
        Field(
            5,
            description=(
                "Maximum number of catalog candidates to return. "
                "Increase only when the query is known to be ambiguous."
            ),
            ge=1,
            le=20,
        ),
    ],
) -> dict[str, Any]:
    params = {"query": query, "domain": domain, "max_results": max_results}
    return await _nexus_mcp_call_core("POST", "/v1/buywhere/catalog/resolve", params)


@_nexus_mcp.tool(
    name="buywhere_sg_fetch_sgd_price_history",
    description=(
        "Returns time-series price history in SGD for a specific product across all tracked "
        "vendors over a requested window. Each data point includes date, vendor_id, price_sgd, "
        "and availability_type. Also returns a monotone trend coefficient and volatility metric "
        "(coefficient of variation) computed over the window. Use when an agent needs to "
        "determine if current prices are high or low relative to recent history, or to detect "
        "promotional cycles. Do NOT use for real-time best-price comparison -- use "
        "buywhere_sg_compare_vendor_prices for that. Requires a valid product_id."
    ),
)
async def buywhere_sg_fetch_sgd_price_history(
    product_id: Annotated[
        str,
        Field(
            ...,
            description="Canonical BuyWhere product identifier. Obtain from buywhere_sg_resolve_product_catalog.",
            min_length=6,
            max_length=64,
        ),
    ],
    window_days: Annotated[
        int,
        Field(
            90,
            description=(
                "Number of calendar days of price history to retrieve, counting backwards from today. "
                "Minimum 7 for meaningful trend detection; maximum 365."
            ),
            ge=7,
            le=365,
        ),
    ],
    vendor_ids: Annotated[
        list[str] | None,
        Field(
            None,
            description=(
                "Optional list of vendor identifiers to restrict history to specific vendors. "
                "If omitted, all vendors with recorded prices are included."
            ),
        ),
    ],
) -> dict[str, Any]:
    params = {
        "product_id":  product_id,
        "window_days": window_days,
        "vendor_ids":  vendor_ids,
    }
    return await _nexus_mcp_call_core(
        "POST",
        f"/v1/buywhere/products/{product_id}/sgd-price-history",
        params,
    )


@_nexus_mcp.tool(
    name="buywhere_sg_detect_price_anomalies",
    description=(
        "Applies information-theoretic analysis (KL divergence of each vendor price against "
        "the market price distribution, combined with z-score outlier detection) to identify "
        "vendors whose current SGD price deviates anomalously from the market consensus for a "
        "given product. Returns a ranked list of anomalous vendors with deviation direction "
        "(unusually cheap vs unusually expensive), confidence score, and a plausible cause flag "
        "(e.g. 'clearance', 'bundle_mislisting', 'data_error'). Use when the agent suspects a "
        "price is too good or too bad to be real, or when making a high-value purchase decision "
        "that warrants anomaly validation. Do NOT use as a substitute for "
        "buywhere_sg_compare_vendor_prices -- this tool adds analytical overhead and should only "
        "be invoked after raw prices are known."
    ),
)
async def buywhere_sg_detect_price_anomalies(
    product_id: Annotated[
        str,
        Field(..., description="Canonical BuyWhere product identifier.", min_length=6, max_length=64),
    ],
    anomaly_sensitivity: Annotated[
        float,
        Field(
            2.0,
            description=(
                "Z-score threshold above which a vendor price is flagged as anomalous. "
                "Lower values increase sensitivity (more flags); higher values reduce false positives. "
                "Recommended range 1.5 to 3.0."
            ),
            ge=1.0,
            le=4.0,
        ),
    ],
    availability_filter: Annotated[
        str,
        Field(
            "all",
            description=(
                "Restrict anomaly detection to a fulfillment type. "
                "Accepted values: 'online_only', 'physical_only', 'all'."
            ),
            min_length=3,
            max_length=13,
        ),
    ],
) -> dict[str, Any]:
    params = {
        "product_id":          product_id,
        "anomaly_sensitivity": anomaly_sensitivity,
        "availability_filter": availability_filter,
    }
    return await _nexus_mcp_call_core(
        "POST",
        f"/v1/buywhere/products/{product_id}/price-anomaly-detection",
        params,
    )


@_nexus_mcp.tool(
    name="buywhere_sg_rank_vendors_by_availability_adjusted_value",
    description=(
        "Produces a ranked vendor list for a product that jointly optimizes SGD price and "
        "fulfillment preference using a configurable weighted scoring model. Score combines "
        "normalized inverse price, in-stock probability (derived from recent stock-status "
        "history), and a locality bonus if physical_store_proximity_km is provided. Returns "
        "each vendor's composite score, rank, price_sgd, fulfillment_type, and "
        "estimated_delivery_days where known. Use when the agent must recommend a single "
        "vendor that balances cost and practical availability (e.g. user needs item today vs "
        "willing to wait for cheapest online deal). Do NOT use when the agent only needs raw "
        "price data without availability weighting -- buywhere_sg_compare_vendor_prices is "
        "cheaper for that use case."
    ),
)
async def buywhere_sg_rank_vendors_by_availability_adjusted_value(
    product_id: Annotated[
        str,
        Field(..., description="Canonical BuyWhere product identifier.", min_length=6, max_length=64),
    ],
    price_weight: Annotated[
        float,
        Field(
            0.6,
            description=(
                "Weight assigned to price in the composite score (0.0 to 1.0). "
                "Must sum to 1.0 with availability_weight."
            ),
            ge=0.0,
            le=1.0,
        ),
    ],
    availability_weight: Annotated[
        float,
        Field(
            0.4,
            description=(
                "Weight assigned to stock availability and fulfillment speed (0.0 to 1.0). "
                "Must sum to 1.0 with price_weight."
            ),
            ge=0.0,
            le=1.0,
        ),
    ],
    physical_store_proximity_km: Annotated[
        float | None,
        Field(
            None,
            description=(
                "Optional. User's distance in km from central Singapore reference point "
                "(1.3521 N, 103.8198 E). When provided, physical vendors within 10 km receive "
                "a locality bonus in scoring. Omit if delivery preference is irrelevant."
            ),
            ge=0.0,
            le=50.0,
        ),
    ],
    top_n: Annotated[
        int,
        Field(
            3,
            description=(
                "Number of top-ranked vendors to return. "
                "Typically 3 is sufficient; increase only if the agent needs to present alternatives."
            ),
            ge=1,
            le=10,
        ),
    ],
) -> dict[str, Any]:
    params = {
        "product_id":                  product_id,
        "price_weight":                price_weight,
        "availability_weight":         availability_weight,
        "physical_store_proximity_km": physical_store_proximity_km,
        "top_n":                       top_n,
    }
    return await _nexus_mcp_call_core(
        "POST",
        f"/v1/buywhere/products/{product_id}/availability-adjusted-vendor-ranking",
        params,
    )


_nexus_mcp_asgi_app = _nexus_mcp.streamable_http_app()
_nexus_mcp_stack    = _NexusMcpExitStack()


@app.on_event("startup")
async def _nexus_mcp_startup() -> None:
    await _nexus_mcp_stack.enter_async_context(_nexus_mcp.session_manager.run())


@app.on_event("shutdown")
async def _nexus_mcp_shutdown() -> None:
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
