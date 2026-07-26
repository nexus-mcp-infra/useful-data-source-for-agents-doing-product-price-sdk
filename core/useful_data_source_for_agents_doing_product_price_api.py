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
    contact={"email": "dasaanrod@gmail.com"},
)

# --- NEXUS: x402 (pago por llamada en USDC, Base Sepolia testnet) ---
from x402.http import FacilitatorConfig, HTTPFacilitatorClient, PaymentOption
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http.types import RouteConfig
from x402.mechanisms.evm.exact import ExactEvmServerScheme
from x402.schemas import Network
from x402.server import x402ResourceServer

_NEXUS_X402_EVM_ADDRESS = "0x70e9f8057bb50e31b6ee06958bcbbe7de9daa98f"
_NEXUS_X402_NETWORK: Network = "eip155:84532"  # Base Sepolia (testnet) -- cambiar a eip155:8453 + facilitator mainnet para produccion
_NEXUS_X402_PRICE = "$0.01"

_nexus_x402_facilitator = HTTPFacilitatorClient(
    FacilitatorConfig(url="https://x402.org/facilitator")
)
_nexus_x402_server = x402ResourceServer(_nexus_x402_facilitator)
_nexus_x402_server.register(_NEXUS_X402_NETWORK, ExactEvmServerScheme())

# --- NEXUS PATCH x402_dynamic_route_matching_buywhere ---
# x402HTTPServerBase._parse_route_pattern() (libreria x402 de terceros)
# no reconoce la sintaxis "{param}" de FastAPI/Starlette para
# segmentos dinamicos -- re.escape() la trata como texto literal, asi
# que "{product_id}" en la clave NUNCA matchea un ID real (solo
# matchearia el string literal "{product_id}" sin resolver). Se usa
# la sintaxis ":param" que x402 SI convierte a "[^/]+" internamente.
# La ruta real de FastAPI mas abajo sigue usando "{product_id}" -- esto
# solo cambia la clave de config de x402, no toca routing de Starlette.
_NEXUS_X402_ROUTES: dict[str, RouteConfig] = {
    "GET /search": RouteConfig(
        accepts=[PaymentOption(scheme="exact", pay_to=_NEXUS_X402_EVM_ADDRESS, price=_NEXUS_X402_PRICE, network=_NEXUS_X402_NETWORK)],
        mime_type="application/json",
        description="Busqueda semantica de productos BuyWhere Singapore por value-score",
    ),
    "GET /product/:product_id/value_breakdown": RouteConfig(
        accepts=[PaymentOption(scheme="exact", pay_to=_NEXUS_X402_EVM_ADDRESS, price=_NEXUS_X402_PRICE, network=_NEXUS_X402_NETWORK)],
        mime_type="application/json",
        description="Desglose del value-score de un producto especifico",
    ),
    "GET /product/:product_id/price_distribution": RouteConfig(
        accepts=[PaymentOption(scheme="exact", pay_to=_NEXUS_X402_EVM_ADDRESS, price=_NEXUS_X402_PRICE, network=_NEXUS_X402_NETWORK)],
        mime_type="application/json",
        description="Distribucion de precios entre vendedores para un producto",
    ),
}

app.add_middleware(PaymentMiddlewareASGI, routes=_NEXUS_X402_ROUTES, server=_nexus_x402_server)

# --- NEXUS: x402scan discovery -- x-payment-info en openapi.json ---
# x402scan (Merit-Systems) no tiene un .well-known/x402 ratificado --
# confirmado 2026-07-26 cruzando x402-foundation/x402 (sin mencion),
# x402scan-skills (deprecado, sin schema) y la spec vigente en
# agentcash.dev/discovery (mismo equipo detras de x402scan): el
# mecanismo real es leer /openapi.json buscando la extension
# "x-payment-info" por operacion. Sin esto, x402scan reporta "no
# discovery document found" pese a que /openapi.json ya se sirve.
# Rutas declaradas explicitas con sintaxis FastAPI ("{product_id}"),
# NO la sintaxis ":product_id" de _NEXUS_X402_ROUTES (son formatos
# distintos -- ver docstring del patch). Precio sale de
# _NEXUS_X402_PRICE ya definido arriba -- nada nuevo se inventa aca.
_NEXUS_X402_OPENAPI_OPERATIONS = [
    ("get", "/search"),
    ("get", "/product/{product_id}/value_breakdown"),
    ("get", "/product/{product_id}/price_distribution"),
]


def _nexus_x402_openapi_with_payment_info():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi as _nexus_get_openapi
    schema = _nexus_get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        contact=app.contact,
    )
    for _nexus_method, _nexus_path in _NEXUS_X402_OPENAPI_OPERATIONS:
        _nexus_operation = schema.get("paths", {}).get(_nexus_path, {}).get(_nexus_method)
        if _nexus_operation is None:
            continue
        _nexus_operation["x-payment-info"] = {
            "price": {
                "mode": "fixed",
                "currency": "USD",
                "amount": _NEXUS_X402_PRICE.lstrip("$"),
            },
            "protocols": [{"x402": {}}],
        }
        _nexus_operation.setdefault("responses", {})["402"] = {"description": "Payment Required"}
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _nexus_x402_openapi_with_payment_info

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


# --- NEXUS PATCH health_openapi_security_override ---
@app.get("/health", response_model=HealthResponse, tags=["ops"], openapi_extra={"security": []})
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
from mcp.server.transport_security import TransportSecuritySettings

# --- NEXUS: PATCH fix_mcp_dns_rebinding_host_deployed_outputs ---
# FastMCP() sin host/transport_security explicito activa proteccion
# anti DNS-rebinding con allowlist localhost-only por default del SDK,
# rechazando con 421 "Invalid Host header" cualquier request real
# contra el dominio publico de Railway. Mismo fix ya validado en
# produccion sobre similarity-search-api -- replicado aca.
_nexus_railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "*")

_nexus_mcp = _NexusFastMCP(
    'nexus-useful-data-source-for-agents-doing-prod',
    stateless_http=True,
    host="0.0.0.0",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        # Railway manda el Host header SIN puerto explicito -- se
        # agrega tanto "dominio" pelado como "dominio:*" para cubrir
        # ambos casos.
        allowed_hosts=[
            "localhost:*",
            "127.0.0.1:*",
            _nexus_railway_domain,
            _nexus_railway_domain + ":*",
        ],
        allowed_origins=[
            "http://localhost:*",
            "http://127.0.0.1:*",
            "https://" + _nexus_railway_domain,
        ],
    ),
)


# --- NEXUS PATCH mcp_x402_auth_gate_buywhere ---
# _nexus_mcp_call_core() eliminada: llamaba a las rutas HTTP reales via
# ASGI in-process, pero esas mismas rutas estan protegidas por el
# middleware x402 (ver "NEXUS: x402" mas arriba) -- cualquier request,
# incluido este interno, exige un pago valido y devuelve 402 Payment
# Required en vez de la respuesta real (confirmado en produccion:
# logs/mcp_tool_grounding_2026-07-16/mcp_client_validation_result_2_post_x402_fix.json).
# Los 2 tools MCP que sobreviven ahora llaman DIRECTO a sus funciones de
# logica de negocio (search_singapore_products_by_value_score,
# get_product_vendor_price_distribution), sin pasar por ASGI/HTTP/x402 --
# mismo criterio que ya uso similarity-search-api (ver
# patch_mcp_tool_grounding_similarity_search_inprocess.py) y que ya usa
# la exclusion de billing de Stripe para tratar estas mismas rutas como
# internas. Los 2 checks que la REST ya exige se aplican aca a mano, igual
# que en similarity-search-api:
#   - auth: _require_api_key() es una funcion comun, no FastAPI-DI-only --
#     llamada directo con un `api_key` explicito del tool (parametro real
#     de esta funcion es `api_key`, no `key` como en similarity-search-api;
#     debe venir del llamador, no de VALID_API_KEYS del propio proceso).
#   - pago: x402.mcp.create_payment_wrapper (integracion MCP oficial del
#     SDK x402, ya instalado como dependencia de este asset) envuelve el
#     handler con el mismo _nexus_x402_server/PaymentRequirements/precio
#     que las rutas REST. Verifica antes del handler; liquida solo si el
#     handler retorna sin excepcion.

from x402.mcp import create_payment_wrapper as _nexus_mcp_x402_wrapper_factory
from x402.schemas.config import ResourceConfig as _NexusX402ResourceConfig

# create_payment_wrapper() necesita PaymentRequirements (asset + amount ya
# resueltos), no PaymentOption (price string sin resolver) -- son tipos
# distintos en el SDK. Se reusa el mismo camino que ya usa la libreria
# puertas adentro para las rutas REST (ver
# x402.http.x402_http_server._build_payment_requirements_from_options):
# ResourceConfig(misma price/network/pay_to que el PaymentOption de REST) ->
# server.build_payment_requirements(). Requiere que el server este
# inicializado (fetch de "supported" contra el facilitator); se garantiza
# una sola vez sin duplicar la llamada si algo mas ya lo inicializo antes.
if not getattr(_nexus_x402_server, "_initialized", False):
    _nexus_x402_server.initialize()

_NEXUS_MCP_X402_RESOURCE_CONFIG = _NexusX402ResourceConfig(
    scheme="exact",
    pay_to=_NEXUS_X402_EVM_ADDRESS,
    price=_NEXUS_X402_PRICE,
    network=_NEXUS_X402_NETWORK,
)
_NEXUS_MCP_X402_ACCEPTS = _nexus_x402_server.build_payment_requirements(_NEXUS_MCP_X402_RESOURCE_CONFIG)
_nexus_mcp_x402_wrapper = _nexus_mcp_x402_wrapper_factory(_nexus_x402_server, accepts=_NEXUS_MCP_X402_ACCEPTS)

@_nexus_mcp.tool(name='nexus_useful_data_source_for_agents_doing_prod_rank_buywhere_products_by_value_score', description="Searches the BuyWhere Singapore catalogue for products matching a natural-language query and returns them ranked by an auditable value-score derived from Shannon entropy of cross-vendor price variance, causal reliability, and price rank. Use this when an agent needs ranked product recommendations for a Singapore market query with auditable justification per product. Do NOT use for real-time stock ticker data, non-SGD markets, or exact product ID lookup (use fetch_buywhere_vendor_price_distribution for that). Requires a valid api_key (same as X-API-Key) and an x402 payment.")
@_nexus_mcp_x402_wrapper
async def rank_buywhere_products_by_value_score(query: Annotated[str, Field(..., description="Natural-language product description, e.g. 'noise-cancelling wireless headphones under SGD 300'. Must be in English.", min_length=3, max_length=300)], limit: Annotated[float, Field(10, description='Maximum number of ranked products to return. Higher values increase latency. Recommended 5-20 for agent consumption.', ge=1, le=50)], min_value_score: Annotated[float, Field(0.0, description='Minimum composite value-score [0.0, 1.0] a product must have to be included. Products below this threshold are excluded before ranking.', ge=0.0, le=1.0)], api_key: Annotated[str, Field(..., description='API key required for this paid operation -- same secret configured as X-API-Key on the REST endpoints (BUYWHERE_API_KEYS). Payment (x402) alone is not sufficient; both gates must pass.')]) -> dict[str, Any]:
    """Rank BuyWhere Products by Value Score"""
    _require_api_key(api_key=api_key)
    response = await search_singapore_products_by_value_score(query, int(limit), min_value_score, _api_key=next(iter(VALID_API_KEYS), ""))
    return response.model_dump()

@_nexus_mcp.tool(name='nexus_useful_data_source_for_agents_doing_prod_fetch_buywhere_vendor_price_distribution', description='Given a BuyWhere product identifier, returns the full cross-vendor price distribution in SGD along with the Shannon entropy of that distribution and the coefficient of variation. Use this when an agent already knows the product and needs to audit price fairness or detect outlier vendor pricing. Do NOT use as a discovery tool — it requires a known BuyWhere product_id from rank_buywhere_products_by_value_score. Requires a valid api_key (same as X-API-Key) and an x402 payment.')
@_nexus_mcp_x402_wrapper
async def fetch_buywhere_vendor_price_distribution(product_id: Annotated[str, Field(..., description='BuyWhere internal product identifier as returned by rank_buywhere_products_by_value_score. Format: alphanumeric string.', min_length=4, max_length=64)], api_key: Annotated[str, Field(..., description='API key required for this paid operation -- same secret configured as X-API-Key on the REST endpoints (BUYWHERE_API_KEYS). Payment (x402) alone is not sufficient; both gates must pass.')]) -> dict[str, Any]:
    """Fetch Vendor Price Distribution for SKU"""
    _require_api_key(api_key=api_key)
    response = await get_product_vendor_price_distribution(product_id, _api_key=next(iter(VALID_API_KEYS), ""))
    return response.model_dump()


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


# --- NEXUS: A2A Agent Card (spec v0.3.0/v1.0) para discovery ---
# Ruta correcta del spec vigente -- NO /agent.json (deprecado en
# v0.1.0), NO /v1/agent-card.json ni /v2/agent-card.json. Debe
# registrarse ANTES del app.mount("/", ...) de mas abajo: Starlette
# matchea rutas en el orden en que se agregan a app.routes, y un Mount
# en "/" intercepta cualquier path si se agrega primero.
@app.get("/.well-known/agent-card.json", include_in_schema=False)
async def _nexus_a2a_agent_card() -> dict:
    return {
        "name": "BuyWhere Singapore Value Intelligence API",
        "description": "Semantic product search over Singapore e-commerce with auditable value-scores derived from Shannon entropy across vendor price distributions.",
        "url": "https://useful-data-source-for-agents-production.up.railway.app",
        "version": "1.0.0",
        "documentationUrl": "https://useful-data-source-for-agents-production.up.railway.app/docs",
        "provider": {
            "organization": "nexus-mcp-infra",
            "url": "https://github.com/nexus-mcp-infra/useful-data-source-for-agents-doing-product-price-sdk",
        },
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "additionalInterfaces": [
            {"url": "https://useful-data-source-for-agents-production.up.railway.app/mcp", "transport": "MCP"},
        ],
        "securitySchemes": {
            "apiKeyHeader": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
            "x402Payment": {
                "type": "apiKey",
                "in": "header",
                "name": "X-PAYMENT",
                "description": "x402 payment proof (USDC, Base Sepolia testnet) required on paid operations -- not a classic auth scheme.",
            },
        },
        "security": [{"apiKeyHeader": [], "x402Payment": []}],
        "skills": [
            {
                "id": "nexus_useful_data_source_for_agents_doing_prod_rank_buywhere_products_by_value_score",
                "name": "Rank BuyWhere Products by Value Score",
                "description": "Searches the BuyWhere Singapore catalogue for products matching a natural-language query and returns them ranked by an auditable value-score derived from Shannon entropy of cross-vendor price variance, causal reliability, and price rank. Do NOT use for real-time stock ticker data, non-SGD markets, or exact product ID lookup. Requires a valid api_key (same as X-API-Key) and an x402 payment.",
                "tags": ["product-search", "price-comparison", "singapore-ecommerce"],
            },
            {
                "id": "nexus_useful_data_source_for_agents_doing_prod_fetch_buywhere_vendor_price_distribution",
                "name": "Fetch Vendor Price Distribution for SKU",
                "description": "Given a BuyWhere product identifier, returns the full cross-vendor price distribution in SGD along with the Shannon entropy of that distribution and the coefficient of variation. Do NOT use as a discovery tool -- requires a known BuyWhere product_id from rank_buywhere_products_by_value_score. Requires a valid api_key (same as X-API-Key) and an x402 payment.",
                "tags": ["price-distribution", "entropy"],
            },
        ],
        "metadata": {
            "protocol_note": (
                "This service implements the Model Context Protocol (MCP) at /mcp, "
                "not A2A's own JSONRPC/gRPC/HTTP+JSON task methods (message/send, "
                "tasks/get, etc.). This Agent Card is provided for discovery/indexing "
                "purposes; A2A-conformant task orchestration is not implemented."
            ),
        },
    }


# --- NEXUS: favicon.ico real para checklist de x402scan ---
# x402scan.com/resources/register senala /favicon.ico ausente al leer
# el listado (2026-07-26). Icono generico solido, sin decision de
# branding -- generado con stdlib puro (struct+zlib), sin Pillow.
# Debe registrarse ANTES del app.mount("/", ...) de mas abajo, mismo
# motivo que el Agent Card: Starlette matchea rutas en el orden en que
# se agregan a app.routes.
import base64 as _nexus_favicon_base64

_NEXUS_FAVICON_ICO = _nexus_favicon_base64.b64decode(
    "AAABAAEAICAAAAEAIABpAAAAFgAAAIlQTkcNChoKAAAADUlIRFIAAAAgAAAAIAgGAAAAc3p69AAAADBJREFUeNrtziEBAAAIAzCS4MlA/1wQ42ZiftWzl1QCAgICAgICAgICAgICAgLpwAO9cwRbSHXMRQAAAABJRU5ErkJggg=="
)


@app.get("/favicon.ico", include_in_schema=False)
async def _nexus_favicon():
    from fastapi import Response
    return Response(content=_NEXUS_FAVICON_ICO, media_type="image/x-icon")


# --- NEXUS: 402index.io domain claim verification file ---
# POST /api/v1/claim (domain=useful-data-source-for-agents-production.up.railway.app,
# 2026-07-26) pide este archivo estatico con el hash exacto, sin espacios
# ni saltos de linea extra. No es DNS TXT -- es un archivo servido por la
# propia app. Debe registrarse ANTES del app.mount("/", ...) de mas abajo,
# mismo motivo que favicon.ico/agent-card.json: Starlette matchea rutas en
# el orden en que se agregan a app.routes.
_NEXUS_402INDEX_VERIFY_HASH = "ecce75e8094569144f8b45fd4d3107cea6da2037889ca619d8626bd8332ea2c0"


@app.get("/.well-known/402index-verify.txt", include_in_schema=False)
async def _nexus_402index_verify():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=_NEXUS_402INDEX_VERIFY_HASH)


app.mount("/", _nexus_mcp_asgi_app)

# --- NEXUS: reporte de uso real a Stripe (inyectado por forge_output_saver_v6) ---
# --- HOTFIX: excluir paths de monitoreo/sistema del billing (ver Fase 0.5) ---
# --- NEXUS PATCH stripe_mcp_billing_exclusion ---
# /mcp agregado como entrada explicita: el sub-app FastMCP montado en "/"
# es Starlette puro (no FastAPI), sus rutas internas nunca setean
# scope["route"] (solo fastapi.routing.APIRoute.matches lo hace) -- para
# cualquier request a /mcp, _nexus_route da None y el fallback cae a
# request.url.path == "/mcp" (el "/" ya presente en el set solo cubre la
# URL raiz, no subrutas del mount). Sin esto, trafico de protocolo MCP
# (initialize, tools/list -- ninguno pasa por gate de auth/pago) se
# facturaba igual que una operacion de negocio real. Confirmado en Railway:
# STRIPE_CUSTOMER_ID/STRIPE_EVENT_NAME/STRIPE_SECRET_KEY reales, modo test.
_NEXUS_BILLING_EXCLUDED_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc", "/favicon.ico", "/mcp", "/search", "/product/{product_id}/value_breakdown", "/product/{product_id}/price_distribution", "/.well-known/agent-card.json", "/.well-known/402index-verify.txt"}  # x402 cubre estas 3 -- Stripe no debe cobrarlas de nuevo; agent-card.json/402index-verify.txt son discovery/verificacion, no negocio
@app.middleware("http")
async def _nexus_usage_middleware(request, call_next):
    response = await call_next(request)
    try:
        # NOTA: usamos la plantilla de ruta (route.path), no request.url.path,
        # porque hay endpoints con segmentos dinamicos (/product/{product_id}/...)
        # -- comparar contra la URL ya resuelta nunca matchearia la exclusion.
        _nexus_route = request.scope.get("route")
        _nexus_path_key = getattr(_nexus_route, "path", None) or request.url.path
        if (
            _nexus_path_key not in _NEXUS_BILLING_EXCLUDED_PATHS
            and response.status_code < 400
        ):
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


# --- NEXUS PATCH rate_limit_useful_data_source ---
# Rate limiting por caller (Fase 1). Identidad resuelta con la mejor senal
# disponible: wallet pagadora x402 (X-PAYMENT) > API key (X-API-Key,
# hasheada) > IP del cliente. Corre como middleware ASGI, por lo tanto
# cubre tanto las rutas REST como el sub-app FastMCP montado en "/" (ver
# docstring de patch_rate_limit_useful_data_source.py para el detalle
# completo de estas decisiones). Sin dependencias nuevas (solo stdlib).
import base64 as _nexus_rl_base64
import hashlib as _nexus_rl_hashlib
import json as _nexus_rl_json
import os as _nexus_rl_os
import threading as _nexus_rl_threading
import time as _nexus_rl_time
from collections import OrderedDict as _NexusRLOrderedDict, deque as _nexus_rl_deque

from fastapi import Request as _NexusRLRequest
from fastapi.responses import JSONResponse as _NexusRLJSONResponse

_NEXUS_RATE_LIMIT_MAX_REQUESTS = int(_nexus_rl_os.environ.get("NEXUS_RATE_LIMIT_PER_MINUTE", "60"))
_NEXUS_RATE_LIMIT_WINDOW_SECONDS = float(_nexus_rl_os.environ.get("NEXUS_RATE_LIMIT_WINDOW_SECONDS", "60"))
_NEXUS_RATE_LIMIT_MAX_TRACKED = int(_nexus_rl_os.environ.get("NEXUS_RATE_LIMIT_MAX_TRACKED", "10000"))
_NEXUS_RATE_LIMIT_EXEMPT_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc", "/favicon.ico", "/.well-known/agent-card.json", "/.well-known/402index-verify.txt"}

_nexus_rate_limit_lock = _nexus_rl_threading.Lock()
_nexus_rate_limit_state = _NexusRLOrderedDict()


def _nexus_rate_limit_extract_wallet(payment_header):
    try:
        padded = payment_header + "=" * (-len(payment_header) % 4)
        payload = _nexus_rl_json.loads(_nexus_rl_base64.b64decode(padded))
        payer = payload.get("payload", {}).get("authorization", {}).get("from")
        return payer.lower() if isinstance(payer, str) and payer else None
    except Exception:
        return None


def _nexus_rate_limit_caller_id(request):
    payment_header = request.headers.get("x-payment")
    if payment_header:
        wallet = _nexus_rate_limit_extract_wallet(payment_header)
        if wallet:
            return f"wallet:{wallet}"
    api_key = request.headers.get("x-api-key")
    if api_key:
        digest = _nexus_rl_hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return f"apikey:{digest}"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


def _nexus_rate_limit_check(caller_id):
    now = _nexus_rl_time.monotonic()
    window = _NEXUS_RATE_LIMIT_WINDOW_SECONDS
    with _nexus_rate_limit_lock:
        bucket = _nexus_rate_limit_state.get(caller_id)
        if bucket is None:
            if len(_nexus_rate_limit_state) >= _NEXUS_RATE_LIMIT_MAX_TRACKED:
                _nexus_rate_limit_state.popitem(last=False)
            bucket = _nexus_rl_deque()
            _nexus_rate_limit_state[caller_id] = bucket
        else:
            _nexus_rate_limit_state.move_to_end(caller_id)
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= _NEXUS_RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(1, int(window - (now - bucket[0])) + 1)
            return False, retry_after
        bucket.append(now)
        return True, 0


@app.middleware("http")
async def _nexus_rate_limit_middleware(request: _NexusRLRequest, call_next):
    if request.url.path in _NEXUS_RATE_LIMIT_EXEMPT_PATHS:
        return await call_next(request)
    caller_id = _nexus_rate_limit_caller_id(request)
    allowed, retry_after = _nexus_rate_limit_check(caller_id)
    if not allowed:
        return _NexusRLJSONResponse(
            status_code=429,
            content={"error": "rate_limited", "detail": f"Too many requests. Retry after {retry_after}s."},
            headers={"Retry-After": str(retry_after)},
        )
    return await call_next(request)
# --- END NEXUS PATCH rate_limit_useful_data_source ---
