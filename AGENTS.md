# BuyWhere Singapore Value Intelligence API — AGENTS.md

> Generado a partir del código fuente real deployado
> (`nexus-mcp-infra/useful-data-source-for-agents-doing-product-price-sdk`,
> `core/useful_data_source_for_agents_doing_product_price_api.py`, fetcheado en vivo desde GitHub
> `main` el 2026-07-17, commit `99850c2c1`). Cada endpoint, tipo, constraint y guía de uso de este
> documento existe literal en ese archivo (los docstrings "Use this when / Do NOT use this for" son
> del código real, no fueron redactados de nuevo). **Sección MCP reescrita el 2026-07-17**: la versión
> anterior de este documento (2026-07-16) afirmaba que "ambos tools responden 200 end-to-end" — eso
> era cierto solo en la ventana entre el fix de grounding y el fix de x402 dynamic route matching: el
> gate de pago de `/product/{product_id}/price_distribution` no se estaba aplicando por un bug de
> matching, no porque el pago no fuera necesario. Con el bug corregido (y confirmado como el
> deployment vigente en Railway, ver sección MCP), **los 2 tools están actualmente rotos en
> producción** — devuelven `402` sin ninguna forma de completar el pago desde MCP. Corregido acá.

## Qué hace

Búsqueda semántica de productos sobre el catálogo BuyWhere Singapore, con un `value_score` auditable
derivado de entropía de Shannon sobre la dispersión de precios entre vendors, más un ajuste de
confiabilidad causal por vendor. Pensado para consumo directo por agentes LLM sin postprocesamiento.

## Base URL

```
https://useful-data-source-for-agents-production.up.railway.app
```

(Railway, sin dominio propio configurado todavía.)

## Autenticación

Header `X-API-Key: <key>`. Env var real: `BUYWHERE_API_KEYS` (lista separada por comas).

**Gotcha real, no cosmético**: si `BUYWHERE_API_KEYS` está vacío en el servidor, `_require_api_key`
loguea un warning ("modo abierto — no apto para producción") pero **deja pasar el request de todos
modos**, sin validar la key. A diferencia del otro asset de NEXUS (que responde `503` si falta la key
del servidor), acá la ausencia de configuración es un auth bypass silencioso. Verificar que
`BUYWHERE_API_KEYS` esté seteada en Railway antes de asumir que el endpoint está protegido.

## Cobro

Las 3 rutas core (`/search`, `/product/{product_id}/value_breakdown`,
`/product/{product_id}/price_distribution`) requieren pago vía **x402** (USDC, red **Base Sepolia —
testnet**), `$0.01` por llamada, misma wallet que `similarity-search-api`. Excluidas del billing de
Stripe para evitar doble cobro. El middleware de exclusión de Stripe compara contra la *plantilla* de
ruta (`request.scope["route"].path`), no la URL resuelta — necesario porque dos de las tres rutas
tienen segmento dinámico `{product_id}`.

## Endpoints

### `GET /search`
Búsqueda semántica del catálogo BuyWhere, rankeada por `value_score` compuesto (entropía + confiabilidad
causal + price rank).

**Usar cuando**: un agente necesita recomendaciones de producto rankeadas para una query de mercado en
Singapur, con justificación auditable por producto.
**No usar para**: datos de stock en tiempo real, mercados fuera de SGD, o lookup exacto por ID de
producto (usar `/product/{product_id}/value_breakdown` para eso).

Query params: `query: string` (requerido, 2–256 chars), `limit: integer` (default 10, 1–50),
`min_value_score: number` (default 0.0, 0.0–1.0).

Response (`ProductSearchResponse`): `query`, `result_count`, `currency`, `products[]`
(`RankedProduct`: `product_id`, `title`, `category`, `image_url`, `vendor_offers[]`, `price_min_sgd`,
`price_max_sgd`, `price_mean_sgd`, `shannon_entropy_bits`, `causal_reliability_score`, `value_score`,
`value_score_explanation`), `computation_ms`.

```bash
curl "https://useful-data-source-for-agents-production.up.railway.app/search?query=noise+cancelling+headphones&limit=5" \
  -H "X-API-Key: $BUYWHERE_API_KEYS"
```

### `GET /product/{product_id}/value_breakdown`
Devuelve la descomposición auditable completa de la fórmula de `value_score` para un producto puntual.

**Usar cuando**: un agente necesita citar la justificación matemática de una recomendación (ej.
"recomendado porque value_score=0.82, impulsado por alta diversidad de vendors").
**No usar para**: exploración masiva de catálogo (usar `/search` para eso).

Response (`ValueScoreBreakdownResponse`): `product_id`, `title`, `shannon_entropy_bits`,
`max_possible_entropy_bits`, `entropy_component`, `reliability_component`, `price_rank_component`,
`composite_value_score`, `formula` (string con la fórmula real evaluada), `recommended_vendor`,
`recommended_price_sgd`.

### `GET /product/{product_id}/price_distribution`
Análisis de entropía de precios entre vendors para un producto: spread, coeficiente de variación,
entropía de Shannon, e interpretación en texto.

**Usar cuando**: un agente necesita determinar si el precio de un producto es estable entre vendors
(baja entropía) o volátil/competitivo (alta entropía) antes de recomendar.
**No usar para**: ranking de value-score entre múltiples productos (usar `/search` para eso).

Response (`PriceDistributionResponse`): `product_id`, `title`, `vendor_count`, `entropy_bits`,
`entropy_normalized`, `price_spread_sgd`, `coefficient_of_variation`, `interpretation`.

### `GET /health`
**Usar para**: health checks, monitores de uptime, probes de deployment readiness.
**No usar para**: queries de producto o cálculos de value-score.

Response (`HealthResponse`): `status`, `upstream_reachable` (si BuyWhere upstream responde),
`api_version`, `timestamp`. Sin autenticación.

## MCP — 2 tools reales, pero actualmente inutilizables en producción (gate x402 sin forma de pagar)

Servidor MCP embebido en `/mcp`, mismo proceso Railway que el REST (`app.mount("/", ...)`) — no hay
segundo servicio; no existe un `mcp_wrapper/src` con wrapper TS para este asset (a diferencia de
`similarity-search-api`), solo `mcp_wrapper/manifest.json` (metadata de distribución, sin código).

> Grounding de rutas: `patch_mcp_tool_grounding_buywhere.py` (commit `b978e1f2a`, 2026-07-16). El
> servidor originalmente tenía 5 tools; 3 (`resolve_buywhere_product_identity`,
> `compare_buywhere_products_causal_rank`, `extract_buywhere_deal_anomalies`) llamaban rutas sin
> implementación real y fueron eliminadas. Los 2 sobrevivientes fueron remapeados a sus rutas reales
> (método y parámetros corregidos), y la llamada interna (`_nexus_mcp_call_core`, vía
> `httpx.ASGITransport` contra el mismo `app`) se corrigió para mandar el header `X-API-Key` que
> `_require_api_key` exige (tomado de `BUYWHERE_API_KEYS`, ya cargada en el mismo proceso) — sin eso,
> incluso con la ruta correcta, la llamada fallaba con `401`.

- **`nexus_useful_data_source_for_agents_doing_prod_rank_buywhere_products_by_value_score`** → `GET /search`.
  Parámetros: `query` (string), `limit` (int), `min_value_score` (number). Los parámetros ficticios
  originales (`max_price_sgd`, `category_filter`, `include_out_of_stock`, que la ruta real no soporta)
  fueron eliminados.
- **`nexus_useful_data_source_for_agents_doing_prod_fetch_buywhere_vendor_price_distribution`** →
  `GET /product/{product_id}/price_distribution`. Parámetro: `product_id` (string). Método corregido de
  `POST` a `GET` (mismatch no solo de path, también de verbo HTTP).

**Estado real, confirmado con un cliente MCP contra producción (`https://useful-data-source-for-agents-production.up.railway.app/mcp`), 2 corridas:**

| Corrida | `rank_buywhere_products_by_value_score` | `fetch_buywhere_vendor_price_distribution` |
|---|---|---|
| 2026-07-16T15:42 UTC (antes del fix de x402 dynamic route matching) | `402 Payment Required` (ruta `/search` sí matcheaba el config de x402) | `503 Service Unavailable` (el config de x402 para `/product/:product_id/price_distribution` **no** matcheaba la ruta real por el bug de `{product_id}` — el pago nunca se chequeó, la llamada pasó directo al upstream real de BuyWhere, que está caído) |
| 2026-07-16T16:24 UTC (después del fix, deployment `80f26026` — **vigente hoy, sin deploys posteriores**) | `402 Payment Required` | `402 Payment Required` |

`patch_x402_dynamic_route_matching_buywhere.py` (commit `99850c2c1`, mergeado y redeployado a las
16:01 UTC) arregló el matching de rutas dinámicas de x402 — efecto secundario: ahora **ambos tools
quedan bloqueados por el gate de pago**, no solo uno. A diferencia de `similarity-search-api`, este
asset **no tiene ningún wrapper de pago MCP instalado**: `requirements.txt` trae `x402[fastapi,evm]`
(sin el extra `mcp`, que es el que provee `x402.mcp.create_payment_wrapper`), y
`_nexus_mcp_call_core()` no adjunta ningún header/payload de pago a la llamada ASGI interna — solo
`X-API-Key`. Resultado: **ningún cliente MCP puede completar una llamada real a estos 2 tools hoy**,
con o sin API key válida, con o sin wallet. Para destrabarlo hace falta portar el mismo fix que ya
tiene `similarity-search-api` (`patch_mcp_x402_auth_gate_similarity_search.py` como referencia) — no
aplicado todavía para este asset, es trabajo nuevo, no un patch existente.

`resolve_buywhere_product_identity`, `compare_buywhere_products_causal_rank` y
`extract_buywhere_deal_anomalies` ya no existen — no tenían ruta real detrás y no hay forma de
invocarlos.

## Errores

`401` — API key inválida (solo si `BUYWHERE_API_KEYS` está seteada; ver gotcha de auth arriba). `422`
— validación de request (`query`/`product_id` fuera de longitud, `limit`/`min_value_score` fuera de
rango). `404` — producto no encontrado en el catálogo BuyWhere. `402` — falta pago x402 válido en una
ruta protegida (REST **y ahora también los 2 tools MCP**, ver arriba — sin forma de pagar desde MCP
hoy). `502`/`503` — upstream de BuyWhere no responde o devuelve error.
