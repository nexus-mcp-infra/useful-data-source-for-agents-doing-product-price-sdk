# BuyWhere Singapore Value Intelligence API — AGENTS.md

> Generado a partir del código fuente real deployado
> (`nexus-mcp-infra/useful-data-source-for-agents-doing-product-price-sdk`,
> `core/useful_data_source_for_agents_doing_product_price_api.py`), re-verificado en vivo con `curl`
> contra `https://useful-data-source-for-agents-production.up.railway.app` el 2026-07-26. Cada
> endpoint, tipo, constraint y guía de uso de este documento existe literal en ese archivo (los
> docstrings "Use this when / Do NOT use this for" son del código real, no fueron redactados de
> nuevo).
>
> **Actualización 2026-07-26 — sección MCP reescrita de nuevo, esta vez para reflejar el estado
> arreglado.** La versión anterior de este documento (2026-07-17) afirmaba que "los 2 tools están
> actualmente rotos en producción" porque el fix de matching de rutas dinámicas de x402
> (`patch_x402_dynamic_route_matching_buywhere.py`, commit `99850c2c1`) había dejado ambos tools
> bloqueados por un `402` sin que el asset tuviera ningún wrapper de pago MCP instalado — cierto en
> ese momento, pero **corregido esa misma noche del 2026-07-17** por
> `patch_mcp_x402_auth_gate_buywhere.py` (merge commit `f97909c3fb`, Railway redeploy `3db06cb5`,
> `SUCCESS`). El código deployado hoy (confirmado leyendo `core/..._api.py` en vivo desde GitHub)
> decora ambos tools con `x402.mcp.create_payment_wrapper()` y exige `api_key` como parámetro
> explícito + `_require_api_key(key=api_key)` como primera línea del handler — mismo patrón que ya
> usa `similarity-search-api` (ver su `AGENTS.md`). Re-verificado en vivo para este documento (sin
> gastar USDC real): ver sección MCP abajo.

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

## MCP — 2 tools reales, auth+pago confirmados en vivo (arreglado 2026-07-17, re-verificado 2026-07-26)

Servidor MCP embebido en `/mcp`, mismo proceso Railway que el REST (`app.mount("/", ...)`) — no hay
segundo servicio; no existe un `mcp_wrapper/src` con wrapper TS para este asset (a diferencia de
`similarity-search-api`), solo `mcp_wrapper/manifest.json` (metadata de distribución, sin código).
`initialize` en vivo devuelve `serverInfo.name = "nexus-useful-data-source-for-agents-doing-prod"`,
`version = "1.28.1"` (versión del SDK `mcp`/`FastMCP` instalado, no la del asset).

> Historia de grounding — 3 rounds de fixes:
>
> **Round 1** (`patch_mcp_tool_grounding_buywhere.py`, commit `b978e1f2a`, 2026-07-16): el servidor
> originalmente tenía 5 tools; 3 (`resolve_buywhere_product_identity`,
> `compare_buywhere_products_causal_rank`, `extract_buywhere_deal_anomalies`) llamaban rutas sin
> implementación real y fueron eliminadas. Los 2 sobrevivientes fueron remapeados a sus rutas reales
> (método y parámetros corregidos).
>
> **Round 2** (`patch_x402_dynamic_route_matching_buywhere.py`, commit `99850c2c1`, 2026-07-16 16:01
> UTC): arregló el matching de rutas dinámicas de x402 para `/product/{product_id}/price_distribution`
> — efecto secundario temporal: ambos tools quedaron bloqueados por el gate de pago porque el asset
> todavía no tenía wrapper de pago MCP instalado (`_nexus_mcp_call_core()` no adjuntaba ningún
> header/payload de pago a la llamada ASGI interna, solo `X-API-Key`).
>
> **Round 3** (`patch_mcp_x402_auth_gate_buywhere.py`, merge commit `f97909c3fb`, 2026-07-17, Railway
> redeploy `3db06cb5` `SUCCESS`): cerró la ventana rota del Round 2. Se eliminó `_nexus_mcp_call_core()`
> (llamada ASGI/HTTP contra las rutas reales) y los 2 tools pasaron a llamar **directo** a las
> funciones de negocio en Python, con un parámetro `api_key` explícito + `_require_api_key(key=api_key)`
> como primera línea del handler, decorados con `x402.mcp.create_payment_wrapper()` (integración MCP
> oficial del paquete `x402`, reusando la misma instancia `_nexus_x402_server`/precio/red que las
> rutas REST) — mismo patrón que `similarity-search-api`
> (`patch_mcp_x402_auth_gate_similarity_search.py` como referencia).

- **`nexus_useful_data_source_for_agents_doing_prod_rank_buywhere_products_by_value_score`** → lógica de
  `GET /search`. Parámetros: `query` (string, 3–300 chars), `limit` (1–50, default 10),
  `min_value_score` (0.0–1.0, default 0.0), `api_key` (string, requerido).
- **`nexus_useful_data_source_for_agents_doing_prod_fetch_buywhere_vendor_price_distribution`** →
  lógica de `GET /product/{product_id}/price_distribution`. Parámetros: `product_id` (string, 4–64
  chars), `api_key` (string, requerido).

`resolve_buywhere_product_identity`, `compare_buywhere_products_causal_rank` y
`extract_buywhere_deal_anomalies` ya no existen — no tenían ruta real detrás y no hay forma de
invocarlos.

**Confirmado en vivo, 2026-07-26** (sin gastar USDC — solo casos que no llegan a settlement):

```
tools/call rank_buywhere_products_by_value_score, api_key="fake-key-test", sin X-PAYMENT
→ isError: true, content: {"x402Version":2, "error":"Payment Required",
   "accepts":[{"scheme":"exact","network":"eip155:84532", "amount":"10000",
   "payTo":"0x70e9f...aa98f", ...}], "resource":{"url":"mcp://tool/rank_buywhere_products_by_value_score", ...}}

tools/call rank_buywhere_products_by_value_score, SIN api_key
→ isError: true, "Field required [type=missing] ... api_key"  (falla en validación de schema MCP,
   antes de llegar al wrapper de pago — mismo orden que similarity-search-api)
```

Caso de pago real + API key incorrecta (payment válido, key inválida → `401`, sin settlement) y caso de
payment válido + key válida ya fueron validados end-to-end el 2026-07-17 contra este mismo deployment
(`logs/buywhere_prod_post_merge_verify_2026-07-17.log`; el caso de key válida se probó contra un fixture
stubbeado, no contra producción, porque el upstream real de BuyWhere estaba — y sigue estando, ver
`GET /health` abajo — caído). No se repitieron esos dos casos en esta verificación para no gastar USDC
de nuevo sin necesidad; los 2 gates (pago, luego auth) siguen activos hoy con el mismo código.

Distinto de todo lo anterior: **`mcp_wrapper/`** solo contiene `manifest.json` (metadata de
distribución, sin código) — no hay wrapper TypeScript separado para este asset, a diferencia del
`mcp_wrapper/src` de `similarity-search-api`.

## Errores

`401` — API key inválida (solo si `BUYWHERE_API_KEYS` está seteada; ver gotcha de auth arriba). `422`
— validación de request (`query`/`product_id` fuera de longitud, `limit`/`min_value_score` fuera de
rango). `404` — producto no encontrado en el catálogo BuyWhere. `402` — falta pago x402 válido, en las
3 rutas REST protegidas y en los 2 tools MCP protegidos; corre antes que la validación de auth
(confirmado en vivo con `api_key` MCP falso, ver sección MCP). `502`/`503` — upstream de BuyWhere no
responde o devuelve error (confirmado en vivo: `GET /health` devuelve `upstream_reachable: false` hoy,
2026-07-26 — el upstream real sigue caído, ver nota de auth/cobro).
