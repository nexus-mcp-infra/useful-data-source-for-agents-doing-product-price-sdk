# Análisis de Complejidad Computacional — BuyWhere SGD Price Intelligence API

## Endpoints Públicos

### `GET /prices/{product_id}`

**Temporal:** O(V · H) donde V = número de vendors con snapshot activo (≤12 en catálogo SG) y H = ventana de historial consultada. El cuello de botella real es el cálculo de `price_dispersion_bits`: Shannon entropy requiere O(V log V) sobre la distribución de precios por vendor, ejecutado en query-time sobre TimescaleDB.
**Espacial:** O(V · H) para materializar los snapshots del JOIN entre `vendor_prices` y `price_history`.
**Casos:** Mejor O(V) con cache Redis hit (solo deserialización); promedio O(V · H) con TimescaleDB chunk scan; peor O(V · H · R) si R vendors disparan re-scrape en background simultáneo compitiendo por asyncpg pool.
**Cuello de botella:** El re-scrape en background comparte el pool de asyncpg (default 10 conexiones). Bajo carga alta, escrituras de flywheel bloquean lecturas analíticas.

---

### `GET /prices/search?q={query}&category={domain}`

**Temporal:** O(N log N + V · M) donde N = productos que matchean la query en el índice de texto (pg_trgm / GIN), V = vendors, M = productos en el resultado. El ranking por precio ajustado por disponibilidad añade O(M log M) de sort.
**Espacial:** O(M · V) para el resultado desnormalizado antes de serialización.
**Casos:** Mejor O(log N) con índice GIN sobre query exacta y cache Redis; promedio O(N log N) con scan parcial de índice; peor O(N · V) si la query es un token de alta frecuencia que invalida la selectividad del índice.
**Cuello de botella:** Queries de baja selectividad (e.g., `q=samsung`) fuerzan seq-scan sobre particiones TimescaleDB grandes.

---

### `GET /prices/{product_id}/dispersion`

**Temporal:** O(V log V) estricto — cálculo de entropía de Shannon H = -Σ p_i · log₂(p_i) sobre V precios normalizados. Sin historial: solo distribución spot actual.
**Espacial:** O(V) — estructura plana de probabilidades por vendor.
**Casos:** Mejor = promedio = peor O(V log V); no hay varianza significativa porque V está acotado a ≤12 vendors SG.
**Cuello de botella:** Ninguno crítico en aislamiento; el problema emerge cuando este cálculo se encadena dentro de `/prices/{product_id}` bajo carga concurrente alta.

---

## Punto de Saturación y Estrategia de Escala

Con asyncpg pool de 10 conexiones, FastAPI async, y Redis L1 cache con TTL de 60s, el punto de saturación empírico estimado es **~380 req/s** para `/prices/{product_id}` (cache hit ratio >70%) y **~90 req/s** para `/search` (predominantemente DB-bound). Más allá de ese umbral, el cuello de botella es el pool de conexiones compitiendo con los background re-scrape tasks. La estrategia de escala prioritaria es separar los workers de ingestión (flywheel) en un proceso asyncio independiente con su propio pool de conexiones read-write, dejando el pool del API server dedicado exclusivamente a lecturas; esto desacopla la latencia de escritura del P99 de lectura y permite escalar horizontalmente las réplicas de lectura de TimescaleDB sin modificar la lógica de flywheel.