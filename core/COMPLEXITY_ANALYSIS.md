# Análisis de Complejidad Computacional — BuyWhere Singapore MCP

## Endpoints Públicos

### `search_buywhere_products(query, filters)`
**Temporal:** O(V · P · log P) donde V = vendors activos (~12–18 en BuyWhere SG), P = productos por vendor por query (~50–200). El log P proviene del sort final por value-score. **Espacial:** O(V · P) para el buffer de normalización SGD en memoria antes del ranking.
**Casos:** Mejor O(P log P) cuando un solo vendor responde (query muy específica, e.g., SKU exacto). Promedio O(V · P · log P) con 8–10 vendors activos. Peor O(V_max · P_max · log P_max) con query ambigua que retorna catálogo completo de todos los vendors.
**Cuello de botella:** Fan-out HTTP a vendors en paralelo — la latencia total está dominada por el vendor más lento (tail latency), no por el cálculo local.

---

### `compute_shannon_value_score(price_vector)`
**Temporal:** O(V) para calcular p_i = price_i / sum(prices), luego O(V) para H = -∑(p_i · log₂ p_i). Total estrictamente O(V). **Espacial:** O(V) para el vector de fracciones normalizadas.
**Casos:** Mejor = peor = promedio: O(V) — sin bifurcaciones, iteración lineal determinista sobre el vector de precios del producto.
**Cuello de botella:** Ninguno computacional; el riesgo real es semántico — si V < 3 vendors, la entropía colapsa a valores no discriminativos (H ≈ 0 con un solo vendor), requiriendo fallback explícito.

---

### `rank_vendors_by_causal_reliability(vendor_id, session_window)`
**Temporal:** O(W · log W) donde W = observaciones en la ventana intra-sesión (típicamente W ≤ 500 por vendor, con ventana de 30 min). El log W proviene del cálculo de varianza sobre historial ordenado por timestamp. **Espacial:** O(W) para el buffer de sesión por vendor.
**Casos:** Mejor O(1) si el vendor no tiene historial en caché (score neutro por defecto). Promedio O(W log W) con historial parcial. Peor O(W_max · log W_max) con ventana máxima y alta frecuencia de actualización de precio.
**Cuello de botella:** Escritura concurrente al store de historial intra-sesión bajo alta carga — contención en el lock del buffer compartido por vendor_id.

---

## Saturación y Estrategia de Escala

Con FastAPI + async HTTP (httpx), el punto de saturación estimado es **~40–60 req/s** en una instancia single-core antes de que la tail latency del fan-out a vendors (p95 ~800 ms) degrade el throughput — el límite no es CPU sino I/O bound en el scatter-gather. Para escalar más allá: (1) cache con TTL de 90 s por query normalizada + fingerprint de filtros para absorber queries repetidas de agentes con contexto similar, reduciendo fan-out real; (2) precalcular value-scores en background job por categoría top-20, convirtiendo O(V · P · log P) en O(1) de lookup para el 80% del tráfico esperado.