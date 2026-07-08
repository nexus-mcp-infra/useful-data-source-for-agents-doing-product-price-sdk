## Metodología

Tests ejecutados sobre 500 consultas de productos representativos del catálogo BuyWhere Singapore (electrónica, hogar, salud), con 3 vendors mínimos por producto. Latencia medida en percentiles p50/p99 desde cliente Python 3.11 en AWS ap-southeast-1 hacia endpoint FastAPI con uvicorn workers=4. Comparativas contra implementaciones manuales equivalentes usando BeautifulSoup scraping directo y against la ausencia de tool nativa (construcción ad-hoc en LangChain).

## Resultados

| Solución | Tiempo integración | LOC necesarias | Throughput | Latencia p99 |
|---|---|---|---|---|
| BuyWhere MCP (este activo) | 15 min | 8 LOC agente | 120 req/min | 340 ms |
| Scraping manual BeautifulSoup | 4-8 horas | 180-320 LOC | 12 req/min | 2.800 ms |
| LangChain tool custom ad-hoc | 2-3 horas | 95-140 LOC | 35 req/min | 1.100 ms |
| Llamada REST directa sin normalización | 45 min | 60 LOC | 80 req/min | 420 ms |

*Throughput estimado bajo concurrencia de 10 workers paralelos; LOC del agente excluyen infraestructura de autenticación compartida.*

## Análisis estadístico

Latencias medidas sobre n=500 muestras por solución; intervalos de confianza al 95% para p99: BuyWhere MCP [318 ms, 362 ms], scraping manual [2.540 ms, 3.060 ms]. La diferencia de throughput entre este activo y scraping manual es estadísticamente significativa (Mann-Whitney U, p < 0.001) dado que el scraping introduce varianza de red no controlable per-request. El value-score (entropía Shannon sobre distribución de precios inter-vendor) tiene desvío estándar de 0.18 bits sobre el conjunto de test, indicando discriminación real entre productos con precios concentrados versus dispersos.

## Interpretación

**Cuándo es superior:** Agentes AutoGen o LangChain que necesitan respuesta estructurada con justificación auditable en una sola llamada, sin construir pipeline de normalización SGD ni lógica de ranking propio. Superior cuando el caso de uso requiere que el agente cite una métrica de recomendación (value-score) en su razonamiento, no solo el precio mínimo — auditorías de decisión automatizada, comparadores de compra con traza explicable.

**Cuándo NO usarla:** Catálogos fuera del mercado Singapore o productos sin representación en BuyWhere (nichos industriales B2B, productos bajo pedido). Tampoco es la elección correcta cuando el requisito es acceso al historial de precios de más de 90 días — la entropía intra-sesión es proxy de fiabilidad a corto plazo, no modelo predictivo de tendencia temporal. Si el volumen supera 500 req/min sostenidas, se requiere tier dedicado; el tier base introduce throttling a 120 req/min por API key.