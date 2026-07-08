# Justificación Matemática: BuyWhere Singapore Price Intelligence API

## 1. Máximo 5 Endpoints (Hick's Law)

El tiempo de decisión de un agente LLM sobre qué tool invocar sigue $T = b \cdot \log_2(n+1)$, donde $b \approx 150\text{ms}$ por bit de información en contextos de tool-calling. Con $n=5$ endpoints semánticos (`search_products`, `get_price_snapshot`, `get_price_history`, `compare_vendors`, `detect_price_anomaly`), el agente resuelve la selección en $\log_2(6) \approx 2.58$ bits — frente a $\log_2(16) \approx 4$ bits con una API de 15 endpoints típica. La reducción del 35% en bits de decisión se traduce directamente en menor tasa de tool-selection error en cadenas AutoGen multi-step.

## 2. Pricing Per-Call vs Por Asiento

La elasticidad precio-demanda para infraestructura de datos de pricing es $\varepsilon = \frac{\partial Q / Q}{\partial P / P}$, y en APIs de investigación de mercado el uso es altamente variable: un agente puede invocar 0 o 400 calls en el mismo día calendario. Una suscripción fija cobra la capacidad esperada $\mathbb{E}[Q]$, capturando cero excedente del productor en los picos; el modelo per-call captura revenue proporcional a $Q_{\text{actual}}$, alineando ingresos con el ciclo de actividad real del agente. Para distribuciones de uso con $\sigma/\mu > 1$ (fat-tail, típico en agentes autónomos), per-call domina en revenue esperado sin sacrificar conversión en cuentas de bajo volumen.

## 3. Estructura de Datos: TimescaleDB + Redis + Schema Normalizado

El historial de precios por SKU × vendor es una serie temporal con cardinalidad $|SKU| \times |V|$ donde $|V| \approx 8$ retailers SG. Indexar sobre `(sku_id, vendor_id, captured_at)` en TimescaleDB con particionamiento por tiempo reduce las queries de ventana temporal a $O(\log N / k)$ donde $k$ es el número de chunks — frente a $O(N)$ en Postgres plano sobre la misma tabla. Redis actúa como capa de memoización con TTL por vendor, reduciendo la complejidad amortizada de lecturas repetidas de snapshots frescos a $O(1)$ con invalidación selectiva en background.

## 4. Invariante Matemático: Entropía de Dispersión de Precios

El invariante de corrección analítica es la entropía de Shannon sobre la distribución discreta de precios normalizados entre vendors: $H = -\sum_{i=1}^{|V|} p_i \log_2 p_i$, donde $p_i = \text{price}_i / \sum_j \text{price}_j$. Este campo `price_dispersion_bits` es invariante bajo escala monetaria (SGD vs USD no cambia $H$) y detecta anomalías cuando $H < H_{\min}$ para la categoría — condición suficiente para que un vendor esté pricing fuera del rango de competencia. Sin este invariante, el agente necesita comparación par a par entre $\binom{|V|}{2}$ pares; con él, la detección es $O(|V|)$.

## 5. Límites Teóricos del Sistema

El flywheel de ingestión lazy garantiza frescura esperada de $\mathbb{E}[\text{age}] = \text{TTL}/2$ bajo demanda uniforme, pero colapsa hacia $\text{TTL}_{\max}$ para SKUs de baja consulta — el sistema no puede garantizar SLA de frescura sobre el long tail del catálogo sin polling activo independiente, que rompe el modelo de costo cero marginal. La entropía $H$ requiere $|V| \geq 2$ vendors con precio disponible; para SKUs con vendor único, `price_dispersion_bits = 0` es matemáticamente correcto pero semánticamente no informativo. Finalmente, el scraping subyacente está acotado por el teorema CAP: durante partición de red a un retailer, el sistema retorna consistencia eventual — el precio puede tener hasta $\text{TTL}_{\text{vendor}}$ segundos de staleness sin violar el contrato de la API, que garantiza timestamp explícito, no atomicidad cross-vendor.