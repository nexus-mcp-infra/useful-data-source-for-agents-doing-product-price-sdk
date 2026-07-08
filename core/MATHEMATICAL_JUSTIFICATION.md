# Justificación Matemática: BuyWhere Singapore MCP Tool

## 1. Máximo 5 Endpoints (Hick's Law)

El tiempo de decisión de un agente LLM al seleccionar una tool sigue $T = b \cdot \log_2(n+1)$, donde $n$ es el número de opciones disponibles. Con $n=5$ endpoints, el overhead de selección es $b \cdot \log_2(6) \approx 2.58b$; duplicar a $n=10$ lo incrementa un 43% sin ganancia proporcional en capacidad expresiva. Para esta primitiva, los cinco endpoints cubren el espacio operacional completo (búsqueda semántica, detalle de producto, comparación multi-vendor, ranking por value-score, normalización de precio) sin solapamiento: son ortogonales en el sentido de que ningún subconjunto puede derivar la salida de otro.

## 2. Pricing Per-Call vs Por Asiento (Elasticidad)

La elasticidad precio-demanda de herramientas de infraestructura de agentes es $\varepsilon = \frac{\Delta Q / Q}{\Delta P / P}$, con $|\varepsilon| > 1$ (demanda elástica) cuando el costo es visible por operación. Un pricing por asiento convierte el costo en fijo-hundido, eliminando la señal de precio en tiempo de ejecución y distorsionando el uso. El modelo per-call alinea el costo marginal de cada consulta de producto con el valor marginal generado: un agente que ejecuta 3 búsquedas para recomendar un SSD en Singapore paga exactamente por esas 3 operaciones, sin subsidiar capacidad ociosa.

## 3. Estructura de Datos: Árbol de Vendors con Índice Invertido

La búsqueda semántica sobre un catálogo de $V$ vendors con $P$ productos por vendor tiene complejidad $O(V \cdot P)$ en fuerza bruta. Un índice invertido sobre embeddings normalizados reduce la búsqueda a $O(\log VP)$ mediante aproximación de vecinos más cercanos (HNSW), mientras que el árbol de vendors permite agregar precios por producto en $O(V)$ en la capa de value-score. La elección no es estética: sin este índice, la latencia de una búsqueda semántica sobre el catálogo BuyWhere excedería el timeout típico de 30s de los frameworks AutoGen/LangChain, haciendo el activo inusable en producción.

## 4. Invariante Matemático: Entropía de Shannon como Métrica Auditable

El value-score se define como $H(x) = -\sum_{i=1}^{V} p_i \log_2(p_i)$ donde $p_i = \text{price}_i / \sum_j \text{price}_j$, combinado con una penalización de fiabilidad $\lambda \cdot \sigma^2_{\text{intra}}$ sobre la varianza histórica intra-sesión del vendor. El invariante es que $H \in [0, \log_2 V]$: $H=0$ cuando un único vendor domina el precio (concentración total, baja competencia real), $H=\log_2 V$ cuando los precios están uniformemente distribuidos (máxima incertidumbre, máximo valor informativo de comparar). Este rango acotado garantiza que el agente LLM siempre recibe un escalar normalizable y citable, independiente del número de vendors activos en la sesión.

## 5. Límites Teóricos del Sistema

El ranking causal asume estacionariedad débil en la distribución de precios intra-sesión: si BuyWhere actualiza precios con frecuencia sub-minuto (flash sales), la varianza histórica $\sigma^2_{\text{intra}}$ deja de ser un proxy válido de fiabilidad y se convierte en ruido. El sistema no puede distinguir entre un vendor con precios dinámicos legítimos y uno con datos inconsistentes sin una ventana temporal mínima de $N \geq 5$ observaciones por vendor; con $N < 5$, el estimador de varianza tiene sesgo $O(1/N)$ que invalida la penalización causal. Adicionalmente, la búsqueda semántica no resuelve ambigüedad de producto cross-category (ej. "Apple" como marca vs fruta): requiere desambiguación explícita en el input del agente.