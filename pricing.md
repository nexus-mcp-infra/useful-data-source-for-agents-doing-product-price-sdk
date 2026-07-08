# Pricing

El modelo de tarifa decreciente por volumen responde directamente a la naturaleza del caso de uso: los agentes autónomos (AutoGen, LangChain) no consultan esta API en sesiones predecibles ni con cadencia fija — consultan en ráfagas durante una sesión de investigación de producto, luego se detienen, luego vuelven. Un tier de suscripción mensual penaliza exactamente el comportamiento que hace valiosa esta primitiva: el agente que lanza cincuenta consultas de comparación de precios en dos minutos para resolver una decisión de compra en SGD paga por esas cincuenta llamadas, no por el mes entero en que solo las usó una vez. El compromiso mínimo crearía fricción de adopción en el único punto donde la primitiva compite: la velocidad a la que un developer puede integrarla en un pipeline de agente sin negociar contratos.

El flywheel de datos refuerza esta decisión desde el lado de los costos operativos. Cada llamada a `/prices` de alto volumen financia el re-scraping de vendors y la actualización de `price_history` — lo que significa que el costo marginal de operar la capa de ingestión cae a medida que sube el volumen agregado de consultas. Trasladar esa economía al cliente vía precio marginal decreciente no es un descuento comercial arbitrario: es la estructura de precios que mantiene el alineamiento entre quien más usa la API y quien más se beneficia de la frescura de datos. Un tier fijo no capturaría esa asimetría — cobrar igual a un agente que hace diez llamadas al mes que a uno que hace decenas de miles significa subvencionar al primero con los recursos que genera el segundo.

Finalmente, el campo `price_dispersion_bits` — la entropía de Shannon sobre la distribución de precios por vendor calculada en query-time — tiene valor analítico que escala con la frecuencia de consulta: cuanto más a menudo un agente consulta una categoría de producto, más resolución temporal obtiene sobre las anomalías de precio. Ese valor acumulado justifica que el precio marginal baje con el volumen, porque el cliente de alto volumen no solo consume datos, sino que también contribuye a la densidad histórica que hace más precisa la detección de anomalías para todos. Un modelo de suscripción fija desconectaría esa contribución del precio pagado; el modelo por llamada con tarifa decreciente la hace explícita y simétrica.

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |