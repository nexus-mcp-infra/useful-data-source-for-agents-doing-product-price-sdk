# Pricing

El modelo de precio por llamada con tarifa decreciente en volumen refleja directamente cómo los agentes LLM consumen esta primitiva en producción: en ráfagas impredecibles, no en flujos constantes y planificables. Un agente haciendo product research para un usuario ejecuta entre una y docenas de consultas en segundos; al día siguiente puede no ejecutar ninguna. Una suscripción fija mensual le cobraría al developer por capacidad ociosa durante los valles y no escalaría justamente durante los picos. El modelo por operación elimina ese desajuste: el costo del developer sigue exactamente la curva de valor que recibe, sin subsidios cruzados entre periodos de alta y baja actividad.

La tarifa decreciente por volumen no es un descuento de fidelidad cosmético — es el reconocimiento de que el costo marginal de servir la centésima consulta de una sesión de research intensiva es structuralmente menor que el de la primera: los índices de productos ya están calientes en caché, los vectores semánticos del query anterior informan el scoring de los siguientes, y el overhead de autenticación y normalización de divisa SGD se amortiza a lo largo de la secuencia. Trasladar esa eficiencia operacional al precio de forma continua, sin escalones que generen cliffs artificiales, mantiene el incentivo del developer alineado con el uso genuino: más consultas significan mejor research, no una penalización por cruzar un tier.

La ausencia de compromiso mínimo es especialmente relevante para el segmento de agentes autónomos en pipelines AutoGen o LangChain: estos sistemas se despliegan en contextos de evaluación, staging y producción con volúmenes radicalmente distintos, y el developer no puede predecir cuántas invocaciones va a generar un agente que toma decisiones de herramienta de forma dinámica. Forzar un mínimo mensual convierte una primitiva de infraestructura en un riesgo financiero para proyectos en fase exploratoria, exactamente el perfil de adopción más valioso para establecer distribución temprana. Sin piso, el developer experimenta libremente; si el value-score calculado con entropía de Shannon produce recomendaciones que el agente puede citar y auditar, la retención se gana por calidad de output, no por lock-in contractual.

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |