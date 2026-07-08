import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import entropy as scipy_entropy
from dataclasses import dataclass, field
from typing import NamedTuple


# Parametros de mercado calibrados para developer tools SEA / Singapore e-commerce agents
P_MIN = 0.001   # USD por operacion (lower bound disposicion a pagar)
P_MAX = 0.050   # USD por operacion (upper bound disposicion a pagar)
Q_BASE = 1_000_000  # operaciones/mes referencia (punto medio geometrico 1K-10M)
PRICE_REF = 0.010   # precio de referencia para elasticidad (centil 50 del rango)


@dataclass
class DemandParams:
    alpha: float        # escala de demanda (intercepto log-log)
    beta: float         # elasticidad precio propia (negativa)
    q_freemium_cap: float  # volumen maximo en tier free antes de conversion
    conversion_rate: float # fraccion de usuarios free que pagan al tocar el cap


class ScenarioResult(NamedTuple):
    name: str
    price_optimal: float
    q_at_optimal: float
    revenue_monthly: float
    elasticity_at_optimal: float
    freemium_equilibrium_price: float


# Demanda tipo power-law: Q(P) = alpha * P^beta
# Justificacion: developer tools siguen adopcion con sensibilidad al precio log-lineal
def buywhere_demand(price: float, params: DemandParams) -> float:
    if price <= 0:
        raise ValueError(f"price debe ser > 0, recibido: {price}")
    return params.alpha * (price ** params.beta)


# Elasticidad precio-demanda puntual: epsilon = beta (constante en power-law)
# Para power-law dQ/dP = alpha * beta * P^(beta-1), entonces epsilon = beta exacto
def price_elasticity(price: float, params: DemandParams) -> float:
    # epsilon = (dQ/dP) * (P / Q) = (alpha * beta * P^(beta-1)) * (P / (alpha * P^beta)) = beta
    return params.beta  # independiente del precio en modelo log-log


def revenue(price: float, params: DemandParams) -> float:
    return price * buywhere_demand(price, params)


def optimal_price(params: DemandParams) -> tuple[float, float, float]:
    # Maximizar R(P) = P * alpha * P^beta = alpha * P^(1+beta)
    # Condicion analitica: dR/dP = alpha*(1+beta)*P^beta = 0 solo si beta = -1 (unit elastic)
    # Para beta != -1, el maximo en el dominio [P_MIN, P_MAX] se halla numericamente
    result = minimize_scalar(
        lambda p: -revenue(p, params),
        bounds=(P_MIN, P_MAX),
        method="bounded"
    )
    p_opt = result.x
    q_opt = buywhere_demand(p_opt, params)
    r_opt = p_opt * q_opt
    return p_opt, q_opt, r_opt


# Punto de equilibrio freemium->paid: precio donde revenue(paid tier) >= costo de oportunidad free
# Modelo: usuario free genera q_freemium_cap ops/mes; convierte si precio <= valor_marginal
# valor_marginal = revenue_por_op_libre estimado como P_MIN * (1 + margen_info)
def freemium_equilibrium(params: DemandParams, info_margin: float = 0.8) -> float:
    # info_margin refleja la prima del value-score (entropia de Shannon) sobre precio minimo puro
    # Un value-score auditable justifica precio mayor que el scraper de precio-minimo
    marginal_value = P_MIN * (1.0 + info_margin)
    # Precio de equilibrio: punto donde conversion_rate * Q(P) * P = Q_free * P_MIN
    # Despejando P: P = (Q_free * P_MIN) / (conversion_rate * alpha * P^beta) -> iteracion
    from scipy.optimize import brentq
    def equilibrium_condition(p: float) -> float:
        paid_revenue = params.conversion_rate * buywhere_demand(p, params) * p
        free_opportunity_cost = params.q_freemium_cap * P_MIN
        return paid_revenue - free_opportunity_cost

    try:
        p_eq = brentq(equilibrium_condition, P_MIN, P_MAX, xtol=1e-7)
    except ValueError:
        # No hay cruce en el rango: retornar bound mas cercano
        p_eq = P_MIN if equilibrium_condition(P_MIN) > 0 else P_MAX
    return p_eq


# Shannon entropy sobre distribucion de precios entre vendors (mismo calculo que value-score)
# p_i = precio_vendor_i / sum(precios); H alta -> alta dispersion -> mayor valor informacional
def vendor_price_entropy(vendor_prices: list[float]) -> float:
    if len(vendor_prices) < 2:
        raise ValueError("Se requieren al menos 2 vendors para calcular entropia de precios")
    prices = np.array(vendor_prices, dtype=float)
    if np.any(prices <= 0):
        raise ValueError("Todos los precios deben ser positivos")
    probs = prices / prices.sum()   # fraccion relativa al total (no al rango, consistente con spec)
    return float(scipy_entropy(probs, base=2))  # bits de incertidumbre entre vendors


# Escenario 1: Early adopters (alto volumen, baja sensibilidad al precio)
# Beta moderado: developers que ya tienen pain point definido y presupuesto
scenario_early_adopter = DemandParams(
    alpha=Q_BASE * (PRICE_REF ** 1.2),  # calibrado para Q=Q_BASE en P=PRICE_REF
    beta=-1.2,
    q_freemium_cap=50_000,
    conversion_rate=0.18
)

# Escenario 2: Mercado masivo SEA (alta sensibilidad, adopcion gradual)
# Beta alto: integradores price-sensitive en mercados emergentes
scenario_sea_mass_market = DemandParams(
    alpha=Q_BASE * (PRICE_REF ** 2.1),
    beta=-2.1,
    q_freemium_cap=200_000,
    conversion_rate=0.06
)

# Escenario 3: Enterprise agents (bajo volumen, muy baja elasticidad, SLA-driven)
# Beta bajo: equipos de producto que valoran auditabilidad del value-score sobre precio
scenario_enterprise_agents = DemandParams(
    alpha=Q_BASE * (PRICE_REF ** 0.7),
    beta=-0.7,
    q_freemium_cap=10_000,
    conversion_rate=0.35
)

SCENARIOS: dict[str, DemandParams] = {
    "early_adopter_sea_devtools": scenario_early_adopter,
    "sea_mass_market_integrators": scenario_sea_mass_market,
    "enterprise_llm_agents": scenario_enterprise_agents,
}


def simulate_all_scenarios() -> list[ScenarioResult]:
    results = []
    for name, params in SCENARIOS.items():
        p_opt, q_opt, r_opt = optimal_price(params)
        eps = price_elasticity(p_opt, params)
        p_eq = freemium_equilibrium(params)
        results.append(ScenarioResult(
            name=name,
            price_optimal=round(p_opt, 6),
            q_at_optimal=round(q_opt, 0),
            revenue_monthly=round(r_opt, 2),
            elasticity_at_optimal=round(eps, 3),
            freemium_equilibrium_price=round(p_eq, 6),
        ))
    return results


def validate_model_consistency(results: list[ScenarioResult]) -> None:
    for r in results:
        assert P_MIN <= r.price_optimal <= P_MAX, (
            f"Precio optimo {r.price_optimal} fuera del rango de disposicion a pagar [{P_MIN}, {P_MAX}]"
        )
        assert r.q_at_optimal > 0, f"Demanda en optimo debe ser positiva para {r.name}"
        assert r.revenue_monthly > 0, f"Revenue mensual debe ser positivo para {r.name}"
        assert r.elasticity_at_optimal < 0, f"Elasticidad debe ser negativa (ley de demanda) para {r.name}"


if __name__ == "__main__":
    results = simulate_all_scenarios()
    validate_model_consistency(results)

    # Ejemplo de calculo de entropia con precios tipicos de vendors en BuyWhere Singapore
    example_vendor_prices_sgd = [45.90, 52.00, 48.50, 61.00, 47.20]
    h = vendor_price_entropy(example_vendor_prices_sgd)

    for r in results:
        print(
            f"scenario={r.name} | "
            f"P_opt=${r.price_optimal:.5f} | "
            f"Q_opt={r.q_at_optimal:.0f} ops/month | "
            f"R_monthly=${r.revenue_monthly:.2f} | "
            f"epsilon={r.elasticity_at_optimal:.3f} | "
            f"freemium_eq_price=${r.freemium_equilibrium_price:.5f}"
        )
    print(f"vendor_price_entropy_example={h:.4f} bits (5 SGD vendors)")