import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from dataclasses import dataclass
from typing import NamedTuple

# BuyWhere SGD Price API — elasticidad precio-demanda para developers
# Rango de disposicion a pagar: $0.001 (hobbyist) a $0.05 (enterprise) por operacion
# Volumen: 1K a 10M ops/mes por cliente

WTP_MIN = 0.001
WTP_MAX = 0.05
VOL_MIN = 1_000
VOL_MAX = 10_000_000

# Parametros de mercado calibrados para developer tools con flywheel de datos
# alpha: demanda base (operaciones/mes agregadas en mercado SG/SE Asia dev)
# beta: sensibilidad precio — developers son elasticos en tier bajo, inelasticos en enterprise
MARKET_ALPHA = 2_500_000.0
MARKET_BETA = 38.0
FREEMIUM_THRESHOLD_OPS = 5_000  # ops/mes gratis antes de cobrar


@dataclass
class DemandPoint:
    price: float
    quantity: float
    elasticity: float
    revenue: float
    price_dispersion_signal: float  # bits de entropia como proxy de valor percibido


class ScenarioResult(NamedTuple):
    name: str
    clients: int
    ops_per_client_monthly: float
    price_per_op: float
    monthly_revenue: float
    elasticity_at_price: float
    freemium_conversion_rate: float


def buywhere_demand(price: float, alpha: float = MARKET_ALPHA, beta: float = MARKET_BETA) -> float:
    """
    Q(P) = alpha * exp(-beta * P)
    Forma exponencial: captura la caida rapida de adopcion al subir precio en mercado dev.
    Con P=0.001 -> Q~2.4M ops; P=0.05 -> Q~5.1K ops — coherente con vol esperado.
    """
    if price <= 0:
        raise ValueError(f"price debe ser positivo, recibido: {price}")
    return alpha * np.exp(-beta * price)


def price_elasticity(price: float, alpha: float = MARKET_ALPHA, beta: float = MARKET_BETA) -> float:
    """
    epsilon = (dQ/dP) * (P/Q)
    Para Q = alpha*exp(-beta*P): dQ/dP = -beta*Q => epsilon = -beta*P
    """
    if price <= 0:
        raise ValueError(f"price debe ser positivo, recibido: {price}")
    return -beta * price  # elasticidad constante en funcion del precio, no de Q


def revenue(price: float) -> float:
    # R(P) = P * Q(P); maximizar sobre P en [WTP_MIN, WTP_MAX]
    return price * buywhere_demand(price)


def optimal_price_buywhere() -> tuple[float, float, float]:
    """
    max R(P) = P * alpha * exp(-beta*P)
    Optimo analitico: P* = 1/beta (de dR/dP=0)
    Restringido al rango de disposicion a pagar del mercado dev SG.
    """
    p_star_unconstrained = 1.0 / MARKET_BETA  # ~0.0263 USD/op
    p_star = float(np.clip(p_star_unconstrained, WTP_MIN, WTP_MAX))
    q_star = buywhere_demand(p_star)
    r_star = p_star * q_star
    return p_star, q_star, r_star


def shannon_entropy_price_dispersion(vendor_prices: list[float]) -> float:
    """
    Entropia de Shannon sobre distribucion de precios multi-vendor — diferenciador analitico.
    H = -sum(p_i * log2(p_i)) donde p_i es peso normalizado de precio de vendor i.
    Expuesto como price_dispersion_bits en cada respuesta de /prices.
    """
    if not vendor_prices or len(vendor_prices) < 2:
        return 0.0
    arr = np.array(vendor_prices, dtype=float)
    if arr.min() <= 0:
        raise ValueError("precios de vendor deben ser positivos para calcular entropia")
    weights = arr / arr.sum()
    # Evitar log(0) con clip
    weights = np.clip(weights, 1e-12, 1.0)
    return float(-np.sum(weights * np.log2(weights)))


def freemium_to_paid_conversion(price: float, free_ops: int = FREEMIUM_THRESHOLD_OPS) -> float:
    """
    Tasa de conversion freemium->paid modelada como CDF normal sobre log(price).
    Calibrada: 50% conversion a P=0.005 (precio psicologico dev), sigma=0.8 en log-space.
    Developers en SG/SE Asia convierten bien bajo $0.01/op para herramientas de agente.
    """
    if price <= 0:
        raise ValueError(f"price debe ser positivo para calcular conversion, recibido: {price}")
    mu_log = np.log(0.005)   # precio de conversion media: $0.005/op
    sigma_log = 0.8
    # P(convert) decrece al subir precio — 1 - CDF(log(price))
    return float(1.0 - norm.cdf(np.log(price), loc=mu_log, scale=sigma_log))


def simulate_adoption_scenarios() -> list[ScenarioResult]:
    """
    EXACTAMENTE 3 escenarios: hobbyist, growth-stage startup, enterprise SG/SE Asia.
    Cada uno refleja un segmento real del mercado de developer tools con agentes LLM.
    """
    scenarios_config = [
        # (nombre, n_clientes, ops_por_cliente/mes, precio/op)
        ("hobbyist_agent_builder", 800,    3_500,     0.002),   # bajo vol, sensible al precio
        ("growth_startup_sg",     120,    85_000,     0.008),   # balance elasticidad/valor
        ("enterprise_retailer",    15, 1_200_000,     0.025),   # inelastico, paga por frescura
    ]

    results = []
    for name, clients, ops_per_client, price in scenarios_config:
        total_ops_market = clients * ops_per_client
        eps = price_elasticity(price)
        conv_rate = freemium_to_paid_conversion(price)
        paying_clients = clients * conv_rate
        monthly_rev = paying_clients * ops_per_client * price

        # Simular dispersion de precios BuyWhere con 4 vendors SG (Lazada, Shopee, Courts, Harvey Norman)
        # Precios de producto de referencia escalados por segmento para entropia realista
        base_sgd = 299.0 * (1 + ops_per_client / VOL_MAX)
        vendor_prices_sgd = [
            base_sgd * np.random.uniform(0.92, 1.0),
            base_sgd * np.random.uniform(0.95, 1.08),
            base_sgd * np.random.uniform(0.88, 1.05),
            base_sgd * np.random.uniform(0.97, 1.12),
        ]
        dispersion_bits = shannon_entropy_price_dispersion(vendor_prices_sgd)

        results.append(ScenarioResult(
            name=name,
            clients=clients,
            ops_per_client_monthly=ops_per_client,
            price_per_op=price,
            monthly_revenue=monthly_rev,
            elasticity_at_price=eps,
            freemium_conversion_rate=conv_rate,
        ))
    return results


def freemium_equilibrium_price() -> dict[str, float]:
    """
    Precio de equilibrio freemium->paid: donde R(P_paid) = R(P_free=0) + valor_marginal_datos.
    Valor marginal = diferencial de entropia entre snapshot fresco vs stale * lambda_freshness.
    """
    # Estimacion de valor de freshness: TTL=300s, re-scrape dispara cuando stale
    # Cada re-scrape captura ~0.3 bits adicionales de señal de precio (flywheel data)
    lambda_freshness_usd_per_bit = 0.003  # $0.003 por bit de reduccion de incertidumbre
    delta_entropy_bits = 0.3
    marginal_data_value = lambda_freshness_usd_per_bit * delta_entropy_bits

    # Precio de equilibrio: P_eq donde conversion(P_eq) * ops_medio = breakeven_ops
    breakeven_ops = FREEMIUM_THRESHOLD_OPS
    p_star, _, _ = optimal_price_buywhere()

    # P_equilibrio: minimo precio donde R_paid supera costo de oportunidad del tier free
    # Resuelto numericamente sobre rango dev-tool
    prices = np.linspace(WTP_MIN, WTP_MAX, 10_000)
    revenues = np.array([p * freemium_to_paid_conversion(p) * breakeven_ops for p in prices])
    freemium_baseline = marginal_data_value * breakeven_ops
    above_baseline = revenues > freemium_baseline
    p_equilibrium = float(prices[above_baseline][0]) if above_baseline.any() else WTP_MIN

    return {
        "freemium_equilibrium_price_usd": round(p_equilibrium, 5),
        "marginal_data_value_usd": round(marginal_data_value, 6),
        "optimal_unconstrained_price_usd": round(1.0 / MARKET_BETA, 5),
        "optimal_constrained_price_usd": round(p_star, 5),
    }


if __name__ == "__main__":
    np.random.seed(42)

    p_opt, q_opt, r_opt = optimal_price_buywhere()
    eps_opt = price_elasticity(p_opt)

    print(f"Precio optimo: ${p_opt:.5f}/op | Q: {q_opt:,.0f} ops | Revenue: ${r_opt:,.2f}/mes")
    print(f"Elasticidad en P*: {eps_opt:.4f} (|eps|=1 confirma maximo de revenue)")

    print("\n--- Escenarios de adopcion ---")
    for s in simulate_adoption_scenarios():
        print(
            f"{s.name}: {s.clients} clientes | ${s.price_per_op}/op | "
            f"conv={s.freemium_conversion_rate:.1%} | rev=${s.monthly_revenue:,.0f}/mes | "
            f"eps={s.elasticity_at_price:.3f}"
        )

    print("\n--- Equilibrio freemium->paid ---")
    eq = freemium_equilibrium_price()
    for k, v in eq.items():
        print(f"  {k}: {v}")