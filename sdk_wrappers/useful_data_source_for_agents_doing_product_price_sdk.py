import os
import time
from typing import Any, Optional
import httpx

DEFAULT_BASE_URL = "https://api.buywhere-sg.nexus.ai/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5


class BuyWhereSGAuthError(Exception):
    pass


class BuyWhereSGValidationError(Exception):
    pass


class BuyWhereSGRateLimitError(Exception):
    pass


class BuyWhereSGAPIError(Exception):
    def __init__(self, message: str, status_code: int, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def _validate_non_empty_string(value: Any, field_name: str) -> str:
    if value is None:
        raise BuyWhereSGValidationError(f"'{field_name}' must not be None")
    if not isinstance(value, str):
        raise BuyWhereSGValidationError(
            f"'{field_name}' must be a string, got {type(value).__name__}"
        )
    stripped = value.strip()
    if not stripped:
        raise BuyWhereSGValidationError(f"'{field_name}' must not be an empty string")
    return stripped


def _validate_product_domain(domain: str) -> str:
    allowed = {"electronics", "appliances", "computing"}
    if domain not in allowed:
        raise BuyWhereSGValidationError(
            f"'product_domain' must be one of {sorted(allowed)}, got '{domain}'"
        )
    return domain


def _validate_positive_int(value: Any, field_name: str, maximum: int = 100) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BuyWhereSGValidationError(
            f"'{field_name}' must be an integer, got {type(value).__name__}"
        )
    if value < 1:
        raise BuyWhereSGValidationError(f"'{field_name}' must be >= 1, got {value}")
    if value > maximum:
        raise BuyWhereSGValidationError(
            f"'{field_name}' must be <= {maximum}, got {value}"
        )
    return value


class Client:
    """
    Thin HTTP wrapper over the BuyWhere Singapore price intelligence API.

    All prices are denominated in SGD. Responses include price_dispersion_bits,
    a Shannon-entropy measure over the vendor price distribution computed
    server-side — use it to flag anomalous pricing without any additional parsing.

    Usage:
        client = Client(api_key="sk-...")
        result = client.search_products("sony wh-1000xm5")
        result = client.get_product_prices("SKU-abc123")
        result = client.compare_vendor_prices("SKU-abc123")
        result = client.get_price_history("SKU-abc123", days=30)
        result = client.detect_price_anomalies("electronics", threshold_bits=1.5)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        resolved_key = api_key or os.environ.get("BUYWHERE_SG_API_KEY")
        if not resolved_key:
            raise BuyWhereSGAuthError(
                "API key is required. Pass api_key= or set BUYWHERE_SG_API_KEY "
                "environment variable."
            )
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            raise BuyWhereSGAuthError(
                "API key must be a non-empty string."
            )
        self._api_key = resolved_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Client": "buywhere-sg-python-sdk/1.0.0",
            },
            timeout=self._timeout,
        )

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
    ) -> dict:
        url = f"{self._base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                response = self._http.request(
                    method, url, params=params, json=json_body
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
                continue
            except httpx.RequestError as exc:
                raise BuyWhereSGAPIError(
                    f"Network error contacting BuyWhere SG API: {exc}",
                    status_code=0,
                ) from exc

            if response.status_code == 401:
                raise BuyWhereSGAuthError(
                    "Invalid or expired API key. Check your credentials."
                )
            if response.status_code == 422:
                try:
                    body = response.json()
                except Exception:
                    body = response.text
                raise BuyWhereSGValidationError(
                    f"Request validation failed: {body}"
                )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "unknown")
                raise BuyWhereSGRateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after} seconds."
                )
            if response.status_code >= 500:
                last_exc = BuyWhereSGAPIError(
                    f"Server error {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                    response_body=response.text,
                )
                wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
                continue
            if response.status_code >= 400:
                try:
                    body = response.json()
                except Exception:
                    body = response.text
                raise BuyWhereSGAPIError(
                    f"Client error {response.status_code}: {body}",
                    status_code=response.status_code,
                    response_body=body,
                )
            try:
                return response.json()
            except Exception as exc:
                raise BuyWhereSGAPIError(
                    "API returned non-JSON response.",
                    status_code=response.status_code,
                    response_body=response.text,
                ) from exc

        if last_exc is not None:
            raise BuyWhereSGAPIError(
                f"Request failed after {self._max_retries} attempts: {last_exc}",
                status_code=0,
            ) from last_exc
        raise BuyWhereSGAPIError(
            f"Request failed after {self._max_retries} attempts.",
            status_code=0,
        )

    def main_method(self, data: Any) -> dict:
        """
        Unified entry point accepting a dict or string query.

        - If data is a string: delegates to search_products(data).
        - If data is a dict with key 'sku': delegates to get_product_prices(data['sku']).
        - If data is a dict with key 'query': delegates to search_products(data['query']).
        - All other keys in the dict are forwarded as optional parameters.

        Raises BuyWhereSGValidationError for None, wrong type, or missing keys.
        """
        if data is None:
            raise BuyWhereSGValidationError(
                "'data' must not be None. Pass a search query string or a dict "
                "with 'query' or 'sku' key."
            )
        if isinstance(data, str):
            return self.search_products(data)
        if isinstance(data, dict):
            if "sku" in data:
                sku = data["sku"]
                limit = data.get("limit", 20)
                return self.get_product_prices(sku, limit=limit)
            if "query" in data:
                query = data["query"]
                product_domain = data.get("product_domain")
                limit = data.get("limit", 20)
                return self.search_products(
                    query, product_domain=product_domain, limit=limit
                )
            raise BuyWhereSGValidationError(
                "'data' dict must contain 'query' (product search string) or "
                "'sku' (product identifier). Got keys: "
                f"{list(data.keys())}"
            )
        raise BuyWhereSGValidationError(
            f"'data' must be a str or dict, got {type(data).__name__}"
        )

    def search_products(
        self,
        query: str,
        product_domain: Optional[str] = None,
        limit: int = 20,
    ) -> dict:
        """
        Full-text search over the BuyWhere SG catalogue.

        Returns matched products with current best SGD price, vendor count,
        and availability_mode ('online'|'physical'|'both').

        Args:
            query: Product search string, e.g. "sony wh-1000xm5" (1-200 chars).
            product_domain: Optional domain filter — 'electronics', 'appliances',
                or 'computing'. Omit to search across all domains.
            limit: Number of results to return (1-100). Default 20.

        Returns:
            dict with keys: products (list), total_count (int), query_time_ms (float).
        """
        validated_query = _validate_non_empty_string(query, "query")
        if len(validated_query) > 200:
            raise BuyWhereSGValidationError(
                f"'query' must not exceed 200 characters, got {len(validated_query)}"
            )
        validated_limit = _validate_positive_int(limit, "limit", maximum=100)
        params: dict = {"q": validated_query, "limit": validated_limit}
        if product_domain is not None:
            params["domain"] = _validate_product_domain(product_domain)
        return self._request("GET", "/products/search", params=params)

    def get_product_prices(
        self,
        sku: str,
        limit: int = 20,
    ) -> dict:
        """
        Retrieve current vendor prices for a specific product SKU in SGD.

        Response includes price_dispersion_bits (Shannon entropy over vendor
        price distribution) — values above 1.5 bits typically indicate a
        pricing anomaly worth surfacing to the end user.

        Args:
            sku: Product SKU identifier as returned by search_products (1-100 chars).
            limit: Max number of vendor price listings to return (1-100). Default 20.

        Returns:
            dict with keys: sku, product_name, vendors (list of vendor+price+
            availability), price_dispersion_bits (float), snapshot_age_seconds (int).
        """
        validated_sku = _validate_non_empty_string(sku, "sku")
        if len(validated_sku) > 100:
            raise BuyWhereSGValidationError(
                f"'sku' must not exceed 100 characters, got {len(validated_sku)}"
            )
        validated_limit = _validate_positive_int(limit, "limit", maximum=100)
        return self._request(
            "GET",
            f"/products/{validated_sku}/prices",
            params={"limit": validated_limit},
        )

    def compare_vendor_prices(
        self,
        sku: str,
        availability_mode: Optional[str] = None,
    ) -> dict:
        """
        Side-by-side vendor comparison for a SKU, ranked by SGD price ascending.

        Filters by availability_mode to restrict results to physical stores,
        online-only, or both. Includes price_dispersion_bits across the filtered
        vendor set so agents can detect whether filtering changes anomaly risk.

        Use this when the agent needs to recommend a specific vendor, not just
        a price range. Do NOT use this as a substitute for search_products
        when the SKU is unknown.

        Args:
            sku: Product SKU identifier (1-100 chars).
            availability_mode: Optional filter — 'online', 'physical', or 'both'.
                Omit to include all vendors regardless of channel.

        Returns:
            dict with keys: sku, ranked_vendors (list), cheapest_vendor,
            price_spread_sgd (float), price_dispersion_bits (float).
        """
        validated_sku = _validate_non_empty_string(sku, "sku")
        if len(validated_sku) > 100:
            raise BuyWhereSGValidationError(
                f"'sku' must not exceed 100 characters, got {len(validated_sku)}"
            )
        params: dict = {}
        if availability_mode is not None:
            allowed_modes = {"online", "physical", "both"}
            if availability_mode not in allowed_modes:
                raise BuyWhereSGValidationError(
                    f"'availability_mode' must be one of {sorted(allowed_modes)}, "
                    f"got '{availability_mode}'"
                )
            params["availability_mode"] = availability_mode
        return self._request(
            "GET", f"/products/{validated_sku}/compare", params=params or None
        )

    def get_price_history(
        self,
        sku: str,
        days: int = 30,
    ) -> dict:
        """
        Time-series of SGD price observations for a SKU across all vendors.

        Data is sourced from TimescaleDB and covers up to 90 days of history.
        Each data point includes vendor_id, price_sgd, and recorded_at (ISO-8601).

        Use this to surface price trends or detect if current price is
        above/below the rolling average. Do NOT use this for real-time
        availability — use get_product_prices instead.

        Args:
            sku: Product SKU identifier (1-100 chars).
            days: Number of historical days to retrieve (1-90). Default 30.

        Returns:
            dict with keys: sku, history (list of price points), period_days (int),
            min_price_sgd (float), max_price_sgd (float), mean_price_sgd (float).
        """
        validated_sku = _validate_non_empty_string(sku, "sku")
        if len(validated_sku) > 100:
            raise BuyWhereSGValidationError(
                f"'sku' must not exceed 100 characters, got {len(validated_sku)}"
            )
        validated_days = _validate_positive_int(days, "days", maximum=90)
        return self._request(
            "GET",
            f"/products/{validated_sku}/price-history",
            params={"days": validated_days},
        )

    def detect_price_anomalies(
        self,
        product_domain: str,
        threshold_bits: float = 1.5,
        limit: int = 20,
    ) -> dict:
        """
        Return products in a domain where price_dispersion_bits exceeds threshold.

        High dispersion (bits > 1.5) indicates vendors have materially divergent
        prices — useful for surfacing arbitrage opportunities or data quality issues.
        Computed server-side via Shannon entropy over the vendor price distribution
        at query time; no client-side parsing required.

        Use this when the agent is doing market-wide price intelligence sweeps.
        Do NOT use this for single-product price lookup — use get_product_prices.

        Args:
            product_domain: Domain to scan — 'electronics', 'appliances', or 'computing'.
            threshold_bits: Minimum entropy threshold in bits (0.0-4.0). Default 1.5.
            limit: Max anomalies to return (1-100). Default 20.

        Returns:
            dict with keys: domain, anomalies (list of sku+name+price_dispersion_bits),
            threshold_bits (float), scanned_product_count (int).
        """
        validated_domain = _validate_product_domain(product_domain)
        if not isinstance(threshold_bits, (int, float)) or isinstance(
            threshold_bits, bool
        ):
            raise BuyWhereSGValidationError(
                f"'threshold_bits' must be a float, got {type(threshold_bits).__name__}"
            )
        if threshold_bits < 0.0 or threshold_bits > 4.0:
            raise BuyWhereSGValidationError(
                f"'threshold_bits' must be between 0.0 and 4.0, got {threshold_bits}"
            )
        validated_limit = _validate_positive_int(limit, "limit", maximum=100)
        return self._request(
            "GET",
            "/prices/anomalies",
            params={
                "domain": validated_domain,
                "threshold_bits": threshold_bits,
                "limit": validated_limit,
            },
        )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()