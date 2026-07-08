"""
buywhere_singapore_sdk.py

Thin HTTP wrapper for the BuyWhere Singapore Price Intelligence API.
Exposes semantic product search with SGD-normalized prices, value-score
ranking, and structured responses ready for direct LLM agent consumption.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

_DEFAULT_BASE_URL = "https://api.buywhere.sg/v1"
_DEFAULT_TIMEOUT = 30.0


class BuyWhereSGAuthError(Exception):
    """Raised when the API key is missing or rejected."""


class BuyWhereSGValidationError(ValueError):
    """Raised when input parameters fail pre-flight validation."""


class BuyWhereSGAPIError(Exception):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API error {status_code}: {detail}")


class BuyWhereSGRateLimitError(BuyWhereSGAPIError):
    """Raised on HTTP 429 — caller should back off before retrying."""


class Client:
    """
    BuyWhere Singapore Price Intelligence SDK.

    Per-call pricing model: each method maps to exactly one billable API call.
    No logic is reimplemented client-side; value-score and causal ranking are
    computed server-side and returned as auditable fields in every response.

    Parameters
    ----------
    api_key:
        Secret key issued by BuyWhere SG. Falls back to the
        ``BUYWHERE_SG_API_KEY`` environment variable when omitted.
    base_url:
        Override the default API base URL (useful for staging environments).
    timeout:
        Per-request timeout in seconds (default: 30).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        resolved_key = api_key or os.environ.get("BUYWHERE_SG_API_KEY")
        if not resolved_key:
            raise BuyWhereSGAuthError(
                "No API key provided. Pass api_key= or set the "
                "BUYWHERE_SG_API_KEY environment variable."
            )
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            raise BuyWhereSGAuthError(
                "api_key must be a non-empty string."
            )

        self._api_key = resolved_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=self._timeout,
        )

    # ------------------------------------------------------------------
    # Primary interface — the method an agent calls directly
    # ------------------------------------------------------------------

    def main_method(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Semantic product search over the BuyWhere Singapore catalogue.

        This is the primary entry point used by LLM agents. It accepts a
        free-form query plus optional filters and returns a ranked list of
        product offers with SGD-normalized prices and auditable value-scores.

        ``data`` fields
        ---------------
        query : str (required)
            Natural-language product query, e.g. "Sony WH-1000XM5 headphones".
            Must be between 3 and 512 characters.
        top_k : int (optional, default 10, range 1-50)
            Maximum number of ranked results to return.
        min_value_score : float (optional, range 0.0-1.0)
            Filter out products whose value-score falls below this threshold.
        category : str (optional)
            Restrict search to a BuyWhere product category slug
            (e.g. "electronics", "home-appliances").
        include_vendor_detail : bool (optional, default False)
            When True, each result embeds per-vendor breakdown with individual
            prices, stock status, and intra-session variance penalties.

        Returns
        -------
        dict with keys:
            query_id        : str — unique identifier for this search call
            query           : str — echo of the submitted query
            results         : list[dict] — ranked product offers (see below)
            metadata        : dict — token usage, latency_ms, call cost SGD

        Each item in ``results`` contains:
            product_id      : str
            title           : str
            best_price_sgd  : float
            value_score     : float  (0.0 – 1.0, higher = better value)
            entropy_bits    : float  (Shannon entropy of the vendor price dist.)
            causal_rank     : int    (1 = best causal rank)
            vendor_count    : int
            vendors         : list[dict] | None  (present if include_vendor_detail)

        Raises
        ------
        BuyWhereSGValidationError   — missing/invalid fields in ``data``
        BuyWhereSGAuthError         — API key rejected (HTTP 401/403)
        BuyWhereSGRateLimitError    — request quota exceeded (HTTP 429)
        BuyWhereSGAPIError          — any other non-2xx API response
        """
        if data is None:
            raise BuyWhereSGValidationError(
                "'data' must be a dict, got None."
            )
        if not isinstance(data, dict):
            raise BuyWhereSGValidationError(
                f"'data' must be a dict, got {type(data).__name__}."
            )

        query = data.get("query")
        if query is None:
            raise BuyWhereSGValidationError(
                "Missing required field 'query' in data."
            )
        if not isinstance(query, str):
            raise BuyWhereSGValidationError(
                f"'query' must be a str, got {type(query).__name__}."
            )
        query = query.strip()
        if len(query) < 3:
            raise BuyWhereSGValidationError(
                "'query' must be at least 3 characters after stripping whitespace."
            )
        if len(query) > 512:
            raise BuyWhereSGValidationError(
                "'query' must not exceed 512 characters."
            )

        top_k = data.get("top_k", 10)
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise BuyWhereSGValidationError(
                f"'top_k' must be an int, got {type(top_k).__name__}."
            )
        if not (1 <= top_k <= 50):
            raise BuyWhereSGValidationError(
                f"'top_k' must be between 1 and 50, got {top_k}."
            )

        min_value_score = data.get("min_value_score")
        if min_value_score is not None:
            if not isinstance(min_value_score, (int, float)) or isinstance(
                min_value_score, bool
            ):
                raise BuyWhereSGValidationError(
                    f"'min_value_score' must be a float, got {type(min_value_score).__name__}."
                )
            if not (0.0 <= float(min_value_score) <= 1.0):
                raise BuyWhereSGValidationError(
                    f"'min_value_score' must be in [0.0, 1.0], got {min_value_score}."
                )

        category = data.get("category")
        if category is not None and not isinstance(category, str):
            raise BuyWhereSGValidationError(
                f"'category' must be a str, got {type(category).__name__}."
            )

        include_vendor_detail = data.get("include_vendor_detail", False)
        if not isinstance(include_vendor_detail, bool):
            raise BuyWhereSGValidationError(
                f"'include_vendor_detail' must be a bool, got "
                f"{type(include_vendor_detail).__name__}."
            )

        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "include_vendor_detail": include_vendor_detail,
        }
        if min_value_score is not None:
            payload["min_value_score"] = float(min_value_score)
        if category is not None:
            payload["category"] = category.strip()

        return self._post("/search/semantic", payload)

    # ------------------------------------------------------------------
    # Additional public methods (surface-minimal: 4 total public methods)
    # ------------------------------------------------------------------

    def get_product_value_score(self, product_id: str) -> dict[str, Any]:
        """
        Retrieve the current value-score and vendor price distribution for a
        single product by its BuyWhere product ID.

        Use this when an agent already has a product_id from a prior
        ``main_method`` call and needs a fresh score without re-running a
        full semantic search.

        Do NOT use this as the first call when only a product name is known —
        use ``main_method`` instead.

        Parameters
        ----------
        product_id : str
            BuyWhere catalogue product identifier (non-empty string, max 64 chars).

        Returns
        -------
        dict with keys:
            product_id      : str
            title           : str
            value_score     : float
            entropy_bits    : float
            causal_rank     : int
            best_price_sgd  : float
            vendor_prices   : list[dict]  (vendor_name, price_sgd, variance_penalty)
            refreshed_at    : str  (ISO-8601 UTC timestamp)
        """
        if not product_id:
            raise BuyWhereSGValidationError(
                "'product_id' must be a non-empty string."
            )
        if not isinstance(product_id, str):
            raise BuyWhereSGValidationError(
                f"'product_id' must be a str, got {type(product_id).__name__}."
            )
        pid = product_id.strip()
        if not pid:
            raise BuyWhereSGValidationError(
                "'product_id' must not be blank after stripping whitespace."
            )
        if len(pid) > 64:
            raise BuyWhereSGValidationError(
                "'product_id' must not exceed 64 characters."
            )

        return self._get(f"/products/{pid}/value-score")

    def list_categories(self) -> dict[str, Any]:
        """
        Return all available BuyWhere Singapore product category slugs and
        their display names.

        Use this to populate the optional ``category`` field in ``main_method``
        with a valid slug. Do NOT use this on every search call — cache the
        result; categories change infrequently (< once per day).

        Returns
        -------
        dict with keys:
            categories : list[dict]  (slug: str, display_name: str, product_count: int)
            total      : int
        """
        return self._get("/categories")

    def get_price_history(
        self,
        product_id: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Retrieve the SGD price history for a single product across all tracked
        vendors over the specified lookback window.

        Use this when an agent needs to reason about price trend or seasonality,
        not just the current best price. The response includes the per-day
        entropy series that feeds the causal ranking model server-side.

        Do NOT use this as a substitute for ``main_method`` — it requires a
        known product_id and does not perform semantic matching.

        Parameters
        ----------
        product_id : str
            BuyWhere catalogue product identifier (non-empty, max 64 chars).
        days : int
            Lookback window in calendar days (1-90, default 30).

        Returns
        -------
        dict with keys:
            product_id      : str
            title           : str
            history         : list[dict]  (date, min_price_sgd, max_price_sgd,
                                           mean_price_sgd, entropy_bits, vendor_count)
            currency        : "SGD"
        """
        if not product_id:
            raise BuyWhereSGValidationError(
                "'product_id' must be a non-empty string."
            )
        if not isinstance(product_id, str):
            raise BuyWhereSGValidationError(
                f"'product_id' must be a str, got {type(product_id).__name__}."
            )
        pid = product_id.strip()
        if not pid:
            raise BuyWhereSGValidationError(
                "'product_id' must not be blank after stripping whitespace."
            )
        if len(pid) > 64:
            raise BuyWhereSGValidationError(
                "'product_id' must not exceed 64 characters."
            )

        if not isinstance(days, int) or isinstance(days, bool):
            raise BuyWhereSGValidationError(
                f"'days' must be an int, got {type(days).__name__}."
            )
        if not (1 <= days <= 90):
            raise BuyWhereSGValidationError(
                f"'days' must be between 1 and 90, got {days}."
            )

        return self._get(f"/products/{pid}/price-history", params={"days": days})

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = self._http.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise BuyWhereSGAPIError(
                408,
                f"Request to {url} timed out after {self._timeout}s: {exc}",
            ) from exc
        except httpx.RequestError as exc:
            raise BuyWhereSGAPIError(
                0,
                f"Network error reaching {url}: {exc}",
            ) from exc
        return self._parse_response(response)

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = self._http.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise BuyWhereSGAPIError(
                408,
                f"Request to {url} timed out after {self._timeout}s: {exc}",
            ) from exc
        except httpx.RequestError as exc:
            raise BuyWhereSGAPIError(
                0,
                f"Network error reaching {url}: {exc}",
            ) from exc
        return self._parse_response(response)

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code == 401 or response.status_code == 403:
            raise BuyWhereSGAuthError(
                f"Authentication failed (HTTP {response.status_code}). "
                "Verify your API key is correct and active."
            )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise BuyWhereSGRateLimitError(
                429,
                f"Rate limit exceeded. Retry after {retry_after} seconds.",
            )
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise BuyWhereSGAPIError(response.status_code, detail)
        try:
            return response.json()
        except Exception as exc:
            raise BuyWhereSGAPIError(
                response.status_code,
                f"Failed to decode JSON response: {exc}. "
                f"Raw body (first 200 chars): {response.text[:200]}",
            ) from exc

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()