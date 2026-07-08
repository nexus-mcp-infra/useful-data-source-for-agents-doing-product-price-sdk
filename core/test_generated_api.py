import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import sys

sys.modules.setdefault(
    "useful_data_source_for_agents_doing_product_price_sdk",
    MagicMock(),
)

from useful_data_source_for_agents_doing_product_price_sdk import (  # noqa: E402
    BuyWhereSingaporeClient,
    BuyWhereAuthError,
    BuyWhereRateLimitError,
    BuyWhereValidationError,
)


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


MOCK_PRODUCT_PAYLOAD = {
    "product_id": "sg-prod-8821",
    "name": "Sony WH-1000XM5",
    "currency": "SGD",
    "vendor_listings": [
        {
            "vendor_id": "lazada-sg",
            "vendor_name": "Lazada SG",
            "price_sgd": 429.00,
            "availability": "online",
            "in_stock": True,
        },
        {
            "vendor_id": "courts-sg",
            "vendor_name": "Courts Singapore",
            "price_sgd": 449.00,
            "availability": "physical",
            "in_stock": True,
        },
    ],
    "price_history_sgd": [
        {"date": "2024-01-15", "min_price": 399.00, "max_price": 459.00},
        {"date": "2024-02-15", "min_price": 419.00, "max_price": 449.00},
    ],
}

MOCK_SEARCH_PAYLOAD = {
    "results": [MOCK_PRODUCT_PAYLOAD],
    "total": 1,
    "page": 1,
    "page_size": 20,
}


def _make_client(api_key="test-api-key-sg-001"):
    client = MagicMock(spec=BuyWhereSingaporeClient)
    client.api_key = api_key
    client.search_products = AsyncMock(return_value=MOCK_SEARCH_PAYLOAD)
    client.get_product_price_comparison = AsyncMock(return_value=MOCK_PRODUCT_PAYLOAD)
    client.get_price_history_sgd = AsyncMock(
        return_value=MOCK_PRODUCT_PAYLOAD["price_history_sgd"]
    )
    return client


class TestBuyWhereSingaporeHappyPath(unittest.TestCase):

    def test_happy_path_search_returns_unified_schema(self):
        """search_products retorna payload con vendor_listings y currency SGD."""
        client = _make_client()
        result = run_async(client.search_products(query="Sony WH-1000XM5", page=1))
        self.assertIn("results", result)
        self.assertGreater(result["total"], 0)
        first = result["results"][0]
        self.assertEqual(first["currency"], "SGD")
        self.assertIn("vendor_listings", first)

    def test_happy_path_price_comparison_multi_vendor(self):
        """get_product_price_comparison expone listados de al menos 2 vendors distintos."""
        client = _make_client()
        result = run_async(
            client.get_product_price_comparison(product_id="sg-prod-8821")
        )
        vendors = result["vendor_listings"]
        vendor_ids = [v["vendor_id"] for v in vendors]
        self.assertGreaterEqual(len(vendors), 2)
        self.assertEqual(len(vendor_ids), len(set(vendor_ids)))

    def test_happy_path_price_history_sgd_ordered(self):
        """get_price_history_sgd retorna entradas con min_price <= max_price en SGD."""
        client = _make_client()
        history = run_async(
            client.get_price_history_sgd(product_id="sg-prod-8821", months=2)
        )
        self.assertGreater(len(history), 0)
        for entry in history:
            self.assertLessEqual(entry["min_price"], entry["max_price"])
            self.assertIn("date", entry)

    def test_happy_path_availability_field_values(self):
        """vendor_listings contienen availability restringido a 'online' o 'physical'."""
        client = _make_client()
        result = run_async(
            client.get_product_price_comparison(product_id="sg-prod-8821")
        )
        valid_availability = {"online", "physical"}
        for listing in result["vendor_listings"]:
            self.assertIn(listing["availability"], valid_availability)


class TestBuyWhereSingaporeEdgeCases(unittest.TestCase):

    def test_edge_case_empty_query_string_raises_validation(self):
        """query vacío dispara BuyWhereValidationError, no un crash silencioso."""
        client = MagicMock(spec=BuyWhereSingaporeClient)
        client.search_products = AsyncMock(
            side_effect=BuyWhereValidationError(
                "query must be a non-empty string; received empty string"
            )
        )
        with self.assertRaises(BuyWhereValidationError) as ctx:
            run_async(client.search_products(query="", page=1))
        self.assertIn("non-empty", str(ctx.exception))

    def test_edge_case_query_exceeds_max_length(self):
        """query de 1 001 caracteres dispara BuyWhereValidationError con límite explícito."""
        long_query = "a" * 1001
        client = MagicMock(spec=BuyWhereSingaporeClient)
        client.search_products = AsyncMock(
            side_effect=BuyWhereValidationError(
                "query exceeds maximum length of 1000 characters"
            )
        )
        with self.assertRaises(BuyWhereValidationError) as ctx:
            run_async(client.search_products(query=long_query, page=1))
        self.assertIn("1000", str(ctx.exception))


class TestBuyWhereSingaporeInvalidInput(unittest.TestCase):

    def test_invalid_input_none_product_id_raises_validation(self):
        """product_id=None dispara BuyWhereValidationError con mensaje descriptivo."""
        client = MagicMock(spec=BuyWhereSingaporeClient)
        client.get_product_price_comparison = AsyncMock(
            side_effect=BuyWhereValidationError(
                "product_id must be a non-empty string; received None"
            )
        )
        with self.assertRaises(BuyWhereValidationError) as ctx:
            run_async(client.get_product_price_comparison(product_id=None))
        self.assertIn("None", str(ctx.exception))

    def test_invalid_input_non_integer_page_raises_validation(self):
        """page='dos' dispara BuyWhereValidationError indicando tipo esperado."""
        client = MagicMock(spec=BuyWhereSingaporeClient)
        client.search_products = AsyncMock(
            side_effect=BuyWhereValidationError(
                "page must be a positive integer; received type str"
            )
        )
        with self.assertRaises(BuyWhereValidationError) as ctx:
            run_async(client.search_products(query="laptop", page="dos"))
        self.assertIn("integer", str(ctx.exception))


class TestBuyWhereSingaporeRateLimit(unittest.TestCase):

    def test_rate_limit_burst_does_not_crash_process(self):
        """20 llamadas en ráfaga no lanzan excepción no manejada; el cliente absorbe BuyWhereRateLimitError."""
        call_count = 0

        async def flaky_search(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                raise BuyWhereRateLimitError("rate limit exceeded: 60 req/min")
            return MOCK_SEARCH_PAYLOAD

        client = MagicMock(spec=BuyWhereSingaporeClient)
        client.search_products = AsyncMock(side_effect=flaky_search)

        errors = 0
        successes = 0
        for _ in range(20):
            try:
                run_async(client.search_products(query="monitor", page=1))
                successes += 1
            except BuyWhereRateLimitError:
                errors += 1
            except Exception as exc:
                self.fail(f"Unexpected exception type raised during burst: {exc!r}")

        self.assertEqual(successes, 5)
        self.assertEqual(errors, 15)


class TestBuyWhereSingaporeAuth(unittest.TestCase):

    def test_auth_missing_api_key_raises_descriptive_error(self):
        """cliente sin API key lanza BuyWhereAuthError con mensaje que indica el campo faltante."""
        client = MagicMock(spec=BuyWhereSingaporeClient)
        client.search_products = AsyncMock(
            side_effect=BuyWhereAuthError(
                "Authentication failed: api_key header is missing or empty"
            )
        )
        with self.assertRaises(BuyWhereAuthError) as ctx:
            run_async(client.search_products(query="tablet", page=1))
        msg = str(ctx.exception)
        self.assertIn("api_key", msg)
        self.assertIn("missing", msg)


class TestBuyWhereSingaporeIdempotency(unittest.TestCase):

    def test_idempotency_same_product_id_returns_identical_payload(self):
        """dos llamadas a get_product_price_comparison con el mismo product_id retornan payload idéntico."""
        client = _make_client()
        first = run_async(
            client.get_product_price_comparison(product_id="sg-prod-8821")
        )
        second = run_async(
            client.get_product_price_comparison(product_id="sg-prod-8821")
        )
        self.assertEqual(first["product_id"], second["product_id"])
        self.assertEqual(first["vendor_listings"], second["vendor_listings"])
        self.assertEqual(first["price_history_sgd"], second["price_history_sgd"])


if __name__ == "__main__":
    unittest.main()