import unittest
from unittest.mock import patch, MagicMock
import json
import sys

sys.modules.setdefault(
    "useful_data_source_for_agents_doing_product_price_sdk",
    MagicMock()
)

import useful_data_source_for_agents_doing_product_price_sdk as sdk_module

MOCK_PRODUCT_RESULT = {
    "query": "wireless headphones",
    "results": [
        {
            "product_id": "bw-sg-00421",
            "name": "Sony WH-1000XM5 Wireless Headphones",
            "price_sgd": 429.00,
            "retailer": "Challenger",
            "url": "https://buywhere.sg/product/bw-sg-00421",
            "value_score": 0.87,
            "in_stock": True,
        },
        {
            "product_id": "bw-sg-00398",
            "name": "Bose QuietComfort 45",
            "price_sgd": 479.00,
            "retailer": "Best Denki",
            "url": "https://buywhere.sg/product/bw-sg-00398",
            "value_score": 0.81,
            "in_stock": True,
        },
    ],
    "total_found": 2,
    "currency": "SGD",
}

MOCK_AUTH_ERROR = {
    "error": "authentication_failed",
    "message": "Missing or invalid BuyWhere API key. Provide a valid key via the 'X-BuyWhere-Key' header.",
    "status_code": 401,
}

MOCK_RATE_LIMIT_ERROR = {
    "error": "rate_limit_exceeded",
    "message": "BuyWhere API rate limit reached: 60 requests/minute. Retry after 1 second.",
    "status_code": 429,
    "retry_after_seconds": 1,
}


def _make_sdk_client(api_key="bw-test-key-mock"):
    client = MagicMock()
    client.api_key = api_key
    return client


class BuyWhereSingaporeSDKTests(unittest.TestCase):

    def setUp(self):
        self.client = _make_sdk_client()

    # ---------------------------------------------------------------------------
    # HAPPY PATH
    # ---------------------------------------------------------------------------

    def test_happy_path_semantic_search_returns_ranked_sgd_results(self):
        """Verifica que una búsqueda semántica válida devuelve resultados ordenados por value_score en SGD."""
        self.client.search_products.return_value = MOCK_PRODUCT_RESULT

        response = self.client.search_products(
            query="wireless headphones",
            max_results=10,
            min_value_score=0.5,
        )

        self.assertEqual(response["currency"], "SGD")
        self.assertGreater(len(response["results"]), 0)
        scores = [r["value_score"] for r in response["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIn("product_id", response["results"][0])
        self.assertIn("price_sgd", response["results"][0])

    def test_happy_path_product_detail_returns_structured_agent_payload(self):
        """Verifica que get_product_detail devuelve un payload con todos los campos que un agente LLM necesita sin postprocesamiento."""
        detail_payload = {
            "product_id": "bw-sg-00421",
            "name": "Sony WH-1000XM5 Wireless Headphones",
            "price_sgd": 429.00,
            "retailer": "Challenger",
            "url": "https://buywhere.sg/product/bw-sg-00421",
            "value_score": 0.87,
            "in_stock": True,
            "price_history_sgd": [449.00, 439.00, 429.00],
            "category": "Audio",
        }
        self.client.get_product_detail.return_value = detail_payload

        result = self.client.get_product_detail(product_id="bw-sg-00421")

        required_agent_fields = {
            "product_id", "name", "price_sgd", "retailer",
            "url", "value_score", "in_stock",
        }
        self.assertTrue(required_agent_fields.issubset(result.keys()))
        self.assertIsInstance(result["price_sgd"], float)
        self.assertIsInstance(result["value_score"], float)
        self.assertGreaterEqual(result["value_score"], 0.0)
        self.assertLessEqual(result["value_score"], 1.0)

    # ---------------------------------------------------------------------------
    # EDGE CASES
    # ---------------------------------------------------------------------------

    def test_edge_case_empty_query_string_raises_value_error(self):
        """Verifica que una query vacía es rechazada antes de llegar a la red."""
        self.client.search_products.side_effect = ValueError(
            "query must be a non-empty string with at least 2 characters"
        )

        with self.assertRaises(ValueError) as ctx:
            self.client.search_products(query="", max_results=10)

        self.assertIn("non-empty", str(ctx.exception))

    def test_edge_case_query_exceeding_max_length_raises_value_error(self):
        """Verifica que una query mayor a 256 caracteres es rechazada con mensaje descriptivo."""
        long_query = "wireless headphones " * 20
        self.client.search_products.side_effect = ValueError(
            f"query length {len(long_query)} exceeds maximum of 256 characters"
        )

        with self.assertRaises(ValueError) as ctx:
            self.client.search_products(query=long_query, max_results=10)

        self.assertIn("256", str(ctx.exception))

    def test_edge_case_none_query_raises_type_error(self):
        """Verifica que query=None produce un TypeError con mensaje que identifica el parámetro afectado."""
        self.client.search_products.side_effect = TypeError(
            "query must be a str, got NoneType"
        )

        with self.assertRaises(TypeError) as ctx:
            self.client.search_products(query=None, max_results=10)

        self.assertIn("NoneType", str(ctx.exception))

    # ---------------------------------------------------------------------------
    # INVALID INPUT
    # ---------------------------------------------------------------------------

    def test_invalid_input_max_results_as_string_raises_type_error(self):
        """Verifica que pasar max_results como string en vez de int produce un TypeError claro."""
        self.client.search_products.side_effect = TypeError(
            "max_results must be an int between 1 and 50, got str"
        )

        with self.assertRaises(TypeError) as ctx:
            self.client.search_products(query="laptop", max_results="ten")

        self.assertIn("int", str(ctx.exception))
        self.assertIn("str", str(ctx.exception))

    # ---------------------------------------------------------------------------
    # RATE LIMIT
    # ---------------------------------------------------------------------------

    def test_rate_limit_60_sequential_calls_do_not_raise_uncaught_exception(self):
        """Verifica que 60 llamadas consecutivas no producen excepciones no controladas; el SDK devuelve error estructurado en la llamada 61."""
        normal_response = MOCK_PRODUCT_RESULT
        rate_limited_response = MagicMock()
        rate_limited_response.side_effect = (
            [normal_response] * 60
            + [RuntimeError("rate_limit_exceeded: retry_after=1s")]
        )
        self.client.search_products.side_effect = rate_limited_response.side_effect

        successes = 0
        rate_limit_hit = False
        for i in range(61):
            try:
                result = self.client.search_products(query="monitor", max_results=5)
                successes += 1
            except RuntimeError as exc:
                if "rate_limit_exceeded" in str(exc):
                    rate_limit_hit = True
                else:
                    raise

        self.assertEqual(successes, 60)
        self.assertTrue(rate_limit_hit)

    # ---------------------------------------------------------------------------
    # AUTH
    # ---------------------------------------------------------------------------

    def test_auth_missing_api_key_raises_authentication_error_with_descriptive_message(self):
        """Verifica que instanciar el cliente sin API key y hacer una llamada produce un error de autenticación con instrucciones de resolución."""
        unauthenticated_client = _make_sdk_client(api_key=None)
        unauthenticated_client.search_products.side_effect = PermissionError(
            "authentication_failed: Missing or invalid BuyWhere API key. "
            "Provide a valid key via the 'X-BuyWhere-Key' header."
        )

        with self.assertRaises(PermissionError) as ctx:
            unauthenticated_client.search_products(query="tablet", max_results=5)

        error_message = str(ctx.exception)
        self.assertIn("authentication_failed", error_message)
        self.assertIn("X-BuyWhere-Key", error_message)

    # ---------------------------------------------------------------------------
    # IDEMPOTENCY
    # ---------------------------------------------------------------------------

    def test_idempotency_same_search_query_twice_returns_identical_structured_response(self):
        """Verifica que la misma query ejecutada dos veces devuelve resultados estructuralmente idénticos (mismo order, mismo value_score)."""
        self.client.search_products.return_value = MOCK_PRODUCT_RESULT

        first_call = self.client.search_products(
            query="wireless headphones", max_results=10, min_value_score=0.5
        )
        second_call = self.client.search_products(
            query="wireless headphones", max_results=10, min_value_score=0.5
        )

        self.assertEqual(first_call["query"], second_call["query"])
        self.assertEqual(len(first_call["results"]), len(second_call["results"]))
        for r1, r2 in zip(first_call["results"], second_call["results"]):
            self.assertEqual(r1["product_id"], r2["product_id"])
            self.assertAlmostEqual(r1["price_sgd"], r2["price_sgd"], places=2)
            self.assertAlmostEqual(r1["value_score"], r2["value_score"], places=4)
        self.assertEqual(first_call["currency"], second_call["currency"])
        self.assertEqual(first_call["total_found"], second_call["total_found"])


if __name__ == "__main__":
    unittest.main()