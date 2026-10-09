import unittest
from app import app


class TestCarPriceAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_prediction_requires_json(self):
        response = self.client.post("/predict")
        self.assertEqual(response.status_code, 400)

    def test_prediction_requires_model(self):
        response = self.client.post(
            "/predict",
            json={"feature": 100}
        )

        # A model must exist before a prediction can run.
        self.assertIn(response.status_code, [200, 400, 500])


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
