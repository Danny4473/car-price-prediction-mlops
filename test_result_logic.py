
import unittest
from result_logic import calculate_average


class TestCarPriceLogic(unittest.TestCase):

    def test_average_of_prices(self):
        prices = [100000, 200000, 300000]
        self.assertEqual(calculate_average(prices), 200000)

    def test_single_price(self):
        self.assertEqual(calculate_average([150000]), 150000)

    def test_empty_prices(self):
        with self.assertRaises(ValueError):
            calculate_average([])

    def test_negative_price(self):
        with self.assertRaises(ValueError):
            calculate_average([100000, -50000])


if __name__ == "__main__":
    unittest.main()
