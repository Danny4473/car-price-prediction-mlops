
import os
import unittest
import pandas as pd

DATASET_PATH = "Used_Car_Price_Feature_Engineered.csv"


class TestCarPriceDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(
                f"Dataset not found: {DATASET_PATH}. "
                "Place the CSV file in the current working directory."
            )

        cls.df = pd.read_csv(DATASET_PATH)

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists(DATASET_PATH))

    def test_dataset_not_empty(self):
        self.assertGreater(len(self.df), 0)

    def test_target_column_exists(self):
        self.assertIn("Selling_Price", self.df.columns)

    def test_target_has_valid_values(self):
        self.assertTrue(self.df["Selling_Price"].notna().all())


# Run tests safely in Jupyter and as a Python script
if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
