import unittest
import pandas as pd
import numpy as np
from workflow import clean_data

# Unit test class
class TestCleanData(unittest.TestCase):
    def test_clean_data(self):
        # Create a sample dataframe
        df = pd.DataFrame({
            'col1': ['1', '2', '3\t', '?', '5'],  # mix of numbers, tab, '?', and string
            'col2': ['10\t', '?', '20', '30', '40'],  # numeric strings with tabs and '?'
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })

        # Expected output (col1 and col2 inferred as object due to mixed types)
        expected = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, np.nan, 5.0],  # float for numbers, NaN for missing, string retained
            'col2': [10.0, np.nan, 20.0, 30.0, 40.0],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })

        # Apply clean_data
        result = clean_data(df)

        # Assert result equals expected
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

# Run the tests
if __name__ == '__main__':
    unittest.main()
