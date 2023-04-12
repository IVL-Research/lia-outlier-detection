import numpy as np
import unittest
from data_chooser import interactive_data_chooser as idc
import pandas as pd

df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [6, 7, 8, 9, 10]})
columns = ['x', 'y']
chooser = idc(df, columns)

class TestRemoveSelectedDataPoints(unittest.TestCase):
    def test_remove_selected_data_points(self):
        # Test case 1: Remove a single point
        current_list_x = np.array([1, 2, 3, 4, 5])
        current_list_y = np.array([6, 7, 8, 9, 10])
        points = MockPoint(2)  # Point at index 2
        result_x, result_y = chooser.remove_selected_data_points(current_list_x, current_list_y, points)
        expected_x = np.array([1, 2, 4, 5])
        expected_y = np.array([6, 7, 9, 10])
        np.testing.assert_array_equal(result_x, expected_x)
        np.testing.assert_array_equal(result_y, expected_y)
        
        # Test case 2: Remove multiple points
        current_list_x = np.array([1, 2, 3, 4, 5])
        current_list_y = np.array([6, 7, 8, 9, 10])
        points = MockPoint([1, 3])  # Points at indices 1 and 3
        result_x, result_y = chooser.remove_selected_data_points(current_list_x, current_list_y, points)
        expected_x = np.array([1, 3, 5])
        expected_y = np.array([6, 8, 10])
        np.testing.assert_array_equal(result_x, expected_x)
        np.testing.assert_array_equal(result_y, expected_y)
        
        # Test case 3: Remove all points
        current_list_x = np.array([1, 2, 3, 4, 5])
        current_list_y = np.array([6, 7, 8, 9, 10])
        points = MockPoint([0, 1, 2, 3, 4])  # All points
        result_x, result_y = chooser.remove_selected_data_points(current_list_x, current_list_y, points)
        expected_x = np.array([])
        expected_y = np.array([])
        np.testing.assert_array_equal(result_x, expected_x)
        np.testing.assert_array_equal(result_y, expected_y)
        
        # Test case 4: Empty input lists
        current_list_x = np.array([])
        current_list_y = np.array([])
        points = MockPoint([0, 1, 2, 3, 4])  # All points
        result_x, result_y = chooser.remove_selected_data_points(current_list_x, current_list_y, points)
        expected_x = np.array([])
        expected_y = np.array([])
        np.testing.assert_array_equal(result_x, expected_x)
        np.testing.assert_array_equal(result_y, expected_y)
        
class MockPoint:
    def __init__(self, point_inds):
        self.point_inds = point_inds

if __name__ == '__main__':
    unittest.main()
