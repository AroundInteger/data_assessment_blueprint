import unittest
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import PHASES, DATA_TYPE_GUIDANCE

class TestMain(unittest.TestCase):
    def test_phases_exist(self):
        """Test that all expected phases exist"""
        expected_phases = ["Overview", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5"]
        self.assertEqual(list(PHASES.keys()), expected_phases)

    def test_data_types_exist(self):
        """Test that all expected data types exist"""
        expected_types = ["Temporal (Race Measurements)", "Spatiotemporal (Sports Motion)", "Fluorescence (Biological Timelapse)"]
        self.assertEqual(list(DATA_TYPE_GUIDANCE.keys()), expected_types)

if __name__ == '__main__':
    unittest.main() 