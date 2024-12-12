# test_calculator.py
import unittest
from calculator import add, subtract, multiply, divide


class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, -1), -2)
        self.assertEqual(add(0, 0), 0)
        self.assertAlmostEqual(add(0.1, 0.2), 0.3, places=1)  # 浮点数

    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)
        self.assertEqual(subtract(-1, -1), 0)
        self.assertEqual(subtract(0, 5), -5)
        self.assertAlmostEqual(subtract(0.3, 0.1), 0.2, places=1)  # 浮点数

    def test_multiply(self):
        self.assertEqual(multiply(2, 3), 6)
        self.assertEqual(multiply(-1, -1), 1)
        self.assertEqual(multiply(0, 5), 0)
        self.assertAlmostEqual(multiply(0.1, 0.2), 0.02, places=2)  # 浮点数

    def test_divide(self):
        self.assertEqual(divide(6, 3), 2)
        self.assertEqual(divide(-6, -2), 3)
        with self.assertRaises(ValueError):
            divide(1, 0)  # 分母为零
        self.assertAlmostEqual(divide(0.3, 0.1), 3.0, places=1)  # 浮点数


if __name__ == "__main__":
    unittest.main()
