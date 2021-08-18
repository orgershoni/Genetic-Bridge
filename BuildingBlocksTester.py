import unittest
import BuildingBlocksGenetator


class MyTestCase(unittest.TestCase):

    def test_something(self):
        triangle = BuildingBlocksGenetator.BuildingBlock()
        triangle.generate_triangle(90, 5, 3)
        self.assertEqual(triangle.get_edges(), [5, 3, 4])
        triangle.get_polygon()


if __name__ == '__main__':
    unittest.main()
