import unittest
import app

class TestColorConversions(unittest.TestCase):

    def test_color_to_rgb(self):
        self.assertEqual(app.color_to_rgb('FFFFFF'), (255, 255, 255))
        self.assertEqual(app.color_to_rgb('000000'), (0, 0, 0))
        self.assertEqual(app.color_to_rgb('FF0000'), (255, 0, 0))
        # Add more test cases as needed

    def test_rgb_to_hex(self):
        self.assertEqual(app.rgb_to_hex((255, 255, 255)), '#ffffff')
        self.assertEqual(app.rgb_to_hex((0, 0, 0)), '#000000')
        self.assertEqual(app.rgb_to_hex((255, 0, 0)), '#ff0000')
        # Add more test cases as needed


if __name__ == '__main__':
    unittest.main()
