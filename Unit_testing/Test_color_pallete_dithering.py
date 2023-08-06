import unittest
import numpy as np
import dithering_module
import color_palette
import app  

class TestColorPalette(unittest.TestCase):
    def test_color_to_rgb(self):
        self.assertEqual(app.color_to_rgb("#FFFFFF"), (255, 255, 255))  # change color_palette to app
        self.assertEqual(app.color_to_rgb("#000000"), (0, 0, 0))  # change color_palette to app
        self.assertEqual(app.color_to_rgb("#123456"), (18, 52, 86))  # change color_palette to app
        with self.assertRaises(ValueError):
            app.color_to_rgb("#12345")  # change color_palette to app


class TestDitheringModule(unittest.TestCase):
    def test_floyd_steinberg_dithering(self):
        input_image = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        expected_output_image = np.array([[0, 255], [255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(dithering_module.floyd_steinberg_dithering(input_image), expected_output_image)

if __name__ == '__main__':
    unittest.main()
