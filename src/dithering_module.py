import numpy as np


def floyd_steinberg_dithering(image):
    """
    Apply Floyd-Steinberg dithering to the image
    """
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            old_pixel = image[y][x].copy()  # copy the original pixel's value
            new_pixel = np.round(old_pixel / 255.0) * 255  # quantize pixel value to 0 or 255
            image[y][x] = new_pixel  # assign new value to the current pixel
            quant_error = old_pixel - new_pixel  # computess the quantization error

            # spread the quantization error to neighboring pixels
            if x < image.shape[1] - 1:
                image[y][x + 1] = np.clip(image[y][x + 1] + quant_error * 7 / 16, 0, 255)
            if y < image.shape[0] - 1:
                image[y + 1][x] = np.clip(image[y + 1][x] + quant_error * 5 / 16, 0, 255)
                if x > 0:
                    image[y + 1][x - 1] = np.clip(image[y + 1][x - 1] + quant_error * 3 / 16, 0, 255)
                if x < image.shape[1] - 1:
                    image[y + 1][x + 1] = np.clip(image[y + 1][x + 1] + quant_error * 1 / 16, 0, 255)
    return image

