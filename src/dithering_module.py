import numpy as np

def floyd_steinberg_dithering(img, num_colors):
    img = img / 255
    for ir in range(img.shape[0]):
        for ic in range(img.shape[1]):
            old_val = img[ir, ic].copy()
            # print("Old Value",old_val)
            new_val = np.round(old_val * (num_colors-1)) / (num_colors-1)
            # print("New Value",new_val)
            img[ir, ic] = new_val
            err = old_val - new_val
            # print("Error",err)
            if ic < img.shape[1] - 1:
                img[ir, ic + 1] += np.clip(err * 7 / 16, 0, 255)
            if ir < img.shape[0] - 1:
                if ic > 0:
                    img[ir + 1, ic - 1] += np.clip(err * 3 / 16, 0, 255)
                img[ir + 1, ic] += np.clip(err * 5 / 16, 0, 255)
                if ic < img.shape[1] - 1:
                    img[ir + 1, ic + 1] += np.clip(err / 16, 0, 255)

    img = np.array((img)*255, dtype=np.uint8)
    return img
