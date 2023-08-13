import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def load_image(path):
    return img.imread(path)

def plot_image(image, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.show()

def kmeans_quantization(image, num_colors=32, initial_centroids=None):
    pixels = image.reshape(-1, 3)
    if initial_centroids is not None:
        kmeans = KMeans(n_clusters=num_colors, init=initial_centroids, random_state=0)
    else:
        kmeans = KMeans(n_clusters=num_colors, random_state=0)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.predict(pixels)
    quantized_pixels = colors[labels]
    quantized_image_array = quantized_pixels.reshape(image.shape)
    return np.uint8(quantized_image_array), kmeans.cluster_centers_

def calculate_metrics(original_image, processed_image):
    psnr = peak_signal_noise_ratio(original_image, processed_image)
    ssim = structural_similarity(original_image, processed_image, multichannel=True, channel_axis=2)
    mse = mean_squared_error(original_image, processed_image)
    print(f"PSNR: {psnr}\nSSIM: {ssim}\nMSE: {mse}")
    return psnr, ssim, mse

