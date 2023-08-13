import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
import matplotlib.pyplot as plt

def compute_histogram(image):
    # Compute the histogram of each color channel of the image
    histogram_r = np.histogram(image[:,:,0].flatten(), bins=256)[0]
    histogram_g = np.histogram(image[:,:,1].flatten(), bins=256)[0]
    histogram_b = np.histogram(image[:,:,2].flatten(), bins=256)[0]
    return histogram_r, histogram_g, histogram_b

def progressive_histogram_quantization(image, desired_bins):
    # Apply PHQ to each color channel of the image
    histogram_r, histogram_g, histogram_b = compute_histogram(image)
    quantized_histogram_r = progressive_histogram_quantization_single_channel(histogram_r, desired_bins)
    quantized_histogram_g = progressive_histogram_quantization_single_channel(histogram_g, desired_bins)
    quantized_histogram_b = progressive_histogram_quantization_single_channel(histogram_b, desired_bins)
    return quantized_histogram_r, quantized_histogram_g, quantized_histogram_b

def progressive_histogram_quantization_single_channel(histogram, desired_bins):
    # Initialize min_val and min_idx
    min_val = np.min(histogram)
    min_idx = np.argmin(histogram)
    while np.count_nonzero(histogram) > desired_bins:
        # Use the stored min_val and min_idx
        K = min_idx
        # Find the first bin to the left of K
        L = K - 1
        # Find the first bin to the right of K
        R = K + 1 if K < len(histogram) - 1 else K
        # Merge bin K to bin R or L based on the conditions
        if histogram[R] >= histogram[K] and abs(K - R) <= abs(K - L):
            histogram[R] += histogram[K]
        else:
            histogram[L] += histogram[K]
        histogram[K] = 0
        # Update min_val and min_idx
        non_zero_histogram = histogram[np.nonzero(histogram)]
        if len(non_zero_histogram) > 0:
            min_val = np.min(non_zero_histogram)
            min_idx = np.nonzero(histogram == min_val)[0][0]
    return histogram

def kmeans_quantization(image, histogram_r, histogram_g, histogram_b, n_clusters):
    print(image.shape)
    print("Clusters:",n_clusters)
    reshaped_image = image.reshape(-1, 3)
    frequencies = np.zeros_like(reshaped_image, dtype=np.float64)
    for i in range(reshaped_image.shape[0]):
        r, g, b = reshaped_image[i]
        frequencies[i, 0] = histogram_r[r] / reshaped_image.shape[0]
        frequencies[i, 1] = histogram_g[g] / reshaped_image.shape[0]
        frequencies[i, 2] = histogram_b[b] / reshaped_image.shape[0]
    reshaped_image = np.hstack((reshaped_image, frequencies))
    kmeans = MiniBatchKMeans(n_clusters)
    kmeans.fit(reshaped_image)
    quantized_image = kmeans.cluster_centers_[kmeans.labels_]
    quantized_image = quantized_image[:, :3].reshape(image.shape).astype(np.uint8)
    return quantized_image, kmeans.cluster_centers_