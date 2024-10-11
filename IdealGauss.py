# Consolidated script for generating a Gaussian surface, a circle, adding an artifact, and analyzing them using FFT

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

# Function to generate a 2D Gaussian surface
def generate_gaussian(size, sigma):
    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x*x + y*y)
    gaussian = np.exp(-(d**2 / (2.0 * sigma**2)))
    return gaussian

# Function to generate a circle with abrupt edges
def generate_circle(size, radius):
    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    x, y = np.meshgrid(x, y)
    circle = np.where(x**2 + y**2 <= radius**2, 1, 0)
    return circle

# Function to add an artifact to an image
def add_artifact(image, position, intensity, size):
    artifact_image = image.copy()
    x, y = position
    artifact_image[x-size:x+size, y-size:y+size] += intensity
    return artifact_image

# High-pass filter function
def high_pass_filter(fft_image, cutoff):
    rows, cols = fft_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0
    return fft_image * mask

# Parameters
size = 256
sigma = 20
radius = 50
artifact_position = (128, 128)  # Center of the image
artifact_intensity = 0.1  # Closer to the Gaussian surface intensity
artifact_size = 2  # Smaller size

# Generate images
gaussian_image = generate_gaussian(size, sigma)
circle_image = generate_circle(size, radius)

# Add artifact to the Gaussian image
artifact_gaussian_image = add_artifact(gaussian_image, artifact_position, artifact_intensity, artifact_size)

# Compute 2D FFT and IFFT for Gaussian
fft_gaussian = fftshift(fft2(gaussian_image))
ifft_gaussian = ifft2(fft_gaussian)

fft_artifact_gaussian = fftshift(fft2(artifact_gaussian_image))
ifft_artifact_gaussian = ifft2(fft_artifact_gaussian)

# Compute 2D FFT and IFFT for Circle
fft_circle = fftshift(fft2(circle_image))
ifft_circle = ifft2(fft_circle)

# Difference Analysis for Gaussian
fft_diff_gaussian = np.abs(fft_artifact_gaussian) - np.abs(fft_gaussian)

# High-pass Filtering for Gaussian
high_pass_original_gaussian = high_pass_filter(fft_gaussian, 20)
high_pass_artifact_gaussian = high_pass_filter(fft_artifact_gaussian, 20)

# Power Spectrum Analysis for Gaussian
power_spectrum_original_gaussian = np.abs(fft_gaussian)**2
power_spectrum_artifact_gaussian = np.abs(fft_artifact_gaussian)**2
power_spectrum_diff_gaussian = power_spectrum_artifact_gaussian - power_spectrum_original_gaussian

# Plotting
fig, axes = plt.subplots(4, 3, figsize=(15, 20))

# Original images
axes[0, 0].imshow(gaussian_image, cmap='gray')
axes[0, 0].set_title('Original Gaussian Image')
axes[0, 1].imshow(artifact_gaussian_image, cmap='gray')
axes[0, 1].set_title('Gaussian with Small Artifact')
axes[0, 2].imshow(circle_image, cmap='gray')
axes[0, 2].set_title('Circle Image')

# FFT images
axes[1, 0].imshow(np.log(np.abs(fft_gaussian) + 1), cmap='gray')
axes[1, 0].set_title('FFT of Original Gaussian')
axes[1, 1].imshow(np.log(np.abs(fft_artifact_gaussian) + 1), cmap='gray')
axes[1, 1].set_title('FFT of Gaussian with Artifact')
axes[1, 2].imshow(np.log(np.abs(fft_circle) + 1), cmap='gray')
axes[1, 2].set_title('FFT of Circle')

# IFFT images
axes[2, 0].imshow(np.abs(ifft_gaussian), cmap='gray')
axes[2, 0].set_title('IFFT of Original Gaussian')
axes[2, 1].imshow(np.abs(ifft_artifact_gaussian), cmap='gray')
axes[2, 1].set_title('IFFT of Gaussian with Artifact')
axes[2, 2].imshow(np.abs(ifft_circle), cmap='gray')
axes[2, 2].set_title('IFFT of Circle')

# Additional Analysis for Gaussian
axes[3, 0].imshow(np.log(np.abs(fft_diff_gaussian) + 1), cmap='gray')
axes[3, 0].set_title('FFT Difference (Gaussian)')
axes[3, 1].imshow(np.log(np.abs(high_pass_artifact_gaussian) + 1), cmap='gray')
axes[3, 1].set_title('High-pass Filtered Artifact (Gaussian)')
axes[3, 2].imshow(np.log(np.abs(power_spectrum_diff_gaussian) + 1), cmap='gray')
axes[3, 2].set_title('Power Spectrum Difference (Gaussian)')

plt.tight_layout()
plt.show()

# Calculate and print some statistics for Gaussian
print("Gaussian Image Analysis:")
print("Max FFT difference: " + str(np.max(fft_diff_gaussian)))
print("Mean FFT difference: " + str(np.mean(fft_diff_gaussian)))
print("Max power spectrum difference: " + str(np.max(power_spectrum_diff_gaussian)))
print("Mean power spectrum difference: " + str(np.mean(power_spectrum_diff_gaussian)))

# Calculate and print some statistics for Circle
print("\
Circle Image Analysis:")
print("Max FFT value: " + str(np.max(np.abs(fft_circle))))
print("Mean FFT value: " + str(np.mean(np.abs(fft_circle))))
print("Max power spectrum value: " + str(np.max(np.abs(fft_circle)**2)))
print("Mean power spectrum value: " + str(np.mean(np.abs(fft_circle)**2)))

