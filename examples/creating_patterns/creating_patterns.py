import os
import numpy as np
import matplotlib.pyplot as plt
from pyblenderSDIC.patterns import create_speckle_BW_image

# Set the parameters for the speckle image
image_shape = (512, 512)
grain_size = (5.0, 5.0)
grain_size_sigma = (1.0, 1.0)
density = 0.8
seed = 42

# Generate the speckle image with specified parameters
speckle_img = create_speckle_BW_image(
    image_shape=image_shape,
    grain_size=grain_size,
    grain_size_sigma=grain_size_sigma,
    density=density,
    seed=seed
)

# Normalize the image data to range [0, 1] for display and saving
img_norm = speckle_img / np.max(speckle_img)

# Display the speckle image with grayscale colormap and no axis
plt.imshow(img_norm, cmap='gray')
plt.title("Speckle Pattern")
plt.axis('off')
plt.show()

# Determine the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the file paths for saving the images
png_path = os.path.join(current_dir, "speckle_image.png")
tiff_path = os.path.join(current_dir, "speckle_image.tiff")

# Save the normalized image as PNG
plt.imsave(png_path, img_norm, cmap='gray')

# Save the normalized image as TIFF
plt.imsave(tiff_path, img_norm, cmap='gray')

# Print confirmation of saved files
print(f"Images saved at:\n - {png_path}\n - {tiff_path}")
