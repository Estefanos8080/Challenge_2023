import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Load the ChestMNIST data file
data = np.load('chestmnist.npz')

# Retrieve one image from the training set
image = data['train_images'][0]

# Display the image using matplotlib
plt.title('Random image from the training set')
plt.imshow(image, cmap='gray')
plt.show()

# Save the image as a grayscale PNG
plt.imsave('Estefanos_kebebew.png', image, cmap='gray')


# Invert the contrast of the image
inverted_image = np.invert(image)

plt.title('Inverted image')
plt.imshow(inverted_image, cmap='gray')
plt.show()

# it can be aslo done by subtracting the image from 255
# inverted_image = 255 - image

plt.imsave('inverted_Estefanos_Kebebew.png', inverted_image, cmap='gray')

# last part is to apply any filter of my choice
# I will use the Gaussian filter. According the research I did, it is the best filter for removing noise from images
# Gaussian filter is a low pass filter that removes high frequency components from the image

# Gaussian filter
gaussian_image = gaussian_filter(image, sigma=1.5)

# Display the image using matplotlib
plt.title('Gaussian image')
plt.imshow(gaussian_image, cmap='gray')
plt.show()

# Save the image as a grayscale PNG
plt.imsave('Gaussian_Estefanos_Kebebew.png', gaussian_image, cmap='gray')