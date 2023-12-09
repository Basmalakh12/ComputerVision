
import cv2
import numpy as np


# Read the input image
image = cv2.imread('photo3.jpg')

width, height = 280, 311

# Define source and destination points
source_points = np.float32([
    [51, 101],
    [217, 82],
    [103, 244],
    [265, 179]
])

destination_points = np.float32([
    [0, 0],
    [400, 0],
    [0, 400],
    [400, 400]
])

# Calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(source_points, destination_points)

# Warp the image using the perspective transform matrix
transformed_image = cv2.warpPerspective(image, M, (width, height))

# Display the original and transformed images
cv2.imshow('Original Image', image)
cv2.imshow('Transformed Image', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
