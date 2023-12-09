import cv2
import numpy as np

# Read the input image
image = cv2.imread('photo2.jpg')

# Apply Canny edge detection
edges = cv2.Canny(image, 50, 150)

cv2.imshow('Original Image', image)
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()



