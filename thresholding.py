
import cv2
import numpy as np

# Read the grayscale image
img = cv2.imread('photo4.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold the grayscale image using Otsu's method
thresh, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Apply adaptive thresholding using MeanShift
adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

# Display the original and thresholded images
cv2.imshow("Original Grayscale Image", img)
cv2.imshow("Simple Thresholding", bw)
cv2.imshow("Adaptive Thresholding", adaptive_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
