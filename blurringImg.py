import cv2


# Read the input image
img = cv2.imread('photo1.jpg')

#  median blurring
blurred_img = cv2.medianBlur(img, 5)

# Display the original and blurred images
cv2.imshow('Original Image', img)
cv2.imshow('Blurred Image', blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
