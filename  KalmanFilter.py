import cv2
import numpy as np

# Create Kalman filter
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32) * 0.03

# Initial state [x, y, vx, vy]
state = np.array([0, 0, 0, 0], np.float32).reshape(-1, 1)

# Video capture
cap = cv2.VideoCapture('car_Moving.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV => color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a red color range in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors , create masks with color range
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour (assuming it's the red car)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Update the measurement based on the center of the bounding box
        measurement = np.array([x + w / 2, y + h / 2], np.float32).reshape(-1, 1)

        # Correct the state using the measurement
        state = kf.correct(measurement)

        # Predict the next state
        prediction = kf.predict()

        # Draw the predicted state on the frame
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 5, (0, 255, 0), -1)

        # Draw the bounding box of the red car
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
