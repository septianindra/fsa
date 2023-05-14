import cv2
import numpy as np

# Load the image
img = cv2.imread('result.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection on the grayscale image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply Hough line transform on the edge-detected image
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Calculate the intersection point of the two detected lines
x1, y1, x2, y2 = lines[0][0]
m1 = (y2 - y1) / (x2 - x1)
b1 = y1 - m1 * x1

x3, y3, x4, y4 = lines[1][0]
m2 = (y4 - y3) / (x4 - x3)
b2 = y3 - m2 * x3

x = int((b2 - b1) / (m1 - m2))
y = int(m1 * x + b1)

# Draw the intersection point on the image
cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

# Display the image with the intersection point
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
