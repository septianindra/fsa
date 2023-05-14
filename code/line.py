import numpy as np
import cv2

# Define the dimensions of the image
height = 500
width = 500

# Create a black image
img = np.zeros((height, width, 3), np.uint8)

# Define the coordinates of the two parallel lines B and C
lineB_start = (50, 100)
lineB_end = (450, 100)
lineC_start = (50, 400)
lineC_end = (450, 400)

# Calculate the distance between lines B and C
distance = lineC_start[1] - lineB_start[1]

# Calculate the angle of the lines in radians
angle = np.deg2rad(30)

# Calculate the length of the line A
length = distance / np.cos(angle)

# Calculate the midpoint of line B
midpointB = ((lineB_start[0] + lineB_end[0]) // 2, (lineB_start[1] + lineB_end[1]) // 2)

# Calculate the midpoint of line C
midpointC = ((lineC_start[0] + lineC_end[0]) // 2, (lineC_start[1] + lineC_end[1]) // 2)

# Calculate the starting point of line A
start_point = (midpointB[0] - int(length/2 * np.sin(angle)), midpointB[1] + int(length/2 * np.cos(angle)))

# Calculate the ending point of line A
end_point = (midpointC[0] + int(length/2 * np.sin(angle)), midpointC[1] - int(length/2 * np.cos(angle)))

# Draw lines B, C, and A on the image
cv2.line(img, lineB_start, lineB_end, (255, 0, 0), 2)
cv2.line(img, lineC_start, lineC_end, (0, 255, 0), 2)
cv2.line(img, start_point, end_point, (0, 0, 255), 2)

# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
