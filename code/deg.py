import cv2
import numpy as np

# Create a blank image
image = np.zeros((1000, 1000, 3), dtype=np.uint8)

# Define the coordinates
coord1 = [596, 347]
coord2 = [488, 477]
coord3 = [596, 887]

# Draw the lines on the image
cv2.line(image, tuple(coord1), tuple(coord2), (0, 0, 255), 2)
cv2.line(image, tuple(coord2), tuple(coord3), (0, 255, 0), 2)

# Calculate the angle between the lines
line1 = np.array(coord1) - np.array(coord2)
line2 = np.array(coord3) - np.array(coord2)
dot_product = np.dot(line1, line2)
norm_product = np.linalg.norm(line1) * np.linalg.norm(line2)
angle = np.arccos(dot_product / norm_product)
angle_degrees = np.degrees(angle)

# Display the image and print the angle
cv2.imshow('Lines', image)
print('Angle between the lines: {:.2f} degrees'.format(angle_degrees))
cv2.waitKey(0)
cv2.destroyAllWindows()
