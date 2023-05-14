import cv2

# Load image
image = cv2.imread('./raw_data/train/6.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 100, 300)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

# Display the result
cv2.namedWindow('Contours', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Contours', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()