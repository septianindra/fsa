import cv2
import numpy as np

# Step 1: Detect edges of femoral shaft using Canny
img = cv2.imread('./raw_data/train/6.jpg', 0)  # Load image in grayscale
edges = cv2.Canny(img, 100, 250)

# Define the dimensions of the image
height = 1404
width = 1404

# Create a black image
white_img = np.ones((height, width, 3), np.uint8)*255

# Step 2: Divide femoral shaft contour into small segments
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
femoral_contour = max(contours, key=cv2.contourArea)  # Select the largest contour

cv2.drawContours(img, contours, -1, (0,255,0), 1)

cnt = contours[1]
rows,cols = white_img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty1 = int((-x*vy/vx) + y)
righty1 = int(((cols-x)*vy/vx)+y)

cnt = contours[2]

rows,cols = white_img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty2 = int((-x*vy/vx) + y)
righty2 = int(((cols-x)*vy/vx)+y)

mid_lefty = (lefty2+lefty1)/2
mid_righty = (righty2+righty1)/2

# Display the result
cv2.namedWindow('edges', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('edges', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.imshow('edges', white_img)
cv2.line(white_img,(cols-1,int(mid_righty)),(0,int(mid_lefty)),(0,255,0),1)

# Display the result
cv2.namedWindow('edges', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('edges', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.imshow('edges', white_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
