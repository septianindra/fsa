import cv2
import numpy as np

# Load the input image
img = cv2.imread('./raw_data/train/5.jpg', cv2.IMREAD_GRAYSCALE)
original_img = cv2.imread('./raw_data/train/5.jpg', cv2.IMREAD_GRAYSCALE)
ori = cv2.imread('./raw_data/train/5.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to the image to reduce noise
img_blur = cv2.GaussianBlur(img, (5,5), 0)

# Apply Hough Circle Transform to detect circles in the image
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=90, maxRadius=0)

# Adding masking
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# Draw detected circles on the original image
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(ori, (x, y), r, (0, 255, 0), 2)
        cv2.circle(ori, (x, y), 2, (0, 0, 255), 3)
        cv2.circle(mask, (x, y), r+16, (255, 255, 255),-1)

result = cv2.bitwise_or(img, mask)

cv2.imwrite("femur_image.jpg", result)

# Apply Canny edge detection to find edges
edges = cv2.Canny(mask, 50, 150)

# Find the contours of the mask
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Initialize a list to store the edge points
edge_points = []

# Loop over each contour
for contour in contours:
    for vertex in contour:
        x, y = vertex[0]
        edge_points.append((x, y))

# Print the edge points
# print(edge_points)

for (x, y) in edge_points:
    cv2.circle(result, (x,y), radius=0, color=(0, 0, 255), thickness=-1)


for (x, y) in edge_points:
    new_center = cv2.circle(original_img, (x,y), radius=0, color=(0, 0, 255), thickness=-1)

circle_points = []

for i in range(1404):
    for j in range(1404):
        if (new_center[i][j]==0):
            circle_points.append((i, j))

arc_result = []
for (x, y) in circle_points:
    if(ori[x][y]!=255):
        arc_result.append((x,y))
        
for (x, y) in arc_result:
    cv2.circle(ori, (y,x), radius=1, color=(0, 0, 255), thickness=-1)


# Define the starting and ending points of the line
start_point = (arc_result[0][1],arc_result[0][0])
end_point = (arc_result[-1][1],arc_result[-1][0])

# Draw a line on the image
color = (0, 0, 255) # Red color
thickness = 2
cv2.line(ori, start_point, end_point, color, thickness)

# Calculate the middle point of the line
middle_point = ((start_point[0]+end_point[0])//2, (start_point[1]+end_point[1])//2)

# Define the two endpoints of the line
pt1 = (596, 347)
pt2 = (664, 264)

# Calculate the slope of the line
m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

# Calculate the y-intercept of the line
b = pt1[1] - m * pt1[0]

# Determine the x-coordinates where the line intersects with the edges of the frame
x1 = 0
y1 = m * x1 + b

x2 = 1040
y2 = m * x2 + b

# Draw the new line
cv2.line(ori, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Display the image with the line and the middle point
cv2.circle(ori, middle_point, 3, (0, 255, 0), -1)
cv2.imshow("Image with line and middle point", ori)
cv2.waitKey(0)
cv2.destroyAllWindows()