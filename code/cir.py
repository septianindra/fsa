import cv2
import numpy as np

# Load the input image
img = cv2.imread('./raw_data/train/5.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to the image to reduce noise
img_blur = cv2.GaussianBlur(img, (5,5), 0)

# Apply Hough Circle Transform to detect circles in the image
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

mask = np.zeros(img.shape[:2], dtype=np.uint8)

# Draw detected circles on the original image
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        # cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        cv2.circle(mask, (x, y), r+5, (255, 255, 255),-1)
        
result = cv2.bitwise_or(img, mask)

cv2.imwrite("femur_image.jpg", result)

# if circles is not None:
#     circles = np.round(circles[0, :]).astype("int")
#     for (x, y, r) in circles:
#         cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

# Show the result
cv2.imshow("Output", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get the center point of the detected circle
# if circles is not None:
#     center = (circles[0][0], circles[0][1])
#     print("Center point of circle:", center)
