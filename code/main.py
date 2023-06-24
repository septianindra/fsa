import cv2
import numpy as np
 
img = cv2.imread('./raw_data/train/dcom')
# print(img.shape) # Print image shape
shwimg= cv2.imshow("original", img)
 
# Cropping an image
cropped_image = img[410:1814, 0:1404]
 
# Display cropped image
shwimg2=cv2.imshow("cropped", cropped_image)

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
  
cv2.imshow('Grayscale', gray_image)

clahe = cv2.createCLAHE(clipLimit = 4)
final_img = clahe.apply(gray_image) + 8

cv2.imshow('Clahe', final_img)
# Save the cropped image
cv2.imwrite("Clahe.jpg", final_img)
 
cv2.waitKey(0)
cv2.destroyAllWindows()
