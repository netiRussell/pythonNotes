import os, cv2

# Convert BGR to RGB colorspace
img = cv2.imread(os.path.join(".", "img", "puppies.jpg"))
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("Operation example", rgb_img)
cv2.waitKey(0)
cv2.imshow("Operation example", hsv_img)
cv2.waitKey(0)
cv2.imshow("Operation example", gray_img)
cv2.waitKey(0)