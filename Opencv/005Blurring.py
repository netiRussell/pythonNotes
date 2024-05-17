import os, cv2

# Blurring
img = cv2.imread(os.path.join(".", "img", "puppies.jpg"))
img_blur = cv2.blur(img, (10,10))
img_gBlur = cv2.GaussianBlur(img, (7,7), 3)
img_meadian = cv2.medianBlur(img, 5)

cv2.imshow("Blur example", img_meadian)
cv2.waitKey(0)