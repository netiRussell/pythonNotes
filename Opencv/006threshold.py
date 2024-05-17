import os, cv2

img = cv2.imread(os.path.join(".", "img", "puppies.jpg"))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Static thresholding
ret, img_threshold = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY) # cv2_img, value of a pixel below which convert to the number mentioned next, this number, threshold to what
img_threshold = cv2.blur(img_threshold, (10,10))

# Adaptive thresholding
img_adaptThreshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 7) # cv2_img, value to threshold, some other arguments...

cv2.imshow("Threshold example", img_adaptThreshold)
cv2.waitKey(0)