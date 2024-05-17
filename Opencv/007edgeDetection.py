import os, cv2
import numpy as np

img = cv2.imread(os.path.join(".", "img", "puppies.jpg"))

# Canny Edge detection
img_canny = cv2.Canny(img, 100, 200)

img_dilate = cv2.dilate(img_canny, np.ones((5,5), dtype=np.int8))

cv2.imshow("Edge detection example", img_dilate)
cv2.waitKey(0)