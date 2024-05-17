import cv2
import os

# Image read
img_path = os.path.join(".", "img", "multipleDisconected.png")
img = cv2.imread(img_path)

# Image write
newImg_path = os.path.join(".", "img", "OUTmultipleDisconected.png")
cv2.imwrite(newImg_path, img)

# Show image
cv2.imshow("Frame name", img)
cv2.waitKey(0) # required line, without it the window closes immediately. 0 = wait 0ms after any key is pressed before you close the window