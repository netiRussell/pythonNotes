import os, cv2

# Resizing
img = cv2.imread(os.path.join(".", "img", "multipleDisconected.png"))

print(img.shape)
resized_img = cv2.resize(img, (498, 342))

# Crop
cropped_img = img[500:600, 30:600] # keep height from : to, keep width from : to 

cv2.imshow("Operation example", cropped_img)
cv2.waitKey(0)