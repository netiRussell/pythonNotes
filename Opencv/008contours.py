import os, cv2

font = cv2.FONT_HERSHEY_COMPLEX 

# Retrieve and prepare an image for better precision
img = cv2.imread(os.path.join(".", "img", "multipleDisconected.png"))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_adaptThreshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 9)

# Finding contours
contours, hierarchy = cv2.findContours(img_adaptThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for obj in contours:
  # Smoothes out the points
  peri = cv2.arcLength(obj, True)
  approx = cv2.approxPolyDP(obj, 0.02 * peri, True)

  cv2.drawContours(img, obj, -1, (0, 255, 0), 2)

  # transform 2+ dimensional array into 1d 
  n = approx.ravel() 
  i = 0

  # Check for redundant data,
  # If are < 0, skip points plotting
  area = cv2.contourArea(obj, True)
  if( area < 0): 
     continue

  for j in n : 
      if(i % 2 == 0): 
          x = n[i] 
          y = n[i + 1] 

          # String containing the co-ordinates. 
          string = str(x) + " " + str(y)  

          # text on remaining co-ordinates. 
          cv2.putText(img, string, (x+2, y-5),  
                    font, 0.8, (0, 255, 0))  
      i = i + 1


cv2.imshow("Contours example", img)
cv2.waitKey(0)