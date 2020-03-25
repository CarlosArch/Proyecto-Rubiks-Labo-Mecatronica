import cv2
import numpy as np
from imutils import contours

image = cv2.imread('rubik.jpg')
original = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = np.zeros(image.shape, dtype=np.uint8)

colors = {
    'naranja': ([0, 219.75, 191.25], [16.17, 245.26, 216.75]),
    'verde': ([68.77, 162.282, 105.24], [86.67, 187.782, 130.7385]),
    'blanco': ([15.90, 0, 181.254], [33.80, 20.63, 206.75]),
    'amarillo': ([13.38, 180.18, 213.26], [31.28, 205.68, 238.76]), 
    'azul': ([108.60, 141.17, 98.25], [126.50, 166.67, 123.75])} 

# Color threshold to find the squares
open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
for color, (lower, upper) in colors.items():
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    color_mask = cv2.inRange(image, lower, upper)

    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=5)

    color_mask = cv2.merge([color_mask, color_mask, color_mask])
    mask = cv2.bitwise_or(mask, color_mask)

gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# cv2.waitKey()
cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# Sort all contours from top-to-bottom or bottom-to-top
(cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

# Take each row of 3 and sort from left-to-right or right-to-left
cube_rows = []
row = []
for (i, c) in enumerate(cnts, 1):
    row.append(c)
    if i % 3 == 0:
        print()
        print(row)
        print(type(row))
        print(len(row))
        (cnts, _) = contours.sort_contours(row, method="left-to-right")
        cube_rows.append(cnts)
        row = []

# Draw text
number = 0
for row in cube_rows:
    for c in row:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)

        cv2.putText(original, "#{}".format(number + 1), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        number += 1

# cv2.imshow('mask', mask)
# cv2.imwrite('mask.png', mask)
# cv2.imshow('original', original)
# cv2.waitKey()