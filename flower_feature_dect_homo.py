import cv2
import numpy as np

main_image_path = 'C:/Users/joelj/Downloads/Flower_detection_algorithm/content/main.jpeg'
template_image_path = 'C:/Users/joelj/Downloads/Flower_detection_algorithm/content/template_image.jpeg'

main_image = cv2.imread(main_image_path)
template = cv2.imread(template_image_path)

if main_image is None:
    raise ValueError(f"Could not load the main image from path: {main_image_path}")
if template is None:
    raise ValueError(f"Could not load the template image from path: {template_image_path}")

#main_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#main_image_gray__diffrence  = cv2.bitwise_xor(img_gray,img_gray_mode)
#cv2.imshow('main_image_gray__diffrence', main_image_gray__diffrence )

main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#cv2.imshow('main_image_gray', main_image_gray )

orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(main_image_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(template_gray, None)

if descriptors1 is None or descriptors2 is None:
    raise ValueError("Descriptors cannot be None")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

print(f"Number of keypoints in main image: {len(keypoints1)}")
print(f"Number of keypoints in template image: {len(keypoints2)}")
print(f"Number of matches found: {len(matches)}")

#Finding valid keypoints we are looking for ... 
valid_matches = []

for match in matches:
    
    if match.trainIdx < len(keypoints2) and match.queryIdx < len(keypoints1):
        #print(len(match.trainIdx))
        valid_matches.append(match)
        #print(f"Valid match found: trainIdx={match.trainIdx}, queryIdx={match.queryIdx}")
    else:
        print(f"Invalid match found: trainIdx={match.trainIdx}, queryIdx={match.queryIdx}")