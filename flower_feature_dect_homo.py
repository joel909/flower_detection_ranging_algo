import cv2
import numpy as np
import matplotlib.pyplot as plt

#################################IMP#########################
# Define paths for main and template images CHANGE IT WHILE RUNNING ON THE PI
main_image_path = 'C:/Users/joelj/Downloads/Flower_detection_algorithm/content/main.jpeg'
template_image_path = 'C:/Users/joelj/Downloads/Flower_detection_algorithm/content/template_image.jpeg'
##########################


#load main and template images
main_image = cv2.imread(main_image_path)
template = cv2.imread(template_image_path)

#check if images loaded successfully
if main_image is None:
    raise ValueError(f"Could not load the main image from path: {main_image_path}")
if template is None:
    raise ValueError(f"Could not load the template image from path: {template_image_path}")

#convert images to grayscale 
main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector fun the descriptrs 
orb = cv2.ORB_create()

# detect keypoints and compute descriptors
keypoints1, descriptors1 = orb.detectAndCompute(main_image_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(template_gray, None)

#check if descriptors are valid no error will be there in this point 
if descriptors1 is None or descriptors2 is None:
    raise ValueError("Descriptors cannot be None")
#initilaisaton of bf matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

print(f"Number of keypoints in main image: {len(keypoints1)}")
print(f"Number of keypoints in template image: {len(keypoints2)}")
print(f"Number of matches found: {len(matches)}")

valid_matches = []
for match in matches:
    if match.trainIdx < len(keypoints2) and match.queryIdx < len(keypoints1):
        valid_matches.append(match)
    else:
        print(f"Invalid match found: trainIdx={match.trainIdx}, queryIdx={match.queryIdx}")

if len(valid_matches) == 0:
    raise ValueError("No valid matches found.")

num_matches_to_draw = min(10, len(valid_matches))

# check if there are enough matches to draw
if num_matches_to_draw == 0:
    raise ValueError("Not enough matches to draw.")

# matches and show image
try:
    match_img = cv2.drawMatches(template, keypoints2, main_image, keypoints1, valid_matches[:num_matches_to_draw], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(match_img)
    plt.show()
except Exception as e:
    print(f"Error during drawing matches: {e}")

# getting the  matched keypoints
src_pts = np.float32([keypoints2[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)

# FIND THE homography matrix
try:
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None or mask is None:
        raise ValueError("Homography could not be computed.")
    matchesMask = mask.ravel().tolist()
    print(f"Homography matrix:\n{M}")
except Exception as e:
    print(f"Error during computing homography: {e}")

# Get dimensions of template image
h, w = template.shape[:2]

# Define template corner points
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

# Perform perspective transformation
try:
    dst = cv2.perspectiveTransform(pts, M)
    print(f"Transformed points: {dst}")
    if dst is not None and len(dst) == 4:
        # Draw bounding box on main image
        main_image_with_box = cv2.polylines(main_image.copy(), [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        plt.imshow(main_image_with_box)
        plt.show()
    else:
        print("Homography transformation failed.")
except Exception as e:
    print(f"Error during perspective transformation: {e}")
