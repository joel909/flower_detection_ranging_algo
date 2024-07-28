import cv2
import numpy as np
import matplotlib.pyplot as plt

#################################IMP#########################
# Define paths for main and template images CHANGE IT WHILE RUNNING ON THE PI
main_image_path = 'C:/Users/joelj/Downloads/Flower_detection_algorithm/content/main.jpeg'
template_image_path = 'C:/Users/joelj/Downloads/Flower_detection_algorithm/content/template.jpeg'
##########################


#load main and template images
main_image = cv2.imread(main_image_path)
template = cv2.imread(template_image_path)
template_image = cv2.imread(template_image_path)
#check if images loaded successfully
if main_image is None:
    raise ValueError(f"Could not load the main image from path: {main_image_path}")
if template is None:
    raise ValueError(f"Could not load the template image from path: {template_image_path}")


scale_percent = 50
width = int(main_image.shape[1] * scale_percent / 100)
height = int(main_image.shape[0] * scale_percent / 100)
dim = (width, height)

resized_main_image = cv2.resize(main_image, dim, interpolation=cv2.INTER_AREA)
resized_template_image = cv2.resize(template_image, (int(template_image.shape[1] * scale_percent / 100), int(template_image.shape[0] * scale_percent / 100)), interpolation=cv2.INTER_AREA)

#convert images to grayscale 
main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
main_gray = cv2.cvtColor(resized_main_image, cv2.COLOR_BGR2GRAY)
template_gray_1 = cv2.cvtColor(resized_template_image, cv2.COLOR_BGR2GRAY)

method = cv2.TM_CCOEFF_NORMED
result = cv2.matchTemplate(main_gray, template_gray_1, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

top_left = max_loc
bottom_right = (top_left[0] + resized_template_image.shape[1], top_left[1] + resized_template_image.shape[0])
cv2.rectangle(resized_main_image, top_left, bottom_right, (0, 255, 0), 2)

print(f"Top left corner: {top_left}")
print(f"Bottom right corner: {bottom_right}")

plt.imshow(resized_main_image)
plt.show()

isolated_pixels = resized_main_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
plt.imshow(isolated_pixels)
plt.show()


isolated_pixels = resized_main_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

print(f"Isolated pixels shape: {isolated_pixels.shape}")

if isolated_pixels.size == 0:
    raise ValueError("Isolated pixels region is empty. Check the bounding box coordinates.")


# round pixel values to the closest primary color
def round_to_rgb(pixel):
    r, g, b = pixel
    if r > g and r > b:
        return [255, 0, 0]  # R
    elif g > r and g > b:
        return [0, 255, 0]  # G
    elif b > r and b > g:
        return [0, 0, 255]  #B
    else:
        if g >= r and g >= b:
            return [0, 255, 0]
        elif r >= g and r >= b:
            return [255, 0, 0]
        else:
            return [0, 0, 255]

rounded_pixels = np.apply_along_axis(round_to_rgb, 2, isolated_pixels).astype(np.uint8)

#  the percentage of r,b,g content
total_pixels = rounded_pixels.shape[0] * rounded_pixels.shape[1]
blue_pixels = np.sum(np.all(rounded_pixels == [0, 0, 255], axis=2))
red_pixels = np.sum(np.all(rounded_pixels == [255, 0, 0], axis=2))
green_pixels = np.sum(np.all(rounded_pixels == [0, 255, 0], axis=2))

blue_percentage = (blue_pixels / total_pixels) * 100 if total_pixels != 0 else 0
red_percentage = (red_pixels / total_pixels) * 100 if total_pixels != 0 else 0
green_percentage = (green_pixels / total_pixels) * 100 if total_pixels != 0 else 0

print(f"Total Pixels: {total_pixels:.2f}")
print(f"Blue Pixels: {blue_pixels:.2f}")
print(f"Red Pixels: {red_pixels:.2f}")
print(f"Green Pixels: {green_pixels:.2f}")

print(f"Percentage of blue content: {blue_percentage:.2f}%")
print(f"Percentage of red content: {red_percentage:.2f}%")
print(f"Percentage of green content: {green_percentage:.2f}%")


plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(resized_main_image, cv2.COLOR_BGR2RGB))
plt.gca().add_patch(plt.Rectangle(top_left, bottom_right[0]-top_left[0], bottom_right[1]-top_left[1], edgecolor='red', facecolor='none', linewidth=2))
plt.title("Template Matching Result")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rounded_pixels, cv2.COLOR_BGR2RGB))
plt.title("Isolated and Rounded Pixels")

plt.show()






# # Initialize ORB detector fun the descriptrs 
# orb = cv2.ORB_create()

# # detect keypoints and compute descriptors
# keypoints1, descriptors1 = orb.detectAndCompute(main_image_gray, None)
# keypoints2, descriptors2 = orb.detectAndCompute(template_gray, None)

# #check if descriptors are valid no error will be there in this point 
# if descriptors1 is None or descriptors2 is None:
#     raise ValueError("Descriptors cannot be None")
# #initilaisaton of bf matcher
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# matches = bf.match(descriptors1, descriptors2)
# matches = sorted(matches, key=lambda x: x.distance)

# print(f"Number of keypoints in main image: {len(keypoints1)}")
# print(f"Number of keypoints in template image: {len(keypoints2)}")
# print(f"Number of matches found: {len(matches)}")

# valid_matches = []
# for match in matches:
#     if match.trainIdx < len(keypoints2) and match.queryIdx < len(keypoints1):
#         valid_matches.append(match)
#     else:
#         print(f"Invalid match found: trainIdx={match.trainIdx}, queryIdx={match.queryIdx}")

# if len(valid_matches) == 0:
#     raise ValueError("No valid matches found.")

# num_matches_to_draw = min(10, len(valid_matches))

# # check if there are enough matches to draw
# if num_matches_to_draw == 0:
#     raise ValueError("Not enough matches to draw.")

# # matches and show image
# try:
#     match_img = cv2.drawMatches(template, keypoints2, main_image, keypoints1, valid_matches[:num_matches_to_draw], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     plt.imshow(match_img)
#     plt.show()
# except Exception as e:
#     print(f"Error during drawing matches: {e}")

# # getting the  matched keypoints
# src_pts = np.float32([keypoints2[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
# dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)

# # FIND THE homography matrix
# try:
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     if M is None or mask is None:
#         raise ValueError("Homography could not be computed.")
#     matchesMask = mask.ravel().tolist()
#     print(f"Homography matrix:\n{M}")
# except Exception as e:
#     print(f"Error during computing homography: {e}")

# # Get dimensions of template image
# h, w = template.shape[:2]

# # Define template corner points
# pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

# # Perform perspective transformation
# try:
#     dst = cv2.perspectiveTransform(pts, M)
#     print(f"Transformed points: {dst}")
#     if dst is not None and len(dst) == 4:
#         # Draw bounding box on main image
#         main_image_with_box = cv2.polylines(main_image.copy(), [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
#         plt.imshow(main_image_with_box)
#         plt.show()
#     else:
#         print("Homography transformation failed.")
# except Exception as e:
#     print(f"Error during perspective transformation: {e}")
