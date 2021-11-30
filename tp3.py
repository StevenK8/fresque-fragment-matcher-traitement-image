from PIL.Image import alpha_composite
import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import floor


# Gets the matches of good features to track between two images
def get_matches(background, fragment):
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    fragment = cv2.cvtColor(fragment, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    fragment_gray = cv2.cvtColor(fragment, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    background_keypoints, des1 = sift.detectAndCompute(background_gray, None)
    fragment_keypoints, des2 = sift.detectAndCompute(fragment_gray, None)

    if len(background_keypoints) == 0 or len(fragment_keypoints) == 0:
        return -1, -1, -1

    # Create the matcher
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = matcher.match(des1, des2)
    # Sort the matches in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    backgroundMatches = []
    fragmentMatches = []
    # If the euclidean distance between two points from background_keypoints and fragment_keypoints is less than a threshold, add it to the list
    if len(matches) > 0:
        for match in matches:
            if match.distance < 500:
                backgroundMatches.append(
                    background_keypoints[match.queryIdx].pt)
                fragmentMatches.append(fragment_keypoints[match.trainIdx].pt)

    i = 0
    # If a point is at a distance more than 500 from the first point, remove it from the list
    if len(backgroundMatches) > 1:
        (referenceX, referenceY) = backgroundMatches[0]
        for match in backgroundMatches:
            if(euclidean_distance(referenceX, referenceY, match[0], match[1]) > 500):
                backgroundMatches.remove(match)
                fragmentMatches.remove(fragmentMatches[i])
                # print("Removed point", fragmentMatches)
                
            i += 1
    elif(len(backgroundMatches) == 1):
        return backgroundMatches[0][0], backgroundMatches[0][1], 0
    elif(len(backgroundMatches) == 0):
        return -1,-1,-1

    if len(backgroundMatches) > 1:
        # print(backgroundMatches[1], fragmentMatches[1])
        # Instantiate empty image
        # img3 = np.zeros((max(background.shape[0], fragment.shape[0]), background.shape[1]+fragment.shape[1], 3), np.uint8)
        # # Draw the real matches
        # img3 = cv2.drawMatches(background,background_keypoints,fragment,fragment_keypoints,realMatches, fragment, flags=2)
        # plt.imshow(img3),plt.show()

        # Calculate the angle between the two vectors
        # print(np.array((backgroundMatches[0][0]-backgroundMatches[1][0],backgroundMatches[0][1]-backgroundMatches[1][1])), np.array((fragmentMatches[0][0]-fragmentMatches[1][0],fragmentMatches[0][1]-fragmentMatches[1][1])))
        angle = angle_between(np.array((backgroundMatches[0][0]-backgroundMatches[1][0], backgroundMatches[0][1]-backgroundMatches[1][1])), np.array(
            (fragmentMatches[0][0]-fragmentMatches[1][0], fragmentMatches[0][1]-fragmentMatches[1][1])))


        offsetX, offsetY = distance_between_points((fragment.shape[:2][1]/2), fragmentMatches[0][0], (fragment.shape[:2][0]/2) , fragmentMatches[0][1])
        v = rotate_vector((offsetX, offsetY), angle)
        # print (v)

        return backgroundMatches[0][0] - v[0], backgroundMatches[0][1] - v[1], -angle
    elif(len(backgroundMatches) == 1):
        return backgroundMatches[0][0], backgroundMatches[0][1], 0
    elif(len(backgroundMatches) == 0):
        return -1,-1,-1


# Calculates the euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

# Calculates the distance in x and y between two points
# def distance(x1, y1, x2, y2):
# returns the horizontal distance and the vertical distance
def distance_between_points(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    return x, y

# Calculates the rotation of a vector
def rotate_vector(vector, angle):
    angle = np.deg2rad(angle)
    return np.array([vector[0]*np.cos(angle) - vector[1]*np.sin(angle), vector[0]*np.sin(angle) + vector[1]*np.cos(angle)])


# Calulates the angle in degrees between two vectors
def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi


# Add a png image over a bigger jpg image
def add_overlay(background, fragment, x, y, angle):
    # Rotate the image
    fragment = rotate_image(fragment, angle)

    b, g, r, a = cv2.split(fragment)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = background[y:y+h, x:x+w]

    # Black-out the area behind the logo in our original ROI
    fragment1_bg = cv2.bitwise_and(
        roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    fragment2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    background[y:y+h, x:x+w] = cv2.add(fragment1_bg, fragment2_fg)

    return fragment


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# Write in a text file i, x, y, angle from a list of points
def write_file(fragmentPositions, filename):
    f = open(filename, "w")
    for i, x, y, angle in fragmentPositions:
        f.write(str(i) + " " + str(x) + " " + str(y) + " " + str(angle) + "\n")
    f.close()


def main():
    background = cv2.imread(
        'Michelangelo_ThecreationofAdam_1707x775.jpg', cv2.IMREAD_UNCHANGED)
    # fragment = cv2.imread('frag_eroded/frag_eroded_4.png',
    #                       cv2.IMREAD_UNCHANGED)

    # x, y, angle = get_matches(background, fragment)
    # if x is not None:
    #     print(x, y, angle)
    
    fragmentPositions = []    
    
    for i in range(0, 327):
        print(str((i/327)*100)+"%")
        x,y,angle = -1,-1,-1
        fragment = cv2.imread('frag_eroded/frag_eroded_'+str(i)+'.png',
                          cv2.IMREAD_UNCHANGED)
        x, y, angle = get_matches(background, fragment)
        if x >= 0:
            fragmentPositions.append((i,x,y,angle))
    
    write_file(fragmentPositions, "test.txt")


main()
