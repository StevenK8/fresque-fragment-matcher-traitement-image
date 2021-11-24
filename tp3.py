import cv2
import numpy as np

#Finds the coordinates and angle of a fragment of a template in an image
def find_template(img, template, threshold=0.8):
    #Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    #Find the template
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    # loc = np.where(res >= threshold)

    #Find the center of the template
    x = int(np.mean(res[1]))
    y = int(np.mean(res[0]))
    print(res)
    # x=100
    # y=200

    #Find the angle of the template
    # angle = cv2.minAreaRect(loc)[-1]
    angle=0

    return x, y, angle

# Add a png image over a bigger jpg image
def add_overlay(img, template, x, y, angle):
    # Rotate the image
    img = rotate_image(img, angle)
    
    b,g,r,a = cv2.split(img)
    overlay_color = cv2.merge((b,g,r))
	
	# Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a,5)

    h, w, _ = overlay_color.shape
    roi = template[y:y+h, x:x+w]

	# Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
	
	# Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

	# Update the original image with our new ROI
    template[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

    return template


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def main():
    img = cv2.imread('frag_eroded/frag_eroded_2.png',cv2.IMREAD_UNCHANGED)
    template = cv2.imread('Michelangelo_ThecreationofAdam_1707x775.jpg')
    x, y, angle = find_template(img, template)
    overlay = add_overlay(img, template, x, y, angle)
    cv2.imshow('img', overlay)
    
    #wait for key to exit
    cv2.waitKey(0)
    
main()