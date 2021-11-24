import cv2
import numpy as np

#Finds the coordinates and angle of a fragment of a template in an image
def find_template(img, template, threshold=0.8):
    #Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    #Find the template
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    #Find the center of the template
    # x = int(np.mean(loc[1]))
    # y = int(np.mean(loc[0]))
    print(loc)
    x=100
    y=200

    #Find the angle of the template
    # angle = cv2.minAreaRect(loc)[-1]
    angle=0

    return x, y, angle

# Add a png image over a bigger jpg image
def add_overlay(img, template, x, y, angle):
    # Rotate the image
    img = rotate_image(img, angle)

    # Add the image to the template	
    template[y:y+img.shape[0], x:x+img.shape[1]] = img

    return template


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def main():
    img = cv2.imread('frag_eroded/frag_eroded_0.png')
    template = cv2.imread('Michelangelo_ThecreationofAdam_1707x775.jpg')
    x, y, angle = find_template(img, template)
    overlay = add_overlay(img, template, x, y, angle)
    cv2.imshow('img', overlay)
    
    #wait for key to exit
    cv2.waitKey(0)
    
main()