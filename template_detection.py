import cv2
import numpy as np
import os

# http://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
# template-matching
def get_hand_cordinates(image, isFrame):
    img = image
    if not isFrame:
        img = cv2.imread(image, 0)
    #img = cv2.resize(img, (304, 540))
    img2 = img.copy()

    width, height = img.shape[::-1]
    #print width, height
    templates = ['temp_hand1.png']
    #templates = ['temp_hand33.png']
    found = None
    #break
    for template_name in templates:
        template = cv2.imread(template_name, 0)
        #template = cv2.resize(template, (200, 300))
        template = cv2.Canny(template, 50, 200)
        w, h = template.shape[::-1]
        #print w,h
        # All the 6 methods for comparison in a list minus 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
        #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        #            'cv2.TM_CCORR_NORMED']
        #methods = ['cv2.TM_CCORR']
        methods = ['cv2.TM_CCOEFF']
        for meth in methods:
            method = eval(meth)
            # loop over the scales of the image
            for scale in np.linspace(0.2, 0.8, 50)[::-1]:
            #for scale in np.linspace(0.2, 0.8, 50)[::-1]:
                img = img2.copy()
                width, height = img.shape[::-1]
                # resize the image according to the scale, and keep track
                # of the ratio of the resizing
                resized = cv2.resize(img, (int(width*scale), int(height* scale)))
                rwidth, rheight = resized.shape[::-1]
                r = img.shape[1] / float(resized.shape[1])

                # if the resized image is smaller than the template, then break
                # from the loop
                if resized.shape[0] < h or resized.shape[1] < w:
                    break

                # detect edges in the resized, grayscale image and apply template
                # matching to find the template in the image
                edged = cv2.Canny(resized, 50, 200)
                result = cv2.matchTemplate(edged, template, method)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                # if we have found a new maximum correlation value, then update
                # the bookkeeping variable
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r, w, h)
                    #print
                    # draw a bounding box around the detected region
                    cv2.rectangle(resized, maxLoc,
                                (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
                    cv2.imwrite("./detected_original_frames/"+image, resized)
                    fileName = "./detected_original_frames/"+str(scale)+"_"+str(maxVal)+"_"+ template_name + "_"+meth+"_"+image.split("/")[-1]
                    #print fileName
                    cv2.imwrite(fileName, resized)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (maxVal, maxLoc, r, w, h) = found
    #print found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))
    #print (startX, startY), (endX, endY)
    # draw a bounding box around the detected result and display the image
    img2 = cv2.imread(image, cv2.IMREAD_COLOR)
    cv2.rectangle(img2, (startX, startY), (endX, endY), (0, 0, 255), 2)
    #cv2.imwrite('final_detected.png', img2)
    cv2.imwrite("./template_detection/" + image.split("/")[-1], img2)
    return img2, (startX, startY), (endX, endY), (startX, startY, endY - startY, endX - startX)


#image_name = "./original_frames/4.jpg"
#get_hand_cordinates(image_name, False)

# images_dir = "./original_frames"

# for image in os.listdir(images_dir):
#     if image.startswith('.'):
#         continue
#     print os.path.join(images_dir,image)
#     get_hand_cordinates(os.path.join(images_dir,image))
