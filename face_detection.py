'''
Face detection using openCV2 [Viola-Jones]
Tutorial - http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
'''

import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#img_name = "img3.jpg"
def generate_skin_probable_image(image, isFrame):
    img = image
    if not isFrame:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
    # img = cv2.resize(img, (304, 540))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = img.copy()
    #print img.shape

    #img_name = "./original_frames/28.jpg"
    #img = cv2.imread(img_name, 1)
    #img1 = cv2.imread(img_name, 1)

    #img = cv2.resize(_img, (340, 240))
    #img1 = cv2.resize(_img1, (340, 240))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsvT = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    nose_area_width = 75
    #print faces
    bestX = bestY = bestW = bestH = bestArea = 0

    for (x,y,w,h) in faces:
        if bestArea < w * h:
            bestX, bestY, bestW, bestH = x,y,w,h
            bestArea = w * h

    x, y, w, h = bestX, bestY, bestW, bestH
    cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
    # from Foster et al. paper - sec 3.1.3.1
    # get 10 X 10 pixel area of the nose, but this is for re-sized image
    center_x, center_y = x + h/2, y + w/2

    # src - http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
    nose_img = img[center_y - nose_area_width/2 : center_y - nose_area_width/2 + nose_area_width,
               center_x - nose_area_width/2 : center_x - nose_area_width/2 + nose_area_width]
    # cv2.rectangle(img1, (center_x - nose_area_width/2, center_y - nose_area_width/2),
    #               (center_x - nose_area_width/2 + nose_area_width, center_y - nose_area_width/2 + nose_area_width),
    #               (0, 255, 0), 2)
    cv2.imwrite('./nose_images/'+image.split('/')[-1]+'_nose.jpg', nose_img)

    #hsv = cv2.cvtColor(nose_img, cv2.COLOR_BGR2HSV)
    # src - http://docs.opencv.org/3.2.0/d1/db7/tutorial_py_histogram_begins.html
    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[center_y - nose_area_width/2 : center_y - nose_area_width/2 + nose_area_width,
    center_x - nose_area_width/2 : center_x - nose_area_width/2 + nose_area_width] = 255
    hist = cv2.calcHist([hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])

    #cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    #plt.plot(hist, color='b')
    #plt.show()

    # apply backprojection
    dst = cv2.calcBackProject([hsvT], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    cv2.filter2D(dst, -1, disc, dst)

    # threshold and binary AND
    ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
    thresh = cv2.merge((thresh, thresh, thresh))
    res = cv2.bitwise_and(img, thresh)
    # cv2.imwrite('./skin_toned_images/'+image.split('/')[-1]+'_skin_toned_binary.jpg', thresh)
    # cv2.imwrite('./skin_toned_images/'+image.split('/')[-1]+'_skin_toned_image.jpg', res)

    # img1 = cv2.imread(image, 0)
    cv2.imwrite('./face_detected_images/'+image.split('/')[-1]+'_detected.jpg', img1)
    return res