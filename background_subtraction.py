import numpy as np
import cv2 as cv2
from template_detection import get_hand_cordinates
from face_detection import generate_skin_probable_image
from hand_tracking import Hand_Tracking
from caffe_test import Classifier
import os
from time import sleep
#cap = cv2.VideoCapture("asl.avi")
#cap = cv2.VideoCapture("IMG_4684.avi")
#cap = cv2.VideoCapture("vid10.mov")
#cap = cv2.VideoCapture("prahlad_1_cropped.mp4")
#cap = cv2.VideoCapture("rutuja_2.MP4")
#cap = cv2.VideoCapture("rutu_2.MP4")
#cap = cv2.VideoCapture("asl.mp4")
#cap = cv2.VideoCapture("pral_1.mov")
cap = cv2.VideoCapture("pral_2.mov")
#cap = cv2.VideoCapture("rutu.mov")
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("asl_cut.mp4")
sleep(1)
counter = 0
ret = True
buffer = 0

# def skin_toned_image(img, counter):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsvT = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     #nose_area_width = 85
#     nose_area_width = 10
#     print faces
#     for (x, y, w, h) in faces:
#         # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         # from Foster et al. paper - sec 3.1.3.1
#         # get 10 X 10 pixel area of the nose, but this is for re-sized image
#         center_x, center_y = x + h / 2, y + w / 2
#
#         # hsv = cv2.cvtColor(nose_img, cv2.COLOR_BGR2HSV)
#         # src - http://docs.opencv.org/3.2.0/d1/db7/tutorial_py_histogram_begins.html
#         # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
#         mask = np.zeros(img.shape[:2], np.uint8)
#         mask[center_y - nose_area_width / 2: center_y - nose_area_width / 2 + nose_area_width,
#         center_x - nose_area_width / 2: center_x - nose_area_width / 2 + nose_area_width] = 255
#         hist = cv2.calcHist([hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
#
#         # apply backprojection
#         dst = cv2.calcBackProject([hsvT], [0, 1], hist, [0, 180, 0, 256], 1)
#
#         # threshold and binary AND
#         ret, thresh = cv2.threshold(dst, 0.6, 255, 0)
#         #thresh = cv2.merge((thresh, thresh, thresh))
#
#         #res = cv2.bitwise_and(img, thresh)
#         cv2.imwrite("./frames/"+str(counter) + '_skin_toned_binary.jpg', thresh)
#         #cv2.imwrite(str(counter).split('/')[-1] + '_skin_toned_image.jpg', res)
#     return thresh

#fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg = cv2.createBackgroundSubtractorKNN()

track_window = None
Hand_Tracking_Obj = Hand_Tracking(None, None)

deploy_prototxt = 'deploy.prototxt'
# Model caffemodel file
#model_trained = 'massey_iter_1000.caffemodel'
model_trained = 'caffenet_paaji_iter_2000.caffemodel'
#model_trained = 'massey_2_iter_1000.caffemodel'
#model_trained = 'caffenet_paaji_iter_1000.caffemodel'
# Path to the mean image (used for input processing)
caffe_root = '/Users/chiteshtewani/Desktop/caffe/caffe/'
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
classifier = Classifier(deploy_prototxt, model_trained, mean_path)
isVideo = True
fileList = []
dir = './AtoZ'
if not isVideo:
    for file in os.listdir(dir):
        if '.JPG' in file:
            fileList.append(os.path.join(dir,file))

print " ----- Detected Characters ---- "
while True:
    ret, frame = cap.read()
    #if not isVideo:
    #    frame = cv2.imread(fileList[counter], 0)

    counter += 1
    #if isVideo and counter % 10 != 1:
    if  counter % 20 != 1:
       continue
    #counter += 1
    #if (not isVideo and counter == len(fileList)) or (isVideo and ret):
    if not ret:
        break
    #if counter < 15:
    #    continue
    # call skin-color detection of frame
    #fgmask = fgbg.apply(frame)

    # TODO - remove this --
    # skip till 24 counter to get the matching template
    fileName = "./original_frames/" + str(counter) + '.jpg'
    cv2.imwrite(fileName, frame)
    # template-matching
    if counter == 1:
        img, _, _, track_window = get_hand_cordinates(fileName, False)
        #print "track_window",track_window
        Hand_Tracking_Obj = Hand_Tracking(frame, track_window)
        # startX, startY, endX, endY = track_window
        # get the hand tracking --

    if counter > 1:
        #print "track_window > 1",track_window
        # CAM-shift tracking - get the coordinates of the hand
        track_window, hand_box_image_file = Hand_Tracking_Obj.hand_tracking(frame, track_window)

        skin_image = generate_skin_probable_image(fileName, False)
        startX, startY, endX, endY = track_window
        c, r, w, h = track_window
        cv2.imwrite("./skin_images/" + str(counter) + '.jpg', skin_image)
        hand_skin_image = skin_image[r - buffer:r + h + buffer, c - buffer:c + w + buffer]
        # hand_skin_image = skin_image[startY - buffer: endY - startY + buffer,
        #                   startX - buffer: endX - startX + buffer]

        skin_hand_image_file = "./skin_probable_hand_images/" + str(counter) + '_skin_hand_image.jpg'
        #print skin_hand_image_file,'----------------------------'
        #cv2.resize(hand_skin_image, (256,256))
        cv2.imwrite(skin_hand_image_file, hand_skin_image)
        labels = classifier.predict(hand_box_image_file)
        label_str = []
        for label in labels:
            label = chr(int(label) + 97)
            label_str.append(label)

        #new_labels = [str(i + 1)+":"+label_str[i] for i in range(len(label_str))]
        print " ".join(label_str)
        print
        cv2.imwrite("./output_labels/" + "_".join(label_str) + "_" + str(counter) + ".jpg", cv2.imread(hand_box_image_file,1))
        #cv2.imwrite("./output_labels/" + "_".join(label_str) + "_" + str(counter) + "_2.jpg", hand_skin_image)
        #output_labels = ""
        #    cv2.imwrite("./output_labels/"+label+"_"+str(counter)+".jpg",hand_skin_image)
        # given the hand coordinates for hand coordinates to CAMShift
        #track_window = start + end
    #counter += 1
    # print counter

    # cv2.imwrite("./original_frames/" + str(counter) + '.jpg', frame)
    #cv2.imwrite("./background_subtraction_images/"+str(counter) + '_fgmask.jpg', fgmask)
    #res = cv2.bitwise_and(skin_image,fgmask)
    #hand_skin_image = res[startY - buffer: endY - startY + buffer,
    #                  startX - buffer: endX - startX + buffer]
    #cv2.imwrite("./skin_probable_hand_image/" + str(counter) + '_fgmask.jpg', fgmask)
    # print fgmask.shape, thresh.shape
    # res = cv2.bitwise_and(fgmask, thresh)
    # #cv2.imwrite("./original_frames/" + str(counter) + '.jpg', frame)
    # cv2.imwrite("./frames/" + str(counter) + '_z.jpg', res)
    # print str(counter)

cap.release()

