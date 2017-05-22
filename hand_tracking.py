import cv2
import numpy as np
from template_detection import get_hand_cordinates

class Hand_Tracking:

    # first frame and first track window
    def __init__(self, frame, track_window):
        if frame is None:
            return
        c, r, w, h  = track_window  # simply hardcoded the values
        #print "track_window",track_window
        cv2.imwrite("frame.jpg", frame)
        #print frame.shape
        roi = frame[r:r + h, c:c + w]
        cv2.imwrite("roi.jpg", roi)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #print "hsv_roi", hsv_roi.shape
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)
        self.counter = 1

    def hand_tracking(self, frame, track_window):
        buffer = 0
        #print "track window in hand tracking", track_window
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        # apply camShift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, self.term_crit)
        #ret, track_window = cv2.CamShift(dst, track_window, self.term_crit)
        #print "after cam shif", track_window, "ret", ret
        x, y, w, h = track_window
        c, r, w, h = track_window
        #print track_window
        hand_box_image = frame[r - buffer:r + h + buffer, c - buffer:c + w + buffer]
        hand_box_image_file = './hand_box_images/' + str(self.counter) + '_hand_box.jpg'
        cv2.imwrite(hand_box_image_file, hand_box_image)
        img2 = cv2.rectangle(frame, (x - buffer, y - buffer), (x + w + buffer, y + h + buffer), 255, 2)
        # Draw it on image
        #pts = cv2.boxPoints(ret)
        #pts = np.int0(pts)
        #img2 = cv2.polylines(frame, [pts], True, 255, 2)
        cv2.imwrite('./cam_shift_images/'+str(self.counter)+'_cam_shift.jpg', img2)
        self.counter += 1
        return track_window, hand_box_image_file