from __future__ import print_function
import roslib
roslib.load_manifest('hide_and_seek_object_detection')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import argparse

# classes of objects that can be detected by the pretrained model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# random colors for color map
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class Detector:

    def __init__(self):
        self.window_name = "Ros Camera View" 
        # arguments for testing purposes 
        # will be replaced with fixed names in final form
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe deploy protxt file")
        self.ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
        self.ap.add_argument("-c", "--confidence", type=float, default=0.2, help="min probability to filter weak detections")
        self.args = vars(self.ap.parse_args())
        print("[INFO] loading model...")
        # reading the pretrained deep neural net
        self.net = cv2.dnn.readNetFromCaffe(self.args["prototxt"], self.args["model"])
        self.bridge = CvBridge()
        # subscribing to images from ros camera topic
        self.image_sub = rospy.Subscriber("/camera/image_raw/", Image, self.callback) 

    def callback(self, data):
        try:
            # converting to opencv image format
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # actual detection in current frame
        self.compute_object_detection(frame)

        # diplaying the frame
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(3)

    def compute_object_detection(self, img):
        (h, w) = img.shape[:2] # image size
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5) # raw image
        print("[INFO] computing object detections...")
        self.net.setInput(blob)
        detections = self.net.forward() # pass the image through the neural net

        # extracting the outputs of the neural net and overlaying the bounding boxes
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
                print("[INFO] {}".format(label))
                cv2.rectangle(img, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


def main(args):
    detc = Detector()
    rospy.init_node("object_detector", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)