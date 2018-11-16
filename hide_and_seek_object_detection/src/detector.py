from __future__ import print_function
import roslib
roslib.load_manifest('hide_and_seek_object_detection')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

class Detector:

    def __init__(self):
        self.window_name = "Ros Camera View"
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw/", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
    
        (rows, cols, channels) = cv_image.shape
        cv2.imshow(self.window_name, cv_image)
        cv2.waitKey(3)

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