#!/usr/bin/env python
import rospy
from hide_and_seek_navigation.msg import listmsg
from std_msgs.msg import String

def planner():
    pub = rospy.Publisher('move_base_goal', listmsg, queue_size=5)
    rospy.init_node('planner')
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        msg = listmsg()
        msg.x = 0.5
        msg.y = 0.5
        msg.yaw = 90
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        planner()
    except rospy.ROSInterruptException:
        pass