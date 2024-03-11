#!/usr/bin/env python3
'''
MIT License

Copyright (c) 2024 High-Assurance Mobility and Control (HMC) Laboratory at Ulsan National Institute of Scienece and Technology (UNIST), Republic of Korea 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from re import L
import rospy
from std_msgs.msg import Bool
from visualization_msgs.msg import MarkerArray
import rospkg

from racepkg.h2h_configs import *
from racepkg.common.utils.file_utils import *
from racepkg.path_generator import PathGenerator

rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('racepkg')

class TrajPub:
    def __init__(self):               
        self.n_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=1.0)                   
        self.center_pub = rospy.Publisher('/center_line',MarkerArray,queue_size = 2)
        self.bound_in_pub = rospy.Publisher('/track_bound_in',MarkerArray,queue_size = 2)
        self.bound_out_pub = rospy.Publisher('/track_bound_out',MarkerArray,queue_size = 2)        
        self.dt = self.t_horizon / self.n_nodes*1.0        
        ## 
        # Generate Racing track info 
        self.track_info = PathGenerator()        
        while self.track_info.track_ready is False:
             rospy.sleep(0.01)
        ##     
        # prediction callback   
        self.traj_hz = rospy.get_param('~traj_hz', default=2)
        self.traj_pub_timer = rospy.Timer(rospy.Duration(1/self.traj_hz), self.traj_timer_callback)         
   
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True            
            rate.sleep()


    def traj_timer_callback(self,event):
        
        if self.track_info.track_ready:     
            self.center_pub.publish(self.track_info.centerline)
            self.bound_in_pub.publish(self.track_info.track_bound_in)
            self.bound_out_pub.publish(self.track_info.track_bound_out)
        
        
   
###################################################################################

def main():
    rospy.init_node("traj_pub")    
    TrajPub()

if __name__ == "__main__":
    main()




 
    


