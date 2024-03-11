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
import numpy as np  
import tf
import rospy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from racepkg.common.tracks.radius_arclength_track import RadiusArclengthTrack

class PathGenerator:
    def __init__(self):                
        self.cur_velocity=0.0
        self.cmd_velocity=0.0
        self.ttc_stop_signal = False
        self.centerline = MarkerArray()
        self.track_bound_in = MarkerArray()
        self.track_bound_out = MarkerArray()
        self.dt = 0.1
        self.track_width = 1.7 
        self.slack = 0.1
        self.cl_segs = None
        self.track = None 
        self.track_ready = False
        self.center_marker = MarkerArray()
        self.gen_path()
        
    def gen_path(self):
        self.track = RadiusArclengthTrack()
        
        curv = 1.0
        curve1 = np.array([[curv*np.pi, curv]])
        curve2 = np.array([[0.5*curv*np.pi, -1*curv]])
        curve3 = np.array([[curv*np.pi, curv]])
        curve4 = np.array([[0.5*curv*np.pi, curv]])
        s1 = np.array([[2.3, 999.0]])
        s2 = np.array([[2.0, 999.0]])
        s3 = np.array([[2.0, 999.0]])
        track = np.vstack([s1,curve1,s1,curve2,curve3,s2,curve4,s3,s1,curve1,s1,curve2,curve3,s2,curve4,s3])         
        self.cl_segs = track        
        self.track.initialize(self.track_width,self.slack, self.cl_segs, init_pos=(0.0, 0.0, 0.0))
        
        self.track_ready = True
        self.get_track_points()        
        return
        
    def get_marker_from_track(self,x,y,psi,color):
        tmpmarkerArray = MarkerArray()
        time = rospy.Time.now()
        for i in range(len(x)):
            tmp_marker = Marker()
            tmp_marker.header.stamp = time
            tmp_marker.header.frame_id = "map"
            tmp_marker.id = i
            tmp_marker.type = Marker.ARROW
            # Set the scale of the marker
            tmp_marker.scale.x = 0.05
            tmp_marker.scale.y = 0.05
            tmp_marker.scale.z = 0.01

            # Set the color
            tmp_marker.color.r = color.r
            tmp_marker.color.g = color.g
            tmp_marker.color.b = color.b
            tmp_marker.color.a = color.a

            tmp_marker.pose.position.x = x[i]
            tmp_marker.pose.position.y =y[i]
            tmp_marker.pose.position.z = 0.0
            quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, psi[i])
            tmp_marker.pose.orientation.x = quaternion[0]
            tmp_marker.pose.orientation.y = quaternion[1]
            tmp_marker.pose.orientation.z = quaternion[2]
            tmp_marker.pose.orientation.w = quaternion[3]
            tmpmarkerArray.markers.append(tmp_marker)
        return tmpmarkerArray

    def get_track_points(self):
        if self.track is None:
            return
        n = int(self.track.track_length/self.dt)        

        x_track = []
        x_bound_in = []
        x_bound_out = []
        y_track = []
        y_bound_in = []
        y_bound_out = []
        psi_track = []
        psi_bound_in = []
        psi_bound_out = []
   
        for i in range(n):
            j = i*self.dt 
            cl_coord = (j, 0, 0)
            xy_coord = self.track.local_to_global(cl_coord)
            x_track.append(xy_coord[0])
            y_track.append(xy_coord[1])
            psi_track.append(xy_coord[2])
            cl_coord = (j, self.track_width / 2, 0)
            xy_coord = self.track.local_to_global(cl_coord)
            x_bound_in.append(xy_coord[0])
            y_bound_in.append(xy_coord[1])
            psi_bound_in.append(xy_coord[2])
            cl_coord = (j, -self.track_width / 2, 0)
            xy_coord = self.track.local_to_global(cl_coord)
            x_bound_out.append(xy_coord[0])
            y_bound_out.append(xy_coord[1])
            psi_bound_out.append(xy_coord[2])

        color_center = ColorRGBA()        
        color_center.g = 1.0
        color_center.a = 0.3
        self.centerline = self.get_marker_from_track(x_track,y_track,psi_track,color_center)

        color_bound = ColorRGBA()        
        color_bound.r = 1.0
        color_bound .a = 0.3
        self.track_bound_in = self.get_marker_from_track(x_bound_in,y_bound_in,psi_bound_in,color_bound)
        self.track_bound_out = self.get_marker_from_track(x_bound_out,y_bound_out,psi_bound_out,color_bound)


