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

import pickle
import rospy
import os
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import threading
import rospkg
from racepkg.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehicleActuation
from racepkg.utils import pose_to_vehicleState, odom_to_vehicleState
from racepkg.path_generator import PathGenerator
from racepkg.h2h_configs import *
from racepkg.common.utils.file_utils import *
from racepkg.common.utils.scenario_utils import RealData
from message_filters import ApproximateTimeSynchronizer, Subscriber

from dynamic_reconfigure.server import Server
from racepkg.cfg import racepkgDynConfig

rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('racepkg')

class Recorder:
    def __init__(self):             
        self.n_nodes = rospy.get_param('~n_nodes', default=10)
        self.t_horizon = rospy.get_param('~t_horizon', default=1.0)                   
        self.torch_device = "cuda:0"   ## Specify the name of GPU 
        # self.torch_dtype  = torch.double
        self.dt = self.t_horizon / self.n_nodes*1.0        
        ## 
        # Generate Racing track info 
        self.track_load = False
        track_file_path = os.path.join(track_dir, 'track.pickle')        
        self.track_info = PathGenerator()
        if self.track_load:
            with open(track_file_path, 'rb') as file:            
                self.track_info = pickle.load(file)
        else: ## save track
            with open(track_file_path, 'wb') as file:
                pickle.dump(self.track_info, file)

        self.track_info = PathGenerator()
        
        while self.track_info.track_ready is False:
             rospy.sleep(0.01)
        
        
        self.cur_ego_odom = Odometry()        
        self.cur_ego_pose = PoseStamped()        
        self.cur_tar_odom = Odometry()
        self.cur_tar_pose = PoseStamped()

        self.cur_ego_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=0.1, x_tran=0.2, e_psi=0.0),
                                      v=BodyLinearVelocity(v_long=0.5),
                                      u=VehicleActuation(t=0.0, u_a=0.0, u_steer = 0.0))
        self.cur_tar_state = VehicleState()            
        
        ego_odom_topic = "/pose_estimate"
        ego_pose_topic = "/tracked_pose"        
        target_odom_topic = "/target/pose_estimate"
        target_pose_topic = "/target/tracked_pose"        
        
        self.ego_odom_ready = False
        self.tar_odom_ready = False

        # prediction callback   
        self.tv_pred = None
        self.tv_cav_pred = None
        self.tv_nmpc_pred = None
        self.tv_gp_pred = None
        self.xy_cov_trace = None 

        ## controller callback        
        self.ego_list = []
        self.tar_list = []
        self.tar_pred_list = []

        self.data_save = False
        self.prev_data_save = False
        self.pred_data_save = False
        self.save_buffer_length = 200

        self.ego_prev_pose = None
        self.tar_prev_pose = None
        self.ego_local_vel = None
        self.tar_local_vel = None

        # Subscribers
        self.ego_odom_sub = rospy.Subscriber(ego_odom_topic, Odometry, self.ego_odom_callback)                                
        self.target_odom_sub = rospy.Subscriber(target_odom_topic, Odometry, self.target_odom_callback)               
        self.ego_pose_sub = Subscriber(ego_pose_topic, PoseStamped)        
        self.target_pose_sub = Subscriber(target_pose_topic, PoseStamped)        
        self.ats = ApproximateTimeSynchronizer([self.ego_pose_sub,  self.target_pose_sub], queue_size=10, slop=0.05)
        self.sync_prev_time = rospy.Time.now().to_sec()
        self.ats.registerCallback(self.sync_callback)

        self.dyn_srv = Server(racepkgDynConfig, self.dyn_callback)        
        self.data_logging_hz = rospy.get_param('~data_logging_hz', default=10)
        self.prev_dl_time = rospy.Time.now().to_sec()
        self.prediction_timer = rospy.Timer(rospy.Duration(1/self.data_logging_hz), self.datalogging_callback)         
        
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True        
            rate.sleep()

 

    def dyn_callback(self,config,level):                
        self.pred_data_save = config.logging_prediction_results
        if config.clear_buffer:
            self.clear_buffer()
        self.predictor_type = config.predictor_type
        print("self.predictor_type = " + str(self.predictor_type))
        self.data_save = config.logging_vehicle_states
        if self.prev_data_save is True and self.data_save is False and config.clear_buffer is not True:
            self.save_buffer_in_thread()
            print("save data by turnning off the switch")
        self.prev_data_save = config.logging_vehicle_states
        print("dyn reconfigured")
        
        return config
        
    def clear_buffer(self):
        if len(self.ego_list) > 0:
            self.ego_list.clear()
            self.tar_list.clear()
            self.tar_pred_list.clear()
    
    def save_buffer_in_thread(self):        
        t = threading.Thread(target=self.save_buffer)
        t.start()

    def save_buffer(self):        
        real_data = RealData(self.track_info.track, len(self.tar_list), self.ego_list, self.tar_list, self.tar_pred_list)
        create_dir(path=real_dir)        
        pickle_write(real_data, os.path.join(real_dir, str(self.cur_ego_state.t) + '_'+ str(len(self.tar_list))+'.pkl'))
        rospy.loginfo("states data saved")
        self.clear_buffer()
        rospy.loginfo("states buffer has been cleaned")

    def sync_callback(self,  ego_pose_topic,  target_pose_topic):
        sync_cur_time = rospy.Time.now().to_sec()
        diff_sync_time = sync_cur_time - self.sync_prev_time         
        if abs(diff_sync_time) > 0.05:
            rospy.logwarn("sync diff time " + str(diff_sync_time))
        self.sync_prev_time = sync_cur_time                        
        self.ego_pose_callback(ego_pose_topic)        
        self.target_pose_callback(target_pose_topic)

    def ego_odom_callback(self,msg):
        if self.ego_odom_ready is False:
            self.ego_odom_ready = True
        self.cur_ego_odom = msg
        
    def ego_pose_callback(self,msg):
        self.cur_ego_pose = msg

    def target_odom_callback(self,msg):
        if self.tar_odom_ready is False:
            self.tar_odom_ready = True
        self.cur_tar_odom = msg
    
    def target_pose_callback(self,msg):
        self.cur_tar_pose = msg
        
    def datalogging_callback(self, event):      
        if self.ego_odom_ready and self.tar_odom_ready:
            pose_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_pose)
            odom_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_odom)
            pose_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_pose)            
            odom_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_odom)
        else:
            rospy.loginfo("state not ready")

        if self.data_save:
                if isinstance(self.cur_ego_state.t,float) and isinstance(self.cur_tar_state.t,float) and self.cur_ego_state.p.s is not None and self.cur_tar_state.p.s is not None and abs(self.cur_tar_state.p.x_tran) < self.track_info.track.track_width and abs(self.cur_ego_state.p.x_tran) < self.track_info.track.track_width:
                    if self.pred_data_save:
                        if self.tv_pred is None:                                                 
                            return
                        else:
                            self.tar_pred_list.append(self.tv_pred)
                    
                    self.ego_list.append(self.cur_ego_state.copy())
                    self.tar_list.append(self.cur_tar_state.copy())                     
                    callback_time = self.ego_list[-1].t                            
                    self.prev_dl_time = callback_time                    
                                        
                    if len(self.tar_list) > self.save_buffer_length:
                        self.save_buffer_in_thread()
                elif len(self.tar_list) > 0 and len(self.ego_list) > 0: 
                    self.save_buffer_in_thread()   
        
            
            



    
    
###################################################################################

def main():
    rospy.init_node("recorder")    
    Recorder()

if __name__ == "__main__":
    main()




 
    


