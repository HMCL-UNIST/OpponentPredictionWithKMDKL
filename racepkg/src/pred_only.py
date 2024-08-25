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

import rospy
import time
import os
import numpy as np 
import threading
import rospkg
import torch 
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from hmcl_msgs.msg import VehiclePredictionROS 
from message_filters import ApproximateTimeSynchronizer, Subscriber
from racepkg.path_generator import PathGenerator
from racepkg.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehicleActuation
from racepkg.utils import pose_to_vehicleState, odom_to_vehicleState, prediction_to_marker, fill_global_info
from racepkg.prediction.covGP.covGPNN_predictor import CovGPPredictor
from racepkg.prediction.multipathPP.multipathPP_predictor import MultipathPPPredictor
from racepkg.prediction.trajectory_predictor import ConstantAngularVelocityPredictor
from racepkg.common.utils.file_utils import *
from racepkg.common.utils.scenario_utils import RealData
from racepkg.utils import prediction_to_rosmsg, rosmsg_to_prediction
from racepkg.h2h_configs import *

from dynamic_reconfigure.server import Server
from racepkg.cfg import racepkgDynConfig



rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('racepkg')


class Predictor:
    def __init__(self):             
        self.n_nodes = rospy.get_param('~n_nodes', default=12)
        self.t_horizon = rospy.get_param('~t_horizon', default=1.2)                   
        self.torch_device = "cuda:0"   ## Specify the name of GPU         
        self.dt = self.t_horizon / self.n_nodes*1.0        
        ## 
        self.track_info = PathGenerator()        
        while self.track_info.track_ready is False:
             rospy.sleep(0.01)
        ##                
        M = 25
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
        

        ## controller callback        
        self.ego_list = []
        self.tar_list = []
        self.tar_pred_list = []        

        self.data_save = False
        self.prev_data_save = False
        self.pred_data_save = False
        self.save_buffer_length = 200

        self.ego_pred = None
        self.gt_tar_pred = None
        self.ego_prev_pose = None
        self.tar_prev_pose = None
        self.ego_local_vel = None
        self.tar_local_vel = None

        # Publishers                
        self.tv_pred_marker_pub = rospy.Publisher('/tv_pred_marker',MarkerArray,queue_size = 2)                        
        self.tar_pred_pub = rospy.Publisher("/tar_pred", VehiclePredictionROS, queue_size=2)

        # Subscribers
        self.ego_pred_sub = rospy.Subscriber('/ego_pred', VehiclePredictionROS, self.ego_pred_callback)          
        self.gt_tar_pred_sub = rospy.Subscriber('/gt_tar_pred', VehiclePredictionROS, self.gt_tar_pred_callback)          
        self.ego_odom_sub = rospy.Subscriber(ego_odom_topic, Odometry, self.ego_odom_callback)                                                  
        self.target_odom_sub = rospy.Subscriber(target_odom_topic, Odometry, self.target_odom_callback)                             
        self.ego_pose_sub = Subscriber(ego_pose_topic, PoseStamped)        
        self.target_pose_sub = Subscriber(target_pose_topic, PoseStamped)        
        self.ats = ApproximateTimeSynchronizer([self.ego_pose_sub,  self.target_pose_sub], queue_size=10, slop=0.05)
        self.sync_prev_time = rospy.Time.now().to_sec()
        self.ats.registerCallback(self.sync_callback)
        # predictor type = 0 : ThetaGP
        #                   1 : CAV
        #                   2: NMPC
        #                   3 : GP
        #                   4: COVGP
        #                   5: Multipathpp

        self.predictor_type = 0  ## initialize the predictor        
                
        args = {   "batch_size": 1024,
                    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                    "input_dim": 10,
                    "n_time_step": 10,
                    "latent_dim": 6,
                    "gp_output_dim": 4,
                    "inducing_points" : 200,
                    "train_nn" : False,
                    "include_kml_loss" : True,
                    "direct_gp" : False,
                    "n_epoch" : 10000,
                    'model_name' : None,
                    'eval' : False,
                    'load_eval_data' : False
                    }        

        self.cav_predictor = ConstantAngularVelocityPredictor(N=self.n_nodes, cov= .01)            
        self.gp_predictor = CovGPPredictor(N=self.n_nodes, track=self.track_info.track,  use_GPU=True, M=M, cov_factor=np.sqrt(2.0), input_predict_model = "naiveGP", args= args.copy())                            
        self.nosim_predictor = CovGPPredictor(N=self.n_nodes, track=self.track_info.track,  use_GPU=True, M=M, cov_factor=np.sqrt(2.0), input_predict_model = "nosimtsGP", args= args.copy())                            
        self.predictor = CovGPPredictor(N=self.n_nodes, track=self.track_info.track,  use_GPU=True, M=M, cov_factor=np.sqrt(2.0), input_predict_model = "simtsGP", args= args.copy())                    
        self.multipathpp_predictor = MultipathPPPredictor(N=self.n_nodes, track=self.track_info.track,  use_GPU=True, M=M, cov_factor=np.sqrt(2.0), input_predict_model = "multipathpp", args= args.copy())                    
        
        self.dyn_srv = Server(racepkgDynConfig, self.dyn_callback)
        self.prediction_hz = rospy.get_param('~prediction_hz', default=10)
        self.prediction_timer = rospy.Timer(rospy.Duration(1/self.prediction_hz), self.prediction_callback)         
        self.data_logging_hz = rospy.get_param('~data_logging_hz', default=10)
        self.prev_dl_time = rospy.Time.now().to_sec()
  
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True            
            rate.sleep()

 
    def ego_pred_callback(self,msg):
        self.ego_pred = rosmsg_to_prediction(msg)
    
    def gt_tar_pred_callback(self,msg):
        self.gt_tar_pred = rosmsg_to_prediction(msg)
        
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
               
    def datalogging_callback(self):      
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
                elif len(self.tar_list) > 0 and len(self.ego_list) > 0: ## if suddent crash or divergence in local, save data and do not continue from the next iteration
                    self.save_buffer_in_thread()   
        
    def prediction_callback(self,event):
        
        start_time = time.time()
        if self.ego_odom_ready is False or self.tar_odom_ready is False or self.ego_pred is None:                    
            return
        
        if self.ego_odom_ready and self.tar_odom_ready:
            pose_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_pose)
            odom_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_odom)
            pose_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_pose)            
            odom_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_odom)            
        else:
            rospy.loginfo("state not ready")
            return 

        if self.predictor and self.cur_ego_state is not None:                            
            self.use_predictions_from_module = True            
            if self.cur_ego_state.t is not None and self.cur_tar_state.t is not None and self.ego_pred.x is not None:            
                if self.predictor_type == 4:   ## KM-DKL predictor               
                    self.tv_pred = self.predictor.get_prediction(self.cur_ego_state, self.cur_tar_state, self.ego_pred)
                elif self.predictor_type == 1: ## CAV predictor
                    self.tv_pred = self.cav_predictor.get_prediction(ego_state = self.cur_ego_state, target_state = self.cur_tar_state, ego_prediction = self.ego_pred)                                                                   
                elif self.predictor_type == 2: ## GT MPCC predictor
                    self.tv_pred = self.gt_tar_pred
                    self.tv_pred.xy_cov = np.repeat(np.diag([0.01, 0.01])[np.newaxis, :, :], len(self.tv_pred.s), axis=0)                                        
                elif self.predictor_type == 3: ## GP Predictor
                    self.tv_pred = self.gp_predictor.get_prediction(ego_state = self.cur_ego_state, target_state = self.cur_tar_state, ego_prediction = self.ego_pred)
                elif self.predictor_type ==0: ## DKL predictor                
                    self.tv_pred = self.nosim_predictor.get_prediction(ego_state = self.cur_ego_state, target_state = self.cur_tar_state, ego_prediction = self.ego_pred)
                elif self.predictor_type == 5: ## multipathpp
                    self.tv_pred = self.multipathpp_predictor.get_prediction(ego_state = self.cur_ego_state, target_state = self.cur_tar_state, ego_prediction = self.ego_pred)
                
                else: 
                    print("select between #0 ~ #4 predictor")                
                           
                tar_pred_msg = None                                
                if self.tv_pred is not None:            
                    fill_global_info(self.track_info.track, self.tv_pred)                    
                    tar_pred_msg = prediction_to_rosmsg(self.tv_pred)   
                    tv_pred_markerArray = prediction_to_marker(self.tv_pred)                    
                    self.datalogging_callback()
                    
                if tar_pred_msg is not None:
                    self.tar_pred_pub.publish(tar_pred_msg)          
                    self.tv_pred_marker_pub.publish(tv_pred_markerArray)                        
                
        end_time = time.time()
        execution_time = end_time - start_time        

###################################################################################

def main():
    rospy.init_node("predictor")    
    Predictor()

if __name__ == "__main__":
    main()




 
    


