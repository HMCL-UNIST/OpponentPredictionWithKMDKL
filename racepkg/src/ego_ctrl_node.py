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
import numpy as np
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from hmcl_msgs.msg import VehiclePredictionROS  
import rospkg
from collections import deque
from racepkg.path_generator import PathGenerator
from racepkg.common.pytypes import VehicleState, ParametricPose, BodyLinearVelocity, VehicleActuation
from racepkg.common.utils.file_utils import *
from racepkg.controllers.PID import PIDLaneFollower
from racepkg.controllers.MPCC_H2H_approx import MPCC_H2H_approx
from racepkg.controllers.utils.controllerTypes import PIDParams
from racepkg.dynamics.models.dynamics_models import CasadiDynamicBicycleFull
from racepkg.utils import prediction_to_rosmsg, rosmsg_to_prediction, LaptimeRecorder, pose_to_vehicleState, odom_to_vehicleState, state_prediction_to_marker
from racepkg.simulation.dynamics_simulator import DynamicsSimulator
from racepkg.h2h_configs import *

from dynamic_reconfigure.server import Server
from racepkg.cfg import racepkgDynConfig

rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('racepkg')

class EgoCtrl:
    def __init__(self):       
        
        self.n_nodes = rospy.get_param('~n_nodes', default=12)
        self.t_horizon = rospy.get_param('~t_horizon', default=1.2)                           
        self.dt = self.t_horizon / self.n_nodes*1.0        
        ## 
        # Generate Racing track info 
        self.track_info = PathGenerator()
        
        while self.track_info.track_ready is False:
             rospy.sleep(0.01)
        ##
        
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
        
        # Publishers                
        self.ackman_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/nav_hmcl',AckermannDriveStamped,queue_size = 2)        
        self.ego_pred_marker_pub = rospy.Publisher("/ego_pred_marker", MarkerArray, queue_size=2)  
        self.ego_pred_pub = rospy.Publisher("/ego_pred",VehiclePredictionROS, queue_size= 2) # open loop prediction of EGO
        
        # Subscribers
        self.tv_pred = None
        self.tv_pred_sub = rospy.Subscriber("/tar_pred",VehiclePredictionROS, self.tar_pred_callback)
        self.ego_odom_sub = rospy.Subscriber(ego_odom_topic, Odometry, self.ego_odom_callback)                        
        self.ego_pose_sub = rospy.Subscriber(ego_pose_topic, PoseStamped, self.ego_pose_callback)     
        self.target_odom_sub = rospy.Subscriber(target_odom_topic, Odometry, self.target_odom_callback)                     
        self.target_pose_sub = rospy.Subscriber(target_pose_topic, PoseStamped, self.target_pose_callback)                           
        
        self.vehicle_model = CasadiDynamicBicycleFull(0.0, ego_dynamics_config, track=self.track_info.track)
        self.gp_mpcc_ego_controller = MPCC_H2H_approx(self.vehicle_model, self.track_info.track, control_params = gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name="test_track")
        self.gp_mpcc_ego_controller.initialize()
        self.warm_start()
        
        #### Laptime recorder
        self.laprecorder = LaptimeRecorder(track = self.track_info.track, vehicle_name = 'ego')
        
        ## controller callback
        self.cmd_hz = 20
        self.cmd_timer = rospy.Timer(rospy.Duration(1/self.cmd_hz), self.cmd_callback)         
        
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():            
            msg = Bool()
            msg.data = True
            # self.status_pub.publish(msg)          
            rate.sleep()

    def tar_pred_callback(self,tar_pred_msg):                
        self.tv_pred = rosmsg_to_prediction(tar_pred_msg)

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
    
    def warm_start(self):
        cur_state_copy = self.cur_ego_state.copy()
        x_ref = cur_state_copy.p.x_tran        
        pid_steer_params = PIDParams()
        pid_steer_params.dt = self.dt
        pid_steer_params.default_steer_params()
        pid_steer_params.Kp = 1
        pid_speed_params = PIDParams()
        pid_speed_params.dt = self.dt
        pid_speed_params.default_speed_params()
        pid_controller_1 = PIDLaneFollower(cur_state_copy.v.v_long, x_ref, self.dt, pid_steer_params, pid_speed_params)
        ego_dynamics_simulator = DynamicsSimulator(0.0, ego_dynamics_config, track=self.track_info.track) 
        input_ego = VehicleActuation()
        t = 0.0
        state_history_ego = deque([], self.n_nodes); input_history_ego = deque([], self.n_nodes)
        n_iter = self.n_nodes+1
        approx = True
        while n_iter > 0:
            pid_controller_1.step(cur_state_copy)
            ego_dynamics_simulator.step(cur_state_copy)            
            self.track_info.track.update_curvature(cur_state_copy)
            input_ego.t = t
            cur_state_copy.copy_control(input_ego)
            q, _ = ego_dynamics_simulator.model.state2qu(cur_state_copy)
            u = ego_dynamics_simulator.model.input2u(input_ego)
            if approx:
                q = np.append(q, cur_state_copy.p.s)
                q = np.append(q, cur_state_copy.p.s)
                u = np.append(u, cur_state_copy.v.v_long)
            state_history_ego.append(q)
            input_history_ego.append(u)
            t += self.dt
            n_iter-=1
           
        compose_history = lambda state_history, input_history: (np.array(state_history), np.array(input_history))
        ego_warm_start_history = compose_history(state_history_ego, input_history_ego)        
        self.gp_mpcc_ego_controller.set_warm_start(*ego_warm_start_history)
        print("warm start done")

    def cmd_callback(self,event):
        if self.ego_odom_ready and self.tar_odom_ready:
            pose_to_vehicleState(self.track_info.track, self.cur_ego_state, self.cur_ego_pose)
            odom_to_vehicleState(self.track_info.track,self.cur_ego_state, self.cur_ego_odom)            
            pose_to_vehicleState(self.track_info.track, self.cur_tar_state, self.cur_tar_pose)
            odom_to_vehicleState(self.track_info.track,self.cur_tar_state, self.cur_tar_odom)            
        else:
            rospy.loginfo("state not ready")
            return 
        
        self.use_predictions_from_module = True
        max_lap_reached = self.laprecorder.update_state(self.cur_ego_state.copy())
        if max_lap_reached:
            pp_cmd = AckermannDriveStamped()
            pp_cmd.header.stamp = self.cur_ego_pose.header.stamp               
            pp_cmd.drive.speed = 0.0                    
            self.ackman_pub.publish(pp_cmd)
            print("Reach the max lap")
            return 

        info, b, exitflag = self.gp_mpcc_ego_controller.step(self.cur_ego_state, tv_state=self.cur_tar_state, tv_pred=self.tv_pred if self.use_predictions_from_module else None)        
        ego_state_pred = self.gp_mpcc_ego_controller.get_prediction()
        ego_pred_msg = prediction_to_rosmsg(ego_state_pred)
        self.ego_pred_pub.publish(ego_pred_msg)

        if ego_state_pred is not None and ego_state_pred.x is not None:
            if len(ego_state_pred.x) > 0:
                ego_marker_color = [0.0, 1.0, 0.0]
                ego_state_pred_marker = state_prediction_to_marker(ego_state_pred,ego_marker_color)
                self.ego_pred_marker_pub.publish(ego_state_pred_marker)
        
        pp_cmd = AckermannDriveStamped()
        pp_cmd.header.stamp = self.cur_ego_pose.header.stamp   
        vel_cmd = 0.0            
        if not info["success"]:
            print(f"EGO infeasible - Exitflag: {exitflag}")                            
        else:
            pred_v_lon = self.gp_mpcc_ego_controller.x_pred[:,0] 
            cmd_accel = self.gp_mpcc_ego_controller.x_pred[1,9]             
            # consider the delay in vel command 
            if cmd_accel < 0.0:                
                vel_cmd = pred_v_lon[6]                
            else:            
                vel_cmd = pred_v_lon[4]                         
        if vel_cmd < 0.3:
            vel_cmd = 0.0
        pp_cmd.drive.speed = vel_cmd        
        pp_cmd.drive.steering_angle = self.cur_ego_state.u.u_steer
        self.ackman_pub.publish(pp_cmd)
    
###################################################################################

def main():
    rospy.init_node("egocontroller")    
    EgoCtrl()

if __name__ == "__main__":
    main()




 
    


