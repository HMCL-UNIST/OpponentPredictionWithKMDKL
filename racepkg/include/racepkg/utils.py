
"""   
 Software License Agreement (BSD License)
 Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************   
"""
#!/usr/bin/env python3
import tf 
import array
import rospy
import math
import pyquaternion
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import torch
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from racepkg.common.tracks.radius_arclength_track import RadiusArclengthTrack
from racepkg.common.pytypes import VehicleState, VehiclePrediction
from typing import List
from hmcl_msgs.msg import VehiclePredictionROS
from racepkg.common.utils.file_utils import *
from geometry_msgs.msg import PoseStamped, Vector3
from tf.transformations import euler_from_quaternion

def compute_local_velocity(pose1:PoseStamped, pose2:PoseStamped):
    time_diff = (pose2.header.stamp - pose1.header.stamp).to_sec()
    # Extract positions and orientations from PoseStamped messages
    pos1 = pose1.pose.position
    pos2 = pose2.pose.position
    quat1 = (pose1.pose.orientation.x, pose1.pose.orientation.y,
             pose1.pose.orientation.z, pose1.pose.orientation.w)
    quat2 = (pose2.pose.orientation.x, pose2.pose.orientation.y,
             pose2.pose.orientation.z, pose2.pose.orientation.w)

    # Convert orientations to Euler angles
    _, _, yaw1 = euler_from_quaternion(quat1)
    _, _, yaw2 = euler_from_quaternion(quat2)

    # Calculate local velocities
    delta_x = pos2.x - pos1.x
    delta_y = pos2.y - pos1.y
    delta_yaw = yaw2 - yaw1

    # Calculate local velocity components

    vx = delta_x / time_diff
    vy = delta_y / time_diff
    omega = delta_yaw / time_diff

    local_velocity = Vector3(vx, vy, omega)
    rotated_velocity = rotate_vector_by_yaw(local_velocity, yaw1)  # Rotating by yaw1
   
    return rotated_velocity

def rotate_vector_by_yaw(vector, yaw):
    # Rotate a 2D vector by an angle in radians (yaw)
    rotated_x = vector.x * math.cos(yaw) - vector.y * math.sin(yaw)
    rotated_y = vector.x * math.sin(yaw) + vector.y * math.cos(yaw)
    rotated_z = vector.z  # Angular velocity remains unchanged

    return Vector3(rotated_x, rotated_y, rotated_z)


def shift_in_local_x(pose_msg: PoseStamped, dist = -0.13):
        # Convert orientation to a rotation matrix
    position = pose_msg.pose.position
    orientation = pose_msg.pose.orientation
    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    rotation_matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]

    # Create a translation vector representing 1 meter displacement in the local coordinate system
    local_translation = np.array([dist, 0., 0.])  # Adjust the values based on your desired displacement
    # Transform the translation vector from local to global coordinate system
    global_translation = rotation_matrix.dot(local_translation)
    # Add the transformed translation vector to the original position
    new_position = np.array([position.x, position.y, position.z]) + global_translation
    # Update the position values in the PoseStamped message
    pose_msg.pose.position.x = new_position[0]
    pose_msg.pose.position.y = new_position[1]


def pose_to_vehicleState(track: RadiusArclengthTrack, state : VehicleState,pose : PoseStamped, line=None):
    state.x.x = pose.pose.position.x
    state.x.y = pose.pose.position.y
    orientation_q = pose.pose.orientation    
    quat = [orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z]
    (cur_roll, cur_pitch, cur_yaw) = quaternion_to_euler (quat)
    state.e.psi = cur_yaw
    xy_coord = (state.x.x, state.x.y, state.e.psi)
    cl_coord = track.global_to_local(xy_coord, line= line)
    if cl_coord is None:
        print('cl_coord is none')
        return
    state.t = pose.header.stamp.to_sec()
    state.p.s = cl_coord[0]
    ##################
    ## WHen track is doubled... wrap with half track length
    if state.p.s > track.track_length:
        state.p.s -= track.track_length
    ###################
    state.p.x_tran = cl_coord[1]
    state.p.e_psi = cl_coord[2]
    track.update_curvature(state)
    
def odom_to_vehicleState(track: RadiusArclengthTrack, state:VehicleState, odom: Odometry):   
    local_vel = get_local_vel(odom, is_odom_local_frame = False)
    if local_vel is None: 
        return 
    
    state.v.v_long = local_vel[0]
    state.v.v_tran = local_vel[1]
    state.w.w_psi = odom.twist.twist.angular.z


def dist2d(point1, point2):
    """
    Euclidean distance between two points
    :param point1:
    :param point2:
    :return:
    """

    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]

    dist2 = (x1 - x2)**2 + (y1 - y2)**2

    return math.sqrt(dist2)

    
def b_to_g_rot(r,p,y): 
    row1 = torch.transpose(torch.stack([torch.cos(p)*torch.cos(y), -1*torch.cos(p)*torch.sin(y), torch.sin(p)]),0,1)
    row2 = torch.transpose(torch.stack([torch.cos(r)*torch.sin(y)+torch.cos(y)*torch.sin(r)*torch.sin(p), torch.cos(r)*torch.cos(y)-torch.sin(r)*torch.sin(p)*torch.sin(y), -torch.cos(p)*torch.sin(r)]),0,1)
    row3 = torch.transpose(torch.stack([torch.sin(r)*torch.sin(y)-torch.cos(r)*torch.cos(y)*torch.sin(p), torch.cos(y)*torch.sin(r)+torch.cos(r)*torch.sin(p)*torch.sin(y), torch.cos(r)*torch.cos(p)]),0,1)
    rot = torch.stack([row1,row2,row3],dim = 1)
    return rot


def wrap_to_pi(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi-0.01:
        angle -= 2.0 * np.pi

    while angle < -np.pi+0.01:
        angle += 2.0 * np.pi

    return angle 


def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

def wrap_to_pi_torch(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    return (((angle + torch.pi) % (2 * torch.pi)) - torch.pi)
    
    



def get_odom_euler(odom):    
    q = pyquaternion.Quaternion(w=odom.pose.pose.orientation.w, x=odom.pose.pose.orientation.x, y=odom.pose.pose.orientation.y, z=odom.pose.pose.orientation.z)
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    q_norm = np.sqrt(np.sum(q ** 2))
 
    return 1 / q_norm * q

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    
    return unit_quat(np.array([qw, qx, qy, qz]))



def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]    
    rot_mat = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])
    return rot_mat


def get_local_vel(odom, is_odom_local_frame = False):
    local_vel = np.array([0.0, 0.0, 0.0])
    if is_odom_local_frame is False: 
        # convert from global to local 
        q_tmp = np.array([odom.pose.pose.orientation.w,odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z])
        euler = get_odom_euler(odom)
        rot_mat_ = q_to_rot_mat(q_tmp)
        inv_rot_mat_ = np.linalg.inv(rot_mat_)
        global_vel = np.array([odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.linear.z])
        local_vel = inv_rot_mat_.dot(global_vel)        
    else:
        local_vel[0] = odom.twist.twist.linear.x
        local_vel[1] = odom.twist.twist.linear.y
        local_vel[2] = odom.twist.twist.linear.z
    return local_vel 


def traj_to_markerArray(traj):

    marker_refs = MarkerArray() 
    for i in range(len(traj[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "ref_states"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.ARROW
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = traj[i,0] 
        marker_ref.pose.position.y = traj[i,1]              
        quat_tmp = euler_to_quaternion(0.0, 0.0, traj[i,3])     
        quat_tmp = unit_quat(quat_tmp)                 
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (0, 255, 0)
        marker_ref.color.a = 0.2
        marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.2, 0.2, 0.15)
        marker_refs.markers.append(marker_ref)
        

    return marker_refs

def predicted_distribution_traj_visualize(x_mean,x_var,y_mean,y_var,mean_predicted_state,color):
    marker_refs = MarkerArray() 
    for i in range(len(x_mean)):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "gplogger_ref"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = x_mean[i]
        marker_ref.pose.position.y = y_mean[i]
        marker_ref.pose.position.z = mean_predicted_state[i,6]  
        quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state[i,2])             
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (color[0], color[1], color[2])
        marker_ref.color.a = 0.5                
        marker_ref.scale.x = 2*np.sqrt(x_var[i])
        marker_ref.scale.y = 2*np.sqrt(y_var[i])
        marker_ref.scale.z = 1
        marker_refs.markers.append(marker_ref)
        i+=1
    return marker_refs


def predicted_trj_visualize(predicted_state,color):        
    marker_refs = MarkerArray() 
    for i in range(len(predicted_state[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "gplogger_ref"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.ARROW
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = predicted_state[i,0] 
        marker_ref.pose.position.y = predicted_state[i,1]              
        marker_ref.pose.position.z = predicted_state[i,6]  
        quat_tmp = euler_to_quaternion(0.0, 0.0, predicted_state[i,2])     
        quat_tmp = unit_quat(quat_tmp)                 
        marker_ref.pose.orientation.w = quat_tmp[0]
        marker_ref.pose.orientation.x = quat_tmp[1]
        marker_ref.pose.orientation.y = quat_tmp[2]
        marker_ref.pose.orientation.z = quat_tmp[3]
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (color[0], color[1], color[2])
        marker_ref.color.a = 0.5                
        marker_ref.scale.x = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_ref.scale.y = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_ref.scale.z = (i+1)/len(predicted_state[:,0])*0.1+0.1
        marker_refs.markers.append(marker_ref)
        i+=1
    return marker_refs


def ref_to_markerArray(traj):
    marker_refs = MarkerArray() 
    for i in range(len(traj[:,0])):
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.ns = "ref_states_"+str(i)
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = traj[i,0] 
        marker_ref.pose.position.y = traj[i,1]              
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (0, 0, 255)
        marker_ref.color.a = 0.5
        marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.1, 0.1, 0.1)
        marker_refs.markers.append(marker_ref)
    return marker_refs



def multi_predicted_distribution_traj_visualize(x_mean_set,x_var_set,y_mean_set,y_var_set,mean_predicted_state_set):
    marker_refs = MarkerArray() 
    for j in range(len(x_mean_set)):
        for i in range(len(x_mean_set[j])):
            marker_ref = Marker()
            marker_ref.header.frame_id = "map"  
            marker_ref.ns = "gplogger_ref"+str(i)+str(j)
            marker_ref.id = j*len(x_mean_set[j])+i
            marker_ref.type = Marker.SPHERE
            marker_ref.action = Marker.ADD                
            marker_ref.pose.position.x = x_mean_set[j][i]
            marker_ref.pose.position.y =  y_mean_set[j][i]
            marker_ref.pose.position.z = mean_predicted_state_set[j,i,6]  
            quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state_set[j,i,2])             
            marker_ref.pose.orientation.w = quat_tmp[0]
            marker_ref.pose.orientation.x = quat_tmp[1]
            marker_ref.pose.orientation.y = quat_tmp[2]
            marker_ref.pose.orientation.z = quat_tmp[3]
            marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (246, 229, 100 + 155/(len(x_mean_set)+0.01)*j)
            marker_ref.color.a = 0.5                    
            marker_ref.scale.x = 0.1 #2*np.sqrt(x_var_set[j][i])
            marker_ref.scale.y = 0.1 #2*np.sqrt(y_var_set[j][i])
            marker_ref.scale.z = 0.1
            marker_refs.markers.append(marker_ref)
            i+=1
    return marker_refs


def mean_multi_predicted_distribution_traj_visualize(mean_predicted_state_set):
    marker_refs = MarkerArray() 
    for j in range(mean_predicted_state_set.shape[0]):        
        for i in range(mean_predicted_state_set.shape[1]):
            marker_ref = Marker()
            marker_ref.header.frame_id = "map"  
            marker_ref.ns = "mean_ref"+str(i)+str(j)
            marker_ref.id = j*mean_predicted_state_set.shape[1]+i
            marker_ref.type = Marker.SPHERE
            marker_ref.action = Marker.ADD                
            marker_ref.pose.position.x = mean_predicted_state_set[j,i,0] 
            marker_ref.pose.position.y = mean_predicted_state_set[j,i,1]              
            marker_ref.pose.position.z = mean_predicted_state_set[j,i,6]  
            quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state_set[j,i,2])     
            quat_tmp = unit_quat(quat_tmp)                 
            marker_ref.pose.orientation.w = quat_tmp[0]
            marker_ref.pose.orientation.x = quat_tmp[1]
            marker_ref.pose.orientation.y = quat_tmp[2]
            marker_ref.pose.orientation.z = quat_tmp[3]
            marker_ref.color.r = 0
            marker_ref.color.g = 255
            marker_ref.color.b = 0 
            marker_ref.color.a = 0.5                
            marker_ref.scale.x = 0.1
            marker_ref.scale.y = 0.1
            marker_ref.scale.z = 0.1
            marker_refs.markers.append(marker_ref)
            i+=1
        

    return marker_refs



def nominal_multi_predicted_distribution_traj_visualize(mean_predicted_state_set):
    marker_refs = MarkerArray() 
    for j in range(mean_predicted_state_set.shape[0]):        
        for i in range(mean_predicted_state_set.shape[1]):
            marker_ref = Marker()
            marker_ref.header.frame_id = "map"  
            marker_ref.ns = "mean_ref"+str(i)+str(j)
            marker_ref.id = j*mean_predicted_state_set.shape[1]+i
            marker_ref.type = Marker.SPHERE
            marker_ref.action = Marker.ADD                
            marker_ref.pose.position.x = mean_predicted_state_set[j,i,0] 
            marker_ref.pose.position.y = mean_predicted_state_set[j,i,1]              
            marker_ref.pose.position.z = mean_predicted_state_set[j,i,6]  
            quat_tmp = euler_to_quaternion(0.0, 0.0, mean_predicted_state_set[j,i,2])     
            quat_tmp = unit_quat(quat_tmp)                 
            marker_ref.pose.orientation.w = quat_tmp[0]
            marker_ref.pose.orientation.x = quat_tmp[1]
            marker_ref.pose.orientation.y = quat_tmp[2]
            marker_ref.pose.orientation.z = quat_tmp[3]
            marker_ref.color.r = 255
            marker_ref.color.g = 0
            marker_ref.color.b = 0 
            marker_ref.color.a = 1.0    
            # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
            marker_ref.scale.x = 0.05
            marker_ref.scale.y = 0.05
            marker_ref.scale.z = 0.05
            marker_refs.markers.append(marker_ref)
            i+=1
        

    return marker_refs

def dist3d(point1, point2):
    """
    Euclidean distance between two points 3D
    :param point1:
    :param point2:
    :return:
    """
    x1, y1, z1 = point1[0:3]
    x2, y2, z2 = point2[0:3]

    dist3d = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2

    return math.sqrt(dist3d)

def gaussianKN2D(rl=4,cl=5, rsig=1.,csig=2.):
    """
    creates gaussian kernel with side length `rl,cl` and a sigma of `rsig,csig`
    """
    rx = np.linspace(-(rl - 1) / 2., (rl - 1) / 2., rl)
    cx = np.linspace(-(cl - 1) / 2., (cl - 1) / 2., cl)
    gauss_x = np.exp(-0.5 * np.square(rx) / np.square(rsig))
    gauss_y = np.exp(-0.5 * np.square(cx) / np.square(csig))
    kernel = np.outer(gauss_x, gauss_y)
    return kernel / (np.sum(kernel)+1e-8)



def torch_path_to_marker(path):
    path_numpy = path.cpu().numpy()
    marker_refs = MarkerArray() 
    marker_ref = Marker()
    marker_ref.header.frame_id = "map"  
    marker_ref.ns = "mppi_ref"
    marker_ref.id = 0
    marker_ref.type = Marker.LINE_STRIP
    marker_ref.action = Marker.ADD     
    marker_ref.scale.x = 0.1 
    for i in range(len(path_numpy[0,:])):                
        point_msg = Point()
        point_msg.x = path_numpy[0,i] 
        point_msg.y = path_numpy[1,i]              
        point_msg.z = path_numpy[3,i] 
        
        color_msg = ColorRGBA()
        color_msg.r = 0.0
        color_msg.g = 0.0
        color_msg.b = 1.0
        color_msg.a = 1.0
        marker_ref.points.append(point_msg)
        marker_ref.colors.append(color_msg)    
    marker_refs.markers.append(marker_ref)
    return marker_refs

def fill_global_info(track,pred):
    if pred.s is not None and len(pred.s) > 0 :
        pred.x = np.zeros(len(pred.s))
        pred.y = np.zeros(len(pred.s))
        pred.psi = np.zeros(len(pred.s))
        for i in range(len(pred.s)):
            cl_coord = [pred.s[i], pred.x_tran[i], pred.e_psi[i]]
            gl_coord = track.local_to_global(cl_coord)
            pred.x[i] = gl_coord[0]
            pred.y[i] = gl_coord[1]
            pred.psi[i] = gl_coord[2]


def prediction_to_marker(predictions):
    
    pred_path_marker_array = MarkerArray()
    if predictions is None or predictions.x is None:
        return pred_path_marker_array
    if len(predictions.x) <= 0:
        return pred_path_marker_array
    for i in range(len(predictions.x)):
  
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.header.stamp = rospy.Time.now()
        marker_ref.ns = "pred"
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = predictions.x[i]
        marker_ref.pose.position.y = predictions.y[i]
        marker_ref.pose.position.z = 0.0        

        marker_ref.lifetime = rospy.Duration(0.2)
        # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
        scale = 1
        if predictions.xy_cov is not None and len(predictions.xy_cov) != 0:
            x_cov = max(predictions.xy_cov[i][0,0],0.01)            
            y_cov = max(predictions.xy_cov[i][1,1],0.01)
        else:
            x_cov = 0.01
            y_cov = 0.01
        marker_ref.scale.x = 2*np.sqrt(x_cov)*scale
        marker_ref.scale.y = 2*np.sqrt(y_cov)*scale
        marker_ref.scale.z = 0.1
        # high uncertainty will get red 
        uncertainty_level = y_cov + x_cov
        # print(uncertainty_level)        
        old_range_min, old_range_max = 0.0, 0.1+0.03*i
        new_range_min, new_range_max = 0, 1
        new_uncertainty = (uncertainty_level - old_range_min) * (new_range_max - new_range_min) / (old_range_max - old_range_min)
        new_uncertainty = max(min(new_uncertainty, new_range_max), new_range_min)  # Clamping to the new range

        

        
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (new_uncertainty, 1-new_uncertainty, 0.0)
        marker_ref.color.a = 0.2     
        pred_path_marker_array.markers.append(marker_ref)
        
    return pred_path_marker_array

def state_prediction_to_marker(predictions,color):
    
    pred_path_marker_array = MarkerArray()
    if predictions is None or predictions.x is None:
        return pred_path_marker_array
    if len(predictions.x) <= 0:
        return pred_path_marker_array
    for i in range(len(predictions.x)):
  
        marker_ref = Marker()
        marker_ref.header.frame_id = "map"  
        marker_ref.header.stamp = rospy.Time.now()
        marker_ref.ns = "pred"
        marker_ref.id = i
        marker_ref.type = Marker.SPHERE
        marker_ref.action = Marker.ADD                
        marker_ref.pose.position.x = predictions.x[i]
        marker_ref.pose.position.y = predictions.y[i]
        marker_ref.pose.position.z = 0.0        
        marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (color[0], color[1], color[2])
        marker_ref.color.a = 0.2     
        marker_ref.lifetime = rospy.Duration(0.2)
        # marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
        scale = 1
        if predictions.xy_cov is not None:
            x_cov = max(predictions.xy_cov[i][0,0],0.00001)
            y_cov = max(predictions.xy_cov[i][1,1],0.00001)
        else:
            x_cov = 0.01
            y_cov = 0.01
        marker_ref.scale.x = 2*np.sqrt(x_cov)*scale
        marker_ref.scale.y = 2*np.sqrt(y_cov)*scale
        marker_ref.scale.z = 0.1
        pred_path_marker_array.markers.append(marker_ref)
        
    return pred_path_marker_array



def prediction_to_rosmsg(vehicle_prediction_obj: VehiclePrediction):
    ros_msg = VehiclePredictionROS()
    ros_msg.header.stamp= rospy.Time.now()
    ros_msg.header.frame_id = "map"
    # Assign values from the VehiclePrediction object to the ROS message
    ros_msg.t = ros_msg.header.stamp.to_sec()
    
    if vehicle_prediction_obj.x is not None:
        ros_msg.x = array.array('f', vehicle_prediction_obj.x)
    if vehicle_prediction_obj.y is not None:
        ros_msg.y = array.array('f', vehicle_prediction_obj.y)
    if vehicle_prediction_obj.v_x is not None:
        ros_msg.v_x = array.array('f', vehicle_prediction_obj.v_x)
    if vehicle_prediction_obj.v_y is not None:
        ros_msg.v_y = array.array('f', vehicle_prediction_obj.v_y)
    if vehicle_prediction_obj.a_x is not None:
        ros_msg.a_x = array.array('f', vehicle_prediction_obj.a_x)
    if vehicle_prediction_obj.a_y is not None:
        ros_msg.a_y = array.array('f', vehicle_prediction_obj.a_y)
    if vehicle_prediction_obj.psi is not None:
        ros_msg.psi = array.array('f', vehicle_prediction_obj.psi)
    if vehicle_prediction_obj.psidot is not None:
        ros_msg.psidot = array.array('f', vehicle_prediction_obj.psidot)
    if vehicle_prediction_obj.v_long is not None:
        ros_msg.v_long = array.array('f', vehicle_prediction_obj.v_long)
    if vehicle_prediction_obj.v_tran is not None:
        ros_msg.v_tran = array.array('f', vehicle_prediction_obj.v_tran)
    if vehicle_prediction_obj.a_long is not None:
        ros_msg.a_long = array.array('f', vehicle_prediction_obj.a_long)
    if vehicle_prediction_obj.a_tran is not None:
        ros_msg.a_tran = array.array('f', vehicle_prediction_obj.a_tran)
    if vehicle_prediction_obj.e_psi is not None:
        ros_msg.e_psi = array.array('f', vehicle_prediction_obj.e_psi)
    if vehicle_prediction_obj.s is not None:
        ros_msg.s = array.array('f', vehicle_prediction_obj.s)
 
    if vehicle_prediction_obj.x_tran is not None:
        ros_msg.x_tran = array.array('f', vehicle_prediction_obj.x_tran)
    if vehicle_prediction_obj.u_a is not None:
        ros_msg.u_a = array.array('f', vehicle_prediction_obj.u_a)
    if vehicle_prediction_obj.u_steer is not None:
        ros_msg.u_steer = array.array('f', vehicle_prediction_obj.u_steer)

    if vehicle_prediction_obj.lap_num is not None:
        ros_msg.lap_num = int(vehicle_prediction_obj.lap_num)
    if vehicle_prediction_obj.sey_cov is not None:             
        ros_msg.sey_cov = vehicle_prediction_obj.sey_cov.tolist()
    if vehicle_prediction_obj.xy_cov is not None:            
        xy_cov_1d = np.array(vehicle_prediction_obj.xy_cov).reshape(-1)        
        ros_msg.xy_cov = xy_cov_1d
    
    return ros_msg  

def prediction_to_std_trace(pred: VehiclePrediction):
    
    xy_cov_trace = 0    
    if pred.xy_cov is not None:            
        xy_cov_1d = np.array(pred.xy_cov).reshape(-1)        
        xy_cov_trace = sum(xy_cov_1d)
    return xy_cov_trace


def rosmsg_to_prediction(ros_msg: VehiclePredictionROS):
    vehicle_prediction_obj = VehiclePrediction()

    # Assign values from the ROS message to the VehiclePrediction object
    vehicle_prediction_obj.t = ros_msg.t
    vehicle_prediction_obj.x = array.array('f', ros_msg.x)
    vehicle_prediction_obj.y = array.array('f', ros_msg.y)
    vehicle_prediction_obj.v_x = array.array('f', ros_msg.v_x)
    vehicle_prediction_obj.v_y = array.array('f', ros_msg.v_y)
    vehicle_prediction_obj.a_x = array.array('f', ros_msg.a_x)
    vehicle_prediction_obj.a_y = array.array('f', ros_msg.a_y)
    vehicle_prediction_obj.psi = array.array('f', ros_msg.psi)
    vehicle_prediction_obj.psidot = array.array('f', ros_msg.psidot)
    vehicle_prediction_obj.v_long = array.array('f', ros_msg.v_long)
    vehicle_prediction_obj.v_tran = array.array('f', ros_msg.v_tran)
    vehicle_prediction_obj.a_long = array.array('f', ros_msg.a_long)
    vehicle_prediction_obj.a_tran = array.array('f', ros_msg.a_tran)
    vehicle_prediction_obj.e_psi = array.array('f', ros_msg.e_psi)
    vehicle_prediction_obj.s = array.array('f', ros_msg.s)
    vehicle_prediction_obj.x_tran = array.array('f', ros_msg.x_tran)
    vehicle_prediction_obj.u_a = array.array('f', ros_msg.u_a)
    vehicle_prediction_obj.u_steer = array.array('f', ros_msg.u_steer)
    vehicle_prediction_obj.lap_num = ros_msg.lap_num
    vehicle_prediction_obj.sey_cov = np.array(ros_msg.sey_cov)
    
    vehicle_prediction_obj.xy_cov = np.array(ros_msg.xy_cov).reshape(-1,2,2)

    return vehicle_prediction_obj


def shift_in_local_x(pose_msg: PoseStamped, dist = -0.13):
        # Convert orientation to a rotation matrix
    position = pose_msg.pose.position
    orientation = pose_msg.pose.orientation
    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    rotation_matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]

    # Create a translation vector representing 1 meter displacement in the local coordinate system
    local_translation = np.array([dist, 0., 0.])  # Adjust the values based on your desired displacement

    # Transform the translation vector from local to global coordinate system
    global_translation = rotation_matrix.dot(local_translation)

    # Add the transformed translation vector to the original position
    new_position = np.array([position.x, position.y, position.z]) + global_translation

    # Update the position values in the PoseStamped message
    pose_msg.pose.position.x = new_position[0]
    pose_msg.pose.position.y = new_position[1]




def state_to_debugmsg(state: VehicleState):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.pose.orientation.x = state.p.s
    msg.pose.orientation.y = state.p.x_tran
    msg.pose.orientation.z = state.p.e_psi

    msg.pose.position.x = state.x.x
    msg.pose.position.y = state.x.y
    msg.pose.position.z = state.e.psi*180.0/3.14195
    return msg

def obstacles_to_markers(obstacles):    
    obs_markers = MarkerArray()    
    if len(obstacles) <= 0:
        print("obstacle length 0")
        return obs_markers
    for i in range(len(obstacles)): 
        obs_marker = Marker()
        obs_marker.header.frame_id = "map"
        obs_marker.header.stamp = rospy.Time.now()
        obs_marker.ns = "line_strip"
        obs_marker.id = i
        obs_marker.type = Marker.LINE_STRIP
        obs_marker.action = Marker.ADD
        obs_marker.pose.orientation.w = 1.0
        obs_marker.scale.x = 0.1  # Line width
        obs_marker.color.a = 1.0  # Alpha
        obs_marker.color.r, obs_marker.color.g, obs_marker.color.b = (0.0, 0.0, 1.0)
        # Define the line strip points
        xy_tmp = obstacles[i].xy        
        for j in range(obstacles[i].xy.shape[0]):            
            point_tmp = Point()
            point_tmp.x = xy_tmp[j,0]
            point_tmp.y = xy_tmp[j,1]
            point_tmp.z = 0.0
            obs_marker.points.append(point_tmp) 

        obs_marker.lifetime = rospy.Duration()
        obs_markers.markers.append(obs_marker)
        
    return obs_markers


def interp_state(state1, state2, t):
    state = state1.copy()    
    dt0 = t - state1.t
    dt = state2.t - state1.t    
    state.t = state1.t+dt
    if abs(state2.p.s - state1.p.s) < 3.0:
        state.p.s = (state2.p.s - state1.p.s) / dt * dt0 + state1.p.s
    state.p.x_tran = (state2.p.x_tran - state1.p.x_tran) / dt * dt0 + state1.p.x_tran
    state.x.x = (state2.x.x - state1.x.x) / dt * dt0 + state1.x.x
    state.x.y = (state2.x.y - state1.x.y) / dt * dt0 + state1.x.y
    state.e.psi = (state2.e.psi - state1.e.psi) / dt * dt0 + state1.e.psi
    state.v.v_long  = (state2.v.v_long - state1.v.v_long) / dt * dt0 + state1.v.v_long
    state.v.v_tran = (state2.v.v_tran - state1.v.v_tran) / dt * dt0 + state1.v.v_tran
    state.w.w_psi  = (state2.w.w_psi - state1.w.w_psi) / dt * dt0 + state1.w.w_psi
    return state




class LaptimeRecorder():
    def __init__(self, track : RadiusArclengthTrack,vehicle_name = 'ego'):
        
        self.n_lap = 0
        self.init_s = 0
        self.track_length = track.track_length
        self.num_max_lap =  np.inf #  3 # np.inf for experiment
        print("track length is set as = " + str(self.track_length))        
        self.laptimes = []
        self.cumulative_laptime = 0.0
        
        self.init_laptime = None
        self.prev_state = None
        self.last_laptime = None
        self.file_name_prefix = vehicle_name+ '_race_state'

        
    
    def update_state(self, cur_state: VehicleState):
        reached_max_lap = False
        if self.prev_state is None:
            self.prev_state = cur_state        
            self.last_laptime = cur_state.t    
            return reached_max_lap
        


        if cur_state.p.s-self.prev_state.p.s < -2:            
            
            self.laptimes.append(cur_state.t - self.last_laptime)            
            self.last_laptime = cur_state.t
            if len(self.laptimes) ==1:
                self.init_laptime = cur_state.t
            if len(self.laptimes) > 5:                                
                file_name = self.file_name_prefix + '_'+str(len(self.laptimes)) + '_' + str(rospy.Time.now().to_sec()) + '.pkl'
                
                self.save(file_name)
            if self.init_laptime is not None:                
                self.n_lap+=1
        
        self.prev_state = cur_state     
        if self.n_lap >= self.num_max_lap:
            reached_max_lap = True
        
        return reached_max_lap  
                

    def get_statistic(self):
        laptime_array = self.get_laptimes()
        avg_laptime = self.get_avg_laptime()
        cum_laptime = self.get_cum_laptime()
        return laptime_array, avg_laptime, cum_laptime

    def get_avg_laptime(self):
        if len(self.laptimes) > 1:
            laptimes = np.array(self.laptimes[1:])
            return np.mean(laptimes)
        else:
            return 0.0
    def get_laptimes(self):
        if len(self.laptimes) > 0:
            return np.array(self.laptimes)
        else:
            return 0.0

    def get_cum_laptime(self):
        if self.last_laptime is not None and self.init_laptime is not None:
            return self.last_laptime - self.init_laptime
        else:
            return 0.0
    
    def save(self,file_name = 'race_statics.pkl'):
        if dir_exists(static_dir) is False:
            create_dir(static_dir,verbose=True)

        laptime_array, avg_laptime, cum_laptime = self.get_statistic()
        stats = dict()
        stats["laptime_array"] = laptime_array
        stats["avg_laptime"] = avg_laptime
        stats["cum_laptime"] = cum_laptime
        pickle_write (stats,os.path.join(static_dir, file_name))


