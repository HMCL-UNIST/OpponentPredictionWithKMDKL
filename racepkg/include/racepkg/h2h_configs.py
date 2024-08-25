'''
MIT License

Copyright (c) 2022 Model Predictive Control (MPC) Laboratory

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
from racepkg.common.pytypes import *
from racepkg.controllers.utils.controllerTypes import *
from racepkg.dynamics.models.model_types import DynamicBicycleConfig

from enum import Enum
import math

class Predictor(Enum):
    GroundTruth = 0
    LSTM = 1
    NMPC = 2
    DirectGP = 3
    AutoGP = 4 
    ConstantInput = 5 
    COVGP = 6          

class Controllers(Enum):
    NMPC = 0
    MPPI = 1
    GPNMPC = 2


# Time discretization
dt = 0.1
# Horizon length
N = 12
# Number of iterations to run PID (need N+1 because of NLMPC predictor warmstart)
n_iter = N+1 
# Track width (should be pre-determined from track generation '.npz')
width = 1.7

# Force rebuild all FORCES code-gen controllers
rebuild = False
# Use a key-point lookahead strategy that is dynamic (all_tracks=True) or pre-generated (all_tracks=False)
all_tracks = True
offset = 32 if not all_tracks else 0

ego_L = 0.33
ego_W = 0.173



tar_L = 0.33
tar_W = 0.173


# Initial track conditions
factor = 1.3  # v_long factor
tarMin = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 2.0, x_tran=-.3 * width, e_psi=-0.02),
                      v=BodyLinearVelocity(v_long=0.8*factor))
tarMax = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 2.2, x_tran=.3* width, e_psi=0.02),
                      v=BodyLinearVelocity(v_long=1.0*factor))
egoMin = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 0.2, x_tran=-.3 * width, e_psi=-0.02),
                      v=BodyLinearVelocity(v_long=0.5*factor))
egoMax = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 0.4, x_tran=.3 * width, e_psi=0.02),
                      v=BodyLinearVelocity(v_long=1.0*factor))



tar_dynamics_config = DynamicBicycleConfig(dt=dt, model_name='dynamic_bicycle_full',
                                           wheel_dist_front=0.165, wheel_dist_rear=0.165, slip_coefficient=.9)
ego_dynamics_config = DynamicBicycleConfig(dt=dt, model_name='dynamic_bicycle_full',
                                           wheel_dist_front=0.165, wheel_dist_rear=0.165, slip_coefficient=.9)

# Controller parameters
gp_mpcc_ego_params = MPCCApproxFullModelParams(
    dt=dt,
    all_tracks=all_tracks,
    solver_dir='' if rebuild else '~/.mpclab_controllers/gp_mpcc_h2h_ego',
    # solver_dir='',
    optlevel=2,

    N=N,
    Qc=5.0, # e_cont , countouring error 
    Ql=500.0, #500.0  # e_lag, lag error 
    Q_theta= 200, # progress speed  v_proj_prev 

    Q_xref=0.0, #  reference tracking for blocking 
    R_d=2.0, # u_a, u_a_dot 
    R_delta=20.0, # 20.0 # u_delta, u_delta_dot

    slack=True,
    l_cs=5, # obstacle_slack
    Q_cs=2.0, # # obstacle_slack_e
    Q_vmax=200.0,    
    vlong_max_soft= 1.845,     
    Q_ts=5000.0, # track boundary
    Q_cs_e=8.0, # obstacle slack
    l_cs_e=35.0,  # obstacle slack
    num_std_deviations= 0.01,
    u_a_max=1.5,     
    vx_max=  1.85, 
    u_a_min=-1.5,
    u_steer_max=0.435,
    u_steer_min=-0.435,
    u_a_rate_max=10,
    u_a_rate_min=-10,
    u_steer_rate_max=2,
    u_steer_rate_min=-2
)



mpcc_tv_params = MPCCApproxFullModelParams(
    dt=dt,
    all_tracks=all_tracks,
    solver_dir='' if rebuild else '~/.mpclab_controllers/mpcc_tv_params',
    # solver_dir='',
    optlevel=2,

    N=N,
    # Qc=300.0, # e_cont , countouring error  10 for blocking 300 for non blockign
    Qc=10.0, # e_cont , countouring error  10 for blocking 300 for non blockign
    
    Ql=500.0, #500.0  # e_lag, lag error 
    Q_theta= 200, # progress speed  v_proj_prev 
    
    # Q_xref=0.0, #  reference tracking for blocking  500 for blocking, 0 for non blocking
    Q_xref=500.0, #  reference tracking for blocking  500 for blocking, 0 for non blocking
    
    R_d=2.0, # u_a, u_a_dot 
    R_delta=20.0, # 20.0 # u_delta, u_delta_dot

    slack=True,
    l_cs=5, # obstacle_slack
    Q_cs=2.0, # # obstacle_slack_e
    Q_vmax=200.0,
    vlong_max_soft=1.6,     
    Q_ts=500.0, # track boundary
    Q_cs_e=8.0, # obstacle slack
    l_cs_e=35.0,  # obstacle slack

    num_std_deviations= 0.1, # 0.01

    u_a_max=1.75,
    vx_max=1.65,        
    u_a_min=-2.0,
    u_steer_max=0.435,
    u_steer_min=-0.435,
    u_a_rate_max=10,
    u_a_rate_min=-10,
    u_steer_rate_max=2,
    u_steer_rate_min=-2
)


mpcc_passive_params = MPCCApproxFullModelParams(
    dt=dt,
    all_tracks=all_tracks,
    solver_dir='' if rebuild else '~/.mpclab_controllers/mpcc_passive_params',
    # solver_dir='',
    optlevel=2,

   
    N=N,
    Qc=300.0, # e_cont , countouring error  10 for blocking 300 for non blockign
    # Qc=10.0, # e_cont , countouring error  10 for blocking 300 for non blockign
    
    Ql=500.0, #500.0  # e_lag, lag error 
    Q_theta= 200, # progress speed  v_proj_prev 


    Q_xref=0.0, #  reference tracking for blocking  500 for blocking, 0 for non blocking
    # Q_xref=500.0, #  reference tracking for blocking  500 for blocking, 0 for non blocking
    
    R_d=2.0, # u_a, u_a_dot 
    R_delta=20.0, # 20.0 # u_delta, u_delta_dot

    slack=True,
    l_cs=5, # obstacle_slack
    Q_cs=2.0, # # obstacle_slack_e
    Q_vmax=200.0,
    vlong_max_soft=1.6, ##0.8 reference speed .. only activate if speed exceeds it         
    Q_ts=500.0, # track boundary
    Q_cs_e=8.0, # obstacle slack
    l_cs_e=35.0,  # obstacle slack

    num_std_deviations= 0.1, # 0.01

    u_a_max=1.75,
    vx_max=1.65,        
    u_a_min=-2.0,
    u_steer_max=0.435,
    u_steer_min=-0.435,
    u_a_rate_max=10,
    u_a_rate_min=-10,
    u_steer_rate_max=2,
    u_steer_rate_min=-2
)


mpcc_timid_params = MPCCApproxFullModelParams(
    dt=dt,
    all_tracks=all_tracks,
    solver_dir='' if rebuild else '~/.mpclab_controllers/gp_mpcc_h2h_timid',
    # solver_dir='',
    optlevel=2,

    num_std_deviations= 0.1, # 0.01

    N=N,
    Qc=1000.0, # e_cont , countouring error  10 for blocking 300 for non blockign
    # Qc=10.0, # e_cont , countouring error  10 for blocking 300 for non blockign
    
    Ql=1000.0, #500.0  # e_lag, lag error 
    Q_theta= 100, # progress speed  v_proj_prev 


    # Q_xref=0.0, #  reference tracking for blocking  500 for blocking, 0 for non blocking
    Q_xref=0.0, #  reference tracking for blocking  500 for blocking, 0 for non blocking
    
    R_d=2.0, # u_a, u_a_dot 
    R_delta=10.0, # 20.0 # u_delta, u_delta_dot

    slack=True,
    l_cs=5, # obstacle_slack
    Q_cs=2.0, # # obstacle_slack_e
    Q_vmax=200.0,
    vlong_max_soft=1.1, ##0.8 reference speed .. only activate if speed exceeds it     
    Q_ts=500.0, # track boundary
    Q_cs_e=8.0, # obstacle slack
    l_cs_e=35.0,  # obstacle slack


    u_a_max=1.55,
    vx_max=1.65,    
    u_a_min=-2.0,
    u_steer_max=0.43,
    u_steer_min=-0.43,
    u_a_rate_max=10,
    u_a_rate_min=-10,
    u_steer_rate_max=1,
    u_steer_rate_min=-1

)

# For NLMPC predictor
nl_mpc_params = NLMPCParams(
        dt=dt,
        solver_dir='' if rebuild else '~/.mpclab_controllers/NL_MPC_solver_forces_pro',
        # solver_dir='',
        optlevel=2,
        slack=False,

        N=N,
        Q=[10.0, 0.2, 1, 15, 0.0, 25.0], # .5 10
        R=[0.1, 0.1],
        Q_f=[10.0, 0.2, 1, 17.0, 0.0, 1.0], # .5 10
        R_d=[5.0, 5.0],
        Q_s=0.0,
        l_s=50.0,

        x_tran_max=width/2,
        x_tran_min=-width/2,
        u_steer_max=0.43,
        u_steer_min=-0.435,
        u_a_max=1.55,
        u_a_min=-2.0,
        u_steer_rate_max=1,
        u_steer_rate_min=-1,
        u_a_rate_max=10.0,
        u_a_rate_min=-10.0
    )
