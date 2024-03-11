#!/usr/bin/env python3
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
import os
import pickle
import string
from dataclasses import dataclass, field
import random
import torch
from typing import List, Tuple
import numpy as np
import scipy.interpolate

from racepkg.simulation.dynamics_simulator import DynamicsSimulator
from racepkg.h2h_configs import *
from racepkg.common.pytypes import VehicleState, VehiclePrediction, ParametricPose, BodyLinearVelocity
from racepkg.common.tracks.radius_arclength_track import RadiusArclengthTrack
from racepkg.common.tracks.track_lib import CurveTrack, StraightTrack, ChicaneTrack
from racepkg.common.tracks.track import get_track

@dataclass
class PostprocessData:
    N: int = field(default=None)  # Number of runs evaluated
    setup_id: str = field(default=None)  # name of the setup
    name: str = field(default=None)  # name of the run (GP1, ...)
    # Win metrics
    win_ids: List = field(default_factory=lambda: [])  # names of pickles won
    num_wins: int = field(default=0)
    win_rate: float = field(default=0)  # Win Rate: wins/N
    win_rate_nc: float = field(default=0)  # Win Rate: wins/(N - num_crashes)
    # Crash/Rule violation Metrics
    crash_ids: List = field(default_factory=lambda: [])  # ids of crash pickles
    crash_ids_ego: List = field(default_factory=lambda:[])
    crash_ids_tv: List = field(default_factory=lambda:[])
    num_crashes: int = field(default=0)
    crash_rate: float = field(default=0)
    crash_x: List = field(default_factory=lambda: [])  # Crash positions x
    crash_y: List = field(default_factory=lambda: [])  # Crash positions y
    left_track_ids: List = field(default_factory=lambda: [])
    num_left_track: int = field(default=0)
    left_track_rate: float = field(default=0)
    # Overtake Metrics
    overtake_ids: List = field(default_factory=lambda: [])  # name of overtake pickles
    num_overtakes: int = field(default=0)  # Number of overtakes
    overtake_s: List = field(default_factory=lambda: [])  # Overtake positions s
    overtake_x: List = field(default_factory=lambda: [])  # Overtake positions x
    overtake_y: List = field(default_factory=lambda: [])  # Overtake positions y
    avg_delta_s: float = field(default=0)  # Average terminal delta s
    # Actuation metrics
    avg_a: float = field(default=0)  # Average Acceleration
    avg_min_a: float = field(default=0)  # Average minimum Acceleration per run
    avg_max_a: float = field(default=0)  # Average maximum Acceleration per run
    avg_abs_steer: float = field(default=0)  # Average abs steer value
    # Prediction metrics
    lateral_errors: List = field(default_factory=lambda: [])
    longitudinal_errors: List = field(default_factory=lambda: [])
    # Feasibility Data
    ego_infesible_ids: List = field(default_factory=lambda: [])
    tv_infesible_ids: List = field(default_factory=lambda: [])
    num_ego_inf: int = field(default=0)
    ego_inf_rate: float = field(default=0)
    num_tv_inf: int = field(default=0)
    tv_inf_rate: float = field(default=0)
    # Track
    track: RadiusArclengthTrack = field(default=None)

    def post(self):
        self.win_rate = self.num_wins/self.N
        self.win_rate_nc = self.num_wins/(self.N - self.num_crashes)
        self.crash_rate = self.num_crashes/self.N
        self.left_track_rate = self.num_left_track/self.N
        self.ego_inf_rate = self.num_ego_inf/self.N
        self.tv_inf_rate = self.num_tv_inf / self.N

@dataclass
class ScenarioDefinition:
    track_type: string = field(default=None)
    track: RadiusArclengthTrack = field(default=None)
    ego_init_state: VehicleState = field(default=None)
    tar_init_state: VehicleState = field(default=None)
    ego_obs_avoid_d: float = field(default=None)
    tar_obs_avoid_d: float = field(default=None)
    length: float = field(default=None)

    def __post_init__(self):
        self.length = self.track.track_length
        if self.track.phase_out:
            self.length = self.track.track_length - self.track.cl_segs[-1][0]

@dataclass
class RealData():
    track: RadiusArclengthTrack
    N: int
    ego_states: List[VehicleState]
    tar_states: List[VehicleState]
    tar_pred: List[VehiclePrediction] = field(default=List[VehiclePrediction])
    

@dataclass
class SimData():
    scenario_def: ScenarioDefinition
    N: int
    ego_states: List[VehicleState]
    tar_states: List[VehicleState]
    ego_preds: List[VehiclePrediction] = field(default=List[VehiclePrediction])
    tar_preds: List[VehiclePrediction] = field(default=List[VehiclePrediction])

@dataclass
class IkdData():
    scenario_def: ScenarioDefinition
    N: int
    ego_states: List[VehicleState]    
    

@dataclass
class EvalData(SimData):
    tar_gp_pred: List[VehiclePrediction] = field(default=List[VehiclePrediction])
    tar_gp_pred_post: List[VehiclePrediction] = field(default=List[VehiclePrediction])
    tv_infeasible: bool = field(default=False)
    ego_infeasible: bool = field(default=False)

@dataclass
class MultiPolicyEvalData():
    ego_config: MPCCApproxFullModelParams = field(default=MPCCApproxFullModelParams)
    tar_config: MPCCApproxFullModelParams = field(default=MPCCApproxFullModelParams)
    evaldata: EvalData = field(default=EvalData)


@dataclass
class ScenarioGenParams:
    types: list = field(
        default_factory=lambda: ['curve', 'straight', 'chicane'])  # curve, straight, chicane, list of types used
    egoMin: VehicleState = field(default=None)
    egoMax: VehicleState = field(default=None)
    tarMin: VehicleState = field(default=None)
    tarMax: VehicleState = field(default=None)
    width: float = field(default=None)


class ScenarioGenerator:
    def __init__(self, genParams: ScenarioGenParams):
        self.types = genParams.types
        self.egoMin = genParams.egoMin
        self.egoMax = genParams.egoMax
        self.tarMin = genParams.tarMin
        self.tarMax = genParams.tarMax
        self.width = genParams.width

    def genScenarioStatic(self, scenarioType : str, ego_init_state : VehicleState, tar_init_state : VehicleState):
        print('Generating scenario of type:', scenarioType)
        if scenarioType == 'curve':
            return self.genCurve(ego_init_state, tar_init_state)
        elif scenarioType == 'straight':
            return self.genStraight(ego_init_state, tar_init_state)
        elif scenarioType == 'chicane':
            return self.genChicane(ego_init_state, tar_init_state)
        elif scenarioType == 'track':
            return self.getLabTrack(ego_init_state, tar_init_state)

    def genScenario(self):
        """
        Generate random scenario from ego_min, ego_max, tar_min, tar_max
        """
        scenarioType = np.random.choice(self.types)
        s, x_tran, e_psi, v_long = self.randomInit(self.egoMin, self.egoMax)
        ego_init_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                      v=BodyLinearVelocity(v_long=v_long))
        s, x_tran, e_psi, v_long = self.randomInit(self.tarMin, self.tarMax)
        while(abs(s-ego_init_state.p.s) < .5 and abs(x_tran - ego_init_state.p.x_tran) < 0.4):
            s, x_tran, e_psi, v_long = self.randomInit(self.egoMin, self.egoMax)
            ego_init_state = VehicleState(t=0.0,
                                          p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                          v=BodyLinearVelocity(v_long=v_long))
            s, x_tran, e_psi, v_long = self.randomInit(self.tarMin, self.tarMax)
        tar_init_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                      v=BodyLinearVelocity(v_long=v_long))

        return self.genScenarioStatic(scenarioType, ego_init_state, tar_init_state)


    def randomInit(self, stateMin, stateMax):
        s = np.random.uniform(stateMin.p.s, stateMax.p.s)
        x_tran = np.random.uniform(stateMin.p.x_tran, stateMax.p.x_tran)
        e_psi = np.random.uniform(stateMin.p.e_psi, stateMax.p.e_psi)
        v_long = np.random.uniform(stateMin.v.v_long, stateMax.v.v_long)
        # print(s, x_tran, e_psi, v_long)
        return s, x_tran, e_psi, v_long
        # return 0.11891171842527565, -0.48029687041213054, 0*0.007710948256320915, 4.577211702712129

    def getLabTrack(self, ego_init_state, tar_init_state):
        return ScenarioDefinition(
            track_type='track',
            track=get_track('Monza_Track'),
            ego_init_state=ego_init_state,
            tar_init_state=tar_init_state,
            ego_obs_avoid_d=0.1,
            tar_obs_avoid_d=0.1
        )

    def genCurve(self, ego_init_state, tar_init_state):
        alpha = np.random.uniform(np.pi/5, np.pi/1.2)
        alpha = np.random.choice([-1, 1]) * alpha
        phase_out_ = False
        ccw_ = np.random.choice([True, False])
        enter_straight_length_ = np.random.uniform(0.05, 0.1)
        exit_straight_length_ = np.random.uniform(5, 5.1)
        s_min = 1.5
        s = np.random.uniform(s_min*10, s_min * 10.1)
        # s_min = 0.5 * self.width * abs(alpha) * 1.05
        # s = np.random.uniform(s_min, s_min * 2)
        return ScenarioDefinition(
            track_type='curve',
            track=CurveTrack(enter_straight_length=enter_straight_length_,
                             curve_length=s,
                             curve_swept_angle=alpha,
                             exit_straight_length=exit_straight_length_,
                             width=self.width,
                             slack=0.8,
                             phase_out=phase_out_,
                             ccw=ccw_),
            ego_init_state=ego_init_state,
            tar_init_state=tar_init_state,
            ego_obs_avoid_d=0.1,
            tar_obs_avoid_d=0.1
        )

    def genStraight(self, ego_init_state, tar_init_state):
        length_ = np.random.uniform(20, 25)
        return ScenarioDefinition(
            track_type='straight',
            track=StraightTrack(length=length_, width=self.width, slack=0.8, phase_out=False),
            ego_init_state=ego_init_state,
            tar_init_state=tar_init_state,
            ego_obs_avoid_d=0.1,
            tar_obs_avoid_d=0.1
        )

    def genChicane(self, ego_init_state, tar_init_state):
        dir = np.random.choice([-1, 1])
        alpha_1 = np.random.uniform(np.pi / 7, np.pi / 2)
        alpha_1 = dir * alpha_1
        s_min = 0.5 * self.width * np.pi * abs(alpha_1) #/ np.pi
        s_1 = np.random.uniform(s_min, s_min * 5.01)

        alpha_2 = np.random.uniform(np.pi / 7, np.pi / 2)
        alpha_2 = dir * alpha_2
        s_min_2 = 0.5 * self.width * np.pi * abs(alpha_2) # / np.pi
        s_2 = np.random.uniform(s_min_2, s_min_2 * 5.01)
        mid_straght_length_ = np.random.uniform(0.1,0.5)
        return ScenarioDefinition(
            track_type='chicane',
            track=ChicaneTrack(enter_straight_length=0.5,
                               curve1_length=s_1,
                               curve1_swept_angle=alpha_1,
                               mid_straight_length=mid_straght_length_,
                               curve2_length=s_2,
                               curve2_swept_angle=alpha_2,
                               exit_straight_length=2,
                               width=self.width,
                               slack=0.8,
                               phase_out=False,
                               mirror=False),
            ego_init_state=ego_init_state,
            tar_init_state=tar_init_state,
            ego_obs_avoid_d=0.1,
            tar_obs_avoid_d=0.1
        )


@dataclass
class Sample():
    input: Tuple[VehicleState, VehicleState]
    output: VehicleState
    s: float


def policy_generator(dynamics_sim : DynamicsSimulator, state : VehicleState):                        
    ## epsi weight 
    tmp_state = state.copy()
    # steering = -0.5*state.p.e_psi
    for i in range(3):
        steering_val = (dynamics_sim.model.track.track_width - abs(tmp_state.p.x_tran) )/ dynamics_sim.model.track.track_width * 0.43
        if tmp_state.p.x_tran > 0:
            steering = abs(steering_val)
        else:
            steering = -1*abs(steering_val)
        
        tmp_state.u.u_steer = steering
        if dynamics_sim.model.track.track_width < abs(tmp_state.p.x_tran)+0.1:            
            tmp_state.u.u_a = -1.5
        else:
            tmp_state.u.u_a = 1.5    
        dynamics_sim.step(tmp_state)               
    dynamics_sim.model.track.update_curvature(tmp_state)                    
    return tmp_state
    

class SampleGenerator():
    '''
    Class that reads simulation results from a folder and generates samples off that for GP training. Can choose a
    function to determine whether a sample is useful or not.
    '''

    def __init__(self, abs_path, randomize=False, elect_function=None, realdata= False, init_all=True):
        '''
        abs path: List of absolute paths of directories containing files to be used for training
        randomize: boolean deciding whether samples should be returned in a random order or by time and file
        elect_function: decision function to choose samples
        init_all: boolean deciding whether all samples should be preloaded, can be set to False to reduce memory usage if
                        needed TODO not implemented yet!
        '''
        
        if elect_function is None:
            elect_function = self.useAll
        self.counter = 0
        invalid_count = 0
        self.abs_path = abs_path
        self.samples = []
        for ab_p in self.abs_path:
            for filename in os.listdir(ab_p):
                if filename.endswith(".pkl"):
                    dbfile = open(os.path.join(ab_p, filename), 'rb')
                    
                    if realdata:
                        scenario_data: RealData = pickle.load(dbfile)
                        track = scenario_data.track
                        
                    else:
                        scenario_data: SimData = pickle.load(dbfile)
                        track = scenario_data.scenario_def.track
                        
                                            ######################## random Policy ############################
                    if realdata:
                        policy_name = ab_p.split('/')[-1]
                    else:
                        policy_name = ab_p.split('/')[-2]
                    
                    policy_gen = False
                    if policy_name == 'wall':
                        policy_gen = True
                        tar_dynamics_simulator = DynamicsSimulator(0, tar_dynamics_config, track=track)                    
                    
                    ###################################################################
                    N = scenario_data.N
                    self.time_horizon = 0
                    if N > self.time_horizon+1:
                        for i in range(N-1-self.time_horizon*2):
                            if i%2 == 0 and scenario_data.tar_states[i] is not None:
                            # if scenario_data.tar_states[i] is not None:
                                ego_st = scenario_data.ego_states[i]
                                tar_st = scenario_data.tar_states[i]
                                if policy_gen:
                                    scenario_data.tar_states[i+1] = policy_generator(tar_dynamics_simulator,scenario_data.tar_states[i])                  
                                ntar_st = scenario_data.tar_states[i + 1]
                                dtar = tar_st.copy()
                                # dtar.p.s = ntar_st.p.s - tar_st.p.s
                                dtar.p.s = wrap_del_s(ntar_st.p.s, tar_st.p.s,track)
                                dtar.p.x_tran = (ntar_st.p.x_tran - tar_st.p.x_tran)
                                dtar.p.e_psi = ntar_st.p.e_psi - tar_st.p.e_psi
                                dtar.v.v_long = ntar_st.v.v_long - tar_st.v.v_long
                                dtar.w.w_psi = ntar_st.w.w_psi - tar_st.w.w_psi
                                
                                valid_data = False
                                real_dt = ntar_st.t - tar_st.t 
                                if (real_dt > 0.05 and real_dt < 0.2):
                                    valid_data = True
                                

                                if valid_data and elect_function(ego_st, tar_st) and abs(dtar.p.s) < track.track_length/2:
                                    self.samples.append(Sample((ego_st, tar_st), dtar, tar_st.lookahead.curvature[0]))
                                else:
                                    invalid_count +=1
                    dbfile.close()
        print('Generated Dataset with', len(self.samples), 'samples!')
        if randomize:
            random.shuffle(self.samples)

    
    
        


    def reset(self, randomize=False):
        if randomize:
            random.shuffle(self.samples)
        self.counter = 0

    def getNumSamples(self):
        return len(self.samples)

    def nextSample(self):
        self.counter += 1
        if self.counter >= len(self.samples):
            print('All samples returned. To reset, call SampleGenerator.reset(randomize)')
            return None
        else:
            return self.samples[self.counter - 1]



    def even_augment(self, param, maxQ):
        '''
        Augments the dataset to even out distribution of samples by param
        Input:
            param: param to even out by
            maxQ: max factor of frequency of samples
        '''
        data_list = []
        n_bins = 10
        if param == 's':
            for i in self.samples:
                data_list.append(i.s)
        hist, bins = np.histogram(data_list, bins=n_bins)
        maxCount = np.max(hist)  # highest frequency bin
        samples_binned = []
        samples = [k for k in self.samples if k.s < bins[0]]
        samples_binned.append(samples)
        for i in range(n_bins-2):
            samples = [k for k in self.samples if bins[i + 1] > k.s >= bins[i]]
            samples_binned.append(samples)
        samples = [k for k in self.samples if k.s >= bins[-1]]
        samples_binned.append(samples)
        for i in samples_binned:
            if len(i) > 0:
                fac = int(round(maxCount*maxQ/(len(i))))
                print(maxCount*maxQ/ (len(i)), fac, maxCount, len(i))
                for j in range(fac-1):
                    self.samples.extend(i)
        
        self.reset(randomize=True)


    def useAll(self, ego_state, tar_state):
        return True



class SampleGeneartorIKDTime(SampleGenerator):
    def __init__(self, abs_path, randomize=False, elect_function=None, init_all=True):
        '''
        abs path: List of absolute paths of directories containing files to be used for training
        randomize: boolean deciding whether samples should be returned in a random order or by time and file
        elect_function: decision function to choose samples
        init_all: boolean deciding whether all samples should be preloaded, can be set to False to reduce memory usage if
                        needed TODO not implemented yet!
        '''
        self.input_dim = 4
        self.horizon_length = 5
        self.target_horizon = 3
        self.output_dim = 2
        self.input_step_back_idx = self.horizon_length-self.target_horizon     ## 2                        
        
        if elect_function is None:
            elect_function = self.useAll
        self.counter = 0
        self.abs_path = abs_path
        self.samples = []
        self.input_data = []
        self.output_data = []
        for ab_p in self.abs_path:
            for filename in os.listdir(ab_p):
                if filename.endswith(".pkl"):
                    dbfile = open(os.path.join(ab_p, filename), 'rb')
                    scenario_data: IkdData = pickle.load(dbfile)
                    N = scenario_data.N
                       
                    if N > self.horizon_length:
                        for t in range(N-1-self.horizon_length):
                            dat = torch.zeros(self.input_dim,self.horizon_length)
                            for i in range(t,t+self.horizon_length):
                                if i == t+self.target_horizon-1:
                                    target_st = scenario_data.ego_states[i]
                                ego_st = scenario_data.ego_states[i]
                                ego_st_next = scenario_data.ego_states[i+1]                                
                                del_s = ego_st_next.p.s - ego_st.p.s
                                del_ey = ego_st_next.p.x_tran - ego_st.p.x_tran
                                del_epsi = ego_st_next.p.e_psi - ego_st.p.e_psi
                                curvature = ego_st.lookahead.curvature[0]
                                dat[:,i-t] = torch.tensor([del_s, del_ey,del_epsi, curvature])

                            
                            self.input_data.append(dat)                            
                            output = torch.tensor([target_st.u.u_a, target_st.u.u_steer])
                            self.output_data.append(output)
                        
                    
                    dbfile.close()
                
        print('Generated Dataset with', len(self.input_data), 'samples!')
        self.samples = self.input_data 
        # if randomize:
        #     random.shuffle(self.samples)
        
     
    
    def get_datasets(self):        
        not_done = True
        sample_idx = 0
        samp_len = self.getNumSamples()    
       
        train_size = int(0.8 * self.getNumSamples())
        val_size = int(0.1 * self.getNumSamples())
        test_size = self.getNumSamples() - train_size - val_size

        inputs= torch.stack(self.input_data).to(torch.device("cuda"))
        labels = torch.stack(self.output_data).to(torch.device("cuda"))
        perm = torch.randperm(len(inputs))
        inputs = inputs[perm]
        labels = labels[perm]
        dataset =  torch.utils.data.TensorDataset(inputs,labels)                
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

  

        return train_dataset, val_dataset, test_dataset



    





class SampleGeneartorEncoder(SampleGenerator):
    def __init__(self, abs_path, randomize=False, elect_function=None, init_all=True):
        '''
        abs path: List of absolute paths of directories containing files to be used for training
        randomize: boolean deciding whether samples should be returned in a random order or by time and file
        elect_function: decision function to choose samples
        init_all: boolean deciding whether all samples should be preloaded, can be set to False to reduce memory usage if
                        needed TODO not implemented yet!
        '''

        # Input for Encoder -> 
        # [(tar_s-ego_s),
        #  ego_ey, ego_epsi, ego_cur,ego_accel, ego_delta,
        #  tar_ey, tar_epsi, tar_cur,tar_accel, tar_delta] 
        #   x time_horizon
        #          
        self.input_dim = 11
        self.time_horizon = 5
        
        if elect_function is None:
            elect_function = self.useAll
        self.counter = 0
        self.abs_path = abs_path
        self.samples = []
        self.info = []
        
        
        for ab_p in self.abs_path:
            for filename in os.listdir(ab_p):
                if filename.endswith(".pkl"):
                    dbfile = open(os.path.join(ab_p, filename), 'rb')
                    scenario_data: SimData = pickle.load(dbfile)
                    N = scenario_data.N                       
                    if N > self.time_horizon+5:
                        for t in range(N-1-self.time_horizon):                            
                            # define empty torch with proper size 
                            dat = torch.zeros(self.time_horizon, self.input_dim)
                            for i in range(t,t+self.time_horizon):                                
                                ego_st = scenario_data.ego_states[i]
                                tar_st = scenario_data.tar_states[i]
                                # [(tar_s-ego_s), ego_ey, ego_epsi, ego_cur,ego_accel, ego_delta,
                                #                 tar_ey, tar_epsi, tar_cur,tar_accel, tar_delta]                                 
                                dat[i-t,:]=torch.tensor([ tar_st.p.s - ego_st.p.s,
                                                            ego_st.p.x_tran,
                                                            ego_st.p.e_psi,
                                                            ego_st.lookahead.curvature[0],
                                                            ego_st.u.u_a,
                                                            ego_st.u.u_steer,
                                                            tar_st.p.x_tran,
                                                            tar_st.p.e_psi,
                                                            tar_st.lookahead.curvature[0],
                                                            tar_st.u.u_a,
                                                            tar_st.u.u_steer])
                                    
                                
                            self.samples.append(dat)      
                            
                        
                    
                    dbfile.close()
                
        print('Generated Dataset with', len(self.samples), 'samples!')
        
        # if randomize:
        #     random.shuffle(self.samples)
        
     
    
    def get_datasets(self):        
        not_done = True
        sample_idx = 0
        samples= torch.stack(self.samples).to(torch.device("cuda"))  
        samp_len = self.getNumSamples()            
        train_size = int(0.8 * samp_len)
        val_size = int(0.1 * samp_len)
        test_size = samp_len - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(samples, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset



    
        


def derive_lateral_long_error_from_true_traj(sim_data : EvalData):
    lateral_error = list()
    longitudinal_error = list()
    track = sim_data.scenario_def.track
    samps = 0
    for timeStep in range(len(sim_data.tar_states)):
        pred = sim_data.tar_gp_pred[timeStep]  # (VehiclePrediction) at current timestep, what is GP prediction
        if pred is not None:
            N = len(pred.s) if pred.s else len(pred.x)
            if N + timeStep - 1 < len(sim_data.tar_states):
                samps += 1
                for i in range(1, N):
                    tar_st = sim_data.tar_states[timeStep + i - 1]  # (VehicleState) current target state from true traveled trajectory
                    if pred.s:
                        current_x, current_y, current_psi = track.local_to_global(
                            (pred.s[i], pred.x_tran[i], pred.e_psi[i]))
                        track.local_to_global_typed(tar_st)
                    else:
                        current_x, current_y, current_psi = pred.x[i], pred.y[i], pred.psi[i]

                    dx = tar_st.x.x - current_x
                    dy = tar_st.x.y - current_y

                    longitudinal = dx * np.cos(current_psi) + dy * np.sin(current_psi)
                    lateral = -dx * np.sin(current_psi) + dy * np.cos(current_psi)
                    longitudinal_error.append(longitudinal)
                    lateral_error.append(lateral)

    return lateral_error, longitudinal_error

def derive_lateral_long_error_from_MPC_preds(sim_data : EvalData):
    lateral_error = list()
    longitudinal_error = list()
    track = sim_data.scenario_def.track
    samps = 0
    for timeStep in range(len(sim_data.tar_states)):
        pred_gp = sim_data.tar_gp_pred[timeStep]  # (VehiclePrediction) at current timestep, what is GP prediction
        pred_mpc = sim_data.tar_preds[timeStep]  # (VehiclePrediction) at current timestep, what is MPCC prediction
        if pred_gp is not None and pred_mpc is not None:
            N = len(pred_gp.s) if pred_gp.s else len(pred_gp.x)
            if N + timeStep - 1 < len(sim_data.tar_states):
                samps += 1
                for i in range(1, N):
                    if pred_gp.s:
                        current_x, current_y, current_psi = track.local_to_global(
                            (pred_gp.s[i], pred_gp.x_tran[i], pred_gp.e_psi[i]))
                    else:
                        current_x, current_y, current_psi = pred_gp.x[i], pred_gp.y[i], pred_gp.psi[i]

                    dx = pred_mpc.x[i] - current_x
                    dy = pred_mpc.y[i] - current_y

                    longitudinal = dx * np.cos(current_psi) + dy * np.sin(current_psi)
                    lateral = -dx * np.sin(current_psi) + dy * np.cos(current_psi)
                    longitudinal_error.append(longitudinal)
                    lateral_error.append(lateral)

    return lateral_error, longitudinal_error





def evaluateGP(sim_data: SimData, post_gp_eval):
    devs = 0
    devx_tran = 0
    devs_p = 0
    devx_tran_p = 0
    maxs = 0
    max_tran = 0
    maxs_p = 0
    max_tran_p = 0
    samps = 0
    for timeStep in range(len(sim_data.tar_states)):
        pred = sim_data.tar_gp_pred[timeStep]
        if post_gp_eval:
            pred_p = sim_data.tar_gp_pred_post[timeStep]
        s_temp = 0
        tran_temp = 0
        s_temp_p = 0
        tran_temp_p = 0
        if pred is not None:
            N = len(pred.s)
            if N + timeStep - 1 < len(sim_data.tar_states):
                samps += 1
                for i in range(1, N):
                    tar_st = sim_data.tar_states[timeStep + i - 1]
                    ds = (pred.s[i] - tar_st.p.s) ** 2
                    dtran = (pred.x_tran[i] - tar_st.p.x_tran) ** 2
                    s_temp += ds / N
                    tran_temp += dtran / N
                    if ds > maxs:
                        maxs = ds
                    if dtran > max_tran:
                        max_tran = dtran
                    if post_gp_eval:
                        ds_p = (pred_p.s[i] - tar_st.p.s) ** 2
                        dtran_p = (pred_p.x_tran[i] - tar_st.p.x_tran) ** 2
                        s_temp_p +=  ds_p / N
                        tran_temp_p +=  dtran_p/ N
                        if ds_p > maxs_p:
                            maxs_p = ds_p
                        if dtran_p > max_tran_p:
                            max_tran_p = dtran_p
                '''s_temp += (pred.s[i] - tar_st.p.s)**2/(N*(max((ego_pred.s[i] - ego_st.p.s)**2, 0.001)))
                    tran_temp += (pred.x_tran[i] - tar_st.p.x_tran)**2/(N*(max((ego_pred.x_tran[i] - ego_st.p.x_tran)**2, 0.001)))'''
                devs += s_temp
                devx_tran += tran_temp
                if post_gp_eval:
                    devs_p += s_temp_p
                    devx_tran_p += tran_temp_p
    devs /= samps
    devx_tran /= samps
    devs_p /= samps
    devx_tran_p /= samps
    if post_gp_eval:
        print('Real sim predictions')
    print('Avg s pred squared error: ', devs, 'max s squared error ', maxs)
    print('Avg x_tran squared error: ', devx_tran,
          ' max x_tran squared error ', max_tran)
    if post_gp_eval:
        print('Post sim predictions')
        print('Avg s pred accuracy: ', devs_p, 'max s squared error ', maxs_p)
        print('Avg x_tran pred accuracy: ', devx_tran_p,
              ' max x_tran squared error ', max_tran_p)


def post_gp(sim_data: EvalData, gp):
    """
    Gets GP predictions based on true ego predictions, eval purposes only
    """
    track_obj = sim_data.scenario_def.track
    sim_data.tar_gp_pred_post = [None]*len(sim_data.tar_gp_pred)
    for timeStep in range(len(sim_data.tar_states)):
        pred = sim_data.tar_gp_pred[timeStep]
        ego_pred = sim_data.ego_preds[timeStep] # TODO check if first entry contains current state
        if pred is not None and ego_pred is not None:
            N = len(pred.s)
            ego_pred_real = ego_pred.copy()
            if N + timeStep - 1 < len(sim_data.tar_states):
                for i in range(1,len(ego_pred_real.s)):
                    c_state = sim_data.ego_states[timeStep - 1 + i]
                    ego_pred_real.s[i] = c_state.p.s
                    ego_pred_real.x_tran[i] = c_state.p.x_tran
                    ego_pred_real.v_long[i] = c_state.v.v_long
                    ego_pred_real.e_psi[i] = c_state.p.e_psi
                pred = gp.get_true_prediction_par(sim_data.ego_states[timeStep-1], sim_data.tar_states[timeStep-1], ego_pred_real, track_obj, M=50)
                sim_data.tar_gp_pred_post[timeStep] = pred



def interp_data(state_list, dt, kind='quadratic'):
    x = []
    y = []
    psi = []
    t = []
    t_end = state_list[-1].t - state_list[0].t
    for st in state_list:
        x.append(st.x.x)
        y.append(st.x.y)
        psi_ = st.e.psi
        while len(psi) > 0 and abs(psi[-1] - psi_) > np.pi / 2:
            if psi[-1] - psi_ > np.pi / 2:
                psi_ = psi_ + np.pi
            else:
                psi_ = psi_ - np.pi
        psi.append(psi_)
        t.append(st.t- state_list[0].t)
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)
    psi = np.array(psi)
    interpStates = dict()
    x_interp = scipy.interpolate.interp1d(t, x, kind=kind)
    y_interp = scipy.interpolate.interp1d(t, y, kind=kind)
    psi_interp = scipy.interpolate.interp1d(t, psi, kind=kind)
    interpStates['x'] = [float(x_interp(dt * i)) for i in range(round(t_end / dt))]
    interpStates['y'] = [float(y_interp(dt * i)) for i in range(round(t_end / dt))]
    interpStates['psi'] = [float(psi_interp(dt * i)) for i in range(round(t_end / dt))]
    interpStates['t'] = [dt * i for i in range(round(t_end / dt))]
    return interpStates


def interp_state(state1, state2, t):
    state = state1.copy()
    dt0 = t - state1.t
    dt = state2.t - state1.t
    state.p.s = (state2.p.s - state1.p.s) / dt * dt0 + state1.p.s
    state.p.x_tran = (state2.p.x_tran - state1.p.x_tran) / dt * dt0 + state1.p.x_tran
    state.x.x = (state2.x.x - state1.x.x) / dt * dt0 + state1.x.x
    state.x.y = (state2.x.y - state1.x.y) / dt * dt0 + state1.x.y
    state.e.psi = (state2.e.psi - state1.e.psi) / dt * dt0 + state1.e.psi
    return state


def wrap_del_s(tar_s, ego_s, track: RadiusArclengthTrack):

    half_track = track.track_length/4
    full_track = track.track_length/2
    tmp_tar_s = tar_s 
    
    if abs(tar_s + full_track - ego_s) < abs(tar_s - ego_s):
        tmp_tar_s += full_track
    elif abs(tar_s - full_track - ego_s) < abs(tar_s - ego_s):
        tmp_tar_s -= full_track
    del_s = tmp_tar_s - ego_s

    if del_s < -15:
        print(1)
    # if abs(del_s) > half_track:
    #     if tar_s > half_track and ego_s < half_track:
    #         tmp = tar_s - full_track
    #         del_s = tmp - ego_s
    #     elif tar_s < half_track and ego_s > half_track:
    #         tmp = ego_s - full_track
    #         del_s = tar_s - tmp
    #     else:
    #         print("NA")
    #         return None
    return del_s


# if abs((tv_state.p.s + self.track_length) - ego_state.p.s) < abs(tv_state.p.s - ego_state.p.s):
#     tv_state.p.s += self.track_length
# elif abs((tv_state.p.s - self.track_length) - ego_state.p.s) < abs(tv_state.p.s - ego_state.p.s):
#     tv_state.p.s -= self.track_length

def wrap_s_np(s_,track_length):
    if len(s_.shape)<1:
        while(s_ <0):
            s_+=track_length
        while(s_ > track_length):
            s_-= track_length
    else:
        while(np.min(s_) < 0):        
            s_[s_ < 0] += track_length    
        while(np.max(s_) >= track_length):        
            s_[s_ >= track_length] -= track_length    
    return s_

def torch_wrap_del_s(tar_s: torch.tensor, ego_s: torch.tensor, track: RadiusArclengthTrack):                        

    half_track = track.track_length/4
    full_track = track.track_length/2
    if len(tar_s.shape)  < 1:
        tar_s = tar_s.unsqueeze(dim=1)
        ego_s = ego_s.unsqueeze(dim=1)
    tar_s[abs(tar_s+full_track-ego_s) <abs(tar_s-ego_s)]+=full_track
    tar_s[abs(tar_s-full_track-ego_s) <abs(tar_s-ego_s)]-=full_track
        
    del_s = tar_s - ego_s    
    
   
    return del_s

##############################
# Example scenario Definitions
##############################


'''
Scenario description:
- straight track
- ego to the left and behind target
- ego same speed as target
'''