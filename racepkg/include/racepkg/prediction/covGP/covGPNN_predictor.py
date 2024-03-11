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
import gc
import torch
from racepkg.common.pytypes import VehicleState, VehiclePrediction
from racepkg.common.tracks.radius_arclength_track import RadiusArclengthTrack
from racepkg.controllers.utils.controllerTypes import *
from racepkg.prediction.trajectory_predictor import BasePredictor
from racepkg.prediction.covGP.covGPNN_model import COVGPNNTrained
from racepkg.prediction.covGP.covGPNN_dataGen import states_to_encoder_input_torch


class CovGPPredictor(BasePredictor):
    def __init__(self,  N: int, track : RadiusArclengthTrack, use_GPU: bool, M: int, cov_factor: float = 1, input_predict_model = "covGP", args = None):
        super(CovGPPredictor, self).__init__(N, track)
        gc.collect()
        torch.cuda.empty_cache()        
        if args is None:
            self.args = {                    
                "batch_size": 150,
                "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                "input_dim": 10,
                "n_time_step": 10, ## how much we see the past history 
                "latent_dim": 9,
                "gp_output_dim": 4,            
                "inducing_points" : 200,
                'model_name' : None
                }
        else:
            self.args = args
        
        self.covgpnn_predict = COVGPNNTrained(input_predict_model, use_GPU, load_trace = True, args = self.args, sample_num = M)
        print("input predict_gp loaded")
    
        self.M = M  # number of samples
        self.cov_factor = cov_factor
        self.ego_state_buffer = []
        self.tar_state_buffer = []
        self.time_length = self.args["n_time_step"]
        self.ego_state_buffer = []
        self.tar_state_buffer = []
        self.encoder_input = torch.zeros(self.args["input_dim"], self.time_length)
        self.buffer_update_count  = 0
        

    def append_vehicleState(self,ego_state: VehicleState,tar_state: VehicleState):
            tmp = self.encoder_input.clone()            
            self.encoder_input[:,0:-1] = tmp[:,1:]            
            self.encoder_input[:,-1] = states_to_encoder_input_torch(tar_state,ego_state, self.track)                        
            self.buffer_update_count +=1
            if self.buffer_update_count > self.time_length:
                self.buffer_update_count = self.time_length+1
                return True
            else:
                return False

    def get_prediction(self, ego_state: VehicleState, target_state: VehicleState,
                       ego_prediction: VehiclePrediction, tar_prediction=None):
        is_encoder_input_ready = self.append_vehicleState(ego_state,target_state)  
        if is_encoder_input_ready: 
            pred = self.covgpnn_predict.sample_traj_gp_par(self.encoder_input,  ego_state, target_state, ego_prediction, self.track, self.M)            
            pred.track_cov_to_local(self.track, self.N, self.cov_factor)  
        else:            
            pred = None 
           
        return pred

