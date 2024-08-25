from racepkg.common.pytypes import VehicleState, VehiclePrediction
import numpy as np
import gc
import torch

from racepkg.common.tracks.radius_arclength_track import RadiusArclengthTrack
from racepkg.controllers.utils.controllerTypes import *
from racepkg.prediction.trajectory_predictor import BasePredictor
from racepkg.prediction.multipathPP.multipathPP_model import MULTIPATHPPTrained
from racepkg.prediction.multipathPP.multipathPP_dataGen import states_to_encoder_input_torch

class MultipathPPPredictor(BasePredictor):
    def __init__(self,  N: int, track : RadiusArclengthTrack, use_GPU: bool, M: int, cov_factor: float = 1, input_predict_model = "multipathpp", args = None):
        super(MultipathPPPredictor, self).__init__(N, track)
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
                "add_noise_data" : False,
                'model_name' : None
                }
        else:
            self.args = args
        
        self.multipathpp_predict = MULTIPATHPPTrained(input_predict_model, use_GPU, load_trace = False, args = self.args, sample_num = M)        
        self.M = M  
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
        if is_encoder_input_ready: ## encoder_is_ready = True            
            pred = self.multipathpp_predict.get_true_prediction_par(self.encoder_input,  ego_state, target_state, ego_prediction, self.track, self.M)            
            pred.track_cov_to_local(self.track, self.N, self.cov_factor)  
        else:            
            pred = None         
        return pred


    def get_eval_prediction(self,input_buffer,  ego_state: VehicleState, target_state: VehicleState,
                       ego_prediction: VehiclePrediction, tar_prediction=None):
        pred = self.multipathpp_predict.get_true_prediction_par(input_buffer,  ego_state, target_state, ego_prediction, self.track, self.M)            
        pred.track_cov_to_local(self.track, self.N, self.cov_factor)  
        
        return pred

    def get_constant_vel_prediction_par(self, target_state: VehicleState):
        target_state.update_global_velocity_from_body()
        t = target_state.t
        x = target_state.x.x
        y = target_state.x.y
        v_x = target_state.v_x
        v_y = target_state.v_y
        psidot = target_state.w.w_psi
        psi = target_state.e.psi

        t_list = np.zeros((self.N))
        x_list = np.zeros((self.N))
        y_list = np.zeros((self.N))
        psi_list = np.zeros((self.N))

        delta_psi = 0
        for i in range(self.N):
            x_list[i] = x
            y_list[i] = y
            t_list[i] = t
            psi_list[i] = psi
            t += self.dt
            x += self.dt * (v_x * np.cos(delta_psi) - v_y * np.sin(delta_psi))
            y += self.dt * (v_y * np.cos(delta_psi) + v_x * np.sin(delta_psi))


            delta_psi += self.dt * psidot
            psi += self.dt * psidot

        pred = VehiclePrediction(t=target_state.t, x=x_list, y=y_list, psi=psi_list)
        pred.xy_cov = np.repeat(np.diag([self.cov, self.cov])[np.newaxis, :, :], self.N, axis=0)
        return pred