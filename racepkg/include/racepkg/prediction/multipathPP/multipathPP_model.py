import torch
import gpytorch
from racepkg.prediction.abstract_gp_controller import GPController
import array
from tqdm import tqdm
import numpy as np
from racepkg.h2h_configs import *
from racepkg.common.utils.file_utils import *
import torch.nn as nn
from racepkg.common.tracks.radius_arclength_track import RadiusArclengthTrack

from racepkg.prediction.multipathPP.multipathPP_nn_model import MULTIPATHPPModel
from torch.utils.data import DataLoader
from typing import Type, List
from racepkg.prediction.multipathPP.multipathPP_dataGen import SampleGeneartorMultiPathPP
import sys
from torch.utils.tensorboard import SummaryWriter
from racepkg.common.utils.scenario_utils import torch_wrap_del_s

class MULTIPATHPP(GPController):
    def __init__(self, args, sample_generator: SampleGeneartorMultiPathPP, enable_GPU=False):
        if args is None:
            self.args = {                    
            "batch_size": 512,
            "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            "input_dim": 10,
            "output_dim": 4,
            "n_time_step": 10,
            "latent_dim": 4,
            "gp_output_dim": 4,            
            "train_nn" : False                
            }
        else: 
            self.args = args
        self.train_nn = self.args["train_nn"]
        input_size = self.args["input_dim"]
        output_size = self.args["gp_output_dim"]
        self.output_size = output_size        
        super().__init__(sample_generator, Type[gpytorch.models.GP], gpytorch.likelihoods.Likelihood, input_size, output_size, enable_GPU)
            
        self.model = MULTIPATHPPModel().to(device='cuda')
        
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        
    def pull_samples(self, holdout=150):        
        return 
       
    def outputToReal(self, output):
        if self.normalize:
            return output

        if self.means_y is not None:
            return output * self.stds_y + self.means_y
        else:
            return output

    def nll_with_covariances(self, gt, predictions, confidences,  covariance_matrices):
        precision_matrices = torch.inverse(covariance_matrices)
        gt = torch.unsqueeze(gt, 1)        
        coordinates_delta = (gt - predictions).unsqueeze(-1)
        errors = coordinates_delta.permute(0, 1, 2, 4, 3) @ precision_matrices @ coordinates_delta
        errors = (-0.5 * errors.squeeze(-1) - 0.5 * torch.logdet(covariance_matrices).unsqueeze(-1))
        assert torch.isfinite(errors).all()
        with np.errstate(divide="ignore"):
            errors = nn.functional.log_softmax(confidences, dim=1) + \
                torch.sum(errors, dim=[2, 3])
        errors = -torch.logsumexp(errors, dim=-1, keepdim=True)
        return torch.mean(errors)


    def train(self,sampGen: SampleGeneartorMultiPathPP,valGEn : SampleGeneartorMultiPathPP,  args = None):
        self.writer = SummaryWriter()
        
        n_epoch = args["n_epoch"]
        self.writer = SummaryWriter()

        train_dataset, _, _  = sampGen.get_datasets()
        batch_size = self.args["batch_size"]
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)        
        valid_dataset, _, _  = valGEn.get_datasets()        
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


        if self.enable_GPU:
            self.model = self.model.cuda()            
          
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)                                                                                                
        max_epochs = n_epoch* len(train_dataloader)        
        no_progress_epoch = 0
        done = False
        epoch = 0        
        best_model = None
        sys.setrecursionlimit(100000)        
        self.model.double()
        while not done:        
            self.model.train()
            train_dataloader = tqdm(train_dataloader)
            valid_dataloader = tqdm(valid_dataloader)
            train_loss = 0
            valid_loss = 0            
            c_loss = 0            
            train_losses = []
            for step, (train_x, train_y) in enumerate(train_dataloader):                                                
                torch.cuda.empty_cache()   
                optimizer.zero_grad()                                                
                train_x_h  = train_x.double()          
                probas, coordinates, covariance_matrices, epsi_vel, epsi_vel_cov, loss_coeff = self.model(train_x_h.cuda())
                s_ey_future_gt = train_y.permute(0,2,1)[:,:,:2]
                epsi_val_future_gt = train_y.permute(0,2,1)[:,:,2:]
                s_ey_loss = self.nll_with_covariances(s_ey_future_gt, coordinates, probas, covariance_matrices) * loss_coeff                
                epsi_val_loss = self.nll_with_covariances(epsi_val_future_gt, epsi_vel, probas, epsi_vel_cov) * loss_coeff                
                loss = s_ey_loss*1.1 + epsi_val_loss
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_loss += loss.item()                    
            
            train_loss_tag = 'MultipathPP/Loss/total_train_loss' 
            self.writer.add_scalar(train_loss_tag, train_loss/ len(train_dataloader), epoch)
                    
            if epoch % 100 ==0:
                snapshot_name = 'multipathPP_'+ str(epoch)+ 'snapshot'                
                self.model.eval()
                self.save_model(snapshot_name)
                self.model.train()
                
            self.model.eval()
            optimizer.zero_grad()     
            for eval_step, (eval_x, eval_y) in enumerate(valid_dataloader):
                torch.cuda.empty_cache()
                with torch.no_grad():                                           
                    eval_x  = eval_x.double()                                                    
                    probas, coordinates, covariance_matrices, epsi_vel, epsi_vel_cov, loss_coeff = self.model(eval_x.cuda())
                    s_ey_future_gt = eval_y.permute(0,2,1)[:,:,:2]
                    epsi_val_future_gt = eval_y.permute(0,2,1)[:,:,2:]
                    eval_s_ey_loss = self.nll_with_covariances(s_ey_future_gt, coordinates, probas, covariance_matrices) * loss_coeff                
                    eval_epsi_val_loss = self.nll_with_covariances(epsi_val_future_gt, epsi_vel, probas, epsi_vel_cov) * loss_coeff                
                    eval_loss = eval_s_ey_loss*1.1 + eval_epsi_val_loss                    
                    valid_loss += eval_loss.item()
                    c_loss = valid_loss / (eval_step + 1)
                    valid_dataloader.set_postfix(log={'valid_loss': f'{(c_loss):.5f}'})                
                  
            no_progress_epoch += 1
            epoch +=1
            if epoch > max_epochs:
                done = True
            

        self.model = best_model        
        print("test done")
    

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

class MULTIPATHPPTrained(GPController):
    def __init__(self, name, enable_GPU, load_trace = False, model=None, args = None, sample_num = 25):        
        self.M = sample_num
        if args is not None:
            self.input_dim = args['input_dim']
            self.n_time_step = args['n_time_step']
        else:
            self.input_dim = 9
            self.n_time_step = 10
        
        self.model_name = name

        if model is not None:
            self.load_model_from_object(model)
        else:
            self.load_model(name)
        
        self.enable_GPU = enable_GPU

        self.load_normalizing_consant(name= name)
        if self.enable_GPU:
            self.model = self.model.cuda()                        
        else:
            self.model.cpu()            
            
        self.load_trace = load_trace
        self.trace_model = None
            

    def load_normalizing_consant(self, name ='normalizing'):        
        
        model = pickle_read(os.path.join(model_dir, name + '_normconstant.pkl'))        
        self.means_x = model['mean_sample'].cuda()        
        self.means_y = model['mean_output'].cuda()
        self.stds_x = model['std_sample'].cuda()        
        self.stds_y = model['std_output'].cuda()                
        print('Successfully loaded normalizing constants', name)


    def get_true_prediction_par(self, input,  ego_state: VehicleState, target_state: VehicleState,
                                ego_prediction: VehiclePrediction, track: RadiusArclengthTrack, M=10):
       
        self.model.eval()
        preds = self.sample_traj_gp_par(input, ego_state, target_state, ego_prediction, track, M)
        pred = self.mean_and_cov_from_list(preds, M, track= track) 
        pred.t = ego_state.t
        return pred



    def insert_to_end(self, roll_input, tar_state, tar_curv, ego_state, track):        
        roll_input[:,:,:-1] = roll_input[:,:,1:]
        input_tmp = torch.zeros(roll_input.shape[0],roll_input.shape[1]).to('cuda')        
        
        input_tmp[:,0] = tar_state[:,0]-ego_state[:,0]                      
        input_tmp[:,0] = torch_wrap_del_s(tar_state[:,0],ego_state[:,0], track)        
        input_tmp[:,1] = tar_state[:,1]
        input_tmp[:,2] = tar_state[:,2]
        input_tmp[:,3] = tar_state[:,3]
        input_tmp[:,4] = tar_curv[:,0]
        input_tmp[:,5] = tar_curv[:,1]
        input_tmp[:,6] = tar_curv[:,2]
        input_tmp[:,7] = ego_state[:,1]
        input_tmp[:,8] = ego_state[:,2] 
        input_tmp[:,9] = ego_state[:,3]                                           
        roll_input[:,:,-1] = input_tmp
        return roll_input.clone()

    
    def sample_traj_gp_par(self, encoder_input,  ego_state: VehicleState, target_state: VehicleState,
                           ego_prediction: VehiclePrediction, track: RadiusArclengthTrack, M):

        if self.load_trace:            
            probas, coordinates, covariance_matrices, epsi_vel, epsi_vel_cov, loss_coeff = self.trace_model(self.standardize(encoder_input).cuda())
        else:
            # pred_delta_dist = self.model(self.standardize(tmp_input))            
            encoder_inputprobas, coordinates, covariance_matrices, epsi_vel, epsi_vel_cov, loss_coeff = self.model(self.standardize(encoder_input.double().unsqueeze(0).cuda()))
        std_ = torch.stack([covariance_matrices[:,:,:,0,0],  covariance_matrices[:,:,:,1,1], epsi_vel_cov[:,:,:,0,0],  epsi_vel_cov[:,:,:,1,1]], dim=-1)
        std_ = torch.sqrt(std_)
        mean_ = torch.concat([coordinates, epsi_vel], dim=-1)
        real_mean, real_std = self.unstandardize_statistics(mean_, std_)                
        sampled_data = torch.distributions.Normal(real_mean.repeat(M,1,1), real_std.repeat(M,1,1)).sample()                    
        prediction_samples = []
      
        for i in range(M):
            tmp_prediction = VehiclePrediction()             
            prediction_samples.append(tmp_prediction)
            prediction_samples[i].s = (array.array('d', target_state.p.s+sampled_data[i,0,:].cpu()))            
            prediction_samples[i].x_tran = (array.array('d', sampled_data[i,1,:].cpu()))
            prediction_samples[i].e_psi = (array.array('d', sampled_data[i,2,:].cpu()))
            prediction_samples[i].v_long = (array.array('d', sampled_data[i,3,:].cpu()))
        
        return prediction_samples





  
    
 

        
