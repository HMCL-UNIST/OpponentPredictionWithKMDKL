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
import torch
import gpytorch
import array
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import sys
from typing import Type, List
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter


from racepkg.h2h_configs import *
from racepkg.common.utils.file_utils import *
from racepkg.common.utils.scenario_utils import torch_wrap_del_s
from racepkg.common.tracks.radius_arclength_track import RadiusArclengthTrack
from racepkg.prediction.abstract_gp_controller import GPController
from racepkg.prediction.covGP.covGPNN_gp_nn_model import COVGPNNModel, COVGPNNModelWrapper
from racepkg.prediction.covGP.covGPNN_dataGen import SampleGeneartorCOVGP
from racepkg.prediction.torch_utils import get_curvature_from_keypts_torch, torch_wrap_s

class COVGPNN(GPController):
    def __init__(self, args, sample_generator: SampleGeneartorCOVGP, model_class: Type[gpytorch.models.GP],
                 likelihood: gpytorch.likelihoods.Likelihood,
                 enable_GPU=False):
        if args is None:
            self.args = {                    
            "batch_size": 512,
            "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            "input_dim": 10,
            "n_time_step": 10,
            "latent_dim": 4,
            "gp_output_dim": 4,
            "inducing_points" : 300,
            "train_nn" : False                
            }
        else: 
            self.args = args
        self.train_nn = self.args["train_nn"]
        input_size = self.args["input_dim"]
        output_size = self.args["gp_output_dim"]
        self.output_size = output_size        
        super().__init__(sample_generator, model_class, likelihood, input_size, output_size, enable_GPU)
        
        self.model = COVGPNNModel(self.args).to(device='cuda')
        self.independent = True        
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

    def train(self,sampGen: SampleGeneartorCOVGP,valGEn : SampleGeneartorCOVGP,  args = None):
        self.writer = SummaryWriter()
        directGP = args['direct_gp']
        include_kml_loss = args['include_kml_loss']
        gp_name = 'simtsGP'
        if directGP:
            gp_name = 'naiveGP'
        else:   
            if include_kml_loss:
                gp_name = 'simtsGP'                
            else:
                gp_name = 'nosimtsGP'                

        n_epoch = args["n_epoch"]
        self.writer = SummaryWriter()
        train_dataset, _, _  = sampGen.get_datasets()
        batch_size = self.args["batch_size"]
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataset, _, _  = valGEn.get_datasets()        
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
          
        self.model.train()        
        self.likelihood.train()
                                                                                       
        optimizer_gp = torch.optim.Adam([{'params': self.model.gp_layer.hyperparameters()},  
                                    {'params': self.model.gp_layer.variational_parameters()},                                      
                                    {'params': self.likelihood.parameters()},
                                        ], lr=0.005)

        optimizer_all = torch.optim.Adam([{'params': self.model.encdecnn.parameters(), 'lr': 0.05, 'weight_decay':1e-9},                                          
                                        {'params': self.model.gp_layer.hyperparameters(), 'lr': 0.005},                                                                                
                                        {'params': self.model.in_covs.parameters(), 'lr': 0.01},
                                        {'params': self.model.out_covs.parameters(), 'lr': 0.01}, 
                                        {'params': self.model.gp_layer.variational_parameters()},
                                        {'params': self.likelihood.parameters()},
                                        ], lr=0.005)
        
        scheduler = lr_scheduler.StepLR(optimizer_gp, step_size=3000, gamma=0.9)
        
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model.gp_layer, num_data=sampGen.getNumSamples()*len(sampGen.output_data[0]))
        mseloss = nn.MSELoss()
        max_epochs = n_epoch* len(train_dataloader)
        last_loss = np.inf
        no_progress_epoch = 0
        done = False
        epoch = 0
        
        best_model = None
        best_likeli = None

        sys.setrecursionlimit(100000)
        self.model.double()
        while not done:        
            train_dataloader = tqdm(train_dataloader)
            valid_dataloader = tqdm(valid_dataloader)
            train_loss = 0
            valid_loss = 0
            c_loss = 0            
            latent_dist_loss_sum = 0
            std_loss_sum = 0
            
            
            cov_loss = 0 
            variational_loss_sum = 0 
            for step, (train_x, train_y) in enumerate(train_dataloader):                                
                torch.cuda.empty_cache()   
                optimizer_all.zero_grad()
                optimizer_gp.zero_grad()                 
                       
                if int(len(train_x.shape)) > 2:
                    train_x_h  = train_x[:,:,:int(train_x.shape[-1]/2)].double()
                else:    
                    train_x_h  = train_x.double()          
             
                output = self.model(train_x_h.cuda())                                

                if include_kml_loss:
                    ## KML Loss 
                    latent_x = self.model.get_hidden(train_x_h.cuda())                    
                    cov_loss = 0    
                    std_loss = 0      
                    
                    for i in range(self.output_size):
                        latent_dist = self.model.in_covs[i](latent_x, latent_x)                     
                        out_dist = self.model.out_covs[i](train_y[:,i].cuda(), train_y[:,i].cuda())
                        out_dist = out_dist.evaluate()
                        latent_dist = latent_dist.evaluate()                                                
                        
                        std_loss  += torch.log((self.model.out_covs[i].lengthscale)/(self.model.in_covs[i].lengthscale + 1e-12))*5e-2                                       
                        
                        cov_loss += mseloss(out_dist, latent_dist)                     

                    latent_std = torch.std(latent_x)                                                                       
                    sig_slope = 10
                    latent_std_loss =  0.1*torch.nn.functional.relu(sig_slope*(latent_std-3.0))
                    ## KML Loss END

                variational_loss = -mll(output, train_y)                
                
                if include_kml_loss:                     
                    loss = cov_loss + variational_loss+  std_loss +latent_std_loss 
                else:
                    loss = variational_loss

                loss.backward()

                if directGP:
                    optimizer_gp.step()
                else:                          
                    optimizer_all.step()
                     
                train_loss += loss.item()    
                variational_loss_sum += variational_loss.item()
                
                if include_kml_loss: 
                    latent_dist_loss_sum +=cov_loss.item()
                    std_loss_sum += std_loss.item()                    
                    self.writer.add_scalar(gp_name+'/stat/latent_max', torch.max(latent_x), epoch*len(train_dataloader) + step)
                    self.writer.add_scalar(gp_name+'/stat/latent_min', torch.min(latent_x), epoch*len(train_dataloader) + step)
                    self.writer.add_scalar(gp_name+'/stat/latent_std', torch.std(latent_x), epoch*len(train_dataloader) + step)

            
            varloss_tag = gp_name+'/Loss/variational_loss'
            self.writer.add_scalar(varloss_tag, variational_loss_sum/ len(train_dataloader), epoch)            
            train_loss_tag = gp_name+'/Loss/total_train_loss' 
            self.writer.add_scalar(train_loss_tag, train_loss/ len(train_dataloader), epoch)
            
            scheduler.step()            
            if epoch % 50 ==0:
                snapshot_name = gp_name + str(epoch)+ 'snapshot'
                self.set_evaluation_mode()
                self.save_model(snapshot_name)
                self.model.train()
                self.likelihood.train()

            
            for step, (test_x, test_y) in enumerate(valid_dataloader):
                torch.cuda.empty_cache()   
                optimizer_gp.zero_grad()                
                optimizer_all.zero_grad()
                
                if int(len(test_x.shape)) > 2:
                    test_x  = test_x[:,:,:int(test_x.shape[-1]/2)].double()
                else:    
                    test_x = test_x.double()         
                
                output = self.model(test_x)
                loss = -mll(output, test_y)
                valid_loss += loss.item()
                c_loss = valid_loss / (step + 1)
                valid_dataloader.set_postfix(log={'valid_loss': f'{(c_loss):.5f}'})                
            
            valid_loss_tag = gp_name+'/Loss/valid_loss'
            self.writer.add_scalar(valid_loss_tag, valid_loss, epoch)
            if c_loss > last_loss:
                if no_progress_epoch >= 30:
                    done = True                    
            else:
                best_model = copy.copy(self.model)
                best_likeli = copy.copy(self.likelihood)
                last_loss = c_loss
                no_progress_epoch = 0            
            no_progress_epoch += 1
            epoch +=1
            if epoch > max_epochs:
                done = True
            

        self.model = best_model
        self.likelihood = best_likeli
        print("test done")
    

class COVGPNNTrained(GPController):
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
            self.likelihood = self.likelihood.cuda()
        else:
            self.model.cpu()
            self.likelihood.cpu()
        self.load_trace = load_trace
        self.trace_model = None
        if self.load_trace:
            self.gen_trace_model()
            

    def gen_trace_model(self):
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
            self.model.eval()
            if self.model_name == 'naiveGP':
                test_x = torch.randn(self.M,self.input_dim).cuda()
            else:
                test_x = torch.randn(self.M,self.input_dim,self.n_time_step).cuda()
            pred = self.model(test_x)  # Do precomputation
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
            self.trace_model = torch.jit.trace(COVGPNNModelWrapper(self.model), test_x)
            
            

    def load_normalizing_consant(self, name ='normalizing'):        
        
        model = pickle_read(os.path.join(model_dir, name + '_normconstant.pkl'))        
        self.means_x = model['mean_sample'].cuda()
        if len(self.means_x.shape) > 1:
            self.means_x = self.means_x[:,:int(self.means_x.shape[1]/2)]
        self.means_y = model['mean_output'].cuda()
        self.stds_x = model['std_sample'].cuda()
        if len(self.stds_x.shape) > 1:
            self.stds_x = self.stds_x[:,:int(self.stds_x.shape[1]/2)]
        self.stds_y = model['std_output'].cuda()
        print('Successfully loaded normalizing constants', name)



    def get_true_prediction_par(self, input,  ego_state: VehicleState, target_state: VehicleState,
                                ego_prediction: VehiclePrediction, track: RadiusArclengthTrack, M=10):
       
        # Set GP model to eval-mode
        self.set_evaluation_mode()
        # draw M samples
        pred = self.sample_traj_gp_par(input, ego_state, target_state, ego_prediction, track, M)
        pred.t = ego_state.t
        return pred


    def insert_to_end(self, roll_input, tar_state, tar_curv, ego_state, track):        
        roll_input[:,:,:-1] = roll_input[:,:,1:]
        input_tmp = torch.zeros(roll_input.shape[0],roll_input.shape[1]).to('cuda')                
        input_tmp[:,0] = tar_state[:,0]-ego_state[:,0]
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

        '''
        encoder_input = batch x feature_dim x time_horizon
        '''     

        horizon = len(ego_prediction.x) 
        pred_tar_state = torch.zeros(M,4,horizon)
        pred_tar_state[:,0,0] = target_state.p.s # s
        pred_tar_state[:,1,0] = target_state.p.x_tran # ey 
        pred_tar_state[:,2,0] = target_state.p.e_psi # epsi
        pred_tar_state[:,3,0] = target_state.v.v_long # vx
 
        roll_input = encoder_input.repeat(M,1,1).to('cuda') 
        roll_tar_state = torch.tensor([target_state.p.s, target_state.p.x_tran, target_state.p.e_psi, target_state.v.v_long]).to('cuda')        
        roll_tar_state = roll_tar_state.repeat(M,1)        
        roll_tar_curv = torch.tensor([target_state.lookahead.curvature[0], target_state.lookahead.curvature[1], target_state.lookahead.curvature[2]]).to('cuda')        
        roll_tar_curv = roll_tar_curv.repeat(M,1)
        roll_ego_state = torch.tensor([ego_state.p.s, ego_state.p.x_tran, ego_state.p.e_psi, ego_state.v.v_long]).to('cuda')
        roll_ego_state = roll_ego_state.repeat(M,1)
 
        # start_time = time.time()
        for i in range(horizon-1):         
            # gp_start_time = time.time()  
            roll_input = self.insert_to_end(roll_input, roll_tar_state, roll_tar_curv, roll_ego_state)                      
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                if self.load_trace:
                    mean, stddev = self.trace_model(self.standardize(roll_input))
                else:
                    pred_delta_dist = self.model(self.standardize(roll_input))
                    mean = pred_delta_dist.mean
                    stddev = pred_delta_dist.stddev
                # pred_delta_dist = self.model(roll_input)            
                # print(stddev.cpu().numpy())
                
                tmp_delta = torch.distributions.Normal(mean, stddev).sample()            
                    
                pred_delta = self.outputToReal(tmp_delta)
            # pred_delta = torch.zeros(roll_tar_state.shape).cuda()
            roll_tar_state[:,0] += pred_delta[:,0] 
            roll_tar_state[:,1] += pred_delta[:,1] 
            roll_tar_state[:,2] += pred_delta[:,2]
            roll_tar_state[:,3] += pred_delta[:,3]  
            ###################################  0.04 ###################################
            roll_tar_curv[:,0] = get_curvature_from_keypts_torch(pred_delta[:,0],track)
            roll_tar_curv[:,1] = get_curvature_from_keypts_torch(pred_delta[:,0]+target_state.lookahead.dl*1,track)                        
            roll_tar_curv[:,2] = get_curvature_from_keypts_torch(pred_delta[:,0]+target_state.lookahead.dl*2,track)                        
            ################################### ###################################
            roll_ego_state[:,0] = ego_prediction.s[i+1]
            roll_ego_state[:,1] = ego_prediction.x_tran[i+1]
            roll_ego_state[:,2] =  ego_prediction.e_psi[i+1]
            roll_ego_state[:,3] =  ego_prediction.v_long[i+1]


            pred_tar_state[:,0,i+1] = roll_tar_state[:,0].clone() # s
            pred_tar_state[:,1,i+1] = roll_tar_state[:,1].clone() # ey 
            pred_tar_state[:,2,i+1] = roll_tar_state[:,2].clone() # epsi
            pred_tar_state[:,3,i+1] = roll_tar_state[:,3].clone() # vx

            # for j in range(M):                          # tar 0 1 2 3 4 5       #ego 6 7 8 9 10 11
            #     prediction_samples[j].s.append(roll_tar_state[j,0].cpu().numpy())
            #     prediction_samples[j].x_tran.append(roll_tar_state[j,1].cpu().numpy())                    
            #     prediction_samples[j].e_psi.append(roll_tar_state[j,2].cpu().numpy())
            #     prediction_samples[j].v_long.append(roll_tar_state[j,3].cpu().numpy())
        
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time taken for GP(over horizons) call: {elapsed_time} seconds")


        # prediction_samples = []
        # for j in range(M):
        #     tmp_prediction = VehiclePrediction() 
        #     tmp_prediction.s = array.array('d', pred_tar_state[j,0,:])
        #     tmp_prediction.x_tran = array.array('d', pred_tar_state[j,1,:])
        #     tmp_prediction.e_psi = array.array('d', pred_tar_state[j,2,:])
        #     tmp_prediction.v_long = array.array('d', pred_tar_state[j,3,:])            
        #     prediction_samples.append(tmp_prediction)

        mean_tar_pred = torch.mean(pred_tar_state,dim=0)        
        mean_pred = VehiclePrediction()
        mean_pred.s = array.array('d', mean_tar_pred[0,:])
        mean_pred.x_tran = array.array('d', mean_tar_pred[1,:])
        mean_pred.e_psi = array.array('d', mean_tar_pred[2,:])
        mean_pred.v_long = array.array('d', mean_tar_pred[3,:])

        std_tar_pred = torch.std(pred_tar_state,dim=0)
        std_tar_pred[3,:] = std_tar_pred[1,:]
        std_tar_pred[2,:] = 0.0
        std_tar_pred[1,:] = 0.0
        mean_pred.sey_cov = array.array('d', std_tar_pred.T.flatten())
        return mean_pred 

     