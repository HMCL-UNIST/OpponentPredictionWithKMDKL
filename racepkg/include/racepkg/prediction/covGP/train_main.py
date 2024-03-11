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

import os
import torch
from racepkg.common.utils.file_utils import *
from racepkg.prediction.covGP.covGPNN_Train import covGPNN_train

args_ = {                    
    "batch_size": 1024,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "input_dim": 10,
    "n_time_step": 10,
    "latent_dim": 8,
    "gp_output_dim": 4,
    "inducing_points" : 100,
    "train_nn" : False,
    "include_kml_loss" : True,
    "direct_gp" : False,
    "n_epoch" : 10000,
    'model_name' : None,
    'eval' : False,
    'load_eval_data' : False
    }

def main_train(train_policy_names = None, valid_policy_names = None):
    train_dirs = []
    for i in range(len(train_policy_names)):
        test_folder = os.path.join(real_dir, train_policy_names[i])
        train_dirs.append(test_folder)

    val_dirs = []

    
    for i in range(len(valid_policy_names)):
        test_folder = os.path.join(real_dir, valid_policy_names[i])
        val_dirs.append(test_folder)
        
    # print("1~~~~~~~~~~~~~Naive GP Train Init~~~~~~~~~~~")    
    # args_["direct_gp"] = True
    # args_["include_kml_loss"] = False
    # args_['model_name'] = 'naiveGP'
    # covGPNN_train(train_dirs, val_dirs, real_data = True, args= args_)    
    # print("~~~~~~~~~~~Naive GP Train Done~~~~~~~~~~")

    # print("2~~~~~~Training Deep Kernel Learning Init ~~")    
    # args_["direct_gp"] = False
    # args_["include_kml_loss"] = False
    # args_['model_name'] = 'nosimtsGP'
    # covGPNN_train(train_dirs, val_dirs, real_data = True, args= args_)    
    # print("~~~~~~~~~~~Training Deep Kernel Learning Init END~~~~~~~~~")


    print("3~~~~~~~Training (Kernel-metric Learning) KML-Deep Kernel Learning Init ~~~~~~~")
    args_["direct_gp"] = False
    args_["include_kml_loss"] = True    
    args_['model_name'] = 'simtsGP'
    covGPNN_train(train_dirs,val_dirs, real_data = True, args= args_)
    print(" Done")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def main():  
    ####################################################        
    train_policy_names = ['dl_1_blocking_train', 'dl_1_real_center_train']
    valid_policy_names = ['dl_1_blocking_eval', 'dl_1_real_center_eval'] 
    main_train(train_policy_names, valid_policy_names)   
    ####################################################
    

if __name__ == "__main__":
    main()





