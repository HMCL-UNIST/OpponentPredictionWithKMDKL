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

import numpy as np
import torch
import gpytorch
from datetime import datetime
from racepkg.common.utils.file_utils import *
from racepkg.prediction.gpytorch_models import IndependentMultitaskGPModelApproximate
from racepkg.prediction.covGP.covGPNN_model import COVGPNN
from racepkg.prediction.covGP.covGPNN_dataGen import SampleGeneartorCOVGP

# Training
def covGPNN_train(dirs = None, val_dirs = None, real_data = False, args = None):
   
    if args is None:
        print("ARGS should be given!!")
        return 
    sampGen = SampleGeneartorCOVGP(dirs, args = args, randomize=True, real_data = real_data)
    
    valid_args = args.copy()
    valGEn = SampleGeneartorCOVGP(val_dirs, load_normalization_constant = True, args = valid_args, randomize=False, real_data = True, tsne = False)
 
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=args["gp_output_dim"]) 
    covgp_predictor = COVGPNN(args, sampGen, IndependentMultitaskGPModelApproximate, likelihood, enable_GPU=True)
    

    covgp_predictor.train(sampGen, valGEn, args = args)
    covgp_predictor.set_evaluation_mode()
    trained_model = covgp_predictor.model, covgp_predictor.likelihood

    create_dir(path=model_dir)
    if args['direct_gp'] is True:
        gp_name = 'naiveGP'
    else:   
        if(args['include_kml_loss']):
            gp_name = 'simtsGP'
        else:
            gp_name = 'nosimtsGP'
    covgp_predictor.save_model(gp_name)
