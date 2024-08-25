#!/usr/bin/env python3

from racepkg.common.utils.file_utils import *
from racepkg.prediction.multipathPP.multipathPP_model import MULTIPATHPP
from racepkg.prediction.multipathPP.multipathPP_dataGen import SampleGeneartorMultiPathPP


# Training
def multipathpp_train(dirs = None, val_dirs = None, real_data = False, args = None):
   
    if args is None:
        print("ARGS should be given!!")
        return 
    sampGen = SampleGeneartorMultiPathPP(dirs, args = args, randomize=True, real_data = real_data)
    valid_args = args.copy()
    valGEn = SampleGeneartorMultiPathPP(val_dirs, load_normalization_constant = True, args = valid_args, randomize=False, real_data = True, tsne = False) 
    if not dir_exists(dirs[0]):
        raise RuntimeError(
            f"Directory: {dirs[0]} does not exist, need to train using `gen_training_data` first")

    multipathpp_predictor = MULTIPATHPP(args, sampGen, enable_GPU=True)
    multipathpp_predictor.train(sampGen, valGEn, args = args)
    multipathpp_predictor.set_evaluation_mode()    
    create_dir(path=model_dir)    
    model_name = 'multipathpp'
    multipathpp_predictor.save_model(model_name)
