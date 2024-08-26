# Kernel-based Metrics Learning for Uncertain Opponent Vehicle Trajectory Prediction in Autonomous Racing (Youtube)
[![race_thumb](https://github.com/HMCL-UNIST/OpponentPredictionWithKMDKL/assets/32535170/82bc89cb-bae4-47bf-9f88-555e1a216b46)](https://www.youtube.com/watch?v=lUekW3UPFJE)

## Experiment setup
<div style="display: flex;">
    <img src="https://github.com/HMCL-UNIST/OpponentPredictionWithKMDKL/assets/32535170/51222632-0402-4be8-9091-209ab43489f8" alt="racecar" width="400">
</div>

## Algorithm overview
We design an autonomous racing solution that predicts opponent trajectory within an autonomous racing context. The proposed KM-DKL effectively encodes the predicted trajectory of the opponent vehicle with diverse driving policies along with its associated uncertainty.

![row_algorithm](https://github.com/HMCL-UNIST/OpponentPredictionWithKMDKL/assets/32535170/dd0a9324-6a15-4afd-b182-06b33fbb4012)

## Comparison to baselines
![trajv3](https://github.com/user-attachments/assets/fc84b8bd-0e19-4136-995c-471baf4b7dc8)


## Dependency
Tested with 
- ROS Noetic
- torch version 1.12.0+cu116  (with GPU)
- gpytorch version 1.8.0 (with GPU)


## Visualize the Track   
    roslaunch racepkg track_pub.launch    

## Run the Predictor
    roslaunch racepkg ego_predictor.launch    
  
To select the predictor type, run rqt. 

## Record vehicle states of Opponent and Ego vehicles during racing
To begin the recording process.

    roslaunch racepkg recorder.launch    
  
    
*To enable data logging, Turn on the rqt and enable recording.

## Training the model 
First, need to make a directory to save models. Detailed paths are defined in `/include/racepkg/common/utils/rile_utils.py`.
Then, refer to `/include/racepkg/prediction/covGP/train_main.py` for training details.

## Ego Controller and Target Controller
The ego controller runs from the desktop. 
    
    roslaunch racepkg ego_controller.launch      
    
    or 
    
    roslaunch racepkg target_controller.launch  
    
If anyone wants to explore it from jetson AGX Orin, they need to build an MPC problem for the platform `NVIDIA-Cortex-A57' by requesting a license from FORCESPRO. 
Additionally, an MPC interface in C for calling the built solver is required. We have built a customized ROS server in C++ for solving the problem in Orin and run the Python node to interface with ROS. 


## Paper 
Under review. 

## Branch Information

For those interested in the comparison method (DNN, multipath++), please refer to the `multipathpp` branch of this repository. 
This branch contains a variant of the MultiPath++ architecture used for our experiments.



## Acknowledgement

**I would like to express my sincere thanks to the following:**

-MPCC formulation and backbone source code from the paper:
  - Zhu, Edward L., et al. "A Gaussian Process Model for Opponent Prediction in Autonomous Racing." Proceedings of the 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE, 2023. Available at: [Source Code Repository](https://github.com/MPC-Berkeley/gp-opponent-prediction-models.git)

  
-Multipath++ backbone code for the comparison method:
  - Konev, Stepan. "MPA: MultiPath++ Based Architecture for Motion Prediction." arXiv, 2022. DOI: 10.48550/arXiv.2206.10041. Available online: [https://arxiv.org/abs/2206.10041](https://arxiv.org/abs/2206.10041).

  
