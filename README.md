# Kernel-based Metric Learning for Uncertainty-Aware Trajectory Prediction in Autonomous Racing


## System architecture

We design an autonomous racing solution that predicts opponent trajectory within an autonomous racing context. The proposed KM-DKL effectively encodes the predicted trajectory of the opponent vehicle with diverse driving policies along with its associated uncertainty.

<div style="display: flex;">
    <img src="https://github.com/HMCL-UNIST/OpponentPredictionWithKMDKL/assets/32535170/108f1568-d47c-41b4-8ec8-3f60a580574f" alt="mainalg" width="400">
    <img src="https://github.com/HMCL-UNIST/OpponentPredictionWithKMDKL/assets/32535170/51222632-0402-4be8-9091-209ab43489f8" alt="racecar" width="400">
</div>

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


## Acknowledgement
 **I would like to express my sincere thanks to the following:**

- MPCC formulation from the below paper.     
```
Zhu, Edward L., et al. "A Gaussian Process Model for Opponent Prediction in Autonomous Racing." 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023.
```
