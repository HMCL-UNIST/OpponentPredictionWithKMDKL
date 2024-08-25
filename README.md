## Multipath++ for Autonomous racing 
In adapting MultiPath++ for our problem, we modify its input and output interfaces to align with our racing domain context, ensuring a fair comparison between baselines. Specifically, the input vector is divided into three parts: EV state history, OV state history, and track information. The EV and OV data are processed through separate LSTM encoders, and the outputs are then passed through the Multi-Context Gating (MCG) encoder to generate the latent features. These features are then merged with track information and further processed by another MCG encoder before being fed into a decoder to generate multi-step OV state predictions.

![multipathpp_v1](https://github.com/user-attachments/assets/c4739271-968a-4739-b691-3055feb7b068)

## Acknowledgments
The implementation of the comparison method (DNN) utilizes concepts and code from the following papers:

Varadarajan, Balakrishnan, et al. "MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction." 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022.

Konev, Stepan. "MPA: MultiPath++ Based Architecture for Motion Prediction." arXiv, 2022. DOI: 10.48550/arXiv.2206.10041. Available online: [https://arxiv.org/abs/2206.10041](https://arxiv.org/abs/2206.10041).

