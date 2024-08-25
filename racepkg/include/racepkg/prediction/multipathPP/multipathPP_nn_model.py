'''
# This code is based on the work presented in:
# Konev, Stepan. "MPA: MultiPath++ Based Architecture for Motion Prediction." arXiv, 2022.
# DOI: 10.48550/arXiv.2206.10041
# URL: https://arxiv.org/abs/2206.10041
'''



import torch
from torch import nn
from racepkg.prediction.multipathPP.modules import MCGBlock, HistoryEncoder, MLP, NormalMLP, Decoder, DecoderHandler, EM, MHA
import yaml
from yaml import Loader

def ff(x,k,s):
    return (x-k)/s+1
def rr(y,k,s):
    return (y-1)*s+k



def get_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader)
    return config


class MULTIPATHPPModel(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        if config is None:
            import os 
            current_dir = os.path.dirname(__file__)
            config_file_path = current_dir + '/config_multipathpp_model.yaml'
            config = get_config(config_file_path)
            config = config["model"]
        self._config = config
        self._agent_history_encoder = HistoryEncoder(config["agent_history_encoder"])
        self._agent_mcg_linear = NormalMLP(config["agent_mcg_linear"])
        self._interaction_mcg_linear = NormalMLP(config["interaction_mcg_linear"])
        self._interaction_history_encoder = HistoryEncoder(config["interaction_history_encoder"])
        self._polyline_encoder = NormalMLP(config["polyline_encoder"])
        self._history_mcg_encoder = MCGBlock(config["history_mcg_encoder"])
        self._interaction_mcg_encoder = MCGBlock(config["interaction_mcg_encoder"])
        self._agent_and_interaction_linear = NormalMLP(config["agent_and_interaction_linear"])
        self._roadgraph_mcg_encoder = MCGBlock(config["roadgraph_mcg_encoder"])
        self._decoder_handler = DecoderHandler(config["decoder_handler_config"])
        if config["multiple_predictions"]:
            self._decoder = Decoder(config["final_decoder"])
        if config["make_em"]:
            self._em = EM()
        if self._config["mha_decoder"]:
            self._mha_decoder = MHA(config["mha_decoder"])
    

    def input_preprocess_for_multipathpp(self,input):      
        '''
        # input = combined history  : dim x Horizon (e.g., 10 x 10)
        # tar_history = 4 x horizon 
        # ego_history = 3 x horizon                         
        # [delta_s,  tar_st.p.x_tran, tar_st.p.e_psi, tar_st.v.v_long,
        # ego_st.p.x_tran, ego_st.p.e_psi, ego_st.v.v_long   
        # tar_st.lookahead.curvature[0], tar_st.lookahead.curvature[1], tar_st.lookahead.curvature[2],
        '''  
        ego_history = input[:,7:,:].permute(0,2,1)
        tar_history = input[:,:4,:].permute(0,2,1)
        track_history = input[:,4:7,-1]
        return ego_history, tar_history, track_history

       
    def _compute_mcg_input_data(self, in_data):                                            
        I = torch.eye(in_data.shape[1], device=in_data.device).unsqueeze(0)
        timestamp_ohe = I.repeat(in_data.shape[0],1, 1)
        out_data= torch.cat([in_data, timestamp_ohe], dim=-1)
        return out_data

    def forward(self, data, num_steps = 0):

        ego_history, tar_history, track_history = self.input_preprocess_for_multipathpp(data)
        ego_history_diff = ego_history[:,1:,:]-ego_history[:,:-1,:]
        tar_history_diff = tar_history[:,1:,:]-tar_history[:,:-1,:]
        ego_history_msg = self._compute_mcg_input_data(ego_history)        
        tar_history_msg = self._compute_mcg_input_data(tar_history)
        target_scatter_numbers = torch.ones(data.shape[0], dtype=torch.long).cuda()
        target_scatter_idx = torch.arange(data.shape[0], dtype=torch.long).cuda()

        target_mcg_input_data_linear = self._agent_mcg_linear(ego_history_msg)
        assert torch.isfinite(target_mcg_input_data_linear).all()
        target_agents_embeddings = self._agent_history_encoder(
            target_scatter_numbers, target_scatter_idx, ego_history,
            ego_history_diff, target_mcg_input_data_linear)
        assert torch.isfinite(target_agents_embeddings).all()

        other_mcg_input_data_linear = self._interaction_mcg_linear(tar_history_msg)
        assert torch.isfinite(other_mcg_input_data_linear).all()

        interaction_agents_embeddings = self._interaction_history_encoder(
            target_scatter_numbers, target_scatter_idx,
            tar_history, tar_history_diff,
            other_mcg_input_data_linear)
        assert torch.isfinite(interaction_agents_embeddings).all()
        target_mcg_embedding = self._history_mcg_encoder(
            target_scatter_numbers, target_scatter_idx, target_agents_embeddings)
        assert torch.isfinite(target_mcg_embedding).all()
        interaction_mcg_embedding = self._interaction_mcg_encoder(
            target_scatter_numbers, target_scatter_idx,
            interaction_agents_embeddings, target_agents_embeddings)
        assert torch.isfinite(interaction_mcg_embedding).all()
        segment_embeddings = self._polyline_encoder(track_history)
        assert torch.isfinite(segment_embeddings).all()
        target_and_interaction_embedding = torch.cat(
            [target_mcg_embedding, interaction_mcg_embedding], axis=-1)
        assert torch.isfinite(target_and_interaction_embedding).all()
        target_and_interaction_embedding_linear = self._agent_and_interaction_linear(
            target_and_interaction_embedding)
        assert torch.isfinite(target_and_interaction_embedding_linear).all()
        roadgraph_mcg_embedding = self._roadgraph_mcg_encoder(
            target_scatter_numbers, target_scatter_idx,
            segment_embeddings, target_and_interaction_embedding_linear)
        assert torch.isfinite(roadgraph_mcg_embedding).all()
        final_embedding = torch.cat(
            [target_mcg_embedding, interaction_mcg_embedding, roadgraph_mcg_embedding], dim=-1)
        assert torch.isfinite(final_embedding).all()      
        probas, coordinates, covariance_matrices, epsi_vel, epsi_vel_cov, loss_coeff = self._decoder_handler(
            target_scatter_numbers, target_scatter_idx, final_embedding, data.shape[0])        
        assert torch.isfinite(probas).all()
        assert torch.isfinite(coordinates).all()
        assert torch.isfinite(covariance_matrices).all()

        return probas, coordinates, covariance_matrices, epsi_vel, epsi_vel_cov, loss_coeff
