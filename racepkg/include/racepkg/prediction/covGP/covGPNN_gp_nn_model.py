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
import torch.nn as nn
from einops import reduce, rearrange
import gpytorch
import torch.nn.functional as F


class CovSparseGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points_num, input_dim, num_tasks):
        # Let's use a different set of inducing points for each task
        inducing_points = torch.rand(num_tasks, inducing_points_num, input_dim)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)


        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=torch.Size([num_tasks])), # nu = 1.5
            batch_shape=torch.Size([num_tasks])
        )
          


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size ==0:
          return x
        return x[:, :, :-self.chomp_size]
    


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()
        
        # Computes left padding so that the applied convolutions are causal
        self.padding = (kernel_size - 1) * dilation
        padding = self.padding
        # First causal convolution
        self.conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
        # self.conv1 = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        self.chomp1 = Chomp1d(padding)
        # self.dropout1 = nn.Dropout(0.1)

        # Second causal convolution
        self.conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
        # self.conv2 = torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        
        # Residual connection
        self.upordownsample = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, 1
        )) if in_channels != out_channels else None

        # Final activation function
        self.relu = None
        
    def forward(self, x):
       
        out_causal=self.conv1(x)
        out_causal=self.chomp1(out_causal)                
        out_causal=F.gelu(out_causal)
        out_causal=self.conv2(out_causal)
        out_causal=self.chomp2(out_causal)        
        out_causal=F.gelu(out_causal)
        res = x if self.upordownsample is None else self.upordownsample(x)
        
        
        if self.relu is None:
            x = out_causal + res
        else:
            x= self.relu(out_causal + res)
        
        return x



class CausalCNNEncoder(torch.nn.Module): 
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, 
                 in_channels,
                 reduced_size,
                 component_dims, 
                 kernel_list=[1,2, 4],
                 ):
        super(CausalCNNEncoder, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = 'cpu'
        self.component_dims = component_dims
        self.in_channels = in_channels
        self.reduced_size = reduced_size
        self.n_time_step = 10
        self.input_fc = CausalConvolutionBlock(in_channels, reduced_size, 1, 1)
        
        self.kernel_list = kernel_list
        self.multi_cnn = nn.ModuleList(
            [nn.Conv1d(reduced_size, component_dims, k, padding=k-1) for k in kernel_list]
        )
        
        
    def print_para(self):
        
        return list(self.multi_cnn.parameters())[0].clone()    
        
    def forward(self, x_h):        
        x_h = self.input_fc(x_h)        
        trend_h = []           

        for idx, mod in enumerate(self.multi_cnn):
            
            out_h = mod(x_h) # b d t
            if self.kernel_list[idx] != 1:
                out_h = out_h[..., :-(self.kernel_list[idx] - 1)]
            trend_h.append(out_h.transpose(1,2))  # b 1 t d                
    
        trend_h = reduce(
            rearrange(trend_h, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )
        
        latent_x = trend_h[:,-1,:]           
        return latent_x


        

class COVGPNNModel(gpytorch.Module):        
    def __init__(
        self, args):
       
        super(COVGPNNModel, self).__init__()        
        self.args = args                
        self.nn_input_dim = args['input_dim']        
        self.n_time_step = args['n_time_step']             
        self.gp_output_dim =  args['gp_output_dim']        
        self.seq_len = args['n_time_step']
        inducing_points = args['inducing_points']
        self.directGP = args['direct_gp']
        self.include_kml_loss = args['include_kml_loss']
                        
        kernel_list=[3,5,7,9]        
        self.encdecnn = CausalCNNEncoder(in_channels = self.nn_input_dim, 
                                reduced_size=50, 
                                component_dims = args['latent_dim'] , 
                                kernel_list= kernel_list)
        
        self.latent_size = args['latent_dim']  

        if self.directGP:
            self.gp_input_dim =  self.nn_input_dim
        else:             
            self.gp_input_dim =  self.latent_size

        self.gp_layer = CovSparseGP(inducing_points_num=inducing_points,
                                                        input_dim=self.gp_input_dim,
                                                        num_tasks=self.gp_output_dim)  # Independent        
        
 
        self.in_covs = torch.nn.ModuleList(
            [gpytorch.kernels.MaternKernel(nu=1.5) for k in range(4)]
        )

        self.out_covs = torch.nn.ModuleList(
            [gpytorch.kernels.MaternKernel(nu=1.5,lengthscale_constraint = gpytorch.constraints.Interval(lower_bound=0.01,upper_bound=1.0))for k in range(4)]
        )
      

    def outputToReal(self, batch_size, pred_dist):
        with torch.no_grad():            
            standardized_mean = pred_dist.mean.view(batch_size,-1,pred_dist.mean.shape[-1])
            standardized_stddev = pred_dist.stddev.view(batch_size,-1,pred_dist.mean.shape[-1])
            return standardized_mean, standardized_stddev
            
            
    def get_hidden(self,input_data):
        if input_data.shape[-1] > self.n_time_step:
            input_data = input_data[:,:,:int(input_data.shape[-1]/2)]
        
        return self.encdecnn(input_data) 

    def forward(self, x_h):  
        if self.directGP: 
            gp_input = x_h.float()
        else:             
            x_h = x_h.double()
            latent_x = self.encdecnn(x_h)   
            gp_input = latent_x
        pred = self.gp_layer(gp_input)
        
        return pred
                



    
class COVGPNNModelWrapper(torch.nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp
    
    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean, output_dist.stddev
    

