import torch
from racepkg.common.utils.scenario_utils import *
from racepkg.h2h_configs import *    
from racepkg.common.utils.scenario_utils import wrap_del_s
from racepkg.common.utils.file_utils import *


def states_to_encoder_input_torch(tar_st,ego_st, track:RadiusArclengthTrack):
    tar_s = tar_st.p.s
    tar_s = wrap_s_np(tar_s,track.track_length/2)
    ego_s = ego_st.p.s
    ego_s = wrap_s_np(ego_s,track.track_length/2)    
    delta_s = wrap_del_s(tar_s, ego_s, track)    
    if len(delta_s.shape) == 1 :
        delta_s = delta_s[0]
   
    input_data=torch.tensor([ delta_s,                        
                        tar_st.p.x_tran,
                        tar_st.p.e_psi,
                        tar_st.v.v_long,
                        tar_st.lookahead.curvature[0],
                        tar_st.lookahead.curvature[1],
                        tar_st.lookahead.curvature[2],
                        ego_st.p.x_tran,
                        ego_st.p.e_psi, 
                        ego_st.v.v_long                       
                        ]).clone()
    
    return input_data



class SampleGeneartorMultiPathPP(SampleGenerator):
    def __init__(self, abs_path,args = None, real_data = False, load_normalization_constant = False, pre_load_data_name = None, randomize=False, elect_function=None, init_all=True, tsne = False):
  
        self.normalized_sample = None
        self.normalized_output = None
        self.args = args
        self.input_dim = args["input_dim"]        
        self.output_dim = args["output_dim"]        
        self.time_horizon = args["n_time_step"]        
        if elect_function is None:
            elect_function = self.useAll
        self.counter = 0
        self.abs_path = abs_path
        self.samples = []
        self.output_data = []
        self.info = []
        self.debug_t = []
        self.means_y = None
        self.stds_y = None        
        self.load_normalization_constant = load_normalization_constant

        if args['eval'] is False:  
            if pre_load_data_name is not None:       
                pre_data_dir = os.path.join(os.path.dirname(abs_path[0]),'preload')              
                pre_data_dir =os.path.join(pre_data_dir,pre_load_data_name+'.pkl')
                self.load_pre_data(pre_data_dir)            
            else:            
                pre_data_dir = os.path.join(os.path.dirname(abs_path[0]),'preload')      
                create_dir(pre_data_dir)        
                pre_data_dir =os.path.join(pre_data_dir,"preload_data.pkl")        
                
                self.gen_samples_with_buffer(real_data = real_data, tsne = tsne, pre_data_dir = pre_data_dir)

            if len(self.samples) < 1 or self.samples is None:
                return 
        
            if randomize:
                self.samples, self.output_data = self.shuffle_in_out_data(self.samples, self.output_data)
            if args['model_name'] is not None:
                self.input_output_normalizing(name = args['model_name'], load_constants=load_normalization_constant)
            print('Generated Dataset with', len(self.samples), 'samples!')
     
    
    def shuffle_in_out_data(self,input, output):
        combined = list(zip(input, output))
        random.shuffle(combined)
        shuffled_list1, shuffled_list2 = zip(*combined)
        shuffled_input = list(shuffled_list1)
        shuffled_output = list(shuffled_list2)
        return shuffled_input, shuffled_output


    def gen_samples_with_buffer(self, real_data = False, tsne = False, pre_data_dir = None):
            for ab_p in self.abs_path:
                for filename in os.listdir(ab_p):
                    if filename.endswith(".pkl"):
                        dbfile = open(os.path.join(ab_p, filename), 'rb')
                        if real_data:
                            scenario_data: SimData = pickle.load(dbfile)                                
                            track_ = scenario_data.track
                        else:
                            scenario_data: RealData = pickle.load(dbfile)                                                            
                            track_ = scenario_data.scenario_def.track

                        if scenario_data.N > self.time_horizon+1+N:                            
                            for t in range(scenario_data.N-1-self.time_horizon-N):                                  
                                def get_x_y_data_from_index(t, scenario_data, tsne = False):                                                                                                               
                                    dat = torch.zeros(self.input_dim, self.time_horizon)                                    
                                    for i in range(t,t+self.time_horizon):                                
                                        ego_st = scenario_data.ego_states[i]
                                        tar_st = scenario_data.tar_states[i]
                                        ntar_orin = scenario_data.tar_states[i+1]
                                        real_dt = ntar_orin.t - tar_st.t 
                                    # [(tar_s-ego_s), ego_ey, ego_epsi, ego_cur,
                                    #                 tar_ey, tar_epsi, tar_cur]      
                                                            
                                        dat[:,i-t]=states_to_encoder_input_torch(tar_st, ego_st, track_)                                          
                               
                                    out = torch.zeros(self.output_dim, N)
                                    ref_tar_st = scenario_data.tar_states[t+self.time_horizon-1].copy()       
                                    for ii in range(t+self.time_horizon, t+self.time_horizon + N):                                        
                                        next_tar_st = scenario_data.tar_states[ii].copy()
                                        tmp_st = scenario_data.tar_states[ii-1].copy()                                    
                                        tar_st = ref_tar_st                                                              
                                        delta_s = wrap_del_s(next_tar_st.p.s,ref_tar_st.p.s, track_)                                                                            
                                                                                                                                                                   
                                        output_t = torch.tensor([delta_s, next_tar_st.p.x_tran, next_tar_st.p.e_psi, next_tar_st.v.v_long ]).clone()                                                                                                                               
                                        out[:,ii-(t+self.time_horizon)] = output_t
                                    
                                    self.debug_t.append(real_dt)
                                    return dat.clone(), out.clone()
                                    
                                    
                                tmp_sample, tmp_output = get_x_y_data_from_index(t,scenario_data, tsne)
                                if tmp_sample is None or tmp_output is None:
                                    continue
                                self.output_data.append(tmp_output)
                                self.samples.append(tmp_sample)

                    
                        dbfile.close()            
            self.save_pre_data(pre_data_dir)
      
    def get_data(self,t, time_length, scenario_data: RealData or SimData, track: RadiusArclengthTrack):
        dat = torch.zeros(self.input_dim, time_length)                                    
        for i in range(t,t+time_length):                                
            ego_st = scenario_data.ego_states[i]
            tar_st = scenario_data.tar_states[i]            
            dat[:,i-t]=states_to_encoder_input_torch(tar_st, ego_st, track)  
        return dat

    def load_pre_data(self,pre_data_dir):        
        model = pickle_read(pre_data_dir)
        self.samples = model['samples']
        self.output_data = model['output_data']    
        print('Successfully loaded data')

    def save_pre_data(self,pre_data_dir):
        model_to_save = dict()
        model_to_save['samples'] = self.samples
        model_to_save['output_data'] = self.output_data        
        pickle_write(model_to_save,pre_data_dir)
        print('Successfully saved data')

    def normalize(self,data):     
        mean = torch.mean(data,dim=0)
        std = torch.std(data,dim=0)        
        if len(data.shape) ==2 :
            new_data = (data - mean.repeat(data.shape[0],1))/std         
        elif len(data.shape) ==3:
            new_data = (data - mean.repeat(data.shape[0],1,1))/std         
        return new_data, mean, std


    def load_normalizing_consant(self, tensor_sample, tensor_output, name ='normalizing'):        
        model = pickle_read(os.path.join(model_dir, name + '_normconstant.pkl'))        
        self.means_x = model['mean_sample']
        self.means_y = model['mean_output']
        self.stds_x = model['std_sample']
        self.stds_y = model['std_output']   

        if tensor_sample is not None:            
            if len(tensor_sample.shape) ==2 :
                self.normalized_sample = (tensor_sample - self.means_x.repeat(tensor_sample.shape[0],1))/self.stds_x         
            elif len(tensor_sample.shape) ==3:
                self.normalized_sample = (tensor_sample - self.means_x.repeat(tensor_sample.shape[0],1,1))/self.stds_x      
        if tensor_output is not None:
            if len(tensor_output.shape) ==2 :
                self.normalized_output = (tensor_output - self.means_y.repeat(tensor_output.shape[0],1))/self.stds_y         
            elif len(tensor_output.shape) ==3:
                self.normalized_output = (tensor_output - self.means_y.repeat(tensor_output.shape[0],1,1))/self.stds_y      

        # self.independent = model['independent'] TODO uncomment        
        print('Successfully loaded normalizing constants', name)


    def input_output_normalizing(self,name = 'normalizing', load_constants = False):
        
        x_tensor = torch.stack(self.samples)

        tensor_output = torch.stack(self.output_data)
        
        if load_constants:
            self.load_normalizing_consant(x_tensor, tensor_output, name=name)
        else:
            self.normalized_sample, mean_sample, std_sample= self.normalize(x_tensor)
            self.normalized_output, mean_output, std_output = self.normalize(tensor_output)        
            model_to_save = dict()
            model_to_save['mean_sample'] = mean_sample
            model_to_save['std_sample'] = std_sample
            model_to_save['mean_output'] = mean_output
            model_to_save['std_output'] = std_output
            pickle_write(model_to_save, os.path.join(model_dir, name + '_normconstant.pkl'))
            print('Successfully saved normalizing constnats', name)
        
    
    
    def nextSample(self):
        self.counter += 1
        if self.counter >= len(self.samples):
            print('All samples returned. To reset, call SampleGenerator.reset(randomize)')
            return None
        else:
            return (self.samples[self.counter - 1], self.output_data[self.counter - 1])
        
    def get_datasets(self, filter = False):        
   

        if self.normalized_output is None:
            inputs= torch.stack(self.samples).to(torch.device("cuda"))  
            labels = torch.stack(self.output_data).to(torch.device("cuda"))
        else:
            inputs = self.normalized_sample.to(torch.device("cuda"))  
            labels = self.normalized_output.to(torch.device("cuda"))  
     
        dataset =  torch.utils.data.TensorDataset(inputs,labels)         
        train_dataset = dataset
        return train_dataset, dataset, dataset
 

