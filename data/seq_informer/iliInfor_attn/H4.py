from task.TaskLoader import TaskDataset
import pandas as pd
from data.base import nn_base,Inf_base
from ray import tune

class aux_model(nn_base):
    def __init__(self):
        super().__init__()
    def hyper_init(self,):

        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  = 0.01
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.67

        self.hyper.msvr_C= 1.0
        self.hyper.msvr_epsilon= 0.1
        self.hyper.msvr_gamma= None


class iliInfor_data(TaskDataset):
    def __init__(self, args):
        super().__init__(args)
    
    def info_config(self):
        self.info.normal = True
        self.info.data_path = 'data/paper/seq/real/ili/ILI.csv'
        self.info.series_name = ['south']
        self.info.num_series = len(self.info.series_name) 
        self.info.num_variate = 1
        self.info.cov_dim = 0
        self.info.input_dim = self.info.num_variate

        self.info.steps = 26
        self.info.period = 4
        self.info.batch_size = 32

        self.info.pred_len = self.info.period
        self.info.label_len = 12  #   'start token length of Informer decoder'

    def sub_config(self,):
        self.seriesPack = [] 
        for i, name in enumerate(self.info.series_name):
            df = pd.read_csv(self.info.data_path, header=0)
            data = df[name+'_ILI']
            if data.isnull().any():
                data= data.interpolate()
            raw_ts = data.values.reshape(-1, )
                    
            sub = self.pack_dataset(raw_ts)
            sub.index = i
            sub.name = name
            sub.H = self.info.H
            sub.merge(self.info)            
            self.seriesPack.append(sub)

class Informer_RL(Inf_base):
    def __init__(self):
        super().__init__()
    
    def base_modify(self,): 
        self.import_path = 'models/seq2seq/Attention/Informer_RL.py' 
        self.class_name = 'Informer_RL'
        self.arch = 'atten_base'

    def hyper_modify(self):
        _aux = aux_model()
        self.hyper.mlp_hidden_size,self.hyper.mlp_lr,self.hyper.mlp_stepLr,self.hyper.mlp_gamma = _aux.hyper.mlp_hidden_size, _aux.hyper.mlp_lr,_aux.hyper.mlp_stepLr,_aux.hyper.mlp_gamma
        self.hyper.msvr_C, self.hyper.msvr_epsilon, self.hyper.msvr_gamma= _aux.hyper.msvr_C, _aux.hyper.msvr_epsilon, _aux.hyper.msvr_gamma
        
        self.hyper.pre_epochs = 20
        self.hyper.base_iter = 50
        self.hyper.recordAgent = True
        self.hyper.agent_size = 128
        self.hyper.agent_lr = 1e-1
        self.hyper.r_rate = 0.5