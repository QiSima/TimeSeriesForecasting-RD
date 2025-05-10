from ray import tune
from data.base import nn_base,Inf_base
from task.TaskLoader import TaskDataset
import pandas as pd
import os

class aux_model(nn_base):
    def __init__(self):
        super().__init__()
    def hyper_init(self,):

        self.hyper.mlp_hidden_size = 500
        self.hyper.mlp_lr  =  0.00235
        self.hyper.mlp_stepLr  = 50
        self.hyper.mlp_gamma = 0.41
        

        self.hyper.msvr_C= 0.025118864315095794
        self.hyper.msvr_epsilon= 0.001
        self.hyper.msvr_gamma= 0.05

class SML1Infor_data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)
    def info_config(self):
        self.info.normal = True
        self.info.data_path = 'data/paper/seq/real/SML2011'
        self.info.series_name = ['NEW-DATA-1.T15']
        self.info.num_series = len(self.info.series_name) 
        self.info.num_variate = 1
        self.info.cov_dim = 0
        self.info.input_dim = self.info.num_variate

        self.info.steps = 4*24*3
        self.info.period = 2*4*12
        self.info.batch_size = 16

        self.info.pred_len = self.info.period
        self.info.label_len = 4*12  

    def sub_config(self,):
        self.seriesPack = []   
        for i in range(self.info.num_series):
            _name = self.info.series_name[i]
            df = pd.read_csv(os.path.join(self.info.data_path, '{}.txt'.format(_name)),sep=' ')
            data = df['3:Temperature_Comedor_Sensor']
            if data.isnull().any():
                data= data.interpolate()
            raw_ts = data.values.reshape(-1, )
            sub = self.pack_dataset(raw_ts)
            sub.index = i
            sub.name = _name
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