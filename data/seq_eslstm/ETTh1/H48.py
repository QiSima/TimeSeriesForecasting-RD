from ray import tune
from data.base import nn_base
from task.TaskLoader import TaskDataset
import numpy as np
import pandas as pd

class ES_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'atten'
        super().__init__()
        
    def hyper_init(self,):
        self.hyper.epochs = 200
        self.hyper.patience = 35
        self.hyper.learning_rate = 5e-2
        self.hyper.hidden_size = 64
        
class aux_model(nn_base):
    def __init__(self):
        super().__init__()
    def hyper_init(self,):
        self.hyper.mlp_hidden_size = 300
        self.hyper.mlp_lr  = 0.0008500000000000001
        self.hyper.mlp_stepLr  = 40
        self.hyper.mlp_gamma = 0.96

        self.hyper.msvr_C= 0.15848931924611143
        self.hyper.msvr_epsilon= 1.0
        self.hyper.msvr_gamma= 0.05

class ETTh1_data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)
        
    def info_config(self):
        self.info.normal = False
        self.info.data_path = 'data/paper/seq/real/ETT/'
        self.start_time = '2018-01-01 00:00:00'
        self.info.series_name = ['ETTh1']
        self.info.num_series = len(self.info.series_name) 
        self.info.num_variate = 7
        self.info.cov_dim = 0
        self.info.input_dim = self.info.num_variate

        self.info.steps = 24*7
        self.info.period = 48
        self.info.batch_size = 32

        self.info.pred_len = self.info.period
        self.info.label_len = 48  

    def sub_config(self,):
        self.seriesPack = []
        for i in range(self.info.num_series):
            _name = self.info.series_name[i]
            df = pd.read_csv(self.info.data_path + self.info.series_name[i] + '.csv',index_col=None,header=0)
            _start = np.where(df.values==self.start_time)[0].item()
            if self.info.num_variate == 1:
                raw_ts = df.values[_start:,-1]
            else:
                raw_ts = df.values[_start:,1:]
            sub = self.pack_dataset(raw_ts)
            sub.index = i
            sub.name = _name
            sub.H = self.info.H
            sub.merge(self.info)            
            self.seriesPack.append(sub)