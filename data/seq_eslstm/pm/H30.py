from ray import tune
from data.base import nn_base
from task.TaskLoader import TaskDataset
import os
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
        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  = 0.0915
        self.hyper.mlp_stepLr  = 40
        self.hyper.mlp_gamma = 0.8
        
        self.hyper.msvr_C= 0.14497406703726315
        self.hyper.msvr_epsilon= 0.0001
        self.hyper.msvr_gamma= 0.05


class pm_data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)
    def info_config(self):
        self.info.normal = False
        self.info.data_path = 'data/paper/seq/real/pm2.5'
        self.info.series_name = ['Chengdu']  
        self.info_yids = [2010,2011,2012,2013,2014,2015]
        self.info.num_series = len(self.info.series_name) 
        self.info.num_variate = 1
        self.info.cov_dim = 0
        self.info.input_dim = self.info.num_variate

        self.info.steps = 180
        self.info.period = 30
        self.info.batch_size = 32

        self.info.pred_len = self.info.period
        self.info.label_len = 30  

    def sub_config(self,):
        self.seriesPack = []
        for i in range(self.info.num_series):
            _name = self.info.series_name[i]
            raw_yid = []
            for yid in self.info_yids:
                city_path = os.path.join(self.info.data_path, '{}'.format(yid),'{}.post.csv'.format(_name))
                df = pd.read_csv(city_path,parse_dates=True, index_col=[0])
                df_h = round(df.resample('d').mean(),2)
                data = df_h['PM_US Post']
                null_num = len(data[data.isnull()].index)
                if null_num > 0:
                    data = data.interpolate()
                raw_yid.append(data)
            raw_ts = pd.concat(raw_yid).values.reshape(-1,)
            sub = self.pack_dataset(raw_ts)
            sub.index = i
            sub.name = _name
            sub.H = self.info.H
            sub.merge(self.info)            
            self.seriesPack.append(sub)

class ESLSTM_RL(ES_base):
    def __init__(self):
        super().__init__()
    
    def base_modify(self,): 
        self.import_path = 'models/seq2seq/ESLSTM/ESLSTM_RL.py' 
        self.class_name = 'ESLSTM_RL'
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
    
    def tuning_modify(self):
        self.tuning.agent_size = tune.choice([32, 64, 72, 96, 128, 256])
        self.tuning.pre_epochs = tune.choice([5, 10, 15, 20])
        self.tuning.r_rate = tune.choice([0.25, 0.5, 0.75])