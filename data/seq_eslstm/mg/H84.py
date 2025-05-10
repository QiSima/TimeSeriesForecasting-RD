from task.TaskLoader import TaskDataset
import numpy as np
from data.base import nn_base
from ray import tune

class ES_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'atten'
        super().__init__()
        
    def hyper_init(self,):
        self.hyper.epochs = 200
        self.hyper.patience = 35
        self.hyper.learning_rate = 1e-2
        self.hyper.hidden_size = 512

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

class mg_data(TaskDataset):
    def __init__(self, args):
        super().__init__(args)

    def info_config(self):
        self.info.normal = False
        self.info.num_series = 1
        self.info.num_variate = 1
        self.info.cov_dim = 0
        self.info.input_dim = self.info.num_variate

        self.info.steps = 200
        self.info.period = 84
        self.info.batch_size = 32

        self.info.pred_len = self.info.period
        self.info.label_len = 38  
    
    def sub_config(self,):
        self.seriesPack = []
        
        for i in range(self.info.num_series):
            raw_ts = np.load(
                'data/paper/seq/synthetic/mg/mg.npy').reshape(-1,)
            sub = self.pack_dataset(raw_ts)
            sub.index = i
            sub.name = 'mg17'
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