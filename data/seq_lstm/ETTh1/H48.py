from data.base import nn_base
from task.TaskLoader import TaskDataset
import numpy as np
import pandas as pd

class seq_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'atten_base'
        super().__init__()
        
    def hyper_init(self,):
        self.hyper.epochs = 100
        self.hyper.patience = 8
        self.hyper.component = 'LSTM'
        self.hyper.encoder_hidden_size = 81
        self.hyper.encoder_num_layer = 1
        self.hyper.decoder_hidden_size = 118
        self.hyper.decoder_num_layer = 1
        self.hyper.decoder_input_size = 85
        self.hyper.learning_rate = 0.01
        self.hyper.step_gamma = 0.9281036169831249

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
        self.info.batch_size = 512

        self.info.pred_len = self.info.period
        self.info.label_len = 48  #   'start token length of Informer decoder'

    def sub_config(self,):
        self.seriesPack = []
        for i in range(self.info.num_series):
            _name = self.info.series_name[i]
            df = pd.read_csv(self.info.data_path + self.info.series_name[i] + '.csv',index_col=None,header=0)
            _start = np.where(df.values==self.start_time)[0].item()
            # _start = 0
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
            
        
class Seq2_RL(seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self,): 
        self.import_path = 'models/seq2seq/RNN/Seq2_RL.py' 
        self.class_name = 'Seq2Seq_RL'
        self.arch = 'seq2seq'
    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        self.hyper.preTrain_epochs = 10  
        self.hyper.train_seq_item = 1

        self.hyper.output_agent = False
        _aux = aux_model()
        self.hyper.mlp_hidden_size,self.hyper.mlp_lr,self.hyper.mlp_stepLr,self.hyper.mlp_gamma = _aux.hyper.mlp_hidden_size, _aux.hyper.mlp_lr,_aux.hyper.mlp_stepLr,_aux.hyper.mlp_gamma
        self.hyper.msvr_C, self.hyper.msvr_epsilon, self.hyper.msvr_gamma= _aux.hyper.msvr_C, _aux.hyper.msvr_epsilon, _aux.hyper.msvr_gamma
        
        self.hyper.agent_hidden_size = 384
        self.hyper.agent_dropout_rate = 0.8
        self.hyper.agent_lr = 0.0076035268671218684
        self.hyper.agent_step_gamma = 0.95
        self.hyper.error_rate = 0.3