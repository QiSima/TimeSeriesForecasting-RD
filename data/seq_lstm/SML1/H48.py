from data.base import nn_base
from task.TaskLoader import TaskDataset
import pandas as pd
import os
        
class seq_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'atten_base'
        super().__init__()
        
    def hyper_init(self,):
        self.hyper.epochs = 100
        self.hyper.patience = 8
        self.hyper.component = 'LSTM'
        self.hyper.encoder_hidden_size = 100
        self.hyper.encoder_num_layer = 1
        self.hyper.decoder_hidden_size = 180
        self.hyper.decoder_num_layer = 2
        self.hyper.decoder_input_size = 99
        self.hyper.learning_rate = 0.005
        self.hyper.step_gamma = 0.972608262931005

class aux_model(nn_base):
    def __init__(self):
        super().__init__()
    def hyper_init(self,):
        self.hyper.mlp_hidden_size = 500
        self.hyper.mlp_lr  = 0.00545
        self.hyper.mlp_stepLr  = 40
        self.hyper.mlp_gamma = 0.45
        
        self.hyper.msvr_C= 0.15848931924611143
        self.hyper.msvr_epsilon= 1.0
        self.hyper.msvr_gamma= 0.05

class SML1_data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)
    def info_config(self):
        self.info.normal = False
        self.info.data_path = 'data/paper/seq/real/SML2011'
        self.info.series_name = ['NEW-DATA-1.T15']
        self.info.num_series = len(self.info.series_name) 
        self.info.num_variate = 1
        self.info.cov_dim = 0
        self.info.input_dim = self.info.num_variate

        self.info.steps = 4*24*3
        self.info.period = 4*12
        self.info.batch_size = 512

        self.info.pred_len = self.info.period
        self.info.label_len = 4*24  #   'start token length of Informer decoder'

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
        
        self.hyper.agent_hidden_size = 320
        self.hyper.agent_dropout_rate = 0.2
        self.hyper.agent_lr = 0.03431202379286479
        self.hyper.agent_step_gamma = 0.97
        self.hyper.error_rate = 0.8    