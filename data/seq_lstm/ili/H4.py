from task.TaskLoader import TaskDataset
import pandas as pd
from data.base import nn_base
        
class seq_base(nn_base):
    def __init__(self):
        self.training = True
        self.arch = 'atten_base'
        super().__init__()
        
    def hyper_init(self,):
        self.hyper.epochs = 100
        self.hyper.component = 'LSTM'
        self.hyper.patience = 8

        self.hyper.encoder_hidden_size = 150
        self.hyper.encoder_num_layer = 1
        self.hyper.decoder_hidden_size = 150
        self.hyper.decoder_num_layer = 1
        self.hyper.decoder_input_size = 80
        self.hyper.learning_rate = 0.005
        self.hyper.step_gamma = 0.99

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


class ili_data(TaskDataset):
    def __init__(self, args):
        super().__init__(args)
    
    def info_config(self):
        self.info.normal = False
        self.info.data_path = 'data/paper/seq/real/ili/ILI.csv'
        self.info.series_name = ['south']
        self.info.num_series = len(self.info.series_name) 
        self.info.num_variate = 1
        self.info.cov_dim = 0
        self.info.input_dim = self.info.num_variate

        self.info.steps = 26
        self.info.period = 4
        self.info.batch_size = 512

        self.info.pred_len = self.info.period
        self.info.label_len = 14  #   'start token length of Informer decoder'

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


class Seq2_RL(seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self,): 
        self.import_path = 'models/seq2seq/RNN/Seq2_RL.py' 
        self.class_name = 'Seq2Seq_RL'
        self.arch = 'seq2seq'
        
    def hyper_modify(self):
        _aux = aux_model()
        self.hyper.mlp_hidden_size,self.hyper.mlp_lr,self.hyper.mlp_stepLr,self.hyper.mlp_gamma = _aux.hyper.mlp_hidden_size, _aux.hyper.mlp_lr,_aux.hyper.mlp_stepLr,_aux.hyper.mlp_gamma
        self.hyper.msvr_C, self.hyper.msvr_epsilon, self.hyper.msvr_gamma= _aux.hyper.msvr_C, _aux.hyper.msvr_epsilon, _aux.hyper.msvr_gamma
        
        self.hyper.preTrain_epochs = 5
        self.hyper.base_iter = 10
        self.hyper.recordAgent = True

        self.hyper.train_seq_item = 10
        
        self.hyper.agent_hidden_size = 256
        self.hyper.agent_dropout_rate = 0.3
        self.hyper.agent_lr = 0.0005
        self.hyper.agent_step_gamma = 0.93
        self.hyper.error_rate = 1