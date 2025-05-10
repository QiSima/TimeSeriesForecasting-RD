from data.base import nn_base
from task.TaskLoader import TaskDataset
import os
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

        self.hyper.encoder_hidden_size = 60
        self.hyper.encoder_num_layer = 1
        self.hyper.decoder_hidden_size = 100
        self.hyper.decoder_num_layer = 1
        self.hyper.decoder_input_size = 80
        self.hyper.learning_rate = 0.008556093932847451
        self.hyper.step_gamma = 0.99

class aux_model(nn_base):
    def __init__(self):
        super().__init__()
    def hyper_init(self,):
        self.hyper.mlp_hidden_size = 100
        self.hyper.mlp_lr  = 0.01245
        self.hyper.mlp_stepLr  = 40
        self.hyper.mlp_gamma = 0.66
        
        self.hyper.msvr_C= 0.15848931924611143
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
        self.info.period = 60
        self.info.batch_size = 512

        self.info.pred_len = self.info.period
        self.info.label_len = 30  #   'start token length of Informer decoder'

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
        
        self.hyper.agent_hidden_size = 96
        self.hyper.agent_dropout_rate = 0.4
        self.hyper.agent_lr = 0.028992973197831437
        self.hyper.agent_step_gamma = 0.93
        self.hyper.error_rate = 0.9        