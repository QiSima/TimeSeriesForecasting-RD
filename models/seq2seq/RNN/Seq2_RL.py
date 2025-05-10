from random import randint
from matplotlib.style import context
import torch
import torch.nn as nn
import numpy as np
from numpy import *
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import sys
import os
from tqdm import tqdm
from task.TaskLoader import Opt
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from models.seq2seq.RL.PG import PolicyGAgent
from models.seq2seq.other.MSVR import MSVR
from models.seq2seq.other.MLP import MLP
import gc
from task.tools import EarlyStopping
import copy
from collections import OrderedDict
import time


class Encoder(nn.Module):
    def __init__(self, component,input_size, hidden_size, num_layers,device):
        super().__init__()
        self.component = component
        if component == 'LSTM':
            self.layer = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True).to(device)
        elif component == 'GRU':
            self.layer = nn.GRU(input_size, hidden_size, num_layers,batch_first=True).to(device)
        else:
            self.layer = nn.RNN(input_size, hidden_size, num_layers,batch_first=True).to(device)

    def forward(self, input_seq):
        if self.component == 'LSTM':
            output, (h, c) = self.layer(input_seq)
            return h, c
        else:
            output, h = self.layer(input_seq)
            return h, None

class Decoder(nn.Module):
    def __init__(self, component,input_size, hidden_size, num_layers,device):
        super().__init__()
        self.component = component
        if component == 'LSTM':
            self.layer = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True).to(device)
        elif component == 'GRU':
            self.layer = nn.GRU(input_size, hidden_size, num_layers,batch_first=True).to(device)
        else:
            self.layer = nn.RNN(input_size, hidden_size, num_layers,batch_first=True).to(device)
        self.linear = nn.Linear(hidden_size, 1).to(device)

    def forward(self, input_x,h=None, c=None):
        # teachers: 2 * batch_size * h
        if self.component == 'LSTM':
            if h is None:
                output, (h, c) = self.layer(input_x)
            else:    
                output, (h, c) = self.layer(input_x, (h, c))
            output = self.linear(h[-1])
            return h,c,output
        else:
            if h is None:
                output, h = self.layer(input_x)
            else:    
                output, h = self.layer(input_x, h)
            output = self.linear(h[-1])
            return h,None,output

class Environment(nn.Module):
    def __init__(self, opts=None, logger=None):
        super().__init__()
        self.opts = opts
        self.logger = logger
        self.other_teachers = None
        
        self.component = opts.component
        
        self.device = opts.device
        self.encoder_input_size = opts.num_variate
        self.encoder_hidden_size = opts.encoder_hidden_size
        self.encoder_num_layer = opts.encoder_num_layer
        self.decoder_input_size = opts.decoder_input_size
        self.decoder_hidden_size = opts.decoder_hidden_size
        self.decoder_num_layer = opts.decoder_num_layer
        self.H = opts.H
        
        self.data_name = opts.data_name
        
        self.environment_lr = opts.learning_rate
        self.environment_gamma = opts.step_gamma
        
        self.Encoder = Encoder(self.component,self.encoder_input_size, self.encoder_hidden_size, self.encoder_num_layer,self.device)
        self.Context = nn.Linear(self.encoder_hidden_size*self.encoder_num_layer,self.decoder_input_size - 1).to(self.device)     
        self.Decoder = Decoder(self.component,self.decoder_input_size, self.decoder_hidden_size, self.decoder_num_layer,self.device)
        
        self.loss_fn = nn.MSELoss()
        paras= [{'params':self.Encoder.parameters()},
                {'params':self.Context.parameters()},
                {'params':self.Decoder.parameters()},
                ]

        self.optimizer = torch.optim.Adam(paras, lr=self.environment_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=self.environment_gamma)
    
    def initial_environment(self,train_loader, val_loader,preTrain_epochs):
        self.get_MLP(train_loader,val_loader)
        self.get_MSVR(train_loader)
        self.get_seq2seq(train_loader,preTrain_epochs)
    
    def reset_environment(self,data_x,data_y,agent_y):
        _h, _= self.Encoder(data_x)           
        context = self.Context(_h.view(_h.size(1),-1)).unsqueeze(1) 
        batch_size  = data_x.size(0)
        decoder_input = torch.tensor(agent_y).to(self.device)[:, :self.H - 1].view(batch_size,-1)
        decoder_input = torch.cat((data_x[:,-1,-1].view(batch_size,1,-1),decoder_input.unsqueeze(2)),dim=1)
        pred = list()
        for t in range(self.H):
            if t == 0 :
                _input = torch.cat((decoder_input[:,t,:].unsqueeze(1),context),dim=2)
                h_t,c_t,output_t = self.Decoder(_input)
            else:
                _input = torch.cat((decoder_input[:,t,:].unsqueeze(1),context),dim=2)
                h_t,c_t,output_t = self.Decoder(_input,h_t,c_t)
            pred.append(output_t)
        pred = torch.cat(pred,dim = 1)
        loss = torch.sqrt(self.loss_fn(pred,data_y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def reset_learning(self):
        self.optimizer.param_groups[0]['lr']  = self.environment_lr
    
    def get_MSVR(self,data_loader):
        model_path = 'Aux_models/{}_{}_H{}.pkl'.format('msvr',self.data_name,self.H)   
        if os.path.exists(model_path):  
            self.svr = torch.load(model_path)
        else:
            msvr_s_time = time.time()   
            self.svr = MSVR(kernel='rbf', degree=3, gamma=self.opts.msvr_gamma, coef0=0.0, tol=0.001, C=self.opts.msvr_C, epsilon=self.opts.msvr_epsilon)
            x,y = concentrateLoader(data_loader,self.device)
            x = x[:,:,-1]
            if torch.cuda.is_available():
                self.svr.fit(x.cpu(),y.cpu())
            else:
                self.svr.fit(x,y)
            msvr_e_time = time.time()
            if not os.path.exists('Aux_models'):
                os.makedirs('Aux_models')
            torch.save(self.svr,model_path)
            self.logger.info('MSVR training time:{}'.format(msvr_e_time-msvr_s_time))
    def get_MLP(self,train_loader,val_loader):
        model_path = 'Aux_models/{}_{}_H{}.pkl'.format('mlp',self.data_name,self.H)
        if os.path.exists(model_path):  
            self.mlp = torch.load(model_path)
            if self.mlp.best_state != None:
                self.mlp.load_state_dict(self.mlp.best_state)    
        else:            
            mlp_s_time = time.time()
            self.mlp = MLP(self.opts.steps, self.H, hidden_size = self.opts.mlp_hidden_size,  device = self.opts.device,learning_rate =  self.opts.mlp_lr ,step_lr = self.opts.mlp_stepLr, gamma = self.opts.mlp_gamma)
            self.mlp.xfit(train_loader,val_loader,epochs=100)   
            self.logger.info(">>> MLP Total params: {:.2f}M".format(
                sum(p.numel() for p in list(self.mlp.parameters())) / 1000000.0))
            mlp_e_time = time.time()
            if not os.path.exists('Aux_models'):
                os.makedirs('Aux_models')
            torch.save(self.mlp,model_path) 
            self.logger.info('MLP training time:{}'.format(mlp_e_time-mlp_s_time))
            
    def get_seq2seq(self,train_loader,preTrain_epochs):
        self.fit_seq2seq(train_loader,preTrain_epochs)
        
    def fit_seq2seq(self,data_loader,preTrain_epochs):
        self.logger.info('PreTrain seq2seq....')
        for _ in tqdm(range(preTrain_epochs)):
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()
                
                _h, _= self.Encoder(data_x)           
                context = self.Context(_h.view(_h.size(1),-1)).unsqueeze(1) 
                batch_size  = data_x.size(0)
                decoder_input = data_y[:, :self.H - 1].view(batch_size,-1)
                decoder_input = torch.cat((data_x[:,-1,-1].view(batch_size,1,-1),decoder_input.unsqueeze(2)),dim=1)
                
                pred = list()
                # decoder step-by-step
                for t in range(self.H):
                    if t== 0:
                        _input  = torch.cat((decoder_input[:,t,:].unsqueeze(1),context),dim=2)
                        h_t,c_t,pred_t = self.Decoder(_input)
                    else:
                        _input  = torch.cat((decoder_input[:,t,:].unsqueeze(1),context),dim=2)
                        h_t,c_t,pred_t = self.Decoder(_input,h_t,c_t)
                    pred.append(pred_t)
                pred = torch.cat(pred,dim = 1)
                loss = torch.sqrt(self.loss_fn(pred,data_y))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        self.logger.info('PreTrain seq2seq Finish.')
                
    def get_other_teachers(self,data_x):
        
        teach_s_time = time.time()
        other_teachers = list()
        with torch.no_grad():
            # data_x_temp = data_x[:,:,-1]
            data_x_temp = data_x
            if torch.cuda.is_available():
                pred_svr = self.svr.predict(data_x_temp.cpu()).to(self.device)
            else:
                pred_svr = self.svr.predict(data_x_temp)
            pred_svr = pred_svr.unsqueeze(1)
            pred_mlp = self.mlp.predict(data_x)
            pred_mlp = pred_mlp.squeeze(2).unsqueeze(1)
        other_teachers.append(pred_svr)
        other_teachers.append(pred_mlp)
        other_teachers = torch.cat(other_teachers,dim = 1)
        teach_e_time = time.time()
        self.logger.info('Get Teachers time:{}'.format(teach_e_time-teach_s_time))
        return other_teachers
    
    def init_state(self,data_x):
        # encoder
        _h, _= self.Encoder(data_x)           
        context = self.Context(_h.view(_h.size(1),-1)).unsqueeze(1) 
        batch_size  = data_x.size(0)
        decoder_input = data_x[:,-1,-1].view(batch_size,1,-1)
        
        _input = torch.cat((decoder_input,context),dim=2)
        # initial state
        h_t,c_t,freeL_t = self.Decoder(_input) 
        
        return h_t,c_t,freeL_t,context
        
    def observation(self,t, context = None, y_t = None, y_t_next = None, action = None, h_t = None,c_t = None,teachers=None,error_rate = 0.5):
        with torch.no_grad():
            pred_t = torch.gather(teachers,1,action).float()
            _rank = self.get_Rank(y_t,teachers,pred_t)
            if y_t_next is None:
                reward = _rank
                h_next,c_next,teachers = None,None,None
            else:
                decoder_input = torch.cat((pred_t.unsqueeze(1),context),dim=2)
                h_next,c_next,freeL_t = self.Decoder(decoder_input,h_t,c_t)
                teachers = torch.cat((freeL_t,self.other_teachers[:,:,t+1]),dim = 1)
                _error = 0.1/(0.1+torch.abs(freeL_t[:,0] - y_t_next))
                reward = error_rate *_error + (1 - error_rate) * _rank
        return h_next,c_next,reward,teachers
    
    def get_Rank(self,y_t,teachers,pred_t):
        error_teachers = torch.abs(teachers - y_t.unsqueeze(1))
        error_agentSelect = torch.abs(pred_t - y_t.unsqueeze(1))

        num_sample = error_teachers.size(0)
        num_model = error_teachers.size(1)
        _rankAll = torch.zeros(num_sample).to(self.device)
        for i in range(num_sample):
            _rank = 0
            for j in range(num_model):
                if error_agentSelect[i] > error_teachers[i][j]:
                    _rank = _rank + 1
            _rankAll[i] = 1 - _rank/(num_model - 1)
        return _rankAll

class Seq2Seq_RL(nn.Module):
    def __init__(self, opts=None, logger=None):
        super(Seq2Seq_RL, self).__init__()
        self.opts = opts
        self.logger = logger
        
        self.preTrain_epochs = opts.preTrain_epochs
        self.agent_lr = opts.agent_lr
        self.agent_step_gamma = opts.agent_step_gamma
        
        self.epochs = opts.epochs
        self.device = opts.device
        self.component = opts.component
        self.H = opts.H
        self.act_n = 3
        self.early_stopping = EarlyStopping(patience= opts.patience)

        self.environment = Environment(opts, logger)
        self.agent = PolicyGAgent(
            obs_n = self.opts.decoder_hidden_size * self.opts.decoder_num_layer,
            act_n= self.act_n,
            device = self.device,
            hidden_size = self.opts.agent_hidden_size,
            dropout_rate = self.opts.agent_dropout_rate,
            learning_rate = self.agent_lr,
            step_gamma = self.agent_step_gamma
            )
    
        self.fit_info = Opt()
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
        
        self.fit_info.seq2seq_loss_list = []
        self.fit_info.seq2seq_vloss_list = []
        
        self.best_environment = None
        self.best_policy = None
        self.best_state = None
    
    def forward(self,encoder_input):
        other_teachers = self.environment.get_other_teachers(encoder_input)
        _h, _ = self.environment.Encoder(encoder_input)
       
        context = self.environment.Context(_h.view(_h.size(1),-1)).unsqueeze(1) 
        # decoder step-by-step agent
        pred = []
        pred_agent = []
        batch_size  = encoder_input.size(0)
        decoder_input = encoder_input[:,-1,-1].view(batch_size,1,-1)
        for t in range(self.H):
            if t == 0:
                _input = torch.cat((decoder_input,context),dim=2)
                h_t,c_t,output_t = self.environment.Decoder(_input)
            else:
                _input = torch.cat((pred_t.unsqueeze(1),context),dim=2)
                h_t,c_t,output_t = self.environment.Decoder(_input,h_t,c_t)
            pred.append(output_t)
            # agent action
            obs = h_t.view(h_t.size(1),-1)
            with torch.no_grad():
                action  = self.agent.predict(obs)
                teachers = torch.cat((output_t,other_teachers[:,:,t]),dim = 1)
            pred_t =  torch.gather(teachers,1,action).float()
            pred_agent.append(pred_t)
        pred = torch.cat(pred,dim = 1)
        pred_agent = torch.cat(pred_agent,dim = 1)
        return pred,pred_agent
    
    def agent_learn(self,data_x,data_y,error_rate = 0.5):
        # 初始化 T_agent:True表示用agent的输出值与真实值的匹配程度进行训练
        self.environment.other_teachers = self.environment.get_other_teachers(data_x)
        h_t,c_t,freeL_t,context = self.environment.init_state(data_x)
        teachers = torch.cat((freeL_t,self.environment.other_teachers[:,:,0]),dim = 1)
        for t in range(self.H):
            obs = h_t.view(h_t.size(1),-1)
            action  = self.agent.select_action(obs)
            if t == self.H - 1:
                data_y_next = None
            else:
                data_y_next = data_y[:,t + 1]
            h_next,c_next,reward,teachers = self.environment.observation(t,context,data_y[:,t],data_y_next, action,h_t, c_t,teachers,error_rate)
            self.agent.policy.rewards.append(reward)
            h_t = h_next
            c_t = c_next
        loss = self.agent.finish_episode()
        self.environment.other_teachers = None
        return loss
    
    def train_agent(self,train_loader, val_loader,iter_num):
        for k in tqdm(range(iter_num)):
            R = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                train_x, train_y = batch_x.detach().clone(), batch_y.detach().clone()
                reward_item = self.agent_learn(train_x,train_y,error_rate=self.opts.error_rate)
                R = R + reward_item
            self.agent.scheduler.step()
            
    def train_environment(self,data_loader, decoder_input,iter_num):
        for _ in range(iter_num):
            index = 0
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                train_x, train_y = batch_x.detach().clone(), batch_y.detach().clone()
                if decoder_input is None:
                    self.environment.reset_environment(train_x,train_y,train_y)
                else:
                    batch_size = train_x.size(0)
                    self.environment.reset_environment(train_x,train_y,decoder_input[index:(index+batch_size),:])
                    index = index + batch_size
            self.environment.scheduler.step() 
    
    def evaluate_seq(self, train_loader, val_loader):
        with torch.no_grad():
            _,y,pred_agent,pred = self.predict(train_loader)
            trmse = np.sqrt(mean_squared_error(y,pred))
            
            _,val_y,_,vpred = self.predict(val_loader)
            vrmse = np.sqrt(mean_squared_error(val_y,vpred))
            
            self.fit_info.loss_list.append(trmse)
            self.fit_info.vloss_list.append(vrmse)

            return trmse,vrmse,pred_agent

        
    def xfit(self, train_loader, val_loader):
        min_vrmse = 9999
        self.logger.info('agent_T_agent_rate: {}\t '.format(self.opts.error_rate))
        
        # 环境初始化
        self.environment.initial_environment(train_loader, val_loader,self.preTrain_epochs)

        # 参数初始化
        self.environment.optimizer.param_groups[0]['lr']  = self.opts.learning_rate
        self.agent.optimizer.param_groups[0]['lr']  = self.opts.agent_lr
        base_iter = self.opts.base_iter  # 10
        # 环境和agent螺旋式提升
        for epoch in tqdm(range(self.epochs)):
            if epoch > 0:
                self.train_environment(train_loader,y,self.opts.train_seq_item)
            if (epoch + 1) % base_iter  == 0 or epoch == 0 or (epoch + 1) == self.epochs:
                self.train_agent(train_loader,val_loader,base_iter)
            trmse,vrmse,y = self.evaluate_seq(train_loader,val_loader)
            if vrmse < min_vrmse :
                min_vrmse = vrmse
                self.fit_info.trmse = trmse
                self.fit_info.vrmse = vrmse
                self.best_epoch = epoch
                self.best_policy =  copy.deepcopy(self.agent.policy.state_dict())
                self.best_environment = copy.deepcopy(self.environment.state_dict())
            self.xfit_logger(epoch)
            torch.cuda.empty_cache() if next(self.environment.parameters()).is_cuda else gc.collect()
            torch.cuda.empty_cache() if next(self.agent.policy.parameters()).is_cuda else gc.collect()
            
            self.early_stopping(vrmse, self)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                break
        if self.best_environment != None:
            self.environment.load_state_dict(self.best_environment)
            self.agent.policy.load_state_dict(self.best_policy)
            self.best_state = copy.deepcopy(self.state_dict())
        return self.fit_info
            
    def xfit_logger(self,epoch):
        self.logger.info('Epoch:{};Training RMSE: : {:.8f} \n Validating RMSE: {:.8f}'.format(
            epoch,self.fit_info.trmse,self.fit_info.vrmse))
    
    def predict(self,data_loader,return_action = False):
        x = []
        y = []
        pred_seq2seq = list()
        pred_agent = list()
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_pred_seq2seq,batch_pred_agent = self.forward(batch_x)
            x.append(batch_x)
            y.append(batch_y)
            pred_seq2seq.append(batch_pred_seq2seq)
            pred_agent.append(batch_pred_agent)
        x = torch.cat(x, dim=0).detach().cpu().numpy()
        y = torch.cat(y, dim=0).detach().cpu().numpy()
        pred_seq2seq = torch.cat(pred_seq2seq, dim=0).detach().cpu().numpy()
        pred_agent = torch.cat(pred_agent, dim=0).detach().cpu().numpy()
        if return_action:
            return x,y,pred_agent,pred_seq2seq
        else:
            return x,y,pred_agent,pred_seq2seq
    
    def loader_pred(self, data_loader, using_best= True,return_action = True):
        if self.best_environment != None and using_best:
            self.environment.load_state_dict(self.best_environment)
            self.agent.policy.load_state_dict(self.best_policy)
        x,y,_,pred_seq2seq = self.predict(data_loader,return_action)
        return x, y, [pred_seq2seq]
        
def concentrateLoader(data_loader,device):
    x = []
    y = []
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        x.append(batch_x)
        y.append(batch_y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    return x,y
        