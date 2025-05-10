import torch
import torch.nn as nn
from tqdm import tqdm
from task.TaskLoader import Opt
from sklearn.metrics import mean_squared_error
import gc
import sys
import os
from models.seq2seq.ESLSTM.modules.cells import ESLSTMCell 
from task.tools import adjust_learning_rate, EarlyStopping
from models.seq2seq.shared_modeuls.mode_RL import PolicyGAgent, Environment, AgentConrol
from models.seq2seq.shared_modeuls.auxNets import MLP,MSVR
from models.seq2seq.ESLSTM.modules.atten import VariableAttention,FusionAttention
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import copy

class ESLSTM_RL(nn.Module):
    def __init__(self, opts=None, logger=None):
        super(ESLSTM_RL, self).__init__()
        self.opts = opts
        self.logger = logger
        self.device = opts.device
        
        self.env = Environment(
            ES_Net(opts.input_dim, opts.hidden_size, opts.pred_len, self.device),
            auxNets = {'MSVR': MSVR(self.device, gamma= opts.msvr_gamma, C = opts.msvr_C, epsilon= opts.msvr_epsilon),
                     'MLP':MLP(opts.steps,opts.pred_len,self.device, opts.mlp_hidden_size, opts.mlp_lr, opts.mlp_stepLr,opts.mlp_gamma)},
            pred_len = opts.pred_len,
            logger = self.logger,
            device = self.device,
            r_rate = opts.r_rate
        )
        self.agent = PolicyGAgent(
            obs_n = 2 * opts.hidden_size * opts.input_dim,
            act_n = 3,
            device = self.device,
            hidden_size = opts.agent_size,
            learning_rate = opts.agent_lr
        )
        
        self.pre_epochs = opts.pre_epochs
        self.epochs = opts.epochs
        self.learning_rate = opts.learning_rate
        self.lradj = 'type1'
        self.early_stopping = EarlyStopping(patience=opts.patience)

        self.recordAgent = opts.recordAgent
        
        self.loss_fn = nn.MSELoss()
        self.best_state = None
        self.fit_info = Opt()
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = [] 

    def train_agent(self, train_loader,val_loader, trainTime, sub_epoch = 20):
        self.logger.info('Agent Learning....')    
        control = AgentConrol(trainTime, self.recordAgent)
        early_stopping = EarlyStopping(sub_epoch//5, reward=True, verbose= False, delta=0.0)
        path = '{}/Agent'.format(self.opts.series_dir)
        for i in range(sub_epoch):
            for batch_x, batch_y in train_loader:
                self.agent.train()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                obs,done = self.env.reset(batch_x, target= batch_y[:,:,-1:], ex_in= batch_y[:,:,:-1]) if batch_y.size(2) > 1 else self.env.reset(batch_x,target= batch_y[:,:,-1:])
                while(not done):
                    action = self.agent.select_action(obs)
                    next_obs, reward, done, _ = self.env.step(action)
                    self.agent.policy.rewards.append(reward)
                    obs = next_obs
                self.agent.finish_episode()
            
            # evaluate
            R, val_R = control.evaluate(self, train_loader, val_loader)
            self.logger.info('\n Agent Epoch:{} ; Training Reward: {:.8f} ; Validating Reward: {:.8f}'.format(i,R, val_R))  
            
            early_stopping(val_R, self.agent, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            adjust_learning_rate(self.agent.optimizer, i + 1 , self.opts.agent_lr)
        if control.recoding:
            control.save(path)
    
    def train_env(self, train_loader, sub_epoch):
        for i in range(sub_epoch):
            for batch_x, batch_y in train_loader:
                self.env.mainNet.train()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.env.mainNet.optimizer.zero_grad()
                obs,done = self.env.reset(batch_x, ex_in= batch_y[:,:,:-1]) if batch_y.size(2) > 1 else self.env.reset(batch_x)
                while(not done):
                    action = self.agent.predict(obs)
                    next_obs, _, done, batch_pred = self.env.step(action) 
                    obs = next_obs
                loss = self.loss_fn(batch_pred, batch_y[:,:,-1:])
                loss.backward()
                self.env.mainNet.optimizer.step()
    
    def forward(self,input, ex_in = None):
        obs,done = self.env.reset(input, ex_in = ex_in)
        batch_actions = list()
        while(not done):
            action = self.agent.predict(obs)
            next_obs, _, done, batch_pred = self.env.step(action)
            obs = next_obs
            batch_actions.append(action)
        return batch_pred, torch.stack(batch_actions,dim=1)
    
    def xfit(self, train_loader, val_loader):   
        min_vrmse = 9999 
        trainTime = 1
        base_iter = self.opts.base_iter
        self.env.gen(train_loader,val_loader, dir = self.opts.series_dir, pre_epochs = self.pre_epochs)
        for epoch in tqdm(range(self.epochs)):
            if (epoch + 1) % base_iter  == 0 or epoch == 0:
                self.train_agent(train_loader,val_loader, trainTime,base_iter)
                trainTime = trainTime + 1
            self.train_env(train_loader,1)
            min_vrmse = self.evaluate(epoch, min_vrmse, train_loader, val_loader)
            torch.cuda.empty_cache() if next(self.parameters()).is_cuda else gc.collect()
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                break
            adjust_learning_rate(self.env.mainNet.optimizer, epoch+1, self.learning_rate,self.lradj)
            
        return self.fit_info
            
    def xfit_logger(self,e, trmse,vrmse):
        self.logger.info('\n MainNet Epoch:{} ; Training RMSE: {:.8f} ; Validating RMSE: {:.8f}'.format(e,trmse,vrmse))    
    
    def evaluate(self, epoch, min_vrmse, train_loader, val_loader):
        with torch.no_grad():
            _,y,pred = self.loader_pred(train_loader,using_best=False,return_action = False)
            trmse = mean_squared_error(y,pred)

            _,val_y,vpred = self.loader_pred(val_loader,using_best=False,return_action = False)
            vrmse = mean_squared_error(val_y,vpred)
            
            self.fit_info.loss_list.append(trmse)
            self.fit_info.vloss_list.append(vrmse)
            if vrmse < min_vrmse:
                min_vrmse = vrmse
                self.fit_info.trmse = trmse
                self.fit_info.vrmse = vrmse
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.state_dict())
        self.xfit_logger(epoch + 1,trmse,vrmse)
        self.early_stopping(vrmse, self, self.opts.series_dir)
        return min_vrmse
    
    def loader_pred(self, data_loader, using_best = True, return_action = True):
        if self.best_state != None and using_best:
            self.load_state_dict(self.best_state)
        x = []
        y = []
        pred = []
        actions = []
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                self.eval()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_pred,batch_actions  = self(batch_x, ex_in= batch_y[:,:,:-1]) if batch_y.size(2) > 1 else self(batch_x)
                x.append(batch_x)
                y.append(batch_y[:,:,-1:])
                pred.append(batch_pred)
                actions.append(batch_actions)
        x = torch.cat(x, dim=0).detach().cpu().numpy()
        y = torch.cat(y, dim=0).detach().cpu().numpy()
        pred = torch.cat(pred, dim=0).detach().cpu().numpy()
        
        # return H * teachers
        actions = torch.cat(actions,dim=0)
        actionsMean = torch.mean(actions,dim=0)
        
        if return_action:
            return x,  y[:,:,0], [pred[:,:,0],actionsMean.detach().cpu().numpy(),actions.detach().cpu().numpy()]
        else:
            return x, y[:,:,0], pred[:,:,0]


class ES_Net(nn.Module):
    def __init__(self, input_dim, hidden_size, pred_len,device):
        super(ES_Net, self).__init__()
        self.netCell = ESLSTMCell(input_dim,hidden_size,device)
        self.varAttn = VariableAttention(input_dim, hidden_size).to(device)
        self.fusionAttn = FusionAttention(1, hidden_size).to(device)
        self.device = device
        self.pred_len  = pred_len
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_fn = nn.MSELoss()

    def initHidden(self, input):
        batchSize = input.data.size(0)
        result = torch.empty(batchSize, 1, self.hidden_size).float()
        result = nn.init.normal_(result, std=0.015)
        return result.to(self.device)

    def enc(self, data_x):
        '''
        Encoding history information, return (one-step-aheah prediction & observation) 
        Input:
            data_x: (torch.FloatTensor) shape: [batch_size, steps, input_dim]
        Return: 
            obs: (torch.FloatTensor) shape: [batch_size, 2 * hidden_size * input_dim]
            y_pred: one-step-aheah prediction 
                    (torch.FloatTensor) shape: [batch_size, 1, 1]
        '''
        batch_size = data_x.size(0)
        self.h_states,self.h_t = self.netCell(data_x,(self.initHidden(data_x),self.initHidden(data_x)))
        p = self.varAttn(self.h_states)
        y_pred = self.fusionAttn(self.h_t[0], p)
        return torch.cat((self.h_t[0], p),dim=2).view(batch_size,-1), y_pred

    def dec(self, dec_inp, ex_in = None):
        '''
        Decoding step by step, return (current-step prediction & observation) 
        Input:
            dec_inp: (torch.FloatTensor) shape: [batch_size, steps, input_dim]
            ex_in: 
                RNN: exogenous variables (torch.FloatTensor) shape: [batch_size, pred_len, exogenous_dim]; 
                Attn: position informations (torch.FloatTensor) shape: [batch_size, pred_len, 1]
        Return: 
            obs: (torch.FloatTensor) shape: [batch_size, 2 * hidden_size * input_dim]
            y_pred: one-step-aheah prediction 
                    (torch.FloatTensor) shape: [batch_size, 1, 1]
        '''        
        batch_size = dec_inp.size(0)
        inp = torch.cat([ex_in, dec_inp],dim=2) if ex_in is not None else dec_inp
        _,self.h_t = self.netCell(inp,self.h_t)
        self.h_states = torch.cat([self.h_states[:,1:,:,:],self.h_t[0].unsqueeze(1)],dim=1)
        p = self.varAttn(self.h_states)
        y_pred = self.fusionAttn(self.h_t[0], p)
        return torch.cat((self.h_t[0], p),dim=2).view(batch_size,-1), y_pred
    
    def forward(self, input_x, ex_in = None):
        # Decoding by free mode
        pred = list()
        _,_pred = self.enc(input_x)
        pred.append(_pred)
        for step in range(self.pred_len-1):
            dec_inp = _pred
            _in = ex_in[:,step : step + 1,:] if ex_in is not None else None
            _,_pred = self.dec(dec_inp, _in)
            pred.append(_pred)
        pred = torch.cat(pred, dim=1)
        return pred
        
    def xfit(self, train_loader,val_loader,logger, pre_epochs=10):
        min_vrmse = 9999
        logger.info('MainNet Autonomous Learning....')    
        for e in range(pre_epochs):
            for batch_x, batch_y in train_loader:
                self.train()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                pred = self(batch_x, batch_y[:,:,:-1]) if self.input_dim > 1 else self(batch_x)
                loss = torch.sqrt(self.loss_fn(pred, batch_y[:,:,-1:]))
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                y,pred = self.loader_pred(train_loader)
                trmse = torch.sqrt(self.loss_fn(y,pred))

                val_y,vpred = self.loader_pred(val_loader)
                vrmse = torch.sqrt(self.loss_fn(val_y,vpred))
                
                logger.info('\nEpoch: {} ; Training RMSE: {:.8f} ; Validating RMSE: {:.8f}'.format(e,trmse,vrmse)) 

                if vrmse < min_vrmse:
                    min_vrmse = vrmse
                    self.best_state = copy.deepcopy(self.state_dict())
            
        self.load_state_dict(self.best_state)
            
    def loader_pred(self, data_loader):
        y = []
        pred = []
        for batch_x, batch_y in data_loader:
            self.eval()
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_pred = self(batch_x, batch_y[:,:,:-1]) if self.input_dim > 1 else self(batch_x)
            y.append(batch_y[:,:,-1:])
            pred.append(batch_pred)
        y = torch.cat(y, dim=0)
        pred = torch.cat(pred, dim=0)
        return y,pred