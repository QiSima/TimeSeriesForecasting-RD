import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os
import copy
from task.TaskLoader import Opt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

class Environment(nn.Module):
    def __init__(self, mainNet, auxNets, pred_len, logger,device, r_rate = 0.5):
        super().__init__()
        self.mainNet = mainNet
        self.auxNets = auxNets
        self.logger = logger
        self.r_rate = r_rate
        self.teacher_num = len(self.auxNets) + 1
        self.pred_len = pred_len
        self.device = device
        self.states = None
    
    def gen(self, train_loader, val_loader, dir, pre_epochs = 5):
        # loading models
        self.auxNets['MLP'] = torch.load('{}/mlp_{}_H{}.pkl'.format('Aux_models',dir.split('/')[3],dir.split('/')[5][1:]))
        if self.auxNets['MLP'].best_state != None:
            self.auxNets['MLP'].load_state_dict(self.auxNets['MLP'].best_state)  
        self.auxNets['MLP'].to(self.device)
        self.auxNets['MSVR'] = torch.load('{}/msvr_{}_H{}.pkl'.format('Aux_models',dir.split('/')[3],dir.split('/')[5][1:]))
        self.mainNet.xfit(train_loader,val_loader,self.logger, pre_epochs)
    
    def reset(self, x, target = None, ex_in = None):
        '''
        Input:
            x: (torch.FloatTensor) shape: [batch_size, steps, in_dim]
            target: (torch.FloatTensor) shape: [batch_size, pred_len, 1]; None
            ex_in: 
                RNN: exogenous variables (torch.FloatTensor) shape: [batch_size, pred_len, exogenous_dim]; 
                Attn: position informations (torch.FloatTensor) shape: [batch_size, pred_len, 1]

        Return: 
            obs: state of environment depending on mainNet 
                RNN: hidden_state (torch.FloatTensor) shape: [batch_size, 1, hidden_size]
                Atten: values after attention  shape: [batch_size, 1, d_model]
            done: bool
        '''
        self.current_step = 0
        self.auxs = self.get_auxs(x)
        self.target = target
        self.ex_in = ex_in
        # encoding & one-step decoding
        self.pred = list()
        obs, pred_t  = self.mainNet.enc(x)
        self.pred.append(pred_t)
        return obs,False

    def step(self, action, return_candicates = False):
        '''
        Input:
            action: taken by the agent, only the selected model is one, others are zeros 
                    (torch.FloatTensor) shape: [batch_size, teacher_num]
        Return: 
            obs: state of environment depending on mainNet 
            reward: given by the environment based on the action
                    (torch.FloatTensor) shape: [batch_size, 1]
            done: bool
            pred: mainNet + RL decoding style output
                    (torch.FloatTensor) shape: [batch_size, pred_len, 1]
        '''
        candidates = torch.cat([self.auxs[:,:,self.current_step: self.current_step + 1,:],self.pred[self.current_step].unsqueeze(1)],dim = 1)
        if self.current_step < self.pred_len - 1:
            dec_inp = torch.einsum("bnld,bn->bld",candidates,action)
            ex_in_t = self.ex_in[:,self.current_step: self.current_step + 1,:] if self.ex_in is not None else None
            obs, pred_t = self.mainNet.dec(dec_inp, ex_in_t)
            self.pred.append(pred_t)
            reward = self.cal_reward(candidates, action, pred_t) if self.target is not None else None
            self.current_step += 1
            done = False 
            _pred = None
        else:
            done = True
            obs = None
            dec_inp = None
            reward = self.cal_reward(candidates, action, None)  if self.target is not None else None
            _pred = torch.cat(self.pred,dim=1)    

        if return_candicates:
            return obs, reward, done, _pred, candidates[:,:,:,-1].permute(0,2,1)
        else:
            return obs, reward, done, _pred

    def cal_reward(self, candidates, action, pred_t, beta = 0.1):
        sorted_E = torch.argsort(torch.abs(candidates[:,:,-1,-1] - self.target[:,self.current_step-1,:]),dim=1)
        action_indices = torch.nonzero(action)[:,1]
        rank = torch.nonzero(sorted_E == action_indices.expand(self.teacher_num, -1).permute(1,0))[:,1:]
        r_rank = 1 - rank/(self.teacher_num-1)
        if pred_t is None:
            reward = r_rank
        else:
            r_loss = beta / (beta + torch.abs(pred_t[:,-1,:] - self.target[:,self.current_step,:]))
            reward = self.r_rate *r_loss + (1 - self.r_rate)* r_rank 
        return reward
    
    def get_auxs(self, x):
        auxs = list()
        with torch.no_grad():
            data_x = copy.deepcopy(x)
            for _, auxNet in self.auxNets.items():
                auxs.append(auxNet.predict(data_x))
        auxs = torch.stack(auxs).permute(1,0,2,3)
        return auxs  # [B, aux_num, H, 1]

class AgentConrol:
    def __init__(self, label, recoding = False):
        self.label = label
        self.recoding = recoding
        if self.recoding:
            self.info = Opt()
            self.info.candicates = []
            self.info.val_candicates = []
            self.info.rewards = []
            self.info.val_rewards = []
            self.info.probs = []
            self.info.val_probs = []
    
    def evaluate(self, model, train_loader, val_loader):
        rewards, probs, candicates =  self._eval(model, train_loader)
        val_rewards, val_probs, val_candicates =  self._eval(model, val_loader)
        if self.recoding:
            self.info.rewards.append(rewards)
            self.info.probs.append(probs)
            self.info.candicates.append(candicates)
            self.info.val_rewards.append(val_rewards)
            self.info.val_probs.append(val_probs)
            self.info.val_candicates.append(val_candicates)

        return rewards, val_rewards
    
    def _eval(self,model,data_loader):
        with torch.no_grad():
            rewards = []
            probs = []
            candicates = []
            for batch_x, batch_y in data_loader:
                model.eval()
                batch_x = batch_x.to(model.device)
                batch_y = batch_y.to(model.device)
                batch_probs = []
                batch_candicates = []
                batch_reward = []
                obs,done = model.env.reset(batch_x, target= batch_y[:,:,-1:], ex_in= batch_y[:,:,:-1]) if batch_y.size(2) > 1 else model.env.reset(batch_x,target= batch_y[:,:,-1:])
                while(not done):
                    action, _prob = model.agent.predict(obs, True)
                    next_obs, reward, done, _, candicate= model.env.step(action, return_candicates = True)
                    obs = next_obs
                    batch_probs.append(_prob)
                    batch_candicates.append(candicate)
                    batch_reward.append(reward)
                rewards.append(torch.cat(batch_reward,dim=1))
                probs.append(torch.stack(batch_probs,dim=1))
                candicates.append(torch.cat(batch_candicates,dim=1))
        del model.agent.policy.rewards[:]         
        del model.agent.policy.saved_log_probs[:]
        return self.cal_rewards(torch.cat(rewards, dim=0), model.agent.gamma), torch.cat(probs, dim=0), torch.cat(candicates, dim=0)
    
    def cal_rewards(self, rewards, gamma):
        R = 0
        returns = []
        step_num = rewards.size(1)
        for i in range(step_num):
            R = rewards[:, step_num - (i + 1): step_num - i] + gamma * R
            returns.insert(0,R)        
        returns = torch.cat(returns,dim = 1)
        record_retruns = torch.mean(returns)
        return record_retruns    

    def save(self, dir):
        path = '{}/Log'.format(dir)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.info, path+'/'+'{}.pth'.format(self.label))
    

class PolicyGAgent(nn.Module):
    def __init__(self, obs_n, act_n, device, hidden_size = 128,dropout_rate = 0.6,learning_rate=1e-2, step_gamma = 0.99,gamma=0.99):
        super().__init__()
        self.obs_n = obs_n
        self.act_n = act_n
        self.gamma = gamma
        self.device = device
        self.learning_rate = learning_rate
        self.policy = Policy(obs_n, act_n,device,hidden_size,dropout_rate)
        self.optimizer = optim.Adam(self.policy.parameters(),lr=learning_rate)
        self.eps = np.finfo(np.float32).eps.item()  
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=step_gamma)

    def select_action(self,state):
        probs = self.policy(state)
        m = Categorical(probs)     
        action = m.sample()       
        self.policy.saved_log_probs.append(m.log_prob(action))   
        _martix = torch.zeros((action.size(0),self.act_n))   
        if action.is_cuda:
            _martix = _martix.to(self.device)
        return _martix.scatter(dim=1,index=action.unsqueeze(1),value = 1)
        
    def predict(self,state,return_prob = False):
        probs = self.policy(state)
        action =torch.max(probs,1)[1]
        _martix = torch.zeros((action.size(0),self.act_n))
        if action.is_cuda:
            _martix = _martix.to(self.device)
        if return_prob:
            return _martix.scatter(dim=1,index=action.unsqueeze(1),value = 1), probs    
        else:
            return _martix.scatter(dim=1,index=action.unsqueeze(1),value = 1)
    
    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0,R)        
        returns = torch.cat(returns,dim = 1)
        saved_log_probs = torch.stack(self.policy.saved_log_probs,dim = 1)
        record_retruns = torch.mean(returns)    
        returns = (returns - torch.mean(returns)) / (torch.std(returns) + self.eps)     
        self.optimizer.zero_grad()
        policy_loss = - torch.sum(torch.mul(saved_log_probs,returns))  
        policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        del self.policy.rewards[:]         
        del self.policy.saved_log_probs[:]
        
        return record_retruns.item()

class Policy(nn.Module):
    def __init__(self,obs_n, act_n,device,hidden_size = 128,dropout_rate = 0.6):
        super(Policy,self).__init__()
        self.affline1 = nn.Linear(obs_n,hidden_size).to(device)
        self.dropout = nn.Dropout(p=dropout_rate).to(device)
        self.affline2 = nn.Linear(hidden_size,act_n).to(device)

        self.saved_log_probs = []
        self.rewards = []
    def forward(self, x):
        x = self.affline1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affline2(x)
        return F.softmax(action_scores,dim=1)