import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PolicyGAgent():
    def __init__(self, obs_n, act_n, device, hidden_size = 128,dropout_rate = 0.6,learning_rate=1e-2, step_gamma = 0.99,gamma=0.99,sample_rate = 0.1):
        self.obs_n = obs_n
        self.act_n = act_n
        self.gamma = gamma
        self.device = device
        self.sample_rate = sample_rate
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
        if action.is_cuda:
            return torch.cuda.LongTensor(action).unsqueeze(1)
        else:
            return torch.LongTensor(action).unsqueeze(1)
    
    def predict(self,state):
        probs = self.policy(state)
        action =torch.max(probs,1)[1]
        if action.is_cuda:
            return torch.cuda.LongTensor(action).unsqueeze(1)
        else:
            return torch.LongTensor(action).unsqueeze(1)
    
    
    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0,R)        
        returns = torch.stack(returns,dim = 1)
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
    
    def reset_learning(self):
        self.optimizer.param_groups[0]['lr'] = self.learning_rate

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
