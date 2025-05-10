import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from task.TaskLoader import Opt

import numpy as np

from sklearn.metrics import mean_squared_error

from tqdm import trange, tqdm
import copy
import torch
import torch.nn as nn
# from torch.nn.utils.rnn import PackedSequence
# import torch.optim as optim



class MLP(nn.Module):
    def __init__(self, timesteps, output_dim, hidden_size,device,learning_rate =  0.0001,step_lr = 20,gamma = 0.9):
        super().__init__()

        self.Input_dim = timesteps
        self.Output_dim = output_dim
        self.Hidden_Size = hidden_size
        
        self.device = device

        self.hidden = nn.Linear(self.Input_dim,self.Hidden_Size).to(device)
        self.fc = nn.Linear(self.Hidden_Size,self.Output_dim).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_lr, gamma)
        self.loss_fn = nn.MSELoss()
        
        self.best_state = None
        
        
    def forward(self, input):
        if len(input.shape) > 2:
            input = input[:,:,-1]
        h=self.hidden(input)
        h=torch.sigmoid(h)
        pred =self.fc(h)
        return pred

    def xfit(self, train_loader, val_loader, epochs = 100):
        min_vrmse = 9999
        min_rmse = 9999
        train_len = len(train_loader)

        epoch = 0
        for epoch in trange(epochs):
            rmse_train = 0
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(torch.float32).to(self.device)
                batch_y = batch_y.to(torch.float32).to(self.device)
                self.optimizer.zero_grad()
                y_pred = self(batch_x)
                loss = self.loss_fn(y_pred, batch_y)
                loss.backward()
                rmse_train += np.sqrt(loss.item())

                self.optimizer.step()
            
            rmse_train = rmse_train / train_len
            
            self.epoch_scheduler.step()
            
            with torch.no_grad():
                _,v_y,v_pred = self.loader_pred(val_loader,using_best=False)
                if len(v_pred.shape) > 2:
                    v_pred = v_pred[:,:,-1]
                rmse_val = np.sqrt(mean_squared_error(v_pred,v_y))
            
            if rmse_val < min_vrmse:
                min_vrmse = rmse_val
                min_rmse = rmse_train
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.state_dict()) 
        return min_rmse

    def predict(self, x):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        '''        
        flag = True
        if len(x.shape) > 2:
            x = x[:,:,-1]
            flag = False
        # test_batch: shape: [full-len, sample, dim]
        output = self(x)
        # output = output.squeeze(1)
        if flag:
            return output
        else:
            return torch.unsqueeze(output,dim=2)
    
    def loader_pred(self, data_loader,using_best= True):
        if self.best_state != None and using_best:
            self.load_state_dict(self.best_state)
        x = []
        y = []
        pred = []
        for batch_x, batch_y in tqdm(data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_pred = self.predict(batch_x)
            x.append(batch_x)
            y.append(batch_y)
            pred.append(batch_pred)
        x = torch.cat(x, dim=0).detach().cpu().numpy()
        y = torch.cat(y, dim=0).detach().cpu().numpy()
        pred = torch.cat(pred, dim=0).detach().cpu().numpy()
        
        return x, y, pred