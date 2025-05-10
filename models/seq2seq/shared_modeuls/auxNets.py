import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import mean_squared_error
from tqdm import trange
import copy
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_size = 100, learning_rate =  0.0001,step_lr = 20,gamma = 0.9):
        super().__init__()

        self.Input_dim = input_dim
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
        
    def predict(self, input):
        input = input[:,:,-1]
        h = torch.sigmoid(self.hidden(input))
        pred = self.fc(h)
        return torch.unsqueeze(pred,dim=2)

    def xfit(self, train_loader, val_loader, logger, epochs = 100):
        min_vrmse = 9999
        logger.info('AuxNet-MLP Learning....')    
        for e in trange(epochs):        
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.predict(batch_x)
                loss = self.loss_fn(y_pred, batch_y[:,:,-1:])
                loss.backward()
                self.optimizer.step()
            self.epoch_scheduler.step()
            
            with torch.no_grad():
                val_y,vpred = self.loader_pred(val_loader)
                vrmse = torch.sqrt(self.loss_fn(val_y,vpred))
                if vrmse < min_vrmse:
                    min_vrmse = vrmse
                    self.best_state = copy.deepcopy(self.state_dict())
        
        self.load_state_dict(self.best_state)
        y,pred = self.loader_pred(train_loader)
        trmse = torch.sqrt(self.loss_fn(y,pred))
        val_y,vpred = self.loader_pred(val_loader)
        vrmse = torch.sqrt(self.loss_fn(val_y,vpred))
        logger.info('Training RMSE: {:.8f} ; Validating RMSE: {:.8f}'.format(trmse,vrmse)) 

    def loader_pred(self, data_loader):
        y = []
        pred = []
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            y.append(batch_y[:,:,-1:])
            pred.append(self.predict(batch_x))
        y = torch.cat(y, dim=0)
        pred = torch.cat(pred, dim=0)
        return y,pred

class MSVR():
    def __init__(self, device, kernel='rbf', degree=3, gamma=None, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1):
        super(MSVR, self).__init__()
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.Beta = None
        self.NSV = None
        self.xTrain = None
        self.device = device
        self.loss_fn = nn.MSELoss()

    def fit(self, x, y):
        x = x[:,:,-1]
        y = y[:,:,-1]
        self.xTrain = (x.numpy()).copy()
        y = (y.numpy()).copy()
        C = self.C
        epsi = self.epsilon
        tol = self.tol

        n_m = np.shape(x)[0]  # num of samples
        n_d = np.shape(x)[1]  # input data dimensionality
        n_k = np.shape(y)[1]  # output data dimensionality (output variables)

        # H = kernelmatrix(ker, x, x, par)
        H = pairwise_kernels(x, x, metric=self.kernel, filter_params=True,
                             degree=self.degree, gamma=self.gamma, coef0=self.coef0)

        self.Beta = np.zeros((n_m, n_k))

        #E = prediction error per output (n_m * n_k)
        E = y - np.dot(H, self.Beta)
        #RSE
        u = np.sqrt(np.sum(E**2, 1, keepdims=True))

        #RMSE
        RMSE = []
        RMSE_0 = np.sqrt(np.mean(u**2))
        RMSE.append(RMSE_0)

        #points for which prediction error is larger than epsilon
        i1 = np.where(u > epsi)[0]

        #set initial values of alphas a (n_m * 1)
        a = 2 * C * (u - epsi) / u

        #L (n_m * 1)
        L = np.zeros(u.shape)

        # we modify only entries for which  u > epsi. with the sq slack
        L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2

        #Lp is the quantity to minimize (sq norm of parameters + slacks)
        Lp = []
        BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
        Lp_0 = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
        Lp.append(Lp_0)

        eta = 1
        k = 1
        hacer = 1
        val = 1

        while(hacer):
            Beta_a = self.Beta.copy()
            E_a = E.copy()
            u_a = u.copy()
            i1_a = i1.copy()

            M1 = H[i1][:, i1] + \
                np.diagflat(1/a[i1]) + 1e-10 * np.eye(len(a[i1]))

            #compute betas
            sal1 = np.dot(np.linalg.inv(M1), y[i1])

            eta = 1
            self.Beta = np.zeros(self.Beta.shape)
            self.Beta[i1] = sal1.copy()

            #error
            E = y - np.dot(H, self.Beta)
            #RSE
            u = np.sqrt(np.sum(E**2, 1)).reshape(n_m, 1)
            i1 = np.where(u >= epsi)[0]

            L = np.zeros(u.shape)
            L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2

            #%recompute the loss function
            BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
            Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
            Lp.append(Lp_k)

            #Loop where we keep alphas and modify betas
            while(Lp[k] > Lp[k-1]):
                eta = eta/10
                i1 = i1_a.copy()

                self.Beta = np.zeros(self.Beta.shape)
                #%the new betas are a combination of the current (sal1)
                #and of the previous iteration (Beta_a)
                self.Beta[i1] = eta*sal1 + (1-eta)*Beta_a[i1]

                E = y - np.dot(H, self.Beta)
                u = np.sqrt(np.sum(E**2, 1)).reshape(n_m, 1)

                i1 = np.where(u >= epsi)[0]

                L = np.zeros(u.shape)
                L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2
                BetaH = np.dot(np.dot(self.Beta.T, H), self.Beta)
                Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
                Lp[k] = Lp_k

                #stopping criterion 1
                if(eta < 1e-16):
                    Lp[k] = Lp[k-1] - 1e-15
                    self.Beta = Beta_a.copy()

                    u = u_a.copy()
                    i1 = i1_a.copy()

                    hacer = 0

            #here we modify the alphas and keep betas
            a_a = a.copy()
            a = 2 * C * (u - epsi) / u

            RMSE_k = np.sqrt(np.mean(u**2))
            RMSE.append(RMSE_k)

            if((Lp[k-1]-Lp[k])/Lp[k-1] < tol):
                hacer = 0

            k = k + 1

            #stopping criterion #algorithm does not converge. (val = -1)
            if(len(i1) == 0):
                hacer = 0
                self.Beta = np.zeros(self.Beta.shape)
                val = -1

        self.NSV = len(i1)

    def _predict(self, input):
        if input.is_cuda:
            input = input.cpu()
        x = input[:,:,-1].numpy()
        H = pairwise_kernels(x, self.xTrain, metric=self.kernel, filter_params=True,
                             degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        yPred = np.dot(H, self.Beta)
        return torch.from_numpy(yPred).unsqueeze(2)
    
    def predict(self, input):
        return self._predict(input).float().to(self.device)
    
    def xfit(self, train_loader, val_loader, logger, epochs = None):
        logger.info('AuxNet-MSVR Learning....')    
        x,y = concentrateLoader(train_loader)
        self.fit(x,y)

        trmse = torch.sqrt(self.loss_fn(y,self._predict(x)))
        val_x,val_y = concentrateLoader(val_loader)
        vrmse = torch.sqrt(self.loss_fn(val_y,self._predict(val_x)))
        logger.info('Training RMSE: {:.8f} ; Validating RMSE: {:.8f}'.format(trmse,vrmse)) 
    
def concentrateLoader(data_loader):
    x = []
    y = []
    for batch_x, batch_y in data_loader:
        x.append(batch_x[:,:,-1:])
        y.append(batch_y[:,:,-1:])
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    if x.is_cuda:
        return x.cpu(), y.cpu()
    else:
        return x,y