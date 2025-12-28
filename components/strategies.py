import torch
import numpy as np
from components.BLR import BayesianLayer as bl
from scipy.stats import norm

class Strategy1(bl):
    '''Strategy 1 uses standard Thompson sampling of the posterior weight distribution
      then chooses the optimal action with respect to the sampled weights'''
    def __init__(self,n_actions,forget,sig_n,sig_w,n_weights=512):

        super().__init__(n_actions,forget,sig_n,sig_w,n_weights)
        self.strat_type = 1
        self.w = torch.tensor(np.random.standard_normal((self.A,self.N)),dtype=torch.float32,device=self.device).detach()*self.sig_w
    
    def select_action(self,state,network):
        
        network.eval()
        self.w = torch.tensor(np.random.standard_normal((self.A,self.N)),dtype=torch.float32,device=self.device).detach()
        for a in range(self.A):
            self.w[a] = self.Cov_decom[a] @ self.w[a] + self.weights[a]

        with torch.no_grad():
            PHI = network(state.to(self.device))
            if PHI.dim() == 2 and PHI.size(0) == 1:
                PHI = PHI.squeeze(0)

        if torch.isnan(PHI).any():
            raise ValueError('Error in action selection: PHI contains NaN')
        
        q_vals = self.w @ PHI
        return q_vals.argmax()

class Strategy2(bl):
    '''Strategy 2 uses the predicted weight vector to calculate posterior predictive distribution,
    it then samples from this distribution for each action and chooses the action that samples the highest
    prediction'''
    def __init__(self,n_actions,forget,sig_n,sig_w,n_weights=512):

        super().__init__(n_actions,forget,sig_n,sig_w,n_weights)
        self.strat_type = 2
    
    def select_action(self,state,network):
        
        network.eval()
        with torch.no_grad():
            PHI = network(state.to(self.device))
            if PHI.dim() == 2 and PHI.size(0) == 1:
                PHI = PHI.squeeze(0)

        if torch.isnan(PHI).any():
            raise ValueError('Error in action selection: PHI contains NaN')

        means = self.weights @ PHI
        L = self.Cov_decom
        v = torch.einsum('aij,j->ai',L.transpose(1,2),PHI)
        vars = (v * v).sum(dim=1)
        stds = torch.sqrt(vars)

        q_vals = torch.tensor(np.random.standard_normal(self.A),device=self.device,dtype=torch.float32)
        q_vals = stds * q_vals + means
        return q_vals.argmax()

class Strategy3(bl):
    '''Strategy 3 forms a distribution for a|s from the predictive stats and parameter alpha
    which is scheduled to increase from 0 to 1 over episode times'''
    def __init__(self,n_actions,forget,sig_n,sig_w,n_weights=512):

        super().__init__(n_actions,forget,sig_n,sig_w,n_weights)
        self.strat_type = 3
        self.t = 0
        self.stochastic = False
    
    def select_action(self,state,network):
        self.t += 1
        self.z_t = max(0.5,3.0*float(np.exp(-self.t/670000)))
        with torch.no_grad():
            PHI = network(state.to(self.device))
            if PHI.dim() == 2 and PHI.size(0) == 1:
                PHI = PHI.squeeze(0)

        if torch.isnan(PHI).any():
            raise ValueError('Error in action selection: PHI contains NaN')
        
        means = self.weights @ PHI
        L = self.Cov_decom
        v = torch.einsum('aij,j->ai',L.transpose(1,2),PHI)
        vars = (v * v).sum(dim=1)
        stds = torch.sqrt(vars)
        q_vals = means + self.z_t * stds

        if self.stochastic:
            probs = torch.softmax(q_vals,dim=0)
            action = torch.multinomial(probs,num_samples=1).item()
            return action
        else:
            return q_vals.argmax()
