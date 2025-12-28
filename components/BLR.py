import numpy as np
import torch
from scipy.stats import norm

##CHECK NOISE VALUES: WHERE IS VARIANCE NEEDED AND WHERE IS STD NEEDED 

class BayesianLayer:
    
    def __init__(self,n_actions,forget,sig_n,sig_w,n_steps,n_weights = 512):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_2 = 'cpu' if self.device == 'cuda' else None

        self.N = int(n_weights)
        self.A = int(n_actions)
        self.N_STEPS = n_steps

        self.sig_n = sig_n
        self.sig_w = sig_w

        self.forget = forget
        self.phiphiT = torch.zeros((self.A,self.N,self.N),dtype=torch.float32,device=self.device).detach()
        self.phiY = torch.zeros((self.A,self.N),dtype=torch.float32,device=self.device).detach()

        self.weights = torch.tensor(np.random.standard_normal((self.A,self.N)),dtype=torch.float32,device=self.device).detach()
        self.Cov = torch.zeros((self.A,self.N,self.N),dtype=torch.float32,device=self.device).detach()
        self.Cov_decom = torch.zeros((self.A,self.N,self.N),dtype=torch.float32,device=self.device).detach()
        for a in range(self.A):
            self.Cov[a] = torch.eye(self.N,dtype=torch.float32,device=self.device).detach()*self.sig_w
            self.Cov_decom[a] = torch.linalg.cholesky(self.Cov[a])
        
    def update_posterior(self,batch,gamma,online,target,w_online,z_t=None):
        
        online.eval()
        target.eval()

        self.phiphiT *= (1 - self.forget)
        self.phiY *= (1 - self.forget)
        self.forget = 0

        w_target = self.weights.clone().detach()

        states = torch.stack([t.state for t in batch]).to(self.device).detach()
        actions = torch.tensor([t.action for t in batch],dtype=torch.long).to(self.device).detach()
        rewards = torch.tensor([t.reward for t in batch],dtype=torch.float32).to(self.device).detach()
        next_states = torch.stack([t.next_state for t in batch]).to(self.device).detach()
        dones = torch.tensor([t.done for t in batch],dtype=torch.float32).to(self.device).detach()

        with torch.no_grad():
            phi = online(states)
            phi_next_online = online(next_states)
            phi_next_target = target(next_states)
        
        next_online_qs = phi_next_online @ w_online.T

        if z_t is not None:
            L = self.Cov_decom
            v = torch.einsum('aij,bj->abi',L.transpose(1,2),phi_next_online)
            vars = (v * v).sum(dim=2)
            stds = torch.sqrt(vars)
            next_online_qs += z_t*stds.T

        next_actions = next_online_qs.argmax(dim=1)
        next_target_qs = (w_target[next_actions] * phi_next_target).sum(dim=1)      

        targets = rewards + (1 - dones)*gamma*next_target_qs

        for a in range(self.A):
            mask = actions == a
            phi_a = phi[mask]
            targets_a = targets[mask]
            if phi_a.size(0) == 0:
                continue

            self.phiphiT[a] += phi_a.T @ phi_a
            self.phiY[a] += phi_a.T @ targets_a


        for a in range(self.A):
            inv_ = self.phiphiT[a]/self.sig_n + torch.eye(self.N,device=self.device,dtype=torch.float32)/self.sig_w
            inv = torch.linalg.inv(inv_).to(self.device).to(torch.float32).detach()
            inv = (inv + inv.T)/2.
            self.weights[a] = (inv @ self.phiY[a]/self.sig_n).to(self.device).to(torch.float32)
            self.Cov[a] = inv
            try:
                self.Cov_decom[a] = torch.linalg.cholesky(self.Cov[a])
            except torch.linalg.LinAlgError:
                self.Cov_decom[a] = torch.linalg.cholesky(torch.eye(self.N,dtype=torch.float32,device=self.device)).detach()
                self.forget = 1
                print('Reverted to Identity for cholesky decomposition. Forgetting set.')