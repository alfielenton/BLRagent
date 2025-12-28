import torch
import torch.nn as nn
import torch.optim as optim
import copy
from components import replay_memory
from scipy.stats import norm

class Agent:

    def __init__(self,env,gamma,model,model_paths,lr,momentum,strategy,
                 memory_capacity,batch_size,priority_ratio=None):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_2 = 'cpu' if self.device == 'cuda' else None
        print(f"Main device: {self.device}")
        print(f"Secondary device: {self.device_2}")

        self.env = env
        self.A = self.env.action_space.n
        self.gamma = gamma

        self.batch_size = batch_size

        self.online = model.to(self.device)
        print('Online model made!')

        self.target = copy.deepcopy(model).to(self.device)
        self.target.eval()
        print('Target model made!')
        print('Models made!')

        self.lr = lr
        self.momentum = momentum
        self.loss_fn = nn.MSELoss()

        self.model_paths = model_paths
        
        self.optimiser = optim.RMSprop(self.online.parameters(),self.lr,momentum=self.momentum)

        self.strategy = strategy
        
        if priority_ratio is not None:
            self.memory = replay_memory.PriorityReplayMemory(memory_capacity,priority_ratio)
        else:
            self.memory = replay_memory.ReplayMemory(memory_capacity)

        self.learn_count = 0
        self.learn_fails = 0

    def select_action(self,state):
        self.online.eval()
        return self.strategy.select_action(state,self.online)
    
    def remember(self,*args):
        self.memory.push(*args)

    def recall(self):
        batch = self.memory.sample(self.batch_size)
        return batch
    
    def learn_network(self, batch):
        
        self.learn_count += 1

        self.online.train()
        self.target.eval()

        w_online = self.strategy.w.clone().detach() if self.strategy.strat_type == 1 else self.strategy.weights.clone().detach()
        w_target = self.strategy.weights.clone().detach()

        states = torch.stack([t.state for t in batch]).to(self.device)  
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device) 
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([t.next_state for t in batch]).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        phi_online = self.online(states)
        with torch.no_grad():
            phi_next_online = self.online(next_states)
            phi_next_target = self.target(next_states)

        next_online_qs = phi_next_online @ w_online.T

        if self.strategy.strat_type == 3:
            L = self.strategy.Cov_decom
            v = torch.einsum('aij,bj->abi',L.transpose(1,2),phi_next_online)
            vars = (v * v).sum(dim=2)
            stds = torch.sqrt(vars)
            next_online_qs += self.strategy.z_t * stds.T

        next_actions = next_online_qs.argmax(dim=1) 
        next_target_qs = (phi_next_target * w_target[next_actions]).sum(dim=1)

        target = rewards + self.gamma * (1 - dones) * next_target_qs
        target = torch.clamp(target,min=-30,max=100)

        q_preds = (w_online[actions] * phi_online).sum(dim=1)
        loss = self.loss_fn(target,q_preds)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        test_sample = states[0]
        self.online.eval()
        phi_test = self.online(test_sample)

        if torch.isnan(phi_test).any():
            self.learn_count += -1
            self.learn_fails += 1
            if self.learn_fails > 10:
                print('Stopping training')
                raise RuntimeError('Found NaN values in output of online')
            
            state_dict = torch.load(self.model_paths['online'],map_location=self.device,weights_only=True)
            self.online.load_state_dict(state_dict)
            self.optimiser = optim.RMSprop(self.online.parameters(),self.lr,momentum=self.momentum)
            print(f'Failed to learn\nFail learn count: {self.learn_fails}')
            
    def update_network(self):
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    def update_strategy(self,batch):
        w_online = self.strategy.w.clone().detach() if self.strategy.strat_type == 1 else self.strategy.weights.clone().detach()
        z_t = self.strategy.z_t if self.strategy.strat_type == 3 else None
        self.strategy.update_posterior(batch,self.gamma,self.online,self.target,w_online,z_t=z_t)