import torch
import time

class StatisticsCollector:

    def __init__(self,timer_path,results_path,model_paths,weight_path):

        self.results_path = results_path
        self.timer_path = timer_path
        self.model_paths = model_paths
        self.weight_path = weight_path
        self.steps_path = 'model_checkpoints//steps_taken.txt'

        with open(self.results_path,'w') as f:
            f.write('')
        
        with open(self.timer_path,'w') as f:
            f.write('')

        self.best_episode = (0,0)
        self.eps_counter = 0
        self.eps_score = 0
        self.good_rewards = 0

    def set_start_step(self,start_step,eps_start):
        self.step_counter = start_step
        self.eps_counter = eps_start

    def record_rewards(self,reward,done):

        with open(self.results_path,'a') as f:
            next_charac = '\n' if done else ','
            f.write(f'{reward}'+next_charac)

        self.step_counter += 1
        self.eps_counter += 1 if done else 0
        self.good_rewards += 1 if reward > 0 else 0
        self.eps_score += reward

        if done:
            if self.eps_score >= self.best_episode[1]:
                self.best_episode = (self.eps_counter,self.eps_score)
            eps_time_secs = int(time.time() - self.eps_timer)

            eps_time_secs = eps_time_secs % (24*3600)
            eps_time_hrs = eps_time_secs // 3600

            eps_time_secs %= 3600

            eps_time_mins = eps_time_secs // 60

            eps_time_secs %= 60

            with open(self.timer_path,'a') as f:
                f.write(f'Eps {self.eps_counter},Eps time:{eps_time_hrs}H{eps_time_mins}M{eps_time_secs}S,Code time:'+self.show_time_taken()+f',Eps reward: {self.eps_score},Good Rewards: {self.good_rewards},Steps taken: {self.step_counter}\n')

            self.eps_timer = time.time()
            self.last_eps_score = self.eps_score
            self.eps_score = 0

    def show_best_episode(self):
        return f'Best episode: {self.best_episode[0]} | Score: {self.best_episode[1]}'
    
    def write_best_episode(self):
        with open('results//best_episode.txt','w') as f:
            f.write(self.show_best_episode())

    def show_time_taken(self):
        time_secs = int(time.time() - self.timer)
        time_secs = time_secs % (24*3600)
        time_hrs = time_secs // 3600
        time_secs %= 3600
        time_mins = time_secs // 60
        time_secs %= 60
        time_string = f'{time_hrs}h.{time_mins}m.{time_secs}s'
        return time_string

    def start_timer(self):
        self.timer = time.time()
        self.eps_timer = time.time()
    
    def save_model(self,agent,steps):
        torch.save(agent.online.state_dict(),self.model_paths['online'])
        torch.save(agent.target.state_dict(),self.model_paths['target'])
        torch.save(agent.optimiser.state_dict(),self.model_paths['optimiser'])

        torch.save(agent.strategy.weights,self.weight_path['weights'])
        torch.save(agent.strategy.Cov,self.weight_path['Cov'])
        torch.save(agent.strategy.phiphiT,self.weight_path['phiphiT'])
        torch.save(agent.strategy.phiY,self.weight_path['phiY'])

        with open(self.steps_path,'w') as f:
            f.write(f'Steps {steps}\nEps: {self.eps_counter}')

    def load_model(self,model,model_type,device):
        model.load_state_dict(torch.load(self.model_paths[model_type],map_location=device))
        if model_type == 'target':
            return model.eval()
        else:
            return model

    def load_weights(self,weight_type,device):
        return torch.load(self.weight_path[weight_type],map_location=device).detach()
    
    def get_load_step(self):        
        with open(self.steps_path,'r') as f:
            step = int(f.readlines()[0][6:])
            eps = int(f.readlines()[1][5:])
        return step , eps