import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation , RecordVideo
import ale_py
gym.register_envs(ale_py)
import agent
import torch
from components.data_handler import *
from components.BDQNet import *
from components import strategies
from stat_collector import StatisticsCollector
from components.parameters import Parameters

opt = Parameters(tetris=False)

env_string = opt.env_string
env = gym.make(env_string,render_mode='rgb_array',max_episode_steps=5000)
if opt.tetris:
    env = FrameStackObservation(env,4)

N_STEPS = opt.N_STEPS
trigger = int(N_STEPS/1000)
env = RecordVideo(env,'videos',episode_trigger = lambda r: True)
print(env_string+' created')

results_path = 'results//training_results.csv'
model_paths = {'online':'model_checkpoints//online.pt',
               'target':'model_checkpoints//target.pt',
               'optimiser':'model_checkpoints//optimiser.pt',
               'phiphiT':'model_checkpoints//phiphi.pt',
               'phiY':'model_checkpoints//phiY.pt'}

weight_path = {'weights':'model_checkpoints//weights.pt',
               'Cov':'model_checkpoints//Cov.pt',
               'phiphiT':'model_checkpoints//phiphi.pt',
               'phiY':'model_checkpoints//phiY.pt'}

timer_path = 'results//timer.csv'
sc = StatisticsCollector(timer_path,results_path,model_paths,weight_path)
print('Stat collector formed')

forget = opt.forget
sig_w = opt.sig_w
sig_n = opt.sig_n
strategy = strategies.Strategy2(env.action_space.n,forget,sig_n,sig_w)
print(f'Strategy {strategy.strat_type} Formed')
if strategy.strat_type == 3:
    print('Stochastic policy' if strategy.stochastic else 'Deterministic policy')

capacity = opt.capacity
model = TetrisNet() if opt.tetris else CartpoleNet()
print('Model parameter set')
gamma = opt.gamma
learning_rate = opt.lr
momentum = opt.momentum
priority_ratio = opt.priority_ratio
batch_size = opt.batch_size
print('Agent parameters set')
myagent = agent.Agent(env,
                      gamma,
                      model,
                      model_paths,
                      learning_rate,
                      momentum,
                      strategy,
                      capacity,
                      batch_size,
                      priority_ratio=priority_ratio)

step_start , eps_start = 0 , 0
load_model = False
if load_model:
    myagent.online = sc.load_model(myagent.online,'online',myagent.device)
    myagent.target = sc.load_model(myagent.target,'target',myagent.device)
    myagent.optimiser = torch.optim.RMSprop(myagent.online.parameters(),learning_rate,momentum=momentum)
    myagent.optimiser = sc.load_model(myagent.optimiser,'optimiser',myagent.device)
    myagent.strategy.weights = sc.load_weights('weights',myagent.device)
    myagent.strategy.Cov = sc.load_weights('Cov',myagent.device)
    myagent.strategy.phiphiT = sc.load_weights('phiphiT',myagent.device)
    myagent.strategy.phiY = sc.load_weights('phiY',myagent.device)
    step_start , eps_start = sc.get_load_step()
    print('Models and weights loaded')
    print(f'Restart at step: {step_start}\nEps: {eps_start}')

print('Agent made')

obs_processor = opt.obs_processor
reward_shaper = opt.reward_shaper
SAVE_INTERVAL = opt.SAVE_INTERVAL
NETWORK_UPDATE = opt.NETWORK_UPDATE
LEARN_STRATEGY = opt.LEARN_STRATEGY
LEARN_NETWORK = opt.LEARN_NETWORK
print('Training parameters set')

sc.set_start_step(step_start,eps_start)
sc.start_timer()
print("Starting training loop...")
done = True

for steps in range(step_start,N_STEPS):

    if done:
        obs = env.reset()
        state = obs_processor(obs,initial=True)
    else:
        state = next_state
    
    action = myagent.select_action(state)
    reward , next_state , done , terminated = obs_processor(env.step(int(action)),initial=False)
    sc.record_rewards(reward,done)
    reward = reward_shaper(reward,terminated)
    myagent.remember(state,action,reward,next_state,terminated)

    if (steps+1) % LEARN_NETWORK == 0:
        batch = myagent.recall()
        myagent.learn_network(batch)

    if int(myagent.learn_count) % NETWORK_UPDATE == 0:
        myagent.update_network()

    if (steps+1) % LEARN_STRATEGY == 0:
        myagent.update_strategy(batch)

    if (steps+1) % int(N_STEPS/1000) == 0:
        print(f"Step {steps+1} | Time taken: {sc.show_time_taken()} | "+sc.show_best_episode())

    if (steps+1) % SAVE_INTERVAL == 0:
        sc.save_model(myagent,steps)

print('Training completed!!')
print(sc.show_best_episode())
sc.write_best_episode()
env.close()    