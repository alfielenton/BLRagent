from .data_handler import *

class Parameters:

    def __init__(self,tetris):

        ##env parameters
        self.tetris = tetris
        self.env_string = 'ALE/Tetris-v5' if tetris else 'CartPole-v1'

        ##Network parameters
        self.in_channels = 4
        self.out_channels = 512

        #BLR and strategies parameters
        self.forget = 0.
        self.sig_w = .01
        self.sig_n = 1.

        ##Agent parameters
        self.gamma = 0.97
        self.batch_size = 32
        self.lr = 1e-4
        self.momentum = 0.95

        ##Main parameters
        self.obs_processor = obs_processor if tetris else cartpole_obs_processor
        self.reward_shaper = reward_shape if tetris else lambda r , d : r 
        self.N_STEPS = int(2e6) if tetris else int(1e6)
        self.capacity = int(1e6) if tetris else int(2e4)
        self.priority_ratio = None
        self.LEARN_NETWORK = 4 if tetris else 1
        self.LEARN_STRATEGY = self.LEARN_NETWORK
        self.NETWORK_UPDATE = 5000 if tetris else 8000
        self.SAVE_INTERVAL = int(self.N_STEPS/50)
