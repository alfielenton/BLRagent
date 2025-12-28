import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self,capacity):
        self.memory = deque(maxlen=capacity)

    def push(self,state,action,reward,next_state,done):
        self.memory.append(Transition(state,action,reward,next_state,done))

    def sample(self,batch_size):
        batch_size = min(batch_size,self.__len__())
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)

class PriorityReplayMemory:
    def __init__(self, capacity,priority_ratio):
        self.memory = deque(maxlen=capacity)  
        self.priority_memory = deque(maxlen=capacity)
        self.priority_ratio = priority_ratio

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))
        if reward == 1.0:
            self.priority_memory.append(Transition(state,action,reward,next_state,done))

    def sample(self, batch_size):

        batch_size = min(self.__len__(),batch_size)
        if self.__len_priority__() == 0:
            return random.sample(self.memory,batch_size)
    
        num_priority = random.randint(0,self.__len_priority__()) if self.__len_priority__() < self.priority_ratio * batch_size else random.randint(0,int(self.priority_ratio*batch_size))
        num_regular = batch_size - num_priority

        priority_batch = random.sample(self.priority_memory,num_priority) if num_priority > 0 else []
        regular_batch = random.sample(self.memory,num_regular)
        batch = priority_batch+regular_batch
        random.shuffle(batch)
        return batch


    def __len__(self):
        return len(self.memory)
    
    def __len_priority__(self):
        return len(self.priority_memory)
