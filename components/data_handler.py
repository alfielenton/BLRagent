import torch
import cv2

def obs_processor(args,initial):
    def process_stack(stack):
        images = []
        for im in range(stack.shape[0]):
            image = stack[im]
            image = image.mean(axis=2)
            image = cv2.resize(image,(84,84),interpolation = cv2.INTER_AREA)
            image = image[:,:40]
            images.append(torch.tensor(image).to(torch.float32))
        return torch.stack(images,dim=0)/255.0

    if initial:
        state , _ = args
        state = process_stack(state)
        return state.detach()
    else: 
        obs , reward , terminated , truncated , _ = args
        next_state = process_stack(obs)
        return reward , next_state.detach() , terminated or truncated , terminated
    
def cartpole_obs_processor(args,initial):
    if initial:
        state , _ = args
        return torch.tensor(state)
    else:
        obs , reward , terminated , truncated , _ = args
        return reward , torch.tensor(obs) , terminated or truncated , terminated
    
def reward_shape(reward,terminated):

    if reward == 1.0:
        return 1.0
    elif terminated and reward == 0.0:
        return -0.1
    else:
        return 0.0