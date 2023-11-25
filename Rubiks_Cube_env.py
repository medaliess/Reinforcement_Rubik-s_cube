import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from Cube import Cube

class RubiksEnv():
    def __init__(self):
        self.action_space = Discrete(12)
        self.observation_space = MultiDiscrete([6]*54)
        self.reset_number = 0
        self.move_number = 0
        self.actions_dict={
            0 : 'R',
            1 : 'R\'',
            2 : 'L',
            3 : 'L\'',
            4 : 'U',
            5 : 'U\'',
            6 : 'D',
            7 : 'D\'',
            8 : 'F',
            9 : 'F\'',
            10 : 'B',
            11 : 'B\''
        }

    
    def step(self, action):
        reward=0
        if not self.cube.isdone():
            move=self.actions_dict[action]
            self.cube.do(move)
            p=self.cube.solvability_percentage()/100
            reward =-1+p
        self.move_number +=1

        
        
        
        done = self.cube.isdone() or self.move_number >= self.shuffle_number*2 +1
        
        if self.cube.isdone():  
            reward = 10

    
        return self.cube.get_array(),reward,done
        
    def reset(self,N_shuffle=5):
        self.cube = Cube()
        self.cube.reset()
        self.cube.shuffle(number_of_moves=N_shuffle)
        self.move_number = 0
        self.reset_number += 1
        self.shuffle_number = N_shuffle
        return self.cube.get_array()
    
    def render(self, mode='human', close=False):
        self.cube.render()
        
    def is_done(self):
        return self.cube.isdone()

