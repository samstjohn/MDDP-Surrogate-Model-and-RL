import logging
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
import pickle
from pickle import load
import random
import dill
import scipy
from sklearn.preprocessing import *
from itertools import count
from collections import namedtuple, deque
import math


from gym import Env
from gym.spaces import Dict, MultiDiscrete, MultiBinary, Discrete, Box

import matplotlib.pyplot as plt



#######################################################################################################
# DEFINE WAAM ENVIRONMENT CLASS (WITH MULTIPLE FUNCTIONS)
class WAAM(Env):
    def __init__(self, scenario):

        # Actions we can take; related to wire feed speed (WFS), arc length correction (Arc), robot velocity (Travel)
        # There are 46 discrete settings that can be applied in combination of the following parameters:
        # WFS: 100, 150, 200, 250
        # Arc: 6, 8, 10, 12
        # Travel: -30, -8, 15
        # Note: at WFS 250, Arc 6, only Travel -8 and 15 are invalid and excluded from the action set
        # These are encoded in a multi-binary space
        #self.action_space = MultiBinary([3, 3])
        self.action_space = Discrete(45)

        # Observation space; all height data for continuous bead is presented
        # x_mm: Horizontal distance from start of bead deposition
        # z_mm: Vertical height of bead
        # target_z_mm: Desired vertical height
        x_mm = []
        z_mm = []
        target_z_mm = []

        generator = (x * 1 for x in range(0, 120))
        for x in generator:
            x_mm.append(x)
            z_mm.append(0)

            if (x < 5) or (x > 100):
                target_z_mm.append(0.0)
            elif x < 50:
                if scenario == "stable_2.75":
                    target_z_mm.append(2.75)
                elif scenario == "2.75_to_3.25":
                    target_z_mm.append(2.75)
                elif scenario == "3.10_to_2.60":
                    target_z_mm.append(3.1)
            else:
                if scenario == "stable_2.75":
                    target_z_mm.append(2.75)
                elif scenario == "2.75_to_3.25":
                    target_z_mm.append(3.25)
                elif scenario == "3.10_to_2.60":
                    target_z_mm.append(2.6)

        self.x_mm = x_mm

        obs_spaces = Dict(
            {'x_mm': Discrete(len(x_mm)),
             'z_mm': Discrete(len(z_mm)),
             'target_z_mm': Discrete(len(target_z_mm)),
             'c_tool_x_mm': Discrete(1)
             })
        self.observation_space = Dict(obs_spaces)

        # Define initial state
        self.state = {'x_mm': x_mm,
                      'z_mm': z_mm,
                      'target_z_mm': target_z_mm,
                      'c_tool_x_mm': 10
                      }

        self.observation_space = {'z_mm': self.state['z_mm'][self.state['c_tool_x_mm']-5:self.state['c_tool_x_mm']+6],
                      'target_z_mm': self.state['target_z_mm'][self.state['c_tool_x_mm']],
                      'c_tool_x_mm': self.state['c_tool_x_mm']
                      }

        self.observation_state = []
        self.observation_state.extend(self.observation_space['z_mm'])
        self.observation_state.extend([self.observation_space['target_z_mm']])
        self.observation_state.extend([self.observation_space['c_tool_x_mm']])

        self.remaining_steps = len(x_mm) - 20

    ###################################################################################################
    # STEP FUNCTION - UPDATE CURRENT STATE BASED ON ACTION AND CALCULATE REWARD
    def step(self, action, stride_length, model_type, scaler):

        current_state = {}

        # Use surrogate model to estimate change to be applied
        wfs, travel_speed, arc_correction = get_action_values(action)

        height_list = env.observation_state[:11]

        if model_type == "cnn":
            wire_feed_speed_list = [0, 0, 0, 0, 0, wfs, 0, 0, 0, 0, 0]
            travel_speed_list = [0, 0, 0, 0, 0, travel_speed, 0, 0, 0, 0, 0]
            arc_correction_list = [0, 0, 0, 0, 0, arc_correction, 0, 0, 0, 0, 0]
        elif model_type == "ffnn":
            wire_feed_speed_list = [wfs]
            travel_speed_list = [travel_speed]
            arc_correction_list = [arc_correction]

        x_simulated = [height_list + wire_feed_speed_list + travel_speed_list + arc_correction_list]
        x_simulated_normalized = scaler.transform(x_simulated)

        ts_data = np.array(x_simulated_normalized, dtype=np.float64)
        ts_data.astype(float)

        if model_type == "cnn":
            ts_tensor = torch.from_numpy(ts_data[0]).view(-1, 1, 4, 11)
        elif model_type == "ffnn":
            ts_tensor = torch.from_numpy(ts_data)

        pred = model(ts_tensor)
        pred = pred.detach().cpu().numpy()
        logging.debug('PRED: %s', pred[0])

        # Update current state
        env.state['z_mm'][env.state['c_tool_x_mm'] - 5:env.state['c_tool_x_mm'] + 6] = \
            env.state['z_mm'][env.state['c_tool_x_mm']-5:env.state['c_tool_x_mm']+6] + \
            pred[0]

        #new_height_list = env.state['z_mm'][env.state['c_tool_x_mm']-5:env.state['c_tool_x_mm']+6]
        #logging.debug('New Height List: %s', new_height_list)

        # Use a placeholder for info
        info = {}

        # The model provides 11 points of height profile
        # No reward should be given for the first or second deposition
        # Ongoing reward should be based on if the first 5 points total height is within a range from target
        # Target is defined as a variable going into the RL
        # If the first five of those points are all +/- 0.05 from target, then reward is incremented by 25
        # If the first five of those points are all +/- 0.10 target, then reward is incremented by 10
        # If the first five of those points have any over +/- 0.20 target, then reward is decremented by 10
        # MAYBE: Any deposition point causes reward to be decremented by 1
        # MAYBE: Any change in WAAM settings causes reward to be decremented by 5

        reward = 0
        avg_diff_target_x = 0

        if stride_length == 1:
            # for stride = 1 scenario, just look at the point of deposition
            eval_x_mm = env.state['z_mm'][env.state['c_tool_x_mm']]
            eval_target_x_mm = env.state['target_z_mm'][env.state['c_tool_x_mm']]

            diff_target_x_mm = abs(np.array(eval_x_mm) - np.array(eval_target_x_mm))
            avg_diff_target_x = diff_target_x_mm

            if np.all(diff_target_x_mm < 0.05):
                reward += 1
            elif np.all(diff_target_x_mm < 0.1):
                reward += 0.25
            elif np.all(diff_target_x_mm > 1.0):
                reward += -2
            elif np.all(diff_target_x_mm > 0.1):
                reward += -0.25


            #if np.all(diff_target_x_mm < 0.05):
            #    reward += 2
            #elif np.all(diff_target_x_mm < 0.1):
            #    reward += 1
            #elif np.all(diff_target_x_mm < 0.5):
            #    reward += 0.5
            #elif np.all(diff_target_x_mm > 1.0):
            #    reward += -2
            #elif np.all(diff_target_x_mm > 0.5):
            #    reward += -0.5

        elif stride_length == 2:

            eval_x_m1_mm = env.state['z_mm'][env.state['c_tool_x_mm'] - 1]
            eval_target_x_m1_mm = env.state['target_z_mm'][env.state['c_tool_x_mm'] - 1]

            eval_x_0_mm = env.state['z_mm'][env.state['c_tool_x_mm']]
            eval_target_x_0_mm = env.state['target_z_mm'][env.state['c_tool_x_mm']]

            diff_target_x_0_mm = abs(np.array(eval_x_0_mm) - np.array(eval_target_x_0_mm))

            if env.state['c_tool_x_mm'] >= 11:
                diff_target_x_m1_mm = abs(np.array(eval_x_m1_mm) - np.array(eval_target_x_m1_mm))
                avg_diff_target_x = (diff_target_x_m1_mm + diff_target_x_0_mm) / 2
            else:
                diff_target_x_m1_mm = 0
                avg_diff_target_x = diff_target_x_0_mm


            if np.all(diff_target_x_m1_mm < 0.05):
                reward += 1
            elif np.all(diff_target_x_m1_mm < 0.1):
                reward += 0.5
            elif np.all(diff_target_x_m1_mm < 0.5):
                reward += 0.25
            elif np.all(diff_target_x_m1_mm > 1.0):
                reward += -1
            elif np.all(diff_target_x_m1_mm > 0.5):
                reward += -0.25

            if np.all(diff_target_x_0_mm < 0.05):
                reward += 1
            elif np.all(diff_target_x_0_mm < 0.1):
                reward += 0.5
            elif np.all(diff_target_x_0_mm < 0.5):
                reward += 0.25
            elif np.all(diff_target_x_0_mm > 1.0):
                reward += -1
            elif np.all(diff_target_x_0_mm > 0.5):
                reward += -0.25


        # Update current location
        env.state['c_tool_x_mm'] += stride_length

        # Reduce remaining steps by one
        self.remaining_steps -= stride_length

        #print("REMAINING STEPS: ", self.remaining_steps)
        # Apply action; I think here is where I would apply the model to estimate next state
        # Essentially feeding in current parameters + updated settings
        # and getting the anticipated resultant size

        if self.remaining_steps <= 0 or avg_diff_target_x > 10:
            done = True
        else:
            done = False

        self.observation_space = {'z_mm': self.state['z_mm'][self.state['c_tool_x_mm']-5:self.state['c_tool_x_mm']+6],
                      'target_z_mm': self.state['target_z_mm'][self.state['c_tool_x_mm']],
                      'c_tool_x_mm': self.state['c_tool_x_mm']
                      }

        self.observation_state = []
        self.observation_state.extend(self.observation_space['z_mm'])
        self.observation_state.extend([self.observation_space['target_z_mm']])
        self.observation_state.extend([self.observation_space['c_tool_x_mm']])

        return self.observation_state, reward, done, False, info

    ###################################################################################################
    # RENDER FUNCTION - DISPLAY PROGRESS
    def render(self):
        pass

    ###################################################################################################
    # RESET FUNCTION - BRING STATE BACK TO ORIGINAL CONFIGURATION
    def reset(self, scenario):
        # Restoring to initial setup
        x_mm = []
        z_mm = []
        target_z_mm = []

        generator = (x * 1 for x in range(0, 120))
        for x in generator:
            x_mm.append(x)
            z_mm.append(0)

            if (x < 5) or (x > 100):
                target_z_mm.append(0.0)
            elif x < 50:
                if scenario == "stable_2.75":
                    target_z_mm.append(2.75)
                elif scenario == "2.75_to_3.25":
                    target_z_mm.append(2.75)
                elif scenario == "3.10_to_2.60":
                    target_z_mm.append(3.1)
            else:
                if scenario == "stable_2.75":
                    target_z_mm.append(2.75)
                elif scenario == "2.75_to_3.25":
                    target_z_mm.append(3.25)
                elif scenario == "3.10_to_2.60":
                    target_z_mm.append(2.6)

        self.x_mm = x_mm

        self.state = {'x_mm': x_mm,
                      'z_mm': z_mm,
                      'target_z_mm': target_z_mm,
                      'c_tool_x_mm': 10
                      }

        self.observation_space = {'z_mm': self.state['z_mm'][self.state['c_tool_x_mm']-5:self.state['c_tool_x_mm']+6],
                      'target_z_mm': self.state['target_z_mm'][self.state['c_tool_x_mm']],
                      'c_tool_x_mm': self.state['c_tool_x_mm']
                      }


        self.observation_state = []
        self.observation_state.extend(self.observation_space['z_mm'])
        self.observation_state.extend([self.observation_space['target_z_mm']])
        self.observation_state.extend([self.observation_space['c_tool_x_mm']])

        self.remaining_steps = len(x_mm) - 20

        return self.state, self.observation_state

#######################################################################################################
# GET_RANDOM_ACTION SELECTS A RANDOM ACTION THAT ALIGNS WITH EXPERIMENT CONFIGURATION
def get_random_action(step_counter):

    current_act = env.action_space.sample()

    if env.remaining_steps > 115:
        current_act = 45

    return(current_act)



####################################################################################################
# REPLAY MEMORY
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

####################################################################################################
# Q-NETWORK

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization, returns tensor
    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

####################################################################################################
# ACTION SELECTOR
def select_action(observation_state, eps_threshold):
    global steps_done
    sample = random.random()

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(observation_state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


#######################################################################################################
# GET_RANDOM_ACTION SELECTS A RANDOM ACTION THAT ALIGNS WITH EXPERIMENT CONFIGURATION
def get_random_action(step_counter):

    current_act = env.action_space.sample()

    if env.remaining_steps > 110:
        current_act = 45

    return(current_act)

#######################################################################################################
# GET_ACTION_VALUES RETURNS THE SETTINGS DERIVED FROM MULTI-BINARY
def get_action_values(action_code):

    if action_code == 0:
        wfs = 100
        travel = 6
        arc_corr = 0
    elif action_code == 1:
        wfs = 100
        travel = 8
        arc_corr = 0
    elif action_code == 2:
        wfs = 100
        travel = 10
        arc_corr = 0
    elif action_code == 3:
        wfs = 100
        travel = 12
        arc_corr = 0
    elif action_code == 4:
        wfs = 150
        travel = 6
        arc_corr = 0
    elif action_code == 5:
        wfs = 150
        travel = 8
        arc_corr = 0
    elif action_code == 6:
        wfs = 150
        travel = 10
        arc_corr = 0
    elif action_code == 7:
        wfs = 150
        travel = 12
        arc_corr = 0
    elif action_code == 8:
        wfs = 200
        travel = 6
        arc_corr = 0
    elif action_code == 9:
        wfs = 200
        travel = 8
        arc_corr = 0
    elif action_code == 10:
        wfs = 200
        travel = 10
        arc_corr = 0
    elif action_code == 11:
        wfs = 200
        travel = 12
        arc_corr = 0
    elif action_code == 12:
        wfs = 250
        travel = 6
        arc_corr = 0
    elif action_code == 13:
        wfs = 250
        travel = 8
        arc_corr = 0
    elif action_code == 14:
        wfs = 250
        travel = 10
        arc_corr = 0
    elif action_code == 15:
        wfs = 250
        travel = 12
        arc_corr = 0
    elif action_code == 16:
        wfs = 100
        travel = 6
        arc_corr = -15
    elif action_code == 17:
        wfs = 100
        travel = 8
        arc_corr = -15
    elif action_code == 18:
        wfs = 100
        travel = 10
        arc_corr = -15
    elif action_code == 19:
        wfs = 100
        travel = 12
        arc_corr = -15
    elif action_code == 20:
        wfs = 150
        travel = 6
        arc_corr = -15
    elif action_code == 21:
        wfs = 150
        travel = 8
        arc_corr = -15
    elif action_code == 22:
        wfs = 150
        travel = 10
        arc_corr = -15
    elif action_code == 23:
        wfs = 150
        travel = 12
        arc_corr = -15
    elif action_code == 24:
        wfs = 200
        travel = 6
        arc_corr = -15
    elif action_code == 25:
        wfs = 200
        travel = 8
        arc_corr = -15
    elif action_code == 26:
        wfs = 200
        travel = 10
        arc_corr = -15
    elif action_code == 27:
        wfs = 200
        travel = 12
        arc_corr = -15
    elif action_code == 28:
        wfs = 250
        travel = 8
        arc_corr = -15
    elif action_code == 29:
        wfs = 250
        travel = 10
        arc_corr = -15
    elif action_code == 30:
        wfs = 250
        travel = 12
        arc_corr = -15
    elif action_code == 31:
        wfs = 100
        travel = 6
        arc_corr = 15
    elif action_code == 32:
        wfs = 100
        travel = 8
        arc_corr = 15
    elif action_code == 33:
        wfs = 100
        travel = 10
        arc_corr = 15
    elif action_code == 34:
        wfs = 100
        travel = 12
        arc_corr = 15
    elif action_code == 35:
        wfs = 150
        travel = 6
        arc_corr = 15
    elif action_code == 36:
        wfs = 150
        travel = 8
        arc_corr = 15
    elif action_code == 37:
        wfs = 150
        travel = 10
        arc_corr = 15
    elif action_code == 38:
        wfs = 150
        travel = 12
        arc_corr = 15
    elif action_code == 39:
        wfs = 200
        travel = 8
        arc_corr = 15
    elif action_code == 40:
        wfs = 200
        travel = 10
        arc_corr = 15
    elif action_code == 41:
        wfs = 200
        travel = 12
        arc_corr = 15
    elif action_code == 42:
        wfs = 250
        travel = 8
        arc_corr = 15
    elif action_code == 43:
        wfs = 250
        travel = 10
        arc_corr = 15
    else:
        wfs = 250
        travel = 12
        arc_corr = 15

    return wfs, travel, arc_corr

#######################################################################################################
# RETURN THE ML MODEL AND OPTIMIZER
def get_model(model_type, learn_rate):

    if model_type == 'cnn':

        class Lambda(nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func
            def forward(self, x):
                return self.func(x)

        model = nn.Sequential(
            nn.Conv2d(1, 11, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(11, 11, kernel_size=3, stride=3, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            Lambda(lambda x: x.view(x.size(0), -1)),
        )

        opt = optim.Adam(model.parameters(), lr=learn_rate)

    elif model_type == 'ffnn':

        model = nn.Sequential(
            nn.Linear(14, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 11),
        )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

        opt = optim.Adam(model.parameters(), lr=learning_rate)

    return model, opt


####################################################################################################
# IMPROVE MODEL
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

#######################################################################################################
# SIMPLIFY NUMBER TO STRING FOR BEAD NUMBER
def get_bead_number(i):
    if i < 10:
        bead_number_string = "0" + str(i)
    else:
        bead_number_string = str(i)

    return bead_number_string

####################################################################################################
# PLOT RESULTS
def save_reward_plots(image_title, image_path, image_file_prepend, cumulative_means):

    plot_file_name = image_file_prepend
    no_title_plot_file_name = image_file_prepend + '_NO_TITLE'

    PLOT_FILE_PATH = os.path.join(image_path, plot_file_name + '.eps')
    NO_TITLE_PLOT_FILE_PATH = os.path.join(image_path, no_title_plot_file_name + '.eps')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = "Times New Roman"

    f1 = plt.figure(figsize=(10,6))
    plt.title(image_title)
    plt.plot(cumulative_means, label="100 Epoch Mean Episode Reward", color='#156082')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    f1.savefig(PLOT_FILE_PATH, bbox_inches='tight', format='eps')

    f2 = plt.figure(figsize=(10,6))
    plt.title('')
    plt.plot(cumulative_means, label="100 Epoch Mean Episode Reward", color='#156082')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    f2.savefig(NO_TITLE_PLOT_FILE_PATH, bbox_inches='tight', format='eps')

#######################################################################################################
# SAVE PLOTS OF COMPARISON
def save_profile_plots(image_title, image_path, image_file_prepend, target_heights, actual_heights):

    plot_file_name = image_file_prepend
    no_title_plot_file_name = image_file_prepend + '_NO_TITLE'

    PLOT_FILE_PATH = os.path.join(image_path, plot_file_name + '.eps')
    NO_TITLE_PLOT_FILE_PATH = os.path.join(image_path, no_title_plot_file_name + '.eps')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = "Times New Roman"

    f1 = plt.figure(figsize=(10,6))
    plt.title(image_title)
    plt.plot(actual_heights, label="Actual Height", color='#FFC000')
    plt.plot(target_heights, linestyle='dashed', label="Target Height", color='#156082')
    plt.xlabel('Step')
    plt.ylabel('Height')
    plt.ylim(top=5)
    plt.legend()
    plt.show()
    f1.savefig(PLOT_FILE_PATH, bbox_inches='tight', format='eps')

    f2 = plt.figure(figsize=(10,6))
    plt.title('')
    plt.plot(actual_heights, label="Actual Height", color='#FFC000')
    plt.plot(target_heights, linestyle='dashed', label="Target Height", color='#156082')
    plt.xlabel('Step')
    plt.ylabel('Height')
    plt.ylim(top=5)
    plt.legend()
    plt.show()
    f2.savefig(NO_TITLE_PLOT_FILE_PATH, bbox_inches='tight', format='eps')


#######################################################################################################
# MAIN LOGIC STARTS HERE
#######################################################################################################
logging.basicConfig(level = logging.INFO, format = '%(asctime)s:%(levelname)s: %(message)s')
logging.info('Initiating script.')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info('Device is %s', device)

stride_lengths = [1, 2]
rnd_seed = 74
torch.manual_seed(rnd_seed)
random.seed(rnd_seed)
np.random.seed(rnd_seed)

model_type = "cnn"

#lf = "MSE"
lf = "L1"
#lf = "SmoothL1"

#scenario = "stable_2.75"
#scenario = "2.75_to_3.25"
scenario = "3.10_to_2.60"

torch.set_default_dtype(torch.float64)

data_set = "4"
logging.info('Data set: %s', data_set)

# Configure parameters for ML
learn_rate = 0.01  # learning rate
momentum = 0.9 # momentum
batch_size = 64  # batch size
epochs = 100  # how many epochs to train for

# Configure parameters for RL
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the 'AdamW' optimizer
#BATCH_SIZE = 128
BATCH_SIZE = 512
#GAMMA = 0.99
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.02
#EPS_DECAY = 20000
TAU = 0.005
LR = 1e-4
REPLAY_MEMORY_SIZE = 10000

# Configure input and output directories
preprocess_dir = "/home/ML_WAAM_defects/01_preprocess/dataset_" + data_set + "/"
root_dir = "/home/ML_WAAM_defects/datasets/dataset_" + data_set + "/"
models_dir = "/home/ML_WAAM_defects/02_model/dataset_" + data_set + "/"
results_dir = "/home/ML_WAAM_defects/04_results/dataset_" + data_set + "/"
image_dir = "/home/ML_WAAM_defects/04_images/dataset_" + data_set + "/"
settings_file = root_dir + "settings.csv"

if not os.path.exists(preprocess_dir):
    os.makedirs(preprocess_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

for stride_length in stride_lengths:
    logging.info('Processing for stride length %s', stride_length)

    scaler_file = models_dir + "waam_scaler_" + data_set + "_stride_" + str(stride_length) + "_epochs_" + \
                  str(epochs) + "_" + model_type + ".pkl"
    scaler = load(open(scaler_file, 'rb'))

    model_state_dict = models_dir + "waam_" + data_set + "_stride_" + str(stride_length) + "_epochs_" + \
                       str(epochs) + "_" + model_type + "_seed_" + str(rnd_seed) + "_loss_" + lf + \
                       "_state_dict.pt"

    torch.set_default_dtype(torch.float64)
    model, opt = get_model(model_type, learn_rate)

    logging.info('Loading state dictionary for model.')
    model.load_state_dict(torch.load(model_state_dict))

    logging.info('Setting up WAAM RL environment.')
    env = WAAM(scenario)

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Retrieve the target height profile to plot later
    target_z_mm = env.state['target_z_mm']

    # Get the number of state observations
    state, observation_state = env.reset(scenario)

    # n_observations is 11 actual x_mm values plus one target x_mm value plus one current_tool_x_mm
    n_observations = len(observation_state)
    logging.info('n_observations: %s', str(n_observations))

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    steps_done = 0
    episode_durations = []

    logging.info('Observation Space: %s', str(env.observation_space))

    ####################################################################################################
    # Run the RL

    if torch.cuda.is_available():
        num_episodes = 1000
    else:
        num_episodes = 50

    # Set EPS_DECAY based on number of episodes and stride length
    # This is set to roughly align epsilon values across episodes for stride lengths of 1 and 2
    EPS_DECAY = num_episodes * 10 / max(1, (stride_length/1.5))

    episode_cumulative_reward = []
    max_cumulative_reward = -1000.00

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, observation_state = env.reset(scenario)
        observation_state = torch.tensor(observation_state, dtype=torch.float64, device=device).unsqueeze(0)
        cumulative_reward = 0.0
        episode_action_set = []

        for t in count():
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * steps_done / EPS_DECAY)
            action = select_action(observation_state, eps_threshold)
            episode_action_set.append(action.item())

            observation, reward, terminated, truncated, _ = env.step(action.item(), stride_length, model_type, scaler)
            cumulative_reward += reward

            reward = torch.tensor([reward], device=device)

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float64, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(observation_state, action, next_state, reward)

            # Move to the next state
            observation_state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)

            target_net.load_state_dict(target_net_state_dict)

            if done:
                logging.info('Episode %s | Epsilon %s | Duration %s Steps | Reward %s',
                             i_episode, round(eps_threshold, 5), t, cumulative_reward)
                episode_cumulative_reward.append(cumulative_reward)

                if cumulative_reward > max_cumulative_reward:
                    max_cumulative_reward = cumulative_reward
                    max_action_set = episode_action_set
                    max_z_mm = env.state['z_mm']

                    logging.info('Max Cumulative Reward Achieved: %s', max_cumulative_reward)
                    logging.debug('Max Action Set: %s', max_action_set)
                    logging.debug('Action Set Size: %s', len(max_action_set))
                    logging.debug('Max z_mm: %s', max_z_mm)

                # plot_durations()
                break

    logging.info('Complete for stride_length of %s', stride_length)

    print()
    logging.info('Max Cumulative Reward: %s', max_cumulative_reward)
    logging.info('Max Action Set: %s', max_action_set)
    logging.info("'Action Set Size: %s", len(max_action_set))

    # Shift max_z_mm height to adjust for initial pre-pend of 5 zeros
    max_z_mm = max_z_mm[5:] + ([0] * 5)
    logging.info('Max z_mm: %s   ', max_z_mm)
    logging.info('Target z_mm: %s', target_z_mm)

    cumulative_rewards_t = torch.tensor(episode_cumulative_reward, dtype=torch.float)
    cumulative_means = cumulative_rewards_t.unfold(0, 100, 1).mean(1).view(-1)
    first_mean = cumulative_means[0].item()
    first_mean_list = [first_mean] * 99
    cumulative_means = torch.cat((torch.tensor(first_mean_list), cumulative_means))

    reward_image_title = "DQN RL 100 Epoch Mean Reward: Stride Length " + str(stride_length)
    reward_image_file_prepend = "DQNRL_Reward__STRIDE_" + str(stride_length) + "_EPOCH_" + str(epochs) + "_LOSS_" \
                                + lf + "_" + scenario

    profile_image_title = "DQL RL Height Alignment: Stride Length " + str(stride_length)
    profile_image_file_prepend = "DQNRL_Height_Alignment__STRIDE_" + str(stride_length) + "_EPOCH_" + str(epochs) + \
                                 "_LOSS_" + lf + "_" + scenario

    save_reward_plots(reward_image_title, image_dir, reward_image_file_prepend, cumulative_means)

    save_profile_plots(profile_image_title, image_dir, profile_image_file_prepend, target_z_mm, max_z_mm)
