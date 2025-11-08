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

from gym import Env
from gym.spaces import Dict, MultiDiscrete, MultiBinary, Discrete, Box

import matplotlib.pyplot as plt



#######################################################################################################
# DEFINE WAAM ENVIRONMENT CLASS (WITH MULTIPLE FUNCTIONS)
class WAAM(Env):
    def __init__(self):

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
                target_z_mm.append(4.0)
            else:
                target_z_mm.append(3.0)

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
        #print("PRED: ", pred[0])

        # Update current state
        env.state['z_mm'][env.state['c_tool_x_mm'] - 5:env.state['c_tool_x_mm'] + 6] = \
            env.state['z_mm'][env.state['c_tool_x_mm']-5:env.state['c_tool_x_mm']+6] + \
            pred[0]

        new_height_list = env.state['z_mm'][env.state['c_tool_x_mm']-5:env.state['c_tool_x_mm']+6]
        #print("New Height List: ", new_height_list)

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
    def reset(self):
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
                target_z_mm.append(4.0)
            else:
                target_z_mm.append(3.0)

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

#######################################################################################################
# SIMPLIFY NUMBER TO STRING FOR BEAD NUMBER
def get_bead_number(i):
    if i < 10:
        bead_number_string = "0" + str(i)
    else:
        bead_number_string = str(i)

    return bead_number_string

#######################################################################################################
# SAVE PLOTS OF COMPARISON
def save_plots(image_title, image_path, image_file_prepend, pred_heights, actual_heights):

    plot_file_name = image_file_prepend
    no_title_plot_file_name = image_file_prepend + '_NO_TITLE'

    PLOT_FILE_PATH = os.path.join(image_path, plot_file_name + '.eps')
    NO_TITLE_PLOT_FILE_PATH = os.path.join(image_path, no_title_plot_file_name + '.eps')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = "Times New Roman"

    f1 = plt.figure(figsize=(10,6))
    plt.title(image_title)
    plt.plot(pred_heights, label="Predicted Height", color='#156082')
    plt.plot(actual_heights, linestyle='dashed', label="Actual Height", color='#FFC000')
    plt.xlabel('Step')
    plt.ylabel('Height')
    plt.ylim(top=5)
    plt.legend()
    plt.show()
    f1.savefig(PLOT_FILE_PATH, bbox_inches='tight', format='eps')

    f2 = plt.figure(figsize=(10,6))
    plt.title('')
    plt.plot(pred_heights, label="Predicted Height", color='#156082')
    plt.plot(actual_heights, linestyle='dashed', label="Actual Height", color='#FFC000')
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

stride_lengths = [1, 2, 3]
rnd_seed = 74
torch.manual_seed(rnd_seed)
random.seed(rnd_seed)
np.random.seed(rnd_seed)

model_type = "cnn"

#lf = "MSE"
#lf = "L1"
lf = "SmoothL1"

torch.set_default_dtype(torch.float64)

data_set = "4"
logging.info('Data set: %s', data_set)

# Configure parameters for ML
learn_rate = 0.01  # learning rate
momentum = 0.9 # momentum
batch_size = 64  # batch size
epochs = 50  # how many epochs to train for

# Configure input and output directories
preprocess_dir = "/home/ML_WAAM_defects/01_preprocess/dataset_" + data_set + "/"
root_dir = "/home/ML_WAAM_defects/datasets/dataset_" + data_set + "/"
models_dir = "/home/ML_WAAM_defects/02_model/dataset_" + data_set + "/"
results_dir = "/home/ML_WAAM_defects/02_results/dataset_" + data_set + "/"
image_dir = "/home/ML_WAAM_defects/02_images/dataset_" + data_set + "/"
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
    env = WAAM()

    # Get number of actions from gym action space
    logging.info('Loading environment action space.')
    n_actions = env.action_space.n

    # Get the number of state observations
    state, observation_state = env.reset()

    # n_observations is 11 actual x_mm values plus one target x_mm value plus one current_tool_x_mm
    n_observations = len(observation_state)
    logging.info('N Observations (n_observations) is %s', n_observations)

    steps_done = 0
    episode_durations = []
    bead_nums = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 27, 28,
                 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

    ####################################################################################################
    # Run the line prediction
    for i in bead_nums:

        action = (i - 1)
        bead_num = get_bead_number(i)

        logging.info('Loading actual height data for bead %s', bead_num)
        timestep_file = preprocess_dir + "waam_torch_bead_timestep_ds_" + data_set + "_bead_" + \
                        bead_num + ".csv"
        timestep_df = pd.read_csv(timestep_file)

        timestep_first_ste = timestep_df['waam_steps_to_end'].iloc[0]
        logging.info('Initial waam_steps_to_end is %s', timestep_first_ste)

        while timestep_first_ste < 110:
            logging.info('Pre-pending row to allow sufficient steps to end.')
            timestep_x_gap = timestep_df['waam_steps_to_end'].iloc[1] - timestep_df['waam_steps_to_end'].iloc[0]
            timestep_x = timestep_df['waam_steps_to_end'].iloc[0] - timestep_x_gap
            waam_steps_from_start = timestep_df['waam_steps_from_start'].iloc[0] - 1
            waam_steps_to_end = timestep_df['waam_steps_to_end'].iloc[0] + 1
            distance_step = timestep_df['distance_step'].iloc[0] -1

            bead_row = [i, 0.0, 0, 0, 0, 0, waam_steps_from_start, waam_steps_to_end, distance_step]

            timestep_df.loc[-1] = bead_row
            timestep_df.index = timestep_df.index + 1  # shifting index
            timestep_df.sort_index(inplace=True)
            timestep_first_ste = timestep_df['waam_steps_to_end'].iloc[0]
            logging.info('Initial waam_steps_to_end is %s', timestep_first_ste)

        timestep_last_ste = timestep_df['waam_steps_to_end'].iloc[len(timestep_df) - 1]
        logging.info('Final waam_steps_to_end is %s', timestep_last_ste)

        while timestep_last_ste > -10:
            logging.info('Appending row to allow sufficient steps to end.')
            timestep_x_gap = (timestep_df['waam_steps_to_end'].iloc[1]) - (timestep_df['waam_steps_to_end'].iloc[0])
            timestep_x = (timestep_df['waam_steps_to_end'].iloc[len(timestep_df) - 1]) + timestep_x_gap
            waam_steps_from_start = (timestep_df['waam_steps_from_start'].iloc[len(timestep_df) - 1]) + 1
            waam_steps_to_end = (timestep_df['waam_steps_to_end'].iloc[len(timestep_df) - 1]) - 1
            distance_step = (timestep_df['distance_step'].iloc[len(timestep_df) - 1]) + 1

            bead_row = [i, 0.0, 0, 0, 0, 0, waam_steps_from_start, waam_steps_to_end, distance_step]

            timestep_df.loc[len(timestep_df)] = bead_row
            timestep_df.sort_index(inplace=True)
            timestep_last_ste = timestep_df['waam_steps_to_end'].iloc[len(timestep_df) -1]
            logging.info('Final waam_steps_to_end is %s', timestep_last_ste)
            logging.info('Length of timestep_df is %s', len(timestep_df))

        start_idx = timestep_df.index[timestep_df['waam_steps_to_end'] == 110].tolist()[0]
        end_idx = start_idx + 120

        actual_heights = []
        actual_heights = timestep_df['z_mm'].iloc[start_idx:end_idx].values.tolist()

        episode_cumulative_reward = []
        state, observation_state = env.reset()
        observation_state = torch.tensor(observation_state, dtype=torch.float64, device=device).unsqueeze(0)
        episode_action_set = []

        continue_processing = 1
        while continue_processing == 1:

            episode_action_set.append(action)

            observation, reward, terminated, truncated, _ = env.step(action, stride_length, model_type, scaler)

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float64, device=device).unsqueeze(0)

            # Move to the next state
            observation_state = next_state

            if done:
                pred_heights = env.state['z_mm']
                logging.info('Predicted heights (pred_heights) are %s', pred_heights)
                logging.info('Actual heights (actual_heights) are %s', actual_heights)

                image_title = "Predicted Height Comparison Bead " + bead_num + " Stride " + str(stride_length) + \
                              ", " + str(epochs) + " Epochs"
                image_file_prepend = "PHC_" + bead_num + "_STRIDE_" + str(stride_length) + "_EPOCH_" + str(epochs) + \
                                     "_LF_" + lf
                save_plots(image_title, image_dir, image_file_prepend, pred_heights, actual_heights)

                continue_processing = 0

