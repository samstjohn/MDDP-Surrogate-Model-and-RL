import logging
import pandas as pd
import os
import numpy as np
import torch
from math import floor
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import random
import dill
import scipy
from sklearn.preprocessing import *

#######################################################################################################
# SIMPLIFY NUMBER TO STRING FOR BEAD NUMBER
def get_bead_number(i):
    if i < 10:
        bead_number_string = "0" + str(i)
    else:
        bead_number_string = str(i)

    return bead_number_string

#######################################################################################################
# BUILDING TIME STEP DATAFRAME FUNCTION
def generate_time_step_data(bead_num, height_df, starting_x_pos, ending_x_pos, num_of_steps):

    # starting_row = height_df[height_df["z_mm"].gt(height_mean)].index[0]

    starting_row = height_df[height_df["x_mm"].gt(starting_x_pos)].index[0]
    starting_x_pos = height_df["x_mm"][starting_row]
    ending_x_pos = starting_x_pos + num_of_steps
    ending_row = height_df[height_df["x_mm"].gt(ending_x_pos)].index[0] - 1

    # Align the x-position with the start of the bead being zero
    height_df["x_mm"] = height_df["x_mm"] - starting_x_pos

    logging.info('Bead %s deposition starting row: %s', bead_num, starting_row)
    logging.info('Bead %s deposition starting x position: %s', bead_num, starting_x_pos)
    logging.info('Bead %s deposition ending x position: %s', bead_num, ending_x_pos)
    logging.info('Bead %s deposition ending x position (by position): %s', bead_num, height_df["x_mm"][ending_row])
    logging.info('Bead %s deposition ending row: %s', bead_num, ending_row)
    logging.info('Bead %s length in rows: %s', bead_num, ending_row - starting_row)

    step_length = int((ending_row - starting_row) / num_of_steps)
    logging.info('Step length: %s', step_length)

    new_df = height_df.iloc[starting_row % step_length::step_length, :].copy()
    new_df.reset_index(inplace=True)
    del new_df['index']

    new_starting_row = new_df[new_df["x_mm"].gt(0.0)].index[0]
    logging.info('Bead deposition starting row: %s', new_starting_row)

    new_df.loc[:, 'distance_step'] = range(1, len(new_df) + 1)
    new_df = new_df.astype({'distance_step': 'int'})

    new_df['waam_steps_from_start'] = new_df['distance_step'] - new_starting_row
    new_df['waam_steps_to_end'] = (new_starting_row + num_of_steps) - new_df['distance_step']

    new_df.loc[new_df["waam_steps_from_start"] >= 0, "wire_feed_speed"] = wfs
    new_df.loc[new_df["waam_steps_to_end"] < 0, "wire_feed_speed"] = 0

    new_df.loc[new_df["waam_steps_from_start"] >= 0, "travel_speed"] = travel_speed
    new_df.loc[new_df["waam_steps_to_end"] < 0, "travel_speed"] = 0

    new_df.loc[new_df["waam_steps_from_start"] >= 0, "arc_correction"] = arc_correction
    new_df.loc[new_df["waam_steps_to_end"] < 0, "arc_correction"] = 0

    logging.info('Height Profile \n %s', new_df)

    return(new_df)

#######################################################################################################
# MAIN LOGIC STARTS HERE
#######################################################################################################

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s:%(levelname)s: %(message)s')
logging.info('Initiating script.')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info('Device is %s', device)

stride_lengths = [1, 2, 3]
rnd_seeds = 74

# Set random seeds to ensure consistency
random.seed(rnd_seed)
np.random.seed(rnd_seed)

generate_start_end_flag = 0
generate_time_step_flag = 0
generate_stride_flag = 1
generate_delta_flag = 1
model_type = "cnn"
num_of_steps = 100

data_set = "4"
logging.info('Data set: %s', data_set)

# Configure parameters for ML
lr = 0.1  # learning rate
bs = 64  # batch size
epochs = 100  # how many epochs to train for

# Configure input and output directories
root_dir = "/home/ML_WAAM_defects/datasets/dataset_" + data_set + "/"
settings_file = root_dir + "settings.csv"
preprocess_dir = "/home/ML_WAAM_defects/01_preprocess/dataset_" + data_set + "/"
models_dir = "/home/ML_WAAM_defects/01_models/dataset_" + data_set + "/"
start_point_file = preprocess_dir + "start_end_points.csv"

if not os.path.exists(preprocess_dir):
    os.makedirs(preprocess_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Document the known bad beads for each dataset
bad_beads = []
if (data_set == '4'):
    bad_beads = [1, 7, 8, 15, 16, 17, 18, 19, 20, 34]
    logging.info('Bad beads: %s', bad_beads)


#######################################################################################################
# Generate a start and endpoint file detailing WAAM deposition beginning and end
if generate_start_end_flag == 1:
    settings = pd.read_csv(settings_file)
    logging.info('Settings \n--------------------------------------\n %s\n', settings)
    num_beads = len(settings)

    start_stop_rows = []

    for i in range(1, num_beads + 1):
        if (i not in bad_beads):
            logging.info('Processing for bead %s', i)
            hfilename = "height_profile_bead_"
            bead_height_profile = pd.read_csv(root_dir + 'height_profiles/' + hfilename + str(i) + '.csv')


            X = bead_height_profile["X(mm)"].tolist()
            Z = bead_height_profile["Z(mm)"].tolist()
            x_array = np.array(X)
            z_array = np.array(Z)
            naan_mask = ~np.isnan(z_array)

            height_profile = z_array[naan_mask]
            length_profile = x_array[naan_mask]

            peak = max(height_profile[0:floor((height_profile.size) / 2)])
            logging.info('Peak %s', str(peak))

            peakIdx = np.where(height_profile == peak)[0][0]
            logging.info('Peak Index %s', str(peakIdx))

            cutoff_flag = 0
            for j in range(95, 30, -1):

                if cutoff_flag == 0:

                    mean_perc = j/100
                    height_cutoff = np.mean(height_profile) * mean_perc

                    for m in range(1, height_profile.size, 1):
                        if height_profile[m] > height_cutoff:
                            startIdx = m
                            break
                        elif height_profile[m] < height_cutoff:
                            continue

                    for n in range(-1, -height_profile.size, -1):
                        if height_profile[n] > height_cutoff:
                            endIdx = n
                            break
                        elif height_profile[n] < height_cutoff:
                            continue

                    length_profile_difference = (length_profile[endIdx] - length_profile[startIdx])

                    if length_profile_difference > 100:
                        logging.debug('Bead %s: Percent of mean used is %s', i, mean_perc)

                        logging.debug('Bead %s: Start Index is %s', i, startIdx)
                        logging.debug('Bead %s: Start Height Profile is %s', i, height_profile[startIdx])
                        logging.debug('Bead %s: Start Length Profile is %s', i, length_profile[startIdx])

                        logging.debug('Bead %s: End Index is %s', i, endIdx)
                        logging.debug('Bead %s: End Height Profile is %s', i, height_profile[endIdx])
                        logging.debug('Bead %s: End Length Profile is %s', i, length_profile[endIdx])

                        cutoff_flag = 1
                        start_stop_row = [i, length_profile[startIdx], (length_profile[startIdx] + 100)]
                        start_stop_rows.append(start_stop_row)

    start_stop_df = pd.DataFrame(start_stop_rows)
    start_stop_df.columns = ['Bead', 'Start', 'End']
    start_stop_df.to_csv(start_point_file, index=False)

#######################################################################################################
# Generate a timestep file detailing height profile and WAAM settings for each bead
if generate_time_step_flag == 1:

    if generate_start_end_flag == 0:
        settings = pd.read_csv(settings_file)
        logging.info('Settings \n--------------------------------------\n %s\n', settings)

    start_point_df = pd.read_csv(start_point_file)
    start_point_df = start_point_df.rename(columns={'Bead': 'bead', 'Start': 'start', 'End': 'end'})
    logging.info('Start Points \n--------------------------------------\n %s\n', start_point_df)

    num_beads = len(settings)
    logging.info('Number of beads: %s', num_beads)

    df_start_indicator = 0

    for i in range(1, num_beads + 1):
        if (i not in bad_beads):
            bead_num = get_bead_number(i)
            timestep_file = preprocess_dir + "waam_torch_bead_timestep_ds_" + data_set + "_bead_" + bead_num + ".csv"

            logging.info('Bead number: %s', i)
            height_df = []

            wfs = settings['WFS'][i - 1]
            travel_speed = settings['Travel'][i - 1]
            arc_correction = settings['Arc Correction'][i - 1]

            logging.info("Wire Feed Speed for this bead is %s", wfs)

            bead_number_string = get_bead_number(i)
            logging.info('Acquiring height data for bead %s', bead_number_string)

            height_file = root_dir + 'height_profiles/height_profile_bead_' + str(i) + '.csv'
            logging.info('Height profile file: %s', height_file)

            csv_df = pd.read_csv(height_file)
            height_df = csv_df.rename(columns={'X(mm)': 'x_mm', 'Z(mm)': 'z_mm'})

            height_df = height_df.fillna(0)
            height_df[height_df < 0] = 0.0

            height_df.insert(loc=0, column='bead_number', value=i)

            height_df['wire_feed_speed'] = 0
            height_df['travel_speed'] = 0
            height_df['arc_correction'] = 0
            height_df['waam_steps_from_start'] = 0
            height_df['waam_steps_to_end'] = 0

            starting_x_pos = start_point_df.loc[start_point_df['bead'] == (i), 'start'].iloc[0]
            ending_x_pos = starting_x_pos + num_of_steps
            logging.info('Bead %s deposition starting x position: %s', i, starting_x_pos)
            logging.info('Bead %s deposition ending x position: %s', i, ending_x_pos)

            new_df = generate_time_step_data(i, height_df, starting_x_pos, ending_x_pos, num_of_steps)
            new_df.to_csv(timestep_file, index=False)


#######################################################################################################
# Generate a stride deposition file deposition point and initial height profile at deposition
if generate_stride_flag == 1:

    if (generate_start_end_flag + generate_time_step_flag) == 0:
        settings = pd.read_csv(settings_file)
        logging.info('Settings \n--------------------------------------\n %s\n', settings)

    start_point_df = pd.read_csv(start_point_file)
    start_point_df = start_point_df.rename(columns={'Bead': 'bead', 'Start': 'start', 'End': 'end'})
    logging.info('Start Points \n--------------------------------------\n %s\n', start_point_df)

    num_beads = len(settings)

    for stride_length in stride_lengths:
        logging.info('Processing for stride length %s', stride_length)
        stride_profile_file = preprocess_dir + "waam_torch_stride_ds_" + data_set + "_stride_" + str(stride_length) + ".csv"
        bead_rows = []

        for i in range(1, num_beads + 1):

            if (i not in bad_beads):
                bead_num = get_bead_number(i)
                logging.info('  Evaluating bead %s', bead_num)
                timestep_file = preprocess_dir + "waam_torch_bead_timestep_ds_" + data_set + "_bead_" + bead_num + ".csv"
                timestep_df = pd.read_csv(timestep_file)
                current_steps_to_end = 100
                previous_steps_to_end = current_steps_to_end

                # Gather z_mm data for the end bead profile (waam_steps_to_end -1 through -6)
                end_waam_steps = -1
                end_idx = timestep_df.index[timestep_df['waam_steps_to_end'] == end_waam_steps].tolist()[0]
                logging.debug('End Index (end_idx) is %s', end_idx)
                end_counter = 1
                end_z_mm_list = []
                while end_counter <= (5 + stride_length):
                    logging.debug('Current End Counter (end_counter) is %s', end_counter)
                    try:
                        end_z_mm = timestep_df.iloc[end_idx, timestep_df.columns.get_loc('z_mm')]
                        end_z_mm_list.append(end_z_mm)
                    except:
                        end_z_mm_list.append(0)
                    end_idx += 1
                    end_counter += 1
                logging.debug('Bead height end profile (end_z_mm_list) is %s', end_z_mm_list)

                # Get current deposition index and set current index to be 5 + stride length prior
                # c_idx indicates the first point that will be included in the array of features
                deposition_idx = timestep_df.index[timestep_df['waam_steps_to_end'] == current_steps_to_end].tolist()[0]
                c_idx = deposition_idx - ( 5 + stride_length )
                time_step = 0


                while ( time_step * stride_length ) < 100:

                    while ( c_idx - deposition_idx ) < 6:

                        # Gather feature information for current index
                        x_mm = timestep_df.iloc[c_idx, timestep_df.columns.get_loc('x_mm')]
                        wire_feed_speed = timestep_df.iloc[c_idx, timestep_df.columns.get_loc('wire_feed_speed')]
                        travel_speed = timestep_df.iloc[c_idx, timestep_df.columns.get_loc('travel_speed')]
                        arc_correction = timestep_df.iloc[c_idx, timestep_df.columns.get_loc('arc_correction')]
                        waam_steps_from_start = timestep_df.iloc[c_idx, timestep_df.columns.get_loc('waam_steps_from_start')]
                        waam_steps_to_end = timestep_df.iloc[c_idx, timestep_df.columns.get_loc('waam_steps_to_end')]
                        distance_step = timestep_df.iloc[c_idx, timestep_df.columns.get_loc('distance_step')]

                        # Set WAAM settings to 0 when not at the deposition point
                        if c_idx != deposition_idx:
                            wire_feed_speed = 0
                            travel_speed = 0
                            arc_correction = 0

                        ################################################################################################
                        # Define height profile for all timesteps

                        #   Step 0: Ste all height profile to zero
                        #   --------------------------------------------------------------------------------------------
                        if time_step == 0:
                            z_mm = 0

                        #   Forward Flow: Use the bead end profile
                        #   --------------------------------------------------------------------------------------------
                        #   For forward flow, the current index c_idx will be greater than or equal to the current
                        #   deposition point less the stride length minus one. In this scenario we use the bead end
                        #   profile to define height information.
                        elif c_idx >= (deposition_idx - (stride_length - 1)):
                            z_mm = end_z_mm_list[c_idx - (deposition_idx - (stride_length - 1))]

                        #   Backward Flow: Use the bead start profile
                        #   --------------------------------------------------------------------------------------------
                        #   For backward flow, the waam_steps_to_end will be greater than 100 minus the stride length.
                        #   In this scenario we use the bead start profile, obtained directly from the timestep_df,
                        #   to define height information.
                        elif waam_steps_to_end > (100 - stride_length):
                            backward_idx = timestep_df.index[timestep_df['waam_steps_to_end'] == waam_steps_to_end].tolist()[0]
                            z_mm = timestep_df.iloc[backward_idx, timestep_df.columns.get_loc('z_mm')]

                        #   Remaining: Use the height profile from the current index position
                        #   --------------------------------------------------------------------------------------------
                        else:
                            z_mm = timestep_df.iloc[c_idx, timestep_df.columns.get_loc('z_mm')]

                        #
                        # End of setting height profile
                        ################################################################################################


                        bead_row = [i, x_mm, z_mm, wire_feed_speed, travel_speed, arc_correction,
                                    waam_steps_from_start, waam_steps_to_end, distance_step, time_step]
                        logging.debug('Bead Row is %s', bead_row)
                        bead_rows.append(bead_row)

                        c_idx += 1

                    deposition_idx += stride_length
                    c_idx = deposition_idx - (5 + stride_length)
                    previous_steps_to_end = current_steps_to_end
                    current_steps_to_end = current_steps_to_end - stride_length
                    time_step += 1


        bead_df = pd.DataFrame(bead_rows)
        bead_df.columns = ['bead_num', 'x_mm', 'z_mm', 'wire_feed_speed', 'travel_speed',
                           'arc_correction', 'waam_steps_from_start', 'waam_steps_to_end',
                           'distance_step', 'time_step']
        bead_df.to_csv(stride_profile_file, index=False)

#######################################################################################################
# Generate a step deposition timestep file detailing height profile and WAAM settings for each bead
if generate_delta_flag == 1:

    if (generate_start_end_flag + generate_time_step_flag + generate_stride_flag) == 0:
        settings = pd.read_csv(settings_file)
        logging.info('Settings \n--------------------------------------\n %s\n', settings)

    for stride_length in stride_lengths:
        logging.info('Generating delta file for stride length %s', stride_length)
        stride_profile_file = preprocess_dir + "waam_torch_stride_ds_" + data_set + "_stride_" + str(stride_length) + ".csv"
        delta_file = preprocess_dir + "waam_torch_delta_ds_" + data_set + "_stride_" + str(stride_length) + ".csv"

        stride_profile = pd.read_csv(stride_profile_file)

        delta_rows = []
        num_beads = len(settings)

        for i in range(1, num_beads + 1):

            if (i not in bad_beads):
                bead_num = get_bead_number(i)

                bead_data = stride_profile.loc[stride_profile['bead_num'] == (i)]
                time_steps = bead_data['time_step'].unique()

                for time_step in time_steps:
                    next_time_step = time_step + 1
                    if next_time_step <= max(time_steps):

                        current_time_step_data = bead_data.loc[bead_data['time_step'] == (time_step)]
                        next_time_step_data = bead_data.loc[bead_data['time_step'] == (next_time_step)]

                        distance_steps = current_time_step_data['distance_step']
                        distance_steps = distance_steps[stride_length:]


                        for distance_step in distance_steps:
                            logging.debug('Distance Step (distance_step) is %s', distance_step)

                            bead_num = current_time_step_data.loc[current_time_step_data['distance_step']
                                                                  == distance_step, 'bead_num'].iloc[0]
                            x_mm = current_time_step_data.loc[current_time_step_data['distance_step']
                                                                  == distance_step, 'x_mm'].iloc[0]
                            z_mm = current_time_step_data.loc[current_time_step_data['distance_step']
                                                                  == distance_step, 'z_mm'].iloc[0]
                            wire_feed_speed = current_time_step_data.loc[current_time_step_data['distance_step']
                                                                  == distance_step, 'wire_feed_speed'].iloc[0]
                            travel_speed = current_time_step_data.loc[current_time_step_data['distance_step']
                                                                  == distance_step, 'travel_speed'].iloc[0]
                            arc_correction = current_time_step_data.loc[current_time_step_data['distance_step']
                                                                  == distance_step, 'arc_correction'].iloc[0]
                            waam_steps_from_start = current_time_step_data.loc[current_time_step_data['distance_step']
                                                                  == distance_step, 'waam_steps_from_start'].iloc[0]
                            waam_steps_to_end = current_time_step_data.loc[current_time_step_data['distance_step']
                                                                  == distance_step, 'waam_steps_to_end'].iloc[0]
                            time_step = current_time_step_data.loc[current_time_step_data['distance_step']
                                                                  == distance_step, 'time_step'].iloc[0]

                            next_z_mm = next_time_step_data.loc[next_time_step_data['distance_step']
                                                                  == distance_step, 'z_mm'].iloc[0]

                            delta_z = next_z_mm - z_mm

                            delta_row = [i, x_mm, z_mm, next_z_mm, delta_z, wire_feed_speed, travel_speed,
                                         arc_correction, waam_steps_from_start, waam_steps_to_end, distance_step,
                                         time_step]
                            logging.debug('Delta Row is %s', delta_row)
                            delta_rows.append(delta_row)

        delta_df = pd.DataFrame(delta_rows)
        delta_df.columns = ['bead_num', 'x_mm', 'z_mm', 'next_z_mm', 'delta_z', 'wire_feed_speed',
                            'travel_speed','arc_correction', 'waam_steps_from_start', 'waam_steps_to_end',
                           'distance_step', 'time_step']
        delta_df.to_csv(delta_file, index=False)