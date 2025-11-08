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
from tqdm import tqdm
import pickle
import random
import dill
import scipy
from sklearn.preprocessing import *


#######################################################################################################
# USE A WRAPPED DATA LOADER TO INCLUDE FUNCTION TO SIZE DATA CORRECTLY
class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))



#######################################################################################################
# Generate the X / y data set
def generate_Xy_data_set(delta_file_df):

    bead_numbers = delta_file_df['bead_num'].unique()
    test_bead_counter = 0
    time_step_counter = 0

    for bead_number in bead_numbers:
        logging.debug('Processing generate_Xy_data_set for bead number %s', bead_number)
        bead_df = delta_file_df.loc[delta_file_df['bead_num'] == bead_number]
        time_steps = bead_df['time_step'].unique()

        for time_step in time_steps:
            logging.debug('Processing generate_Xy_data_set for time step %s', time_step)
            bead_ts_df = bead_df.loc[bead_df['time_step'] == time_step]

            x_list = []
            y_list = []

            height = (bead_ts_df[['z_mm']]).values.tolist()
            for h in height:
                x_list.append(h[0])

            wire_feed_speed = (bead_ts_df[['wire_feed_speed']]).to_numpy()
            for wfspd in wire_feed_speed:
                x_list.append(wfspd[0])

            travel_speed = (bead_ts_df[['travel_speed']]).to_numpy()
            for tspd in travel_speed:
                x_list.append(tspd[0])

            arc_correction = (bead_ts_df[['arc_correction']]).to_numpy()
            for acor in arc_correction:
                x_list.append(acor[0])

            delta_height = (bead_ts_df[['delta_z']]).to_numpy()
            for dh in delta_height:
                y_list.append(dh[0])

            ts_data = np.array(x_list)
            ts_target = np.array(y_list)

            if time_step_counter == 0:
                ts_X = ts_data
                ts_y = ts_target
                time_step_counter += 1
            else:
                ts_X = np.vstack([ts_X, ts_data])
                ts_y = np.vstack([ts_y, ts_target])

            logging.debug('ts_X shape is %s', ts_X.shape)
            logging.debug('ts_y shape is %s', ts_y.shape)

        if (test_bead_counter == 0):
            X = ts_X
            y = ts_y
        else:
            X = np.vstack([X, ts_X])
            y = np.vstack([y, ts_y])

            test_bead_counter += 1

        logging.debug('X shape is %s', X.shape)
        logging.debug('y shape is %s', y.shape)

        x_list.append(h[0])

    # At this point X is a numpy array with the following:
    # Shape: (Deposition Examples, Features)
    # Each row has 44 features
    # - 1-11: original height
    # - 12-22: wire feed speed settings (current deposition)
    # - 23-33: travel speed settings (current deposition)
    # - 34-44: arc correction settings (current deposition)
    #
    # At this point y is a numpy array with the following:
    # Shape: (Deposition Examples, Features)
    # Each row has 11 features
    # - 1-11: increase in height

    X = X.astype(float)
    y = y.astype(float)

    return X, y


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
# CONVERT THE TRAIN AND VALID DATA INTO DATALOADER FOR BATCHING
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

#######################################################################################################
# FORMAT THE X TENSOR TO THE CORRECT SIZE FOR CNN
def preprocess(x, y):
    return x.view(-1, 1, 4, 11), y

#######################################################################################################
# DEFINE LOSS BATCH
def loss_batch(model, loss_func, xb, yb, opt=None):

    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

#######################################################################################################
# FIT THE MODEL
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):

    for epoch in range(epochs):
        with tqdm(train_dl, unit="batch") as tepoch:

            for xb, yb in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                train_loss, train_num = loss_batch(model, loss_func, xb.float(), yb.float(), opt)
                model.eval()

                with torch.no_grad():
                    test_losses, nums = zip(*[loss_batch(model, loss_func, xb.float(), yb.float()) for xb, yb in valid_dl])

                tepoch.set_postfix(loss=train_loss)

            test_loss = np.sum(np.multiply(test_losses, nums)) / np.sum(nums)

    return test_loss

#######################################################################################################
# FIND MEAN AND CONFIDENCE INTERVAL BASED ON LIST
def mean_confidence_interval(data, confidence):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h



#######################################################################################################
# MAIN LOGIC STARTS HERE
#######################################################################################################
logging.basicConfig(level = logging.INFO, format = '%(asctime)s:%(levelname)s: %(message)s')
logging.info('Initiating script.')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info('Device is %s', device)

stride_lengths = [1, 2, 3]
rnd_seeds = [74, 10, 41, 50, 91, 41, 61, 8, 77, 25]

model_type = "cnn"

# Define loss function
#loss_func = nn.MSELoss()
#lf = "MSE"
#loss_func = nn.L1Loss()
#lf = "L1"
#loss_func = nn.SmoothL1Loss()
#lf = "SmoothL1"

data_set = "4"
logging.info('Data set: %s', data_set)

# Configure parameters for ML
learn_rate = 0.01  # learning rate
momentum = 0.9 # momentum
batch_size = 64  # batch size
epochs = 100  # how many epochs to train for

# Configure input and output directories
preprocess_dir = "/home/ML_WAAM_defects/01_preprocess/dataset_" + data_set + "/"
root_dir = "/home/ML_WAAM_defects/datasets/dataset_" + data_set + "/"
models_dir = "/home/ML_WAAM_defects/02_model/dataset_" + data_set + "/"
results_dir = "/home/ML_WAAM_defects/02_results/dataset_" + data_set + "/"
settings_file = root_dir + "settings.csv"

if not os.path.exists(preprocess_dir):
    os.makedirs(preprocess_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Document the known bad beads for each dataset
bad_beads = []
if (data_set == '4'):
    bad_beads = [1, 7, 8, 15, 16, 17, 18, 19, 20, 34]
    logging.info('Bad beads: %s', bad_beads)

for stride_length in stride_lengths:
    logging.info('Processing for stride length %s', stride_length)
    delta_file = preprocess_dir + "waam_torch_delta_ds_" + data_set + "_stride_" + str(stride_length) + ".csv"

    scaler_file = models_dir + "waam_scaler_" + data_set + "_stride_" + str(stride_length) + "_epochs_" + \
                  str(epochs) + "_" + model_type + ".pkl"
    results_file = results_dir + "waam_" + data_set + "_stride_" + str(stride_length) + "_epochs_" + \
                   str(epochs) + "_loss_" + lf + "_" + model_type + "_results.csv"


    delta_file_df = pd.read_csv(delta_file)

    # Generate the X and y data for processing
    logging.info('Generating Xy data set')
    X, y = generate_Xy_data_set(delta_file_df)
    logging.debug('X shape is %s', X.shape)
    logging.debug('y shape is %s', y.shape)

    # Create the train and validation datasets
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    pickle.dump(scaler, open(scaler_file, 'wb'))


    overall_max_diff_list = []
    execution_num = 1

    for rnd_seed in rnd_seeds:

        logging.info('Using random seed %s', rnd_seed)

        # Set random seeds to ensure consistency
        torch.manual_seed(rnd_seed)
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)

        model_state_dict = models_dir + "waam_" + data_set + "_stride_" + str(stride_length) + "_epochs_" + \
                           str(epochs) + "_" + model_type + "_seed_" + str(rnd_seed) + "_loss_" + lf + \
                           "_state_dict.pt"
        model_complete =   models_dir + "waam_" + data_set + "_stride_" + str(stride_length) + "_epochs_" + \
                           str(epochs) + "_" + model_type + "_seed_" + str(rnd_seed) + "_loss_" + lf + \
                           "_model.pt"

        # Split data into train / test sets, using Pytorch nomenclature (train / valid)
        Xtr, Xva, Ytr, Yva = train_test_split(X_scaled, y, train_size=0.7, shuffle=True)
        x_train, y_train, x_valid, y_valid = map(torch.tensor, (Xtr, Ytr, Xva, Yva))
        logging.debug('x_train data type is %s', x_train.dtype)
        logging.debug('y_train data type is %s', y_train.dtype)
        logging.debug('x_valid data type is %s', x_valid.dtype)
        logging.debug('y_valid data type is %s', y_valid.dtype)

        logging.debug('x_train shape is %s', x_train.shape)
        logging.debug('y_train shape is %s', y_train.shape)
        logging.debug('x_valid shape is %s', x_valid.shape)
        logging.debug('y_valid shape is %s', y_valid.shape)

        # Create data loader for the train and validation datasets
        train_ds = TensorDataset(x_train, y_train)
        valid_ds = TensorDataset(x_valid, y_valid)

        logging.info("Loading train and test data into dataloader structure.")
        train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)

        logging.info('Formatting tensors.')
        train_dl = WrappedDataLoader(train_dl, preprocess)
        valid_dl = WrappedDataLoader(valid_dl, preprocess)

        logging.info('Defining model.')
        model, opt = get_model(model_type, learn_rate)

        logging.info('Fitting the model.')
        test_loss = fit(epochs, model, loss_func, opt, train_dl, valid_dl)


        logging.info('Saving the state dictionary and model.')
        torch.save(model.state_dict(), model_state_dict)
        torch.save(obj=model, f=model_complete, pickle_module=dill)


        logging.info('Test Loss is %s', test_loss)

        num_valid_rows = x_valid.shape[0]
        overall_diff_max = 0
        max_diff_list = []

        # Calculate the maximum difference of predicted values
        for valid_row in range(num_valid_rows):
            pred = model(x_valid[valid_row].view(-1, 1, 4, 11).float())
            tensor_diff = torch.abs(torch.sub(y_valid[valid_row], pred, alpha=1))
            tensor_diff_max = torch.max(tensor_diff)

            if (tensor_diff_max > overall_diff_max):
                overall_diff_max = tensor_diff_max

            max_diff_list.append(tensor_diff_max.item())
            overall_max_diff_list.append(tensor_diff_max.item())

        # Calculate max difference mean and maximum difference confidence interval
        max_diff_mean, max_diff_conf_interval = mean_confidence_interval(np.asarray(max_diff_list), 0.95)
        logging.info('Max Diff Mean is %s', max_diff_mean)
        logging.info('Max Diff Conf Interval is %s', max_diff_conf_interval)

        # Populate statistics list to be saved into CSV later
        stats_list = [execution_num, test_loss, "N/A", max_diff_mean, max_diff_conf_interval]

        if (execution_num == 1):
            results_array = np.array(stats_list)
            index_values = [execution_num - 1]
            execution_num += 1
        else:
            results_array = np.vstack([results_array, np.array(stats_list)])
            index_values.append(execution_num - 1)


    all_val_losses = results_array[:, 1].astype(float)

    overall_max_diff_mean, overall_max_diff_conf_interval = mean_confidence_interval(
        np.asarray(overall_max_diff_list), 0.95)
    all_val_loss_mean, all_val_loss_conf_interval = mean_confidence_interval(all_val_losses, 0.95)

    overall_stats = ["Combined", all_val_loss_mean, all_val_loss_conf_interval, overall_max_diff_mean,
                     overall_max_diff_conf_interval]

    logging.info('Overall Diff Mean is %s', overall_max_diff_mean)
    logging.info('Overall Diff Conf Interval is %s', overall_max_diff_conf_interval)

    execution_df = pd.DataFrame(results_array,
                                columns=['execution_num', 'val_loss', 'val_loss_conf_int', 'max_diff_mean',
                                         'max_diff_conf_int'])
    overall_df = pd.DataFrame(np.array(overall_stats).reshape(1, 5),
                              columns=['execution_num', 'val_loss', 'val_loss_conf_int', 'max_diff_mean',
                                       'max_diff_conf_int'])

    # Save the results to CSV
    full_df = pd.concat([execution_df, overall_df], axis=0)
    full_df.to_csv(results_file)

