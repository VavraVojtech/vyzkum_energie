import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from datetime import timedelta
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib  # for saving and loading scalers

from processing.combiner import get_train_data
# from executors.run_prediction import prepare_date
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# ===== DRAGON Imports for Hyperparameter Optimization =====
from dragon.search_operators.base_neighborhoods import ArrayInterval, FloatInterval, IntInterval
from dragon.search_space.base_variables import ArrayVar, FloatVar, IntVar
from dragon.search_algorithm.mutant_ucb import Mutant_UCB

# ===== DRAGON extension for Hyperparameter Optimization =====
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# worker = comm.Get_rank()
# size = comm.Get_size()
# print(f"Worker {worker} of {size}")
# =============================================================

# Global variables to store the best candidate model (from step 1 optimization) during HPO.
# these are stored befor loss_function --> there it is used

# =============================================================================
# ENERGY CONSUMPTION MODEL (Same architecture as before)
# =============================================================================
class EnergyConsumptionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(EnergyConsumptionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        out = self.fc(gru_out[:, -1, :])
        return out

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# Create sliding window dataset (if needed)
def create_sequences(input_data, target_data, seq_length):
    in_sequences = []
    out_sequences = []
    data_len = len(input_data)
    for i in range(data_len - seq_length):
        in_sequences.append(input_data[i:i + seq_length])
        out_sequences.append(target_data[i + seq_length])
    return torch.stack(in_sequences), torch.stack(out_sequences)

def scaling_parameters():
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    return scaler_features, scaler_target

def evaluate_model(model, X_test, y_test, criterion, scaler_target):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        loss = criterion(predictions, y_test)
        print(f'Test Loss: {loss.item():.4f}')
        # Inverse scaling for interpretability
        predictions_inv = scaler_target.inverse_transform(predictions.numpy())
        y_test_inv = scaler_target.inverse_transform(y_test.numpy())
    # (Optionally, add additional evaluation/plotting here.)

def load_model(model_path, input_size):
    # Use the same hyperparameters as optimized or default ones if needed
    hidden_size = 377
    output_size = 1
    num_layers = 3
    # HiddenSize=296, NumLayers=3, LR=0.003992992690842621
    
    model = EnergyConsumptionModel(input_size, hidden_size, output_size, num_layers)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()
    return model

def load_model_from_step1(checkpoint_path, best_parameters_path, input_size):
    
    # read parameters
    df = pd.read_csv(best_parameters_path)
    best_hidden_size = int(df['best_hidden_size'].iloc[0])
    best_num_layers = int(df['best_num_layers'].iloc[0])
    best_lr = float(df['best_lr'].iloc[0])

    print(f"Best parameters for prediction: HiddenSize={best_hidden_size}, NumLayers={best_num_layers}, LR={best_lr}")

    # read model
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # prepare model
    model = EnergyConsumptionModel(input_size, best_hidden_size, 1, best_num_layers)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def prepare_data(df, input_cols, label_col):
    df_features = df[input_cols]
    df_target = df[[label_col]]

    sequence_length = 5  # Number of previous days
    X_sequences = []
    y_targets = []
    num_features = len(input_cols)
    for i in range(len(df) - sequence_length):
        past_features = df_features.iloc[i:i + sequence_length - 1].values
        current_features = df_features.iloc[i + sequence_length - 1].copy()
        current_features['C_ote'] = 0.0  # set target feature to 0 for prediction day
        current_features = current_features.values.reshape(1, num_features)
        sequence = np.vstack([past_features, current_features])
        target = df_target.iloc[i + sequence_length - 1].values
        X_sequences.append(sequence)
        y_targets.append(target)
    X = np.array(X_sequences)
    y = np.array(y_targets)
    X = np.nan_to_num(X, nan=0.0)
    
    scaler_features, scaler_target = scaling_parameters()
    num_samples, seq_len, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)
    X_scaled = scaler_features.fit_transform(X_reshaped).reshape(num_samples, seq_len, num_features)
    y_scaled = scaler_target.fit_transform(y)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler_features, scaler_target

# =============================================================================
# DRAGON HPO: Define Global Variables for the Optimization Evaluation
# =============================================================================
# (These will be set once data is prepared)
X_train_global = None
X_test_global = None
y_train_global = None
y_test_global = None
best_candidate_loss_step1 = float('inf')
best_candidate_state_step1 = None
best_candidate_hyperparams = None

def loss_function(args, idx, verbose=False):
    """
    DRAGON loss function used to evaluate a given set of hyperparameters.
    args[0]: HiddenSize (float; will be cast to int)
    args[1]: NumLayers (float; will be cast to int)
    args[2]: LR (learning rate)
    """
    global best_candidate_loss_step1, best_candidate_state_step1, best_candidate_hyperparams
    hidden_size = int(args[0])
    num_layers = int(args[1])
    lr = args[2]
    input_size = X_train_global.shape[2]
    model = EnergyConsumptionModel(input_size, hidden_size, 1, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
    
    best_val_loss = float('inf')
    # Train for a limited number of epochs during HPO (e.g. 50 epochs)
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_global)
        train_loss = criterion(outputs, y_train_global)
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_global)
            val_loss = criterion(val_outputs, y_test_global)
        scheduler.step(val_loss)
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
    
    # Update global best candidate if this candidate is better.
    if best_val_loss < best_candidate_loss_step1:
        best_candidate_loss_step1 = best_val_loss
        best_candidate_state_step1 = model  # Save the candidate's state dict.
        best_candidate_hyperparams = (hidden_size, num_layers, lr)
    print(f"Idx: {idx}, Params: HiddenSize={hidden_size}, NumLayers={num_layers}, LR={lr:.6f}, Val Loss: {best_val_loss:.6f}")
    return best_val_loss, model


# =============================================================================
# TRAINING FUNCTION WITH DRAGON OPTIMIZATION
# =============================================================================
def train_model(X_train, y_train, X_test, y_test, model_path, model_path_1st_step, best_parameters_path, epochs, scaler_target, input_cols, run_optimization=True, load_optimized_model=False):
    # Use global variables so the loss_function can access the prepared data.
    global X_train_global, X_test_global, y_train_global, y_test_global
    X_train_global, X_test_global, y_train_global, y_test_global = X_train, X_test, y_train, y_test

    save_dir = "save/V5_dragon_opt"

    if run_optimization:
        # START DRAGON COMMENT
        if not load_optimized_model:
            logging.info("Running DRAGON Optimization")
            # --- DRAGON Hyperparameter Search Space ---
            hidden_size_var = IntVar("HiddenSize", lower=256, upper=512, neighbor=IntInterval(32)) # IntInterval(128)
            num_layers_var = IntVar("NumLayers", lower=1, upper=5, neighbor=IntInterval(1))
            lr_var = FloatVar("LR", lower=0.0001, upper=0.08, neighbor=FloatInterval(0.01)) # lower=0.001, upper=0.1, neighbor=FloatInterval(0.01))
            search_space = ArrayVar(hidden_size_var, num_layers_var, lr_var, label="Search Space", neighbor=ArrayInterval())
            # -------------------------------------------

            # Run the DRAGON search algorithm to tune hyperparameters
            search_algorithm = Mutant_UCB(
                search_space,
                save_dir=save_dir,
                T = 1500,   # 100,      # Number of iterations.
                N = 5,      # 5,        # Maximum number of partial training for one configuration.
                K = 20,     # 20,       # Size of the population.
                E = 0.01,   # 0.01,     # Exploratory parameters. [lower (0) = exploitation, go for local minimas; higher (1) = exploration, explore for more global minimas]
                evaluation = loss_function,
                models = None,    # List of configurations that should be included into the initial population.
                pop_path = None,  # Path towards a directory containing an former evaluation that we aim to continue.
                verbose = False,   # print current state
                set_mpi = False
            )
            search_algorithm.run()           # .run_no_mpi() |OR| .run_mpi() |OR| .run()
            # search_algorithm.run_mpi()
            # search_algorithm

            # Save the best candidate from step 1
            torch.save(best_candidate_state_step1.state_dict(), model_path_1st_step)
            # torch.save(model.state_dict(), model_path)

            params_dict = {
                "best_hidden_size": [best_candidate_hyperparams[0]],
                "best_num_layers": [best_candidate_hyperparams[1]],
                "best_lr": [best_candidate_hyperparams[2]]
            }
            df_params = pd.DataFrame(params_dict)
            df_params.to_csv(best_parameters_path, index=False)
            
            logging.info(f"Step 1 (HPO) optimized model saved to {model_path_1st_step} and parameters to {best_parameters_path}")
        else:
            # Load the best candidate from step 1 from the checkpoint (instead of the pickle file)
            df = pd.read_csv(best_parameters_path)
            best_hidden_size = int(df['best_hidden_size'].iloc[0])
            best_num_layers = int(df['best_num_layers'].iloc[0])
            best_lr = float(df['best_lr'].iloc[0])

            logging.info("Loaded optimized hyperparameters from step 1:")
            logging.info(f"HiddenSize={best_hidden_size}, NumLayers={best_num_layers}, LR={best_lr}")

        # Load the candidate configuration from the corresponding pickle file
        config_file = os.path.join(save_dir, f"best_model/x.pkl")
        with open(config_file, "rb") as f:
            best_config = pickle.load(f)

        # Assuming best_config is a list/tuple with the three parameters:
        best_hidden_size = int(best_config[0])
        best_num_layers = int(best_config[1])
        best_lr = best_config[2]

        logging.info("Best parameters found:")
        logging.info(f"[DRAGON HPO] Best parameters: HiddenSize={best_hidden_size}, NumLayers={best_num_layers}, LR={best_lr}\n")
        logging.info("Building New Neural Network with best optimized parameters by DRAGON Optimization")
        # END DRAGON COMMENT
        # logging.info('please uncomment and use python 3.9 pred_short_ng_optimize')
    else:
        best_hidden_size = 377      # 285
        best_num_layers = 3         # 2
        best_lr = 0.00236           # 0.0023560198058878027
        best_hidden_size,best_num_layers,best_lr

        logging.info("Building New Neural Network with best parameters by hand")

    # Build the final model with the best (or default) hyperparameters.
    input_size = X_train.shape[2]
    model = EnergyConsumptionModel(input_size, best_hidden_size, 1, best_num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    scheduler_type = 1  # using ReduceLROnPlateau
    if scheduler_type == 0:
        scheduler = StepLR(optimizer, step_size=100, gamma=0.8)
    elif scheduler_type == 1:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=25)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 45
    val_check_interval = 5
    best_model_path = 'best_model.pth'
    early_stop = False

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        train_loss.backward()
        optimizer.step()

        if scheduler_type == 0:
            scheduler.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
        if scheduler_type == 1:
            scheduler.step(val_loss)

        if (epoch + 1) % val_check_interval == 0:
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Epoch {epoch+1}: Validation loss decreased to {best_val_loss:.6f}. Saving model.")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                early_stop = True
                break
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}, LR: {current_lr:.6f}')
        else:
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.6f}, LR: {current_lr:.6f}')

    if not early_stop:
        model.load_state_dict(torch.load(best_model_path))

    evaluate_model(model, X_test, y_test, criterion, scaler_target)
    torch.save(model.state_dict(), model_path)
    logging.info('NN build and train finished')
    return model

# =============================================================================
# PREDICTION FUNCTION (Same logic as before)
# =============================================================================

# def predict_nn(model_path, model_path_1st_step, best_parameters_path, predict_1st_step, input_cols, label_col, scaler_features, scaler_target, day_shift):
#     start_date = pd.to_datetime('today').normalize() - timedelta(days=day_shift)
#     end_date = start_date + timedelta(days=5)
#     sequence_length = 5  # Number of days in sequence
#     data_start_date = start_date - timedelta(days=sequence_length + 5 - 1)

#     df = prepare_date(start_date, end_date, Open_Meteo_database=True)
#     df.set_index('date', inplace=True)
#     columns_needed = input_cols.copy()
#     if label_col not in input_cols:
#         columns_needed += [label_col]
#     df = df[columns_needed]

#     predicted_C_ote = {}
#     all_prediction_dates = pd.date_range(start=start_date - timedelta(days=5), end=end_date)
#     future_prediction_dates = pd.date_range(start=start_date, end=end_date)

#     if predict_1st_step:
#         model = load_model_from_step1(model_path_1st_step, best_parameters_path, input_size=len(input_cols))
#     else:
#         model = load_model(model_path, input_size=len(input_cols))

#     for current_date in all_prediction_dates:
#         sequence_dates = pd.date_range(end=current_date, periods=sequence_length)
#         sequence_features = []
#         for seq_date in sequence_dates:
#             if seq_date in df.index:
#                 features = df.loc[seq_date, input_cols].copy()
#             else:
#                 features = pd.Series(0.0, index=input_cols)
#             if seq_date == current_date:
#                 features['C_ote'] = 0.0
#             else:
#                 if pd.isna(features['C_ote']):
#                     if seq_date in predicted_C_ote and seq_date in future_prediction_dates:
#                         features['C_ote'] = predicted_C_ote[seq_date]
#                     elif seq_date in df.index and not pd.isna(df.loc[seq_date, 'C_ote']):
#                         features['C_ote'] = df.loc[seq_date, 'C_ote']
#                     else:
#                         features['C_ote'] = 0.0
#             if 'cons_c_2d' in features.index and pd.isna(features['cons_c_2d']):
#                 cons_c_2d_date = seq_date - timedelta(days=2)
#                 if cons_c_2d_date in predicted_C_ote and cons_c_2d_date in future_prediction_dates:
#                     features['cons_c_2d'] = predicted_C_ote[cons_c_2d_date]
#                 elif cons_c_2d_date in df.index and not pd.isna(df.loc[cons_c_2d_date, 'C_ote']):
#                     features['cons_c_2d'] = df.loc[cons_c_2d_date, 'C_ote']
#                 else:
#                     features['cons_c_2d'] = 0.0
#             sequence_features.append(features.values)
#         sequence_np = np.array(sequence_features)
#         sequence_np = np.nan_to_num(sequence_np, nan=0.0)
#         num_samples, num_features = sequence_np.shape
#         sequence_np_reshaped = sequence_np.reshape(-1, num_features)
#         sequence_scaled = scaler_features.transform(sequence_np_reshaped)
#         sequence_scaled = sequence_scaled.reshape(1, sequence_length, num_features)
#         sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32)
#         model.eval()
#         with torch.no_grad():
#             prediction = model(sequence_tensor)
#             prediction_inv = scaler_target.inverse_transform(prediction.numpy())
#             predicted_value = prediction_inv.flatten()[0]
#         predicted_C_ote[current_date] = predicted_value
#         df.loc[current_date, 'Predicted_C_ote'] = predicted_value
#         if current_date in future_prediction_dates:
#             df.loc[current_date, 'C_ote'] = predicted_value
#             future_date = current_date + timedelta(days=2)
#             if future_date in df.index and pd.isna(df.loc[future_date, 'cons_c_2d']):
#                 df.loc[future_date, 'cons_c_2d'] = predicted_value
#     history_start_date = start_date - timedelta(days=5)
#     full_dates = pd.date_range(start=history_start_date, end=end_date)
#     df_result = df.loc[full_dates, ['C_ote', 'Predicted_C_ote']]
#     df_result['Predicted_C_ote'] = df_result['Predicted_C_ote'].round(0)
#     return df_result

# =============================================================================
# PLOTTING FUNCTION (Unchanged)
# =============================================================================
def plot_predictions(model, X_test, y_test):
    scaler_features, scaler_target = scaling_parameters()
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        predictions = scaler_target.inverse_transform(predictions)
        real_values = scaler_target.inverse_transform(y_test.numpy())
    for i in range(10):
        print(f'Real Value: {real_values[i][0]:.4f}, Predicted Value: {predictions[i][0]:.4f}, Diff: {real_values[i][0] - predictions[i][0]:.4f}')
    plt.figure(figsize=(10, 6))
    plt.plot(real_values, label='Real Values')
    plt.plot(predictions, label='Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.title('Predicted vs Real Energy Consumption')
    plt.show()

# =============================================================================
# MAIN CHAIN LIGHTNING FUNCTION (New version: NN_CHL_V5_OM_DRAGON)
# =============================================================================
def run_short_term_nn_chain_lightning_V5_OM_DRAGON(train=False, load_optimized_model = False, predict=True, predict_1st_step = False, output_to_db=True, day_shift=0):
    model_path = 'model/NN_chain_lightning_V5_OM_DRAGON.pth'
    model_path_1st_step = 'model/best_model_step1_optimized_V5.pth'
    best_parameters_path = 'model/best_parameters_NN_CHL_V5.csv'
    # Feature columns and target column
    input_cols = ['cons_c_2d', 'seasonality', 'heating_omt', 'dd_omt','is_weekend', 'is_working_day',
                  'direct_radiation', 'direct_radiation_1d',
                  'temperature_2m', 'apparent_temperature',
                  'temperature_2m_diff_01', 'apparent_temperature_diff_01',
                  'direct_radiation_diff_01', 'relative_humidity_2m_diff_01',
                  'dew_point_2m_diff_01', 'cloud_cover_diff_01',
                  'wind_speed_10m_diff_01',
                  'mix_temperature_diff_00_11',
                  'C_ote']
    label_col = 'C_ote'

    if train:
        start_date = pd.to_datetime('2022-10-01')
        end_date = pd.to_datetime('2025-02-29')
        df = get_train_data(start_date, end_date)
        X_train, X_test, y_train, y_test, scaler_features, scaler_target = prepare_data(df, input_cols, label_col)

        # Save scalers
        scaler_features_path = 'model/NN_chain_lightning_V5_OM_DRAGON_scaler_features.pkl'
        scaler_target_path = 'model/NN_chain_lightning_V5_OM_DRAGON_scaler_target.pkl'
        joblib.dump(scaler_features, scaler_features_path)
        joblib.dump(scaler_target, scaler_target_path)

        # Train using DRAGON-based hyperparameter optimization
        model = train_model(X_train, y_train, X_test, y_test, model_path, model_path_1st_step, best_parameters_path, epochs=2000, scaler_target=scaler_target, input_cols=input_cols,
                            run_optimization=False, load_optimized_model=load_optimized_model) # epochs = 1400

    # if predict:
    #     print("Predicting")
    #     scaler_features_path = 'model/NN_chain_lightning_V5_OM_DRAGON_scaler_features.pkl'
    #     scaler_target_path = 'model/NN_chain_lightning_V5_OM_DRAGON_scaler_target.pkl'
    #     scaler_features = joblib.load(scaler_features_path)
    #     scaler_target = joblib.load(scaler_target_path)
    
    #     df_result = predict_nn(model_path, model_path_1st_step, best_parameters_path, predict_1st_step,
    #                            input_cols, label_col, scaler_features, scaler_target, day_shift=day_shift)

    #     # prepare for print and for db save
    #     start_date = pd.to_datetime('today').normalize() - timedelta(days=day_shift)
    #     end_date = start_date + timedelta(days=5)
    #     df_cut = df_result.loc[start_date:end_date]
    #     df_cut = df_cut.drop(columns=['C_ote'])
    #     df_cut = df_cut.reset_index().rename(columns={'index': 'date'})
    #     df_cut.rename(columns={'Predicted_C_ote': 'cons_c_prediction'}, inplace=True)
    #     df_cut['issue_date'] = pd.to_datetime(start_date).date()
    #     df_cut['date'] = pd.to_datetime(df_cut['date']).dt.date
    #     start_date = start_date + timedelta(1)
    #     print(df_cut)

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
                        format='%(asctime)s %(levelname)s | %(message)s')
    # Uncomment the next line if you need to check historical C_ote
    train_flag              = True         # Set True to run training with DRAGON HPO
    load_optimized_model    = True        # Set True load optimized model by DRAGON
    predict_flag            = False        # To make predictions
    predict_1st_step        = False        # True = predict from "1st" step, False = predict from "2nd" step
    output_to_db_flag       = False        # To save predictions to DB
    day_shift = 0  # adjust day shift as needed
    # for d in range(1, 33+1):
    #     day_shift = d
    run_short_term_nn_chain_lightning_V5_OM_DRAGON(train=train_flag, load_optimized_model = load_optimized_model,
                                                   predict=predict_flag, predict_1st_step = predict_1st_step,
                                                   output_to_db=output_to_db_flag, day_shift=day_shift)
    
    # This is not in production, this file is for future experiments!
