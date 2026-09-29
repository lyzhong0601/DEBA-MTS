import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tqdm
import numpy as np
import pandas as pd
from dataset import TimeDataset, AttackEvaluateSet
from torch.utils.data import DataLoader
from attack import Attacker
from sklearn.metrics import mean_absolute_error, mean_squared_error
from forecast_models import TimesNet, Autoformer, FEDformer, DLinear, LightTS
import os

MODEL_MAP = {
    'TimesNet': TimesNet,
    'Autoformer': Autoformer,
    'FEDformer': FEDformer,
    'DLinear': DLinear,
    'LightTS': LightTS
}

class Trainer:
    """
    Model trainer.
    Main functions:
    1. train: initialize the static trigger and inject poisoned points.
    2. validate: evaluate attack and clean forecasting performance.
    3. test: train a new forecasting model from scratch on poisoned data.
    """

    def __init__(self, config, atk_vars, target_pattern, train_mean, train_std,
                 train_data, test_data, train_data_stamps, test_data_stamps, device, logger):
        self.config = config
        self.mean = train_mean
        self.std = train_std
        self.test_data = test_data
        self.device = device
        self.logger = logger

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs

        self.train_data_stamps = train_data_stamps
        self.test_data_stamps = test_data_stamps

        train_set = TimeDataset(train_data, train_mean, train_std, device, num_for_hist=12, num_for_futr=12, timestamps=train_data_stamps)
        self.attacker = Attacker(train_set, atk_vars, config, target_pattern, device)
        self.use_timestamps = config.Dataset.use_timestamps

        self.prepare_data()

    def prepare_data(self):
        self.train_set = self.attacker.dataset
        self.cln_test_set = TimeDataset(self.test_data, self.mean, self.std, self.device, num_for_hist=12,
                                           num_for_futr=12, timestamps=self.test_data_stamps)
        self.atk_test_set = AttackEvaluateSet(self.attacker, self.test_data, self.mean, self.std, self.device,
                                              num_for_hist=12, num_for_futr=12, timestamps=self.test_data_stamps)

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.cln_test_loader = DataLoader(self.cln_test_set, batch_size=self.batch_size, shuffle=False)
        self.atk_test_loader = DataLoader(self.atk_test_set, batch_size=self.batch_size, shuffle=False,
                                          collate_fn=self.atk_test_set.collate_fn)

    def train(self):
        if not hasattr(self.attacker, 'atk_ts'):
            if self.config.method != 'joint_ae_midhigh':
                raise ValueError(f"Unsupported timestamp selection method: {self.config.method}. Kairos only supports joint_ae_midhigh.")
            self.attacker.select_atk_timestamp_joint_ae_midhigh()

        self.logger.info(
            f"Attack timestamp selection completed | method={self.config.method} | "
            f"num_ts={len(self.attacker.atk_ts)}"
        )
        self.logger.info(f"atk_ts[:20] = {self.attacker.atk_ts[:20].detach().cpu().numpy()}")

        self.attacker.sparse_inject()

        # Rebuild the DataLoader with poisoned data.
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.logger.info("Poisoning completed. Triggers were injected with the current timestamp strategy.")

    def validate(self, model, epoch, atk_eval_epoch=0):
        model.eval()
        cln_info = atk_info = ''
        with torch.no_grad():
            cln_preds = []
            atk_preds = []
            cln_targets = []
            atk_targets = []

            for batch_index, batch_data in enumerate(self.cln_test_loader):
                # calculate the clean performance
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                x_des = torch.zeros_like(labels)
                outputs = model(encoder_inputs, x_mark, x_des, None)
                outputs = self.cln_test_set.denormalize(outputs)
                cln_targets.append(labels.cpu().detach().numpy())
                cln_preds.append(outputs.cpu().detach().numpy())

            cln_preds = np.concatenate(cln_preds, axis=0)
            cln_targets = np.concatenate(cln_targets, axis=0)
            cln_mae = mean_absolute_error(cln_targets.reshape(-1, 1), cln_preds.reshape(-1, 1))
            cln_rmse = mean_squared_error(cln_targets.reshape(-1, 1), cln_preds.reshape(-1, 1)) ** 0.5

            cln_info = f' | clean MAE: {cln_mae:.2f}, clean RMSE: {cln_rmse:.2f}'

            if epoch >= atk_eval_epoch:
                for batch_index, batch_data in enumerate(self.atk_test_loader):
                    # calculate the attacked performance
                    if not self.use_timestamps:
                        encoder_inputs, labels, clean_labels, idx = batch_data
                        x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                    else:
                        encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                    encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                    labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                    x_des = torch.zeros_like(labels)
                    outputs = model(encoder_inputs, x_mark, x_des, None)
                    outputs = self.atk_test_set.denormalize(outputs)

                    labels = labels[:, :self.attacker.pattern_len, self.attacker.atk_vars]
                    outputs = outputs[:, :self.attacker.pattern_len, self.attacker.atk_vars]
                    atk_targets.append(labels.cpu().detach().numpy())
                    atk_preds.append(outputs.cpu().detach().numpy())

                atk_preds = np.concatenate(atk_preds, axis=0)
                atk_targets = np.concatenate(atk_targets, axis=0)
                atk_mae = mean_absolute_error(atk_targets.reshape(-1, 1), atk_preds.reshape(-1, 1))
                atk_rmse = mean_squared_error(atk_targets.reshape(-1, 1), atk_preds.reshape(-1, 1)) ** 0.5

                atk_info = f' | attacked MAE: {atk_mae:.2f}, attacked RMSE: {atk_rmse:.2f}'

        info = 'Epoch: {}'.format(epoch) + cln_info + atk_info
        self.logger.info(info)

        return {'clean_mae': cln_mae, 'clean_rmse': cln_rmse, 'attack_mae': atk_mae if 'atk_mae' in locals() else None}

    def test(self):
        # Train a new model from scratch on poisoned data.
        model = MODEL_MAP[self.config.model_name](self.config.Model).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        self.attacker.sparse_inject()
        self.train_set = self.attacker.dataset
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        # Track metrics for each epoch.
        clean_maes = []
        attack_maes = []

        for epoch in range(self.num_epochs):
            pbar = tqdm.tqdm(self.train_loader, desc=f'Training new forecasting model {epoch}/{self.num_epochs}')
            for batch_index, batch_data in enumerate(pbar):
                if not self.use_timestamps:
                    encoder_inputs, labels, clean_labels, idx = batch_data
                    x_mark = torch.zeros(encoder_inputs.shape[0], encoder_inputs.shape[-1], 4).to(self.device)
                else:
                    encoder_inputs, labels, clean_labels, x_mark, y_mark, idx = batch_data
                encoder_inputs = torch.squeeze(encoder_inputs).to(self.device).permute(0, 2, 1)
                labels = torch.squeeze(labels).to(self.device).permute(0, 2, 1)

                optimizer.zero_grad()

                x_des = torch.zeros_like(labels)
                outputs = model(encoder_inputs, x_mark, x_des, None)
                outputs = self.train_set.denormalize(outputs)

                loss = F.smooth_l1_loss(outputs, labels)
                loss.backward()
                optimizer.step()

            # Record metrics for the current epoch.
            current_metrics = self.validate(model, epoch, 0)
            if current_metrics:
                clean_maes.append(current_metrics['clean_mae'])
                if current_metrics['attack_mae'] is not None:
                    attack_maes.append(current_metrics['attack_mae'])

        # Find the epoch with the best clean MAE among the last 5 epochs.
        if len(clean_maes) > 0:
            last_n = min(5, len(clean_maes))

            # Get clean MAE values from the last N epochs.
            last_n_clean = clean_maes[-last_n:]

            # Locate the minimum value and its index in last_n_clean.
            min_clean_val = min(last_n_clean)
            min_idx_in_last_n = last_n_clean.index(min_clean_val)

            log_msg = f'Best clean MAE in the last {last_n} epochs: {min_clean_val:.2f}'

            # If an attack MAE exists for the same epoch, report it as well.
            if len(attack_maes) >= len(clean_maes): # Ensure the list is aligned or long enough.
                last_n_attack = attack_maes[-last_n:]
                best_epoch_attack_mae = last_n_attack[min_idx_in_last_n]
                log_msg += f' | Corresponding attack MAE: {best_epoch_attack_mae:.2f}'

            self.logger.info(log_msg)
