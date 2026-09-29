#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import os
import random
from dataset import load_raw_data
from trainer import Trainer
import yaml
from easydict import EasyDict as edict
from utils.log_handler import get_logger
import argparse

# Import the node selector from attack.py.
from attack import NodeSelector


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parser_args():
    default_config = yaml.load(open('configs/default_config.yaml', 'r'), Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser(description='Time Series Backdoor Attack')

    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--model_name', type=str, default='DLinear')
    parser.add_argument('--dataset', type=str, default='PEMS03')

    parser.add_argument('--alpha_s', type=float, default=0.3)
    parser.add_argument('--alpha_t', type=float, default=0.03)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument(
        '--method',
        type=str,
        default='joint_ae_midhigh',
        choices=['joint_ae_midhigh'],
        help='timestamp selection method'
    )

    # Node selection method.
    parser.add_argument(
        '--node_select_method',
        type=str,
        default='joint_ae',
        choices=['joint_ae'],
        help='attack node selection method'
    )

    # joint AE node selection parameters.
    parser.add_argument('--node_window', type=int, default=12, help='window size for node selection')
    parser.add_argument('--node_ae_epochs', type=int, default=20, help='epochs for joint AE node selector')
    parser.add_argument('--node_ae_max_windows', type=int, default=2000, help='max sampled windows for node selection AE')

    # selected-node temporal AE parameters.
    parser.add_argument('--temporal_ae_epochs', type=int, default=30, help='epochs for temporal AE timestamp selector')
    parser.add_argument('--temporal_ae_max_windows', type=int, default=2000, help='max sampled windows for temporal AE')
    parser.add_argument('--temporal_bin_low_q', type=float, default=0.20, help='low quantile for mid-high temporal bin')
    parser.add_argument('--temporal_bin_high_q', type=float, default=0.80, help='high quantile for mid-high temporal bin')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--attack_lr', type=float, default=0.005)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('--pattern_type', type=str, default='cone')
    parser.add_argument('--trigger_len', type=int, default=4)
    parser.add_argument('--pattern_len', type=int, default=7)
    parser.add_argument('--bef_tgr_len', type=int, default=6)

    args = parser.parse_args()

    config = vars(args)
    config['Dataset'] = default_config['Dataset'][config['dataset']]
    config['Target_Pattern'] = default_config['Target_Pattern'][config['pattern_type']]

    config['Model'] = default_config['Model'][config['model_name']]
    config['Model']['c_out'] = config['Dataset']['num_of_vertices']
    config['Model']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Model']['dec_in'] = config['Dataset']['num_of_vertices']

    config = edict(config)
    return config


def main(config):
    log_dir = './logging/' + config.method + '/'
    log_filename = '{}_{}_t{}_{}_{}_seed{}.log'.format(
        config.dataset,
        config.model_name,
        config.alpha_t,
        config.method,
        config.node_select_method,
        config.seed
    )
    logger = get_logger(log_dir, log_filename=log_filename)
    config.logger = logger

    gpuid = config.gpuid
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

    logger.info(f"CUDA: {USE_CUDA}, {DEVICE}")

    seed_torch(config.seed)

    data_config = config.Dataset
    if not data_config.use_timestamps:
        train_mean, train_std, train_data_seq, test_data_seq = load_raw_data(data_config)
        train_data_stamps = test_data_stamps = None
    else:
        train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps = load_raw_data(data_config)

    spatial_poison_num = max(
        int(round(train_data_seq.shape[1] * config.alpha_s)),
        1
    )

    ########################################################
    # Node selection is centralized in NodeSelector from attack.py.
    ########################################################
    node_selector = NodeSelector(
        train_data_seq=train_data_seq,
        config=config,
        device=DEVICE,
        logger=logger
    )

    atk_vars = node_selector.select_nodes(
        k=spatial_poison_num,
        method=config.node_select_method
    )

    atk_vars = torch.from_numpy(np.array(atk_vars)).long().to(DEVICE)

    logger.info(f'node_select_method: {config.node_select_method}')
    logger.info(f'timestamp_method: {config.method}')
    logger.info(f'shape of attacked_variables: {atk_vars.shape}')
    logger.info(f'attacked_variables: {atk_vars.detach().cpu().numpy()}')
    logger.info(
        f'temporal_selector_cfg: epochs={config.temporal_ae_epochs}, '
        f'max_windows={config.temporal_ae_max_windows}, '
        f'mid_high_q=[{config.temporal_bin_low_q}, {config.temporal_bin_high_q})'
    )

    ########################################################
    # Target pattern.
    ########################################################
    target_pattern = config.Target_Pattern
    target_pattern = torch.tensor(target_pattern).float().to(DEVICE) * train_std

    exp_trainer = Trainer(
        config,
        atk_vars,
        target_pattern,
        train_mean,
        train_std,
        train_data_seq,
        test_data_seq,
        train_data_stamps,
        test_data_stamps,
        DEVICE,
        logger
    )

    logger.info('=' * 20 + ' [ Stage 1 ] ' + '=' * 20)
    logger.info('Initializing the static trigger and poisoned points')
    exp_trainer.train()

    logger.info('=' * 20 + ' [ Stage 2 ] ' + '=' * 20)
    logger.info('Evaluating attack performance on a new model')
    exp_trainer.test()


if __name__ == "__main__":
    config = parser_args()
    main(config)
