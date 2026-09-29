import torch
import numpy as np
from math import ceil

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


########################################################
# Shared autoencoder
########################################################
class MultiChannelConv1DAE(nn.Module):
    def __init__(self, in_channels, seq_len):
        super(MultiChannelConv1DAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(16, in_channels, kernel_size=3, padding=1)
        )

        self.align = nn.AdaptiveAvgPool1d(seq_len)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.align(decoded)


########################################################
# Node selector: Kairos joint AE
########################################################
class NodeSelector:
    def __init__(self, train_data_seq, config, device='cuda', logger=None):
        """
        train_data_seq: (T, N, C)
        """
        self.train_data_seq = train_data_seq
        self.config = config
        self.device = device
        self.logger = logger

        # Analyze channel 0 by default.
        if train_data_seq.ndim == 3:
            self.data = train_data_seq[:, :, 0]   # (T, N, C) -> (T, N)
        elif train_data_seq.ndim == 2:
            self.data = train_data_seq

        self.data = np.asarray(self.data, dtype=np.float32)

        self.node_window = getattr(config, 'node_window', 12)
        self.node_ae_epochs = getattr(config, 'node_ae_epochs', 20)
        self.node_ae_max_windows = getattr(config, 'node_ae_max_windows', 2000)

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    ####################################################
    # Kairos: multi-node joint AE node selection
    ####################################################
    def select_joint_ae_nodes(self, k):
        """
        Multi-node joint AE node selection:
        1. Train the AE with joint windows from all nodes.
        2. Compute each node's average reconstruction error across all windows.
        3. Select high-error nodes as high-risk nodes.
        """
        data = self.data  # (T, N)
        T, N = data.shape
        window = self.node_window

        self._log(f'[NodeSelector] Start joint AE node selection | T={T}, N={N}, window={window}')

        # 1. Build joint windows with shape (num_windows, N, window).
        windows = []
        for i in range(T - window + 1):
            w = data[i:i + window, :].T   # (N, window)
            windows.append(w)

        windows = np.array(windows)  # (num_windows, N, window)

        # Randomly subsample windows for efficiency.
        if len(windows) > self.node_ae_max_windows:
            idx = np.random.choice(len(windows), self.node_ae_max_windows, replace=False)
            windows = windows[idx]

        # Apply per-node z-score normalization within each local window to emphasize shape.
        means = np.mean(windows, axis=2, keepdims=True)
        stds = np.std(windows, axis=2, keepdims=True) + 1e-6
        windows = (windows - means) / stds

        windows_tensor = torch.tensor(windows, dtype=torch.float32).to(self.device)

        # 2. Train the joint AE.
        ae_model = MultiChannelConv1DAE(in_channels=N, seq_len=window).to(self.device)
        optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        dataset = TensorDataset(windows_tensor, windows_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        self._log(f'[NodeSelector] Training joint AE on {len(windows_tensor)} sampled windows...')

        ae_model.train()
        for epoch in range(self.node_ae_epochs):
            total_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = ae_model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self._log(f'[NodeSelector][AE] epoch {epoch+1}/{self.node_ae_epochs}, loss={avg_loss:.6f}')

        # 3. Compute each node's average reconstruction error.
        ae_model.eval()
        node_error_sum = np.zeros(N, dtype=np.float64)
        node_error_cnt = 0

        eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)

        with torch.no_grad():
            for batch_x, _ in eval_loader:
                output = ae_model(batch_x)
                # per-node error: (B, N)
                per_node_err = torch.mean((output - batch_x) ** 2, dim=2)
                node_error_sum += per_node_err.sum(dim=0).cpu().numpy()
                node_error_cnt += per_node_err.shape[0]

        node_scores = node_error_sum / (node_error_cnt + 1e-12)

        idx = np.argsort(-node_scores)

        self._log('[NodeSelector] joint_ae selection finished.')
        self._log(f'[NodeSelector] top-10 node scores: {[(int(i), float(node_scores[i])) for i in idx[:10]]}')

        return idx[:k]

    ####################################################
    # Unified entry point
    ####################################################
    def select_nodes(self, k, method='joint_ae'):
        if method != 'joint_ae':
            raise ValueError(f'Unsupported node selection method: {method}. Kairos only supports joint_ae.')
        return self.select_joint_ae_nodes(k)


########################################################
# Attacker
########################################################
class Attacker:
    def __init__(self, dataset, atk_vars, config, target_pattern, device='cuda'):
        self.device = device
        self.dataset = dataset
        self.config = config
        self.target_pattern = target_pattern
        self.atk_vars = atk_vars

        self.trigger_len = config.trigger_len
        self.pattern_len = config.pattern_len
        self.bef_tgr_len = config.bef_tgr_len

        self.fct_input_len = config.Dataset.len_input
        self.fct_output_len = config.Dataset.num_for_predict
        self.alpha_t = config.alpha_t
        self.alpha_s = config.alpha_s
        self.temporal_poison_num = ceil(self.alpha_t * len(self.dataset))

        self.temporal_ae_epochs = getattr(config, 'temporal_ae_epochs', 30)
        self.temporal_ae_max_windows = getattr(config, 'temporal_ae_max_windows', 4000)
        self.temporal_bin_low_q = getattr(config, 'temporal_bin_low_q', 0.70)
        self.temporal_bin_high_q = getattr(config, 'temporal_bin_high_q', 0.95)
        self.temporal_selection_mode = getattr(config, 'temporal_selection_mode', 'mid_high')

        self.generated_static_trigger = self.generate_trigger(self.dataset.std)

    def _log(self, msg):
        logger = getattr(self.config, 'logger', None)
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    def set_atk_timestamp(self, atk_ts):
        self.atk_ts = atk_ts

    def generate_trigger(self, std_val):
        delta_tgr = 0.2 * std_val
        trigger = (torch.rand(1, 1, self.trigger_len).to(self.device) * 2 - 1) * delta_tgr
        return trigger

    def sparse_inject(self):
        assert hasattr(self, 'atk_vars'), 'Please set attack variables first.'
        assert hasattr(self, 'atk_ts'), 'Please set attack timestamps first.'
        self.dataset.init_poison_data()

        trigger_len = self.trigger_len
        pattern_len = self.target_pattern.shape[-1]

        if not hasattr(self, 'generated_static_trigger'):
            self.generated_static_trigger = self.generate_trigger(self.dataset.std)

        current_trigger = self.generated_static_trigger.expand(len(self.atk_vars), 1, -1)

        for beg_idx in self.atk_ts.tolist():
            self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx:beg_idx + trigger_len] = \
                self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx - 1:beg_idx] + current_trigger.detach()

            self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx + trigger_len:beg_idx + trigger_len + pattern_len] = \
                self.target_pattern + self.dataset.poisoned_data[self.atk_vars, 0:1, beg_idx - 1:beg_idx]

    def predict_trigger(self, data_bef_trigger):
        c = data_bef_trigger.shape[1]
        n = data_bef_trigger.shape[0]
        if not hasattr(self, 'generated_static_trigger'):
            self.generated_static_trigger = self.generate_trigger(self.dataset.std)
        return self.generated_static_trigger.expand(n, c, -1), torch.zeros_like(self.generated_static_trigger)

    ####################################################
    # Kairos: selected-node temporal score with mid-high bin sampling
    ####################################################
    def _build_selected_node_temporal_windows(self):
        data = self.dataset.data
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        atk_vars_np = self.atk_vars.detach().cpu().numpy() if torch.is_tensor(self.atk_vars) else np.asarray(self.atk_vars)
        series = data[atk_vars_np, 0, :]  # (K, T)

        window_size = self.bef_tgr_len + self.trigger_len
        dataset_len = len(self.dataset)

        valid_indices = []
        windows_list = []

        for beg_idx in range(self.bef_tgr_len, dataset_len):
            if beg_idx + self.trigger_len + self.pattern_len < data.shape[-1]:
                w = series[:, beg_idx - self.bef_tgr_len: beg_idx + self.trigger_len]  # (K, window_size)
                if w.shape[1] != window_size:
                    continue

                means = np.mean(w, axis=1, keepdims=True)
                stds = np.std(w, axis=1, keepdims=True) + 1e-6
                norm_w = (w - means) / stds

                windows_list.append(norm_w)
                valid_indices.append(beg_idx)

        if len(windows_list) == 0:
            return None, None

        windows_np = np.asarray(windows_list, dtype=np.float32)
        valid_indices = np.asarray(valid_indices, dtype=np.int64)

        if len(windows_np) > self.temporal_ae_max_windows:
            sampled_idx = np.random.choice(len(windows_np), self.temporal_ae_max_windows, replace=False)
            sampled_idx = np.sort(sampled_idx)
            windows_np = windows_np[sampled_idx]
            valid_indices = valid_indices[sampled_idx]

        return windows_np, valid_indices

    def _train_temporal_ae_and_get_scores(self, windows_np):
        windows_tensor = torch.tensor(windows_np, dtype=torch.float32).to(self.device)

        num_atk_nodes = windows_tensor.shape[1]
        window_size = windows_tensor.shape[2]

        self._log(
            f"[Attacker] Training selected-node temporal AE | "
            f"nodes={num_atk_nodes}, samples={len(windows_tensor)}, window={window_size}"
        )

        ae_model = MultiChannelConv1DAE(in_channels=num_atk_nodes, seq_len=window_size).to(self.device)
        optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        dataset = TensorDataset(windows_tensor, windows_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        ae_model.train()
        for epoch in range(self.temporal_ae_epochs):
            total_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = ae_model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(dataloader), 1)
            self._log(f'[Attacker][Temporal AE] epoch {epoch + 1}/{self.temporal_ae_epochs}, loss={avg_loss:.6f}')

        ae_model.eval()
        scores = []
        eval_loader = DataLoader(dataset, batch_size=512, shuffle=False)
        with torch.no_grad():
            for batch_x, _ in eval_loader:
                output = ae_model(batch_x)
                # One temporal score per window: average over nodes and time.
                mse_per_sample = torch.mean((output - batch_x) ** 2, dim=(1, 2))
                scores.extend(mse_per_sample.detach().cpu().numpy())

        scores = np.asarray(scores, dtype=np.float64)
        self._log(
            f'[Attacker] temporal score stats | '
            f'min={scores.min():.6f}, median={np.median(scores):.6f}, max={scores.max():.6f}'
        )
        return scores

    def _sample_non_overlapping_indices(self, candidate_indices, dataset_len):
        if len(candidate_indices) == 0:
            return np.array([], dtype=np.int64)

        candidate_indices = np.asarray(candidate_indices, dtype=np.int64).copy()
        np.random.shuffle(candidate_indices)

        select_pos_mark = torch.zeros(dataset_len, dtype=torch.int)
        selected_idx = []

        for beg_idx in candidate_indices:
            end_idx = beg_idx + self.trigger_len + self.pattern_len + 8
            if end_idx < dataset_len and beg_idx > self.bef_tgr_len:
                if torch.sum(select_pos_mark[beg_idx:end_idx]) == 0:
                    selected_idx.append(int(beg_idx))
                    select_pos_mark[beg_idx:end_idx] = 1

            if len(selected_idx) >= self.temporal_poison_num:
                break

        return np.asarray(sorted(selected_idx), dtype=np.int64)

    def _select_high_score_non_overlapping_indices(self, valid_indices, scores, dataset_len):
        order = np.argsort(-scores)
        select_pos_mark = torch.zeros(dataset_len, dtype=torch.int)
        selected_idx = []

        for score_idx in order:
            beg_idx = int(valid_indices[score_idx])
            end_idx = beg_idx + self.trigger_len + self.pattern_len + 8
            if end_idx < dataset_len and beg_idx > self.bef_tgr_len:
                if torch.sum(select_pos_mark[beg_idx:end_idx]) == 0:
                    selected_idx.append(beg_idx)
                    select_pos_mark[beg_idx:end_idx] = 1

            if len(selected_idx) >= self.temporal_poison_num:
                break

        return np.asarray(sorted(selected_idx), dtype=np.int64)

    def select_atk_timestamp_joint_ae_midhigh(self):
        windows_np, valid_indices = self._build_selected_node_temporal_windows()
        if windows_np is None or len(windows_np) == 0:
            raise RuntimeError('[Attacker] Failed to build temporal AE windows; Kairos timestamp selection cannot run.')

        scores = self._train_temporal_ae_and_get_scores(windows_np)

        low_q = float(self.temporal_bin_low_q)
        high_q = float(self.temporal_bin_high_q)
        if not (0.0 <= low_q < high_q <= 1.0):
            self._log(f'[Attacker] Invalid quantile interval ({low_q}, {high_q}); falling back to [0.70, 0.95).')
            low_q, high_q = 0.70, 0.95

        q_low = np.quantile(scores, low_q)
        q_high = np.quantile(scores, high_q)

        # mid-high bin: [low_q, high_q)
        candidate_mask = (scores >= q_low) & (scores < q_high)
        candidate_indices = valid_indices[candidate_mask]

        self._log(
            f'[Attacker] joint temporal mid-high sampling | '
            f'quantile=[{low_q:.2f}, {high_q:.2f}), '
            f'score_range=[{q_low:.6f}, {q_high:.6f}), '
            f'candidates={len(candidate_indices)}'
        )

        selected_idx = self._sample_non_overlapping_indices(candidate_indices, len(self.dataset))

        # If mid-high candidates are insufficient, expand to all candidates above low_q.
        if len(selected_idx) < self.temporal_poison_num:
            self._log(
                f'[Attacker] Insufficient mid-high candidates: selected {len(selected_idx)} / {self.temporal_poison_num}; '
                f'expanding to all candidates above low_q.'
            )
            fallback_candidate_indices = valid_indices[scores >= q_low]
            selected_idx = self._sample_non_overlapping_indices(fallback_candidate_indices, len(self.dataset))

        if len(selected_idx) < self.temporal_poison_num:
            self._log(
                f'[Attacker] AE candidates are still insufficient: selected {len(selected_idx)} / {self.temporal_poison_num}; '
                f'using temporal AE score ranking to fill the selection.'
            )
            selected_idx = self._select_high_score_non_overlapping_indices(valid_indices, scores, len(self.dataset))

        if len(selected_idx) == 0:
            raise RuntimeError('[Attacker] Temporal AE selection failed; no valid attack timestamp was found.')

        atk_ts = torch.tensor(selected_idx, dtype=torch.long).to(self.device)
        atk_ts = torch.sort(atk_ts)[0]
        self.set_atk_timestamp(atk_ts)
        self._log(f'[Attacker] joint_ae_midhigh selected {len(atk_ts)} attack timestamps.')
