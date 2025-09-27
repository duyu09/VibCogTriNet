# -*- coding: utf-8 -*-
'''
- File: model_train.py
- Author: HE Feifan*; DU Yu(11250717@stu.lzjtu.edu.cn); YANG Shasha - School of Electronic and Information Engineering, Lanzhou Jiaotong University
- Date: 2025/09/23~2025/09/25
'''
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from itertools import combinations

# ========== Step 1: Define Dataset for Contrastive and Supervised Learning ==========
class TimeSeriesDataset(Dataset):
    def __init__(self, ts_root, stats_root, pair_sample_ratio=0.03, mode='supervised'):
        """
        ts_root: root directory for time series data (4 subdirs for classes)
        stats_root: root directory for statistical features (same structure and files as ts_root)
        pair_sample_ratio: percentage of data to sample for contrastive learning pairs
        mode: 'supervised' or 'contrastive'
        """
        self.mode = mode
        self.pair_sample_ratio = pair_sample_ratio
        self.data = []          # list of (time_series, stats_feature)
        self.targets = []       # class index
        self.class_names = []   # directory names
        self.pairs = []

        subdirs = sorted([d for d in os.listdir(ts_root) if os.path.isdir(os.path.join(ts_root, d))])
        self.class_names = subdirs

        for class_idx, class_name in enumerate(subdirs):
            ts_dir = os.path.join(ts_root, class_name)
            stats_dir = os.path.join(stats_root, class_name)

            csv_files = [f for f in os.listdir(ts_dir) if f.endswith('.csv')]
            for file in csv_files:
                ts_file = os.path.join(ts_dir, file)
                stats_file = os.path.join(stats_dir, file)

                # time series: one column, 8000 points
                series = pd.read_csv(ts_file, header=None).values.flatten().astype(np.float32)
                assert len(series) == 8000, f"Expected 8000 points, got {len(series)} in {ts_file}"

                # stats features: second column values
                stats_vals = pd.read_csv(stats_file, header=None)[1].values.astype(np.float32)

                self.data.append((series, stats_vals))
                self.targets.append(class_idx)

        self.targets = np.array(self.targets)

        if self.mode == 'contrastive':
            self._create_pairs()

    def _create_pairs(self):
        indices_by_class = [np.where(self.targets == i)[0] for i in range(len(self.class_names))]
        self.pairs = []

        for class_idx in range(len(self.class_names)):
            class_indices = indices_by_class[class_idx]
            sample_size = max(1, int(len(class_indices) * self.pair_sample_ratio))
            sampled_indices = random.sample(list(class_indices), sample_size)

            # Positive pairs
            pos_pairs = list(combinations(sampled_indices, 2))
            self.pairs.extend([(i1, i2, 1) for i1, i2 in pos_pairs])

            # Negative pairs
            for other_class_idx in range(class_idx + 1, len(self.class_names)):
                other_indices = random.sample(list(indices_by_class[other_class_idx]), sample_size)
                for i1, i2 in zip(sampled_indices, other_indices):
                    self.pairs.append((i1, i2, 0))

        random.shuffle(self.pairs)

    def __len__(self):
        if self.mode == 'contrastive':
            return len(self.pairs)
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'contrastive':
            idx1, idx2, label = self.pairs[idx]
            ts1, stats1 = self.data[idx1]
            ts2, stats2 = self.data[idx2]
            return (
                torch.tensor(ts1, dtype=torch.float32),
                torch.tensor(stats1, dtype=torch.float32),
                torch.tensor(ts2, dtype=torch.float32),
                torch.tensor(stats2, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32)
            )
        else:
            ts, stats = self.data[idx]
            y = self.targets[idx]
            return (
                torch.tensor(ts, dtype=torch.float32),
                torch.tensor(stats, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long)
            )

# ========== Step 2: Define Model ==========
class LocalCNN(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=20, padding=10),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):
        return self.conv(x).squeeze(-1)

class TransformerEncoder(nn.Module):
    def __init__(self, dim=32, depth=4, heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
    def forward(self, x):
        out = self.transformer(x)
        return out.mean(dim=1)

class SpectrogramCNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=16, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, hidden_dim, kernel_size=16, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.hidden_dim = hidden_dim
    def forward(self, x):
        feat = self.conv(x)
        return feat.view(x.size(0), -1)

class DualPathModel(nn.Module):
    def __init__(self, hidden_dim=32, n_classes=4, stats_feature_dim=32, class_names=None):
        super().__init__()
        self.localcnn = LocalCNN(1, hidden_dim)
        self.transformer = TransformerEncoder(dim=hidden_dim)
        self.spec_cnn = SpectrogramCNN(hidden_dim=hidden_dim)
        fusion_dim = hidden_dim * 2 + stats_feature_dim
        self.fc_sequential = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.use_classifier = False
        self.class_names = class_names

    def forward(self, x, stats_feature):
        B = x.size(0)
        chunks = x.view(B, 1, -1)
        local_feat = self.localcnn(chunks)
        trans_feat = self.transformer(local_feat.unsqueeze(1))
        spec = torch.stft(x, n_fft=64, hop_length=16, return_complex=True, window=torch.ones(64))  # 先使用矩形窗
        spec_power = spec.abs() ** 2
        spec_img = spec_power.unsqueeze(1)
        spec_feat = self.spec_cnn(spec_img)
        feat = torch.cat([trans_feat, spec_feat, stats_feature], dim=1)
        feat = self.fc_sequential(feat)
        if self.use_classifier:
            out = self.classifier(feat)
            return out
        return feat

# ========== Step 3: Contrastive Loss ==========
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, feat1, feat2, label):
        dist = F.pairwise_distance(feat1, feat2)
        loss = torch.mean(label * dist ** 2 + (1 - label) * torch.clamp(self.margin - dist, min=0.0) ** 2)
        return loss

# ========== Step 4: Training and Evaluation ==========
def train_model(ts_root, stats_root, save_dir, hidden_dim=256, epochs=100, contrastive_epochs=30, lr=2e-4, 
                batch_size=32, pair_sample_ratio=0.003, test_ratio=0.15, margin=0.72, step_size=16, gamma=0.5):
    contrastive_dataset = TimeSeriesDataset(ts_root, stats_root, pair_sample_ratio=pair_sample_ratio, mode='contrastive')
    supervised_dataset = TimeSeriesDataset(ts_root, stats_root, mode='supervised')

    # Split supervised dataset into train/test
    total_len = len(supervised_dataset)
    test_len = int(total_len * test_ratio)
    train_len = total_len - test_len
    train_set, test_set = random_split(supervised_dataset, [train_len, test_len])

    contrastive_loader = DataLoader(contrastive_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats_dim = supervised_dataset.data[0][1].shape[0]
    model = DualPathModel(hidden_dim=hidden_dim, n_classes=len(supervised_dataset.class_names), stats_feature_dim=stats_dim, class_names=supervised_dataset.class_names).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    supervised_criterion = nn.CrossEntropyLoss()
    contrastive_criterion = ContrastiveLoss(margin=margin)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        if epoch < contrastive_epochs:
            model.use_classifier = False
            data_loader = contrastive_loader
            criterion = contrastive_criterion
        else:
            model.use_classifier = True
            data_loader = train_loader
            criterion = supervised_criterion

        for batch in data_loader:
            optimizer.zero_grad()
            if epoch < contrastive_epochs:
                ts1, stats1, ts2, stats2, labels = batch
                ts1, stats1, ts2, stats2, labels = ts1.to(device), stats1.to(device), ts2.to(device), stats2.to(device), labels.to(device)
                feat1 = model(ts1, stats1)
                feat2 = model(ts2, stats2)
                loss = criterion(feat1, feat2, labels)
            else:
                ts, stats, yb = batch
                ts, stats, yb = ts.to(device), stats.to(device), yb.to(device)
                out = model(ts, stats)
                loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(data_loader)
        phase = "Contrastive" if epoch < contrastive_epochs else "Supervised"
        print(f"Epoch {epoch+1}: {phase} avg_loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        # Validation every 8 epochs after supervised begins
        if epoch >= contrastive_epochs and (epoch - contrastive_epochs + 1) % 8 == 0:
            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for ts, stats, yb in test_loader:
                    ts, stats, yb = ts.to(device), stats.to(device), yb.to(device)
                    out = model(ts, stats)
                    loss = supervised_criterion(out, yb)
                    val_loss += loss.item()
                    preds = out.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            val_loss /= len(test_loader)
            acc = correct / total
            print(f"Validation: loss={val_loss:.4f}, acc={acc:.4f}")

            # Save checkpoint
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                'model_state': model.state_dict(),
                'class_names': model.class_names,
                'stats_dim': stats_dim
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

# ========== Step 5: Inference Function ==========
def use_model(model_path, ts_list, stats_list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)
    class_names = ckpt['class_names']
    stats_dim = ckpt['stats_dim']
    model = DualPathModel(hidden_dim=256, n_classes=len(class_names), stats_feature_dim=stats_dim, class_names=class_names).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.use_classifier = True
    model.eval()

    ts = torch.tensor(ts_list, dtype=torch.float32).unsqueeze(0).to(device)
    stats = torch.tensor(stats_list, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(ts, stats)
        pred = out.argmax(dim=1).item()
        return class_names[pred]

if __name__ == "__main__":
    # train_model(
    #     ts_root=r".\dataset_final",  # Replace with actual path
    #     stats_root=r".\dataset_final_features_selected",  # Replace with actual path
    #     save_dir=r".\model_ckpt",  # Replace with actual path
    #     hidden_dim=256,
    #     epochs=500,
    #     contrastive_epochs=42,
    #     lr=2e-4,
    #     batch_size=32,
    #     pair_sample_ratio=0.027,
    #     test_ratio=0.15,
    #     margin=0.72,
    #     step_size=16,
    #     gamma=0.7
    # )
    ts = pd.read_csv(r".\dataset_final\OR\OR-20.csv", usecols=[0], header=None).squeeze("columns").tolist()
    stats = pd.read_csv(r".\dataset_final_features_selected\OR\OR-20.csv", usecols=[1], header=None).squeeze("columns").tolist()
    ts_list = list(map(float, ts))
    stats_list = list(map(float, stats))
    pred = use_model(
        model_path=r".\model_ckpt\checkpoint_epoch386.pt",  # Replace with actual path
        ts_list=ts_list,
        stats_list=stats_list
    )
    print("Predicted class:", pred)
