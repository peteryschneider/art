import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

torch.manual_seed(seed)

def load_data(data_dir):
    raw_data = {'states': torch.load(os.path.join(data_dir, 'torch_states.pth')),
                'actions': torch.load(os.path.join(data_dir, 'torch_actions.pth')),
                'rtgs': torch.load(os.path.join(data_dir, 'torch_rtgs.pth')),
                'ctgs': torch.load(os.path.join(data_dir, 'torch_ctgs.pth')),
                'attention_mask': torch.load(os.path.join(data_dir, 'torch_masks.pth')),
                'timesteps': torch.load(os.path.join(data_dir, 'torch_timesteps.pth')),
                }
    return raw_data

def train_val_split(tensor, train_ratio=0.8):
    num_samples = tensor.size(0)
    num_train = int(num_samples * train_ratio)
    indices = torch.randperm(num_samples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    return train_indices, val_indices

class RpodDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.max_len = self.data['states'].shape[1]

    def __len__(self):
        return len(self.data['states'])

    def __getitem__(self, idx):
        return self.data['states'][idx, :, :], self.data['actions'][idx, :, :], self.data['rtgs'][idx, :, :], self.data['ctgs'][idx, :, :], self.data['attention_mask'][idx, :], self.data['timesteps'][idx, :], idx

    def get_data_size(self):
        return len(self.data['states'])

def normalize_data(data, indices, timesteps=None, norm_stats=None):
    if timesteps is None:
        timesteps = data['timesteps'].shape[-1]

    torch_states_filtered = data['states'][indices , -timesteps:, :]
    torch_actions_filtered = data['actions'][indices, -timesteps:, :]
    torch_rtgs_filtered = data['rtgs'][indices, -timesteps:]
    torch_ctgs_filtered = data['ctgs'][indices, -timesteps:]

    torch_masks_filtered = data['attention_mask'][indices, -timesteps:]
    torch_timesteps_filtered = data['timesteps'][indices, -timesteps:]

    # Get normalization stats for train data
    # For validation, used provided stats from train data
    if norm_stats is None:
        norm_stats = {}

        expanded_mask = torch_masks_filtered.unsqueeze(-1).expand_as(torch_states_filtered)
        norm_stats['states_mean'] = (expanded_mask * torch_states_filtered).mean(dim=(0, 1))
        norm_stats['states_std'] = ((expanded_mask * torch_states_filtered).std(dim=(0, 1)) + 1e-6)

        expanded_mask = torch_masks_filtered.unsqueeze(-1).expand_as(torch_actions_filtered)
        norm_stats['actions_mean'] = (expanded_mask * torch_actions_filtered).mean(dim=(0, 1))
        norm_stats['actions_std'] = ((expanded_mask * torch_actions_filtered).std(dim=(0, 1)) + 1e-6)

        norm_stats['rtgs_mean'] = (torch_masks_filtered * torch_rtgs_filtered).mean(dim=(0, 1))
        norm_stats['rtgs_std'] = ((torch_masks_filtered * torch_rtgs_filtered).std(dim=(0, 1)) + 1e-6)

        norm_stats['ctgs_mean'] = (torch_masks_filtered * torch_ctgs_filtered).mean(dim=(0, 1))
        norm_stats['ctgs_std'] = ((torch_masks_filtered * torch_ctgs_filtered).std(dim=(0, 1)) + 1e-6)

    # Apply normalization
    torch_states_filtered = ((torch_states_filtered - norm_stats['states_mean']) / (norm_stats['states_std'] + 1e-6))
    torch_actions_filtered= ((torch_actions_filtered - norm_stats['actions_mean']) / (norm_stats['actions_std'] + 1e-6))
    torch_rtgs_filtered = ((torch_rtgs_filtered - norm_stats['rtgs_mean']) / (norm_stats['rtgs_std'] + 1e-6))
    torch_ctgs_filtered = ((torch_ctgs_filtered - norm_stats['ctgs_mean']) / (norm_stats['ctgs_std'] + 1e-6))

    torch_rtgs_filtered = torch.unsqueeze(torch_rtgs_filtered, -1)
    torch_ctgs_filtered = torch.unsqueeze(torch_ctgs_filtered, -1)

    normalized_data = {'states': torch_states_filtered,
                       'actions': torch_actions_filtered,
                       'rtgs': torch_rtgs_filtered,
                       'ctgs': torch_ctgs_filtered,
                       'attention_mask': torch_masks_filtered,
                       'timesteps': torch_timesteps_filtered
                       }
    return normalized_data, norm_stats

def create_dataloader(data, batch_size=10, shuffle=True):
    rpod_dataset = RpodDataset(data)
    data_loader = DataLoader(
        rpod_dataset,
        shuffle=shuffle,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=0,
    )
    return data_loader
