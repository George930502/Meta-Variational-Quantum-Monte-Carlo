import torch
import torch.nn as nn
import numpy as np

class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        # Use std=0.05 for initialization as per the paper
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.05)
        self.v_bias = nn.Parameter(torch.zeros(num_visible))
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))
    
    def forward(self, v):
        v = v.float()
        wx = torch.matmul(v, self.W)
        log_prob = torch.sum(v * self.v_bias, dim=-1) + torch.sum(nn.functional.softplus(wx + self.h_bias), dim=-1)
        return log_prob

class CNN(nn.Module):
    def __init__(self, num_spins, depth):
        super(CNN, self).__init__()
        self.num_spins = num_spins
        self.L = int(np.sqrt(num_spins))
        if self.L * self.L != num_spins:
            raise ValueError("CNN model requires a square number of spins.")
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=depth, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(depth * self.L * self.L, num_spins)

    def forward(self, v):
        v = v.float()
        batch_size = v.shape[0]
        v_reshaped = v.view(batch_size, 1, self.L, self.L)
        x = torch.relu(self.conv1(v_reshaped))
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        #log_prob = torch.sum(nn.functional.softplus(x), dim=-1)
        log_prob = torch.sum(x, dim=-1)
        return log_prob

def get_model(config):
    model_type = config['experiment']['model_type']
    num_spins = config['problem_params']['num_spins']
    if model_type == 'RBM':
        alpha = config['model_params']['rbm_alpha']
        num_hidden = int(alpha * num_spins)
        return RBM(num_spins, num_hidden)
    elif model_type == 'CNN':
        depth = config['model_params']['cnn_depth']
        return CNN(num_spins, depth)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")