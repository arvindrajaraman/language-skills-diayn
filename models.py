import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import ResidualBlock, make_residual_layer

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=512):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class SkillDiscriminatorNetwork(nn.Module):
    def __init__(self, state_size, skill_size, seed, fc1_units=512, fc2_units=512):
        super(SkillDiscriminatorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.skill_size = skill_size
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(fc2_units, skill_size)

    def forward(self, state):
        x = state
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(F.gelu(self.fc2(x)))
        return F.softmax(self.fc3(x), dim=1)
    
class SkillConvDiscriminatorNetwork(nn.Module):
    def __init__(self, state_shape, skill_size, seed, fc1_units=512, fc2_units=512):
        super(SkillConvDiscriminatorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.height, self.width, self.n_channels = state_shape
        assert self.height == self.width, "Code is not optimized for non-square images. You may comment this assertion out."
        self.skill_size = skill_size

        self.res1 = make_residual_layer(self.height,     self.height,     3)
        self.res2 = make_residual_layer(self.height,     self.height * 2, 4, stride=2)
        self.res3 = make_residual_layer(self.height * 2, self.height * 4, 6, stride=2)
        self.res4 = make_residual_layer(self.height * 4, self.height * 8, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(self.height * 8, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, skill_size)

    def forward(self, state):
        # Residual layers
        if state.dim() == 3:
            x = state.unsqueeze(0)
        else:
            x = state

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        # Average pool and flatten
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        if state.dim() == 3:
            x = x.squeeze()

        return x

class QSkillNetwork(nn.Module):
    """Actor (Policy) Model with skill input."""

    def __init__(self, state_size, action_size, skill_size, seed, fc1_units=512, fc2_units=512):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            skill_size (int): Dimension of skill
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QSkillNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.skill_size = skill_size

        self.fc1 = nn.Linear(state_size + skill_size, fc1_units)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state, skill):
        """Build a network that maps state -> action values."""
        inp = torch.cat((state, skill), dim=1)
        x = self.dropout1(F.gelu(self.fc1(inp)))
        x = self.dropout2(F.gelu(self.fc2(x)))
        return self.fc3(x)
    
class QConvSkillNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_shape, action_size, skill_size, seed, fc1_units=512, fc2_units=512):
        """Initialize parameters and build model.
        Params
        ======
            state_shape (int): tuple of state_shape (height, width, n_channels)
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QConvSkillNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.height, self.width, self.n_channels = state_shape
        assert self.height == self.width, "Code is not optimized for non-square images. You may comment this assertion out."
        self.action_size = action_size
        self.skill_size = skill_size

        self.res1 = make_residual_layer(self.height,     self.height,     3)
        self.res2 = make_residual_layer(self.height,     self.height * 2, 4, stride=2)
        self.res3 = make_residual_layer(self.height * 2, self.height * 4, 6, stride=2)
        self.res4 = make_residual_layer(self.height * 4, self.height * 8, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear((self.height * 8) + skill_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state, skill):
        """Build a network that maps state -> action values."""
        if state.dim() == 3:
            x = state.unsqueeze(0)
        else:
            x = state

        # Residual layers
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        # Average pool and flatten
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        skill_tiled = skill.tile((x.shape[0], 1))
        x = torch.cat((x, skill_tiled), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
