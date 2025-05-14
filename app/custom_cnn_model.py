import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch
from gym.spaces import Box

_, nn = try_import_torch()


class CustomCNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Assume obs is [C, H, W] — if not, permute in forward
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),  # [3,128,40] → [32,31,9]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # [32,31,9] → [64,14,3]
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute output size after conv to feed into FC
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 40)
            conv_out = self.conv_layers(dummy_input)
            self._conv_out_size = conv_out.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self._conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

        self.value_branch = nn.Sequential(
            nn.Linear(self._conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._value = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]

        # Convert from [B, H, W, C] → [B, C, H, W]
        if x.dim() == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)

        # Convert to float32 and normalize to [0, 1]
        x = x.float() / 255.0
        x = self.conv_layers(x)
        self._value = self.value_branch(x)
        return self.fc(x), state

    @override(ModelV2)
    def value_function(self):
        return self._value.squeeze(1)
