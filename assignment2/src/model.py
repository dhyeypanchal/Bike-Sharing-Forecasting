import torch
import torch.nn as nn


class DeepBikeNet(nn.Module):
    def __init__(self, input_dim=27, hidden_dims=[256, 128, 64, 32, 16], dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, 1)

    def forward(self, x):
        return self.out(self.net(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Preset network configurations for hyperparameter tuning
ARCHITECTURES = {
    '3_layer': [128, 64, 32],
    '5_layer': [256, 128, 64, 32, 16],
    '5_layer_wide': [512, 256, 128, 64, 32],
    '7_layer': [512, 256, 128, 64, 32, 16, 8],
}


def build_model(arch='5_layer', input_dim=27, dropout_rate=0.2):
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch}")
    return DeepBikeNet(input_dim=input_dim, hidden_dims=ARCHITECTURES[arch], dropout_rate=dropout_rate)


if __name__ == '__main__':
    dummy = torch.randn(64, 27)
    model = build_model('5_layer')
    pred = model(dummy)
    print("5-layer network:", model.hidden_dims)
    print("Output shape:", pred.shape)
    print("Trainable parameters:", model.count_parameters())
