import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)  # skip connection!


class ECGCNN(nn.Module):
    def __init__(self, num_classes=94):
        super().__init__()

        # Initial feature extraction — 12 leads → 64 channels
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Residual blocks — learn deeper patterns
        self.layer1 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # Downsample + expand channels
        self.down1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # Downsample + expand channels
        self.down2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Global average pool — any length → fixed size
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.classifier(x)


if __name__ == "__main__":
    model = ECGCNN()
    test_input = torch.randn(4, 12, 5000)
    output = model(test_input)

    total_params = sum(p.numel() for p in model.parameters())
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)
    print(f"Total parameters: {total_params:,}")