import torch
import torch.nn as nn


# ==============================================================================
# ORIGINAL ARCHITECTURE (100% PRESERVED FOR FULL BACKWARD COMPATIBILITY)
# ==============================================================================

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
    """
    Original 1D CNN Architecture with residual connections.
    Maintained verbatim for full compatibility with existing checkpoints.
    """
    def __init__(self, num_classes=94):
        super().__init__()

        # Initial feature extraction — 12 leads -> 64 channels
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

        # Global average pool — any length -> fixed size
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


# ==============================================================================
# ADVANCED ARCHITECTURE: ECGResNetSE (Lead-Aware Squeeze-and-Excitation 1D ResNet)
# ==============================================================================

class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation block for 1D signals.
    Recalibrates channel-wise feature responses by explicitly modelling
    interdependencies between leads and temporal feature maps.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.fc(x).unsqueeze(-1)
        return x * weights


class ResNetSEBlock(nn.Module):
    """
    1D Residual Block with extended receptive field kernel (k=15 or 7)
    and Squeeze-and-Excitation channel attention.
    """
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=15):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock1D(out_channels)

        # Shortcut projection if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + residual)
        return out


class ECGResNetSE(nn.Module):
    """
    Lead-Aware Multi-Scale 1D ResNet with Squeeze-and-Excitation attention.
    Designed specifically for 500 Hz 12-lead ECG signals:
    - Large 1D kernels (k=15, 7) capture wide QRS, P, and T wave complexes.
    - Squeeze-and-Excitation adaptively weights informative leads and feature projections.
    - Exactly 94 logits output.
    """
    def __init__(self, in_channels=12, num_classes=94, base_channels=64):
        super().__init__()

        # Stem: capture multi-lead ECG morphology
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 1: 64 channels
        self.layer1 = nn.Sequential(
            ResNetSEBlock(base_channels, base_channels, stride=1, kernel_size=15),
            ResNetSEBlock(base_channels, base_channels, stride=1, kernel_size=15)
        )

        # Stage 2: 128 channels
        self.layer2 = nn.Sequential(
            ResNetSEBlock(base_channels, base_channels * 2, stride=2, kernel_size=11),
            ResNetSEBlock(base_channels * 2, base_channels * 2, stride=1, kernel_size=11)
        )

        # Stage 3: 256 channels
        self.layer3 = nn.Sequential(
            ResNetSEBlock(base_channels * 2, base_channels * 4, stride=2, kernel_size=7),
            ResNetSEBlock(base_channels * 4, base_channels * 4, stride=1, kernel_size=7)
        )

        # Stage 4: 512 channels
        self.layer4 = nn.Sequential(
            ResNetSEBlock(base_channels * 4, base_channels * 8, stride=2, kernel_size=5),
            ResNetSEBlock(base_channels * 8, base_channels * 8, stride=1, kernel_size=5)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(base_channels * 8, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.classifier(x)


# ==============================================================================
# MODEL FACTORY
# ==============================================================================

def get_model(arch="ecg_cnn", num_classes=94):
    """
    Factory function to instantiate models by architecture name.
    Supported:
      - 'ecg_cnn' (original baseline architecture)
      - 'ecg_resnet_se' (advanced 1D ResNet with SE attention)
    """
    arch = arch.lower().strip()
    if arch in ("ecg_cnn", "cnn", "baseline"):
        return ECGCNN(num_classes=num_classes)
    elif arch in ("ecg_resnet_se", "resnet_se", "resnet1d"):
        return ECGResNetSE(in_channels=12, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: '{arch}'. Supported: 'ecg_cnn', 'ecg_resnet_se'")


if __name__ == "__main__":
    for arch_name in ["ecg_cnn", "ecg_resnet_se"]:
        model = get_model(arch_name, num_classes=94)
        x = torch.randn(4, 12, 5000)
        out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"[{arch_name}] Input: {tuple(x.shape)} -> Output: {tuple(out.shape)} | Params: {params:,}")