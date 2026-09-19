#cnn+dataloader - made 94 outputneurons for 94 diag, 12 leads, 5000 samples, batchsize=4

import torch
import torch.nn as nn



class ECGCNN(nn.Module):
    def __init__(self, num_classes=94):
        super().__init__()
        #feature extractor, pattern learnig, reLU

        self.features = nn.Sequential(
            nn.Conv1d(12,32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            #convu bklock 2   featurechannel(fc)=dooble
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            #3rd block , triple
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            #variable to value per fc
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):

        # Pass ECG signal through convolution layers
        x = self.features(x)

        x = x.squeeze(-1)

        return self.classifier(x)
        

if __name__ == "__main__":
    model = ECGCNN()

    test_input = torch.randn(4, 12, 5000)
    output = model(test_input)

    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)





#op was: 
'''
Input shape: torch.Size([4, 12, 5000]) - for ref - batchsize=4, 12 leads, 5000 samples
Output shape: torch.Size([4, 94])   - 4 leads, 94 diag


'''