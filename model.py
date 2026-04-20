import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SiameseCNN(nn.Module):
    """
    A Siamese architecture for Change Detection.
    Extracts features from both images using shared weights, 
    then computes the absolute difference of features and decodes to a mask.
    """
    def __init__(self):
        super(SiameseCNN, self).__init__()
        
        # Shared Encoder (Siamese branches)
        self.encoder1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(32, 64)
        
        # Decoder (Takes the absolute difference of features)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(32, 32)
        
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(16, 16)
        
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward_once(self, x):
        # Extract features
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        return e3

    def forward(self, img1, img2):
        # Pass both images through the shared encoder
        feat1 = self.forward_once(img1)
        feat2 = self.forward_once(img2)
        
        # Compute distance metric (absolute difference)
        diff = torch.abs(feat1 - feat2)
        
        # Decode the difference to a pixel-wise mask
        d2 = self.upconv2(diff)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = self.decoder1(d1)
        
        out = self.out_conv(d1)
        
        # Output probability map (0 to 1)
        return torch.sigmoid(out)

if __name__ == "__main__":
    # Test model
    model = SiameseCNN()
    b = torch.randn(2, 3, 256, 256)
    r = torch.randn(2, 3, 256, 256)
    out = model(b, r)
    print("Output shape:", out.shape) # Should be [2, 1, 256, 256]
