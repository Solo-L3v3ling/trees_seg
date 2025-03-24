import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    """Double convolution block: (conv -> ReLU) × 2"""
    
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


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse to use from last to first
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsample
            skip = skip_connections[idx//2]
            
            # Handle cases where dimensions don't match perfectly
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
                
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx+1](x)
            
        return self.final_conv(x)


if __name__ == "__main__":
    # Example usage
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn((1, 3, 240, 240))
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(model)
    summary(model, (3, 240, 240))
    plt.imshow(x.squeeze().permute( 1,2,0).detach().numpy())
    plt.show()
    plt.imshow(output.squeeze().detach().numpy())
    plt.show()
