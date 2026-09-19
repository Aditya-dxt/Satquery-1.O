import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialTemporalAttentionModule(nn.Module):
    """
    Attention module to suppress seasonal/lighting noise and focus on structural human developments.
    Applies spatial and channel difference attention between time features.
    """
    def __init__(self, in_channels: int):
        super(SpatialTemporalAttentionModule, self).__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        # Absolute temporal difference
        diff = torch.abs(feat_a - feat_b)
        
        # Channel Attention
        concat_feat = torch.cat([feat_a, feat_b], dim=1)
        c_weight = self.channel_attn(concat_feat)
        diff_c = diff * c_weight
        
        # Spatial Attention
        avg_out = torch.mean(diff_c, dim=1, keepdim=True)
        max_out, _ = torch.max(diff_c, dim=1, keepdim=True)
        s_weight = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        
        return diff_c * s_weight


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d((in_channels // 2) + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNetAttention(nn.Module):
    """
    Temporal Siamese Convolutional Network with ResNet34 Encoder Backbone
    and Spatial-Temporal Attention Difference Modules.
    The primary value is ResNet34_Weights.DEFAULT, which is equivalent to ResNet34_Weights.IMAGENET1K_V1
    hese weights were trained on ImageNet-1K and reproduce the results of the original "Deep Residual Learning for Image Recognition" paper.
    
    Key details include:

    Usage: Pass weights=ResNet34_Weights.DEFAULT or weights='DEFAULT' to torchvision.models.resnet34(). 
    Model Size: The pretrained model contains approximately 21.8 million parameters and has a file size of roughly 83.3 MB.
    Inference Transforms: The class also provides access to the specific image preprocessing transforms required for inference with these pretrained weights. 
    Alternative: Users can also set weights=None to initialize the model with random weights instead of pretrained ones. 
    """
    def __init__(self, pretrained: bool = True):
        super(SiameseUNetAttention, self).__init__()
        
        # Shared Encoder Backbone (ResNet34)
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        ) # Output: [64, H/2, W/2]
        self.maxpool = resnet.maxpool # Output: [64, H/4, W/4]
        
        self.layer1 = resnet.layer1 # [64, H/4, W/4]
        self.layer2 = resnet.layer2 # [128, H/8, W/8]
        self.layer3 = resnet.layer3 # [256, H/16, W/16]
        self.layer4 = resnet.layer4 # [512, H/32, W/32]
        
        # Attention Modules across scales
        self.attn1 = SpatialTemporalAttentionModule(64)
        self.attn2 = SpatialTemporalAttentionModule(128)
        self.attn3 = SpatialTemporalAttentionModule(256)
        self.attn4 = SpatialTemporalAttentionModule(512)
        
        # Decoder Blocks
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        
        # Final Classification Head
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1) # Logits (binary classification)
        )

    def extract_features(self, x: torch.Tensor):
        x0 = self.initial(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        # Pass both images through shared encoder twin
        f_a0, f_a1, f_a2, f_a3, f_a4 = self.extract_features(img_a)
        f_b0, f_b1, f_b2, f_b3, f_b4 = self.extract_features(img_b)
        
        # Apply Spatial-Temporal Attention Differences
        diff1 = self.attn1(f_a1, f_b1)
        diff2 = self.attn2(f_a2, f_b2)
        diff3 = self.attn3(f_a3, f_b3)
        diff4 = self.attn4(f_a4, f_b4)
        
        # UNet Decoding Path
        d4 = self.dec4(diff4, diff3)
        d3 = self.dec3(d4, diff2)
        d2 = self.dec2(d3, diff1)
        d1 = self.dec1(d2, f_a0)
        
        out = self.final_up(d1)
        logits = self.final_conv(out)
        return logits