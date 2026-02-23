import torch
import torch.nn as nn
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.features(x)
def cnn_to_tokens(x):
    B, C, H, W = x.shape
    x = x.view(B, C, H * W)
    x = x.permute(0, 2, 1)
    return x
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
class CNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        d_model=128,
        num_heads=4,
        ff_dim=256,
        num_layers=2
    ):
        super().__init__()
        self.cnn = CNNBackbone()
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = cnn_to_tokens(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
model = CNNTransformer(num_classes=5)
dummy_image = torch.randn(2, 3, 224, 224)
output = model(dummy_image)
print(output.shape)