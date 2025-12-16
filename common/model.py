# common/model.py
import paddle.nn as nn
from paddlespeech.cls.models import cnn14

def build_backbone(pretrained: bool = True):
    # 你原来就是 extract_embedding=True
    return cnn14(pretrained=pretrained, extract_embedding=True)

class SoundClassifier(nn.Layer):
    def __init__(self, backbone, num_class: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.backbone.emb_size, num_class)

    def forward(self, x):
        # x: [B, T, n_mels]
        x = x.unsqueeze(1)      # -> [B, 1, T, n_mels]
        x = self.backbone(x)    # -> [B, emb]
        x = self.dropout(x)
        return self.fc(x)       # -> [B, num_class]
