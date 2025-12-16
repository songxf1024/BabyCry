# common/feat.py
import paddle
from paddle.audio.features import LogMelSpectrogram

def build_feature_extractor(feat_conf: dict) -> LogMelSpectrogram:
    return LogMelSpectrogram(**feat_conf)

def wav_to_feats(feature_extractor: LogMelSpectrogram, wav_batch: paddle.Tensor) -> paddle.Tensor:
    """
    wav_batch: [B, T]
    return:    [B, Tm, n_mels]  (给 cnn14 前会再 unsqueeze(1))
    """
    feats = feature_extractor(wav_batch)      # 常见输出: [B, n_mels, frames]
    feats = paddle.transpose(feats, [0, 2, 1])  # -> [B, frames, n_mels]
    return feats
