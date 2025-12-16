# common/config.py
from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class Config:
    label_list = ['awake', 'diaper', 'hug', 'hungry', 'sleepy', 'uncomfortable']
    train_dir: str = "./dataset/train"
    sample_rate: int = 16000

    # 和预训练 cnn14 对齐
    feat_conf: dict = None
    # 训练时固定音频时长，多截少补
    clip_seconds = 30.0
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-4
    dropout: float = 0.1
    num_workers: int = 2

    # 统一把模型与配置存这里
    ckpt_dir: str = "./checkpoints"
    ckpt_name: str = "SoundClassifier.pdparams"
    cfg_name: str = "config.json"

    def __post_init__(self):
        if self.feat_conf is None:
            self.feat_conf = dict(
                sr=self.sample_rate,
                n_fft=1024,
                hop_length=320,
                window="hann",
                win_length=1024,
                f_min=50.0,
                f_max=14000.0,
                n_mels=64,
            )

def save_config(cfg: Config):
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / cfg.cfg_name).write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_config(ckpt_dir: str, cfg_name: str = "config.json") -> Config:
    p = Path(ckpt_dir) / cfg_name
    d = json.loads(p.read_text(encoding="utf-8"))
    # dataclass 反序列化
    cfg = Config()
    for k, v in d.items():
        setattr(cfg, k, v)
    return cfg
