# common/data.py
from paddleaudio.backends.soundfile_backend import soundfile_load  # ✅带 sr 参数并可自动重采样
import paddle
from paddle.io import Dataset
from pathlib import Path

class CryDataset(Dataset):
    def __init__(self, root_dir: str, label_list: list[str], sample_rate: int):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.label_list = label_list
        self.sample_rate = sample_rate

        self.files, self.labels = [], []
        self._scan()

    def _scan(self):
        for idx, name in enumerate(self.label_list):
            d = self.root_dir / name
            if not d.exists():
                continue
            for p in sorted(d.rglob("*.wav")):
                self.files.append(str(p))
                self.labels.append(idx)
        if not self.files:
            raise RuntimeError(f"No wav files found under: {self.root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        wav_path = self.files[i]
        label = self.labels[i]

        # ✅读取时直接重采样到 self.sample_rate，并转单声道
        y, sr = soundfile_load(
            wav_path,
            sr=self.sample_rate,      # 关键：目标采样率
            mono=True,                # 单声道
            normal=True,
            resample_mode="kaiser_fast"
        )  # docs: soundfile_load(file, sr=..., mono=..., resample_mode=...) :contentReference[oaicite:2]{index=2}

        # 保险起见：确认已经被重采样到目标 sr
        if sr != self.sample_rate:
            # 理论上不会发生，但保留兜底
            raise ValueError(f"{wav_path} resample failed: sr={sr}, expected={self.sample_rate}")

        wav = paddle.to_tensor(y, dtype="float32")           # [T]
        lab = paddle.to_tensor(label, dtype="int64")         # []

        return wav, lab
