# common/collate.py
import random
import paddle
import paddle.nn.functional as F

def _fix_length_1d(w: paddle.Tensor, target_len: int, random_crop: bool = True) -> paddle.Tensor:
    n = int(w.shape[0])
    if n == target_len:
        return w
    if n > target_len:
        start = random.randint(0, n - target_len) if random_crop else 0
        return w[start:start + target_len]
    # n < target_len -> pad
    return F.pad(w, pad=[0, target_len - n])  # 右侧补零

def collate_wav(
    batch,
    sample_rate: int,
    clip_seconds: float = 5.0,
    random_crop: bool = True,
):
    """
    batch: List[(wav[T], label[])]
    return:
      wavs: [B, T_fixed]
      labels: [B]
    """
    wavs, labels = zip(*batch)

    target_len = int(sample_rate * clip_seconds)
    wavs = [_fix_length_1d(w, target_len, random_crop=random_crop) for w in wavs]
    wavs = paddle.stack(wavs, axis=0)  # [B, T_fixed]

    # labels 是标量 tensor，拼成 [B]
    labels = paddle.concat([l.reshape([1]) for l in labels], axis=0).astype("int64")
    return wavs, labels
