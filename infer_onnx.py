import os
import argparse
import numpy as np
import onnxruntime as ort
import paddle

from common.config import load_config
from common.feat import build_feature_extractor, wav_to_feats
from paddleaudio.backends.soundfile_backend import soundfile_load


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints",
                    help="目录中包含 config.json 和 SoundClassifier.onnx")
    ap.add_argument("--wav", type=str, required=True,
                    help="待预测 wav 文件路径")
    ap.add_argument("--topk", type=int, default=3,
                    help="输出前 topk 类别")
    ap.add_argument("--onnx", type=str, default="",
                    help="onnx 文件路径（不填则默认 ckpt_dir/SoundClassifier.onnx）")
    ap.add_argument("--provider", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                    help="ONNXRuntime provider：auto/cpu/cuda")
    return ap.parse_args()


def build_providers(provider: str):
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # auto：优先 CUDA，失败自动回退 CPU
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def main():
    args = parse_args()

    ckpt_dir = args.ckpt_dir
    wav_path = args.wav
    topk = args.topk

    cfg = load_config(ckpt_dir)

    # 1) 读 wav（自动重采样到训练 sample_rate）
    y, sr = soundfile_load(
        wav_path,
        sr=cfg.sample_rate,
        mono=True,
        normal=True,
        resample_mode="kaiser_fast",
    )
    wav = paddle.to_tensor(y, dtype="float32").unsqueeze(0)  # [1, T]

    # 2) wav -> LogMel 特征（与训练一致）
    feature_extractor = build_feature_extractor(cfg.feat_conf)
    feats = wav_to_feats(feature_extractor, wav)             # [1, frames, n_mels]
    feats_np = feats.numpy().astype(np.float32)

    # 3) ONNXRuntime 推理
    onnx_path = args.onnx if args.onnx else os.path.join(ckpt_dir, "SoundClassifier.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    sess = ort.InferenceSession(
        onnx_path,
        providers=build_providers(args.provider),
    )

    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: feats_np})[0]  # [1, C]
    probs = softmax(logits)[0]

    topk = min(topk, len(cfg.label_list))
    idxs = probs.argsort()[-topk:][::-1]

    print(f"[{wav_path}] (sr={cfg.sample_rate})")
    for i in idxs:
        print(f"{cfg.label_list[i]}: {probs[i]:.5f}")


if __name__ == "__main__":
    main()
