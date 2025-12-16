# infer.py
import argparse
from paddleaudio.backends.soundfile_backend import soundfile_load
import paddle
import paddle.nn.functional as F
from common.config import load_config
from common.feat import build_feature_extractor, wav_to_feats
from common.model import SoundClassifier, build_backbone
from common.collate import _fix_length_1d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./checkpoints")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.ckpt_dir)
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")

    feature_extractor = build_feature_extractor(cfg.feat_conf)

    model = SoundClassifier(build_backbone(pretrained=False), num_class=len(cfg.label_list), dropout=cfg.dropout)
    state = paddle.load(f"{args.ckpt_dir}/{cfg.ckpt_name}")
    model.set_state_dict(state)
    model.eval()

    y, sr = soundfile_load(args.wav, sr=cfg.sample_rate, mono=True, normal=True)
    wav = paddle.to_tensor(y, dtype="float32")
    # target_len = int(cfg.sample_rate * 30.0)
    # wav = _fix_length_1d(wav, target_len, random_crop=False)
    wav = wav.unsqueeze(0)  # [1, T_fixed]


    with paddle.no_grad():
        feats = wav_to_feats(feature_extractor, wav)
        logits = model(feats)
        probs = F.softmax(logits, axis=1).numpy()[0]

    idxs = probs.argsort()[-args.topk:][::-1]
    print(f"[{args.wav}]")
    for i in idxs:
        print(f"{cfg.label_list[i]}: {probs[i]:.5f}")

if __name__ == "__main__":
    main()
