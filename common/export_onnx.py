# export_onnx.py
import argparse
import paddle
from common.config import load_config
from common.model import SoundClassifier, build_backbone

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./checkpoints")
    ap.add_argument("--out", default="SoundClassifier")
    ap.add_argument("--frames", type=int, default=1701)
    args = ap.parse_args()

    cfg = load_config(args.ckpt_dir)

    model = SoundClassifier(build_backbone(pretrained=False), num_class=len(cfg.label_list), dropout=cfg.dropout)
    model.set_state_dict(paddle.load(f"{args.ckpt_dir}/{cfg.ckpt_name}"))
    model.eval()

    input_spec = paddle.static.InputSpec(shape=[None, args.frames, cfg.feat_conf["n_mels"]], dtype="float32")
    paddle.onnx.export(model, args.out, input_spec=[input_spec])
    print(f"Exported ONNX to: {args.out}.onnx")

if __name__ == "__main__":
    main()
