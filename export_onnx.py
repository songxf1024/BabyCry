import os
import paddle

from common.config import load_config
from common.model import SoundClassifier, build_backbone

def main():
    ckpt_dir = "./checkpoints"
    out_prefix = "./checkpoints/SoundClassifier"   # 会生成 SoundClassifier.onnx

    cfg = load_config(ckpt_dir)

    model = SoundClassifier(
        build_backbone(pretrained=False),
        num_class=len(cfg.label_list),
        dropout=cfg.dropout,
    )
    model.set_state_dict(paddle.load(os.path.join(ckpt_dir, cfg.ckpt_name)))
    model.eval()

    x_spec = paddle.static.InputSpec(
        shape=[None, None, cfg.feat_conf["n_mels"]],
        dtype="float32",
        name="x",
    )

    # opset 推荐 11（兼容性一般更好；paddle.onnx.export 支持 9/10/11）:contentReference[oaicite:0]{index=0}
    paddle.onnx.export(model, out_prefix, input_spec=[x_spec], opset_version=11)
    print("Exported:", out_prefix + ".onnx")

if __name__ == "__main__":
    main()
