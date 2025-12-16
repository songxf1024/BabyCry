# train.py
import os
import paddle
import paddle.nn as nn
from paddleaudio.utils import logger
from common.config import Config, save_config
from common.data import CryDataset
from common.feat import build_feature_extractor, wav_to_feats
from common.model import SoundClassifier, build_backbone
from common.collate import collate_wav

def main():
    cfg = Config()
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")

    train_ds = CryDataset(cfg.train_dir, cfg.label_list, cfg.sample_rate)

    train_loader = paddle.io.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        return_list=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: collate_wav(b, sample_rate=cfg.sample_rate, clip_seconds=cfg.clip_seconds, random_crop=True),
    )

    feature_extractor = build_feature_extractor(cfg.feat_conf)

    model = SoundClassifier(
        backbone=build_backbone(pretrained=True),
        num_class=len(cfg.label_list),
        dropout=cfg.dropout,
    )

    optimizer = paddle.optimizer.Adam(learning_rate=cfg.lr, parameters=model.parameters())
    criterion = nn.CrossEntropyLoss()

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    save_config(cfg)

    log_freq = 10
    steps_per_epoch = len(train_loader)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        avg_loss, num_corrects, num_samples = 0.0, 0, 0

        for step, (wav, labels) in enumerate(train_loader, start=1):
            # wav: [B, T]
            feats = wav_to_feats(feature_extractor, wav)

            logits = model(feats)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            avg_loss += float(loss.item())
            preds = paddle.argmax(logits, axis=1)
            num_corrects += int((preds == labels).numpy().sum())
            num_samples += int(labels.shape[0])

            if step % log_freq == 0 or step == steps_per_epoch:
                logger.train(
                    f"Epoch={epoch}/{cfg.epochs} Step={step}/{steps_per_epoch} "
                    f"loss={avg_loss/log_freq:.4f} acc={num_corrects/max(num_samples,1):.4f} lr={optimizer.get_lr():.6f}"
                )
                avg_loss, num_corrects, num_samples = 0.0, 0, 0

    paddle.save(model.state_dict(), os.path.join(cfg.ckpt_dir, cfg.ckpt_name))
    logger.train(f"Saved: {os.path.join(cfg.ckpt_dir, cfg.ckpt_name)}")

if __name__ == "__main__":
    main()
