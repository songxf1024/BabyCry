# BabyCry
Baby Cry Classification Based on PaddleSpeech


## Usage
1. Install dependencies.

```bash
conda create -n cry python=3.10
conda activate cry

pip install -r requirements.txt
```

2. Download code.

```bash
git clone https://github.com/songxf1024/BabyCry.git

cd BabyCry
```

### Training
> It is recommended to use a GPU for training.

1. Download [dataset](https://github.com/songxf1024/BabyCry/releases/tag/dataset).

2. Modify config in `common/config.py`, optional.

3. Start train.

```bash
python train.py
```

The output might be:  
> [2025-12-16 21:08:06,417] [   TRAIN] - Epoch=49/50 Step=15/15 loss=0.0748 acc=0.9496 lr=0.000100  
> [2025-12-16 21:08:06,417] [   TRAIN] - Epoch=49/50 Step=15/15 loss=0.0748 acc=0.9496 lr=0.000100   
> [2025-12-16 21:08:16,663] [   TRAIN] - Epoch=50/50 Step=10/15 loss=0.1153 acc=0.9625 lr=0.000100  
> [2025-12-16 21:08:16,663] [   TRAIN] - Epoch=50/50 Step=10/15 loss=0.1153 acc=0.9625 lr=0.000100  
> [2025-12-16 21:08:19,432] [   TRAIN] - Epoch=50/50 Step=15/15 loss=0.0748 acc=0.9568 lr=0.000100  
> [2025-12-16 21:08:19,432] [   TRAIN] - Epoch=50/50 Step=15/15 loss=0.0748 acc=0.9568 lr=0.000100  
> [2025-12-16 21:08:20,119] [   TRAIN] - Saved: ./checkpoints/SoundClassifier.pdparams  
> [2025-12-16 21:08:20,119] [   TRAIN] - Saved: ./checkpoints/SoundClassifier.pdparams  


### Inference
1. Download [checkpoint](https://github.com/songxf1024/BabyCry/releases/tag/checkpoint).

2. Start infer.

```bash
python infer.py --ckpt_dir ./checkpoints --wav ./dataset/test/test_4.wav --topk 3
```

The output might be:   
> [./dataset/test/test_4.wav]  
> sleepy: 0.96702  
> uncomfortable: 0.03200  
> hug: 0.00070   




## TODO
- Deploy to Android Termux.

---

## More

- [【教程】旧手机别丢! 教你做一个哭声/声音检测器](https://xfxuezhang.blog.csdn.net/article/details/155922809)



