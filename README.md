# BabyCry
Baby cry sound classification using PaddleSpeech (PANNs CNN14 backbone).

- **Labels**: awake / diaper / hug / hungry / sleepy / uncomfortable
- **Training**: PaddlePaddle + PaddleSpeech
- **Inference**: Paddle (native) + ONNXRuntime (optional)

## Project Structure
```bash
BabyCry/
├─ common/                # shared config / dataset / feature / model
├─ train.py               # training entry
├─ infer.py               # paddle inference
├─ export_onnx.py         # export ONNX
├─ infer_onnx.py          # ONNXRuntime inference
├─ dataset/               # (download separately)
└─ checkpoints/           # (generated or downloaded)
```


## Setup
1. Install dependencies.

```bash
conda create -n cry python=3.10 -y
conda activate cry

sudo apt-get install -y cmake g++ make protobuf-compiler libprotobuf-dev build-essential python3-dev
python -m pip install -U pip setuptools wheel

pip install -r requirements.txt
```

> GPU is recommended for training.  

1. Download code.

```bash
git clone https://github.com/songxf1024/BabyCry.git
cd BabyCry
```

## Dataset
Download [dataset](https://github.com/songxf1024/BabyCry/releases/tag/dataset) and unzip.  

Expected format:  

```bash
dataset/train/
  awake/*.wav
  diaper/*.wav
  hug/*.wav
  hungry/*.wav
  sleepy/*.wav
  uncomfortable/*.wav

dataset/test/
  test_*.wav
```

> Note:   
> Audio files can have different sample rates / lengths. The code will resample and pad/crop automatically during training/inference.


## Training

1. (Optional) Edit configs in `common/config.py`.

2. Start training.

```bash
python train.py
```

Outputs will be saved to:  
- ./checkpoints/SoundClassifier.pdparams
- ./checkpoints/config.json


The output might be:  
> [2025-12-16 21:08:19,432] [   TRAIN] - Epoch=50/50 Step=15/15 loss=0.0748 acc=0.9568 lr=0.000100  
> [2025-12-16 21:08:20,119] [   TRAIN] - Saved: ./checkpoints/SoundClassifier.pdparams  
> [2025-12-16 21:08:20,119] [   TRAIN] - Saved: ./checkpoints/SoundClassifier.pdparams  


## Inference (Paddle)
1. Download [checkpoint](https://github.com/songxf1024/BabyCry/releases/tag/checkpoint).

2. Run inference.

```bash
python infer.py --ckpt_dir ./checkpoints --wav ./dataset/test/test_4.wav --topk 3
```

The output might be:   
> [./dataset/test/test_4.wav]  
> sleepy: 0.96702  
> uncomfortable: 0.03200  
> hug: 0.00070   

## ONNX
### Export ONNX

1. Export ONNX from Paddle weights.

```bash
export_onnx.py
```

The ONNX file will be generated at:  
- ./checkpoints/SoundClassifier.onnx

### Inference (ONNXRuntime)

```bash
python infer_onnx.py --ckpt_dir ./checkpoints --wav ./dataset/test/test_4.wav --topk 3
```

If you installed onnxruntime-gpu:  
```bash
python infer_onnx.py --ckpt_dir ./checkpoints --wav ./dataset/test/test_4.wav --topk 3 --provider cuda
```


## TODO
- Deploy to Android Termux.

---

## More

- [【教程】旧手机别丢! 教你做一个哭声/声音检测器](https://xfxuezhang.blog.csdn.net/article/details/155922809)



