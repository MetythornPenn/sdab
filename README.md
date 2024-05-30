# Sdab

#### Khmer Automatic Speech Recognition

 
Sdab is a Python package for Automatic Speech Recognition with focus on Khmer language. It have offline khmer automatic speech recognition model from my Pretrain Model and other that using Wav2Vec2 model.

License: [Apache-2.0 License](https://github.com/MetythornPenn/sdab/blob/main/LICENSE)

Pretrain Model: [Huggingface](https://huggingface.co/metythorn/khmer-asr-openslr)

## Installation


#### Install from PyPI
```sh
pip install sdab
```

#### Install from source

```sh

# clone repo 
git clone https://github.com/MetythornPenn/sdab.git

# install lib from source
pip install -e .
```

## Usage

#### Download sample audio

```bash
wget -O audio.wav https://github.com/MetythornPenn/sdab/blob/main/sample/audio.wav
```

#### Python API

```python
from sdab import Sdab

file_path = "audio.wav"
model_name = "metythorn/khmer-asr-openslr"  # or local directory path

sdab = Sdab( file_path = file_path, model_name = model_name)
print(sdab.result)

# result : ស្ពានកំពងចំលងអ្នកលើងនៅព្រីវែញជាស្ពានវេញជាងគេសក្នុងព្រសរាជាអាចកម្ពុជា
```

- `file_path`: path of audio file
- `model_name` : pretrain model path from `huggingface` or `local`
- `device` : should be `cpu` or `cuda` but I use `cpu` by default
- `tokenized`: show `[PAD]` in output, `False` by default
- `return`: Khmer text from ASR

## Reference 
- Inspired by [Techcast](https://www.youtube.com/watch?v=ekhFo-6JzLQ&t=28s)
- Khmer word segmentation from SeangHay [khmercut](https://github.com/seanghay/khmercut.git) | [khmersegment](https://github.com/seanghay/khmersegment)
- Wav2Vec2 from Facebook [Wav2Vec2](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md)
