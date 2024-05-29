# Sdab

Khmer Automatic Speech Recognition

 <a href="https://pypi.python.org/pypi/pythaiasr"><img alt="pypi" src="https://img.shields.io/pypi/v/pythaiasr.svg"/></a><a href="https://opensource.org/licenses/Apache-2.0"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/></a><a href="https://github.com/MetythornPenn/sdab">

Sdab is a Python package for Automatic Speech Recognition with focus on Khmer language. It have offline khmer automatic speech recognition model from my Pretrain Model and other that using Wav2Vec2 model.

License: [Apache-2.0 License](https://github.com/PyThaiNLP/pythaiasr/blob/main/LICENSE)

Model homepage: https://huggingface.co/metythorn/khmer-asr-openslr

## Install From Source

```sh
# clone repo 
git clone https://github.com/MetythornPenn/sdab.git

# install lib from source
pip install -e .
```

## Usage

inference code : inference.ipynb

```python
from sdab import Sdab

file_path = "sample/audio.wav"
model_name = "metythorn/khmer-asr-openslr"  # or local directory path
sdab = Sdab( file_path = file_path, model_name = model_name ,device='cpu', tokenized= False)
print(sdab.result)

# result : ស្ពានកំពងចំលងអ្នកលើងនៅព្រីវែញជាស្ពានវេញជាងគេសក្នុងព្រសរាជាអាចកម្ពុជា

```


- file_path: path of sound file
- model_name : pretrain model path from huggingface or local
- device : Should be CPU or CUDA but I use CPU by default
- tokenized: show [PAD] in output
- return: Khmer text from ASR

## Reference 
- Khmer word segmentation from SeangHay [khmercut](https://github.com/seanghay/khmercut.git) | [khmersegment](https://github.com/seanghay/khmersegment)
- Wav2Vec2 from Facebook [Wav2Vec2](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md)
- Inspired by [Thai Dev](https://www.youtube.com/watch?v=ekhFo-6JzLQ&t=28s)
