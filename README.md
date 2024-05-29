# PyThaiASR

Python Thai Automatic Speech Recognition

 <a href="https://pypi.python.org/pypi/pythaiasr"><img alt="pypi" src="https://img.shields.io/pypi/v/pythaiasr.svg"/></a><a href="https://opensource.org/licenses/Apache-2.0"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/></a><a href="https://github.com/MetythornPenn/sdab">

Sdab is a Python package for Automatic Speech Recognition with focus on Khmer language. It have offline khmer automatic speech recognition model from my Pretrain Model and other that using Wav2Vec2 model.

License: [Apache-2.0 License](https://github.com/PyThaiNLP/pythaiasr/blob/main/LICENSE)

Google Colab: [Link Google colab](https://colab.research.google.com/drive/1zHt3GoxXWCaNSMRzE5lrvpYm9RolcxOW?usp=sharing)

Model homepage: https://huggingface.co/airesearch/wav2vec2-large-xlsr-53-th

## Install

```sh
pip install -e .
```

## Usage

```python
from sdab import Sdab

file_path = "path_to_your_audio_file.wav"
model_name = "metythorn/khmer-asr-openslr"  # or local directory path
sdab = Sdab( file_path = file_path, model_name = model_name ,device='cpu', tokenized= False)
print(sdab.result)

```


- file_path: path of sound file
- model_name : pretrain model path from huggingface or local
- device : Should be CPU or CUDA but I use CPU by default
- tokenized: show [PAD] in output
- return: Khmer text from ASR