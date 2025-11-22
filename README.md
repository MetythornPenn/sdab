# Sdab

#### Khmer Automatic Speech Recognition (Whisper + Wav2Vec2)

Sdab is a lightweight helper around Hugging Face ASR models with a focus on Khmer language. It can load sequence-to-sequence Whisper checkpoints (default) or CTC-style Wav2Vec2 models, convert audio to the expected format, and return a transcription in a single call.

- License: [Apache-2.0](https://github.com/MetythornPenn/sdab/blob/main/LICENSE)
- Default Whisper model: [metythorn/whisper-large-v3](https://huggingface.co/metythorn/whisper-large-v3)
- Optional turbo Whisper model: [metythorn/whisper-large-v3-turbo](https://huggingface.co/metythorn/whisper-large-v3-turbo)
- Example Wav2Vec2 model: [metythorn/wav2vec2-xls-r-300m](https://huggingface.co/metythorn/wav2vec2-xls-r-300m)

## Features
- 🔁 Automatically detects Whisper vs Wav2Vec2 when you pass a Hugging Face repo ID or local directory.
- 🎧 Handles loading, mono conversion, and resampling to 16 kHz with `torchaudio`.
- ⚙️ Lets you pick CPU/GPU device and numerical precision to match your hardware.
- 🧪 Includes a sample audio clip for quick testing.

## Installation

> It is highly recommended to work inside a virtual environment.

```sh
python -m pip install --upgrade pip
pip install torch torchaudio transformers soundfile
pip install sdab
```

GPU users should install the right `torch`/`torchaudio` binaries for their CUDA version as described on https://pytorch.org/get-started/locally/.

To install from source:

```sh
git clone https://github.com/MetythornPenn/sdab.git
cd sdab
pip install -e .
```

## Quick Start

Download the bundled sample audio (Khmer speech, 16 kHz WAV):

```bash
wget -O audio.wav https://raw.githubusercontent.com/MetythornPenn/sdab/main/sample/audio.wav
```

### Whisper (default)

```python
from sdab import Sdab

sd = Sdab("audio.wav")  # defaults to metythorn/whisper-large-v3
print(sd.transcribe())
```

### Explicit Whisper model

```python
from sdab import Sdab

sd = Sdab(
    "audio.wav",
    model_name="metythorn/whisper-large-v3",
    device="cuda:0",         # or "cpu"
)
print(sd.transcribe())
```

Need the faster turbo checkpoint? Provide it explicitly:

```python
sd = Sdab("audio.wav", model_name="metythorn/whisper-large-v3-turbo")
print(sd.transcribe())
```

### Wav2Vec2 / CTC model

```python
from sdab import Sdab

sd = Sdab(
    "audio.wav",
    model_name="metythorn/wav2vec2-xls-r-300m",
    model_type="wav2vec2",  # optional; Sdab will infer from the model name
)
print(sd.transcribe())
```

### Important parameters
- `file_path`: Path to your WAV/FLAC/etc. file.
- `model_name`: Hugging Face repo ID or local directory with the pretrained model.
- `model_type`: Force `"whisper"` or `"wav2vec2"` if autodetect is not correct.
- `device`: `"cpu"` or any PyTorch device string (for example `"cuda:0"`).
- `torch_dtype`: Override the dtype (defaults to `float32` on CPU and `float16` on CUDA).

## Tips
- Whisper expects mono 16 kHz input; Sdab automatically downsamples and squeezes channels.
- Models are downloaded from Hugging Face the first time you reference them. Keep an eye on cache size in `~/.cache/huggingface`.
- For long recordings consider chunking/streaming outside of Sdab to stay within GPU memory.
- Results are returned from `sd.transcribe()` directly; the class no longer stores a separate `sd.result`.
- Errors while loading a model are wrapped in a helpful `RuntimeError` with the model name.

## References
- Inspired by [Bong Vitou Phy](https://huggingface.co/vitouphy/wav2vec2-xls-r-300m-khmer) and the accompanying [Techcast episode](https://www.youtube.com/watch?v=ekhFo-6JzLQ&t=28s).
- Khmer word segmentation libraries from SeangHay: [khmercut](https://github.com/seanghay/khmercut.git) and [khmersegment](https://github.com/seanghay/khmersegment).
- Whisper: [paper](https://cdn.openai.com/papers/whisper.pdf) | [Hugging Face models](https://huggingface.co/models?search=whisper).
- Wav2Vec2 paper and resources from Facebook AI Research: [fairseq examples](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md).
