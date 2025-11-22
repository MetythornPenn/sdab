import os
import warnings
from typing import Optional, Tuple

import torch
import torchaudio

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline as hf_pipeline,
)

warnings.filterwarnings("ignore", category=UserWarning)


class Sdab:
    DEFAULT_WHISPER_MODEL = "metythorn/whisper-large-v3"
    TURBO_WHISPER_MODEL = "metythorn/whisper-large-v3-turbo"

    """Small helper to transcribe a single audio file using either Wav2Vec2 (CTC)
    or Whisper-style sequence-to-sequence models.

    The class will download models from Hugging Face when `model_name` is not
    a local directory. By default it will autodetect Whisper models when the
    model name contains the substring "whisper"; you can override detection
    with `model_type`.
    """

    def __init__(
        self,
        file_path: str,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,  # 'wav2vec2' or 'whisper'
        tokenized: bool = False,
        device: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.file_path = file_path
        self.model_name = model_name or self.DEFAULT_WHISPER_MODEL
        self.tokenized = tokenized
        self.device_str = device
        self.device = torch.device(device)
        self.torch_dtype = self._resolve_dtype(torch_dtype)
        self.model_type = self._resolve_model_type(self.model_name, model_type)

        # model / processor placeholders
        self.model = None
        self.processor = None
        self.pipe = None

        self._load_model_components()

    def _resolve_dtype(self, torch_dtype: Optional[torch.dtype]) -> torch.dtype:
        if torch_dtype is not None:
            return torch_dtype
        return torch.float16 if self.device.type == "cuda" else torch.float32

    def _resolve_model_type(self, model_name: str, model_type: Optional[str]) -> str:
        if model_type is not None:
            return model_type.lower()
        lowered = model_name.lower()
        if "whisper" in lowered or "speechseq2seq" in lowered:
            return "whisper"
        return "wav2vec2"

    def _hf_device_index(self) -> int:
        if self.device.type != "cuda":
            return -1
        # torch.device("cuda") has index None; default to first GPU
        return 0 if self.device.index is None else self.device.index

    def _load_model_components(self) -> None:
        try:
            if self.model_type == "whisper":
                self._load_whisper_components()
            elif self.model_type == "wav2vec2":
                self._load_wav2vec2_components()
            else:
                raise ValueError(f"Unsupported model_type '{self.model_type}'")
        except Exception as exc:
            raise RuntimeError(f"Failed to load model/processor for '{self.model_name}'") from exc

    def _load_whisper_components(self) -> None:
        """Load Whisper-style model and build a HF pipeline."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.pipe = hf_pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=self._hf_device_index(),
        )

    def _load_wav2vec2_components(self) -> None:
        """Load Wav2Vec2 model + processor."""
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self.model.to(self.device)

    def _load_audio(self) -> Tuple:
        """Load audio file, convert to mono and resample to 16 kHz.

        Returns a 1-D numpy array (float32) suitable for HF pipelines.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")

        waveform, sr = torchaudio.load(self.file_path)

        # convert to mono by averaging channels if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)

        return waveform.squeeze().numpy(), 16000

    def transcribe(self) -> str:
        """Run transcription for the configured model and return text."""
        if self.model_type == "whisper":
            return self._transcribe_whisper()
        return self._transcribe_wav2vec2()

    def _transcribe_whisper(self) -> str:
        waveform_np, _ = self._load_audio()
        if self.pipe is None:
            raise RuntimeError("Whisper pipeline not initialized")
        # pipeline returns a dict with 'text'
        try:
            result = self.pipe(waveform_np)
        except Exception as exc:
            raise RuntimeError(f"Whisper pipeline failed: {exc}") from exc
        return result.get("text", "")

    def _transcribe_wav2vec2(self) -> str:
        if self.processor is None or self.model is None:
            raise RuntimeError("Wav2Vec2 components not initialized")

        waveform_np, sampling_rate = self._load_audio()
        processor_inputs = self.processor(
            waveform_np,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_tensor = processor_inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]

        text = self.processor.decode(pred_ids)
        if not self.tokenized:
            text = text.replace(" ", "")
        return text


# Example usage:
# sd = Sdab("path/to/audio.wav", model_name="metythorn/whisper-large-v3")
# print(sd.transcribe())
