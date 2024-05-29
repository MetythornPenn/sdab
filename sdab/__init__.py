import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import numpy as np
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class Sdab:
    def __init__(self, 
                 file_path: str, 
                 model_name: str = "metythorn/khmer-asr-openslr",
                 tokenized: bool = False, 
                 device: str = 'cpu'):
        if os.path.isdir(model_name):
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        self.device = torch.device(device)
        self.file_path = file_path
        self.tokenized = tokenized

        # Perform ASR process and store the result
        self.result = self._perform_asr()

    def speech_file_to_array_fn(self, batch: dict) -> dict:
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = speech_array[0]
        batch["sampling_rate"] = sampling_rate
        return batch

    def resample(self, batch: dict) -> dict:
        resampler = torchaudio.transforms.Resample(batch['sampling_rate'], 16_000)
        batch["speech"] = resampler(batch["speech"]).numpy()
        batch["sampling_rate"] = 16_000
        return batch

    def prepare_dataset(self, batch: dict) -> dict:
        batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values
        return batch

    def _perform_asr(self) -> str:
        """
        Perform ASR and return the transcribed text.
        :return: Transcribed text from the ASR model.
        :rtype: str
        """
        b = {'path': self.file_path}
        a = self.prepare_dataset(self.resample(self.speech_file_to_array_fn(b)))
        input_dict = self.processor(a["input_values"][0], return_tensors="pt", padding=True)
        logits = self.model(input_dict.input_values.to(self.device)).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]

        if self.tokenized:
            txt = self.processor.decode(pred_ids)
        else:
            txt = self.processor.decode(pred_ids).replace(' ', '')

        return txt

# Example initialization and usage:
# file_path = "path_to_your_audio_file.wav"
# model_name = "metythorn/khmer-asr-openslr"  # or local directory path
# sdab = Sdab(file_path, model_name)
# print(sdab.result)
