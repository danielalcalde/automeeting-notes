import torch
from whisper.transcribe import transcribe as transcribe_whisper
from .whisper_extension import transcribe
from .core import speaker_transition_timestamps
from whisper import load_model
import time

class Models:
    def __init__(self, whisper_model_name="large", device=None, verbose=False):
        self.whisper_model_name = whisper_model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.verbose = verbose
        self._whisper_model = None
        self._pyannote_pipeline = None

    def load_whisper_model(self, model_name=None):
        if model_name is None:
            model_name = self.whisper_model_name
        
        if self._whisper_model is not None:
            self._whisper_model = None
        
        self._whisper_model = load_model(model_name, device=self.device)

    def load_pyannote_pipeline(self, model_name="pyannote/speaker-diarization-3.1"):
        from pyannote.audio import Pipeline
        self._pyannote_pipeline = Pipeline.from_pretrained(model_name)
        if self.device == "cuda":
            self._pyannote_pipeline = self._pyannote_pipeline.to(torch.device(0))
    
    @property
    def whisper_model(self):
        if self._whisper_model is None:
            if self.verbose: print(f"Loading {self.whisper_model_name} Whisper model...")
            self.load_whisper_model(self.whisper_model_name)
            if self.verbose:print("Loaded Whisper model.")
        return self._whisper_model
    
    @property
    def pyannote_pipeline(self):
        if self._pyannote_pipeline is None:
            if self.verbose: print("Loading Pyannote pipeline...")
            self.load_pyannote_pipeline()
            if self.verbose: print("Loaded Pyannote pipeline.")
        return self._pyannote_pipeline
    
    def transcribe(self, audio_path, temperature=1.0, diarization=None, **kwargs):
        t0 = time.time()
        if self.verbose:
            print(f"Transcribing {audio_path} with {self.whisper_model_name} Whisper model...")

        if diarization is None:
            res = transcribe_whisper(self.whisper_model, audio_path, temperature=temperature, **kwargs)
        else:
            cut_timestamps = speaker_transition_timestamps(diarization)
            res = transcribe(self.whisper_model, audio_path, cut_timestamps, temperature=temperature, **kwargs)
        
        if self.verbose:
            print(f"Transcribed in {time.time() - t0:.2f} seconds.")

        return res

    def diarize(self, audio_path):
        t0 = time.time()
        if self.verbose:
            print(f"Diarizing {audio_path} with Pyannote pipeline...")

        res = self.pyannote_pipeline(audio_path)

        if self.verbose:
            print(f"Diarized in {time.time() - t0:.2f} seconds.")
        return res

models = Models(verbose=True)