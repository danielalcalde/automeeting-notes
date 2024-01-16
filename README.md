# AutoMeeting Notes
A Python web UI for generating speaker-diarized meeting transcriptions from audio recordings. Utilizing the state-of-the-art Whisper speech-to-text model and Pyannote speaker diarization techniques, the library produces precise transcripts that accurately differentiate between speakers in a meeting or conversation. With multi-language support and an optional command-line interface, AutoMeeting Notes simplifies the transcription process.

## Key Features

- Automatic transcription of audio files with speaker diarization.
- Web-based user interface + command line interface.
- Configurable output formatting to suit user requirements.


## Installation Steps

- Clone the repository using the following command:
    
    ```sh
    git clone https://github.com/danielalcalde/automeeting-notes
    
    ```
    
- Navigate to the cloned directory and install the required packages:
    
    ```sh
    cd automeeting_notes
    ./install.sh
    
    ```
- Accepting eula from pyannote in hf.co in [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and add your huggingface token with:
    ```sh
    huggingface-cli login
    ``` 

## Usage Instructions

### Command Line Interface
```py
automeeting_notes path/to/audio_file --output_dir path/to/output_dir
```

The command will generate a text file containing the transcribed and diarized meeting notes in the specified output directory.


### Web-based User Interface
Launch the web application by executing:
```
automeeting-webui
```

## Acknowledgements
Inspired by https://github.com/yinruiqing/pyannote-whisper and https://github.com/openai/whisper/discussions/264