# AutoMeeting Notes
An Python webui for generating speaker-diarized meeting transcriptions from audio recordings
Leveraging a state-of-the-art speech-to-text model (whisper) and speaker diarization techniques (pyannote), the library delivers a comprehensive transcript that accurately differentiates between various speakers in a meeting or conversation. With support for multiple languages and an optional command line user interface, AutoMeeting Notes streamlines the transcription process.

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