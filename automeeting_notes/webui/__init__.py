import os

import numpy as np
import torch
import gradio as gr

from whisper import available_models

from automeeting_notes import diarize_text, res_to_txt
from automeeting_notes import models
args = {'verbose': True, 'task': 'transcribe', 'language': None, 'best_of': 5, 'beam_size': 5, 'patience': None, 'length_penalty': None, 'suppress_tokens': '-1', 'initial_prompt': None, 'condition_on_previous_text': True, 'fp16': True, 'compression_ratio_threshold': 2.4, 'logprob_threshold': -1.0, 'no_speech_threshold': 0.6}

def transcribe_and_diarize(audio_path, verbose, model, transcript_format):
    # if audio is not wav, convert it with ffmpeg
    if not audio_path.endswith(".wav"):
        audio_path_new = audio_path + ".wav"
        os.system(f"ffmpeg -i {audio_path} -acodec pcm_s16le -ac 1 -ar 16000 {audio_path_new}")
        try:
            os.remove(audio_path)
        except OSError:
            pass
        
        audio_path = audio_path_new

    models.load_whisper_model(model)
    
    args['verbose'] = verbose
    models.verbose = verbose

    result = models.models.transcribe(audio_path, temperature=0.7, **args)
    diarization_result = models.models.pyannote_pipeline(audio_path)

    try:
        os.remove(audio_path)
    except OSError:
        pass

    res = diarize_text(result, diarization_result)
    text = res_to_txt(res, transcript_format)
    return text

css = """                     
                             .gradio-input,
                             .gradio-output {
                                 border-radius: 15px;
                             }
                             .gradio-header {
                                 background-color: #4B9CD3;
                                 color: white;
                                 font-weight: bold;
                                 font-size: 24px;
                             }
                             .gradio-description {
                                 font-size: 16px;
                                 padding: 10px 0;
                             }
                         """

def webui():
    # Define input and output components
    
    # Choose a model
    with gr.Blocks() as demo:
        gr.Markdown("## Audio Transcription with Dialogue Diarization")
        with gr.Row().style(equal_height=True):
            with gr.Column():
                audio_input = gr.components.Audio(label="Audio", type="filepath")

                with gr.Row().style(equal_height=True):
                    models = gr.inputs.Dropdown(
                        label="Model",
                        choices=available_models(),
                        default="large",
                    )

                    transcript_format = gr.inputs.Dropdown(
                        label="Format",
                        choices=["timestamps", "simple"],
                        default="simple",
                    )

                    check_box_verbose = gr.components.Checkbox(label="Verbose")
                
            text_output = gr.components.Textbox(label="Transcription")
        
        
        
        button = gr.Button("Transcribe", label="Transcribe")
        button.click(transcribe_and_diarize,
                        inputs=[audio_input, check_box_verbose, models, transcript_format],
                        outputs=[text_output])

    demo.launch()