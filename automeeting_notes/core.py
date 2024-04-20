import pyannote.core as pcore

def obtain_timed_text_segments(transcript):
    """
    Extracts a list of timed text segments from a transcript.

    Parameters
    ----------
    transcript : dict
        The transcript from which to extract the text segments.

    Returns
    -------
    list
        A list of tuples, each tuple contains a `Segment` object
        and a string representing the text of that segment.
    """
    segment_text_pairs = []
    for seg in transcript['segments']:
        start_time, end_time, txt = seg['start'], seg['end'], seg['text']
        segment_text_pairs.append((pcore.Segment(start_time, end_time), txt))
    return segment_text_pairs

def allocate_speaker_to_segments(timed_text_segments, spk_data):
    """
    Assigns speakers to the timed text segments.

    Parameters
    ----------
    timed_text_segments : list
        A list of timed text segments.

    spk_data : SpeakerDiarization
        Speaker diarization data.

    Returns
    -------
    list
        A list of tuples with time range, speaker, and text information.
    """
    speaker_text_details = []
    for seg, txt in timed_text_segments:
        spk = spk_data.crop(seg).argmax()
        speaker_text_details.append(((seg.start, seg.end), spk, txt))
    return speaker_text_details

def consolidate_sentences(speaker_text_details):
    """
    Merges consecutive sentences spoken by the same speaker.

    Parameters
    ----------
    speaker_text_details : list
        A list of tuples with time range, speaker, and text information.

    Returns
    -------
    list
        A list of merged tuples with time range, speaker, and text information.
    """
    combined_speaker_text = []
    current_speaker = None
    sentence_group = []

    for timing, speaker, text in speaker_text_details:
        if speaker != current_speaker and len(sentence_group) > 0:
            combined_speaker_text.append(merge_grouped_sentences(sentence_group))
            sentence_group = [(timing, speaker, text)]
            current_speaker = speaker
        else:
            sentence_group.append((timing, speaker, text))
            current_speaker = speaker

    if len(sentence_group) > 0:
        combined_speaker_text.append(merge_grouped_sentences(sentence_group))

    return combined_speaker_text

def merge_grouped_sentences(sentence_group):
    """
    Merges sentences for a single speaker.

    Parameters
    ----------
    sentence_group : list
        A list of tuples with time range, speaker, and text information.

    Returns
    -------
    tuple
        A merged tuple with time range, speaker, and text information.
    """
    combined_start, combined_end = sentence_group[0][0][0], sentence_group[-1][0][1]
    combined_speaker = sentence_group[0][1]
    combined_text = ''.join([entry[-1] for entry in sentence_group]).lstrip()
    return ((combined_start, combined_end), combined_speaker, combined_text)

def diarize_text(transcribe_res, diarization_result):
    """
    Diarizes the text from a transcription.
    Parameters
    ----------
    transcribe_res : dict
        The transcription result.
    diarization_result : pyannote.core.Annotation
        The diarization result.
    Returns
    -------
    list
        A list of tuples with time range, speaker, and text information.
    """
    timed_segments = obtain_timed_text_segments(transcribe_res)
    spk_segment = allocate_speaker_to_segments(timed_segments, diarization_result)
    processed_res = consolidate_sentences(spk_segment)
    return processed_res

def speaker_transition_timestamps(diarization):
    """
    Get the timestamps where the speaker changes.
    Parameters
    ----------
    diarization : pyannote.core.Annotation
        The diarization result.
    Returns
    -------
    list
        A list of timestamps where the speaker changes.
    """
    timestamps = []
    current_speaker = None
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if current_speaker is None:
            current_speaker = speaker
        
        if speaker != current_speaker:
            timestamps.append((turn.start + previouse_turn.end)/2)
            current_speaker = speaker
        previouse_turn = turn

    return timestamps

# Create a function to convert the processed results into text
def res_to_txt(res_processed, style="simple"):
    # Initialize an empty string to store the output
    res_txt = ''
    
    # Iterate over each segment, speaker, and sentence in the processed results
    for seg, spk, sentence in res_processed:
        # Create a line with the timestamp, speaker, and sentence
        start_minutes, start_seconds = divmod(int(seg[0]), 60)
        end_minutes, end_seconds = divmod(int(seg[1]), 60)

        if style == "full_timestamp":
            line = f'{start_minutes:02d}:{start_seconds:02d}-{end_minutes:02d}:{end_seconds:02d} {spk}: {sentence}\n'
        
        elif style == "timestamp":
            line = f'{start_minutes:02d}:{start_seconds:02d} {spk}: {sentence}\n'
        
        elif style == "simple":
            line = f'{spk}: {sentence}\n'
        
        # Append the line to the output string
        res_txt += line
    
    # Return the output string
    return res_txt

def write_to_txt(spk_sent, file, **kwargs):
    with open(file, 'w') as f:
        f.write(res_to_txt(spk_sent, **kwargs))
