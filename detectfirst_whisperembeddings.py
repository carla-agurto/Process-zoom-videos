#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 17:35:30 2025

@author: carlaagurto
"""

import os
import wave
import contextlib
import webrtcvad
from pydub import AudioSegment

import torch
import whisper
import numpy as np
import pandas as pd



def convert_to_wav_16k(file, out_file):
    """Convert audio file to mono wav, 16kHz, 16-bit PCM."""
    audio = AudioSegment.from_file(file)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(out_file, format="wav")
    return out_file

def read_wave(path):
    """Read a WAV file and return PCM audio data, sample rate."""
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, "Audio must be mono"
        sample_width = wf.getsampwidth()
        assert sample_width == 2, "Audio must be 16-bit"
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000), "Unsupported sample rate"
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def detect_first_speech_time(file, vad_level=2, frame_duration_ms=30):
    """Return time (s) when speech first starts in file, or None if no speech."""
    pcm_data, sample_rate = read_wave(file)
    vad = webrtcvad.Vad(vad_level)
    frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # bytes per frame
    frames = [pcm_data[i:i+frame_size] for i in range(0, len(pcm_data), frame_size)]

    for i, frame in enumerate(frames):
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sample_rate):
            return (i * frame_duration_ms) / 1000.0  # seconds
    return None

def compare_two_files(folder):
    """Find which of two files in folder speaks first."""
    files = [os.path.join(folder, f) for f in os.listdir(folder) if not f.startswith(".")]
    assert len(files) == 2, "Folder must contain exactly 2 audio files"

    converted_files = []
    for f in files:
        base, _ = os.path.splitext(f)
        out_file = base + "_16k.wav"
        converted_files.append(convert_to_wav_16k(f, out_file))

    times = {}
    for f in converted_files:
        t = detect_first_speech_time(f, vad_level=3)
        times[f] = t if t is not None else float("inf")

    sorted_files = sorted(times.items(), key=lambda x: x[1])
    first_file = os.path.basename(sorted_files[0][0])
    second_file = os.path.basename(sorted_files[1][0])

    return first_file, second_file, times




def extract_whisper_features_df(wav_file, model_name="base", chunk_sec=30, overlap_sec=15):
    """
    Extract Whisper encoder embeddings from long audio with overlapping windows
    and save results into a DataFrame with embedding dimensions as columns.

    Parameters:
        wav_file (str): Path to audio file (.wav).
        model_name (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        chunk_sec (int): Window size in seconds (default=30).
        overlap_sec (int): Overlap in seconds (default=15).

    Returns:
        pd.DataFrame: DataFrame with columns ['start_time', 'end_time', 'e0', ..., 'eN'].
    """
    # Load Whisper model
    model = whisper.load_model(model_name)

    # Load audio
    audio = whisper.load_audio(wav_file)
    sr = whisper.audio.SAMPLE_RATE  # 16000 Hz

    # Convert to samples
    chunk_size = chunk_sec * sr
    step_size = (chunk_sec - overlap_sec) * sr

    rows = []
    start_sample = 0

    while start_sample < len(audio):
        end_sample = int(start_sample + chunk_size)
        chunk = audio[int(start_sample):int(end_sample)]

        # Pad or trim
        chunk = whisper.pad_or_trim(chunk)
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)

        # Encode
        with torch.no_grad():
            feats = model.encoder(mel.unsqueeze(0))  # (1, frames, dim)
            feats = feats.squeeze(0)  # (frames, dim)

        # Pool into single embedding per chunk (mean across time)
        pooled_embedding = feats.mean(dim=0).cpu().numpy()

        # Compute start/end time in seconds
        start_time = start_sample / sr
        end_time = min(end_sample, len(audio)) / sr

        # Row: start_time, end_time, embedding dims
        row = [start_time, end_time] + pooled_embedding.tolist()
        rows.append(row)

        # Advance window
        start_sample += step_size
        if end_sample >= len(audio):
            break

    # Column names: start_time, end_time, e0...eN
    dim = pooled_embedding.shape[0]
    columns = ["start_time", "end_time"] + [f"whisp_emb{i}" for i in range(dim)]

    df = pd.DataFrame(rows, columns=columns)
    return df



