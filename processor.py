import gc
import os

import numpy as np

from core.analysis import Analyzer
from core.denoising import Denoiser
from core.upscaling import Upscaler
from core.dynamics import DynamicsProcessor
from utils.audio_io import load_audio, save_audio, finalize_audio


MAX_DURATION_SECONDS = 6 * 60
CHUNK_DURATION_SECONDS = 30
OVERLAP_SECONDS = 1

class AudioProcessor:
    def __init__(self):
        self.analyzer = Analyzer()
        self.denoiser = Denoiser()
        self.upscaler = Upscaler()
        self.dynamics = DynamicsProcessor()

    def _process_chunk(self, y, sr, analysis_data):
        y_denoised = self.denoiser.process(y, sr)
        y_upscaled = self.upscaler.process(y_denoised, sr, analysis_data)
        return self.dynamics.process(y_upscaled, sr)

    def process(self, file_path, progress_callback=None):
        # 1. Load Audio
        y, sr = load_audio(file_path)
        sample_count = y.shape[-1]
        duration = sample_count / sr
        if duration > MAX_DURATION_SECONDS + 0.1:
            raise ValueError("音源は6分以内にしてください。")

        if progress_callback:
            progress_callback(0.03, "音源を解析しています")

        # Analyze one representative excerpt instead of constructing a full-song
        # spectrogram. This keeps memory bounded for tracks up to six minutes.
        analysis_length = min(sample_count, CHUNK_DURATION_SECONDS * sr)
        analysis_start = max(0, (sample_count - analysis_length) // 2)
        analysis_excerpt = y[:, analysis_start:analysis_start + analysis_length]
        analysis_data = self.analyzer.analyze(analysis_excerpt, sr)

        # Process fixed-size overlapping chunks. Whole-song STFTs can consume
        # several gigabytes for a three-to-six-minute stereo file.
        chunk_size = CHUNK_DURATION_SECONDS * sr
        overlap = OVERLAP_SECONDS * sr
        step = chunk_size - overlap
        starts = list(range(0, sample_count, step))
        y_processed = np.zeros_like(y, dtype=np.float32)
        weights = np.zeros(sample_count, dtype=np.float32)

        for index, start in enumerate(starts):
            end = min(start + chunk_size, sample_count)
            chunk = y[:, start:end].copy()
            processed = self._process_chunk(chunk, sr, analysis_data)
            processed = np.asarray(processed, dtype=np.float32)

            window = np.ones(end - start, dtype=np.float32)
            fade_length = min(overlap, end - start)
            if start > 0 and fade_length:
                window[:fade_length] = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
            if end < sample_count and fade_length:
                window[-fade_length:] = np.minimum(
                    window[-fade_length:],
                    np.linspace(1.0, 0.0, fade_length, dtype=np.float32),
                )

            y_processed[:, start:end] += processed[:, :end - start] * window
            weights[start:end] += window

            del chunk, processed, window
            gc.collect()
            if progress_callback:
                progress_callback(
                    0.08 + 0.82 * ((index + 1) / len(starts)),
                    f"分割処理中 ({index + 1}/{len(starts)})",
                )

        y_processed /= np.maximum(weights, 1e-6)[np.newaxis, :]

        if progress_callback:
            progress_callback(0.93, "音量とピークを調整しています")

        # Finalize (LUFS, True Peak) once after all chunks are joined.
        y_final = finalize_audio(y_processed, sr)

        # Save output
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, _ = os.path.splitext(base_name)
        # WAV is consistently supported by SoundFile and web browsers even when
        # the uploaded source is MP3, M4A, or another compressed format.
        output_path = os.path.join(dir_name, f"{name}_lifter.wav")

        save_audio(output_path, y_final, sr)
        if progress_callback:
            progress_callback(1.0, "完了しました")
        return output_path
