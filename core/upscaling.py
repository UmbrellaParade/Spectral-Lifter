import numpy as np
import librosa

class Upscaler:
    def __init__(self, target_sr=48000):
        self.target_sr = target_sr
        
    def _shape_transients(self, y, sr):
        # Basic transient shaping via harmonic-percussive separation
        _, y_percussive = librosa.effects.hpss(y, margin=2.0)
        # Boost percussive component
        y_enhanced = y + y_percussive * 0.5
        return y_enhanced
        
    def process(self, y, sr, analysis_data):
        """
        Reconstruct >16kHz band and shape transients.
        """
        if y.ndim > 1:
            y_mono = librosa.to_mono(y if y.shape[0] <= 2 else y.T)
        else:
            y_mono = y

        # --- 1. Spectral Bandwidth Extension (Neural/Heuristic) ---
        # V1.2 uses a lightweight heuristic approach. The previous PyTorch model
        # had no trained weights and did not affect the output.
        # Pitch shift by one octave to generate harmonics above original cutoff
        y_exciter = librosa.effects.pitch_shift(y_mono, sr=sr, n_steps=12, res_type='soxr_hq')
        
        # High-pass the generated signal above the detected cutoff (or 14kHz fallback)
        cutoff = analysis_data.get("cutoff_frequency", 14000.0)
        S_exc = librosa.stft(y_exciter)
        freqs = librosa.fft_frequencies(sr=sr)
        hp_mask = (freqs > max(cutoff - 1000, 12000)).astype(float)[:, np.newaxis]
        S_exc_hp = S_exc * hp_mask
        y_hf = librosa.istft(S_exc_hp)
        
        # Mix HF back into original signal
        if y.ndim > 1:
            y_out = y.copy()
            for ch in range(y.shape[0]):
                length = min(len(y_out[ch]), len(y_hf))
                y_out[ch][:length] += y_hf[:length] * 0.25 # Adjust mix amount
        else:
            y_out = y.copy()
            length = min(len(y), len(y_hf))
            y_out[:length] += y_hf[:length] * 0.25
            
        # --- 2. Transient Shaping ---
        if y_out.ndim > 1:
            for ch in range(y_out.shape[0]):
                y_out[ch] = self._shape_transients(y_out[ch], sr)
        else:
            y_out = self._shape_transients(y_out, sr)
            
        return y_out
