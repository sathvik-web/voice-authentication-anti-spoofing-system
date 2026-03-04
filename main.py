import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import welch
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from speechbrain.inference import EncoderClassifier
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

# SETTINGS
SAMPLE_RATE        = 16000
DURATION           = 15
SIMILARITY_THRESHOLD = 75      # % cosine similarity to accept same speaker
SPOOF_THRESHOLD      = 35      # Lower = stricter anti-spoof (was too high before)
LOG_FILE             = "auth_log.json"

# ANSI color codes for terminal output
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# LOAD SPEAKER MODEL
print(f"{CYAN}{BOLD}Loading Speaker Verification Model...{RESET}")
speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"}
)
print(f"{GREEN}Speaker Model Loaded.{RESET}\n")


# RECORD AUDIO
def record_audio(filename: str) -> None:
    """Record audio from the microphone and save as WAV."""
    print(f"\n{CYAN}Recording '{filename}' for {DURATION} seconds...{RESET}")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    print(f"{GREEN}Saved: {filename}{RESET}")

# SILENCE / NOISE CHECK
def check_audio_quality(file_path: str) -> dict:
    """
    Check if audio contains enough speech and estimate SNR.
    Returns a dict with 'valid', 'rms_db', 'snr_db'.
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    rms = np.sqrt(np.mean(y ** 2))
    rms_db = 20 * np.log10(rms + 1e-9)

    # Estimate noise floor from the quietest 10% of frames
    frame_length = 512
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=256)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
    noise_floor = np.percentile(frame_rms, 10)
    signal_peak = np.percentile(frame_rms, 90)
    snr_db = 20 * np.log10((signal_peak + 1e-9) / (noise_floor + 1e-9))

    # Voice activity: fraction of frames above noise floor threshold
    vad_threshold = noise_floor * 3
    voice_ratio = np.mean(frame_rms > vad_threshold)

    valid = rms_db > -40 and snr_db > 5 and voice_ratio > 0.2

    return {
        "valid": valid,
        "rms_db": float(rms_db),
        "snr_db": float(snr_db),
        "voice_ratio": float(voice_ratio)
    }

# SPEAKER EMBEDDING
def get_embedding(file_path: str) -> torch.Tensor:
    """Extract ECAPA-TDNN speaker embedding from a WAV file."""
    signal, fs = torchaudio.load(file_path)

    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    if fs != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=SAMPLE_RATE)
        signal = resampler(signal)

    embedding = speaker_model.encode_batch(signal)
    return embedding.squeeze()

# IMPROVED ANTI-SPOOF DETECTOR
def detect_spoof(file_path: str) -> tuple[str, float, dict]:
    """
    Multi-feature anti-spoof analysis.
    
    Features used:
      1. MFCC delta variance   — real voices have natural fluctuation
      2. Pitch jitter          — synthetic voices are often too regular
      3. Spectral flux         — measures frame-to-frame spectral change
      4. Zero-crossing rate    — helps distinguish natural vs. processed audio
      5. Harmonic-to-noise ratio (HNR) — low HNR can indicate synthesis artifacts
      6. Spectral rolloff      — real speech typically has energy spread
    
    Each feature contributes a sub-score; the final score is a weighted average.
    Score < SPOOF_THRESHOLD → bonafide (real), ≥ threshold → spoof.
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    details = {}

    # 1. MFCC delta variance 
    # Real voices have rich, varied MFCC deltas; TTS tends to be smoother
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfccs)
    mfcc_var = np.mean(np.var(delta_mfcc, axis=1))
    # Map: low variance → higher spoof likelihood
    mfcc_score = float(np.clip(1.0 - mfcc_var / 5.0, 0, 1) * 100)
    details["mfcc_delta_var"] = round(float(mfcc_var), 4)

    # 2. Pitch jitter (F0 irregularity)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, fill_na=None
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    if len(f0_voiced) > 10:
        f0_diff = np.diff(f0_voiced[~np.isnan(f0_voiced)])
        jitter = np.std(f0_diff) / (np.mean(np.abs(f0_voiced[~np.isnan(f0_voiced)])) + 1e-6)
        # Very low jitter → too regular → more likely synthetic
        pitch_score = float(np.clip(1.0 - jitter * 10, 0, 1) * 100)
    else:
        pitch_score = 50.0  # No voiced frames detected — neutral score
    details["pitch_jitter"] = round(float(jitter) if len(f0_voiced) > 10 else -1, 4)

    # 3. Spectral flux 
    S = np.abs(librosa.stft(y))
    flux = np.mean(np.diff(S, axis=1) ** 2)
    # Very low flux → overly static spectrum → suspicious
    flux_score = float(np.clip(1.0 - flux / 0.05, 0, 1) * 100)
    details["spectral_flux"] = round(float(flux), 6)

    # 4. Zero-crossing rate variance 
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_var = np.var(zcr)
    # Real voices have more ZCR variability
    zcr_score = float(np.clip(1.0 - zcr_var / 0.01, 0, 1) * 100)
    details["zcr_variance"] = round(float(zcr_var), 6)

    # 5. Harmonic-to-noise ratio proxy 
    harmonics, percussive = librosa.effects.hpss(y)
    hnr = np.sum(harmonics ** 2) / (np.sum(percussive ** 2) + 1e-6)
    # Extremely high HNR can indicate synthetic purity; very low = noise
    hnr_score = float(np.clip(abs(np.log10(hnr + 1e-6) - 1.5) / 2.0, 0, 1) * 100)
    details["hnr"] = round(float(hnr), 4)

    # 6. Spectral rolloff spread 
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_std = np.std(rolloff)
    # Real speech has natural rolloff variation; too consistent → suspicious
    rolloff_score = float(np.clip(1.0 - rolloff_std / 2000.0, 0, 1) * 100)
    details["rolloff_std"] = round(float(rolloff_std), 2)

    # Weighted final score
    weights = {
        "mfcc":    0.30,
        "pitch":   0.25,
        "flux":    0.20,
        "zcr":     0.10,
        "hnr":     0.08,
        "rolloff": 0.07,
    }
    spoof_score = (
        weights["mfcc"]    * mfcc_score   +
        weights["pitch"]   * pitch_score  +
        weights["flux"]    * flux_score   +
        weights["zcr"]     * zcr_score    +
        weights["hnr"]     * hnr_score    +
        weights["rolloff"] * rolloff_score
    )
    spoof_score = float(np.clip(spoof_score, 0, 100))

    label = "spoof" if spoof_score >= SPOOF_THRESHOLD else "bonafide"
    return label, spoof_score, details

# PLOT SPECTROGRAMS + FEATURES
def plot_results(ref_path: str, test_path: str,
                 similarity: float, spoof_score: float,
                 spoof_label: str, final: str) -> None:
    """Plot mel spectrograms and a result summary panel."""
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Voice Authentication Analysis", fontsize=16, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Reference spectrogram 
    ax1 = fig.add_subplot(gs[0, 0])
    y_ref, sr = librosa.load(ref_path, sr=SAMPLE_RATE)
    S_ref = librosa.feature.melspectrogram(y=y_ref, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S_ref, ref=np.max),
                             sr=sr, x_axis='time', y_axis='mel', ax=ax1)
    ax1.set_title("Reference Voice")

    # Test spectrogram 
    ax2 = fig.add_subplot(gs[0, 1])
    y_test, _ = librosa.load(test_path, sr=SAMPLE_RATE)
    S_test = librosa.feature.melspectrogram(y=y_test, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S_test, ref=np.max),
                             sr=sr, x_axis='time', y_axis='mel', ax=ax2)
    ax2.set_title("Test Voice")

    # MFCC comparison
    ax3 = fig.add_subplot(gs[0, 2])
    mfcc_ref  = librosa.feature.mfcc(y=y_ref,  sr=sr, n_mfcc=13)
    mfcc_test = librosa.feature.mfcc(y=y_test, sr=sr, n_mfcc=13)
    ax3.plot(np.mean(mfcc_ref,  axis=1), label="Reference", marker='o', ms=4)
    ax3.plot(np.mean(mfcc_test, axis=1), label="Test",      marker='s', ms=4)
    ax3.set_title("Mean MFCCs")
    ax3.set_xlabel("MFCC Coefficient")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # Score gauges 
    ax4 = fig.add_subplot(gs[1, 0])
    colors_sim = ["#27ae60" if similarity >= SIMILARITY_THRESHOLD else "#e74c3c"]
    ax4.barh(["Similarity"], [similarity], color=colors_sim, height=0.4)
    ax4.barh(["Similarity"], [100], color="#ecf0f1", height=0.4)
    ax4.barh(["Similarity"], [similarity], color=colors_sim, height=0.4)
    ax4.axvline(SIMILARITY_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax4.set_xlim(0, 100)
    ax4.set_title(f"Speaker Similarity: {similarity:.1f}%")
    ax4.text(similarity + 1, 0, f"{similarity:.1f}%", va='center', fontsize=10)

    ax5 = fig.add_subplot(gs[1, 1])
    color_spoof = "#e74c3c" if spoof_label == "spoof" else "#27ae60"
    ax5.barh(["Spoof Score"], [spoof_score], color=color_spoof, height=0.4)
    ax5.axvline(SPOOF_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax5.set_xlim(0, 100)
    ax5.set_title(f"Spoof Score: {spoof_score:.1f}%")
    ax5.text(spoof_score + 1, 0, f"{spoof_score:.1f}%", va='center', fontsize=10)

    # Final decision panel 
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    bg_color = "#d5f5e3" if final == "ACCESS GRANTED" else "#fadbd8"
    txt_color = "#1e8449" if final == "ACCESS GRANTED" else "#922b21"
    ax6.set_facecolor(bg_color)
    ax6.patch.set_visible(True)
    ax6.text(0.5, 0.55, final, ha='center', va='center',
             fontsize=15, fontweight='bold', color=txt_color,
             transform=ax6.transAxes)
    ax6.text(0.5, 0.30,
             f"Speaker: {'✓ SAME' if similarity >= SIMILARITY_THRESHOLD else '✗ DIFF'}\n"
             f"Liveness: {'✓ REAL' if spoof_label == 'bonafide' else '✗ FAKE'}",
             ha='center', va='center', fontsize=10, color=txt_color,
             transform=ax6.transAxes)

    plt.savefig("auth_result.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"{GREEN}Plot saved to auth_result.png{RESET}")

# LOGGING
def _make_serializable(obj):
    """Recursively convert numpy / torch types to Python native types for JSON."""
    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # torch tensors
    if isinstance(obj, torch.Tensor):
        try:
            return obj.detach().cpu().numpy().tolist()
        except Exception:
            return None
    # dict / list recursion
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj
def log_result(similarity: float, spoof_score: float,
               spoof_label: str, final: str, quality: dict) -> None:
    """Append authentication result to a JSON log file."""
    entry = {
        "timestamp":      datetime.datetime.now().isoformat(),
        "similarity_pct": round(similarity, 2),
        "spoof_score":    round(spoof_score, 2),
        "spoof_label":    spoof_label,
        "final_decision": final,
        "audio_quality":  _make_serializable(quality),
    }
    log = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []
    log.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print(f"{CYAN}Result logged to {LOG_FILE}{RESET}")

# PRINT SUMMARY
def print_summary(similarity: float, spoof_score: float,
                  spoof_label: str, quality: dict, final: str) -> None:
    w = 38
    print(f"\n{BOLD}{'=' * w}{RESET}")
    print(f"{BOLD}  VOICE AUTHENTICATION REPORT{RESET}")
    print(f"{BOLD}{'=' * w}{RESET}")

    # Audio quality
    q_color = GREEN if quality["valid"] else RED
    print(f"\n{'Audio Quality':.<25} {q_color}{'OK' if quality['valid'] else 'POOR'}{RESET}")
    print(f"  RMS level    : {quality['rms_db']:>6.1f} dB")
    print(f"  Est. SNR     : {quality['snr_db']:>6.1f} dB")
    print(f"  Voice ratio  : {quality['voice_ratio'] * 100:>5.1f}%")

    # Speaker similarity
    s_color = GREEN if similarity >= SIMILARITY_THRESHOLD else RED
    s_label = "SAME SPEAKER" if similarity >= SIMILARITY_THRESHOLD else "DIFFERENT SPEAKER"
    print(f"\n{'Similarity Score':.<25} {s_color}{similarity:>6.2f}%{RESET}")
    print(f"{'Speaker Decision':.<25} {s_color}{s_label}{RESET}")

    # Liveness
    l_color = GREEN if spoof_label == "bonafide" else RED
    l_label = "AUTHENTIC VOICE" if spoof_label == "bonafide" else "SPOOF / DEEPFAKE"
    print(f"\n{'Spoof Score':.<25} {l_color}{spoof_score:>6.2f}%{RESET}")
    print(f"{'Liveness Decision':.<25} {l_color}{l_label}{RESET}")

    # Final
    f_color = GREEN if final == "ACCESS GRANTED" else RED
    print(f"\n{BOLD}{'=' * w}{RESET}")
    print(f"{BOLD}  FINAL: {f_color}{final}{RESET}")
    print(f"{BOLD}{'=' * w}{RESET}\n")

# MAIN PIPELINE
def main() -> None:
    print(f"\n{BOLD}{CYAN}=== LIVE VOICE AUTHENTICATION SYSTEM ==={RESET}\n")

    # Record reference 
    input(f"{YELLOW}Press ENTER to record Reference Voice...{RESET}")
    record_audio("ref.wav")

    # Quality check on reference
    ref_quality = check_audio_quality("ref.wav")
    if not ref_quality["valid"]:
        print(f"{RED}⚠  Reference audio quality is poor "
              f"(SNR={ref_quality['snr_db']:.1f} dB). "
              f"Please re-record in a quieter environment.{RESET}")

    # Record test
    input(f"\n{YELLOW}Press ENTER to record Test Voice...{RESET}")
    record_audio("test.wav")

    test_quality = check_audio_quality("test.wav")
    if not test_quality["valid"]:
        print(f"{RED}⚠  Test audio quality is poor "
              f"(SNR={test_quality['snr_db']:.1f} dB).{RESET}")

    # Speaker similarity
    print(f"\n{CYAN}Extracting speaker embeddings...{RESET}")
    emb_ref  = get_embedding("ref.wav")
    emb_test = get_embedding("test.wav")
    score = F.cosine_similarity(emb_ref, emb_test, dim=0)
    similarity_percent = float(score) * 100

    # Anti-spoof 
    print(f"{CYAN}Running Anti-Spoof Analysis...{RESET}")
    spoof_label, spoof_score, spoof_details = detect_spoof("test.wav")

    print(f"\n  Feature breakdown:")
    for k, v in spoof_details.items():
        print(f"    {k:<22}: {v}")

    # Decision logic 
    speaker_ok    = similarity_percent >= SIMILARITY_THRESHOLD
    liveness_ok   = spoof_label == "bonafide"
    audio_ok      = test_quality["valid"]
    final_decision = "ACCESS GRANTED" if (speaker_ok and liveness_ok and audio_ok) \
                     else "ACCESS BLOCKED"

    # Output
    print_summary(similarity_percent, spoof_score, spoof_label,
                  test_quality, final_decision)

    log_result(similarity_percent, spoof_score, spoof_label,
               final_decision, test_quality)

    # Visualize
    print(f"{CYAN}Generating visualisation...{RESET}")
    plot_results("ref.wav", "test.wav",
                 similarity_percent, spoof_score,
                 spoof_label, final_decision)

    print(f"\n{BOLD}Demo Complete.{RESET}")


if __name__ == "__main__":
    main()