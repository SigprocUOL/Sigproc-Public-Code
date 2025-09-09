"""
process a sample utterance (from DNS synthetic test dataset without REVERB,
https://dns-challenge.azurewebsites.net/Interspeech2020)
"""

import time

import soundfile as sf
import torch
from pypesq import pesq

import deep_mfmvdr as dm

EVAL = False
NUM_CORES = 4
torch.device("cpu")
torch.set_num_threads(NUM_CORES)

def main():
    # load test utterance
    noisy, fs = sf.read("noisy.wav")
    noisy_t = torch.Tensor(noisy).unsqueeze(0)  # (batch_size x num_samples)
    duration = noisy_t.shape[-1] / fs

    if EVAL:
        clean, _ = sf.read("clean.wav")
        pesq_scores = {'noisy': pesq(clean, noisy, fs)}
    
    # init model and load weights
    config = {'filter_length': 5}
    state_dict = torch.load('dmfmvdr_best_weights.pt', map_location="cpu")
    deep_mfmvdr = dm.DeepMFMVDR(config)
    deep_mfmvdr.load_state_dict(state_dict)

    # get model output
    with torch.no_grad():
        start_time = time.time()
        output = deep_mfmvdr(noisy_t)
        end_time = time.time()
        print(f"processing audio of length {duration:.1f} s took {(end_time - start_time):.1f} s using {NUM_CORES} cores of your CPU.")

    # (evaluate and) save enhanced wave
    enhanced = output["speech_estimate_wave"].numpy()[0]
    sf.write('enhanced.wav', enhanced, fs)

    if EVAL:
        pesq_scores['enhanced'] = pesq(clean, enhanced, fs)
        print("PESQ scores for this utterance:")
        print(f"noisy: {pesq_scores['noisy']:.2f}, enhanced: {pesq_scores['enhanced']:.2f}")
        print(f"-> Delta PESQ = {(pesq_scores['enhanced'] - pesq_scores['noisy']):.2f}")


if __name__ == "__main__":
    main()
