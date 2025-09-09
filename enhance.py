import argparse
import time

import soundfile as sf
import torch

import deep_mfmvdr as dm

NUM_CORES = 4
torch.device("cpu")
torch.set_num_threads(NUM_CORES)


def main(input_path, output_path):
    # load test utterance
    noisy, fs = sf.read(input_path)
    noisy_t = torch.Tensor(noisy).unsqueeze(0)  # (batch_size x num_samples)
    duration = noisy_t.shape[-1] / fs

    # init model and load weights
    config = {"filter_length": 5}
    state_dict = torch.load("dmfmvdr_best_weights.pt", map_location="cpu")
    deep_mfmvdr = dm.DeepMFMVDR(config)
    deep_mfmvdr.load_state_dict(state_dict)

    # get model output
    with torch.no_grad():
        start_time = time.time()
        output = deep_mfmvdr(noisy_t)
        end_time = time.time()
        print(
            f"processing audio of length {duration:.1f} s took {(end_time - start_time):.1f} s using {NUM_CORES} cores of your CPU."
        )

    # (evaluate and) save enhanced wave
    enhanced = output["speech_estimate_wave"].numpy()[0]
    sf.write(output_path, enhanced, fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some audio files.")
    parser.add_argument(
        "input_path", type=str, help="Input path for the noisy wav file"
    )
    parser.add_argument(
        "output_path", type=str, help="Output path for the enhanced wav file"
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path)
