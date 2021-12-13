import yaml
import pathlib
import librosa as li
from ddsp.core import extract_loudness, extract_pitch
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
from scipy.io import wavfile


def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(x, sampling_rate, block_size, signal_length, oneshot, model_capacity='full',**kwargs):
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N)) #pad only the end of the array.

    if oneshot:
        x = x[..., :signal_length]

    pitch, _ = extract_pitch(x, sampling_rate, block_size, model_capacity=model_capacity)
    loudness = extract_loudness(x, sampling_rate, block_size)
    
    # reshape x to keep signal_length in the last dimension.
    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        p = torch.from_numpy(self.pitchs[idx])
        l = torch.from_numpy(self.loudness[idx])
        return s, p, l


def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    files = get_files(**config["data"])
    pb = tqdm(files)

    signals = []
    pitchs = []
    loudness = []

    for f in pb:
        pb.set_description(str(f))
        x, _ = li.load(f, **config["preprocess"]["sampling_rate"])
        x, p, l = preprocess(x, **config["preprocess"])
        signals.append(x)
        pitchs.append(p)
        loudness.append(l)

    # Concatenate all signals
    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    out_dir = config["preprocess"]["out_dir"]
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)


if __name__ == "__main__":
    main()