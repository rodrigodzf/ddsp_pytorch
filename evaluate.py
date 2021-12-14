import torch
import yaml
from argparse import ArgumentParser
from pathlib import Path
from ddsp.model import DDSP, DDSP_noseq
from ddsp.core import extract_loudness, extract_pitch
from preprocess import preprocess
import numpy as np
import librosa as li
from torch.nn.functional import l1_loss
import soundfile as sf
import matplotlib.pyplot as plt
# import torchaudio
# import torchcrepe


# def extract_loudness_pitch(f, sampling_rate, block_size, signal_length, oneshot, **kwargs):
#     audio, sr = li.load(f, sampling_rate)
#     N = (signal_length - len(audio) % signal_length) % signal_length
#     x = np.pad(audio, (0, N))
#     # N = (signal_length - audio.shape[-1] % signal_length) % signal_length
#     # audio = np.pad(audio, ((0, 0), (0, N)))

#     if oneshot:
#         audio = audio[..., :signal_length]


#     pitch = torchcrepe.predict(torch.from_numpy(audio).unsqueeze(0),
#                            sampling_rate,
#                            hop_length=block_size,
#                            fmin=1,
#                            fmax=2000,
#                            device='cpu')
#     loudness = extract_loudness(audio, sampling_rate, block_size)
#     return audio, pitch, loudness

def make_plots(gt_pitch_midi, gen_pitch_midi, loudness, gen_loudness):
    fig, ax = plt.subplots(3,1)
    ax[0].plot(gen_pitch_midi)
    ax[1].plot(gt_pitch_midi)
    ax[2].plot(np.abs(gen_pitch_midi - gt_pitch_midi))

    for a in ax:
        a.set_xlabel('Frame')
        a.set_ylabel('Semitones')
        a.grid()

    plt.tight_layout()
    plt.savefig('eval_pitch.png')

    fig, ax = plt.subplots(3,1)
    ax[0].plot(gen_loudness)
    ax[1].plot(loudness)
    ax[2].plot(np.abs(gen_loudness - loudness))

    ax[2].set_xlabel('Frame')
    ax[2].set_ylabel('Difference')
    ax[2].grid()

    for a in ax[:2]:
        a.set_xlabel('Frame')
        a.set_ylabel('Loudness')
        a.grid()
    plt.tight_layout()
    plt.savefig('eval_loudness.png')

    # %%

if __name__ == '__main__':
    parser = ArgumentParser(description='DDSP graphs')
    parser.add_argument('--input_dir', type=Path, default=".")
    parser.add_argument('--target_a', type=Path, default=".")
    parser.add_argument('--target_b', type=Path, default=".")
    parser.add_argument('--interpolate_f0', action='store_true')
    parser.add_argument('--interpolate_loudness', action='store_true')
    parser.add_argument('--confidence_threshold', type=float, default=0.85)
    parser.add_argument('--device', type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="GPU to use; -1 is CPU (default 0)")

    args = parser.parse_args()

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    # Load model and config
    # TODO: save metadata also in pth
    with open(Path(args.input_dir) / "config.yaml", "r") as config:
        config = yaml.safe_load(config)
    ddsp = DDSP(**config["model"]).to(device)
    state = ddsp.state_dict()
    pretrained = torch.load(Path(args.input_dir) / "state.pth", map_location=device)
    state.update(pretrained)
    ddsp.load_state_dict(state)
    ddsp.eval()

    # Preprocess ground truth
    preprocess_config = config["preprocess"]
    sampling_rate = preprocess_config["sampling_rate"]
    block_size = preprocess_config["block_size"]
    model_capacity = preprocess_config["model_capacity"]

    audio_a, _ = li.load(args.target_a, sampling_rate)
    if args.target_b and (args.interpolate_f0 or args.interpolate_loudness):
        audio_b, _ = li.load(args.target_b, sampling_rate)
        # cut the longest wav to match the shortest
        min_lenght = min(audio_a.size, audio_b.size)
        audio_a = audio_a[:min_lenght]
        audio_b = audio_b[:min_lenght]

    # preprocess_config["oneshot"] = True
    signals, pitchs, loudness = preprocess(audio_a, **preprocess_config)

    if args.target_b and (args.interpolate_f0 or args.interpolate_loudness):
        signals_b, pitchs_b, loudness_b = preprocess(audio_b, **preprocess_config)

    if args.target_b and args.interpolate_loudness:
        loudness = torch.from_numpy(loudness_b.astype(np.float32))
    else:
        loudness = torch.from_numpy(loudness.astype(np.float32))

    if args.target_b is not None and args.interpolate_f0:
        pitchs = torch.from_numpy(pitchs_b.astype(np.float32))
    else:
        pitchs = torch.from_numpy(pitchs.astype(np.float32))

    # for all frames
    with torch.no_grad():
        mean_loudness = loudness.mean()
        std_loudness = loudness.std()
        p = pitchs.unsqueeze(-1).to(device)
        l = loudness.unsqueeze(-1).to(device)
        l = (l - config["data"]["mean_loudness"]) / config["data"]["std_loudness"]

        y, _, _ = ddsp(p, l)

        y = y.reshape(-1).cpu().numpy()
        gen_pitch, gen_confidence = extract_pitch(y, sampling_rate, block_size, model_capacity)
        gen_loudness = extract_loudness(y, sampling_rate, block_size)

        # Use pitches with confidence above threshold 
        mask = gen_confidence[np.newaxis, :] > args.confidence_threshold

        gen_pitch = torch.from_numpy(gen_pitch.astype(np.float32))[mask]
        gt_pitch = pitchs.reshape(-1)[mask]
        gt_pitch_midi = li.core.hz_to_midi(gt_pitch)
        gen_pitch_midi = li.core.hz_to_midi(gen_pitch)

        pitch_l1 = np.abs(gen_pitch_midi - gt_pitch_midi).mean()
        loudness_l1 = np.abs(torch.from_numpy(gen_loudness.astype(np.float32)) - loudness.reshape(-1)).mean()

        make_plots(gt_pitch_midi, gen_pitch_midi, loudness.reshape(-1).numpy(), gen_loudness)
        print(f'Loudness L1 {loudness_l1}')
        print(f'Pitch L1 {pitch_l1}')

        sf.write(f"eval.wav", y, sampling_rate)

