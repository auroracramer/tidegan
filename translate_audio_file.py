import argparse
import json
import os
import numpy as np
import scipy.signal
import librosa
import torch
from torch import autograd
from wavegan import load_wavegan_generator


def translate_frames(model, frames):
    """
    Run audio frames through timbre translation model and return the output
    """
    frames_v = autograd.Variable(torch.FloatTensor(frames).cuda())
    out_frames = model(frames_v)
    return out_frames.cpu().numpy()


def translate_audio(model, audio_data, hop_divisor_power=1):
    """
    Perform timbre translation on overlapping frames of audio and produce output
    signal via WOLA-style aggregation
    """
    if hop_divisor_power > 13:
        err_msg = 'hop_divisor_power must be less than 14. Got {}'
        raise ValueError(err_msg.format(hop_divisor_power))

    win_size = 2**14
    # Only use powers of 2 to ensure that overlapping windows result in a constant
    hop_size = win_size // (2**hop_divisor_power)
    scale_factor = (2*hop_size)/win_size

    # Create scaled Hanning window, such that sum of overlapping windows is 1
    win = scipy.signal.hanning(win_size, sym=False) * scale_factor

    # Pad input so the the window tapering doesn't affect the boundaries of the
    # signal
    inp = np.pad(audio_data, (win_size//2, win_size//2))
    out = np.zeros(inp.shape)

    # Split input into overlapping frames
    inp_frames = librosa.utils.frame(inp, frame_length=win_size, hop_length=hop_size).T
    # Translate each of the frames
    out_frames = translate_frames(model, inp_frames)

    # Perform weighted overlap-add style method of combining frames into output
    for idx, out_frame in enumerate(out_frames):
        n = idx * hop_size
        out[n:n+win_size] += win * out_frame

    # Truncate padded part
    out = out[win_size//2:-win_size//2]
    return out


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Apply timbre translation to an audio file')
    parser.add_argument('model_path', type=str, help='Path to model file')
    parser.add_argument('model_config_path', type=str, help='Path to model training config file')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('output_path', type=str,
                        help='Path where output audio file will be saved')
    parser.add_argument('--hop-divisor-power', '-hdp', dest='hop_divisor_power',
                        type=int, default=1,
                        help='Exponent of 2 used to divde the window size to get the hop_size')

    args = parser.parse_args()
    return vars(args)


def main(model_path, model_config_path, audio_path, output_path, hop_divisor_power=1):
    print ("Loading model")
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    model = load_wavegan_generator(filepath, **model_config)

    print ("Loading audio")
    audio_data, fs = librosa.load(audio_path, sr=44100)

    print ("Translating audio")
    output_audio = translate_audio(model, audio_data, hop_divisor_power)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Writing audio")
    librosa.output.write_wav(output_path, audio_data, fs)


if __name__ == '__main__':
    main(**parse_arguments())
