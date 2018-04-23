import os
import librosa
import torch
import numpy as np

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
    
def np_to_input_tensor(data, use_cuda):
    data = data[:,np.newaxis,:]
    data = torch.Tensor(data)
    if use_cuda:
        data = data.cuda()
    return data


def save_tidegan_samples(output_dir, current_audibles, step, fs=16000):
    samples_dir = os.path.join(output_dir, 'samples', str(step))
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
        
    for out_type, data in current_audibles.items():
        for idx, sample in enumerate(data):
            output_path = os.path.join(samples_dir, "{}_{}.wav".format(out_type, idx+1))
            librosa.output.write_wav(output_path, sample, sr=fs)
            
            
def tensor2audio(audio_tensor):
    audio_numpy = audio_tensor[0].cpu().float().numpy()
    return audio_numpy