# ==============================================================================
# Adapted from predict.py mos-finetune-ssl by Erica Cooper
# ==============================================================================

import torch
from .mos_fairseq import MosPredictor, MyDataset
import numpy as np
import argparse
import fairseq
# import soundfile as sf
import torchaudio
import os
import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fairseq-base-model', type=str, required=True, help='Path to pretrained fairseq base model.')
    # parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--finetuned-checkpoint', type=str, required=True, help='Path to finetuned MOS prediction checkpoint.')
    # parser.add_argument('--outfile', type=str, required=False, default='answer.txt', help='Output filename for your answer.txt file for submission to the CodaLab leaderboard.')
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help='Device to use for inference.')
    parser.add_argument('--wav-fpath', type=str, default=None, help='Path to wav file to predict MOS for.')
    parser.add_argument('--wav-dir', type=str, default=None, help='Path to directory containing wav files to predict MOS for.')
    args = parser.parse_args()
    return args

def get_mos_model(cp_path, my_checkpoint, device):
    # datadir = args.datadir
    # outfile = args.outfile

    # system_csv_path = os.path.join(datadir, 'mydata_system.csv')

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))

    return model

def load_wav(wav_fpath, device, resample_tol=1e-1):
    wav, sr = torchaudio.load(wav_fpath)
    if sr != 16000:
        # print('Resampling from {} to 16000'.format(sr))
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    # assert sr == 16000, "Sample rate mismatch: {} vs 16000".format(sr)
    assert torch.max(torch.abs(wav)) - resample_tol <= 1.0, "wav file should be normalized to [-1, 1]"
    wav[torch.max(wav) > 1.0] = 1.0
    wav[torch.min(wav) < -1.0] = -1.0
    wav = wav.to(device)
    wav = wav.unsqueeze(0)
    return wav

def predict_mos(wav_fpath, model, device):
    wav = load_wav(wav_fpath, device)

    with torch.no_grad():
        mos = model(wav)
        mos = mos.cpu().numpy()
    return mos[0]

def predict_mos_dir(wav_dir, model, device, save_in_dir=True):
    mos_dict = {}
    print("Predicting MOS for files in", wav_dir)
    wav_fpaths = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]
    for wav_fpath in tqdm.tqdm(wav_fpaths):
        wav = load_wav(wav_fpath, device)
        wav_fname = os.path.basename(wav_fpath)
        with torch.no_grad():
            mos_dict[wav_fname] = float(model(wav).cpu().numpy()[0])
    mos = np.array(list(mos_dict.values()))
    print("mean mos:", np.mean(mos))
    print("std mos:", np.std(mos))
    print("total audio", len(mos))

    if save_in_dir:
        json.dump(mos_dict, open(os.path.join(wav_dir, 'predicted_mos.json'), 'w'))
    return mos

def main():
    args = parse_args()
    model = get_model(cp_path = args.fairseq_base_model, my_checkpoint = args.finetuned_checkpoint, device=args.device)
    device = torch.device(args.device)

    if args.wav_fpath is not None:
        mos = predict_mos(args.wav_fpath, model, device)
        print("predicted mos:", mos)
    elif args.wav_dir is not None:
        mos = predic_mos_dir(args.wav_dir, model, device)
        # print("predicted mos:", mos)

if __name__ == '__main__':
    main()