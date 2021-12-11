import os
import subprocess

# Preprocess can be called independently
# or while on Train if prep is set, preprocess will be executed.
def preprocess(midpath, txtpath, wavpath):
    pre = subprocess.Popen(["python", "-m", "data.serialize", "--mid_path", midpath, "--txt_path", txtpath, "--wav_path", wavpath])
    return pre

# Infer requires paths of Midi and TXT files.
# Optionally checkpoint path, song to infer, and path to save wav can be passed (str)
def infer(midpath, txtpath, checkpoint=None, song=None, savepath=None):
    args = ["python", "inference.py", "--mid_path", midpath, "--txt_path", txtpath]
    if checkpoint:
        args.extend(["--checkpoint_path", checkpoint])
    if song:
        args.extend(["--song_path", song])
    if savepath:
        args.extend(["--save_path", savepath])
    inf = subprocess.Popen(args)
    return inf

# Train basically needs no arguments, but only when preprocess is required.
# If preprocess is not done yet, set prep to True and pass paths of input files.
# config is for option --config_path of the original program
def train(midpath=None, txtpath=None, wavpath=None, prep=True, config=None):
    if prep:
        preprocess(midpath, txtpath, wavpath)
    args = ["python", "train.py"]
    if config:
        args.extend(["--config_path", config])
    tr = subprocess.Popen(args)
    return tr