## Sogang Capstone Design 2021

Edited MLPSinger (https://github.com/neosapience/mlp-singer) to support GUI interface and various input file paths.
By calling functions in wrapper.py, Training or Inference will be started as a subprocess.
If you want to cease the progress on runtime, just call out kill() to the return of the function. (Popen)
It requires each paths of files (.mid, .txt, .wav), otherwise it will be executed as default (data/raw)
