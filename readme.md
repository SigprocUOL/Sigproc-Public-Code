This work is licensed under a [CC BY-NC-SA 3.0 US](https://creativecommons.org/licenses/by-nc-sa/3.0/us/) license.

In this repository, you can find a minimal implementation of the deep multi-frame minimum variance distortionless response (MFMVDR) filter, which has been published in
M. Tammen, S. Doclo, Deep multi-frame MVDR filtering for single-microphone speech enhancement, in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Jun. 2021, pp. 8443-8447. The corresponding preprint is available at [arXiv](https://arxiv.org/abs/2011.10345).

The repository also includes the deep MFMVDR model weights used in the evaluation of the above submission. Note that `git lfs` is required to download these weights.

Audio samples of a trained version of this model are available at [this link](https://uol.de/en/sigproc/research/audio-demos/multi-frame-speech-enhancement/deep-multi-frame-mvdr-filtering-for-single-microphone-speech-enhancement).

## Installation Instructions
To install the required dependencies in a new conda environment (GPU-enabled PyTorch version), use the following:
```
conda create -n dmfmvdr -y 
conda activate dmfmvdr
# PyTorch with GPU-support
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch -y
# OR PyTorch without GPU-support
conda install pytorch==1.6.0 cpuonly torchvision==0.7.0 cpuonly -c pytorch -y
conda install pip
pip install soundfile pypesq
```
