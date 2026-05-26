% -- Online Incremental Averaging Method for Graph-Based TDOA Estimation --
% 
% Code Author: Klaus Brümann
% Email: klaus.bruemann@uni-oldenburg.de
% Last edited: 26 May 2026
% 
% This code is an optimized and refined implementation of the TDOA 
% estimation method proposed in [1] (and some baseline method discussed in 
% the paper). This implementation uses no source tracking, it simply 
% estimates the TDOAs in each frame based on recursively averaged cross-
% power spectral densities. 
% 
% We have supplied following example scenarios and corresponding signals:
% - Scenario 1:    one stationary source,     compact microphone array
% - Scenario 2:    one moving source,         compact microphone array
% - Scenario 3:    one stationary source,     distributed microphone array
% - Scenario 4:    one moving source,         distributed microphone array
% 
% Exemplary clean speech signals are from TIMIT database [2].
% Clean speech signals were convolved with RIRs using RAZR [3] (adapted for
% moving sources).
% Spherically isotropic babble noise is generated using [4].
% 
% The user can mix spatially isotropic babble noise at a desired SNR by
% setting SNR_dB.
% Also, the recursive smoothing time 'Smoothing_time' can be changed for
% your application. 
% 
% If you assume that the microphone geometry is unknown, define:
% assume_mic_geometry_known_yn = false;
% 
% Standard GCC-PHAT algorithm parameters are stored in the struct "Params".
% E.g., if the target is a narrowband signal with known frequency limits, 
% the lower and upper considered frequency can be varied by changing: 
% Params.f_low and Params.f_high;
% For speech signals, the considered methods work well using all 
% frequencies between 0 - 8 kHz. 
% 
% References:
% [1] K. Brümann, K. Yamaoka, N. Ono, and S. Doclo, "Incremental averaging
% method to improve graph-based time-difference-of-arrival estimation,"
% in Proc. IEEE Workshop on Applications of Signal Processing to Audio and
% Acoustics (WASPAA), Lake Tahoe, CA, USA, 2025.
%
% [2] J. S. Garofolo, L. F. Lamel, W. M. Fisher, J. G. Fiscus, D. S. 
% Pallett, and N. L. Dahlgren, TIMIT Acoustic-Phonetic Continuous Speech 
% Corpus LDC93S1. Philadelphia, PA, USA: Linguistic Data Consortium, 1993.
%
% [3] T. Wendt, S. van de Par, and S. D. Ewert, "A computationally 
% efficient and perceptually plausible algorithm for binaural room impulse 
% response simulation," J. Audio Eng. Soc., vol. 62, no. 11, pp. 748-766, 
% 2014.
%
% [4] E. A. P. Habets, I. Cohen, and S. Gannot, "Generating nonstationary
% multisensor signals under a spatial coherence constraint," J. Acoust.
% Soc. Am., vol. 124, no. 5, pp. 2911-2917, Nov. 2008.