function [X,f,xBlock,timeIndices] = calc_STFT(x,fs,nfft,noverlap,wType,NFFT)
%CALC_STFT short-time fourier transform using OLA. The STFT uses a
%sqrt(hann(nfft)) window.
%
% INPUT:
%   x           : input time signal(s) (samples x channels)
%   fs          : sampling rate
%   nfft        : FFT size
%   noverlap    : frame overlap; default: 2 (50%)
%
% OUTPUT:
%   X           : STFT matrix (channels x bins x frames)
%   f           : frequency vector for bins
%   xBlock      : individual time frames (nfft x channels x frameLength)
%   timeIndices : time indices corresponding to xBlock (nft x frameLength)
%
%   See also: calc_ISTFT, fft.

% Author: Daniel Marquardt & Nico Goessling
% Date: 27.05.2016
% Modified by Daniel Fejgin on 02.07.2019 (added variables "xBlock" and "timeIndices")
% Modified by Daniel Fejgin on 16.09.2019 (added options for window)

if nargin < 4
    noverlap = 2;
end

% synthesis window
if strcmpi(wType,'sqrtHann')
    window  = sqrt(hann(nfft, 'periodic'));
elseif strcmpi(wType,'Hann')
    window  = hann(nfft, 'periodic');
elseif strcmpi(wType,'Hamming')
    window  = hamming(nfft, 'periodic');
else
    error(['Unsupported window type "', wType,'"'])
end

% use only half FFT spectrum
N_half = nfft / 2 + 1;

% get frequency vector
f = 0:(fs / 2) / (N_half - 1):fs / 2;

% init
L = floor((length(x) - nfft + (nfft / noverlap)) / (nfft / noverlap));
X = zeros(floor(NFFT / 2) + 1, L, size(x,2));
timeIndices = nan(nfft,L);
xBlock = nan(nfft,size(x,2),L);% time x channel x frame

% OLA processing
for l = 0:L-1 % Frame index
    timeIndices(:,l+1) = (floor(l*(nfft / noverlap) + 1):floor(l*(nfft / noverlap) + nfft)).';
    x_frame = x(floor(l*(nfft / noverlap) + 1):floor(l*(nfft / noverlap) + nfft),:);
    xBlock(:,:,l+1) = x_frame;
    x_windowed = x_frame.*repmat(window, 1, size(x_frame,2));
    X_frame =  fft(x_windowed,NFFT,1);
    X(:,l+1,:) = X_frame(1 : floor(NFFT / 2) + 1, :);
%     X(:,l+1,:) = X_frame(1 : floor(size(X_frame,1) / 2) + 1, :);
end

X = permute(X, [3 1 2]);

end