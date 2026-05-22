function vad = vad_opt(x, fs, thr, min_on)
% Calculated a perfect VAD from a clean speech signal x.
% INPUT
%   x: Speech input vector (dim: T x 1)
%   fs: Sample rate (Hz)
%   thr: Threshold relative to maximum value in signal (dB)
%   min_on: Minimum interval of voice activity (seconds)
%
% OUTPUT
%   vad: Sample-wise binary voice activity detection
%
% Nico Goessling, 2017

if nargin <= 3
     min_on = 100e-3;
     if nargin <= 2
        % if normalizing by max(abs()), then set to about -30, 
        % if normalizing by var(x), then set to about 21 
        thr = 21; 
     end
end

% Normalisation
x = x - mean(x);
x = x ./ var(x); % --- var seems more reliable
% x = x ./ max(abs(x));

% Set VAD
vad = zeros(size(x,1),1);
vad(10*log10(mean(x.^2,2)) >= thr) = 1;

min_period = round(min_on*fs);
idx  = find(vad == 1);
for k = 1:length(idx)
     vad(idx(k):idx(k)+min_period) = 1;
end
vad = vad(1:length(x));

end
