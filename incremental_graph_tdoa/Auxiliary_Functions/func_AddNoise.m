function [y_t] = func_AddNoise(x_t, SNR_dB, M_mtx, M_L, fs, c)

Param.x_t = x_t; 
Param.N = size(M_mtx,2); 
Param.fs = fs; 
Param.c = c; 
Param.K = 1024; 
Param.R = Param.K/4;
[Babble] = func_gen_babble_speech_moving(Param, M_mtx);

% Mix noise
offset = 0.05;%0.01;%05; %0.05;
vad = logical(vad_opt(x_t, fs, 20, offset));
noise_data = Babble(1:size(x_t,1),:);

% SNR_tmp = 10 * log10(var(x_t(vad, :), [], 1) ./ var(noise_data(vad, :), [], 1));%zeros(Param.N,1);%size(z_t,2),1);
% SNR_current_dB = mean(SNR_tmp(1:M_L)); % average SNR over all mics rather than just the first mic. (mean of dB is equal to geometric mean of powers)
Px = mean(var(x_t(vad, 1:M_L), [], 1));
Pn = mean(var(noise_data(vad, 1:M_L), [], 1));
SNR_current_dB = 10*log10(Px / Pn);

scalingFactor = db2mag(SNR_current_dB - SNR_dB);

% test:
% noise_data(:,1:M_L) = noise_data(:,1:M_L)*3;

y_t = x_t + scalingFactor * noise_data;


end