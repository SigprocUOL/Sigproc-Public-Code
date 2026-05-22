function [X] = func_calc_STFT(x,fs,K,R,Ana_window,NFFT)

if length(x) == size(x,2) % arrange so that rows: signal dim, cols: mics.
    x = x';
end
M = size(x,2);

X = repmat(permute(calc_STFT(x(:,1),fs,K,K/R,Ana_window,NFFT),[1,4,2,3]),M,1,1,1); % preallocation

% Y = repmat(permute(calc_STFT(y_t(:,1),fs,K,K/R,Ana_window,NFFT),[1,4,2,3]),size(y_t,2),1,1,1); % preallocation
for channel_ind = 2:M % floor((size(x,1) - nfft + (nfft / noverlap)) / (nfft / noverlap));
    X(channel_ind,1,:,:) = permute(calc_STFT(x(:,channel_ind),fs,K,K/R,Ana_window,NFFT),[1,4,2,3]);
    % Y(channel_ind,1,:,:) = permute(calc_STFT(y_t(:,channel_ind),fs,K,K/R,Ana_window,NFFT),[1,4,2,3]);
end


end