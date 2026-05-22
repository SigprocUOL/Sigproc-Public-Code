function [VAD_f_ind] = func_Compute_VAD_idx(x_t,M_L,fs,K,F,Ana_window,NFFT,plot_yn,ylims)
VAD = logical(vad_opt(x_t(:,1:M_L), fs, 21, 0.05)); %vad_opt(mean(x_t(:,1:M_L),2), fs, -22, 0.08));

VAD_f_ind = permute(any( db( ...
    func_calc_STFT(VAD,fs,K,F,Ana_window,NFFT) ...
    ) > -100,3),[3,4,1,2]);
% VAD_f_ind = permute(VAD_f,[3,4,1,2]);


% add delayed VAD offset:
offset_len = 0.15*fs/K;
% find transitions from 1 to 0:
Transitions = reshape((find(([VAD_f_ind';0]==1) & ([0;VAD_f_ind']==0)) + (0:ceil(offset_len)))',[],1);
Transitions(Transitions>length(VAD_f_ind)) = [];
VAD_f_ind(Transitions) = 0;

if plot_yn
    x = 0:(size(VAD_f_ind,2));
    idx = find([diff(~VAD_f_ind)]);
    if mod(length(idx),2) && ~VAD_f_ind(1)   % if 0,1,0,1,0 - odd length
        idxm = reshape([0, idx],2,[]);
    elseif mod(length(idx),2) && VAD_f_ind(1) % if 1,0,1,0,1 - odd length
        idxm = reshape([idx,length(VAD_f_ind)],2,[]);
    elseif ~mod(length(idx),2) && ~VAD_f_ind(1) % 0,1,0,1,0,1
        idxm = reshape([0, idx ,length(VAD_f_ind)],2,[]);
    else % 1,0,1,0,1,0
        idxm = reshape(idx,2,[]);
    end

    % ylims = ylim;
    hold on
    for k = 1:size(idxm,2)
        xrng = (idxm(1,k) : idxm(2,k))+1;
        patch([x(xrng) flip(x(xrng))]+0.5, [min(ylims)*ones(size(x(xrng))), flip(max(ylims)*ones(size(x(xrng))))], ...
            'k', 'FaceAlpha',0.7)
    end
    ylim(ylims);
    hold off;
end
end