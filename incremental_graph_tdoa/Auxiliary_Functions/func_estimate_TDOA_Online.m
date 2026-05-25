function [TDOA_est_mtx,CPSD] = func_estimate_TDOA_Online(Y_block,CPSD,Params,TDOA_Method)
M = size(Y_block,1);
K = Params.NIFFT/Params.GCC_Resampling;

% Taper lowest and highest frequencies to 0.
freqz = permute(Params.fs*((1:(K/2+1))-1)/K,[1,3,2,4]);
w = ones(size(freqz));
flow(1) = 600; % CPSDs below this frequency will have 0 magnitude
flow(2) = 1000; % start tapering CPSD below this frequency
flow(flow(1)>=flow(2)) = flow(2);
idx = freqz > flow(1) & freqz < flow(2); 
w(:,:,idx,:) = (0.5*(1 - cos(pi*(freqz(idx)-flow(1))/(flow(2)-flow(1)))));
w(:,:,freqz <= flow(1),:) = 0;
fhigh(1) = 7800; % start tapering CPSD above this frequency
fhigh(2) = 8000; % CPSDs above this frequency will have 0 magnitude
fhigh(fhigh(2)<=fhigh(1)) = fhigh(1);
idx = freqz > fhigh(1) & freqz < fhigh(2); 
w(:,:,idx,:) = (0.5*(1 + cos(pi*(freqz(idx)-fhigh(1))/(fhigh(2)-fhigh(1)))));
w(:,:,freqz >= fhigh(2),:) = 0;

% OnlyRef = true;
% Ref_ind = 1; % microphone closest to centre;

% Update recursive estimate of cross-power spectral density
% CPSD = mean( Y_block.*permute(conj(Y_block(Ref_ind,1,:,:)),[2,1,3,4]) ,4); % .* Weight(:,:,:,ll);

CPSD = Params.lambda*CPSD + (1-Params.lambda)*Y_block.*permute(conj(Y_block),[2,1,3]);% .* Weight(:,:,:,ll);
% CPSD = mean( permute(sqrt(hann(size(Y_block,4),'symmetric')),[4,3,2,1]) .* Y_block.*permute(conj(Y_block),[2,1,3,4])  ,4); % .* w_WG
% CPSD_Pow = squeeze(sum(CPSD_full .* eye(size(CPSD_full,1)), [1 2]));
% CPSD_Pow = squeeze(func_pagetrace(CPSD));

% Phase-transform
mag = abs(CPSD);
eps_static = 3.5e-4; % 
eps_dynamic = 7.5e-4 * prctile(mag,85,3); % 0; % median(mag, 3); % 
CPSD_PHAT = CPSD ./ max(mag, eps_static+eps_dynamic); % (mag+eps_static+eps_dynamic); % (abs(CPSD)+eps_static+eps_dynamic);
CPSD_PHAT = CPSD_PHAT .* w; % Taper frequencies close to Nyquist
GCC_PHAT = ifftshift(  ifft(cat(3,CPSD_PHAT, ...
    zeros(M,M,Params.NIFFT-K,1), ...%zeros(M,length(Ref_ind),NFFT-K,1), ...
    conj(CPSD_PHAT(:,:,(end-1):-1:2))), Params.NIFFT, 3,'symmetric') ,3);
GCC_PHAT = GCC_PHAT*Params.NIFFT/(sum(w) + sum(w(1,1,2:(end-1)))); % undo scaling of tapering and zero-padding

lags_3D =  reshape(( (0:(Params.NIFFT-1)) - Params.NIFFT/2) / Params.GCC_Resampling / Params.fs , 1, 1, []);
select_TDOAs = abs(lags_3D) <= (Params.TDOA_Limits(:,:));
GCC_PHAT(~select_TDOAs) = 0; % -Inf; % 0;

% [TDOA_est_mtx, Reliability_1, Reliability_2] = func_Optimized_Findpeaks3(GCC_PHAT, lags_3D, num_candidates);
[TDOA_est_mtx, Reliability_1, ~] = func_Optimized_Findpeaks3_(GCC_PHAT, lags_3D, Params.TDOA_Limits, Params.num_candidates);

switch TDOA_Method
    case {'Baseline'}
        ref_idx = randi(M,1);
        TDOA_est_mtx = TDOA_est_mtx(:,ref_idx) + TDOA_est_mtx(ref_idx,:);
    case {'R'}
        % Reliability_1((1:(M+1):(M^2)) + ((1:Params.num_candidates) -1)' * M^2 ) = Inf;
        [~,Refmic] = max(min(Reliability_1,[],1));
        TDOA_est_mtx = (TDOA_est_mtx(:,Refmic) + TDOA_est_mtx(Refmic,:));
    case {'MST'}
        TDOA_est_mtx = func_Compute_MST(Reliability_1,TDOA_est_mtx);
    case {'MST+'}
        %% Incremental TDOA re-estimation
        TDOA_est_mtx = func_MSTplus(Reliability_1,CPSD,GCC_PHAT,lags_3D,M,Params.fs,Params.TDOA_Limits,select_TDOAs,w);
end
TDOA_est_mtx((1:(M+1):(M^2)) + ((1:Params.num_candidates) -1)' * M^2 ) = 0;
end
