function [TDOA_Mtx_full] = func_MSTplus(...
    Reliability_full,CPSD,GCC_PHAT,lags,N,fs,Upper_TDOA,select_TDOAs,w)%,Extra_Vars)
K = (size(CPSD,3)-1)*2;
NIFFT = size(GCC_PHAT,3);

Reliability_full = tril(Reliability_full,-1) + tril(Reliability_full,-1).';

[Reliability] = deal(tril(Reliability_full,-1));

[TDOA_Mtx,TDOA_Mtx_full] = deal(zeros(size(Reliability_full)));


%% 1) Start by picking the microphone pair with the most reliable

% GCC-PHAT function (based on pair-wise reliability)
[~,max_ind] = max(Reliability(:,:,1,1),[],'all');
[r_max,c_max] = ind2sub([N,N],max_ind);

%s elect_TDOAs = abs(lags(1,1,:)) <= Upper_TDOA(r_max,c_max);
% TDOA_Mtx(r_max,c_max) = func_Optimized_Findpeaks3(GCC_PHAT(r_max,c_max,:,1), lags(1,1,:), 1, 0);
TDOA_Mtx(r_max,c_max) = func_Optimized_Findpeaks3_(GCC_PHAT(r_max,c_max,:,1), lags(1,1,:), Upper_TDOA(r_max,c_max), 1);
% end

TDOA_Mtx(c_max,r_max) = -TDOA_Mtx(r_max,c_max);

% Reliability(r_max,c_max) = 0;
Reliability_full(r_max,c_max) = 0;
Reliability_full(c_max,r_max) = 0;

% Regularization for phase-normalization
eps_static = 3.5e-4; % 

consistency_idx_vec = sort([r_max;c_max],'ascend');
while ~all(Reliability_full==0,'all') %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Reliability_Augmented = zeros(size(Reliability_full));
    % Determine most reliable microphone pair for augmentation
    Reliability_Augmented(:,consistency_idx_vec) = Reliability_full(:,consistency_idx_vec); % technically Reliability_full(:,consistency_idx_vec)+Reliability_full(:,consistency_idx_vec)'; but no need
    [~,aug_max_ind] = max(Reliability_Augmented,[],'all');
    [r_aug,c_aug] = ind2sub([N,N],aug_max_ind);

    consistency_idx_vec = sort(unique([consistency_idx_vec; r_aug;c_aug]),'ascend');
    column_consistency_idx = consistency_idx_vec((consistency_idx_vec ~= r_aug) & (consistency_idx_vec ~= c_aug));

    CPSDs(:,1) = CPSD(r_aug,c_aug,:,1);

    % % ---
    for ii = 1:length(column_consistency_idx)
        % These are the time shifts which we will apply to adjacent columns
        % of the augmented row.
        tdoa_shift = TDOA_Mtx(column_consistency_idx(ii),c_aug); % Non-integer phase shift

        %% Averaging CPSDs:
        frequencies_ = (0:(K/2)).' * fs / (K); % *intfactor % Frequency bins for positive frequencies
        CPSDs_full(:,1) = CPSD(r_aug,column_consistency_idx(ii),:,1);%cat(3, CPSD(r_aug,column_consistency_idx(ii),:,1), ...
            %conj(CPSD(r_aug,column_consistency_idx(ii),(end-1):-1:2,1)) );
        CPSDs(:,ii+1) = CPSDs_full .* exp(-1j * 2 * pi * frequencies_ * tdoa_shift);
    end

    Mean_CPSD = permute( mean(CPSDs,2) , [3,2,1,4]);
    % Consistent_CPSD_PHAT = Mean_CPSD./(abs(Mean_CPSD)+1e-9);
    % Consistent_CPSD_PHAT([1,end],:) = 1;
    
    mag = abs(Mean_CPSD);
    eps_dynamic = 7.5e-4 * prctile(mag,85,3); % 0; % median(mag, 3); % 
    Consistent_CPSD_PHAT = Mean_CPSD./ max(mag, eps_static+eps_dynamic); % (mag+eps_static+eps_dynamic); % (abs(CPSD)+eps_static+eps_dynamic);
    Consistent_CPSD_PHAT = Consistent_CPSD_PHAT .* w; % Taper frequencies close to Nyquist
    % Consistent_CPSD_PHAT(:,:,1) = 1; % Fix DC
    Consistent_GCC_PHAT = ifftshift(  ifft(cat(3,Consistent_CPSD_PHAT,...
        zeros(1,1,NIFFT-K,1), ...
        conj(Consistent_CPSD_PHAT(:,:,(end-1):-1:2))), NIFFT, 3,'symmetric') ,3); 
    Consistent_GCC_PHAT = Consistent_GCC_PHAT*NIFFT/(sum(w) + sum(w(1,1,2:(end-1)))); % undo scaling of tapering and zero-padding
    
    Consistent_GCC_PHAT(1,1,~select_TDOAs(r_aug,c_aug,:)) = 0; % -Inf; % 0;
    
    TDOA_est = func_Optimized_Findpeaks3_(Consistent_GCC_PHAT, lags(1,1,:), Upper_TDOA(r_aug,c_aug), 1); % max(Upper_TDOA,[],'all'), 1, 0); % 

    % fill out augmented row of TDOA matrix
    % TDOA_Mtx(r_aug,c_aug) = base_time(locs(1));
    idx_excl_r = consistency_idx_vec((consistency_idx_vec ~= r_aug));
    % idx_excl_c = consistency_idx_vec((consistency_idx_vec ~= c_aug));
    % idx = consistency_idx_vec((consistency_idx_vec ~= r_aug)&(consistency_idx_vec ~= c_aug));
    for ii = 1:length(idx_excl_r)
        % delta_TDOA = TDOA_Mtx(idx_excl_r(ii),c_aug)-TDOA_Mtx(idx_excl_r(ii),c_aug);
        TDOA_Mtx(r_aug,idx_excl_r(ii)) = TDOA_est - TDOA_Mtx(idx_excl_r(ii),c_aug);
        TDOA_Mtx(idx_excl_r(ii),r_aug) = - TDOA_Mtx(r_aug,idx_excl_r(ii));
    end

    Reliability_full(consistency_idx_vec,consistency_idx_vec) = 0;
end
TDOA_Mtx_full(:,:,1,1) = TDOA_Mtx;


end