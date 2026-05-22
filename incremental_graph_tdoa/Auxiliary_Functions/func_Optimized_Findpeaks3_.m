function [TDOA_est, Reliability_1, Reliability_2] = func_Optimized_Findpeaks3_(Signal, tau_lags_3D, Upper_TDOA, num_candidates)
% Signal: [M, M, NFFT]
% tau_lags_3D: [1, 1, NFFT]
% Upper_TDOA: [M, M] matrix of physical limits

[M1, M2, N] = size(Signal);
num_cands_original = num_candidates;
num_candidates = max(2, num_candidates);

TDOA_est = zeros(M1, M2, num_candidates);
val_max  = zeros(M1, M2, num_candidates);
TempSignal = Signal;
dt = tau_lags_3D(2) - tau_lags_3D(1);

% Constant grid for 3D masking
k_indices = reshape(1:N, 1, 1, []);
k_diff = reshape(1:N-1, 1, 1, []);
page_size = M1 * M2;
spatial_idx = reshape(1:page_size, M1, M2);

if nargout > 2
    C = max(2,num_candidates);
else 
    C = num_candidates;
end
for cands_ind = 1:C
    % Update Mask for next candidate iteration
    if cands_ind > 1
        peak_idx_3D = reshape(ind_max, M1, M2, 1);

        grad3 = diff(TempSignal, 1, 3);

        finite_pairs = isfinite(TempSignal(:,:,1:end-1)) & ...
            isfinite(TempSignal(:,:,2:end));

        % Left side of peak:
        % slope k -> k+1 should be >= 0 when moving toward peak
        left_ok = (grad3 >= 0) & finite_pairs;

        % Right side of peak:
        % slope k -> k+1 should be <= 0 when moving away from peak
        right_ok = (grad3 <= 0) & finite_pairs;

        % Last invalid left-side gradient before the peak
        bad_left_before_peak = (~left_ok) & (k_diff < peak_idx_3D);
        left_edge = max(bad_left_before_peak .* k_diff, [], 3) + 1;

        % First invalid right-side gradient at or after the peak
        bad_right_after_peak = (~right_ok) & (k_diff >= peak_idx_3D);
        right_edge_candidates = k_diff + N * (~bad_right_after_peak);
        right_edge = min( min(right_edge_candidates, [], 3) , N);

        % Mask the complete lobe
        mask = isfinite(TempSignal) & ...
            (k_indices >= reshape(left_edge, M1, M2, 1)) & ...
            (k_indices <= reshape(right_edge, M1, M2, 1));
        
        % Flatten each local lobe to the lower of its two edge values
        left_lin  = spatial_idx + (left_edge  - 1) * page_size;
        right_lin = spatial_idx + (right_edge - 1) * page_size;

        edge_floor = min(TempSignal(left_lin), TempSignal(right_lin));
        edge_floor_3D = repmat(edge_floor, 1, 1, N);

        TempSignal(mask) = edge_floor_3D(mask);
    end

    % Find discrete peak
    [v_raw, ind_max] = max(TempSignal, [], 3); % many samples of TempSignal are already set to 0

    % Initialize current candidate with discrete values
    discrete_lags = tau_lags_3D(ind_max);
    TDOA_est(:,:,cands_ind) = discrete_lags;
    val_max(:,:,cands_ind) = v_raw;

    % Vectorized Parabolic Interpolation
    % Only attempt if peak is not on the absolute NFFT window edge
    valid_peak = (ind_max > 1) & (ind_max < N);
    
    if any(valid_peak, 'all')
        idx_spatial = find(valid_peak);
        
        % Get linear indices for the volume
        % mid: peak, alpha: peak-1, gamma: peak+1
        mid = idx_spatial + (ind_max(idx_spatial) - 1) * page_size;
        
        alpha = Signal(mid - page_size); 
        beta  = Signal(mid);
        gamma = Signal(mid + page_size);

        denom = (alpha - 2*beta + gamma);
        denom(denom == 0) = -1e-12; 
        p = 0.5 * (alpha - gamma) ./ denom;
        
        % Interpolated TDOA
        interp_TDOA = discrete_lags(idx_spatial) + p * dt;
        
        % --- BOUNDARY LOGIC ---
        % Only update if: 
        % 1. It's a valid concave peak (denom < 0)
        % 2. The result is strictly within the +/- Upper_TDOA bounds
        limit = Upper_TDOA(idx_spatial);
        keep_interp = (denom < 0) & (interp_TDOA > -limit) & (interp_TDOA < limit);
        
        if any(keep_interp)
            % Indices relative to the full M x M matrix
            final_idx = idx_spatial(keep_interp);
            
            % Apply TDOA update
            temp_TDOA = TDOA_est(:,:,cands_ind);
            temp_TDOA(final_idx) = interp_TDOA(keep_interp);
            TDOA_est(:,:,cands_ind) = temp_TDOA;
            
            % Apply refined magnitude update
            temp_vals = val_max(:,:,cands_ind);
            temp_vals(final_idx) = beta(keep_interp) - 0.25 * (alpha(keep_interp) - gamma(keep_interp)) .* p(keep_interp);
            val_max(:,:,cands_ind) = temp_vals;
        end
    end % else, if peak is not valid, simply select boundary
end

% TDOA Reliability Metrics
if nargout > 1
    Reliability_1 = val_max(:,:,1);
    Reliability_1((1:(M1+1):(M1*M2))) = 1;
end

if nargout > 2
    % Ratio-based reliability (closer to 1 is more reliable)
    Reliability_2 = max(1 - val_max(:,:,2) ./ (Reliability_1 + 1e-12), 1e-6);
    Reliability_2((1:(M1+1):(M1*M2))) = 1;
% else
%     Reliability_2 = ones(M1, M2) * 1e-6;
%     Reliability_2((1:(M1+1):(M1*M2))) = 1;
end

% Rels_row = min(Reliability_1);
% [~,Ref_ind] = max(Rels_row);
% TDOA_est = (TDOA_est(:,Ref_ind,:) + TDOA_est(Ref_ind,:,:))/2; % Reference microphone-based estimate

TDOA_est = TDOA_est(:,:,1:num_cands_original);

end