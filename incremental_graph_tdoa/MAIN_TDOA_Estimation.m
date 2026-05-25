% -- Online Incremental Averaging Method for Graph-Based TDOA Estimation --
% 
% Code Author: Klaus Brümann
% Email: klaus.bruemann@uni-oldenburg.de
% Last edited: 25 May 2026
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


% Preliminaries (restores path to default and adds relevant folders).
clear;
restoredefaultpath;
rng(1337);
addpath(genpath('Auxiliary_Functions'));%,genpath('Data')); 

%% --- Import acoustic signal and relevant data for results processing ---
list = {'stationary source, compact array',...
    'moving source, compact array',...
    'stationary source, distributed array',...                   
    'moving source, distributed array'};
[indx] = listdlg('PromptString',{'Select a scenario'},'SelectionMode','single','ListString',list);
Scenario = ['Scenario ' num2str(indx)]; % Options: {'Scenario 1', 'Scenario 2'} (described above)
Filename = Scenario; 
Filename(Filename==' ') = [];
Data = load(fullfile('Data',[Filename '.mat']));
[M_mtx,s_traj] = deal(Data.M_mtx,Data.s_traj);
[x_t,Params.fs] = audioread(fullfile('Data',[Filename '.wav'])); % [signal, sampling frequency [Hz]]

switch Scenario
    case 'Scenario 1' % Exemplary stationary source scenario, compact microphone array
        SNR_dB = -5; % signal-to-noise ratio [dB]
        Moving_source_yn = false;
        Array_type = 'Compact';
        assume_mic_geometry_known_yn = true; % microphone geometry is available (for setting upper TDOA limits)
    case 'Scenario 2' % Exemplary moving source scenario, compact microphone array
        SNR_dB = 0; % signal-to-noise ratio [dB]
        Moving_source_yn = true;
        Array_type = 'Compact';
        assume_mic_geometry_known_yn = true; % microphone geometry is available (for setting upper TDOA limits)
    case 'Scenario 3' % Exemplary stationary source scenario, distributed microphone array
        SNR_dB = -10; % signal-to-noise ratio [dB]
        Moving_source_yn = false;
        Array_type = 'Distributed';
        assume_mic_geometry_known_yn = true; % microphone geometry is available (for setting upper TDOA limits)
    case 'Scenario 4' % Exemplary moving source scenario, distributed microphone array
        SNR_dB = 5; % signal-to-noise ratio [dB]
        Moving_source_yn = true;
        Array_type = 'Distributed';
        assume_mic_geometry_known_yn = true; % microphone geometry is available (for setting upper TDOA limits)
    otherwise
        error('If you select your own signal, please define an appropriate SNR and define whether source is moving or not.') ...
        % SNR_dB = Inf; % [dB]
        % Moving_source_yn = true;
        % assume_mic_geometry_known_yn = true; % is microphone geometry available (for setting upper TDOA limits)? [true / false]
end

% CPSD smoothing time constant
if ~Moving_source_yn % if source is stationary:
    Smoothing_time = 1.2; % temporal CPSD averaging [s]
else % if source is moving:
    Smoothing_time = 0.3; % temporal CPSD averaging [s]
end
% For fast/jittery source movement, use: Smoothing_time = [0.03 - 0.10] s. 
% For slow/stationary scenarios, more smoothing works better.
% For moving scenarios, the more smoothing, the fewer outliers there are
% in the TDOA estimation, but the less accurate they become (due to over-
% reliance on out-of-date CPSDs). 

% To listen to the spatial properties of the moving signal, listen to e.g., channel 3 and 4:
% soundsc(x_t(:,[3,4]),Params.fs)

[x_t,M] = func_check_signal_dimensions(x_t); % signal dimensions of x_t: (samples x Num. microphones).
c = 343; % speed of sound [m/s]

% Add spatially isotropic noise to the reverberant speech signal: 
y_t = func_AddNoise(x_t, SNR_dB, M_mtx, M, Params.fs, c);

% To listen to the noisy signal:
% soundsc(y_t(:,[3,4]),Params.fs)

%% Variable definition and initialization:
% STFT parameters
K = 2048; % Frame length: should be long enough to capture enough correlated speech between microphones
F = K/4; % Frame shift 
Ana_window = 'sqrtHann'; % STFT analysis window
[Params.GCC_Resampling, Params.NIFFT] = func_Compute_GCC_Resampling_from_Frame_Length(K,Params.fs); % This function automatically determines some GCC-PHAT upsampling factors (using zero-padding of CPSDs) based on frame length.

if assume_mic_geometry_known_yn
    D = squeeze(vecnorm(M_mtx - permute(M_mtx',[2,3,1]))); % Inter-microphone distances for upper TDOA limits
    Params.TDOA_Limits = D/c + 1/(Params.fs*Params.GCC_Resampling); % Geometric limits for minimum and maximum possible TDOAs
else
    Params.TDOA_Limits = Inf(M,M); %#ok<UNRCH>
end

[Params.lambda,~] = func_Compute_lambda(Smoothing_time,F,Params.fs); % input: Smoothing_time, Frame_shift, fs

Params.num_candidates = 1; % 1, ..., 3 or so
max_num_sources = 1;

Implementations = {'Baseline','R','MST','MST+'}; % TDOA Estimation Methods
% Baseline: Selects a random reference microphone in each frame
% R: Selects reference microphone with the highest minimal reliability
% MST: Minimum spanning tree
% MST+: Proposed incremental averaging method
Processing_time_method = zeros(length(Implementations),1);
TDOA_est_mtx_full = cell(1,length(Implementations));

% If you want to edit the minimum and maximal frequencies considered for
% the GCC-PHAT function, edit these values:
Params.f_low = 0;
Params.f_high = Params.fs/2;

% Method selection
disp(['Processing ' num2str(round(size(x_t,1)/Params.fs,2)) ' s signal.'])
for Implementation_idx = 1:length(Implementations)
    Implementation = Implementations{Implementation_idx};
    
    % Redefine starting indices for every method, to begin processing from 
    % start of signal:
    frame_idx = 1;
    signal_indices = 1:K;

    CPSD = zeros(M,M,K/2+1); % Temporally-averaged cross-power spectral density/covariance matrix

    TDOA_est_mtx_full{Implementation_idx} = zeros(M,M,1,floor(size(y_t,1)/F));
    
    % -------------- Online processing: --------------
    while signal_indices(1) < size(y_t,1) 
        if signal_indices(end) > size(y_t,1) % if approaching end of signal:
            y_frame = y_t(signal_indices(1):size(y_t,1),:);
            y_frame = [y_frame; zeros(K-size(y_frame,1),M)]; %#ok<AGROW>
        else % Default:
            y_frame = y_t(signal_indices,:);
        end

        tic;
        [Y] = calc_STFT(y_frame,Params.fs,K,K/F,Ana_window,K); % (signal,fs,nfft,overlap_factor,wType,NFFT)

        %% ----------------------------------------------------------------
        %% DO PROCESSING HERE ---------------------------------------------
        %% ----------------------------------------------------------------
        [TDOA_est_mtx,CPSD] = func_estimate_TDOA_Online( ... % Online TDOA estimation function
            permute(Y, [1,3,2,4]), ... % STFT frame in format: (Mics x 1 x Frequencies)
            CPSD, ... % Cross-power spectral densities in format: (Mics x Mics x Frequencies)
            Params,Implementation);
        %% ----------------------------------------------------------------
        %% END OF PROCESSING ----------------------------------------------
        %% ----------------------------------------------------------------

        % Cumulatively add processing time for operations which are
        % relevant to the processing (STFT and TDOA estimation):
        Processing_time_method(Implementation_idx) = Processing_time_method(Implementation_idx) + toc;

        TDOA_est_mtx_full{Implementation_idx}(:,:,1,frame_idx) = TDOA_est_mtx; % save TDOA estimates for later plotting
        frame_idx = frame_idx + 1;
        signal_indices = signal_indices + F;
    end
    
    disp(['Time elapsed for method [' Implementations{Implementation_idx} ']: ' num2str(Processing_time_method(Implementation_idx)) ' s'])
end
frames = frame_idx-1;

%% Plot and display results
s_traj_STFT = permute( interp1(1:size(s_traj, 2), s_traj.', linspace(1, size(s_traj, 2), frames), 'linear') , [2,3,4,1]);
TDOA_true = permute( vecnorm(s_traj_STFT-M_mtx) - vecnorm(s_traj_STFT- M_mtx(:,1)) ,[2,1,3,4])/c;

[VAD_idxs] = func_Compute_VAD_idx(x_t,M,Params.fs,K,F,Ana_window,K,false);
VAD_idxs = [VAD_idxs, false(1,frames-size(VAD_idxs,2))];

[TDOA_Err,TDOA_Acc] = deal( Inf(length(Implementations),1) );
TDOA_Errors = cell(1,length(Implementations));


figure(1); subplot(1,1,1);
plot3(M_mtx(1,:),M_mtx(2,:),M_mtx(3,:),'o'); axis([0,6,0,6,0,2.4]); grid on;
hold on;
plot3(s_traj_STFT(1,:),s_traj_STFT(2,:),s_traj_STFT(3,:),'r')
plot3(s_traj_STFT(1,1),s_traj_STFT(2,1),s_traj_STFT(3,1),'r.','MarkerSize',10)
legend('Microphone array','Source trajectory','Source starting position','AutoUpdate','off')
plot3([M_mtx(1,:); M_mtx(1,:)], [M_mtx(2,:); M_mtx(2,:)], [M_mtx(3,:); zeros(1,size(M_mtx,2))],'k:');
hold off;
axis([0, 6, 0, 6, 0, 2.4])
view(351, 20);
drawnow;

fig = figure(2);
% Set TDOA accuracy threshold differently for compact and distributed 
% microphone arrays:
if strcmp(Array_type,'Compact')
    Acc_Threshold = 0.1e-3;
else% strcmp(Array_type,'Distributed')
    Acc_Threshold = 1e-3;
end
max_TDOA_err = Acc_Threshold;
for Implementation_idx = 1:length(Implementations)
    for plot_idx = 1:M
        subplot(M,1,plot_idx)
        if plot_idx == 1
            plot(((1:size(x_t,1))-1)/Params.fs, y_t(:,1),'r')
            hold on
            plot(((1:size(x_t,1))-1)/Params.fs, x_t(:,1),'g')
            hold off
            max_val = max( max(abs(x_t(:,1)),[],'all') , max(abs(y_t(:,1)),[],'all'));
            ylim(max_val*[-1,1])
            xlabel('Time [s]')
            legend('Noisy & reverberant speech','Reverberant speech')
        else
            if Implementation_idx == 1
                plot(squeeze(TDOA_true(plot_idx,1,1,:)),'k')
                xlim([1,frames])
                ylim(Params.TDOA_Limits(plot_idx,1)*[-1.2,1.2])
            end
            hold on;
            plot(squeeze(TDOA_est_mtx_full{Implementation_idx}(plot_idx,1,1,:)),'--')
            ylabel(['Mic. pair (' num2str(plot_idx) ',1)'])
            if Implementation_idx == length(Implementations)
                if plot_idx == M
                    legend('TDOA','Estimated TDOA: Baseline','Estimated TDOA: Reference mic-based','Estimated TDOA: MST','Estimated TDOA: MST+','AutoUpdate','off')
                end
                func_Compute_VAD_idx(x_t,M,Params.fs,K,F,Ana_window,K,true,ylim);
            end
            if plot_idx == 2
                title('Estimated TDOAs')
            end

        end
        hold off;
    end
    xlabel('STFT frame')

    TDOA_Errors{Implementation_idx} = abs(TDOA_true(2:end,:,:,:) - TDOA_est_mtx_full{Implementation_idx}(2:end,1,1,:)); % omit first microphone pair (1,1), because TDOA is 0 anyway.
    TDOA_Err(Implementation_idx,1) = mean( TDOA_Errors{Implementation_idx}(:,1,1,VAD_idxs) , [1,4]);
    TDOA_Acc(Implementation_idx,1) = sum(TDOA_Errors{Implementation_idx}(2:end,:,:,VAD_idxs) < Acc_Threshold,'all')/numel(TDOA_Errors{Implementation_idx}(2:end,:,:,VAD_idxs)); % omit first microphone pair (1,1), because TDOA is 0 anyway.

    max_TDOA_err = max(max(TDOA_Errors{Implementation_idx}(:)),max_TDOA_err);
end
figure(3);
for Implementation_idx = 1:length(Implementations)
    subplot(2,2,Implementation_idx)
    histogram(TDOA_Errors{Implementation_idx}(:)*1e3,'BinWidth',1e3*Acc_Threshold/2)
    ylim([0,numel(TDOA_Errors{Implementation_idx})])
    xlim([0,max_TDOA_err*1e3])
    grid on;
    xlabel('TDOA Error [ms]')
    title(Implementations{Implementation_idx})
    sgtitle('TDOA Error Histograms')
end


disp(['Results for ' Scenario ' (' list{indx} '): '])
disp(array2table([round(Processing_time_method,2),round(TDOA_Err*1e3,3),TDOA_Acc*100], ...
    'VariableNames', {'Processing time [s]','TDOA Error [ms]',...
    ['TDOA Accuracy [%] (percentage of errors within ' num2str(Acc_Threshold*1e3) ' ms)']}, ...
    'RowNames', Implementations));