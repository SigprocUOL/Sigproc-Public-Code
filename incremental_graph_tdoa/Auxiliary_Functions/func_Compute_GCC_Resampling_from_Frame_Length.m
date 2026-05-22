function [GCC_Resampling,NIFFT] = func_Compute_GCC_Resampling_from_Frame_Length(K,fs)

if fs > 16000
    % Determine GCC-PHAT Resampling factor
    if K <= 256                % K <= 256
        GCC_Resampling = 20;
    elseif K > 256 && K <= 512 % K == 512 
        GCC_Resampling = 10;
    elseif K > 512 && K < 2048 % k == 1024
        GCC_Resampling = 4;
    else                       % K >= 2048
        GCC_Resampling = 3;
    end
else % if fs == 16000
    if K <= 256 % Determine GCC-PHAT Resampling factor
        GCC_Resampling = 10;
    elseif K > 256 && K <= 512
        GCC_Resampling = 8;
    elseif K > 512 && K < 2048
        GCC_Resampling = 6;
    else % K >= 2048
        GCC_Resampling = 4;
    end
end
NIFFT = GCC_Resampling*K;
end