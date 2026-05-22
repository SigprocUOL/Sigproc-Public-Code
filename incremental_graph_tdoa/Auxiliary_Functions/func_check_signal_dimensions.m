function [y_t,M_L] = func_check_signal_dimensions(y_t)

M_L = size(y_t,2); % Number of microphones

if M_L > size(y_t,1)
    error('Please make sure the signal samples are defined along the first dimension and channels are along the second dimension')
end

end