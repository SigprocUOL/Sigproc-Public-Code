function [lambda,lambda_str] = func_Compute_lambda(lambda_s,R,fs)

if ~exist('fs','var')
    fs = 16e3;
end

lambda = exp(-R./(lambda_s*fs));
lambda_str = num2str(lambda);
lambda_str(lambda_str=='.')='_';
disp(['Smoothing time: ' num2str( round( -R/fs /(log(lambda)) ,2) )  ' s. lambda: ' num2str(round(lambda,3)) '.'])

end