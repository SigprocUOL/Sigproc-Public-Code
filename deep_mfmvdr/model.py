"""
contains deep MFMVDR model implementation
"""
import torch
import torch.nn.functional as F

from . import building_blocks as bb
from . import utils

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


class DeepMFMVDR(torch.nn.Module):
    """
    deep MFMVDR filter
    """
    def __init__(self, config):
        super().__init__()

        config_defaults = {
            'filter_length': 5,
            'frame_length': 128,
            'shift_length': 32,
            'reg': 1e-3,
            'minimum_gain': -17.,
            'minimum_gain_k': 10.,
            'return_wave': True,
            'hidden_dim': 128,
            'layer': 4,
            'stack': 2,
            'kernel': 3,
        }
        config_defaults.update(config)
        self.config = utils.Bunch(config_defaults)

        self.config.frequency_bins = int(self.config.frame_length / 2 + 1)
        self.config.minimum_gain = utils.db2mag(self.config.minimum_gain)
        self.config.tcn_params = {
            "hidden_dim": self.config.hidden_dim,
            "layer": self.config.layer,
            "stack": self.config.stack,
            "kernel": self.config.kernel
        }
        self.stft = utils.STFTTorch(self.config.frame_length,
                                    self.config.frame_length -
                                    self.config.shift_length,
                                    window=torch.hann_window)
        output_size = self.config.filter_length**2 * self.config.frequency_bins

        self.Phin_estimator = bb.TCNEstimator(
            input_dim=2 * self.config.frequency_bins,
            output_dim=output_size,
            BN_dim=self.config.tcn_params["hidden_dim"],
            hidden_dim=4 * self.config.tcn_params["hidden_dim"],
            layer=self.config.tcn_params['layer'],
            stack=self.config.tcn_params['stack'],
            kernel=self.config.tcn_params['kernel'])
        self.Phiy_estimator = bb.TCNEstimator(
            input_dim=2 * self.config.frequency_bins,
            output_dim=output_size,
            BN_dim=self.config.tcn_params["hidden_dim"],
            hidden_dim=4 * self.config.tcn_params["hidden_dim"],
            layer=self.config.tcn_params['layer'],
            stack=self.config.tcn_params['stack'],
            kernel=self.config.tcn_params['kernel'])

        self.xi_estimator = bb.TCNEstimator(
            input_dim=self.config.frequency_bins,
            output_dim=self.config.frequency_bins,
            BN_dim=self.config.tcn_params["hidden_dim"],
            hidden_dim=4 * self.config.tcn_params["hidden_dim"],
            layer=self.config.tcn_params['layer'],
            stack=self.config.tcn_params['stack'],
            kernel=self.config.tcn_params['kernel'])

    def forward(self, batch):
        """
        batch: tensor of shape (batch_size x num_samples)
        """
        noisy = batch
        batch_size = noisy.shape[0]
        # to STFT-domain (batch_size x frequency_bins x time_frames x 2)
        noisy = torch.stack([self.stft.get_stft(x) for x in noisy], dim=0)
        # concatenate real and imaginary component over frequency dimension
        noisy_stacked = torch.cat([noisy[..., 0], noisy[..., 1]], dim=1)
        # get multi-frame vectors
        noisy_adj = self.get_adjacent(noisy)

        noisy_mag_log = torch.log10(utils.complex_tensor_abs(noisy))
        a_priori_snr = F.softplus(self.xi_estimator(noisy_mag_log))

        correlation_noise = self.Phin_estimator(noisy_stacked)
        correlation_noise = correlation_noise.view(
            (batch_size, self.config.frequency_bins,
             int(self.config.filter_length**2), -1)).permute(0, 1, 3,
                                                             2).contiguous()
        correlation_noisy = self.Phiy_estimator(noisy_stacked)
        correlation_noisy = correlation_noisy.view(
            (batch_size, self.config.frequency_bins,
             int(self.config.filter_length**2), -1)).permute(0, 1, 3,
                                                             2).contiguous()

        # assemble Hermitian matrices
        correlation_noise = utils.vector_to_Hermitian(correlation_noise)
        correlation_noisy = utils.vector_to_Hermitian(correlation_noisy)

        # force matrices to be psd
        correlation_noise = utils.complex_tensor_matrix_matrix_product(
            correlation_noise,
            utils.complex_tensor_hermitian(correlation_noise))
        correlation_noisy = utils.complex_tensor_matrix_matrix_product(
            correlation_noisy,
            utils.complex_tensor_hermitian(correlation_noisy))

        # Tikhonov regularization (if desired)
        if self.config.reg != 0.0:
            correlation_noise = utils.tik_reg(correlation_noise,
                                              reg=self.config.reg)
            correlation_noisy = utils.tik_reg(correlation_noisy,
                                              reg=self.config.reg)

        # compute IFC vectors
        gamman = utils.complex_tensor_division(
            correlation_noise[..., -1, :],
            correlation_noise[..., -1, -1, :][..., None, :] + EPS)
        gammay = utils.complex_tensor_division(
            correlation_noisy[..., -1, :],
            correlation_noisy[..., -1, -1, :][..., None, :] + EPS)
        gammax = ((1 + a_priori_snr) /
                  (a_priori_snr + EPS))[..., None, None] * gammay - (
                      1 / (a_priori_snr + EPS))[..., None, None] * gamman

        # compute MFMVDR filters
        filters = utils.get_mvdr(gammax, correlation_noise)

        # filter to obtain speech estimate
        speech_estimate, _ = utils.filter_minimum_gain_like(
            self.config.minimum_gain,
            filters,
            noisy_adj,
            k=self.config.minimum_gain_k)

        output = {"speech_estimate": speech_estimate}
        if self.config.return_wave:
            speech_estimate_wave = torch.stack(
                [self.stft.get_istft(x) for x in speech_estimate])
            # normalize wave
            speech_estimate_wave = 0.03 * speech_estimate_wave / (speech_estimate_wave**2).mean(-1).sqrt()
            output["speech_estimate_wave"] = speech_estimate_wave

        return output

    def get_adjacent(self, stft: torch.Tensor) -> torch.Tensor:
        # zero-pad and unfold stft, i.e.,
        # add zeros to the beginning so that, using the multi-frame signal model,
        # there will be as many output frames as input frames
        return F.pad(stft, pad=[0, 0, self.config.filter_length - 1,
                                0]).unfold(dimension=-2,
                                           size=self.config.filter_length,
                                           step=1).transpose(-2,
                                                             -1).contiguous()
