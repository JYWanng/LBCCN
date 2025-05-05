import torch
import torch.nn as nn
from torch.nn import Module
from torch_stoi import NegSTOILoss


EPS = 1e-6

class Stft(nn.Module):
    def __init__(self, n_dft=1024, hop_size=512, win_length=None,
                 onesided=True, is_complex=True):

        super().__init__()

        self.n_dft = n_dft
        self.hop_size = hop_size
        self.win_length = n_dft if win_length is None else win_length
        self.onesided = onesided
        self.is_complex = is_complex

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels, time_steps)"

        window = torch.hann_window(self.win_length, device=x.device)

        y = torch.stft(x, self.n_dft, hop_length=self.hop_size,
                       win_length=self.win_length, onesided=self.onesided,
                       return_complex=True, window=window, normalized=True)

        if not self.is_complex:
            y = torch.view_as_real(y)
            y = y.movedim(-1, 1) # move complex dim to front

        return y


class IStft(Stft):

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels=freq_bins, time_steps)"
        window = torch.hann_window(self.win_length, device=x.device)

        y = torch.istft(x, self.n_dft, hop_length=self.hop_size,
                        win_length=self.win_length, onesided=self.onesided,
                        window=window,normalized=True)

        return y

class BinauralLoss(Module):
    def __init__(self, sr=16000, ild_weight=0.5, ipd_weight=5, stoi_weight=5, snr_loss_weight=0.5):

        super().__init__()
        self.stft = Stft(n_dft = 256, hop_size = 128)
        self.stoi_loss = NegSTOILoss(sample_rate=sr)
       
        self.ild_weight = ild_weight
        self.ipd_weight = ipd_weight
        self.stoi_weight = stoi_weight
        self.snr_loss_weight = snr_loss_weight

        
    def forward(self, voice, noise, vlc, vrc, nlc, nrc, voice_target, noise_target):
        voice_target_stft_l = self.stft(voice_target[:, 0])
        voice_target_stft_r = self.stft(voice_target[:, 1])
        noise_target_stft_l = self.stft(noise_target[:, 0])
        noise_target_stft_r = self.stft(noise_target[:, 1])
        voice_target_stft_l = voice_target_stft_l[:,:40,:]
        voice_target_stft_r = voice_target_stft_r[:,:40,:]
        noise_target_stft_l = noise_target_stft_l[:,:40,:]
        noise_target_stft_r = noise_target_stft_r[:,:40,:]

        loss = 0
        if self.snr_loss_weight > 0:
            
            snr_l = snr_loss(voice[:, 0], voice_target[:, 0])
            snr_r = snr_loss(voice[:, 1], voice_target[:, 1])
          
            voice_snr_loss_lr = - (snr_l + snr_r)/2
           
            voice_bin_snr_loss = self.snr_loss_weight*voice_snr_loss_lr
            
            print('\n Voice SNR Loss = ', voice_bin_snr_loss)
            loss += voice_bin_snr_loss

            snr_l = snr_loss(noise[:, 0], noise_target[:, 0])
            snr_r = snr_loss(noise[:, 1], noise_target[:, 1])
          
            noise_snr_loss_lr = - (snr_l + snr_r)/2
           
            noise_bin_snr_loss = self.snr_loss_weight*noise_snr_loss_lr
            
            print('\n Noise SNR Loss = ', noise_bin_snr_loss)
            loss += noise_bin_snr_loss
        
        if self.stoi_weight > 0:
            stoi_l = self.stoi_loss(voice[:, 0], voice_target[:, 0])
            stoi_r = self.stoi_loss(voice[:, 1], voice_target[:, 1])

            voice_stoi_loss = (stoi_l+stoi_r)/2
            voice_stoi_loss_mean = voice_stoi_loss.mean()
            voice_bin_stoi_loss = self.stoi_weight*voice_stoi_loss_mean
            print('\n Voice STOI Loss = ', voice_bin_stoi_loss)
            loss += voice_bin_stoi_loss

            stoi_l = self.stoi_loss(noise[:, 0], noise_target[:, 0])
            stoi_r = self.stoi_loss(noise[:, 1], noise_target[:, 1])

            noise_stoi_loss = (stoi_l+stoi_r)/2
            noise_stoi_loss_mean = noise_stoi_loss.mean()
            noise_bin_stoi_loss = self.stoi_weight*noise_stoi_loss_mean
            print('\n Noise STOI Loss = ', noise_bin_stoi_loss)
            loss += noise_bin_stoi_loss

        if self.ild_weight > 0:
            voice_ild_loss = ild_loss_db(voice_target_stft_l.abs(), voice_target_stft_r.abs(),
                                   vlc.abs(), vrc.abs()).mean()
            
            voice_bin_ild_loss = self.ild_weight*voice_ild_loss
            print('\n Voice ILD Loss = ', voice_bin_ild_loss)
            loss += voice_bin_ild_loss

            noise_ild_loss = ild_loss_db(noise_target_stft_l.abs(), noise_target_stft_r.abs(),
                                   nlc.abs(), nrc.abs()).mean()
            
            noise_bin_ild_loss = self.ild_weight*noise_ild_loss
            print('\n Noise ILD Loss = ', noise_bin_ild_loss)
            loss += noise_bin_ild_loss

        if self.ipd_weight > 0:
            voice_ipd_loss = ipd_loss_rads(voice_target_stft_l, voice_target_stft_r,
                                     vlc, vrc).mean()
            voice_bin_ipd_loss = self.ipd_weight*voice_ipd_loss
            
            print('\n Voice IPD Loss = ', voice_bin_ipd_loss)
            loss += voice_bin_ipd_loss
        
            noise_ipd_loss = ipd_loss_rads(noise_target_stft_l, noise_target_stft_r,
                                     nlc, nrc).mean()
            noise_bin_ipd_loss = self.ipd_weight*noise_ipd_loss
            
            print('\n Noise IPD Loss = ', noise_bin_ipd_loss)
            loss += noise_bin_ipd_loss
        
        target_stft_l = self.stft(voice_target[:, 0])
        target_stft_r = self.stft(voice_target[:, 1])
        stft_l = self.stft(voice[:, 0])
        stft_r = self.stft(voice[:, 1])
        ild_loss = ild_loss_db(target_stft_l.abs(), target_stft_r.abs(), stft_l.abs(), stft_r.abs())
        ipd_loss = ipd_loss_rads(target_stft_l, target_stft_r, stft_l, stft_r)

        return loss, (voice_snr_loss_lr, noise_snr_loss_lr, voice_stoi_loss_mean, noise_stoi_loss_mean, voice_ild_loss, noise_ild_loss, voice_ipd_loss, noise_ipd_loss, ild_loss, ipd_loss)
        
 


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def snr_loss(s1, s_target, eps=EPS, reduce_mean=True):
    
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    snr_norm = snr  # /max(snr)
    if reduce_mean:
        snr_norm = torch.mean(snr_norm)

    return snr_norm


def ild_db(s1, s2, eps=EPS):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    l1 = 20*torch.log10(s1 + eps)
    l2 = 20*torch.log10(s2 + eps)
    ild_value = (l1 - l2)

    return ild_value


def ild_loss_db(target_stft_l, target_stft_r,
                output_stft_l, output_stft_r, avg_mode=None):
    # amptodB = T.AmplitudeToDB(stype='amplitude')

    target_ild = ild_db(target_stft_l.abs(), target_stft_r.abs())
    output_ild = ild_db(output_stft_l.abs(), output_stft_r.abs())
    mask = speechMask(target_stft_l,target_stft_r,threshold=20)
    
    ild_loss = (target_ild - output_ild).abs()
    # breakpoint()
    masked_ild_loss = ((ild_loss * mask).sum(dim=2)).sum(dim=1)/(mask.sum(dim=2)).sum(dim=1)
   
    return masked_ild_loss

def msc_loss(target_stft_l, target_stft_r,
                output_stft_l, output_stft_r):
    
    

    # Calculate the Auto-Power Spectral Density (APSD) for left and right signals
    # Calculate the Auto-Power Spectral Density (APSD) for left and right signals
    cpsd = target_stft_l * target_stft_r.conj()
    cpsd_op = output_stft_l * output_stft_r.conj()
    
    # Calculate the Aucpsd = target_stft_l * target_stft_r.conj()to-Power Spectral Density (APSD) for left and right signals
    left_apsd = target_stft_l * target_stft_l.conj()
    right_apsd = target_stft_r * target_stft_r.conj()
    
    left_apsd_op = output_stft_l * output_stft_l.conj()
    right_apsd_op = output_stft_r * output_stft_r.conj()
    
    # Calculate the Magnitude Squared Coherence (MSC)
    msc_target = torch.abs(cpsd)**2 / ((left_apsd.abs() * right_apsd.abs())+1e-8)
    msc_output = torch.abs(cpsd_op)**2 / ((left_apsd_op.abs() * right_apsd_op.abs())+1e-8)
    
    mask = speechMask(target_stft_l,target_stft_r,threshold=20)
    
    msc_error = (msc_target - msc_output).abs()
    


    # Plot the MSC values as a function of frequency
    
    
    # breakpoint()
    # masked_msc_error = ((msc_error * mask).sum(dim=2)).sum(dim=1)/(mask.sum(dim=2)).sum(dim=1)
    
    return msc_error.mean()
    

def ipd_rad(s1, s2, eps=EPS, avg_mode=None):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    ipd_value = ((s1 + eps)/(s2 + eps)).angle()

    return ipd_value


def ipd_loss_rads(target_stft_l, target_stft_r,
                  output_stft_l, output_stft_r, avg_mode=None):
    # amptodB = T.AmplitudeToDB(stype='amplitude')
    target_ipd = ipd_rad(target_stft_l, target_stft_r, avg_mode=avg_mode)
    output_ipd = ipd_rad(output_stft_l, output_stft_r, avg_mode=avg_mode)

    ipd_loss = ((target_ipd - output_ipd).abs())

    mask = speechMask(target_stft_l,target_stft_r, threshold=20)
    
    masked_ipd_loss = ((ipd_loss * mask).sum(dim=2)).sum(dim=1)/(mask.sum(dim=2)).sum(dim=1)
    return masked_ipd_loss


def speechMask(stft_l,stft_r, threshold=15):
    # breakpoint()
    _,_,time_bins = stft_l.shape
    thresh_l,_ = (((stft_l.abs())**2)).max(dim=2) 
    thresh_l_db = 10*torch.log10(thresh_l) - threshold
    thresh_l_db=thresh_l_db.unsqueeze(2).repeat(1,1,time_bins)
    
    thresh_r,_ = (((stft_r.abs())**2)).max(dim=2) 
    thresh_r_db = 10*torch.log10(thresh_r) - threshold
    thresh_r_db=thresh_r_db.unsqueeze(2).repeat(1,1,time_bins)
    
    
    bin_mask_l = BinaryMask(threshold=thresh_l_db)
    bin_mask_r = BinaryMask(threshold=thresh_r_db)
    
    mask_l = bin_mask_l(20*torch.log10((stft_l.abs())))
    mask_r = bin_mask_r(20*torch.log10((stft_r.abs())))
    mask = torch.bitwise_and(mask_l.int(), mask_r.int())
    
    return mask


class BinaryMask(Module):
    def __init__(self, threshold=0.5):
        super(BinaryMask, self).__init__()
        self.threshold = threshold

    def forward(self, magnitude):
        # Compute the magnitude of the complex spectrogram
        # magnitude = torch.sqrt(spectrogram[:,:,0]**2 + spectrogram[:,:,1]**2)

        # Create a binary mask by thresholding the magnitude
        mask = (magnitude > self.threshold).float()
        # breakpoint()
        return mask

