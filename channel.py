import torch
import torch.nn as nn

# Assumes no ICI/ISI (Proper CP handling is assumed)
# Low computation/memory cost
class EfficientChannel(nn.Module):
    def __init__(self, N, delta_f, device, batch_size, model_type='EVA'):
        super(EfficientChannel, self).__init__()
        self.N = N
        self.delta_f = delta_f
        self.device = device
        self.batch_size = batch_size
        self.model_type = model_type

    def gen_channel_params(self):
        delay_res = 1 / (self.N * self.delta_f)
        if self.model_type == 'EPA':
            tau = torch.tensor([0, 30, 70, 90, 110, 190, 410], device=self.device) * 1e-9
            PDP = torch.tensor([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8], device=self.device)
        elif self.model_type == 'EVA':
            tau = torch.tensor([0, 30, 150, 310, 370, 710, 1090, 1730, 2510], device=self.device) * 1e-9
            PDP = torch.tensor([0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9], device=self.device)
        elif self.model_type == 'ETU':
            tau = torch.tensor([0, 50, 120, 200, 230, 500, 1600, 2300, 5000], device=self.device) * 1e-9
            PDP = torch.tensor([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0], device=self.device)
        else:
            raise ValueError("Invalid channel model type.")
        
        taps = tau.shape[0]
        pow_norm = 10 ** (PDP / 10)
        pow_norm /= torch.sum(pow_norm)
        delay_taps = torch.round(tau / delay_res).long()
        
        chan_coef = (torch.sqrt(pow_norm) * 
                    (torch.randn(self.batch_size, taps, device=self.device, dtype=torch.float32) + 
                     1j * torch.randn(self.batch_size, taps, device=self.device, dtype=torch.float32)) 
                    ) / torch.sqrt(torch.tensor(2.0, device=self.device))
        
        return chan_coef, delay_taps, taps

    def gen_frequency_response(self):
        chan_coef, delay_taps, _ = self.gen_channel_params()
        k = torch.arange(self.N, device=self.device).float()
        
        phase_shifts = -2j * torch.pi * k.unsqueeze(0) * delay_taps.unsqueeze(1) / self.N
        phase_shifts = torch.exp(phase_shifts)  
        
        H_eff = torch.einsum('bt,tn->bn', chan_coef, phase_shifts) 
        return H_eff

    def gen_noise(self, sigma_2):
        return (torch.sqrt(torch.tensor(sigma_2/2, device=self.device)) * 
                (torch.randn(self.batch_size, self.N, device=self.device) + 
                 1j * torch.randn(self.batch_size, self.N, device=self.device)))

    def forward(self, x, sigma_2, fading=True, h=None, equalization='MMSE'):
        if fading:
            if sigma_2 != 0:
                H_eff = h if h is not None else self.gen_frequency_response()
                noise = self.gen_noise(sigma_2)
                y = H_eff * x + noise
                if equalization == 'MMSE':
                    x_hat = (torch.conj(H_eff) * y) / (torch.abs(H_eff)**2 + sigma_2)
                elif equalization == 'ZF':
                    x_hat = y / H_eff
                elif equalization == 'None':
                    x_hat = y
                else:
                    raise ValueError("Invalid equalization type")
                return x_hat
            else: # For noiseless feedback setting (no distortion)
                return x
        else:
            if sigma_2 != 0:
                noise = self.gen_noise(sigma_2)
                y = x + noise
                return y
            else: # For noiseless feedback setting 
                return x

