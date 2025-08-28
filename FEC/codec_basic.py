import torch
import numpy as np
from galois import GF2
from .parity_matrix import ParityCheckMatrix

class LinearCodec:
    def __init__(self, info_len, coded_len, zc, k_ldpc, n_ldpc, H=None, G=None, device='cpu', llr_clip=5, max_decode_iter=100):
        if H is None and G is None:
            raise AttributeError('parity matrix and generator matrix cannot both be none')
        elif H is None:
            self.G = torch.from_numpy(G).to(device=device, dtype=torch.float32)
            H = np.array(GF2(G.astype(int)).null_space())
            self.H = ParityCheckMatrix(H, device=device)
        elif G is None:
            self.H = ParityCheckMatrix(H, device=device)
            G = GF2(H.astype(int)).null_space()
            self.G = torch.from_numpy(G).to(device=device, dtype=torch.float32)
        else:
            if np.sum(np.array(H.dot(G.T))) > 0:
                raise ValueError('the dot product of generator and parity matrix is not zero')
            self.G = torch.from_numpy(G).to(device=device, dtype=torch.float32)
            self.H = ParityCheckMatrix(H, device=device)

        self.info_len = info_len
        self.coded_len = coded_len
        self.llr_clip = llr_clip
        self.device = device
        self.max_decode_iter = max_decode_iter
        self.zc = zc
        self.k_ldpc = k_ldpc
        self.n_ldpc = n_ldpc

    def set_device(self, device):
        self.H.set_device(device=device)
        self.G = self.G.to(device=device)
        self.device = device

    def encode(self, x):
        x = x.to(dtype=torch.float32)
        batch_size = x.shape[0]
        k = x.shape[1]
        if self.k_ldpc < k:
            raise ValueError("k_ldpc must be >= k (number of information bits)")
        filler = torch.zeros((batch_size, self.k_ldpc - k), dtype=torch.float32, device=x.device)
        x_filled = torch.cat([x, filler], dim=1)
        full_codeword = torch.matmul(x_filled, self.G) % 2
        full_codeword = full_codeword.to(torch.int)
        if self.n_ldpc < self.k_ldpc:
            raise ValueError("n_ldpc must be >= k_ldpc")
        c_no_filler1 = full_codeword[:, :k]
        c_no_filler2 = full_codeword[:, self.k_ldpc:]
        c_no_filler = torch.cat([c_no_filler1, c_no_filler2], dim=1)
        if c_no_filler.shape[1] < 2 * self.zc + self.coded_len:
            raise ValueError("Not enough bits available after shortening for the target length")
        final_codeword = c_no_filler[:, 2 * self.zc : 2 * self.zc + self.coded_len]
        return final_codeword
    
    def undo_puncturing_and_shortening(
        self,
        received_llr,
        _output_dtype=torch.float32
    ):
        batch_size = received_llr.shape[0]
        front_zeros = torch.zeros(
            (batch_size, 2 * self.zc),
            dtype=_output_dtype,
            device=received_llr.device
        )
        llr_5g = torch.cat([front_zeros, received_llr], dim=1)
        k_filler = self.k_ldpc - self.info_len
        nb_punc_bits = (self.n_ldpc - k_filler) - self.coded_len - 2 * self.zc
        back_zeros = torch.zeros(
            (batch_size, nb_punc_bits),
            dtype=_output_dtype,
            device=received_llr.device
        )
        llr_5g = torch.cat([llr_5g, back_zeros], dim=1)
        x1 = llr_5g[:, :self.info_len]
        nb_par_bits = self.n_ldpc - k_filler - self.info_len
        x2 = llr_5g[:, self.info_len : self.info_len + nb_par_bits]
        z = self.llr_clip * torch.ones(
            (batch_size, k_filler),
            dtype=_output_dtype,
            device=received_llr.device
        )
        llr_5g = torch.cat([x1, z, x2], dim=1)
        return llr_5g
    
    def remove_filler_and_shorten(self, x_hat):
        batch_size = x_hat.shape[0]
        u_hat = x_hat[:batch_size, :self.info_len]
        return u_hat

    def phi(self, u: torch.Tensor) -> torch.Tensor:
        u_clamped = u.clamp(min=8.5e-8, max=16.635532) #float32 specific clamping
        return torch.log(((torch.exp(u_clamped) + 1.0) / (torch.exp(u_clamped) - 1.0)))

    def decode(self, received_llr, gamma=1.0):
        ch_llr = self.undo_puncturing_and_shortening(received_llr)
        
        ch_llr = ch_llr.transpose(0, 1).clamp(-self.llr_clip, self.llr_clip)
        shape = (self.H.E, ch_llr.shape[1])
        msg_C2V = torch.zeros(*shape, device=self.device)
        msg_C2V_old = torch.zeros(*shape, device=self.device)
        msg_V2C_old = torch.zeros(*shape, device=self.device)
        for it in range(self.max_decode_iter):
            int_llr = self.H.col_gather(ch_llr)
            ext_llr = self.H.col_sum_loo(msg_C2V)
            msg_V2C_new = int_llr + ext_llr
            msg_V2C = gamma * msg_V2C_new + (1.0 - gamma) * msg_V2C_old
            sgn = (-1.0) ** self.H.row_sum_loo((msg_V2C < 0).float())
            sgn = sgn.detach()
            abs_msg = msg_V2C.abs()
            phi_msg = self.phi(abs_msg)
            sum_phi_all = self.H.row_sum_loo(phi_msg)
            msg_C2V_new = sgn * self.phi(sum_phi_all)
            msg_C2V = gamma * msg_C2V_new + (1.0 - gamma) * msg_C2V_old
            msg_V2C_old = msg_V2C.detach().clone()
            msg_C2V_old = msg_C2V.detach().clone()
        output_llr = ch_llr + self.H.col_sum(msg_C2V)
        output_llr = output_llr.transpose(0, 1)

        # For E2E training stability, clip output LLR values
        output_llr = output_llr.clamp(-self.llr_clip, self.llr_clip)

        final_llr = self.remove_filler_and_shorten(output_llr)
        return final_llr
