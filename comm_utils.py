import torch

class QAMModulator:
    def __init__(self, M, device, mod_temp=0.225):
        self.device = device
        self.M = M
        self.mod_temp = mod_temp
        self.L = int(torch.sqrt(torch.tensor(float(M))).item())
        self.n_bits = int(torch.log2(torch.tensor(float(self.L))).item())
        self.norm_factor = torch.sqrt(
            torch.tensor((2 / 3) * (M - 1), dtype=torch.float, device=self.device)
        )
        self.candidate_levels = torch.linspace(
            -self.L + 1, self.L - 1, steps=self.L, device=self.device
        )
        self.candidate_bits = self._generate_gray_table(self.L, self.n_bits)

    def _generate_gray_table(self, L, n_bits):
        table = []
        for i in range(L):
            gray = i ^ (i >> 1)
            bits = [(gray >> (n_bits - 1 - j)) & 1 for j in range(n_bits)]
            table.append(bits)
        return torch.tensor(table, dtype=torch.float, device=self.device)

    def modulate(self, bits):
        bits = bits.to(self.device)
        total_bits = bits.shape[-1]
        block_size = 2 * self.n_bits
        if total_bits % block_size != 0:
            raise ValueError(
                f"Input bit length must be a multiple of {block_size} for M = {self.M}."
            )
        num_blocks = total_bits // block_size
        bits = bits.view(-1, num_blocks, block_size)
        I_bits = bits[..., :self.n_bits]
        Q_bits = bits[..., self.n_bits:]

        # Use softmax with temperature for differentiability (and to avoid vanishing gradients!)
        diff_I = torch.sum(
            (I_bits.unsqueeze(-2) - self.candidate_bits.unsqueeze(0).unsqueeze(0)) ** 2,
            dim=-1,
        )
        weights_I = torch.softmax(-diff_I / self.mod_temp, dim=-1)
        I = torch.sum(weights_I * self.candidate_levels, dim=-1)

        diff_Q = torch.sum(
            (Q_bits.unsqueeze(-2) - self.candidate_bits.unsqueeze(0).unsqueeze(0)) ** 2,
            dim=-1,
        )
        weights_Q = torch.softmax(-diff_Q / self.mod_temp, dim=-1)
        Q = torch.sum(weights_Q * self.candidate_levels, dim=-1)

        modulated = torch.complex(I, Q) / self.norm_factor
        return modulated

