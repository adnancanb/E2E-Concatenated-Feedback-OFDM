import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, pdb

# ========== LIGHTCODE ==========

class LIGHTCODE(nn.Module):
	def __init__(self, mod, input_size, m, d_model, dropout, multclass = False, NS_model=0):
		super(LIGHTCODE, self).__init__()
		self.mod = mod
		self.multclass = multclass
		self.m = m

		self.fe1 = FE_LIGHTCODE(mod, NS_model, input_size, d_model)
		self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
   
		# project to the output space
		if mod == "trx":
			self.out = nn.Linear(d_model, 2)
		else:
			if multclass:
				self.out1 = nn.Linear(d_model, d_model)
				self.out2 = nn.Linear(d_model, 2**m)
			else:
				self.out1 = nn.Linear(d_model, d_model)
				self.out2 = nn.Linear(d_model, 2*m)
		self.dropout = nn.Dropout(dropout)

	def forward(self, src):
		enc_out = self.fe1(src)
		enc_out = self.norm1(enc_out)
		
		if self.mod == "rec":
			enc_out = self.out1(enc_out)
			enc_out = self.out2(enc_out)
		else:
			enc_out = self.out(enc_out)
   
		if self.mod == "rec":
			if self.multclass == False:
				batch = enc_out.size(0)
				ell = enc_out.size(1)
				enc_out = enc_out.contiguous().view(batch, ell*self.m,2)
				output = F.softmax(enc_out, dim=-1)
			else:
				output = F.softmax(enc_out, dim=-1)
		else:
			# encoders
			output = enc_out
		return output

class FE_LIGHTCODE(nn.Module):
    def __init__(self, mod, NS_model, input_size, d_model):
        super(FE_LIGHTCODE, self).__init__()
        self.mod = mod
        self.NS_model = NS_model
        self.reserve = 3 + 8
        if self.NS_model == 0:
            self.FC1 = nn.Linear(input_size, d_model, bias=True)
        elif self.NS_model == 1:
            self.FC1 = nn.Linear(input_size, d_model*3, bias=True)
            self.activation1 = F.relu
            self.FC2 = nn.Linear(d_model*3, d_model, bias=True)
        elif self.NS_model == 2:
            self.FC1 = nn.Linear(input_size, d_model*3, bias=True)
            self.activation1 = nn.GELU()
            self.FC2 = nn.Linear(d_model*3, d_model*3, bias=True)
            self.activation2 = nn.GELU()
            self.FC3 = nn.Linear(d_model*3, d_model, bias=True)
        elif self.NS_model == 3:
            self.FC1 = nn.Linear(input_size, d_model*2, bias=True)
            self.activation1 = nn.ReLU()
            self.FC2 = nn.Linear(d_model*2, d_model*2, bias=True)
            self.activation2 = nn.ReLU()
            self.FC3 = nn.Linear(d_model*2, d_model*2, bias=True)
            self.FC4 = nn.Linear(d_model*4, d_model, bias=True)

    def forward(self, src):
        if self.NS_model == 0:
            x = self.FC1(src)
        elif self.NS_model == 1:
            x = self.FC1(src)
            x = self.FC2(self.activation1(x))
        elif self.NS_model == 2:
            x = self.FC1(src)
            x = self.FC2(self.activation1(x))
            x = self.FC3(self.activation2(x))
        elif self.NS_model == 3:
            x1 = self.FC1(src)
            src1 = x1 * (-1)
            x1 = self.FC2(self.activation1(x1))
            x1 = self.FC3(self.activation2(x1))
            x = self.FC4(torch.cat([x1, src1], dim = 2))
        return x

