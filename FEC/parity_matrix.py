import torch
import numpy as np

class ParityCheckMatrix:
    def __init__(self, H, device='cpu'):
        self.H = torch.tensor(H, dtype=torch.float32)
        # number of factor nodes, variable nodes, and edges in factor graph
        (self.M, self.N), self.E = H.shape, int(H.sum())
        # Construct auxiliary matrix for sparse row sum and sparse column sum
        I_row, I_col = np.where(H != 0)
        self.I_row = torch.tensor(I_row)
        self.I_col = torch.tensor(I_col)
        # Create indices for sparse tensor for row/column sum, sort in lexicographic order
        order = np.arange(self.E)
        r = torch.LongTensor(np.array([I_row, order]))
        c = torch.LongTensor(np.array([np.sort(I_col), order[np.argsort(I_col)]]))
        # Construct sparse tensor for row/column sum
        ones = torch.tensor(np.ones(order.shape)).float()
        self.S_row = torch.sparse_coo_tensor(r, ones, (self.M, self.E), dtype=torch.float32)
        self.S_col = torch.sparse_coo_tensor(c, ones, (self.N, self.E), dtype=torch.float32)
        # Dispatch data to GPU
        self.set_device(device=device)

    def __repr__(self):
        return 'Parity check matrix of size ({0}, {1})'.format(self.M, self.N)

    # Define functions for row/column gather, sparse row/column sum and sparse leave-one-out row/column sum
    def row_gather(self, msg):
        # length-M input, length-E output
        return msg.index_select(0, self.I_row)

    def col_gather(self, msg):
        # length-N input, length-E output
        return msg.index_select(0, self.I_col)

    def row_sum(self, msg):
        # length-E input, length-M output
        return self.S_row @ msg

    def col_sum(self, msg):
        # length-E input, length-N output
        return self.S_col @ msg

    def row_sum_loo(self, msg):
        # length-E input, length-E output
        return self.row_gather(self.row_sum(msg)) - msg

    def col_sum_loo(self, msg):
        # length-E input, length-E output
        return self.col_gather(self.col_sum(msg)) - msg

    def set_device(self, device):
        self.H = self.H.to(device=device)
        self.I_row = self.I_row.to(device=device)
        self.I_col = self.I_col.to(device=device)
        self.S_row = self.S_row.to(device=device)
        self.S_col = self.S_col.to(device=device)
