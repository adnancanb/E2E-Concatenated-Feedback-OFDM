import os
import random
import numpy as np
import torch
import json
import math
import torch.nn as nn
from utils import *
from nn_layers import *
from lightcode_model import *
from parameters import *
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import time
import torch.profiler

from channel import EfficientChannel
from comm_utils import QAMModulator
from FEC.nr_ldpc import get_5g_ldpc_parity_matrix
from FEC.codec_basic import LinearCodec
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

torch.cuda.empty_cache()

class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()

        self.args = args

        if args.nn_model == 'GBAF':
            self.pe = PositionalEncoder_fixed(lenWord=args.d_model_trx)
            self.Tmodel = BERT("trx", 2*(args.m + 2 * (args.T - 1)), args.m, args.d_model_trx,
                                args.N_trx, args.heads_trx, args.dropout, args.custom_attn,
                                args.multclass, args.NS_model)
            
            self.Rmodel = BERT("rec", 2*args.T, int(args.m * math.log2(args.ModOrder)), args.d_model_trx,
                                args.N_trx + 1, args.heads_trx, args.dropout, args.custom_attn,
                                args.multclass, args.NS_model)
        elif args.nn_model == 'LIGHTCODE':
            self.Tmodel = LIGHTCODE("trx", 2*(args.m + 2 * (args.T - 1)), args.m, args.d_model_trx_ligthcode, args.dropout,args.multclass, args.enc_NS_model_lightcode)
            self.Rmodel = LIGHTCODE("rec", 2*args.T, int(args.m * math.log2(args.ModOrder)), args.d_model_rec_lightcode, args.dropout, args.multclass, args.dec_NS_model_lightcode)
        else:
            raise ValueError("Invalid Neural Model.")  
        
        if args.intlv:
            self.interleaver = self.random_interleaving_matrix()

        self.chan = EfficientChannel(args.ell, args.delta_f, args.device, args.batchSize, args.channelMod)

        self.modulator = QAMModulator(args.ModOrder, args.device)

        if self.args.reloc == 1:
            self.total_power_reloc = Power_reallocate(args)

    def random_interleaving_matrix(self):
        if self.args.coded:
            intlv_depth = self.args.n
        else:
            intlv_depth = self.args.k
        perm = torch.randperm(intlv_depth, device=self.args.device)
        eye_matrix = torch.eye(intlv_depth, device=self.args.device, dtype=torch.float32)
        interleaver = eye_matrix[perm]
        return interleaver.unsqueeze(0)
    
    def power_constraint(self, inputs, isTraining, eachbatch, idx=0):
        if isTraining == 1:
            this_mean = torch.mean(inputs, 0)
            if torch.is_complex(inputs):
                this_std = torch.sqrt(torch.mean(torch.abs(inputs - this_mean)**2, 0))
            else:
                this_std = torch.std(inputs, 0)
        elif isTraining == 0:
            stat_file = 'statistics' + str(args.coded) + str(args.ModOrder) + str(args.intlv) + str(args.llr_data) + str(args.equalizer) + str(args.nn_model) + str(args.E2E) + str(args.snr2) + str(args.testintlv)
            if eachbatch == 0:
                this_mean = torch.mean(inputs, 0)
                if torch.is_complex(inputs):
                    this_std = torch.sqrt(torch.mean(torch.abs(inputs - this_mean)**2, 0))
                else:
                    this_std = torch.std(inputs, 0)

                if not os.path.exists(stat_file):
                    os.mkdir(stat_file)
                torch.save(this_mean, stat_file + '/this_mean' + str(idx))
                torch.save(this_std, stat_file + '/this_std' + str(idx))
                if idx == 0:
                    print('this_mean and this_std saved ...', flush=True)
            else:
                this_mean = torch.load(stat_file + '/this_mean' + str(idx))
                this_std = torch.load(stat_file + '/this_std' + str(idx))
        outputs = (inputs - this_mean) / (this_std + 1e-8)
        return outputs

    def forward(self, eachbatch, bVec, fwd_noise_var, fb_noise_var, isTraining=1, codec=None):
        
        # Interleaving
        if self.args.intlv:
            bVec = (torch.matmul(self.interleaver, bVec.unsqueeze(-1))).squeeze(-1)
        
        # Modulation Part
        if self.args.ModOrder == 2:
            bVec_md = 1 - 2 * bVec
            bVec_md = torch.complex(bVec_md, torch.zeros_like(bVec_md))
        else:
            bVec_md = self.modulator.modulate(bVec)

        bVec_md = bVec_md.view(self.args.batchSize, self.args.ell, self.args.m)
        
        if self.args.fading:
            #Frequency selectivity does not change over communication rounds!
            h_eff = self.chan.gen_frequency_response()
        else:
            h_eff = None

        for idx in range(self.args.T):
         
            if idx == 0:
                src = torch.cat([
                    bVec_md,
                    torch.zeros(self.args.batchSize, self.args.ell,
                                2 * (self.args.T - 1)).to(self.args.device)
                ], dim=-1)
            elif idx == self.args.T - 1:
                src = torch.cat([
                    bVec_md,
                    parity_all,
                    parity_fb
                ], dim=-1)
            else:
                src = torch.cat([
                    bVec_md,
                    parity_all,
                    torch.zeros(self.args.batchSize, self.args.ell,
                                self.args.T - (idx + 1)).to(self.args.device),
                    parity_fb,
                    torch.zeros(self.args.batchSize, self.args.ell,
                                self.args.T - (idx + 1)).to(self.args.device)
                ], dim=-1)
            
            src_split = split_complex_to_real_vector(src) 

            # Neural Network Part
            if self.args.nn_model == 'GBAF':
                output = self.Tmodel(src_split, None, self.pe)
            else:
                output = self.Tmodel(src_split)

            output_complex = reconstruct_complex_from_real_vector(output)   
            parity_complex = self.power_constraint(output_complex, isTraining, eachbatch, idx)
            parity_split = split_complex_to_real_vector(parity_complex)
            parity_split = self.total_power_reloc(parity_split, idx)
            parity = reconstruct_complex_from_real_vector(parity_split) # freq. domain parity symbols

            y_complex = (self.chan(parity.view(args.batchSize, -1), fwd_noise_var, self.args.fading, h_eff, self.args.equalizer)).view(args.batchSize, args.ell, 1)

            if idx == 0:
                parity_fb = (self.chan(y_complex.view(args.batchSize, -1), fb_noise_var, self.args.fading, h_eff, self.args.equalizer)).view(args.batchSize, args.ell, 1)
                parity_all = parity
                received = y_complex
            else:
                parity_fb = torch.cat([parity_fb, (self.chan(y_complex.view(args.batchSize, -1), fb_noise_var, self.args.fading, h_eff, self.args.equalizer)).view(args.batchSize, args.ell, 1)], dim=-1)
                parity_all = torch.cat([parity_all, parity], dim=-1)
                received = torch.cat([received, y_complex], dim=-1)
        
        received_split = split_complex_to_real_vector(received)

        if self.args.nn_model == 'GBAF':
            decSeq = self.Rmodel(received_split, None, self.pe)
        else:
            decSeq = self.Rmodel(received_split)

        #Deinterleaver
        if self.args.intlv:
            decSeq_int = torch.matmul(self.interleaver.transpose(-2, -1), decSeq)
        else:
            decSeq_int = decSeq

        if self.args.llr_data:
            if self.args.coded:
                preds = torch.log(decSeq_int + 1e-10)
                #Soft LLR values based on log-probabilities
                llr_vals = preds[..., 0] - preds[..., 1]   
                return llr_vals
            else:
                raise ValueError("Invalid output!")
        else:
            if self.args.coded:
                if self.args.E2E:
                    preds = torch.log(decSeq_int + 1e-8)
                    #Soft LLR values based on log-probabilities
                    llr_vals = preds[..., 0] - preds[..., 1]
                    llr_hat = codec.decode(llr_vals)
                else:
                    return decSeq_int
            else:
                llr_hat = decSeq_int

            return llr_hat

def individual_log_probs(llr):
    p0 = torch.exp(llr) / (1 + torch.exp(llr))
    p1 = 1 - p0
    return torch.stack((torch.log(p0), torch.log(p1)), dim=-1) # return (B, N, 2)

def train_model(model, args):
    print("-->-->-->-->-->-->-->-->-->--> start training ...", flush=True)

    best_loss = float('inf')

    model.train()

    if args.coded:
        H, zc, k_ldpc, n_ldpc = get_5g_ldpc_parity_matrix(args.k, args.n)
        codec = LinearCodec(info_len=args.k, coded_len=args.n, zc=zc, k_ldpc=k_ldpc, n_ldpc=n_ldpc, H=H, device=args.device, llr_clip=args.llr_sat, max_decode_iter=args.bp_iter)
    else:
        codec = None

    for eachbatch in range(args.totalbatch):

        u = torch.randint(0, 2, (args.batchSize, args.k), dtype=torch.float32).to(args.device)

        if args.coded:
            with torch.no_grad():
                bVec = codec.encode(u).float()
        else:
            bVec = u

        snr1 = random.uniform(args.snr_low, args.snr_high)

        fwd_noise_var = 10 ** (-snr1 * 1.0 / 10) 
        
        if args.snr2 == 100:
            fb_noise_var = 0 
        else:
            fb_noise_var = 10 ** (-args.snr2 * 1.0 / 10) 

        preds = model(eachbatch, bVec, 
                      fwd_noise_var,
                      fb_noise_var,
                      isTraining=1,
                      codec=codec)
        
        # torch.cuda.reset_peak_memory_stats()
        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        #     torch.cuda.synchronize()
        #     start_time = time.time()

        #     preds = model(eachbatch, bVec,
        #                 fwd_noise_var,
        #                 fb_noise_var,
        #                 isTraining=1,
        #                 codec=codec)
            
        #     torch.cuda.synchronize()
        #     elapsed_time = time.time() - start_time
        # peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        # print(f"Forward pass took: {elapsed_time:.4f} seconds")
        # print(f"Peak GPU memory usage: {peak_memory:.2f} GB")
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        args.optimizer.zero_grad()

        if args.coded:
            if args.E2E:
                llr_hat = individual_log_probs(preds)
                preds = llr_hat.contiguous().view(-1, llr_hat.size(-1))
            else:
                preds = preds.contiguous().view(-1, preds.size(-1))
                preds = torch.log(preds)
        else:
            preds = preds.contiguous().view(-1, preds.size(-1))
            preds = torch.log(preds)
        
        if (args.coded and args.E2E) or not(args.coded):
            ys = u.long().contiguous().view(-1)
        else:
            ys = bVec.long().contiguous().view(-1)

        # Mathematically equivalent to BCE (considering the setup and the dimensions)
        loss = F.nll_loss(preds, ys)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)

        args.optimizer.step()

        if args.use_lr_schedule:
            args.scheduler.step()

        if eachbatch % 1000 == 0: 
            with torch.no_grad():
                print('Training:', 'Idx, lr, snr1, snr2, BS, loss=',
                    (eachbatch, args.lr, snr1, args.snr2, args.batchSize,
                    round(loss.item(), 7)), flush=True)

        if loss.item() < best_loss and bool(snr1 < 7.5) and eachbatch > 80000: 
            best_loss = loss.item()
            if not os.path.exists('weights'):
                os.mkdir('weights')
            saveDir = 'weights/model_weights_best' + str(args.coded) + str(args.ModOrder) + str(args.intlv) + str(args.equalizer) + str(args.nn_model) + str(args.E2E) + str(args.snr2) + '.pth'
            torch.save(model.state_dict(), saveDir)
            print(f"Model saved at batch {eachbatch} with new best loss: {best_loss:.4f}", flush=True)

    if not os.path.exists('weights'):
        os.mkdir('weights')
    saveDir = 'weights/model_weights' + str(args.coded) + str(args.ModOrder) + str(args.intlv) + str(args.equalizer) + str(args.nn_model) + str(args.E2E) + str(args.snr2) + '.pth'
    torch.save(model.state_dict(), saveDir)
    print(f"Model saved (final)", flush=True)
    
    #save the model directory to the arguments
    args.saveDir = saveDir
    # Run the evaluation after training
    EvaluateNets(model, args)
    if args.coded and not(args.intlv):
        # Reevaluate with interleaver:
        args.intlv = True
        args.testintlv = 'Different'
        model.interleaver = model.random_interleaving_matrix()
        print(f"Testing with interleaving ...", flush=True)
        # Re-evaluate with interleaving
        EvaluateNets(model, args)


def EvaluateNets(model, args):

    print("-->-->-->-->-->-->-->-->-->--> start testing ...", flush=True)
    print(args.saveDir, flush=True)
    print(f"Number of Test batches: {args.numTestbatchSize}", flush=True)
    print(f'Interleaving enabled: {args.intlv}', flush=True)

    if args.coded:
        H, zc, k_ldpc, n_ldpc = get_5g_ldpc_parity_matrix(args.k, args.n)
        codec = LinearCodec(info_len=args.k, coded_len=args.n, zc=zc, k_ldpc=k_ldpc, n_ldpc=n_ldpc, H=H, device=args.device, llr_clip=args.llr_sat, max_decode_iter=args.bp_iter)
    else:
        codec = None

    checkpoint = torch.load(args.saveDir)
    model.load_state_dict(checkpoint)
    model.eval()

    snr_vector = [6, 7, 8, 9, 10]
    print(snr_vector, flush=True)
    ber_list = []

    with torch.no_grad():

        for snr1 in snr_vector:

            print(f'Testing for forward snr: {snr1} ...', flush=True)
            print(f"Number of Test batches: {args.numTestbatchSize}", flush=True)

            bitErrors = 0

            for eachbatch in range(args.numTestbatchSize):

                u = torch.randint(0, 2, (args.batchSize, args.k), dtype=torch.float32).to(args.device)

                if args.coded:
                    bVec = codec.encode(u).float()
                else:
                    bVec = u

                fwd_noise_var = 10 ** (-snr1 * 1.0 / 10)

                if args.snr2 == 100:
                    fb_noise_var = 0
                else:
                    fb_noise_var = 10 ** (-args.snr2 * 1.0 / 10)

                preds = model(eachbatch, bVec, fwd_noise_var, fb_noise_var, isTraining=0, codec=codec)

                if (args.coded and args.E2E):
                    # preds are LLRs (output of outer decoder) in the coded (outer) case
                    dec_bits = (preds < 0)
                    decodeds = dec_bits.long().contiguous().view(-1)
                elif (args.coded and not(args.E2E)):
                    preds = torch.log(preds + 1e-8)
                    #Soft LLR values based on log-probabilities
                    llr_vals = preds[..., 0] - preds[..., 1]
                    llr_hat = codec.decode(llr_vals)
                    dec_bits = (llr_hat < 0)
                    decodeds = dec_bits.long().contiguous().view(-1)
                elif not(args.coded):
                    preds = preds.contiguous().view(-1, preds.size(-1))
                    _, decodeds = preds.max(dim=1)
                else:
                    raise ValueError("Invalid Setup.")  

                ys = u.long().contiguous().view(-1)

                bitErrors += int(torch.sum(decodeds != ys.to(args.device)))

            BER = bitErrors / (args.numTestbatchSize * args.batchSize * args.k)
            ber_list.append(BER)
            print('SNR:', snr1, 'BER:', BER, flush=True)
        
        if args.coded:
            berFile = 'ber_vector' + str(args.coded) + str(args.ModOrder) + str(args.intlv) + str(args.equalizer) + str(args.nn_model) + str(args.E2E) + str(args.snr2) + str(args.testintlv) 
        else:
            berFile = 'ber_vector' + str(args.coded) + str(args.ModOrder) + str(args.equalizer) + str(args.nn_model) + str(args.snr2) # No interleaving in the baseline case

        np.savetxt(berFile, np.array(ber_list))



def llr_analysis(model, args):

    print("-->-->-->-->-->-->-->-->-->--> start collecting llr ...", flush=True)

    if args.coded:
        H, zc, k_ldpc, n_ldpc = get_5g_ldpc_parity_matrix(args.k, args.n)
        codec = LinearCodec(info_len=args.k, coded_len=args.n, zc=zc, k_ldpc=k_ldpc,
                            n_ldpc=n_ldpc, H=H, device=args.device,
                            llr_clip=args.llr_sat, max_decode_iter=args.bp_iter)
    else:
        codec = None

    model.load_state_dict(torch.load(args.saveDir))
    model.eval()

    snr_vector = [6, 10]
    args.numTestbatch = 25000
    print(f"Number of Test batches: {args.numTestbatch}", flush=True)

    bins = np.linspace(-21, 21, 66)
    bin_width = bins[1] - bins[0]
    hist_counts = {snr: np.zeros(len(bins) - 1, dtype=np.int64) for snr in snr_vector}
    totals = {snr: 0 for snr in snr_vector}

    with torch.no_grad():
        for snr1 in snr_vector:
            fwd_noise_var = 10 ** (-snr1 / 10)
            fb_noise_var = 0 if args.snr2 == 100 else 10 ** (-args.snr2 / 10)

            for eachbatch in range(args.numTestbatch):
                if args.coded:
                    bVec = torch.randint(0, 2, (args.batchSize, args.n), dtype=torch.float32,
                                         device=args.device)
                else:
                    bVec = torch.randint(0, 2, (args.batchSize, args.k), dtype=torch.float32,
                                         device=args.device)

                llr = model(eachbatch, bVec, fwd_noise_var, fb_noise_var, isTraining=0, codec=codec)
                llr_cpu = llr.view(-1).cpu().numpy()
                counts, _ = np.histogram(llr_cpu, bins=bins)
                hist_counts[snr1] += counts
                totals[snr1] += llr_cpu.size
                del llr, llr_cpu
                torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, len(snr_vector), figsize=(10, 6))
    for i, snr in enumerate(sorted(snr_vector)):
        density = hist_counts[snr] / (totals[snr] * bin_width)
        ax = axes[i]
        ax.bar(bins[:-1], density, width=bin_width, align='edge', alpha=0.75, edgecolor='black')
        ax.set_title(f"Feedforward SNR: {snr} dB", fontsize=22)
        ax.set_xlabel("LLR Values", fontsize=22)
        ax.set_ylabel("Density", fontsize=22)
        ax.set_xlim(-21, 21)
        ax.set_ylim(0, 0.07)
        ax.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.savefig("llr_histogram_fb_noisy_direct.pdf", dpi=1000)
    plt.close(fig)

if __name__ == '__main__':

    args = args_parser()
    # args.saveDir = "weights/model_weightsTrue4FalseZFGBAFTrue20final.pth" # for testing
    args.saveDir = "weights/model_weightsTrue4FalseZFGBAFFalse20final.pth" # for testing
    args.d_model_trx = args.heads_trx * args.d_k_trx

    json_file_path = "parameters/args" + str(args.coded) + str(args.ModOrder) + str(args.intlv) + str(args.equalizer) + str(args.nn_model) + str(args.E2E) + str(args.snr2) + ".json"

    # Rescale based on Modulation Order
    args.m = int(args.m / math.log2(args.ModOrder))

    if args.train:
        os.makedirs("parameters", exist_ok=True)
        with open(json_file_path, "w") as f:
            json.dump(vars(args), f, indent=8)
    
    model = AE(args).to(args.device)
    print("Neural Model:", args.nn_model, flush=True)
    print(f'Outer coding enabled: {args.coded}, BP iterations: {args.bp_iter}', flush=True)
    print(f'Interleaving enabled: {args.intlv}', flush=True)
    print("Modulation Order:", args.ModOrder, flush=True)
    print("Equalizer:", args.equalizer, flush=True)
    print('E2E training with the outer code:', args.E2E, flush=True)
    print('Batch size:', args.batchSize, flush=True)
    print('Total batches:', args.totalbatch, flush=True)
    print('Feedback SNR (dB):', args.snr2, flush=True)
    print('Number of Test batches:', args.numTestbatchSize, flush=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}", flush=True)

    # Analysis of learned LLR values (output of the neural decoder)
    if args.llr_data:
        if args.coded:
            llr_analysis(model, args)
            exit(0)
        else:
            raise ValueError('Invalid LLR Analysis')

    if args.train == 1:
        if args.opt_method == 'adamW':
            args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                               betas=(0.9, 0.999), eps=1e-08,
                                               weight_decay=args.wd, amsgrad=False)
        else:
            args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                              betas=(0.9, 0.98), eps=1e-9)
        if args.use_lr_schedule:
            lambda1 = lambda epoch: (1 - epoch / args.totalbatch)
            args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)
        train_model(model, args)
    else:
        EvaluateNets(model, args)
        if args.coded and not(args.intlv):
            # Reevaluate with interleaver:
            args.intlv = True
            args.testintlv = 'Different'
            model.interleaver = model.random_interleaving_matrix()
            print(f"Testing with interleaving ...", flush=True)
            # Re-evaluate with interleaving
            EvaluateNets(model, args)

