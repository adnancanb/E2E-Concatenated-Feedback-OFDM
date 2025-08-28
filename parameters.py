import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', type=int, default=384)
    parser.add_argument('--n', type=int, default=768)

    # Set to True when training for the train/test interleaver case
    # Otherwise, set to False
    parser.add_argument('--intlv', type=bool, default=False)

    parser.add_argument('--testintlv', type=str, default='Same') # Don't change this (only for saving purposes)
    parser.add_argument('--coded', type=bool, default=True)

    parser.add_argument('--ModOrder', type=int, default=4)
    parser.add_argument('--bp_iter', type=int, default=10) # 10 for E2E/ 20 for w/o E2E
    parser.add_argument('--llr_sat', type=float, default=10.0)

    parser.add_argument('--snr_low', type=float, default=5) 
    parser.add_argument('--snr_high', type=float, default=11) 
    
    parser.add_argument('--delta_f', type=int, default=30000)
    parser.add_argument('--channelMod', type=str, default='EVA')
    parser.add_argument('--fading', type=bool, default=True)

    parser.add_argument('--equalizer', type=str, default='ZF') # ZF as main equalization
    parser.add_argument('--llr_data', type=bool, default=False) # LLR analysis
    parser.add_argument('--E2E', type=bool, default=True, help='E2E training with the outer channel code (SET TRUE FOR FULLY NEURAL MODELS AS WELL!)')

    parser.add_argument('--nn_model', type=str, default='GBAF', help="GBAF/LIGHTCODE")

    parser.add_argument('--snr1', type=float, default=-1) # Not used
    parser.add_argument('--snr2', type=float, default=20) # feedback SNR (dB) 
    parser.add_argument('--m', type=int, default=12) # 12 for Coded/6 for Uncoded
    parser.add_argument('--ell', type=int, default=64)
    parser.add_argument('--T', type=int, default=9)
    parser.add_argument('--seq_reloc', type=int, default=1)

    parser.add_argument('--NS_model', type=int, default=2) # Selected in GBAF paper
    parser.add_argument('--heads_trx', type=int, default=1) 
    parser.add_argument('--d_k_trx', type=int, default=32) 
    parser.add_argument('--N_trx', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--custom_attn', type=bool, default=True)

    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--reloc', type=int, default=1)
    parser.add_argument('--totalbatch', type=int, default=150000) 
    parser.add_argument('--batchSize', type=int, default=4096) 
    parser.add_argument('--numTestbatchSize', type=int, default=50000) 
    parser.add_argument('--opt_method', type=str, default='adamW')
    parser.add_argument('--clip_th', type=float, default=0.5)
    parser.add_argument('--use_lr_schedule', type=bool, default=True)
    parser.add_argument('--multclass', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.0025) 
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda:0')

    # Lightcode model params
    parser.add_argument('--d_model_trx_ligthcode', type=int, default=16, help="feature dimension")
    parser.add_argument('--d_model_rec_lightcode', type=int, default=16, help="feature dimension")
    parser.add_argument('--enc_NS_model_lightcode', type=int, default=3)
    parser.add_argument('--dec_NS_model_lightcode', type=int, default=3)

    args = parser.parse_args()

    return args
