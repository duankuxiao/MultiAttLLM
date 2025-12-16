from utils.print_args import print_args
from utils.tools import load_content
import torch


def model_hyparameter_setup(args):
    args.use_norm = True
    args.learning_rate = 0.0001
    args.d_model = 512
    args.d_ff = 2048
    args.e_layers = 4
    args.d_layers = 1
    args.factor = 3
    args.moving_avg = 25

    if args.model == 'LLMformer':
        args.train_epochs = 30
        args.batch_size = 24
        args.learning_rate = 0.001
        args.patience = 6
        args.llm_model = 'BERT'  #GPT2
        args.d_model = 16
        args.d_ff = 64
        args.e_layers = 2
        args.d_layers = 2  # 3
        args.llm_layers = 2  # 6

    if 'RNN' in args.model:
        args.use_norm = False
        args.rnn_dim = 256
        args.rnn_layers = 2

    if args.model == 'Transformer':
        args.use_norm = False

    if args.model == 'DLinear':
        args.use_norm = False

    if args.model == 'Informer':
        args.train_epochs = 20
        args.use_norm = False
        args.factor = 5
        args.d_layers = 2  # default 2


    if args.model == 'Autoformer':
        args.e_layers = 2
        args.use_norm = False

    if args.model == 'iTransformer':  # default
        args.train_epochs = 20
        args.e_layers = 3
        args.d_model = 512
        args.d_ff = 512

    if args.model == 'PatchTST':
        args.train_epochs = 20
        args.use_norm = True
        args.d_model = 128
        args.d_ff = 256
        args.e_layers = 2
        args.n_heads = 8

    if args.model == 'TimesNet':
        args.e_layers = 2
        args.use_norm = True
        args.train_epochs = 10
        args.d_model = 32  # min{max[2**log(seq_dim),32],512} for forecast   min{max[2**log(seq_dim),64],128} for imputation
        args.d_ff = 32
        args.top_k = 5  # 5 for forecast   3 for imputation, classification, anomaly detection

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    args.inverse = True
    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.feature_cols is not None:
        args.enc_in = len(args.feature_cols)
        args.dec_in = len(args.feature_cols)
    args.c_out = len(args.target)

    if args.features == 'S':
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1

    print('Args in experiment:')
    print_args(args)
    return args
