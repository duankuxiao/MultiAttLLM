from utils.print_args import print_args
from utils.tools import load_content
import torch


def model_hyperparameter_setup(args):
    args.use_norm = True
    args.learning_rate = 0.001
    args.patience = 6
    args.train_epochs = 100

    args.d_model = 512
    args.d_ff = 2048
    args.e_layers = 4
    args.d_layers = 1
    args.factor = 3
    args.moving_avg = 25

    if args.model == 'LLMformer':
        args.batch_size = 10
        args.use_norm = True
        args.learning_rate = 0.001

        args.patience = 6
        args.llm_model = 'GPT2'
        args.d_model = 16
        args.d_ff = 64
        args.e_layers = 2
        args.d_layers = 3
        args.llm_layers = 6

    if 'RNN' in args.model:
        args.use_norm = False
        args.rnn_dim = 256
        args.rnn_layers = 2

    if args.model == 'Transformer':
        args.use_norm = False
        args.d_model = 256
        args.d_ff = 512

    if args.model == 'DLinear':
        args.use_norm = False

    if args.model == 'Informer':
        args.use_norm = False
        args.factor = 5
        args.d_layers = 2  # default 2  imputatiaon 1
        args.d_model = 128
        args.d_ff = 128

    if args.model == 'Autoformer':
        args.e_layers = 2
        args.d_model = 512
        args.d_ff = 2048
        args.use_norm = False

    if args.model == 'iTransformer':  # default
        args.e_layers = 3
        args.d_model = 512
        args.d_ff = 512

    if args.model == 'PatchTST':
        args.use_norm = True
        args.d_model = 128
        args.d_ff = 256
        args.e_layers = 2
        args.n_heads = 8
        args.d_model = 512
        args.d_ff = 512
        args.e_layers = 4

    if args.model == 'TimesNet':
        args.e_layers = 2
        args.use_norm = True
        args.train_epochs = 10
        args.d_model = 64  # min{max[2**log(seq_dim),32],512} for forecast   min{max[2**log(seq_dim),64],128} for imputation
        args.d_ff = 64
        args.top_k = 3  # 5 for forecast   3 for imputation, classification, anomaly detection

    if 'LLM' in args.model:
        args.content = load_content(args)
        if args.llm_model == 'LLAMA8b':
            args.llm_dim = 4096
        elif args.llm_model == 'LLAMA3b':
            args.llm_dim = 3072
        elif args.llm_model == 'LLAMA1b':
            args.llm_dim = 2048
        elif 'BERT' in args.llm_model:
            args.llm_dim = 768
        elif 'GPT2' in args.llm_model:
            args.llm_dim = 768
        else:
            raise ValueError('Unknown llm model')

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
