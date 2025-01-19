import torch
import os
from exp.exp_forecasting import Exp_Forecast
from utils.print_args import print_args
from utils.tools import load_content
import random
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

fix_seed = 4213
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def get_setting(args,ii):
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_sd{}_td{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_dropout{}_eb{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.enc_in,
        args.c_out,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.dropout,
        args.embed,
        args.des, ii)

    if 'TimeLLM' in args.model:
        setting += '_{}_llmd{}_llmf{}_tk{}'.format(args.llm_model, args.llm_dim, args.llm_layers, args.top_k)
        if args.use_prompt:
            setting += '_prompt'
    if 'RNN' in args.model:
        setting += '_{}_rnnd{}_rnnf{}'.format(args.rnn_model, args.rnn_dim, args.rnn_layers, )

    if args.use_forecast:
        setting += '_forecast'
    if args.percent != 100:
        setting = 'few-shot{}_'.format(args.percent) + setting
    if args.scale:
        setting += '_scale'
    return setting


def main(args):
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    args.inverse = True
    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.content = load_content(args)

    if 'TimeLLM' in args.model:
        args.batch_size = 24
        args.learning_rate = 0.01
        args.content = load_content(args)
        if args.llm_model == 'LLAMA3b':
            args.llm_dim = 3072
        elif args.llm_model == 'LLAMA1b':
            args.llm_dim = 2048
        elif 'BERT' in args.llm_model:
            args.llm_dim = 768
        elif 'GPT2' in args.llm_model:
            args.llm_dim = 768
        else:
            raise ValueError('Unknown llm model')

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

    Exp = Exp_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            # setting record of experiments
            setting = get_setting(args,ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = get_setting(args,ii)

        exp = Exp(args)  # set experiments
        print(' >>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MultiAttLLMpaper')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='forecast', help='model id')
    parser.add_argument('--model_comment', type=str, default='PV', help='prefix when saving test results')
    parser.add_argument('--model', type=str, default='DLinear',
                        help='model name, options: [TimeLLM, TimesNet, DLinear, iTransformer, RNN, MultiAttLLM]')

    # data loader
    parser.add_argument('--data', type=str, default='electricity', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/electricity', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='tokyo.csv', help='data file PV_hour.csv or PV_hour_7f.csv')
    parser.add_argument('--source_data_path', type=str, default='tokyo.csv', help='data file PV_hour.csv or PV_hour_7f.csv')

    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; '
                                                                   'M:multivariate predict multivariate, S: univariate predict univariate, ' 'MS:multivariate predict univariate')
    parser.add_argument('--target', type=list, default=['Electricity'], help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, '
                                                              'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                                                              'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./results/', help='location of model checkpoints')
    parser.add_argument('--use_forecast', action='store_true', help='input forecast data', default=False)

    parser.add_argument('--num_train', type=int, default=8760 * 2 + 24, help='train number of data')
    parser.add_argument('--num_test', type=int, default=8760, help='test number of data')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=24 * 3, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24 * 7, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24 * 7, help='prediction sequence length')
    parser.add_argument('--seq_dim', type=int, default=11, help='input sequence length')
    parser.add_argument('--pred_dim', type=int, default=1, help='input sequence length')
    parser.add_argument('--forecast_dim', type=int, default=2, help='input sequence length')
    parser.add_argument('--feature_cols', type=list, default=None, help="input features")

    parser.add_argument('--seasonal_patterns', type=str, default='Hourly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)

    parser.add_argument('--enc_in', type=int, default=11, help='encoder input size (seq features dim)')
    parser.add_argument('--dec_in', type=int, default=11, help='decoder input size (forecast dim)')
    parser.add_argument('--c_out', type=int, default=1, help='output size (pred dim)')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=8, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--hidden_size', nargs='+', default=[256], help='output mlp layer')

    # Autoformer
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average for Autoformer')
    # TimeMixer
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size for TimeMixer')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers for TimeMixer')
    parser.add_argument('--down_sampling_method', type=str, default=None, help='down sampling method, only support avg, max, conv')
    parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

    # TimeLLM
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=1, help='')
    parser.add_argument('--llm_model', type=str, default='BERT', help='LLM model')  # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default=768, help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--llm_layers', type=int, default=6, help='bert_layers=6 llama_layers=32')
    parser.add_argument('--use_prompt', action='store_true', help='input forecast data', default=True)

    # RNN
    parser.add_argument('--rnn_model', type=str, default='LSTM', help='RNN model')  # GRU, LSTM, seq2seq
    parser.add_argument('--rnn_dim', type=int, default=512, help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--rnn_layers', type=int, default=2, help='bert_layers=6 llama_layers=32')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--accelerate', type=bool, default=False, help='accelerator')

    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate 0.0001 for other models  0.01 for LLM')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='PEM', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--percent', type=int, default=100)

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    args = parser.parse_args()

    for model in ['RNN', 'DLinear', 'iTransformer', 'TimesNet','TimeLLM', 'MultiAttLLM']:

        args.model_id = 'test'
        args.features = 'M'
        args.model = model
        args.llm_model = 'LLAMA1b'
        args.is_training = 1
        args.use_prompt = False

        args.d_model = 32
        args.d_ff = 64
        args.e_layers = 4
        args.d_layers = 4
        args.llm_layers = 32

        if args.model == 'MultiAttLLM':
            args.lradj = 'PEMS'
            args.d_model = 32
            args.d_ff = 64
            args.e_layers = 4
            args.d_layers = 4
            args.llm_layers = 6

        if args.model == 'DLinear':
            args.d_model = 256
            args.d_ff = 1024
            args.e_layers = 4
            args.d_layers = 1

        if args.model == 'iTransformer':
            args.d_model = 512
            args.d_ff = 2048
            args.e_layers = 4
            args.d_layers = 1

        if args.model == 'TimesNet':
            args.d_model = 64
            args.d_ff = 256
            args.e_layers = 2
            args.d_layers = 1

        args.feature_cols = ['Electricity','Renewable_energy', 'Nuclear', 'Coal', 'Hydro', 'Geothermal', 'Biomass','Solar', 'Solar_curtailment', 'Wind', 'Wind_ccurtailment','Water_pumping',
                             'Interconnection', 'Temperature', 'Relative_humidity', 'Precipitation', 'Dew_point', 'Vapor_pressure', 'Wind_speed', 'Sunshine_duration',
                             'Snowfall', 'Global_horizontal_irradiance']
        args.target = ['Electricity', 'Renewable_energy', 'Coal']

        if args.model == 'TimeLLM':
            args.feature_cols = ['Electricity', 'Renewable_energy', 'Coal']
            args.target = ['Electricity', 'Renewable_energy', 'Coal']
            args.d_model = 16
            args.d_ff = 32
            args.e_layers = 1
            args.d_layers = 1
            args.llm_layers = 6

        main(args)


