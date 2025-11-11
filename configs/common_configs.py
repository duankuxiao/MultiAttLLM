import argparse

parser = argparse.ArgumentParser(description='Time-LLM')

# basic config
parser.add_argument('--task_name', type=str,  default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection,interval_forecast]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='forecast', help='model id')
parser.add_argument('--model_comment', type=str, default='PV', help='prefix when saving test results')
parser.add_argument('--model', type=str, default='DLinear',
                    help='model name, options: [Autoformer, TimeLLM,TimeLLMForecast, TimesNet, DLinear, Informer, Transformer, TimeMixer, iTransformer, TransformerX, RNN, PatchTST]')

# data loader
parser.add_argument('--data', type=str, default='electricity', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/electricity', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='kyushu.csv', help='data file PV_hour.csv or PV_hour_7f.csv')
parser.add_argument('--source_data_path', type=str, default='kyushu.csv', help='data file PV_hour.csv or PV_hour_7f.csv')

parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, ' 'MS:multivariate predict univariate')
parser.add_argument('--target', type=list, default=['Electricity'], help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./results/', help='location of model checkpoints')
parser.add_argument('--use_forecast', action='store_true', help='input forecast data', default=False)

parser.add_argument('--num_train', type=int, default=8760*2+24, help='train number of data')
parser.add_argument('--num_test', type=int, default=8760, help='test number of data')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# forecasting task
parser.add_argument('--seq_len', type=int, default=24*3, help='input sequence length')
parser.add_argument('--label_len', type=int, default=24, help='start token length')
parser.add_argument('--pred_len', type=int, default=24*7, help='prediction sequence length')
parser.add_argument('--seq_dim', type=int, default=11, help='input sequence length')
parser.add_argument('--pred_dim', type=int, default=1, help='input sequence length')
parser.add_argument('--forecast_dim', type=int, default=2, help='input sequence length')
parser.add_argument('--feature_cols', nargs='+', default=None, help="input features")

# imputation task
parser.add_argument('--mask_rate', type=float, default=0.2, help='mask_rate')
parser.add_argument('--mask_method', type=str, default='rdo', help='mask_method')

parser.add_argument('--ori_weight', type=float, default=1, help='ori_weight')
parser.add_argument('--missing_weight', type=float, default=1, help='missing_weight')
parser.add_argument('--pred_weight', type=float, default=1, help='pred_weight')
parser.add_argument('--input_inter',default=False, help='whether to use input interpolation')
parser.add_argument('--interpolate_method', type=str, default='linear', help=['linear','polynomial', 'spline'])
parser.add_argument('--interpolate_order', type=int, default=1, help='1<order<5 for spline, 2 or odd number for polynomial')
parser.add_argument('--mask_target_only', default=False, help='mask_target_only')
parser.add_argument('--output_ori', default=True, help='mask_target_only')


parser.add_argument('--seasonal_patterns', type=str, default='Hourly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--distil', action='store_false',help='whether to use distilling in encoder, using this argument means not using distilling',default=True)

parser.add_argument('--enc_in', type=int, default=11, help='encoder input size (seq features dim)')
parser.add_argument('--dec_in', type=int, default=11, help='decoder input size (forecast dim)')
parser.add_argument('--c_out', type=int, default=1, help='output size (pred dim)')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
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
parser.add_argument('--use_norm', type=bool, default=True, help='whether to use normalize; True  False ')

parser.add_argument('--factor', type=int, default=1, help='attn factor')
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
parser.add_argument('--scale', action='store_true', help='scale', default=True)

# RNN
parser.add_argument('--rnn_model', type=str, default='LSTM', help='RNN model')  # GRU, LSTM, seq2seq
parser.add_argument('--rnn_dim', type=int, default=256, help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--rnn_layers', type=int, default=2, help='bert_layers=6 llama_layers=32')

# optimization
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--accelerate', type=bool, default=False, help='accelerator')
parser.add_argument('--val', type=bool, default=True, help='validation')

parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=6, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate 0.0001 for other models  0.01 for LLM')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function MSE MAE')
parser.add_argument('--loss_method', type=str, default='MSE', help='mse msemu (fix missing) adaptive hybird')

parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--likelihood', type=str, default='g', help='exp description  g or nb')
parser.add_argument("--sample_size", type=int, default=500)
parser.add_argument("--fix_seed", type=int, default=4213)


# metrics (dtw)
parser.add_argument('--use_dtw', type=bool, default=False, help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

args = parser.parse_args()


