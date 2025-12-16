from configs.common_configs import args

args.model_comment = 'electricity'
args.data = 'electricity'
args.root_path = './dataset/electricity'
args.data_path = 'tokyo.csv'

args.source_data_path = args.data_path

args.features = 'M'
args.feature_cols = ['Electricity','Renewable_energy', 'Nuclear', 'Coal', 'Hydro', 'Geothermal', 'Biomass','Solar', 'Solar_curtailment', 'Wind', 'Wind_ccurtailment',
                     'Water_pumping', 'Interconnection', 'Temperature', 'Relative_humidity', 'Precipitation', 'Dew_point', 'Vapor_pressure', 'Wind_speed', 'Sunshine_duration',
                     'Global_horizontal_irradiance']
args.target = ['Electricity', 'Renewable_energy', 'Coal']

args.num_train = 8760*2+24
args.num_test = 8760
args.seq_len = 72
args.pred_len = 168
args.label_len = args.seq_len
args.forecast_dim = 1
args.enc_in = 22
args.dec_in = 22
args.c_out = 1
args.scale = True
