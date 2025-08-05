from configs.common_configs import args

args.model_comment = 'electricity'  # 能源供应
args.data = 'electricity'
args.root_path = './dataset/electricity'
args.data_path = 'tokyo.csv'
# args.data_path = 'kyushu.csv'
# args.data_path = 'Texas.csv'

args.source_data_path = args.data_path

args.features = 'M'
args.feature_cols = ['Electricity','Renewable_energy', 'Nuclear', 'Coal', 'Hydro', 'Geothermal', 'Biomass','Solar', 'Solar_curtailment', 'Wind', 'Wind_ccurtailment',
                     'Water_pumping', 'Interconnection', 'Temperature', 'Relative_humidity', 'Precipitation', 'Dew_point', 'Vapor_pressure', 'Wind_speed', 'Sunshine_duration',
                     'Global_horizontal_irradiance']
args.target = ['Electricity', 'Renewable_energy', 'Coal']

if args.data_path == 'Texas.csv':
    args.feature_cols = ['COAST', 'EAST', 'FWEST', 'NORTH', 'NCENT', 'SOUTH', 'SCENT', 'WEST', 'REGDN', 'REGUP', 'RRS', 'NSPIN', 'WIND_ACTUAL_SYSTEM_WIDE',
                         'SOLAR_ACTUAL_SYSTEM_WIDE', 'LZ_AEN', 'LZ_CPS', 'LZ_HOUSTON', 'LZ_LCRA', 'LZ_NORTH', 'LZ_RAYBN', 'LZ_SOUTH', 'LZ_WEST']
    args.target = args.feature_cols
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


# ['Electricity', 'Nuclear', 'Coal', 'Hydro', 'Geothermal', 'Biomass',
#        'Solar', 'Solar_curtailment', 'Wind', 'Wind_ccurtailment',
#        'Water_pumping', 'Interconnection', 'Total', 'Temperature',
#        'Relative_humidity', 'Precipitation', 'Dew_point', 'Vapor_pressure',
#        'Wind_speed', 'Sunshine_duration', 'Snowfall',
#        'Global_horizontal_irradiance']

if __name__ == '__main__':
    import pandas as pd
    import os
    data = pd.read_csv(os.path.join(r'D:\Time-LLM-main\dataset\electricity', 'Texas.csv'))
    print(data.columns)

