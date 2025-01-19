import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_cumstom(Dataset):
    def __init__(self,configs, root_path, flag='train', size=None,
                 features='S', data_path='PV_power.csv',
                 target=['PV'], scale=True, timeenc=0, freq='h', percent=100,seasonal_patterns=None):

        self.num_train = configs.num_train
        self.num_test = configs.num_test

        self.source_data_path = configs.source_data_path
        self.forecast_dim = configs.forecast_dim
        self.feature_cols = configs.feature_cols
        self.c_out = configs.c_out
        if size == None:
            self.seq_len = 24 * 3
            self.label_len = 24
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        # s_begin = (index // 24) * 24
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        x_forecast = self.data_forecast[r_begin:r_end, :self.forecast_dim]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, x_forecast

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        df_source_domain = pd.read_csv(os.path.join(self.root_path, self.source_data_path))
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.feature_cols is None:
            self.feature_cols = df_raw.columns[1:]
        cols = list(self.feature_cols.copy())

        if 'date' in cols:
            cols.remove('date')

        for s in self.target:
            if s in cols:
                cols.remove(s)
        df_raw = df_raw[['date'] + cols + self.target]
        df_source_domain = df_source_domain[['date'] + cols + self.target]

        num_vali = len(df_raw) - self.num_train - self.num_test
        border1s = [0, self.num_train - self.seq_len, len(df_raw) - self.num_test - self.seq_len]
        border2s = [self.num_train, self.num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_target = df_raw[self.target]

        cols_data_source_domain = df_source_domain.columns[1:]
        df_data_source_domain = df_source_domain[cols_data_source_domain]
        df_target_source_domain = df_source_domain[self.target]

        if self.scale:
            '''
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.target_scaler.fit(df_target.values)
            '''

            train_data = df_data_source_domain[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.target_scaler.fit(df_target_source_domain.values)

        else:
            data = df_data.values

        self._get_dataset(df_raw, data, border1, border2)

    def _get_dataset(self,df_raw,data,border1,border2):
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)


        if self.features == 'S':
            self.data_x = data[border1:border2, -1:]
            self.data_y = data[border1:border2, -1:]
        else:
            self.data_x = data[border1:border2, :len(self.feature_cols)]
            self.data_y = data[border1:border2, -len(self.target):]

        self.data_forecast = data[border1:border2, :self.forecast_dim]
        self.data_stamp = data_stamp


if __name__ == '__main__':
    from utils.tools import heatmap
    # folder = r'D:\Time-LLM-main\dataset\pv'
    # data_file = 'PV_hour.csv'
    # data = pd.read_csv(os.path.join(folder, data_file),index_col=0)
    # print(data.columns)
    # # data = data[['Temperature', 'Relative_humidity',  'Wind_speed', 'Global_horizontal_irradiance', 'System_price', 'Sell_volume', 'Buy_volume', 'Total_volume',
    # #                      'Sell_volume_block_orders', 'Sell_volume_contracted_block_orders', 'Buy_volume_block_orders', 'Buy_volume_contracted_block_orders',  'Price']]
    # data = data[['Temperature','Relative_humidity','Precipitation','Dew_point','Vapor_pressure','Wind_speed','Sunshine_duration','Snowfall','Global_horizontal_irradiance']]
    # data_path = 'pv'
    # heatmap(data,data_path)

    folder = r'D:\Time-LLM-main\dataset\electricity'
    # data_file = 'kyushu.csv'
    # data = pd.read_csv(os.path.join(folder, data_file), index_col=0)
    # data_path = 'kyushu_electricity'
    # heatmap(data, data_path)

    # data_file = 'spot_index.csv'
    # data = pd.read_csv(os.path.join(folder, data_file))
    # df_repeated = data.loc[data.index.repeat(24)].reset_index(drop=True)
    # df_repeated.to_csv(os.path.join(folder, data_file[:-4] + '_24h.csv'), index=False)
    # print(df_repeated)

    # data_file = 'intraday_30min.csv'
    # df = pd.read_csv(os.path.join(folder, data_file))
    # data = df.iloc[::2, :]
    # print(data)
    # data.to_csv(os.path.join(folder, data_file[:-4] + '_1h.csv'))

    folder = r'D:\Time-LLM-main\dataset\electricity\hokkaido'
    file_names = [f'sup_dem_results_{year}_{quarter}q.csv' for year in range(2019, 2024) for quarter in range(1, 5)]
    file_names = file_names[3:-1]  # 限制到 '23_3q'


    # 读取并合并所有文件
    dfs = []
    for file_name in file_names:
        file_path = os.path.join(folder, file_name)
        # # 读取txt文件为DataFrame，假设是逗号分隔
        # df = pd.read_csv(os.path.join(folder,file_name[:-4]+'.txt'), delimiter=',', encoding='utf-8')
        # output_csv_path = file_path
        # # 将DataFrame保存为csv文件
        # df.to_csv(output_csv_path, index=False, encoding='utf-8')

        if os.path.exists(file_path):  # 检查文件是否存在
            df = pd.read_csv(file_path,encoding='SHIFT-JIS').iloc[3:,:]
            dfs.append(df)

    # 合并所有DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # 保存为一个新的csv文件
    combined_df.to_csv(os.path.join(folder,'hokkaido.csv'), index=False, encoding='SHIFT-JIS')