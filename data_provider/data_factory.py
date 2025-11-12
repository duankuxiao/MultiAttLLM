from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from torch.utils.data import DataLoader
from data_provider.data_loader_cumstom import Dataset_cumstom, Dataset_classification
from models.AutoTimes import Dataset_Custom as Dataset_Custom_AutoTimes
from data_provider.data_loader_LLM import Dataset_cumstom_llm
from data_provider.uea import collate_fn

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
}


def data_provider(args, flag):
    if args.data in data_dict:
        Data = data_dict[args.data]
    else:
        Data = Dataset_cumstom

    if args.model == 'TimeLLM':
        Data = Dataset_cumstom  # Dataset_solar_radiation_llm

    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    drop_last = True
    freq = args.freq
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    batch_size = 1 if (flag == 'test' or flag == 'TEST') else args.batch_size

    if args.task_name == 'classification':
        drop_last = False
        data_set = Dataset_classification(
            configs=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader

    else:
        data_set = Data(
            configs=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            scale=args.scale,
            seasonal_patterns=args.seasonal_patterns,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
