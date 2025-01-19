from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from exp.exp_forecasting import Exp_Forecast
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.tools import results_evaluation, save_config
from torch.optim import lr_scheduler


class Exp_FewShot(Exp_Forecast):
    def __init__(self, args):
        super(Exp_FewShot, self).__init__(args)

    def few_shot_train(self, setting, shot_num, path=None):
        assert 0 < shot_num < 1, "shot_num should be less than 1 and more than 0"
        print('loading model')
        if path is None:
            model_path = os.path.join(self.args.checkpoints, setting, 'checkpoints')
        else:
            model_path = os.path.join(path, 'checkpoints')
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint')))

        for name, param in self.model.named_parameters():
            if "output_projection" not in name:  # 假设最后的全连接层命名为 "fc"
                param.requires_grad = False
            if "output_projection" in name:
                param.requires_grad = True

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 仅采样shot_num%数量的数据进行few-shot训练
        length = len(train_data)
        train_data_sampler = torch.utils.data.Subset(train_data, torch.randperm(length)[:int(length * shot_num)])
        few_shot_loader = torch.utils.data.DataLoader(train_data_sampler, batch_size=self.args.batch_size, shuffle=True)

        path = os.path.join(self.args.checkpoints, setting, 'checkpoints')
        if not os.path.exists(path):
            os.makedirs(path)
        save_config(self.args, os.path.join(path, 'configs.pkl'))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(model_optim, train_loader)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast) in enumerate(few_shot_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                x_forecast = x_forecast.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)

                        outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)

                    outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                verbose_interval = (len(train_data) // 10) if len(train_data) > 10 else 1
                if (i + 1) % verbose_interval == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.2f}min'.format(speed, left_time / 60))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            cost_time = round((time.time() - epoch_time) / 60, 2)
            print("     Epoch: {} cost time: {} min".format(epoch + 1, cost_time))
            print("         Train Loss: {0:.7f} Vali Loss: {1:.7f}".format(train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            left_time = 1 + (self.args.patience - early_stopping.counter) * cost_time
            print("          Left time: {} min".format(left_time))

            if self.args.lradj != 'TST':
                if self.args.lradj == 'COS':
                    scheduler.step()
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        self.args.learning_rate = model_optim.param_groups[0]['lr']
                        print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)

            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
