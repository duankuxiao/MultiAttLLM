import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.interval_forecasting_tools import negative_binomial_loss, gaussian_sample, negative_binomial_sample, GaussianLikelihoodLoss
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import results_probability_forecast_evaluation, results_evaluation
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.tools import save_config
from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')

class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)
        self.likelihood = args.likelihood
        self.loss_method = args.loss_method
        if self.loss_method == "adaptive":
            self.log_sigma_mse = nn.Parameter(torch.zeros(1))
            self.log_sigma_nll = nn.Parameter(torch.zeros(1))

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        param_list = list(self.model.parameters())
        if self.loss_method == "adaptive":
            # 将不确定性参数也加入到优化器中
            param_list += [self.log_sigma_mse, self.log_sigma_nll]
        model_optim = optim.Adam(param_list, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.loss_method == 'MSE':
            criterion = nn.MSELoss()
            return criterion
        if self.loss_method == 'g':
            # criterion = nn.GaussianNLLLoss(reduction='mean')
            criterion = GaussianLikelihoodLoss(full=True,reduction='mean')
            return criterion
        if self.loss_method == "adaptive" or self.loss_method == "hybridmu":
            MSEcriterion = nn.MSELoss()
            NLLcriterion = GaussianLikelihoodLoss(full=True,reduction='mean')
            return [MSEcriterion, NLLcriterion]

    def _select_scheduler(self, model_optim, train_loader):
        train_steps = len(train_loader)
        if self.args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        return scheduler

    def _loss_function(self, criterion, pred, true, mu, sigma):
        if self.loss_method == "g":
            loss = criterion(mu, true, sigma)
        elif self.loss_method == "nb":
            loss = negative_binomial_loss(mu, true, sigma)
        elif self.loss_method == "mse":
            loss = criterion(pred, true)
        elif self.loss_method == "msemu":
            loss = criterion(mu, true)
        elif self.loss_method == "hybridmu":
            loss = criterion[0](pred, true) + criterion[1](mu, true, sigma)
        elif self.loss_method == "adaptive":
            mse_loss = criterion[0](pred, true)
            nll_loss = criterion[1](mu, true, sigma)
            # loss = 0.5 * (torch.exp(-self.log_sigma_mse.to(true.device)) * mse_loss + torch.exp(-self.log_sigma_nll.to(true.device)) * nll_loss)
            loss = 0.5 * (torch.exp(-self.log_sigma_mse.to(true.device)) * mse_loss + torch.exp(-self.log_sigma_nll.to(true.device)) * nll_loss +
                          self.log_sigma_mse.to(true.device) + self.log_sigma_nll.to(true.device))
        return loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                x_forecast = x_forecast.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast():
                        outputs, mu, sigma = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)
                else:
                    outputs, mu, sigma = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)

                if self.args.accelerate:
                    outputs, batch_y = self.accelerator.gather_for_metrics((outputs, batch_y))
                outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                mu, sigma = mu.detach().cpu(), sigma.detach().cpu()

                loss = self._loss_function(criterion, pred, true, mu, sigma)
                total_loss.append(np.array(loss))
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting, 'checkpoints')
        if not os.path.exists(path):
            os.makedirs(path)
        save_config(self.args, os.path.join(path, 'configs.pkl'))

        time_now = time.time()
        time_start = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,accelerator=self.accelerator)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(model_optim, train_loader)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        if self.args.accelerate:
            self.model,train_loader,vali_loader, model_optim,scheduler = self.accelerator.prepare(self.model,train_loader,vali_loader,model_optim,scheduler)
            self.accelerator.print(f"Process {self.accelerator.process_index} is using device {self.accelerator.device}")

        # Initialize a dictionary to store loss values
        loss_records = {"epoch": [],"time": [], "train_loss": [], "vali_loss": []}
        if self.loss_method == 'adaptive':
            loss_records = {"epoch": [], "time": [],"train_loss": [], "vali_loss": [], "mse_weight": [], "nll_weight": []}

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                if self.args.accelerate:
                    batch_x = batch_x.float()
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float()
                    batch_y_mark = batch_y_mark.float()
                    x_forecast = x_forecast.float()

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                else:
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
                    with torch.cuda.amp.autocast():
                        outputs, mu, sigma = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)

                        outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]

                        if self.args.accelerate:
                            batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:]
                        else:
                            batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)
                        loss = self._loss_function(criterion, outputs, batch_y, mu, sigma)

                        train_loss.append(loss.item())
                else:
                    outputs, mu, sigma = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)
                    outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]

                    if self.args.accelerate:
                        batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:]
                    else:
                        batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)

                    loss = self._loss_function(criterion, outputs, batch_y, mu, sigma)
                    train_loss.append(loss.item())
                    if torch.isnan(loss):
                        break

                if self.args.accelerate:
                    self.accelerator.print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch - 1) * train_steps - i)
                    self.accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.2f}min'.format(speed, left_time / 60))
                else:
                    verbose_interval = (len(train_loader) // 5) if len(train_loader) > 5 else 1
                    if (i + 1) % verbose_interval == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch - 1) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.2f}min'.format(speed, left_time / 60))
                        iter_count = 0
                        time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if self.args.accelerate:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()
                        model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False,accelerator=self.accelerator)
                    scheduler.step()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            # Record loss values
            loss_records["epoch"].append(epoch + 1)
            loss_records["time"].append(round((time.time() - time_start)/60,4))
            loss_records["train_loss"].append(train_loss)
            loss_records["vali_loss"].append(vali_loss)
            if self.loss_method == 'adaptive':
                loss_records["mse_weight"].append(self.log_sigma_mse.detach().numpy())
                loss_records["nll_weight"].append(self.log_sigma_nll.detach().numpy())

            # test_loss = self.vali(test_data, test_loader, criterion)
            cost_time = round((time.time() - epoch_time) / 60, 2)
            print(" Epoch: {} cost time: {} min".format(epoch + 1, cost_time))
            print("☆☆☆☆☆Train Loss: {0:.7f} Vali Loss: {1:.7f}".format(train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)

            if np.isnan(vali_loss):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break

            left_time = 1 + (self.args.patience - early_stopping.counter) * cost_time
            print("  Left time: {} min".format(round(left_time,2)))

            if self.args.lradj != 'TST':
                if self.args.lradj == 'COS':
                    scheduler.step()
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    # if epoch == 0:
                    #     self.args.learning_rate = model_optim.param_groups[0]['lr']
                    #     print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)

            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        if self.args.accelerate:
            self.accelerator.wait_for_everyone()

        best_model_path = path + '/' + 'checkpoint'
        if self.args.accelerate:
            self.model = self.accelerator.load_state(best_model_path)
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        # Convert loss records to DataFrame and save as CSV
        folder_path = os.path.join(self.args.checkpoints, setting)
        loss_df = pd.DataFrame(loss_records)
        loss_df.to_csv(os.path.join(folder_path, "loss_records.csv"), index=False)
        print("Loss records saved to:", os.path.join(folder_path, "loss_records.csv"))
        report = torch.cuda.memory_summary(device=self.device, abbreviated=False)
        print(report)
        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        used_bytes = torch.cuda.memory_allocated(self.device)
        with open(os.path.join(folder_path, "memory_summary_{}_{}.txt".format(round(used_bytes * 1024 / (10 ** 9), 1), round(speed * 1000, 2))), "w") as f:
            f.write(report)
        print("Saved CUDA memory summary to cuda_memory_summary.txt")
        if self.loss_method == "adaptive":
            print('mse weight: {}, nll weight: {}'.format(self.log_sigma_mse,self.log_sigma_nll))
        return self.model, loss_df

    def test(self, setting, test=0, path=None):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            if path is None:
                model_path = os.path.join(self.args.checkpoints, setting, 'checkpoints')
                folder_path = os.path.join(self.args.checkpoints, setting)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            else:
                model_path = os.path.join(path, 'checkpoints')
                folder_path = path
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint')))
        else:
            folder_path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        preds ,trues = [], []
        mus,sigamas = [],[]

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast) in enumerate(test_loader):
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
                        outputs, mu, sigma = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)

                else:
                    outputs, mu, sigma = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)


                outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:]  # .to(self.device)
                mu = mu[:, -self.args.pred_len:, -self.f_dim:]
                sigma = sigma[:, -self.args.pred_len:, -self.f_dim:]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                mu = mu.detach().cpu().numpy()
                sigma = sigma.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    mu = test_data.inverse_transform(mu.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    sigma = sigma * test_data.std_

                outputs = outputs[:, :, -self.f_dim:]
                batch_y = batch_y[:, :, -self.f_dim:]
                mu = mu[:, :, -self.f_dim:]
                sigma = sigma[:, :, -self.f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                mus.append(mu)
                sigamas.append(sigma)
                verbose_interval = (len(test_data) // 2) if len(test_data) > 2 else 1
                verbose = False
                if verbose:
                    if (i + 1) % verbose_interval == 0:
                        input = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        res_path = os.path.join(folder_path + '/test_results/')
                        if not os.path.exists(res_path):
                            os.makedirs(res_path)
                        visual(gt, pd, os.path.join(res_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mus = np.concatenate(mus, axis=0)
        sigamas = np.concatenate(sigamas, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        [mse, rmse,nrmse, mae,mape,rae, r2,corr] = results_evaluation(trues.flatten(), preds.flatten())
        # print('mae:{}, r2:{}, dtw:{}'.format(mae, r2, dtw))
        f = open(os.path.join('./results', "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, r2:{}, dtw:{}'.format(mae, r2, dtw))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(os.path.join(folder_path, 'metrics_{}_{}.npy'.format(self.args.data,self.args.model_id)), np.array([mae, mse, rmse, r2, corr]))
        np.save(os.path.join(folder_path, 'pred_{}_{}.npy'.format(self.args.data,self.args.model_id)), preds)
        np.save(os.path.join(folder_path, 'true_{}_{}.npy'.format(self.args.data,self.args.model_id)), trues)
        np.save(os.path.join(folder_path, 'mu_{}_{}.npy'.format(self.args.data,self.args.model_id)), mus)
        np.save(os.path.join(folder_path, 'sigma_{}_{}.npy'.format(self.args.data,self.args.model_id)), sigamas)

        pred_res,metrics_df = self.res_evaluation_multi_target(trues, preds,mus,sigamas,trainable_params, folder_path)
        return pred_res,metrics_df

    def run_quantile_regression_baseline(self):
        import lightgbm as lgb
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        def train_quantile_model(X, y, quantile):
            params = {
                'objective': 'quantile',
                'alpha': quantile,
                'verbosity': -1
            }
            train_set = lgb.Dataset(X, y)
            model = lgb.train(params, train_set, num_boost_round=100)
            return model

        X_list, y_list = [], []
        for batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast in train_loader:
            # batch_x: [B, seq_len, num_features]
            # batch_y: [B, pred_len, num_targets]
            # 只取最后一个时间点作为输入（或 mean pooling）
            input_feats = batch_x[:, -1, :]  # [B, num_features]
            target_vals = batch_y[:, 0, :]  # [B, num_targets]，预测未来第一步

            X_list.append(input_feats.numpy())
            y_list.append(target_vals.numpy())

        X_all = np.concatenate(X_list, axis=0)  # [N, num_features]
        y_all = np.concatenate(y_list, axis=0)  # [N, num_targets]
        num_targets = y_all.shape[1]
        models = []

        for i in range(num_targets):
            y_target = y_all[:, i]

            lower_model = train_quantile_model(X_all, y_target, 0.05)
            median_model = train_quantile_model(X_all, y_target, 0.5)
            upper_model = train_quantile_model(X_all, y_target, 0.95)

            models.append((lower_model, median_model, upper_model))
        X_test = []
        Y_true = []

        for batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast in test_loader:
            input_feats = batch_x[:, -1, :]  # [B, num_features]
            X_test.append(input_feats.numpy())
            Y_true.append(batch_y[:, 0, :].numpy())  # 只预测第一步

        X_test = np.concatenate(X_test, axis=0)  # [N, num_features]
        Y_true = np.concatenate(Y_true, axis=0)  # [N, num_targets]

        picp90_list, piw90_list = [], []
        mse_list, rmse_list, mae_list, r2_list, corr_list, mape_list = [], [], [], [], [], []
        nrmse_list,rae_list = [],[]
        columns_list = []
        for i in self.args.target:
            columns_list.append('{}_true'.format(i))
            columns_list.append('{}_mean'.format(i))
            columns_list.append('{}_95'.format(i))
            columns_list.append('{}_5'.format(i))

        res_df = pd.DataFrame(columns=columns_list)

        for i in range(num_targets):

            y_lower_all, y_pred_all, y_upper_all = [], [], []

            lower_model, median_model, upper_model = models[i]

            y_lower_all.append(lower_model.predict(X_test))
            y_pred_all.append(median_model.predict(X_test))
            y_upper_all.append(upper_model.predict(X_test))

            # Stack: [N, num_targets]
            y_lower_all = np.stack(y_lower_all, axis=1)
            y_pred_all = np.stack(y_pred_all, axis=1)
            y_upper_all = np.stack(y_upper_all, axis=1)

            res_df['{}_true'.format(self.args.target[i])] = Y_true[:,i]
            res_df['{}_mean'.format(self.args.target[i])] = y_pred_all
            res_df['{}_95'.format(self.args.target[i])] = y_upper_all
            res_df['{}_5'.format(self.args.target[i])] = y_lower_all

            [mse, rmse, nrmse, mae, mape, rae, r2, corr] = results_evaluation(Y_true[:,i].reshape(-1,1), y_pred_all)
            picp90, piw90 = results_probability_forecast_evaluation(Y_true[:,i].reshape(-1,1), mu=y_pred_all, y_lower=y_lower_all, y_upper=y_upper_all)
            picp90_list.append(picp90)
            piw90_list.append(piw90)

            mse_list.append(mse)
            rmse_list.append(rmse)
            nrmse_list.append(nrmse)
            mae_list.append(mae)
            mape_list.append(mape)
            rae_list.append(rae)
            r2_list.append(r2)
            corr_list.append(corr)
        res_metrics_df = pd.DataFrame(
            columns=['picp90', 'piw90',  'mse', 'rmse', 'nrmse', 'mae', 'mape', 'rae', 'r2', 'corr'],
            index=[i for i in self.args.target])
        res_metrics_df['picp90'] = picp90_list
        res_metrics_df['piw90'] = piw90_list
        res_metrics_df['mse'] = mse_list
        res_metrics_df['rmse'] = rmse_list
        res_metrics_df['nrmse'] = nrmse_list
        res_metrics_df['rae'] = rae_list
        res_metrics_df['mae'] = mae_list
        res_metrics_df['mape'] = mape_list
        res_metrics_df['r2'] = r2_list
        res_metrics_df['corr'] = corr_list
        res_metrics_df.loc['mean'] = res_metrics_df.mean()
        print(res_metrics_df.loc['mean'])
        plt.show()
        path = r'D:\Time-LLM-main\results'
        res_df.to_csv(os.path.join(path, 'pred_res_regression_{}.csv'.format(self.args.data_path[:-4])))
        res_metrics_df.to_csv(os.path.join(path, 'res_metrics_regression_{}.csv'.format(self.args.data_path[:-4])))
        return res_metrics_df

    def res_evaluation_multi_target(self,true,pred,mus, sigmas, trainable_params,path):
        stride = self.args.pred_len
        true = true[::stride,:,:].reshape(-1,len(self.args.target))
        pred = pred[::stride,:,:].reshape(-1,len(self.args.target))
        mus = mus[::stride,:,:].reshape(-1,len(self.args.target))
        sigmas = sigmas[::stride,:,:].reshape(-1,len(self.args.target))
        columns_list = []
        for i in self.args.target:
            columns_list.append('{}_true'.format(i))
            columns_list.append('{}_mean'.format(i))
            columns_list.append('{}_95'.format(i))
            columns_list.append('{}_5'.format(i))
            columns_list.append('{}_85'.format(i))
            columns_list.append('{}_15'.format(i))

        res_df = pd.DataFrame(columns=columns_list)
        nll_list, crps_list, picp90_list, picp70_list, piw90_list, piw70_list,pinaw90_list,pinaw70_list = [], [], [], [], [],[],[],[]
        mse_list, rmse_list, mae_list, r2_list, corr_list,mape_list = [], [], [], [], [], []
        nrmse_list,rae_list = [],[]
        for i in self.args.target:
            y_true,y_pred,mu,sigma,p50,p95,p5,p85,p15,p70,p30 = self._show_plot(i,true[:, self.args.target.index(i)],pred[:, self.args.target.index(i)],mus[:,self.args.target.index(i)],sigmas[:,self.args.target.index(i)],path)

            res_df['{}_true'.format(i)] = true[:, self.args.target.index(i)]
            res_df['{}_mean'.format(i)] = p50
            res_df['{}_95'.format(i)] = p95
            res_df['{}_5'.format(i)] = p5
            res_df['{}_85'.format(i)] = p85
            res_df['{}_15'.format(i)] = p15
            if self.loss_method == 'adaptive' or self.loss_method == 'hybridmu':
                [mse, rmse,nrmse, mae,mape,rae, r2,corr] = results_evaluation(y_true, y_pred)
            else:
                [mse, rmse,nrmse, mae,mape,rae, r2,corr] = results_evaluation(y_true, p50)

            nll, crps, picp90, picp80, picp70, piw90, piw80, piw70, pinaw90, pinaw80, pinaw70 = results_probability_forecast_evaluation(y_true, mu, sigma)
            print('{} nll:{}, crps:{}, mse:{}, rmse:{} mae:{} mape:{} r2:{} corr:{}'.format(i,nll, crps, mse, rmse, mae,mape, r2, corr))

            nll_list.append(nll)
            crps_list.append(crps)
            picp90_list.append(picp90)
            picp70_list.append(picp70)
            piw90_list.append(piw90)
            piw70_list.append(piw70)
            pinaw90_list.append(pinaw90)
            pinaw70_list.append(pinaw70)

            mse_list.append(mse)
            rmse_list.append(rmse)
            nrmse_list.append(nrmse)
            mae_list.append(mae)
            mape_list.append(mape)
            rae_list.append(rae)
            r2_list.append(r2)
            corr_list.append(corr)

        res_metrics_df = pd.DataFrame(columns=['trainable_params','nll','crps','picp90','pinaw90','piw90','picp70','pinaw70','piw70','mse', 'rmse','nrmse', 'mae','mape','rae', 'r2','corr'],
                                      index=[i for i in self.args.target])
        res_metrics_df['trainable_params'] = trainable_params
        res_metrics_df['nll'] = nll_list
        res_metrics_df['crps'] = crps_list
        res_metrics_df['picp90'] = picp90_list
        res_metrics_df['picp70'] = picp70_list
        res_metrics_df['pinaw70'] = pinaw70_list
        res_metrics_df['pinaw90'] = pinaw90_list
        res_metrics_df['piw90'] = piw90_list
        res_metrics_df['piw70'] = piw70_list
        res_metrics_df['mse'] = mse_list
        res_metrics_df['rmse'] = rmse_list
        res_metrics_df['nrmse'] = nrmse_list
        res_metrics_df['rae'] = rae_list
        res_metrics_df['mae'] = mae_list
        res_metrics_df['mape'] = mape_list
        res_metrics_df['r2'] = r2_list
        res_metrics_df['corr'] = corr_list
        res_metrics_df.loc['mean'] = res_metrics_df.mean()
        print(res_metrics_df.loc['mean'])
        plt.show()
        res_df.to_csv(os.path.join(path, 'pred_res_{}.csv'.format(self.args.data_path[:-4])))
        res_metrics_df.to_csv(os.path.join(path, 'res_metrics_df_{}.csv'.format(self.args.data_path[:-4])))
        return res_df, res_metrics_df

    def _show_plot(self,i,y_true,y_pred,mu,sigma,path):
        y_sample = []
        res_df = pd.DataFrame(columns=['true','p50','p95','p5','p85','p15','p70','p30'])
        for _ in tqdm(range(self.args.sample_size)):
            if self.likelihood == 'g':
                y_ = gaussian_sample(torch.tensor(mu), torch.tensor(sigma))
            elif self.likelihood == 'nb':
                y_= negative_binomial_sample(torch.tensor(mu), torch.tensor(sigma))
            else:
                y_ = gaussian_sample(torch.tensor(mu), torch.tensor(sigma))

            y_sample.append(y_.reshape(-1,1))
        y_sample = np.concatenate(y_sample, axis=1)
        p50 = np.quantile(y_sample, 0.5, axis=1)
        p95 = np.quantile(y_sample, 0.95, axis=1)
        p85 = np.quantile(y_sample, 0.85, axis=1)
        p15 = np.quantile(y_sample, 0.15, axis=1)
        p5 = np.quantile(y_sample, 0.05, axis=1)
        p30 = np.quantile(y_sample, 0.3, axis=1)
        p70 = np.quantile(y_sample, 0.7, axis=1)
        res_df['true'] = y_true
        res_df['p50'] = p50
        res_df['p95'] = p95
        res_df['p5'] = p5
        res_df['p85'] = p85
        res_df['p15'] = p15
        res_df['p70'] = p70
        res_df['p30'] = p30
        res_df.to_csv(os.path.join(path, 'if_res_{}_{}.csv'.format(self.args.model_id,i)))

        p50_ = p50[-self.args.pred_len*7:]
        p95_ = p95[-self.args.pred_len*7:]
        p85_ = p85[-self.args.pred_len*7:]
        p15_ = p15[-self.args.pred_len*7:]
        p5_ = p5[-self.args.pred_len*7:]
        x_range = np.arange(self.args.num_train - self.args.pred_len*7, self.args.num_train)
        plt.figure(self.args.target.index(i)+1, figsize=(20, 5))
        plt.plot(x_range, p50_, "r-", label="P50 forecast")
        plt.fill_between(x_range, p5_, p95_, alpha=0.5, color="orange", label="P5-P95 quantile")
        plt.fill_between(x_range, p15_, p85_, alpha=0.5, color="green", label="P15-P85 quantile")

        yplot = y_true[-self.args.pred_len*7:]
        plt.plot(x_range, yplot, "k-", label="True values")
        ymin, ymax = plt.ylim()
        plt.vlines(self.args.num_train - self.args.pred_len*7, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        plt.ylim(ymin, ymax)
        plt.legend(loc="upper left")
        plt.title('Prediction uncertainty')
        plt.xlabel("Periods")
        plt.ylabel("Y")
        plt.savefig(os.path.join(path,'{}.png'.format(i)))
        plt.close()
        return y_true,y_pred,mu,sigma,p50,p95,p5,p85,p15,p70,p30
