import pandas as pd

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.tools import results_evaluation, save_config
from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')

class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

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
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)

                if self.args.accelerate:
                    outputs, batch_y = self.accelerator.gather_for_metrics((outputs, batch_y))
                outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting, 'checkpoints')
        if not os.path.exists(path):
            os.makedirs(path)
        save_config(self.args, os.path.join(path, 'configs.pkl'))

        time_now = time.time()

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
        loss_records = {"epoch": [], "train_loss": [], "vali_loss": []}

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
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)

                        outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]
                        if self.args.accelerate:
                            batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:]
                        else:
                            batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)

                    outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]
                    if self.args.accelerate:
                        batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:]
                    else:
                        batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                if self.args.accelerate:
                    self.accelerator.print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.2f}min'.format(speed, left_time / 60))
                else:
                    verbose_interval = (len(train_loader) // 5) if len(train_loader) > 5 else 1
                    if (i + 1) % verbose_interval == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
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
            loss_records["train_loss"].append(train_loss)
            loss_records["vali_loss"].append(vali_loss)

            # test_loss = self.vali(test_data, test_loader, criterion)
            cost_time = round((time.time() - epoch_time) / 60, 2)
            print(" Epoch: {} cost time: {} min".format(epoch + 1, cost_time))
            print("☆☆☆☆☆Train Loss: {0:.7f} Vali Loss: {1:.7f}".format(train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            left_time = 1 + (self.args.patience - early_stopping.counter) * cost_time
            print("  Left time: {} min".format(left_time))

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
        return self.model

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

        preds = []
        trues = []

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
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast)

                if self.args.accelerate:
                    self.accelerator.wait_for_everyone()
                    outputs = self.accelerator.gather_for_metrics(outputs)

                outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:]  # .to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, -self.f_dim:]
                batch_y = batch_y[:, :, -self.f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
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
        print('mae:{}, r2:{}, dtw:{}'.format(mae, r2, dtw))
        f = open(os.path.join('./results', "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, r2:{}, dtw:{}'.format(mae, r2, dtw))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(os.path.join(folder_path, 'metrics_{}_{}.npy'.format(self.args.data,self.args.data_path[:-4])), np.array([mae, mse, rmse, r2, corr]))
        np.save(os.path.join(folder_path, 'pred_{}_{}.npy'.format(self.args.data,self.args.data_path[:-4])), preds)
        np.save(os.path.join(folder_path, 'true_{}_{}.npy'.format(self.args.data,self.args.data_path[:-4])), trues)

        if self.args.features == 'M':
            pred_res,metrics_df = self.res_evaluation_multi_target(trues, preds,trainable_params, folder_path)
        else:
            pred_res,metrics_df = self.res_evaluation(trues,preds,trainable_params, folder_path)
        return pred_res,metrics_df

    def res_evaluation(self,true, pred,trainable_params, path):
        stride = self.args.pred_len
        pred_output = np.squeeze(pred,axis=-1)[::stride, :].reshape(-1, 1)
        true_output = np.squeeze(true,axis=-1)[::stride, :].reshape(-1, 1)

        pred_res = pd.DataFrame({'pred': pred_output.flatten(), 'true': true_output.flatten()})
        pred_res.loc[pred_res['true'] < 1e-3, 'true'] = 0
        pred_res.loc[pred_res['true'] < 1e-3, 'pred'] = 0

        [mse, rmse,nrmse, mae,mape,rae, r2,corr] = results_evaluation(pred_res['true'].values, pred_res['pred'].values)

        pred_res.to_csv(os.path.join(path, 'pred_results_{}_{}.csv'.format(self.args.data,self.args.data_path[:-4])))
        metrics_df = pd.DataFrame({'trainable_params':trainable_params,'mse':mse,'rmse': rmse,'nrmse':nrmse, 'mae': mae, 'mape': mape,'rae':rae,'r2': r2,'corr':corr}, index=[0])
        metrics_df.to_csv(os.path.join(path, 'metrics_results_{}_{}.csv'.format(self.args.data,self.args.data_path[:-4])))

        print('RMSE: {} MAE: {} R2: {}'.format(rmse, mae, r2))
        return pred_res,metrics_df

    def res_evaluation_multi_target(self,true,pred,trainable_params,path):
        stride = self.args.pred_len
        true = true[::stride,:,:].reshape(-1,len(self.args.target))
        pred = pred[::stride,:,:].reshape(-1,len(self.args.target))
        columns_list = []
        for i in self.args.target:
            columns_list.append('{}_true'.format(i))
            columns_list.append('{}_pred'.format(i))
        res_df = pd.DataFrame(columns=columns_list)
        mse_list, rmse_list, mae_list, r2_list, corr_list,mape_list = [], [], [], [], [], []
        nrmse_list,rae_list = [],[]
        for i in self.args.target:
            res_df['{}_pred'.format(i)] = pred[:, self.args.target.index(i)]
            res_df['{}_true'.format(i)] = true[:, self.args.target.index(i)]

            [mse, rmse,nrmse, mae,mape,rae, r2,corr] = results_evaluation(true[:, self.args.target.index(i)], pred[:, self.args.target.index(i)])
            print('{} mse:{}, rmse:{} mae:{} r2:{} corr:{}'.format(i, mse, rmse, mae, r2, corr))
            np.save(os.path.join(path, 'metrics_{}.npy'.format(i)), np.array([mse, rmse, mae, r2, corr]))

            mse_list.append(mse)
            rmse_list.append(rmse)
            nrmse_list.append(nrmse)
            mae_list.append(mae)
            mape_list.append(mape)
            rae_list.append(rae)
            r2_list.append(r2)
            corr_list.append(corr)

        res_metrics_df = pd.DataFrame(columns=['trainable_params','mse', 'rmse','nrmse', 'mae','mape','rae', 'r2','corr'],
                                      index=[i for i in self.args.target])
        res_metrics_df['trainable_params'] = trainable_params
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
        res_df.to_csv(os.path.join(path, 'pred_res_{}.csv'.format(self.args.data_path[:-4])))
        res_metrics_df.to_csv(os.path.join(path, 'res_metrics_df_{}.csv'.format(self.args.data_path[:-4])))
        return res_df,res_metrics_df