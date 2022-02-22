import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from transformer.batch import subsequent_mask
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle

from torch.utils.tensorboard import SummaryWriter
import individual_TF
from baselineUtils import cal_ade, cal_fde, show

dim = 3


class Infer(object):
    def __init__(self, dataset_name, batch_size, emb_size, obs, preds, use_cuda = True, delim = '\t'):

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.obs = obs
        self.preds = preds
        self.delim = delim
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")

    def load_mat(self):
        nor = scipy.io.loadmat(os.path.join("models/Individual/{}/norm.mat".format(args.dataset_name)))
        mean = torch.from_numpy(nor['mean'])
        std = torch.from_numpy(nor['std'])
        return mean, std

    def load_model(self):

        model = individual_TF.IndividualTF(dim, dim + 1, dim + 1, N=6,
                                           d_model=self.emb_size, d_ff=2048, h=8, dropout=0.1,
                                           mean=[0, 0, 0], std=[0, 0, 0]).to(self.device)

        model.load_state_dict(torch.load(f'models/Individual/{args.name}/{args.epoch}.pth'))

        model.to(self.device)
        return model

    def get_test_dl(self):
        test_dataset, _ = baselineUtils.create_dataset('datasets', self.dataset_name, 0, self.obs, self.preds,
                                                       delim=self.delim, train=False, eval=True, verbose=args.verbose)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return test_dl

    def get_one(self):
        test_dl = self.get_test_dl()
        mean, std = self.load_mat()
        model = self.load_model()
        with torch.no_grad():
            for id_b, batch in enumerate(test_dl):
                inp = (batch['src'][:, 1:, dim:2 * dim].to(self.device) - mean.to(self.device)) / std.to(self.device)
                src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(self.device)
                start_of_seq = torch.Tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                    self.device)
                dec_inp = start_of_seq

                for i in range(args.preds):
                    trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(self.device)
                    out = model(inp, dec_inp, src_att, trg_att)
                    dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

                preds_tr_b = (dec_inp[:, 1:, 0:dim] * std.to(self.device) + mean.to(self.device)).cpu().numpy().cumsum(1) \
                             + batch['src'][:, -1:,0:dim].cpu().numpy()

                obs = batch['src'][:, :, dim:2 * dim].cpu().numpy()
                pred_gt, pred_fake = batch['trg'][:, :, 0:dim].cpu().numpy(), torch.from_numpy(preds_tr_b).cpu().numpy()
                return obs, pred_gt, pred_fake


    def check_accuary(self):
        test_dl = self.get_test_dl()
        mean, std = self.load_mat()
        model = self.load_model()
        with torch.no_grad():
            model.eval()
            gt = []
            pr = []
            inp_ = []
            peds = []
            frames = []
            dt = []
            metrics = {}
            disp_error, f_disp_error = [], []
            total_traj = 0
            for id_b, batch in enumerate(test_dl):
                inp_.append(batch['src'])
                gt.append(batch['trg'][:, :, 0:dim])
                frames.append(batch['frames'])
                peds.append(batch['peds'])
                dt.append(batch['dataset'])
                inp = (batch['src'][:, 1:, dim:2 * dim].to(self.device) - mean.to(self.device)) / std.to(self.device)
                src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(self.device)
                start_of_seq = torch.Tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                    self.device)
                dec_inp = start_of_seq

                for i in range(args.preds):
                    trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(self.device)
                    out = model(inp, dec_inp, src_att, trg_att)
                    dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

                preds_tr_b = (dec_inp[:, 1:, 0:dim] * std.to(self.device) + mean.to(self.device)).cpu().numpy().cumsum(1) \
                             + batch['src'][:, -1:, 0:dim].cpu().numpy()
                pr.append(preds_tr_b)
                pred_gt, pred_fake = batch['trg'][:, :, 0:dim].cpu().numpy(), torch.from_numpy(preds_tr_b).cpu().numpy()

                ade = cal_ade(pred_gt, pred_fake)
                fde = cal_fde(pred_gt, pred_fake)
                disp_error.append(ade.item())
                f_disp_error.append(fde.item())

                total_traj += pred_gt.size(0)


            gt = np.concatenate(gt, 0)
            pr = np.concatenate(pr, 0)

            mad, fad, errs = baselineUtils.distance_metrics(gt, pr)

            metrics['ade'] = sum(disp_error) / (total_traj * args.preds)
            metrics['fde'] = sum(f_disp_error) / total_traj
            metrics['mad'] = mad
            metrics['fad'] = fad
            return metrics


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='state0907_20210327_3d')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=8)
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="state0907_20210327_3d")
    parser.add_argument('--epoch',type=str,default="00450")
    parser.add_argument('--num_samples', type=int, default="20")

    args=parser.parse_args()
    model_name=args.name

    infer = Infer(args.dataset_name, args.batch_size, args.emb_size, args.obs, args.preds)
    obs, pred_gt, pred_fake = infer.get_one()

    show(obs, pred_gt, pred_fake)

