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


dim = 3


def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='state0907_20210312')
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
    parser.add_argument('--name', type=str, default="state0907_20210312")
    parser.add_argument('--epoch',type=str,default="00000")
    parser.add_argument('--num_samples', type=int, default="20")




    args=parser.parse_args()
    model_name=args.name

    try:
        os.mkdir('models')
    except:
        pass
    try:
        os.mkdir('output')
    except:
        pass
    try:
        os.mkdir('output/Individual')
    except:
        pass
    try:
        os.mkdir(f'models/Individual')
    except:
        pass

    try:
        os.mkdir(f'output/Individual/{args.name}')
    except:
        pass

    try:
        os.mkdir(f'models/Individual/{args.name}')
    except:
        pass

    #log=SummaryWriter('logs/%s'%model_name)

    # log.add_scalar('eval/mad', 0, 0)
    # log.add_scalar('eval/fad', 0, 0)
    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True


    ## creation of the dataloaders for train and validation

    test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)

    inp = scipy.io.loadmat(os.path.join("output/Individual/{}/det_{}.mat".format(args.dataset_name,args.epoch)))

    nor = scipy.io.loadmat(os.path.join("models/Individual/{}/norm.mat".format(args.dataset_name)))

    mean = torch.from_numpy(nor['mean'])
    std = torch.from_numpy(nor['std'])

    model=individual_TF.IndividualTF(dim, dim+1, dim+1, N=args.layers,
                   d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout,mean=[0,0,0],std=[0,0,0]).to(device)

    model.load_state_dict(torch.load(f'models/Individual/{args.name}/{args.epoch}.pth'))
    model.to(device)


    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    #sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)




    # DETERMINISTIC MODE
    with torch.no_grad():
        model.eval()
        gt=[]
        pr=[]
        inp_=[]
        peds=[]
        frames=[]
        dt=[]
        for id_b,batch in enumerate(test_dl):
            inp_.append(batch['src'])
            gt.append(batch['trg'][:, :, 0:dim])
            frames.append(batch['frames'])
            peds.append(batch['peds'])
            dt.append(batch['dataset'])
            inp = (batch['src'][:, 1:, dim:2*dim].to(device) - mean.to(device)) / std.to(device)
            src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
            start_of_seq = torch.Tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                device)
            dec_inp = start_of_seq

            for i in range(args.preds):
                trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                out = model(inp, dec_inp, src_att, trg_att)
                dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

            preds_tr_b = (dec_inp[:, 1:, 0:dim] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + batch[
                                                                                                                  'src'][
                                                                                                              :, -1:,
                                                                                                              0:dim].cpu().numpy()
            pr.append(preds_tr_b)


        peds=np.concatenate(peds,0)
        frames=np.concatenate(frames,0)
        dt=np.concatenate(dt,0)
        gt=np.concatenate(gt,0)
        dt_names=test_dataset.data['dataset_name']
        pr=np.concatenate(pr,0)
        mad,fad,errs=baselineUtils.distance_metrics(gt,pr)

        #log.add_scalar('eval/DET_mad', mad, epoch)
        #log.add_scalar('eval/DET_fad', fad, epoch)

        scipy.io.savemat(f"output/Individual/{args.name}/MM_deterministic.mat",{'input':inp,'gt':gt,'pr':pr,'peds':peds,'frames':frames,'dt':dt,'dt_names':dt_names})

        print("Determinitic:")
        print("mad: %6.3f"%mad)
        print("fad: %6.3f" % fad)

































if __name__=='__main__':
    main()
