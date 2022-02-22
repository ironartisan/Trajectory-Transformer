from torch.utils.data import Dataset
import os, time
import pandas as pd
import numpy as np
import torch
import random
import scipy.spatial
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dim = 3

def create_dataset(dataset_folder,dataset_name,val_size,gt,horizon,delim="\t",train=True,eval=False,verbose=False):
        """
        gt:obs length
        horizon:pred length
        """
        if train==True: #训练集
            datasets_list = os.listdir(os.path.join(dataset_folder,dataset_name, "train"))
            full_dt_folder=os.path.join(dataset_folder,dataset_name, "train")
        if train==False and eval==False: #验证集
            datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, "val"))
            full_dt_folder = os.path.join(dataset_folder, dataset_name, "val")
        if train==False and eval==True: #测试集
            datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, "test"))
            full_dt_folder = os.path.join(dataset_folder, dataset_name, "test")


        datasets_list=datasets_list
        data={}
        data_src=[]
        data_trg=[]
        data_seq_start=[]
        data_frames=[]
        data_dt=[]
        data_peds=[]

        val_src = []
        val_trg = []
        val_seq_start = []
        val_frames = []
        val_dt = []
        val_peds=[]

        if verbose:
            print("start loading dataset")
            print("validation set size -> %i"%(val_size))

        # dt为txt文件
        for i_dt, dt in enumerate(datasets_list):
            if verbose:
                print("%03i / %03i - loading %s"%(i_dt+1,len(datasets_list),dt))
            raw_data = pd.read_csv(os.path.join(full_dt_folder, dt), delimiter=delim,
                                            names=["frame", "ped", "x", "y", "z"],usecols=[0, 1, 2, 3, 4],na_values="?")

            raw_data.sort_values(by=['frame','ped'], inplace=True)

            inp,out,info=get_strided_data_clust(raw_data,gt,horizon,1)

            dt_frames=info['frames']
            dt_seq_start=info['seq_start']
            dt_dataset=np.array([i_dt]).repeat(inp.shape[0])
            dt_peds=info['peds']



            if val_size>0 and inp.shape[0]>val_size*2.5:
                if verbose:
                    print("created validation from %s" % (dt))
                k = random.sample(np.arange(inp.shape[0]).tolist(), val_size)
                val_src.append(inp[k, :, :])
                val_trg.append(out[k, :, :])
                val_seq_start.append(dt_seq_start[k, :, :])
                val_frames.append(dt_frames[k, :])
                val_dt.append(dt_dataset[k])
                val_peds.append(dt_peds[k])
                inp = np.delete(inp, k, 0)
                out = np.delete(out, k, 0)
                dt_frames = np.delete(dt_frames, k, 0)
                dt_seq_start = np.delete(dt_seq_start, k, 0)
                dt_dataset = np.delete(dt_dataset, k, 0)
                dt_peds = np.delete(dt_peds,k,0)
            elif val_size>0:
                if verbose:
                    print("could not create validation from %s, size -> %i" % (dt,inp.shape[0]))

            data_src.append(inp)
            data_trg.append(out)
            data_seq_start.append(dt_seq_start)
            data_frames.append(dt_frames)
            data_dt.append(dt_dataset)
            data_peds.append(dt_peds)





        data['src'] = np.concatenate(data_src, 0)
        data['trg'] = np.concatenate(data_trg, 0)
        data['seq_start'] = np.concatenate(data_seq_start, 0)
        data['frames'] = np.concatenate(data_frames, 0)
        data['dataset'] = np.concatenate(data_dt, 0)
        data['peds'] = np.concatenate(data_peds, 0)
        data['dataset_name'] = datasets_list

        mean= data['src'].mean((0,1))
        std= data['src'].std((0,1))

        if val_size>0:
            data_val={}
            data_val['src']=np.concatenate(val_src,0)
            data_val['trg'] = np.concatenate(val_trg, 0)
            data_val['seq_start'] = np.concatenate(val_seq_start, 0)
            data_val['frames'] = np.concatenate(val_frames, 0)
            data_val['dataset'] = np.concatenate(val_dt, 0)
            data_val['peds'] = np.concatenate(val_peds, 0)

            return IndividualTfDataset(data, "train", mean, std), IndividualTfDataset(data_val, "validation", mean, std)

        return IndividualTfDataset(data, "train", mean, std), None




        return IndividualTfDataset(data,"train",mean,std), IndividualTfDataset(data_val,"validation",mean,std)



class IndividualTfDataset(Dataset):
    def __init__(self,data,name,mean,std):
        super(IndividualTfDataset,self).__init__()

        self.data=data
        self.name=name

        self.mean= mean
        self.std = std

    def __len__(self):
        return self.data['src'].shape[0]


    def __getitem__(self,index):
        return {'src':torch.Tensor(self.data['src'][index]),
                'trg':torch.Tensor(self.data['trg'][index]),
                'frames':self.data['frames'][index],
                'seq_start':self.data['seq_start'][index],
                'dataset':self.data['dataset'][index],
                'peds': self.data['peds'][index],
                }







def create_folders(baseFolder,datasetName):
    try:
        os.mkdir(baseFolder)
    except:
        pass

    try:
        os.mkdir(os.path.join(baseFolder,datasetName))
    except:
        pass



def get_strided_data(dt, gt_size, horizon, step):
    """
    gt:obs length
    horizon:pred length
    step:步长
    """

    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt
    # 所有的行人ID
    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step): # 将gt和horizon的
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:].values)
            ped_ids.append(p)

    frames=np.stack(frame) # list转成numpy，将所有的frame整合
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    inp_no_start = inp_te_np[:,1:,:] - inp_te_np[:, :-1, :]
    inp_std = inp_no_start.std(axis=(0, 1))
    inp_mean = inp_no_start.mean(axis=(0, 1))
    inp_norm=inp_no_start
    #inp_norm = (inp_no_start - inp_mean) / inp_std

    #vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    #inp_norm=np.concatenate((inp_norm,vis),2)

    return inp_norm[:,:gt_size-1],inp_norm[:,gt_size-1:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}


def get_strided_data_2(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    inp_relative_pos= inp_te_np-inp_te_np[:,:1,:]
    inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0],1,dim)),inp_te_np[:,1:,0:dim] - inp_te_np[:, :-1, 0:dim]),1)
    inp_accel = np.concatenate((np.zeros((inp_te_np.shape[0],1,dim)),inp_speed[:,1:,0:dim] - inp_speed[:, :-1, 0:dim]),1)
    #inp_std = inp_no_start.std(axis=(0, 1))
    #inp_mean = inp_no_start.mean(axis=(0, 1))
    #inp_norm= inp_no_start
    #inp_norm = (inp_no_start - inp_mean) / inp_std

    #vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    #inp_norm=np.concatenate((inp_norm,vis),2)
    inp_norm=np.concatenate((inp_te_np,inp_relative_pos,inp_speed,inp_accel),2)
    inp_mean=np.zeros(8)
    inp_std=np.ones(8)

    return inp_norm[:,:gt_size],inp_norm[:,gt_size:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}

def get_strided_data_clust(dt, gt_size, horizon, step):
    """
    dt: 读取的dataset数据
    gt:obs length
    horizon:pred length
    step:步长
    """
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique() # 找出唯一ID
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            # 找出p相等的数据，2：代表去除前两列（frame,id）
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    #inp_relative_pos= inp_te_np-inp_te_np[:,:1,:]
    # inp_te_np shape is [batch, len_obs + len_pred, coordinate],求出相对速度
    inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0],1,dim)),inp_te_np[:,1:,0:dim] - inp_te_np[:, :-1, 0:dim]),1)
    #inp_accel = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_speed[:,1:,0:2] - inp_speed[:, :-1, 0:2]),1)
    #inp_std = inp_no_start.std(axis=(0, 1))
    #inp_mean = inp_no_start.mean(axis=(0, 1))
    #inp_norm= inp_no_start
    #inp_norm = (inp_no_start - inp_mean) / inp_std

    #vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    #inp_norm=np.concatenate((inp_norm,vis),2)
    # 第三个维度累加 inp_norm
    inp_norm=np.concatenate((inp_te_np,inp_speed),2)
    inp_mean=np.zeros(dim*2)
    inp_std=np.ones(dim*2)

    return inp_norm[:,:gt_size],inp_norm[:,gt_size:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}


def distance_metrics(gt,preds):
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
    return errors.mean(),errors[:,-1].mean(),errors


def displacement_error(pred_traj_gt, pred_traj, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """

    loss = pred_traj_gt - pred_traj
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss
    elif mode == 'max':
        return torch.max(loss)


def final_displacement_error(
        pred_pos_gt, pred_pos, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 3). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 3). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def cal_ade(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_gt, pred_traj_fake)
    return ade


def cal_fde(pred_traj_gt, pred_traj_fake):
    fde = final_displacement_error(pred_traj_gt[-1], pred_traj_fake[-1])
    return fde


def l2_loss(pred_traj, pred_traj_gt, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (batch, seq_len, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape ( batch,seq_len, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    batch = pred_traj.size(0)
    loss = (pred_traj_gt - pred_traj)**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / batch
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def show(obs_traj, pred_traj_gt, pred_traj, save_fig=False, fig_name=None):
    """
    obs_traj:(batch, obs_len, dim)
    pred_traj_gt:(batch, gt_len, dim)
    pred_traj:(batch, pred_len, dim)
    """
    pred_traj = np.concatenate((np.expand_dims(obs_traj[:, -1, :], axis=1), pred_traj),axis=1)
    # pred_traj_gt = np.concatenate((np.expand_dims(obs_traj[:, -1, :], axis=1), pred_traj_gt), axis=1)

    fig = plt.figure()
    ax = Axes3D(fig)
    line_width = 1
    marker_size = 3

    ax.plot(obs_traj[0, :, 0], obs_traj[0, :, 1], obs_traj[0, :, 2], color='r', label='input', linewidth=line_width)
    ax.plot(pred_traj_gt[0, :, 0], pred_traj_gt[0, :, 1], pred_traj_gt[0, :, 2], color='g', label='groudtruth',
            linewidth=line_width, marker='D', markersize=marker_size)
    # ax.plot(pred_traj[0, :, 0], pred_traj[0, :, 1], pred_traj[0, :, 2], color='b', label='prediction',
    #         linewidth=line_width, marker='x', markersize=marker_size)

    ax.legend()
    if save_fig:
        if not fig_name:
            fig_name = int(time.time())
        plt.savefig('{}.png'.format(fig_name))
    else:
        plt.show()

def show_traj(obs_traj, save_fig=False, fig_name=None):
    """
    obs_traj:(batch, obs_len, dim)
    """
    # pred_traj = np.concatenate((np.expand_dims(obs_traj[:, -1, :], axis=1), pred_traj),axis=1)
    # pred_traj_gt = np.concatenate((np.expand_dims(obs_traj[:, -1, :], axis=1), pred_traj_gt), axis=1)

    fig = plt.figure()
    ax = Axes3D(fig)
    line_width = 1
    marker_size = 3

    ax.plot(obs_traj[0, :, 0], obs_traj[0, :, 1], obs_traj[0, :, 2], color='r', label='input', linewidth=line_width)
    # ax.plot(pred_traj_gt[0, :, 0], pred_traj_gt[0, :, 1], pred_traj_gt[0, :, 2], color='g', label='groudtruth',
    #        linewidth=line_width, marker='D', markersize=marker_size)
    # ax.plot(pred_traj[0, :, 0], pred_traj[0, :, 1], pred_traj[0, :, 2], color='b', label='prediction',
    #         linewidth=line_width, marker='x', markersize=marker_size)

    ax.legend()
    if save_fig:
        if not fig_name:
            fig_name = int(time.time())
        plt.savefig('{}.png'.format(fig_name))
    else:
        plt.show()