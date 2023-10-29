import argparse
import glob, os
import torch
import sys
import time
import torch.nn as nn
import pickle
from torch.autograd import Variable
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from dataloader import * 
from loss import *
import cv2
from model import *
from utils import *
import torch.distributed as dist

def train(model, optimizer, loader, epoch, device, args):
    model.train()
    tic = time.time()

    total_loss = AverageMeter()
    cur_loss = AverageMeter()

    for idx, sample in enumerate(loader):
        img_clips = sample[0]
        gt_sal = sample[1]
        if args.use_sound or args.use_vox:
            audio_feature = sample[2].to(device)
        else:
            audio_feature = torch.zeros(args.batch_size, 1024, 3, 1).to(device)
        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0, 2, 1, 3, 4))
        gt_sal = gt_sal.to(device)
        optimizer.zero_grad()
        pred_sal = model(img_clips, audio_feature)
        loss = loss_func(pred_sal[0], gt_sal, args) + loss_func(pred_sal[1], gt_sal, args) * 0.5 + loss_func(
            pred_sal[2], gt_sal, args) * 0.25
        loss.backward()
        optimizer.step()
        total_loss.update(loss.item())
        cur_loss.update(loss.item())

        if idx % args.log_interval == (args.log_interval - 1):
            print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss.avg, (time.time() - tic) / 60))
            cur_loss.reset()
            sys.stdout.flush()

    print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss.avg))
    sys.stdout.flush()

    return total_loss.avg


def validate(model, loader, epoch, device, args):
    model.eval()
    tic = time.time()
    total_loss = AverageMeter()
    total_cc_loss = AverageMeter()
    total_sim_loss = AverageMeter()
    tic = time.time()
    for idx, sample in enumerate(loader):
        img_clips = sample[0]
        gt_sal = sample[1]
        if args.use_sound or args.use_vox:
            audio_feature = sample[2].to(device)
        else:
            audio_feature = torch.zeros(args.batch_size, 1024, 3, 1).to(device)
        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0, 2, 1, 3, 4))

        pred_sal = model(img_clips, audio_feature)

        gt_sal = gt_sal.squeeze(0).numpy()

        pred_sal = pred_sal.cpu().squeeze(0).numpy()
        pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
        pred_sal = blur(pred_sal).unsqueeze(0).cuda()

        gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

        assert pred_sal.size() == gt_sal.size()

        loss = loss_func(pred_sal, gt_sal, args)
        cc_loss = cc(pred_sal, gt_sal)
        sim_loss = similarity(pred_sal, gt_sal)

        total_loss.update(loss.item())
        total_cc_loss.update(cc_loss.item())
        total_sim_loss.update(sim_loss.item())

    print(
        '[{:2d}, val] avg_loss : {:.5f} cc_loss : {:.5f} sim_loss : {:.5f}, time : {:3f}'.format(epoch, total_loss.avg, total_cc_loss.avg, total_sim_loss.avg, (time.time() - tic) / 60))
    sys.stdout.flush()

    return total_loss.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_epochs',default=40, type=int)
    parser.add_argument('--lr',default=1e-4, type=float)
    parser.add_argument('--kldiv',default=True, type=bool)
    parser.add_argument('--cc',default=False, type=bool)
    parser.add_argument('--nss',default=False, type=bool)
    parser.add_argument('--sim',default=False, type=bool)
    parser.add_argument('--nss_emlnet',default=False, type=bool)
    parser.add_argument('--nss_norm',default=False, type=bool)
    parser.add_argument('--l1',default=False, type=bool)
    parser.add_argument('--lr_sched',default=False, type=bool)
    parser.add_argument('--optim',default="Adam", type=str)

    parser.add_argument('--kldiv_coeff',default=1.0, type=float)
    parser.add_argument('--step_size',default=5, type=int)
    parser.add_argument('--cc_coeff',default=-1.0, type=float)
    parser.add_argument('--sim_coeff',default=-1.0, type=float)
    parser.add_argument('--nss_coeff',default=1.0, type=float)
    parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
    parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
    parser.add_argument('--l1_coeff',default=1.0, type=float)

    parser.add_argument('--batch_size',default=2, type=int)
    parser.add_argument('--log_interval',default=10, type=int)
    parser.add_argument('--no_workers',default=8, type=int)
    parser.add_argument('--model_val_path',default="/path/to/eval.pt", type=str)
    parser.add_argument('--clip_size',default=32, type=int)
    parser.add_argument('--nhead',default=4, type=int)
    parser.add_argument('--num_encoder_layers',default=3, type=int)
    parser.add_argument('--num_decoder_layers',default=3, type=int)
    parser.add_argument('--transformer_in_channel',default=768, type=int)
    parser.add_argument('--train_path_data',default='/path/to/train', type=str)
    parser.add_argument('--val_path_data',default="/path/to/train", type=str)
    parser.add_argument('--decoder_upsample',default=1, type=int)
    parser.add_argument('--frame_no',default="last", type=str)
    parser.add_argument('--load_weight',default="pretrain_weight.pt", type=str)
    parser.add_argument('--num_hier',default=3, type=int)
    parser.add_argument('--dataset',default="myDataset", type=str)
    parser.add_argument('--alternate',default=1, type=int)
    parser.add_argument('--spatial_dim',default=-1, type=int)
    parser.add_argument('--split',default=-1, type=int)
    parser.add_argument('--use_sound',default=True, type=bool)
    parser.add_argument('--use_transformer',default=True, type=bool)
    parser.add_argument('--use_vox',default=False, type=bool)
    args = parser.parse_args()
    print(args)
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    print(world_size, rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    file_weight = './S3D_kinetics400.pt'

    model = VideoAudioSaliencyModel(
            transformer_in_channel=args.transformer_in_channel,
            nhead=args.nhead,
            use_transformer=args.use_transformer,
            num_encoder_layers=args.num_encoder_layers,
            use_upsample=bool(args.decoder_upsample),
            num_hier=args.num_hier,
            num_clips=args.clip_size,
            use_sound = args.use_sound
        )
    np.random.seed(0)
    torch.manual_seed(0)


    if args.dataset == "DHF1KDataset":
        train_dataset = DHF1KDataset(args.train_path_data, args.clip_size, mode="train", alternate=args.alternate)
        val_dataset = DHF1KDataset(args.val_path_data, args.clip_size, mode="val", alternate=args.alternate)

    elif args.dataset=="SoundDataset":
        train_dataset_diem = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='DIEM', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
        val_dataset_diem = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='DIEM', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

        train_dataset_coutrout1 = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='Coutrot_db1', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
        val_dataset_coutrout1 = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='Coutrot_db1', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

        train_dataset_coutrout2 = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='Coutrot_db2', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
        val_dataset_coutrout2 = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='Coutrot_db2', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

        train_dataset_avad = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='AVAD', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
        val_dataset_avad = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='AVAD', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

        train_dataset_etmd = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='ETMD_av', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
        val_dataset_etmd = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='ETMD_av', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

        train_dataset_summe = SoundDatasetLoader(args.clip_size, mode="train", dataset_name='SumMe', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
        val_dataset_summe = SoundDatasetLoader(args.clip_size, mode="test", dataset_name='SumMe', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

        train_dataset = torch.utils.data.ConcatDataset([
                    train_dataset_diem, train_dataset_coutrout1,
                    train_dataset_coutrout2,
                    train_dataset_avad, train_dataset_etmd,
                    train_dataset_summe
            ])

        val_dataset = torch.utils.data.ConcatDataset([
                    val_dataset_diem, val_dataset_coutrout1,
                    val_dataset_coutrout2,
                    val_dataset_avad, val_dataset_etmd,
                    val_dataset_summe
            ])
    elif args.dataset == "Hollywoood" or args.dataset == "UCF":
        train_dataset = Hollywood_UCFDataset(args.train_path_data, args.clip_size, mode="train")
        # print(len(train_dataset))
        val_dataset = Hollywood_UCFDataset(args.val_path_data, args.clip_size, mode="val")

    else:
        train_dataset = MySoundDatasetLoader(args.clip_size, mode="train", dataset_name='myDataset', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
        val_dataset = MySoundDatasetLoader(args.clip_size, mode="test", dataset_name='myDataset', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

    Datasampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.no_workers, sampler=Datasampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)
    print("training iteras:{}", len(train_loader))
    if args.load_weight!="None":
        msg = model.load_state_dict(torch.load(args.load_weight),strict =False)
        print(msg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(rank), device_ids=[rank], find_unused_parameters=True)

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    print(device)
    best_model = None
    for epoch in range(0, args.no_epochs):
        Datasampler.set_epoch(epoch)
        loss = train(model, optimizer, train_loader, epoch, device, args)

        if rank==0:
            with torch.no_grad():
               val_loss = validate(model, val_loader, epoch, device, args)
               if epoch == 0 :
                   val_loss = np.inf
                   best_loss = val_loss
               if val_loss <= best_loss:
                   best_loss = val_loss
                   best_model = model
                   print('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
                   torch.save(model.module.state_dict(), args.model_val_path)
        print()
