import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import copy
import argparse
import math
from tqdm import tqdm
from model.Resnet import resnet34
from Dataset import create_con_cla_dataloader,natural_sort
from torch.utils.tensorboard import SummaryWriter

class TrainOptions():
    def __init__(self):
        self.parser=argparse.ArgumentParser(description="Train Priori Distillation of GP-UNIT")
        self.parser.add_argument("--task", type=str, default='Get_classifer',
                                 help="task name")
        self.parser.add_argument("--lr", type=float, default=0.0002,
                                 help="learning rate")
        self.parser.add_argument("--iter", type=int, default=45000,
                                 help="iterations")
        self.parser.add_argument("--batch", type=int, default=16,
                                 help="batch size")
        self.parser.add_argument("--lambda_Cla", type=float, default=1.0,
                                 help="the weight of shape distant loss")
        self.parser.add_argument("--paired_data_root", type=str,
                                 help="the path to the synImageNet291")
        self.parser.add_argument("--save_every", type=int, default=5000,
                                 help="inteval of saving a checkpoint")
        self.parser.add_argument("--save_begin", type=int, default=30000,
                                 help="when to start saving a chekckpoint")
        self.parser.add_argument("--visualize_every",type=int, default=500,
                                 help="interval of saving an intermediate result")
        self.parser.add_argument("--model_path",type=str,default='./checkpoint/',
                                 help="path to the saved models")

        def parse(self):
            self.opt = self.parser.parse_args()
            args = vars(self.opt)
            print('Load options')
            for name, value in sorted(args.items()):
                print('%s %s' % (str(name), str(value)))
            return self.opt

def train(args, dataloader, Classifier, optimizer_Cla,  device='cuda'):
    pbar = tqdm(range(args.iter), initial=0, smoothing=0.01, ncols=120, dynamic_ncols=False)

    Classifier.train()
    iterator = iter(dataloader)
    writer = SummaryWriter(comment='priori_distillation')
    for idx in pbar:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            data = next(iterator)

        imgs,labels=data['img'].to(device),data['label'].to(device)
        pred=Classifier[imgs]
        Lpred=F.cross_entropy(pred,labels)

        loss_dict = {}

        loss_dict['Lpred'] =Lpred

        ae_loss = args.lambda_Cla*Lpred

        Classifier.zero_grad()
        ae_loss.backward()
        Classifier.step()

        message = ''
        for k,v in loss_dict.items():
            v = v.mean().float
            message +='L%s : %.3f' % (k,v)
        pbar.set_description((message))

        if((idx +1) >= args.save_begin and (idx + 1) % args.save_every == 0) or (idx + 1) == args.iter:
            torch.save(
                {
                    "Cla_ema": Classifier.state_dict(),
                    "ae_optim": optimizer_Cla.state_dict(),
                    "idx": idx
                },
                f"%s/%s-%05d.pt" % (args.model_path, args.task, idx+1), #the last comma is dispensable
            )
            if(idx +1) == args.iter:
                torch.save(Classifier.state_dict(),f"%s/Classifier-%05d.pt" % (args.model_path,idx+1))
        writer.add_scalar('Lpred' , Lpred , global_step=idx)
if __name__ == "__main__":

    parser = TrainOptions()
    args = parser.parse()
    print('*' * 98)
    if not os.path.exists("log/%s/" % (args.task)):
        os.makedirs("log/%s/" % (args.task))

    device = 'cuda'
    Classifier=resnet34(num_classes=600).to(device)
    Classifier=nn.DataParallel(Classifier)
    if isinstance(Classifier, nn.DataParallel):
        netAE = Classifier.module

    optimizer_Cla = torch.optim.Adam(Classifier.parameters(), lr=args.lr,betas=(0.9,0.999))

    print('create models successfully!')

    files = os.listdir(args.paired_data_root)
    natural_sort(files)
    dataset_sizes = [600]*len(files)

    #for paired data
    dataloader = create_con_cla_dataloader(args.paired_data_root, files, dataset_sizes,  batchSize=args.batch)

    print('Create dataloaders successfully!')

    train(args, dataloader, Classifier, optimizer_Cla,  device)










