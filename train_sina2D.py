import argparse
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import trange

from models import Siren, weights_init
from ssim import ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sharp Implicit Neural Atlas 2D')
    parser.add_argument('--dataset', type=str, help='Path to Image Data as .pth file in [NxHxW]', required=True)
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train')
    parser.add_argument('--channels', type=int, default=64, help='Number of channels')
    parser.add_argument('--layers', type=int, default=7, help='Number of layers')
    parser.add_argument('--grid_sz', type=int, default=28, help='Grid size')
    parser.add_argument('--random_sampling', action='store_true', help='If VRAM limited use random voxel sampling')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--displacement_lr', type=float, default=1e-3, help='Displacement learning rate')
    parser.add_argument('--lambda_linear', type=int, default=500, help='Lambda linear')
    parser.add_argument('--model_weights_init', type=int, default=20, help='Scale')
    parser.add_argument('--displacement_optimizer', type=str, default='Adam', help='Displacement optimizer')
    parser.add_argument('--displacement_scheduler', type=str, default='LambdaLR', help='Displacement scheduler')
    parser.add_argument('--displacement_warmup', type=int, default=-1, help='Displacement warmup')
    parser.add_argument('--val_epochs', type=int, default=50, help='Validation epochs')
    parser.add_argument('--save-epochs', type=int, default=500, help='Save epochs')
    parser.add_argument('--save_path', type=str, default='results', help='Path to save results')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()



    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        device = torch.device('cpu')



    exp_name = 'SINA_2D_'+time.strftime("%Y%m%d-%H%M")

    args.save_path = os.path.join(args.save_path,exp_name)
    os.makedirs(args.save_path,exist_ok=True)


    with open(f'{args.save_path}/args.txt', 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    args.save_path = os.path.join(args.save_path,exp_name)
    os.makedirs(args.save_path,exist_ok=True)

    with open(f'{args.save_path}/args.txt', 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    imgs = torch.load(args.dataset, map_location=device)
    N, H, W = imgs.shape



    model = Siren(in_features=2,out_features=1, hidden_ch=args.channels,num_layers=args.layers)
    model.apply(partial(weights_init, scale=args.model_weights_init))
    model.to(device)


    #initialize displacement field
    disp = torch.zeros(N,2,args.grid_sz,args.grid_sz)
    disp = disp.to(device)
    disp.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    disp_optimizer = torch.optim.Adam([disp],lr=args.displacement_lr)


    if args.displacement_warmup > 0:
        disp_scheduler_lambda = lambda epoch: 1 if epoch>args.displacement_warmup else epoch/args.displacement_warmup
    else:
        disp_scheduler_lambda = lambda epoch: 1 if epoch>(args.epochs//4) else epoch/(args.epochs//4)  
    disp_scheduler = torch.optim.lr_scheduler.LambdaLR(disp_optimizer, lr_lambda=disp_scheduler_lambda)

    
    mesh = F.affine_grid(torch.eye(2,3).unsqueeze(0).cuda(),(1,1,H,W)).repeat(args.batch_size,1,1,1)


    losses = []


    for epoch in trange(args.epochs):
        model.train()
        minibatches = torch.stack([torch.randperm(N)[i:i+args.batch_size] for i in range(0,N,args.batch_size)][:-1])

        for j in range(minibatches.shape[0]):

            disp_optimizer.zero_grad()
            optimizer.zero_grad()

            imgs_train = imgs[minibatches[j]].cuda().float().unsqueeze(1)

            disp_spline = F.interpolate(F.avg_pool2d(F.avg_pool2d(disp[minibatches[j]],5,stride=1,padding=2),5,stride=1,padding=2),\
                                    size=(H,W),mode='bilinear').permute(0,2,3,1)
            
            out = model((mesh+disp_spline).view(-1,2)).view(-1)
            loss =  1 - ssim(out.view(-1,H,W),imgs_train.view(-1,H,W))
            loss.backward()
            losses.append(loss.item())

            disp_optimizer.step()
            optimizer.step()

        disp_scheduler.step()


        if epoch % args.val_epochs == 0 or epoch == args.epochs-1:
            model.eval()
            with torch.no_grad():
                atlas = model(mesh[:1].view(-1,2)).view(1,1,H,W)
                loss_last_iter = torch.tensor(losses[-minibatches.shape[0]:]).mean()
                print(f'Loss: {loss_last_iter:.4f}')
                f,ax = plt.subplots(1,1)
                ax.imshow(atlas.squeeze().cpu().numpy(),'gray')
                plt.savefig(os.path.join(args.save_path, f'atlas_{epoch}.png'))
                plt.close()
    

        if epoch % args.save_epochs == 0 or epoch == args.epochs-1:
            if epoch > 0:
                torch.save({'model':model.state_dict(),
                            'disp':disp,
                            'optimizer':optimizer.state_dict(),
                            'disp_optimizer':disp_optimizer.state_dict(),
                            'epoch':epoch,
                            'losses':losses},
                            os.path.join(args.save_path, f'model_{epoch}.pth'))
                