import os
import argparse
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from models import Siren
import matplotlib.pyplot as plt
from tqdm import trange






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sharp Implicit Neural Atlas 2D')
    parser.add_argument('--trained_model', type=str, help='Path to trained model', required=True)
    parser.add_argument('--output_path', type=str, help='Output directory path', default='results')
    parser.add_argument('--save_nifti', action='store_true', help='Save Atlas as .nii.gz')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor for the atlas')
    args = parser.parse_args()


    dataset = args.trained_model.split('/')[-1].split('.')[0]
    print(f'Generating Atlas for {dataset}')
    if dataset == 'AbdomenCTCT':
        H = 192
        W = 160
        D = 256
        in_features = 3
        out_features = 1
        hidden_ch = 256
        num_layers = 10
        dims = 3

    elif dataset == 'OASIS':
        H = 192
        W = 160
        in_features = 2
        out_features = 1
        hidden_ch = 64
        num_layers = 7
        dims = 2

    elif dataset == 'JSRT':
        H = W = 224
        in_features = 2
        out_features = 1
        hidden_ch = 64
        num_layers = 7
        dims = 2
        
    else:
        raise NotImplementedError
    
    
    model_state_dict = torch.load(args.trained_model, map_location='cpu')['model']
    sina = Siren(in_features=in_features,out_features=out_features, hidden_ch=hidden_ch,num_layers=num_layers)
    if dims == 3:
        print('Compiling Model')
        sina = torch.compile(sina)
        _ = sina(torch.zeros(1,3))
    sina.load_state_dict(model_state_dict)
    sina.eval()
    
    if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
    
    if dataset == 'AbdomenCTCT':
        out_H = int(H*args.scale_factor)
        out_W = int(W*args.scale_factor)
        out_D = int(D*args.scale_factor)
        
        

        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,out_H,out_W,out_D), align_corners=False).squeeze()
        atlas = torch.zeros(H,W,D)
        print(f'Generating {dataset} Atlas with grid size: {out_H}x{out_W}x{out_D}')
        with torch.no_grad():
            for i in trange(D):
                atlas[:,:,i] = sina(mesh[:,:,i,:]).squeeze().detach()

        #save slices as .pngs
        f,ax = plt.subplots(3,1)
        ax[0].set_title('SINA: AbdomenCTCT')
        ax[0].imshow(atlas[96,:,:], 'gray')
        ax[1].imshow(atlas[:,80,:], 'gray')
        ax[2].imshow(atlas[:,:,128], 'gray')
        for a in ax:
            a.axis('off')
        plt.savefig(f'{args.output_path}/atlas.png', bbox_inches='tight', pad_inches=0)

        if args.save_nifti:
            atlas = atlas.numpy()
            atlas = nib.Nifti1Image(atlas, np.eye(4))
            nib.save(atlas, f'{args.output_path}/SINA-AbdomenCTCT.nii.gz')
        

    elif dataset in ['JSRT', 'OASIS']:
        out_H = int(H*args.scale_factor)
        out_W = int(W*args.scale_factor)
        mesh = F.affine_grid(torch.eye(2,3).unsqueeze(0),(1,1,out_H,out_W),align_corners=False).repeat(1,1,1,1)
        print(f'Generating {dataset} Atlas with grid size: {out_H}x{out_W}')
        with torch.no_grad():
            atlas = sina(mesh.view(-1,2)).view(1,1,out_H,out_W).detach()

        f,ax = plt.subplots(1,1)
        ax.set_title(f'SINA: {dataset}')
        ax.imshow(atlas.squeeze().cpu().numpy(),'gray')
        ax.axis('off')
        plt.savefig(f'{args.output_path}/SINA-{dataset}.png', bbox_inches='tight')

        if args.save_nifti:
            atlas = atlas.squeeze().cpu().numpy()
            atlas = nib.Nifti1Image(atlas, np.eye(4))
            nib.save(atlas, f'{args.output_path}/SINA-{dataset}.nii.gz')

    print(f'{dataset} Atlas saved at: "{args.output_path}"')


        
