import argparse
import os
import tarfile

import nibabel as nib
import torch
import wget


def wget_data(url, path):
    if not os.path.exists(path):
        wget.download(url, path)
    return path

def main(input_path, output_path):
    # Search subfolders for files ending with 'slice_norm.nii.gt'
    fpaths = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('slice_norm.nii.gz'):
                fpaths.append(os.path.join(root, file))
    fpaths = sorted(fpaths)

    imgs = []
    for fpath in fpaths:
        img = nib.load(fpath).get_fdata()
        imgs.append(torch.tensor(img).permute(2,1,0).squeeze().squeeze().float())
    imgs = torch.stack(imgs)

    # Determine output path
    if output_path is None:
        output_path = input_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save as OASIS_imgs.pth
    torch.save(imgs, os.path.join(output_path, 'OASIS_imgs.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files ending with 'slice_norm.nii.gt' in a directory.")
    parser.add_argument("-o", "--output_path", type=str, help="Output directory path (default: input_path)")

    wget_data('http://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.2d.v1.0.tar', 'data.tar')
    # The data.tar contains many folders, we want to extract them to a folder called 'OASIS'
    with tarfile.open('data.tar') as tar:
        tar.extractall('OASIS')
    
    
    args = parser.parse_args()

    main('OASIS', args.output_path)
    
    # force remove the OASIS folder
    # remove the data.tar file
    os.remove('data.tar')
    import shutil
    shutil.rmtree('OASIS')
    
