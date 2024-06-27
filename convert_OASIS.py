import os
import argparse
import torch
import nibabel as nib

def main(input_path, output_path):
    # Search subfolders for files ending with 'slice_norm.nii.gt'
    fpaths = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('slice_norm.nii.gt'):
                fpaths.append(os.path.join(root, file))
    fpaths = sorted(fpaths)

    imgs = []
    for fpath in fpaths:
        img = nib.load(fpath).get_fdata()
        imgs.append(torch.tensor(img).T.float())
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
    parser.add_argument("input_path", type=str, help="Input directory path")
    parser.add_argument("-o", "--output_path", type=str, help="Output directory path (default: input_path)")

    args = parser.parse_args()

    main(args.input_path, args.output_path)
