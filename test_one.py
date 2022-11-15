import argparse
import pickle
from train import ClfDataset, ClfModel
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import clip
import skimage.io as io
from PIL import Image
import time
def test_one(info, model, args, output_dir, output_prefix):
    imgpath, label = info
    device = torch.device(f'cuda:{args.cuda_num}') if (torch.cuda.is_available() and args.enable_cuda) else torch.device('cpu')
    clip_model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    image = io.imread(imgpath)
    image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        img_embedding = clip_model.encode_image(image).cpu()
        output = model(img_embedding)
        output = torch.exp(output)

    with open(f'data/{args.prefix}_performance.pkl', 'wb') as f:
        pickle.dump(dict(output=output, label=label), f)
    print(f'performance saved')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/train-019.pt', help='path to load model')
    parser.add_argument('--out_dir', type=str, default='./data', help='path to save output')
    parser.add_argument('--prefix', type=str, default='test_one', help='prefix of name of saved output')
    parser.add_argument('--cuda_num', type=int, default=2, help='0 to 3')
    parser.add_argument('--enable_cuda', type=bool, default=False)
    args = parser.parse_args()


    model = ClfModel(prefix_length=512)
    with open(args.checkpoint, 'rb') as f:
        ckp = torch.load(f)
    model.load_state_dict(ckp)

    info = ('data/152_0105.jpg', 151)

    test_one(info, model, args, output_dir=args.out_dir, output_prefix=args.prefix)

    return 0

if __name__ == '__main__':
    print(time.time())
    main()
    print(time.time())
    