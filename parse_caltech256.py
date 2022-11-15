import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
from tqdm import tqdm
import argparse
from pathlib import Path


def main(clip_model_type: str, cuda_num: int):
    ## [cfg]
    phases = ['train', 'validation', 'test']
    device = torch.device(f'cuda:{cuda_num}')
    clip_model_name = clip_model_type
    #avoid / in directory
    clip_model_name = clip_model_name.replace('/', '_')
    out_path = f"./data/{clip_model_name}.pkl"
    if not Path(out_path).parent.exists():
        Path(out_path).parent.mkdir(parents=True)
    ## [cfg]

    ## [load compute save]
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    
    for phase in phases:
        out_path = f"./data/{clip_model_name}_{phase}.pkl"
        full_dir = Path.cwd()/Path(phase)
        classes = [i.name for i in full_dir.iterdir()]

        all_embeddings = []
        all_labels = []
        for cla in tqdm(classes):
            files = [i for i in (full_dir/Path(cla)).iterdir()]
            label = [cla]

            for i in range(len(files)):
                filename = files[i]
                # if Path(filename).is_dir():
                #     print(filename)
                # continue
                # rm -r 056.dog/greg
                image = io.imread(filename)
                image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_embedding = clip_model.encode_image(image).cpu()
                all_embeddings.append(img_embedding)
                all_labels.append(label)
            #     if i == 2:
            #         break
            # break
        
        with open(out_path, 'wb') as f:
            pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "labels": all_labels}, f)
        print(f'{phase} done')

    return 0
    ## [load compute save]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--cuda_num', type=int, default=3, help='0 to 3')
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.cuda_num))
