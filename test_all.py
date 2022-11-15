import argparse
import pickle
from train import ClfDataset, ClfModel
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
def test_all(testset, model, args, output_dir, output_prefix):
    
    device = torch.device(f'cuda:{args.cuda_num}') if (torch.cuda.is_available() and args.enable_cuda) else torch.device('cpu')
    batch_size = args.bs

    loss_func = nn.NLLLoss()

    with torch.no_grad():
        model.eval()
        test_data = DataLoader(testset, batch_size, shuffle=False)
        test_loss, test_acc, counter_test = 0, 0, 0

        ## [iter on dataloader]
        for j, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _values, predictions = torch.max(outputs.data, 1)
            correct_prediction_counts = predictions.eq(labels)

            # Convert correct_prediction_counts to float and then compute the mean
            acc = torch.mean(correct_prediction_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc +=acc.item() * len(labels)
            counter_test += len(labels)
        test_acc /= counter_test
        ## [iter on dataloader]

        ## [performance]
        with open(f'data/{args.prefix}_performance.pkl', 'wb') as f:
            pickle.dump(dict(test_loss=test_loss, test_acc=test_acc, counter_test=counter_test), f)
        print(f'performance saved')
        ## [performance]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/train-019.pt', help='path to load model')
    parser.add_argument('--out_dir', type=str, default='./data', help='path to save output')
    parser.add_argument('--prefix', type=str, default='test_all', help='prefix of name of saved output')
    parser.add_argument('--bs', type=int, default=10000)
    parser.add_argument('--cuda_num', type=int, default=2, help='0 to 3')
    parser.add_argument('--enable_cuda', type=bool, default=False)
    args = parser.parse_args()

    data_dir = './data/ViT-B_32_train.pkl'
    testset = ClfDataset(data_dir)
    model = ClfModel(prefix_length=512)
    with open(args.checkpoint, 'rb') as f:
        ckp = torch.load(f)
    model.load_state_dict(ckp)

    test_all(testset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)

    return 0

if __name__ == '__main__':
    main()