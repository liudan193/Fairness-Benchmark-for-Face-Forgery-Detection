"""
evaluate pretained model.
"""
import os
import numpy as np
import random
import yaml
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from dataset.fairfd import FairFD
from detectors import DETECTOR
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--detector_path', type=str, default='', required=True)
parser.add_argument('--weights_path', type=str, default='', required=True)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])

def prepare_my_testing_data(config, root_path):
    paths = ["data/African", "data/Asian", "data/Caucasian", "data/Indian",
             "FaceSwap/African", "FaceSwap/Asian", "FaceSwap/Caucasian", "FaceSwap/Indian",
             "SimSwap/African", "SimSwap/Asian", "SimSwap/Caucasian", "SimSwap/Indian",
             "FastReen/African", "FastReen/Asian", "FastReen/Caucasian", "FastReen/Indian",
             "Dual_Generator_Face_Reen/African", "Dual_Generator_Face_Reen/Asian", "Dual_Generator_Face_Reen/Caucasian", "Dual_Generator_Face_Reen/Indian",
             "MaskGan/African", "MaskGan/Asian", "MaskGan/Caucasian", "MaskGan/Indian",
             "StyGAN/African", "StyGAN/Asian", "StyGAN/Caucasian", "StyGAN/Indian",
             "SDSwap/African", "SDSwap/Asian", "SDSwap/Caucasian", "SDSwap/Indian",
             "StarGAN/African", "StarGAN/Asian", "StarGAN/Caucasian", "StarGAN/Indian",
             "Face2Diffusion/African", "Face2Diffusion/Asian", "Face2Diffusion/Caucasian", "Face2Diffusion/Indian",
             "FSRT/African", "FSRT/Asian", "FSRT/Caucasian", "FSRT/Indian",
             "DCFace/African", "DCFace/Asian", "DCFace/Caucasian", "DCFace/Indian", ]

    # paths = ["data/African", "FaceSwap/African"]

    test_data_loaders = {}
    for i in range(len(paths)):
        test_set = FairFD(config, os.path.join(root_path, paths[i]))
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
            )
        test_data_loaders[paths[i]] = test_data_loader
    return test_data_loaders

def test_one_dataset(model, data_loader):
    confidences = []
    for i, data_dict in tqdm(enumerate(data_loader)):
        # get data
        data, label, mask, landmark = data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)
        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        predictions['cls'] = F.softmax(predictions['cls'], dim=1)
        confidences.append(predictions['cls'][:, 1].cpu().detach().numpy())
    confidences = np.concatenate(confidences, axis=0)
    print(confidences.shape)
    print(confidences[0:20])
    return confidences

def create_dir(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

def test_epoch(model, test_data_loaders, model_name):
    # set model to eval mode
    model.eval()
    # define test recorder
    metrics_all_datasets = {}
    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        save_name = "./saved_confidences/" + model_name + "/" + key.replace('/', '_') + ".npy"
        # if os.path.exists(save_name):
        #     continue
        create_dir(save_name)
        # compute loss for each dataset
        confidences = test_one_dataset(model, test_data_loaders[key])
        np.save(save_name, confidences)
    print('===> Test Done!')
    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

def main():
    # config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    weights_path = args.weights_path
    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loaders
    rootpath = "../dataset/test"
    test_data_loaders = prepare_my_testing_data(config, rootpath)

    # prepare the model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)

    # start evaluating
    test_epoch(model, test_data_loaders, config['model_name'])
    print("Finished...")


if __name__ == '__main__':
    main()
