import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import Custom_AnomalyCLIP_PromptLearner
from utils import normalize
from dataset import Custom_Dataset
from logger import get_logger
from tqdm import tqdm
import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from sklearn.metrics import accuracy_score
import yaml
from QFormer.models.swin.qformer_simple import QFormer
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_qformer_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def check_for_nan(tensor, name):
    if isinstance(tensor, dict):
        for key, value in tensor.items():
            check_for_nan(value, f"{name}[{key}]")
    elif torch.is_tensor(tensor):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            print(f"{name} values: {tensor}")

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnable_text_embedding_depth": args.depth, "learnable_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Custom_Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    specie_results = {'gt': [], 'pred': []}
    specie_metrics = {'accuracy': 0}

    prompt_learner = Custom_AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    # Load Q-Former model and configuration
    qformer_config = load_qformer_config('QFormer/configs/swin/qformer_tiny_patch4_window7_224.yaml')
    expected_keys = ['embed_dim', 'depths', 'num_heads', 'mlp_ratio', 'qkv_bias', 'qk_scale', 
                     'drop_rate', 'attn_drop_rate', 'drop_path_rate', 'norm_layer', 'use_checkpoint', 'num_classes']
    qformer_params = {k: qformer_config['MODEL'][k] for k in expected_keys if k in qformer_config['MODEL']}
    qformer = QFormer(**qformer_params)
    qformer.load_state_dict(checkpoint["qformer"])
    qformer.to(device)
    qformer.eval()

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    
    print(compound_prompts_text)
    decoded_prompts = [_tokenizer.decode(tokens.tolist()) for tokens in tokenized_prompts]

    # Print the decoded prompts
    for i, prompt in enumerate(decoded_prompts):
        print(f"Prompt {i}: {prompt}")
    # 打印或保存learnable prompt text
    # print("Learnable Prompts:")
    # for i, prompt in enumerate(prompts):
    #     print(f"Prompt {i+1}: {prompt}")

    # # 或者将其保存到文件中
    # with open("learnable_prompts.txt", "w") as f:
    #     for i, prompt in enumerate(prompts):
    #         f.write(f"Prompt {i+1}: {prompt}\n")

    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=text_features.shape[0]), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for idx, items in enumerate(tqdm(test_dataloader)):
        cls_name = items['cls_name'][0]
        specie_name = np.array(items['specie_name'])
        # print('specie_name: ', specie_name)
        if specie_name == '':
            continue

        image = items['img'].to(device)
        specie_id = int(specie_name)

        with torch.no_grad():
            image_features, _ = model.encode_image(image, features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Generate type-attention image features using Q-Former
            type_attention_image_features = qformer.forward_image_features(image_features)
            type_attention_image_features = type_attention_image_features / type_attention_image_features.norm(dim=-1, keepdim=True)

            text_probs = type_attention_image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...] / 0.07

            specie_pred_probs = text_probs[:, 2:7]
            specie_pred = torch.argmax(specie_pred_probs, dim=-1).cpu().numpy()
            specie_pred = int(specie_pred)

            specie_results['gt'].append(specie_id)
            specie_results['pred'].append(specie_pred)
    print(len(specie_results['gt']), len(specie_results['pred']))
    specie_gt = specie_results['gt']
    specie_pred = specie_results['pred']
    specie_accuracy = accuracy_score(specie_gt, specie_pred)
    specie_metrics['accuracy'] = specie_accuracy
    specie_table_ls = [['specie_accuracy', str(np.round(specie_accuracy * 100, decimals=1))]]

    specie_results_table = tabulate(specie_table_ls, headers=['metric', 'accuracy'], tablefmt="pipe")
    logger.info("\n%s", specie_results_table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--subset_size", type=int, default=50, help="number of samples to load for testing")
    
    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
