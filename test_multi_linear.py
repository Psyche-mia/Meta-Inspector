import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import Custom_AnomalyCLIP_PromptLearner, AnomalyCLIP_PromptLearner
from utils import normalize
from dataset import Original_Dataset, Custom_Dataset
from logger import get_logger
from tqdm import tqdm
import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from visualization import visualizer
from metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, accuracy_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    # Load only a subset of the dataset
    # subset_size = min(args.subset_size, len(test_data))
    # test_data = torch.utils.data.Subset(test_data, range(subset_size))

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    # obj_list = ['bottle', 'cable']  # Calculate results for 'bottle' and 'cable'
    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    specie_results = {'gt': [], 'pred': []}
    specie_metrics = {'accuracy': 0}
    
    for obj in obj_list:
        results[obj] = {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': []}
        metrics[obj] = {'pixel-auroc': 0, 'pixel-aupro': 0, 'image-auroc': 0, 'image-ap': 0}

    prompt_learner = Custom_AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    # print("Generated Prompts:", prompts)
    # print("Tokenized Prompts:", tokenized_prompts)
    # print("Compound Prompts Text:", compound_prompts_text)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=text_features.shape[0]), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # 打印 text_features 的形状
    print("Text features shape:", text_features.shape)

    for idx, items in enumerate(tqdm(test_dataloader)):
        cls_name = items['cls_name'][0]
            
        image = items['img'].to(device)
        specie_name = items['specie_name'][0]
        specie_id = items['specie_id'].to(device)
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]['imgs_masks'].append(gt_mask)
        results[cls_name]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            # print("Text probs shape:", text_probs.shape)
            text_probs = text_probs[:, 0, 1]
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map = anomaly_map.sum(dim=0)
            results[cls_name]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0)
            results[cls_name]['anomaly_maps'].append(anomaly_map)

            specie_logits = model.specie_classifier(image_features)
            specie_pred = torch.argmax(specie_logits, dim=1).cpu().numpy()
            specie_results['gt'].append(specie_id.cpu().numpy())
            specie_results['pred'].append(specie_pred)

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    for obj in obj_list:
        if not results[obj]['imgs_masks'] or not results[obj]['anomaly_maps']:
            continue
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        table_ls.append(table)

    specie_gt = np.concatenate(specie_results['gt'])
    specie_pred = np.concatenate(specie_results['pred'])
    specie_accuracy = accuracy_score(specie_gt, specie_pred)
    specie_metrics['accuracy'] = specie_accuracy
    specie_table_ls = [['specie_accuracy', str(np.round(specie_accuracy * 100, decimals=1))]]

    if args.metrics == 'image-level':
        table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results_table = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))])
        results_table = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results_table = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
    
    specie_results_table = tabulate(specie_table_ls, headers=['metric', 'accuracy'], tablefmt="pipe")
    
    logger.info("\n%s", results_table)
    logger.info("\n%s", specie_results_table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    parser.add_argument("--dataset", type=str, default='visa')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--subset_size", type=int, default=50, help="number of samples to load for testing")

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)

