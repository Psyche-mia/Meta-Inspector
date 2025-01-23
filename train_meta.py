import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import Custom_AnomalyCLIP_PromptLearner, AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset, Custom_Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from sklearn.preprocessing import LabelEncoder
from utils import get_transform
from torch import nn
import yaml
from prompt_ensemble import tokenize
# from AnomalyCLIP_lib.AnomalyCLIP import QFormer, load_qformer_model

# 从Q-Former文件夹中导入Q-Former模型及相关配置
from QFormer.models.swin.qformer_simple import QFormer
# from QFormer.configs.swin.qformer_tiny_patch4_window7_224 import get_config

# 配置Q-Former模型参数
def load_qformer_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnable_text_embedding_depth": args.depth, "learnable_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()
    
    ########### qformer ###########
    qformer_config = load_qformer_config('QFormer/configs/swin/qformer_tiny_patch4_window7_224.yaml')
        # 过滤掉不需要的配置项
    expected_keys = ['embed_dim', 'depths', 'num_heads', 'mlp_ratio', 'qkv_bias', 'qk_scale', 
                     'drop_rate', 'attn_drop_rate', 'drop_path_rate', 'norm_layer', 'use_checkpoint', 'num_classes']
    qformer_params = {k: qformer_config['MODEL'][k] for k in expected_keys if k in qformer_config['MODEL']}
    qformer = QFormer(**qformer_params)
    # qformer = QFormer(**qformer_config['MODEL'])
    qformer.to(device)
    qformer.train()
    ####################################

    train_data = Custom_Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    prompt_learner = Custom_AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    optimizer = torch.optim.Adam(list(prompt_learner.parameters()) + list(model.parameters()) + list(qformer.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_ce = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(list(prompt_learner.parameters()) + list(model.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_ce = torch.nn.CrossEntropyLoss()

    # Define categories
    categories = {
        "Physical Damage": ["Cracks", "Scratches", "Breaks", "Tears", "Breakages", "manipulated", "scratch"],
        "Contamination": ["Contamination", "Stains", "Dirt", "Foreign Materials", "Impurities", "Glue", "Liquid"],
        "Morphological Anomalies": ["Deformations", "Warping", "Poke", "Thread", "Cut", "Bent", "Squeeze"],
        "Surface Defects": ["Surface Irregularities", "Discoloration", "Wrong Texture", "Faulty Imprint", "Print", "Color", "hole"],
        "Manufacturing Defects": ["Incorrect Assembly", "Misalignment", "Swap", "Misplaced", "Missing Parts", "fold", "flip"],
    }

    category_labels = {category: idx for idx, category in enumerate(categories.keys())}

    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []
        specie_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly'].to(device)
            specie_name = items['specie_name']
            
            # 将 specie_name 转换为整数，跳过空字符串
            specie_index = [int(name) if name != '' else -1 for name in specie_name]
            specie_index = torch.tensor(specie_index, dtype=torch.long).to(device)
            # print("specie_index: ", specie_index)
            # 创建 one-hot 编码的张量
            num_classes = 5  # 假设类别数为 5
            # batch_size = len(specie_index)
            # subtype = torch.zeros((batch_size, num_classes), dtype=torch.float).to(device)
            # for i, index in enumerate(specie_index):
            #     if index != -1:  # 跳过空字符串转化的 -1
            #         subtype[i, index] = 1

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            # print(text_features.shape)

            # if text_features.shape[0] % 3 != 0:
            #     pad_size = 3 - (text_features.shape[0] % 3)
            #     padding = torch.zeros((pad_size, text_features.shape[1]), dtype=text_features.dtype, device=text_features.device)
            #     text_features = torch.cat([text_features, padding], dim=0)
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=text_features.shape[0]), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 通过Q-Former处理image_features生成type-attention image feature
            type_attention_image_features = qformer.forward_image_features(image_features)
            type_attention_image_features = type_attention_image_features / type_attention_image_features.norm(dim=-1, keepdim=True)

            # print("type_attention_image_features shape:", type_attention_image_features.shape)
            # print("image_features shape:", image_features.shape)
            
            # 计算type-attention text_probs
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...] / 0.07
            # print("text_probs shape:", text_probs.shape)
            
            type_attention_text_probs = type_attention_image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            type_attention_text_probs = type_attention_text_probs[:, 0, ...] / 0.07
            # print("Text probs shape:", text_probs[:, 0:2].shape)
            # print("label shape:", label.shape)
            image_loss = F.cross_entropy(text_probs[:, 0:2], label.long().cuda())
            image_loss_list.append(image_loss.item())
            
            # print("type_attention_text_probs shape:", type_attention_text_probs.shape)
            # 获取 2:7 范围内的分类概率
            specie_pred_probs = type_attention_text_probs[:, 2:7]
            specie_logits = specie_pred_probs.view(-1, num_classes)
            # specie_index = torch.tensor(specie_indices).to(device)
            
            # 移除 specie_index 中值为 -1 的项
            valid_indices = specie_index != -1
            specie_logits = specie_logits[valid_indices]
            specie_index = specie_index[valid_indices]

            # print("specie_logits shape:", specie_logits.shape)
            # print("specie_logits:", specie_logits)
            # print("specie_index shape:", specie_index.shape)
            # print("specie_index:", specie_index)

            if specie_logits.size(0) > 0:
                specie_loss = loss_ce(specie_logits, specie_index)
                specie_loss_list.append(specie_loss.item())

            similarity_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1 - gt)
                
            # print("loss: ", loss)
            print(loss.requires_grad)  # 应为 True
            print(image_loss.requires_grad)  # 应为 True
            print(specie_loss.requires_grad)  # 应为 True

            optimizer.zero_grad()
            (loss + image_loss + specie_loss).backward(retain_graph=True)
            # no specie loss
            # (loss + image_loss).backward()
            optimizer.step()
            loss_list.append(loss.item())
        
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}, specie_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list), np.mean(specie_loss_list)))

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict(), "model": model.state_dict(), "qformer": qformer.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="/mnt/IAD_datasets/visa_anomaly_detection/VisA_20220922", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')

    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=1, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
