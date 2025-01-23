import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import Custom_AnomalyCLIP_PromptLearner2, AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset, Custom_Dataset2
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
from QFormer.models.swin.qformer import QFormer
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

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads, output_length):
        super(AttentionPooling, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.output_length = output_length

        # Linear layer to project the downsampled sequence to the desired length
        self.linear_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, embed_dim] => [8, 1370, 768]
        
        # Attention pooling: Query, Key, and Value are all the input sequence
        attn_output, _ = self.attention(x, x, x)  # Shape remains [8, 1370, 768]
        
        # Downsampling using attention scores
        downsampled_output = attn_output[:, :self.output_length, :]  # Select first `output_length` elements
        
        # Optionally apply a linear transformation
        downsampled_output = self.linear_proj(downsampled_output)  # Shape: [8, 77, 768]
        
        return downsampled_output

def train(args):
    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnable_text_embedding_depth": args.depth, "learnable_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()
    
    ########### qformer ###########
    # qformer_config = load_qformer_config('QFormer/configs/swin/qformer_tiny_patch4_window7_224.yaml')
    #     # 过滤掉不需要的配置项
    # expected_keys = ['embed_dim', 'depths', 'num_heads', 'mlp_ratio', 'qkv_bias', 'qk_scale', 
    #                  'drop_rate', 'attn_drop_rate', 'drop_path_rate', 'norm_layer', 'use_checkpoint', 'num_classes']
    # qformer_params = {k: qformer_config['MODEL'][k] for k in expected_keys if k in qformer_config['MODEL']}
    # qformer = QFormer(**qformer_params)
    # # qformer = QFormer(**qformer_config['MODEL'])
    # qformer.to(device)
    # qformer.train()
    ####################################

    train_data = Custom_Dataset2(root=args.train_data_path, json_path=args.json_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    prompt_learner = Custom_AnomalyCLIP_PromptLearner2(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    
    ################ Attention Pooling ################
    attention_pooling_layer = AttentionPooling(embed_dim=768, num_heads=8, output_length=77).to(device)
    
    # projection_layer = nn.Linear(68, 768).to(device)
    # optimizer = torch.optim.Adam(list(prompt_learner.parameters()) + list(model.parameters()) + list(qformer.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    # optimizer = torch.optim.Adam(list(prompt_learner.parameters()) + list(model.parameters()) + list(qformer.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # # losses
    # loss_focal = FocalLoss()
    # loss_dice = BinaryDiceLoss()
    # loss_ce = torch.nn.CrossEntropyLoss()

    # losses
    # optimizer = torch.optim.Adam(list(prompt_learner.parameters()) + list(model.parameters()) + list(attention_pooling_layer.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer = torch.optim.Adam(list(prompt_learner.parameters()) + list(model.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
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
    
    finegrained_categories = ['bent', 'breakages', 'breaks', 'color', 'contamination', 'cracks', 'cut', 'deformations', 
        'dirt', 'discoloration', 'faulty imprint', 'flip', 'fold', 'foreign materials', 'glue', 
        'hole', 'impurities', 'incorrect assembly', 'liquid', 'manipulated', 'misalignment', 
        'misplaced', 'missing parts', 'poke', 'print', 'scratches', 'squeeze', 'stains', 'surface irregularities', 
        'swap', 'tears', 'thread', 'warping', 'wrong texture']


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
            # print(items['img'])
            # print(items['anomaly'])
            image = items['img'].to(device)
            label = items['anomaly'].to(device)
            specie_name = items['specie_name']
            # print(specie_name)
            # 移除 specie_name 中值为 good 的项
            # valid_indices = specie_name != "good"
            valid_indices = [name != "good" for name in specie_name]
            # print("valid_indices: ", valid_indices)
            # specie_name = specie_name[valid_indices]
            specie_name = [name for name, valid in zip(specie_name, valid_indices) if valid]
            specie_name = [name.replace('_', ' ') for name in specie_name]
            encoded_inputs = tokenize(specie_name).to(device)
            # print(encoded_inputs.shape)
            
            
            # 将 specie_name 转换为整数，跳过空字符串
            # specie_index = [int(name) if name != '' else -1 for name in specie_name]
            # specie_index = torch.tensor(specie_index, dtype=torch.long).to(device)
            # print("specie_index: ", specie_index)
            # 创建 one-hot 编码的张量
            # num_classes = 5  # 假设类别数为 5
            # batch_size = len(specie_index)
            # subtype = torch.zeros((batch_size, num_classes), dtype=torch.float).to(device)
            # for i, index in enumerate(specie_index):
            #     if index != -1:  # 跳过空字符串转化的 -1
            #         subtype[i, index] = 1

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
            # print("compound_prompts_text: ", compound_prompts_text)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            # text_features2 = model.encode_text_learn(prompts, tokenized_prompts).float()
            # print("text features shape original:", text_features.shape)

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)



                # specie_name_encoded = model.encode_text_simple(encoded_inputs)  # 形状: (num_specie_classes, text_feature_dim)
                # specie_name_encoded = specie_name_encoded / specie_name_encoded.norm(dim=-1, keepdim=True)  # 归一化
                # print("spcie_name_encoded shape: ", specie_name_encoded.shape)
            
            # 对 specie_name 进行编码
            normalized_texts = []
            # Iterate over each input (of length 77) in encoded_inputs
            
            for i in range(encoded_inputs.size(0)):
                single_text = encoded_inputs[i]  # Get the i-th text input, shape: [77]
                
                # Encode the single text input
                single_text_encoded = model.encode_text_simple(single_text.unsqueeze(0), compound_prompts_text)  # Shape: (1, text_feature_dim)
                
                # Normalize the encoded text
                single_text_encoded = single_text_encoded / single_text_encoded.norm(dim=-1, keepdim=True)
                
                # Append the result to the list
                normalized_texts.append(single_text_encoded.squeeze(0))

            # After processing all inputs, combine them into a single tensor
            specie_name_encoded = torch.stack(normalized_texts)  # Shape: (n, text_feature_dim)


            # if text_features.shape[0] % 3 != 0:
            #     pad_size = 3 - (text_features.shape[0] % 3)
            #     padding = torch.zeros((pad_size, text_features.shape[1]), dtype=text_features.dtype, device=text_features.device)
            #     text_features = torch.cat([text_features, padding], dim=0)
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=text_features.shape[0]), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 通过Q-Former处理image_features生成type-attention image feature
            # type_attention_image_features = qformer.forward_image_features(image_features)
            # type_attention_image_features = type_attention_image_features / type_attention_image_features.norm(dim=-1, keepdim=True)

            # print("type_attention_image_features shape:", type_attention_image_features.shape)
            # print("image_features shape:", image_features.shape)
            # print("text_features shape:", text_features.shape)
            # print("text_features:", text_features)
            # print("image_features shape 2:", image_features.unsqueeze(1).shape)
            # print("text_features shape 2:", text_features.permute(0, 2, 1).shape)
            
            # 计算type-attention text_probs
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            print("text_probs shape:", text_probs.shape)
            text_probs = text_probs[:, 0, ...] / 0.07
            print("text_probs shape:", text_probs.shape)
            print("text_probs:", text_probs)
            # print("text_probs shape:", text_probs.shape)
            
            # type_attention_text_probs = type_attention_image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            # type_attention_text_probs = type_attention_text_probs[:, 0, ...] / 0.07
            # print("Text probs shape:", text_probs[:, 0:2].shape)
            # print("label shape:", label.shape)
            image_loss = F.cross_entropy(text_probs[:, 0:2], label.long().cuda())
            image_loss_list.append(image_loss.item())
            # print("label: ", label.long().cuda())
            # print("text_probs: ", text_probs[:, 0:2])
            
            # print("type_attention_text_probs shape:", type_attention_text_probs.shape)
            # 获取 2:7 范围内的分类概率
            # specie_pred_probs = type_attention_text_probs[:, 2:7]
            # specie_logits = specie_pred_probs.view(-1, num_classes)
            # specie_index = torch.tensor(specie_indices).to(device)

            # print("specie_logits shape:", specie_logits.shape)
            # print("specie_logits:", specie_logits)
            # print("specie_index shape:", specie_index.shape)
            # print("specie_index:", specie_index)


            # print("number of feature maps: ", len(patch_features))
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
            
            # # 对 specie_name 进行编码
            # specie_name_encoded = model.encode_text(specie_name.to(device))  # 形状: (num_specie_classes, text_feature_dim)
            # specie_name_encoded = specie_name_encoded / specie_name_encoded.norm(dim=-1, keepdim=True)  # 归一化
            # print("spcie_name_encoded shape: ", specie_name_encoded.shape)
            
            # 选择最后一个 feature map 进行处理
            # last_feature_map = patch_features[-1]  # 取最后一个 feature map
            # valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.bool)
            # last_feature_map = last_feature_map[valid_indices_tensor]
            # type_attention_image_features = attention_pooling_layer(last_feature_map)
            
            valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.bool)
            type_attention_image_features = image_features[valid_indices_tensor]
            # type_attention_image_features = attention_pooling_layer(last_feature_map)
            
            # 计算type-attention text_probs
            # print("type_attention_image_features shape:", type_attention_image_features.shape)
            # print("specie_name_encoded shape:", specie_name_encoded.shape)
            # print("type_attention_image_features shape 2:", type_attention_image_features.unsqueeze(1).shape)
            # print("specie_name_encoded shape 2:", specie_name_encoded.unsqueeze(0).permute(0, 2, 1).shape)
            type_attention_text_probs = type_attention_image_features @ specie_name_encoded.unsqueeze(0).permute(0, 2, 1)
            type_attention_text_probs = type_attention_text_probs[:, 0, ...] / 0.07
            # print("type_attention_text_probs shape: ", type_attention_text_probs.shape)
            # print("type_attention_text_probs: ", type_attention_text_probs)
            type_attention_text_probs = type_attention_text_probs.squeeze(0)
            # print("type_attention_text_probs shape: ", type_attention_text_probs.shape)
            specie_label = torch.ones_like(type_attention_text_probs, dtype=torch.float).cuda()
            # print("specie_label shape: ", specie_label.shape)
            specie_loss = F.mse_loss(type_attention_text_probs, specie_label)
            # print(specie_loss)
            specie_loss_list.append(specie_loss.item())
            # 使用 Q-Former 处理最后一个 feature map
            # print("last_feature_map shape: ", last_feature_map.shape)
            # type_attention_image_features = qformer(last_feature_map)
            
            # 添加一个投影层，将 `type_attention_image_features` 投影到与 `specie_name_encoded` 相同的维度
            # print(type_attention_image_features.shape, specie_name_encoded.shape)
            
            # projection_layer = nn.Linear(type_attention_image_features.shape[-1], text_features.shape[-1]).to(device)
            # projected_image_features = projection_layer(type_attention_image_features)  # 形状: (batch_size, num_patches, text_feature_dim)

            # 如果需要，你可以对 `projected_image_features` 进行 pooling 操作，将它们缩减到 batch_size * text_feature_dim 的形状
            # 例如使用 mean pooling
            # projected_image_features = torch.mean(projected_image_features, dim=1)  # 形状: (batch_size, text_feature_dim)
            
            # Apply the projection layer to map to the same dimensionality as specie_name_encoded
            # projected_image_features = projection_layer(type_attention_image_features)
            # pr_specie_name = type_attention_image_features / type_attention_image_features.norm(dim=-1, keepdim=True)
            
            # print("pr_specie_name shape before filtering: ", pr_specie_name.shape)
            # print("valid_indices: ", valid_indices)
            # Convert valid_indices to a tensor
            # valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.bool)
            # print("valid_indices_tensor: ", valid_indices_tensor)
            # Filter pr_specie_name 
            # pr_specie_name = pr_specie_name[valid_indices_tensor]
            # print("pr_specie_name shape after filtering: ", pr_specie_name.shape)
            
            # if len(specie_name) > 0:
            #     # specie_loss = loss_ce(specie_name, specie_index)
            #     # specie_loss_list.append(specie_loss.item())
            #     # specie_loss = F.cosine_similarity(projected_image_features.unsqueeze(1), specie_name_encoded.unsqueeze(0), dim=-1)            
            #     specie_loss = 1 - F.cosine_similarity(pr_specie_name, specie_name_encoded, dim=-1).mean()
            #     specie_loss_list.append(specie_loss.item())
            
            # print(loss.requires_grad)  # 应为 True
            # print(image_loss.requires_grad)  # 应为 True
            # print(specie_loss.requires_grad)  # 应为 True
            
            ############ Analyse the gradients of loss
            # Initialize dictionaries to store gradients
            gradients_after_image_loss = {}
            gradients_after_specie_loss = {}

            # Analyze and print gradients for each model or layer
            def print_gradients(model, model_name):
                print(f"Gradients for {model_name}:")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name} - Gradient sum: {param.grad.sum().item()}")
                    else:
                        print(f"{name} - No gradient")
                        
            # Analyze gradients after each backward pass
            def store_gradients(model, gradient_storage):
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradient_storage[name] = param.grad.clone()

            # Analyze and print the gradients affected by specie_loss
            def print_specie_loss_affected_gradients(model, gradients_after_image_loss, gradients_after_specie_loss):
                print("Gradients affected by specie_loss:")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_diff = gradients_after_specie_loss[name] - gradients_after_image_loss.get(name, torch.zeros_like(param.grad))
                        if grad_diff.abs().sum().item() > 0:
                            print(f"{name} - Gradient difference sum after specie_loss: {grad_diff.abs().sum().item()}")

            # Zero out any previously accumulated gradients
            optimizer.zero_grad()

            # Backward pass for image_loss and store the gradients
            image_loss.backward(retain_graph=True)
            store_gradients(model, gradients_after_image_loss)

            # Backward pass for specie_loss and store the gradients
            specie_loss.backward(retain_graph=True)
            store_gradients(model, gradients_after_specie_loss)

            # Backward pass for the general loss
            loss.backward(retain_graph=True)

            # Print gradients for all components
            print_gradients(model, "Main Model")
            print_gradients(prompt_learner, "Prompt Learner")

            # Uncomment if using this layer
            # print_gradients(attention_pooling_layer, "Attention Pooling Layer")

            # Print out the gradients specifically affected by specie_loss
            print_specie_loss_affected_gradients(model, gradients_after_image_loss, gradients_after_specie_loss)

            # Apply the optimizer step after analyzing gradients
            optimizer.step()

                
            #################### Original back propogation code
            # optimizer.zero_grad()
            # (loss + image_loss + specie_loss).backward(retain_graph=True)
            # # no specie loss
            # # (loss + image_loss).backward()
            # optimizer.step()
            # loss_list.append(loss.item())
        
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}, specie_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list), np.mean(specie_loss_list)))

        # if (epoch + 1) % args.save_freq == 0:
        #     ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
        #     torch.save({"prompt_learner": prompt_learner.state_dict(), "model": model.state_dict(),
        #     "attention_pooling_layer": attention_pooling_layer.state_dict()}, ckp_path)
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict(), "model": model.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="/mnt/IAD_datasets/visa_anomaly_detection/VisA_20220922", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    parser.add_argument("--json_path", type=str, default="./data/visa", help="path to json file")
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
