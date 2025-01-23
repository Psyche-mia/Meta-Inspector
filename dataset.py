import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os


def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Original_Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]} 

import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class Custom_Dataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, dataset_name=None, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/processed_meta2.json', 'r'))
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id, self.specie_map_class_id = self.generate_class_info(dataset_name)
        self.specie_list = list(self.specie_map_class_id.keys())

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly']
        
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(img_mask) if self.target_transform is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        
        # print(cls_name)
        # print(specie_name)

        return {
            'img': img, 
            'img_mask': img_mask, 
            'cls_name': cls_name, 
            'specie_name': specie_name, 
            'anomaly': anomaly,
            'img_path': os.path.join(self.root, img_path), 
            "cls_id": self.class_name_map_class_id[cls_name],
            "specie_id": self.specie_map_class_id[specie_name]
        }
    
    def generate_class_info(self, dataset_name):
        # obj_list = []  # Replace with actual logic to generate object list if needed
        if dataset_name == 'mvtec':
            obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        elif dataset_name == 'visa':
            obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                        'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        elif dataset_name == 'mpdd':
            obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
        elif dataset_name == 'btad':
            obj_list = ['01', '02', '03']
        elif dataset_name == 'DAGM_KaggleUpload':
            obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
        elif dataset_name == 'SDD':
            obj_list = ['electrical commutators']
        elif dataset_name == 'DTD':
            obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
        elif dataset_name == 'colon':
            obj_list = ['colon']
        elif dataset_name == 'ISBI':
            obj_list = ['skin']
        elif dataset_name == 'Chest':
            obj_list = ['chest']
        elif dataset_name == 'thyroid':
            obj_list = ['thyroid']
        class_name_map_class_id = {cls_name: idx for idx, cls_name in enumerate(self.cls_names)}
        
        specie_set = set()
        for data in self.data_all:
            specie_set.add(data['specie_name'])
        specie_map_class_id = {specie_name: idx for idx, specie_name in enumerate(specie_set)}

        return obj_list, class_name_map_class_id, specie_map_class_id
    
class Custom_Dataset2(data.Dataset):
    def __init__(self, root, json_path, transform=None, target_transform=None, dataset_name=None, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(json_path, 'r'))
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id, self.specie_map_class_id = self.generate_class_info(dataset_name)
        self.specie_list = list(self.specie_map_class_id.keys())

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly']
        
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(img_mask) if self.target_transform is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        
        # print(cls_name)
        # print(specie_name)

        return {
            'img': img, 
            'img_mask': img_mask, 
            'cls_name': cls_name, 
            'specie_name': specie_name, 
            'anomaly': anomaly,
            'img_path': os.path.join(self.root, img_path), 
            "cls_id": self.class_name_map_class_id[cls_name],
            "specie_id": self.specie_map_class_id[specie_name]
        }
    
    def generate_class_info(self, dataset_name):
        # obj_list = []  # Replace with actual logic to generate object list if needed
        if dataset_name == 'mvtec':
            obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        elif dataset_name == 'visa':
            obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                        'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        elif dataset_name == 'mpdd':
            obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
        elif dataset_name == 'btad':
            obj_list = ['01', '02', '03']
        elif dataset_name == 'DAGM_KaggleUpload':
            obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
        elif dataset_name == 'SDD':
            obj_list = ['electrical commutators']
        elif dataset_name == 'DTD':
            obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
        elif dataset_name == 'colon':
            obj_list = ['colon']
        elif dataset_name == 'ISBI':
            obj_list = ['skin']
        elif dataset_name == 'Chest':
            obj_list = ['chest']
        elif dataset_name == 'thyroid':
            obj_list = ['thyroid']
        class_name_map_class_id = {cls_name: idx for idx, cls_name in enumerate(self.cls_names)}
        
        specie_set = set()
        for data in self.data_all:
            specie_set.add(data['specie_name'])
        specie_map_class_id = {specie_name: idx for idx, specie_name in enumerate(specie_set)}

        return obj_list, class_name_map_class_id, specie_map_class_id

        
# class Custom_Dataset(data.Dataset):
#     def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
#         self.root = root
#         self.transform = transform
#         self.target_transform = target_transform
#         self.data_all = []
#         meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
#         meta_info = meta_info[mode]

#         self.cls_names = list(meta_info.keys())
#         for cls_name in self.cls_names:
#             self.data_all.extend(meta_info[cls_name])
#         self.length = len(self.data_all)

#         self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
#         self.specie_map_class_id = {specie: i for i, specie in enumerate(set(item['specie_name'] for item in self.data_all))}

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         data = self.data_all[index]
#         img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly']
#         img = Image.open(os.path.join(self.root, img_path))
#         if anomaly == 0:
#             img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
#         else:
#             if os.path.isdir(os.path.join(self.root, mask_path)):
#                 img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
#             else:
#                 img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
#                 img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
#         # transforms
#         img = self.transform(img) if self.transform is not None else img
#         img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
#         img_mask = [] if img_mask is None else img_mask
#         # Return species information
#         return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'specie_name': specie_name, 'anomaly': anomaly,
#                 'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name],
#                 "specie_id": self.specie_map_class_id[specie_name]}

#########################Test Custom_Dataset###############################
# import AnomalyCLIP_lib
# import torch
# import argparse
# import torch.nn.functional as F
# from prompt_ensemble import AnomalyCLIP_PromptLearner
# from loss import FocalLoss, BinaryDiceLoss
# from utils import normalize
# from logger import get_logger
# from tqdm import tqdm

# import os
# import random
# import numpy as np
# from tabulate import tabulate
# from utils import get_transform
# if __name__ == "__main__":
#     # n_ctx = 12
#     # depth = 9
#     # t_n_ctx= 4
#     # device = "cuda" if torch.cuda.is_available() else "cpu"
#     # AnomalyCLIP_parameters = {"Prompt_length": n_ctx, "learnable_text_embedding_depth": depth, "learnable_text_embedding_length": t_n_ctx}
#     # model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
#     # model.eval()
#     # prompt_learner = Custom_AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
#     parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
#     # paths
#     parser.add_argument("--data_path", type=str, default="/mnt/IAD_datasets/mvtec_anomaly_detection", help="path to test dataset")
#     parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
#     parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
#     # model
#     parser.add_argument("--dataset", type=str, default='mvtec')
#     parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
#     parser.add_argument("--image_size", type=int, default=518, help="image size")
#     parser.add_argument("--depth", type=int, default=9, help="image size")
#     parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
#     parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
#     parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
#     parser.add_argument("--metrics", type=str, default='image-pixel-level')
#     parser.add_argument("--seed", type=int, default=111, help="random seed")
#     parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    
#     args = parser.parse_args()
#     preprocess, target_transform = get_transform(args)
#     data_path = '/mnt/IAD_datasets/mvtec_anomaly_detection'
#     dataset = 'mvtec'
#     # test_data = Custom_Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
#     test_data = Custom_Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
#     test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
#     for idx, items in enumerate(tqdm(test_dataloader)):
#         # image = items['img'].to(device)
#         cls_name = items['cls_name']
#         print(cls_name)
#         cls_id = items['cls_id']
#         gt_mask = items['img_mask']
#         gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
#         results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
#         results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())