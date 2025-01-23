import os
import json
import pandas as pd

class VisASolver(object):
    CLSNAMES = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        'pcb4', 'pipe_fryum',
    ]

    def __init__(self, root='data/visa'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test']
        self.csv_data = pd.read_csv(f'{root}/split_csv/1cls.csv', header=0)

    def run(self):
        columns = self.csv_data.columns  # [object, split, label, image, mask]
        info = {phase: {} for phase in self.phases}
        anomaly_samples = 0
        normal_samples = 0

        for cls_name in self.CLSNAMES:
            csv_path = os.path.join(self.root, cls_name, 'image_anno.csv')
            anno_data = pd.read_csv(csv_path, header=0)
            cls_data = self.csv_data[self.csv_data[columns[0]] == cls_name]

            for phase in self.phases:
                cls_info = []
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                cls_data_phase.index = list(range(len(cls_data_phase)))

                for idx in range(cls_data_phase.shape[0]):
                    data = cls_data_phase.loc[idx]
                    is_abnormal = data.iloc[2] != 'normal'
                    
                    # 从 anno_data 中提取 specie_name
                    anno_specie_name = anno_data[anno_data['image'] == data.iloc[3]]['label'].values[0]
                    
                    info_img = dict(
                        img_path=data.iloc[3],
                        mask_path=data.iloc[4] if is_abnormal else '',
                        cls_name=cls_name,
                        specie_name=anno_specie_name,  # 从 image_anno.csv 中提取
                        anomaly=1 if is_abnormal else 0,
                    )
                    cls_info.append(info_img)
                    if phase == 'test':
                        if is_abnormal:
                            anomaly_samples += 1
                        else:
                            normal_samples += 1
                info[phase][cls_name] = cls_info

        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)


if __name__ == '__main__':
    runner = VisASolver(root='/mnt/IAD_datasets/visa_anomaly_detection/VisA_20220922')
    runner.run()
