import os
import yaml
import torch
import random
import numpy as np
from datetime import datetime
from competition_utils import *

def submission_3_20220515(yaml_path, output_json_path):
    ###### can be modified (Only Hyperparameters, which can be modified in demo) ######
    data_config = load_yaml_config(yaml_path)
    model_name = 'yolo12'
    ex_dict = {}
    epochs = 20 # less than 20
    batch_size = 16 # less than 16
    optimizer = 'AdamW' 
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-4

    ###### can be modified (Only Models, which can't be modified in demo) ######
    from ultralytics import YOLO
    
    Experiments_Time = datetime.now().strftime("%y%m%d_%H%M%S")
    ex_dict['Iteration'] = int(yaml_path.split('.yaml')[0][-2:])
    image_size = 640
    output_dir = 'tmp'
    optim_args = {'optimizer': optimizer, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
    # devices = [0]
    # device = torch.device(f"cuda:{devices[0]}") if devices else torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Record experiment settings
    ex_dict['Experiment Time'] = Experiments_Time
    ex_dict['Epochs'] = epochs
    ex_dict['Batch Size'] = batch_size
    ex_dict['Device'] = device
    ex_dict['Optimizer'] = optimizer
    ex_dict['LR'] = optim_args['lr']
    ex_dict['Weight Decay'] = optim_args['weight_decay']
    ex_dict['Momentum'] = optim_args['momentum']  
    ex_dict['Image Size'] = image_size
    ex_dict['Output Dir'] = output_dir

    # Dataset information
    Dataset_Name = yaml_path.split('/')[1]
    ex_dict['Dataset Name'] = Dataset_Name
    ex_dict['Data Config'] = yaml_path
    ex_dict['Number of Classes'] = data_config['nc']
    ex_dict['Class Names'] = data_config['names']

    # Dataset Augmentation
    ex_dict['augment']    = False
    
    # 기본적으로 켜질 augmentation 외에 추가로 0으로 세팅할 항목들
    ex_dict['mosaic']      = 0.0   # Mosaic 합성 확률
    ex_dict['mixup']       = 0.0   # MixUp 합성 확률
    ex_dict['copy_paste']  = 0.0   # Copy‑Paste 합성 확률
    
    ex_dict['hsv_h']       = 0.015 # HSV hue 변화 범위
    ex_dict['hsv_s']       = 0.7   # HSV saturation 변화 범위
    ex_dict['hsv_v']       = 0.4   # HSV value 변화 범위
    
    ex_dict['degrees']     = 10.0  # 회전 각도 범위
    ex_dict['translate']   = 0.1   # 이동 비율
    ex_dict['scale']       = 0.5   # 크기 조정 비율
    
    ex_dict['flipud']      = 0.0   # 수직 뒤집기 확률
    ex_dict['fliplr']      = 0.5   # 수평 뒤집기 확률
    # ----------------------------------------------------------

    control_random_seed(42)

    # Initialize and train model
    model = YOLO(f'ultralytics/cfg/models/12/{model_name}.yaml')
    os.makedirs(output_dir, exist_ok=True)
    ex_dict['Model Name'] = model_name
    ex_dict['Model'] = model
    ex_dict = train_model(ex_dict)

    # Inference and save results
    test_images = get_test_images(data_config)
    results_dict = detect_and_save_bboxes(ex_dict['Model'], test_images)
    save_results_to_file(results_dict, output_json_path)


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


def get_test_images(config):
    test_path = os.path.join(config['path'], config['test'])
    if os.path.isdir(test_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_paths = []
        for root, _, files in os.walk(test_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    elif test_path.endswith('.txt'):
        with open(test_path, 'r') as f:
            return [line.strip() for line in f]
    else:
        return []


def control_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


from tqdm import tqdm

def detect_and_save_bboxes(model, image_paths):
    """
    Run object detection on a list of images and collect bounding boxes.
    Displays a progress bar with tqdm.
    """
    results_dict = {}
    for img_path in tqdm(image_paths, desc="Detecting bounding boxes"):
        results = model(img_path, verbose=False, task='detect')
        img_results = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                # Reorder to [x1, x2, y1, y2]
                img_results.append({
                    'bbox': [x1, x2, y1, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        results_dict[img_path] = img_results
    return results_dict



def save_results_to_file(results_dict, output_path):
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"결과가 {output_path}에 저장되었습니다.")
