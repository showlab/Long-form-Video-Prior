import os
import random

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List, Tuple, Union
from torch import FloatTensor
import torch
from prdc import compute_prdc
from pytorch_fid.fid_score import calculate_frechet_distance
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm
from PIL import Image

Feats = Union[FloatTensor, List[FloatTensor]]
Layout = Tuple[np.ndarray, np.ndarray]

# selected keypoint index in whole body annotation
face_s, face_e = 21, 90
body_s, body_e = 5, 16
foot_s, foot_e = 17, 20
hand_s, hand_e = 17, 20
keypoint_idx_93 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 40,
                   41, 42, 43, 44, 45, 46, 47, 48, 49, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 83, 84, 85, 86,
                   87, 88, 89, 90, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                   111, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]

keypoint_idx_17 = [i for i in range(17)]

# body, foot, hand, face
limb_93 = [[5, 6], [6, 8], [8, 10], [5, 7], [7, 9], [6, 12], [12, 11], [5, 11], [12, 14], [14, 16], [11, 13], [13, 15],
           [16, 20], [16, 21], [15, 17], [15, 18], [129, 130], [130, 131], [131, 132], [125, 126], [126, 127],
           [127, 128], [121, 122], [122, 123], [123, 124], [117, 118], [118, 119], [119, 120], [114, 115], [115, 116],
           [10, 129], [10, 125], [10, 121], [10, 117], [10, 114], [93, 94], [94, 95], [96, 97], [97, 98], [98, 99],
           [100, 101], [101, 102], [102, 103], [104, 105], [105, 106], [106, 107], [108, 109], [109, 110], [110, 111],
           [9, 93], [9, 96], [9, 100], [9, 104], [9, 108]]

limb_17 = [[0, 2], [0, 1], [2, 4], [1, 3], [5, 6], [6, 8], [8, 10], [5, 7], [7, 9], [6, 12], [12, 11], [5, 11],
           [12, 14], [14, 16], [11, 13], [13, 15]]

face_idx = [[23, 25], [25, 27], [27, 29], [29, 31], [31, 33], [33, 35], [35, 37], [37, 39], [40, 41], [41, 42],
            [42, 43], [43, 44], [59, 60], [60, 61], [61, 62], [62, 63], [63, 64], [64, 59], [45, 46], [46, 47],
            [47, 48], [48, 49], [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [70, 65], [83, 84], [84, 85],
            [85, 86], [86, 87], [87, 88], [88, 89], [89, 90], [90, 83]]

coco_classes = \
    ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
     'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
     'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
     'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
     'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
     'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
     'scissors', 'teddy bear', 'hair drier', 'toothbrush']

coco_palette = \
    [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
     (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
     (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
     (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
     (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
     (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
     (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
     (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
     (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
     (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
     (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
     (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
     (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
     (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
     (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
     (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
     (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
     (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
     (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
     (246, 0, 122), (191, 162, 208)]

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)


# Plots one bounding box on image img
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness

    if label is None:
        color = color or [random.randint(0, 255) for _ in range(3)]
    else:
        if color is None:
            color = [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1)  # filled
    cv2.putText(img, label, c1, cv2.FONT_HERSHEY_DUPLEX, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def plot_one_person(fig, ax, x):
    colors = []
    for ci in range(133):
        c = matplotlib.cm.get_cmap('tab20')((ci % 20 + 0.05) / 20)
        colors.append(c)

    for i, (ptr_s, ptr_e) in enumerate(limb_17 if x.shape[0] != 93 else limb_93):
        if x.shape[0] != 93:
            s_, e_ = keypoint_idx_17.index(ptr_s), keypoint_idx_17.index(ptr_e)
        else:
            s_, e_ = keypoint_idx_93.index(ptr_s), keypoint_idx_93.index(ptr_e)
        # if sum(x[s_]) == 0 or sum(x[e_]) == 0:
        if 0 in x[s_] or 0 in x[e_]:
            continue
        ax.plot((x[s_, 0], x[e_, 0]), (x[s_, 1], x[e_, 1]), c=colors[s_],
                linestyle="-", linewidth=1.8, solid_capstyle='round')

    if x.shape[0] == 93:
        for i, (ptr_s, ptr_e) in enumerate(face_idx):
            s_, e_ = keypoint_idx_93.index(ptr_s), keypoint_idx_93.index(ptr_e)
            if 0 in x[s_] or 0 in x[e_]:
                continue
            ax.plot((x[s_, 0], x[e_, 0]), (x[s_, 1], x[e_, 1]), c=colors[s_],
                    linestyle="-", linewidth=1, solid_capstyle='round')
    for p in x:
        if 0 in p:
            continue
        ax.scatter(p[0], p[1], color='white', s=1, zorder=2, marker='o', alpha=1, edgecolors='none')

    return fig, ax


def remove_border(fig, canvas):
    height, width, channels = canvas.shape
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')


better_color = [[108, 104, 60], [53, 117, 126], [77, 45, 201], [123, 61, 144], [254, 155, 64], [25, 20, 141],
                [164, 39, 14], [199, 160, 146], [186, 203, 70], [141, 71, 218]]


# visualize a storyboard
def visualize_storyboard(storyboard, idx, save_dir='debug/', data_root=None):
    H, W = storyboard['resolution']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # plot bboxes
    color_dict = {}

    color_cnt = 0
    for i in range(len(storyboard['main characters'])):
        save_path = os.path.join(save_dir, str(idx))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        keypoints = storyboard['main characters'][i]
        objects = storyboard['objects'][i]
        if storyboard.__contains__('movie_id') and storyboard.__contains__('key_frames') and data_root is not None:
            # movie_id = storyboard['movie_id'][i]
            keyframe = storyboard['key_frames'][i]
            img_path = os.path.join(data_root, keyframe)
            canvas = np.array(Image.open(img_path).convert('RGB'))
            canvas = cv2.resize(canvas, (W, H))
        else:
            canvas = np.zeros((H, W, 3), dtype=np.uint8) + 50
        # canvas = np.zeros((H, W, 3), dtype=np.uint8) + 30
        fig, ax = plt.subplots()
        for box_dict in objects:
            for k, v in box_dict.items():
                v = [int(j) for j in v.split(' ')]
                if k != 'person':
                    if k not in color_dict.keys():
                        if color_cnt < len(better_color):
                            color_dict[k] = better_color[color_cnt]
                            color_cnt += 1
                        else:
                            color_dict[k] = [random.randint(0, 255) for _ in range(3)]
                    canvas = plot_one_box(v, canvas, color=color_dict[k], label=k)
                ax.imshow(canvas)
                remove_border(fig, canvas)

        for kpt_dict in keypoints:
            for k, v in kpt_dict.items():
                v = [int(j) for j in v.split(' ')]
                if k not in color_dict.keys():
                    if color_cnt < len(better_color):
                        color_dict[k] = better_color[color_cnt]
                        color_cnt += 1
                    else:
                        color_dict[k] = [random.randint(0, 255) for _ in range(3)]
                try:
                    if len(v) == 190 or len(v) == 38:
                        box = np.array(v[-4:])
                        kpt = np.array(v[:-4]).reshape(-1, 2)
                        canvas = plot_one_box(box, canvas, color=color_dict[k], label=k)
                        ax.imshow(canvas)
                        fig, ax = plot_one_person(fig, ax, kpt)
                        remove_border(fig, canvas)
                    elif len(v) == 4:
                        box = np.array(v)
                        canvas = plot_one_box(box, canvas, color=color_dict[k], label=k)
                        ax.imshow(canvas)
                        remove_border(fig, canvas)
                    else:
                        raise NotImplementedError
                except Exception as e:
                    print(e)
        plt.savefig(f'{save_path}/{i}.png', dpi=300)
        plt.close()


# visualize a storyboard
def visualize_storyboard_inference(storyboard, idx, save_dir='debug/', resolution=(320, 512)):
    H, W = resolution
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # plot bboxes
    color_dict = {'boy': [244, 139, 141], 'knife': [79, 24, 159], 'Gloria': [167, 49, 24], 'blood': [111, 228, 58]}
    color_cnt = 0
    for i in range(len(storyboard['main characters'])):
        save_path = os.path.join(save_dir, str(idx))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        keypoints = storyboard['main characters'][i]
        objects = storyboard['objects'][i]

        canvas = np.zeros((H, W, 3), dtype=np.uint8) + 50

        fig, ax = plt.subplots()
        for box_dict in objects:
            for k, v in box_dict.items():
                v = [int(j) for j in v.split(' ')]
                if k != 'person':
                    if k not in color_dict.keys():
                        if color_cnt < len(better_color):
                            color_dict[k] = better_color[color_cnt]
                            color_cnt += 1
                        else:
                            color_dict[k] = [random.randint(0, 255) for _ in range(3)]
                    canvas = plot_one_box(v, canvas, color=color_dict[k], label=k)
                ax.imshow(canvas)
                remove_border(fig, canvas)

        for kpt_dict in keypoints:
            for k, v in kpt_dict.items():
                v = [int(j) for j in v.split(' ')]
                if k not in color_dict.keys():
                    if color_cnt < len(better_color):
                        color_dict[k] = better_color[color_cnt]
                        color_cnt += 1
                    else:
                        color_dict[k] = [random.randint(0, 255) for _ in range(3)]
                try:
                    if len(v) == 190 or len(v) == 38:
                        box = np.array(v[-4:])
                        kpt = np.array(v[:-4]).reshape(-1, 2)
                        canvas = plot_one_box(box, canvas, color=color_dict[k], label=k)
                        ax.imshow(canvas)
                        fig, ax = plot_one_person(fig, ax, kpt)
                        remove_border(fig, canvas)
                    elif len(v) == 4:
                        box = np.array(v)
                        canvas = plot_one_box(box, canvas, color=color_dict[k], label=k)
                        ax.imshow(canvas)
                        remove_border(fig, canvas)
                    else:
                        raise NotImplementedError
                except:
                    continue
        plt.savefig(f'{save_path}/{i}.png', dpi=300)
        plt.close()


# eval rougel and bertscore
def eval_score(pred_txt, gt_txt, args=None):
    new_pred_txt = []
    new_gt_txt = []
    for p in pred_txt:
        for gt in gt_txt:
            new_pred_txt.append(p)
            new_gt_txt.append(gt)

    # rouge-l
    rougescore = ROUGEScore()
    score = []
    for i in tqdm(range(len(new_pred_txt))):
        score.append(rougescore(new_pred_txt[i], new_gt_txt[i])['rougeL_fmeasure'])

    rougel = np.array(score).reshape(len(pred_txt), -1).max(axis=-1).mean()

    new_pred_txt = np.array_split(new_pred_txt, int(len(new_pred_txt) / 128))
    new_gt_txt = np.array_split(new_gt_txt, int(len(new_gt_txt) / 128))

    # bertscore
    bertscore = BERTScore(device=args.device, batch_size=args.batch_size, max_length=args.max_length)
    score_list = []
    for i in tqdm(range(len(new_pred_txt))):
        score = bertscore(new_pred_txt[i], new_gt_txt[i])
        score_list.append(score['f1'])

    f1 = np.concatenate(score_list, 0).reshape(len(pred_txt), -1).max(axis=-1).mean()

    return rougel, f1

# eval rougel and bertscore
def eval_score_v2(pred_txt, gt_txt, args=None):

    # rouge-l
    rougescore = ROUGEScore()
    score = []
    for i in tqdm(range(len(pred_txt))):
        score.append(rougescore(pred_txt[i], gt_txt[i])['rougeL_fmeasure'])

    rougel = np.array(score).reshape(-1).mean()

    pred_txt = np.array_split(pred_txt, int(len(pred_txt) / 128))
    gt_txt = np.array_split(gt_txt, int(len(gt_txt) / 128))

    # bertscore
    bertscore = BERTScore(device=args.device, batch_size=args.batch_size, max_length=args.max_length)
    score_list = []
    for i in tqdm(range(len(pred_txt))):
        score = bertscore(pred_txt[i], gt_txt[i])
        score_list.append(score['f1'])
    f1 = np.concatenate(score_list, 0).reshape(-1).mean()

    return rougel, f1

def prepare_tensors(one_frame_box, one_frame_label, args=None):
    one_frame_box = one_frame_box[:args.max_bbox]
    one_frame_label = one_frame_label[:args.max_bbox]
    if len(one_frame_box) == args.max_bbox:
        box_tensor = torch.from_numpy(np.stack(one_frame_box).astype(np.float32)).to(args.device)
        label_tensor = torch.from_numpy(np.array(one_frame_label)).to(args.device)
        padding_mask_tensor = torch.zeros((args.max_bbox), dtype=torch.bool).to(args.device)
    else:
        zeros_pad = np.zeros((args.max_bbox - len(one_frame_box), 4))
        box_tensor = torch.from_numpy(
            np.concatenate([np.stack(one_frame_box), zeros_pad], axis=0).astype(np.float32)).to(args.device)
        label_tensor = torch.from_numpy(
            np.pad(one_frame_label, (0, (args.max_bbox - len(one_frame_box))), mode='constant', constant_values=0)).to(
            args.device)
        padding_mask_tensor = torch.cat([torch.zeros((len(one_frame_box)), dtype=torch.bool),
                                         torch.ones((args.max_bbox - len(one_frame_box)), dtype=torch.bool)]).to(
            args.device)

    return box_tensor, label_tensor, padding_mask_tensor


def __to_numpy_array(feats: Feats) -> np.ndarray:
    if isinstance(feats, list):
        # flatten list of batch-processed features
        feats = [x.detach().cpu().numpy() for x in feats]
    else:
        feats = feats.detach().cpu().numpy()
    return np.concatenate(feats)


def compute_generative_model_scores(
        feats_real: Feats,
        feats_fake: Feats,
) -> Dict[str, float]:
    """
    Compute precision, recall, density, coverage, and FID.
    """
    feats_real = __to_numpy_array(feats_real)
    feats_fake = __to_numpy_array(feats_fake)

    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    results = compute_prdc(
        real_features=feats_real, fake_features=feats_fake, nearest_k=5
    )
    results["fid"] = calculate_frechet_distance(
        mu_real, sigma_real, mu_fake, sigma_fake
    )

    return results
