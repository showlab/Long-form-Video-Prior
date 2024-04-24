"""
Encode the visual locations into text sequences
@author sierkinhane.github.io
"""
# 2024-04-11 15:14:29
# clean the code
# preprocess the raw json files to our structural prompt
import argparse
import copy
import json
import os
import random

import numpy as np
from interval import Interval
from tqdm import tqdm
from utils import keypoint_idx_93


# ----- kinhane
def get_size(coordinate, type, small=Interval(0, 32 ** 2), medium=Interval(32 ** 2, 96 ** 2, lower_closed=False)):
    if type == 'box':
        coordinate = np.array(coordinate)
        mean_area = np.mean((coordinate[2] - coordinate[0]) * (coordinate[3] - coordinate[1]))
    elif type == 'keypoint' or type == 'mask':
        area_list = []
        for coord in coordinate:
            if type == 'mask':
                coord = np.array(coord).squeeze(1)
            else:
                # delete unannotated key points
                tmp = []
                for kpt in coord:
                    _, _, v = kpt
                    if v != 0:
                        tmp.append(kpt)
                coord = np.array(tmp)
            area = (np.max(coord[:, 0]) - np.min(coord[:, 0])) * (np.max(coord[:, 1]) - np.min(coord[:, 1]))
            area_list.append(area)
        mean_area = np.mean(area_list)
    else:
        raise NotImplementedError

    if mean_area in small:
        return 'small'
    elif mean_area in medium:
        return 'medium'
    else:
        return 'large'


# ----- kinhane

def filter_keypoint(keypoints):
    output = []
    for kp_list in keypoints:
        output_single = []
        for kp in kp_list:
            for name, point in kp.items():
                if np.array(point).sum() > 0:
                    output_single.append({name: point})
        if len(output_single) > 0:
            output.append(output_single)
    return output


def sample_data(data, num_samples):
    # compute the number of samples
    num_epochs = num_samples // len(data)
    if num_samples != len(data):
        num_epochs += 1

    data *= num_epochs
    print(num_epochs, len(data))
    data = random.sample(data, num_samples)

    return data


# balancing number of instances of each category via over-sampling
def balance_data_sampling(data, categories):
    num_instances = {v: 0 for k, v in categories.items()}
    category_seq_list = {v: [] for k, v in categories.items()}

    for i in tqdm(range(len(data))):
        temp = []
        for j in range(len(data[i])):
            for k, v in data[i][j].items():
                num_instances[k] += 1
                if k not in temp:
                    category_seq_list[k].append(data[i])
                temp.append(k)

    num_img_per_cat = {k: len(v) for k, v in category_seq_list.items()}
    num_aug = int(np.median(np.array(list(num_img_per_cat.values()))))  # median
    aug_num_instances = copy.deepcopy(num_instances)
    aug_category_seq_list = {v: [] for k, v in categories.items()}

    # over-sampling at image-level
    for k, v in tqdm(category_seq_list.items()):
        n = num_img_per_cat[k]
        if n == 0:  # fix open images
            continue
        multiplier = round(num_aug / n)
        if multiplier == 0:
            multiplier += 1
        aug_category_seq_list[k] = category_seq_list[k] * multiplier
        aug_num_instances[k] *= multiplier

    aug_data = []
    for k, v in aug_category_seq_list.items():
        aug_data.extend(v)

    return aug_data


instruction_templates = [
    "Create a storyboard.",
    "Develop a storyboard.",
    "Craft a storyboard.",
    "Design a storyboard.",
    "Generate a storyboard.",
    "Could you develop a storyboard?",
    "Could you generate a storyboard?",
    "Create a storyboard titled {}.",
    "Develop a storyboard that encompasses {}.",
    "Create a story board that takes place in {}.",
    "Craft a storyboard consisting of {} key frames.",
    "Design a storyboard that includes {} characters.",
    "Construct a storyboard featuring {} as the main characters.",
    "Generate a storyboard with a summary stating that {}.",
    "Create a storyboard that showcases {} in character expressions.",
    "Could you create a storyboard with a summary stating that {} and featuring {} as the main characters?",
    "Create a storyboard titled {} and develop it with elements of {}.",
    "Craft a storyboard that takes place in {} and showcases {} in character expressions.",
    "Design a storyboard consisting of {} shots and includes {} characters.",
    "Develop a storyboard titled {} that encompasses {} and showcases {} in character expressions.",
    "Could you generate a storyboard that takes place in {} and consists of {} key frames?",
    "Create a storyboard with a summary stating that {} and showcases {} in character expressions.",
    "Craft a storyboard that encompasses {} and features {} as the main characters.",
    "Develop a storyboard titled {} that takes place in {}.",
    "Construct a storyboard with a summary stating that {} and includes {} shots.",
    "Could you develop a storyboard titled {} that encompasses {} and takes place in {}?",
    "Create a storyboard featuring {} as the main characters and consists of {} shots with a summary stating that {}.",
    "Design a storyboard showcasing {} in character expressions that includes {} characters and takes place in {}.",
    "Generate a storyboard with a summary stating that {} and encompasses {}.",
    "Develop a storyboard that takes place in {} and showcases {} in character expressions, featuring {} as the main characters.",
    "Craft a storyboard consisting of {} shots titled {} with a summary stating that {}.",
    "Create a storyboard that takes place in {} and includes {} characters with elements of {}.",
    "Construct a storyboard featuring {} as the main characters and takes place in {} with a summary stating that {}.",
    "Generate a storyboard with a summary stating that {}, and showcasing {} in character expressions with elements of {}.",
    "Develop a storyboard that includes {} and takes place in {}, featuring {} as the main characters with a title of {}."
]

elements = [[], [], [], [], [], [], [], ['title'], ['genre'], ['scene'], ['#shots'], ['#characters'],
            ['main characters'], ['summary'], ['emotion'],
            ['summary', 'main characters'], ['title', 'genre'],
            ['scene', 'emotion'], ['#shots', '#characters'],
            ['title', 'genre', 'emotion'], ['scene', '#shots'],
            ['summary', 'emotion'], ['genre', 'main characters'],
            ['title', 'scene'], ['summary', '#shots'], ['title', 'genre', 'scene'],
            ['main characters', '#shots', 'summary'], ['emotion', '#characters', 'scene'],
            ['summary', 'genre'], ['scene', 'emotion', 'main characters'],
            ['#shots', 'title', 'summary'], ['scene', '#characters', 'genre'],
            ['main characters', 'scene', 'summary'], ['summary', 'emotion', 'genre'],
            ['#characters', 'scene', 'main characters', 'title']
            ]


# per-sequence
def format_kpt_rep(keypoints, bboxes, args=None):
    kpts_with_boxes = []
    for kpts, boxes in zip(keypoints, bboxes):
        if len(boxes) == 0:
            kpts_with_boxes.append([])
            continue

        # per-frame
        a_list = []
        rid = random.randint(1, 100)
        random.seed(rid)
        random.shuffle(boxes)
        random.seed(rid)
        random.shuffle(kpts)
        for k, b in zip(kpts, boxes):
            cn = list(k.keys())[0]
            bn = list(b.keys())[0]
            b = np.array(b[bn])
            # remove negative numbers
            b[b <= 0] = 0
            b[:4] = b[:4] * args.res

            k = np.array(k[cn])
            # remove negative numbers
            k[k <= 0] = 0
            k[:, :2] = k[:, :2] * args.res

            # filter keypoint
            for i in range(k.shape[0]):
                if k[i, -1] < args.kpt_thr:
                    k[i] = [0, 0, 0]

            valid_mask = copy.deepcopy(k)
            valid_mask[valid_mask > 0] = 1

            # add noise
            if args.noise:
                noise = np.random.uniform(-50, 50)
                b[:4] += noise
                k[:, :2] += noise
                k = np.clip(k, 0, 512) * valid_mask
                b = np.clip(b, 0, 512)

            box_str = str(list(np.round(b[:4]).astype(np.int16).reshape(-1))).replace(',', '')
            if np.sum(valid_mask[:, -1]) <= 2:
                a_list.append({cn: box_str})
                continue

            size = get_size(b, 'box')
            # modify keypoint representation
            # 1. long shot   - (small)  : only bbox
            # 2. medium shot - (medium) : only bbox
            # 3. close-up    - (large)  : 93 keypoints

            # import ipdb
            # ipdb.set_trace()

            if args.prefix == 'box':
                a_list.append({cn: box_str})
            else:
                if size == 'small':
                    # a_list.append({'person': box_str})
                    # kpt_str = str(list(k[keypoint_idx_17][:, :2].astype(np.int16).reshape(-1))).replace(',', '')
                    # a_list.append({cn: (kpt_str + box_str).replace('][', ' ')})
                    a_list.append({cn: box_str})
                elif size == 'medium':
                    # kpt_str = str(list(k[keypoint_idx_17][:, :2].astype(np.int16).reshape(-1))).replace(',', '')
                    # a_list.append({cn: (kpt_str + box_str).replace('][', ' ')})
                    a_list.append({cn: box_str})
                else:
                    kpt_str = str(list(np.round(k[keypoint_idx_93][:, :2]).astype(np.int16).reshape(-1))).replace(',','')
                    a_list.append({cn: (kpt_str + box_str).replace('][', ' ')})
        kpts_with_boxes.append(a_list)

    return kpts_with_boxes


# per-sequence
def format_box_rep(bboxes, args=None):
    box_list = []
    for boxes in bboxes:
        if len(boxes) == 0:
            box_list.append([])
            continue
        a_list = []
        random.shuffle(boxes)
        cnt = 0
        for b in boxes:
            if cnt >= args.max_num_objects:
                break
            for k, v in b.items():
                v = np.array(v)
                v[v <= 0] = 0
                v[:4] *= args.res
                # add noise
                if args.noise:
                    noise = np.random.uniform(-50, 50)
                    v[:4] += noise
                    v = np.clip(v, 0, 512)
                if v[-1] >= args.box_thr:
                    a_list.append(
                        {k: str(list((v[:4]).astype(np.int16).reshape(-1))).replace(',', '')})
                    cnt += 1

        box_list.append(a_list)
    return box_list


def cal_iou(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def filter_repeated_bboxes(bboxes_all_cats, bboxes_person, args=None):
    filtered_bboxes = []
    for bbc, bbp in zip(bboxes_all_cats, bboxes_person):
        for bp in bbp:
            if bp in bbc:
                bbc.remove(bp)
        filtered_bboxes.append(bbc)
    # filter those bboxes with large overlapping (same class)
    for bbc in bboxes_all_cats:
        random.shuffle(bbc)
        for bb in bbc:
            bbc_ = copy.deepcopy(bbc)
            for i in range(len(bbc_)):
                # if bb == bbc_[i] or list(bb.keys())[0] != list(bbc_[i].keys())[0]:
                if bb == bbc_[i]:
                    continue
                iou = cal_iou(np.array(list(bb.values())[0]) * args.res, np.array(list(bbc_[i].values())[0]) * args.res)
                if iou > args.max_iou:
                    # two different classes with large overlapping, select according their predicted confidence
                    if list(bb.values())[0][-1] > list(bbc_[i].values())[0][-1]:
                        bbc.remove(bbc_[i])
    return bboxes_all_cats


num_2_char = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten',
    11: 'eleven',
    12: 'twelve',
    13: 'thirteen',
    14: 'fourteen',
    15: 'fifteen',
}

def to_sequences(data, args=None):
    output = []
    sidx = 3 if args.instruct else 2
    for cnt in tqdm(range(len(data))):
        item = data[cnt]
        null_frame_flag = False
        for idx, (obj_box, person_box, keypoint) in enumerate(zip(item['bboxes_object'], item['bboxes_person'], item['keypoints'])):
            if len(obj_box) + len(person_box) + len(keypoint) == 0:
                null_frame_flag = True
        if null_frame_flag:
            print('skip a storyboard that contains null frames!')
            continue

        if isinstance(item['main characters'], list):
            temp_list = []
            for l in item['main characters']:
                temp_list.extend(l)
            temp_list = list(set(temp_list))
            item['main characters'] = ' and '.join(temp_list).lower()
            item['#characters'] = len(temp_list)

        H, W = item['resolution']
        factor = 512 / max(H, W)
        item['resolution'] = [int(H * factor), int(W * factor)]

        item['#shots'] = len(item['keypoints'])
        if args.num_instructions != -1:
            instructions = random.sample(instruction_templates, args.num_instructions)
        else:
            instructions = instruction_templates
        for inst in instructions:
            element = elements[instruction_templates.index(inst)]
            element_values = []
            if len(element) != 0:
                for ele in element:
                    values = item[ele]

                    if ele in ['#shots', '#characters']:
                        # added in 2024-04-08
                        if ele == '#shots':
                            if len(item['keypoints']) <= args.max_frames:
                                values = len(item['keypoints'])
                            else:
                                values = args.max_frames
                        # added at 2024-04-08
                        if random.random() > 0.5:
                            element_values.append(values)
                        else:
                            element_values.append(num_2_char[values])
                    else:
                        element_values.append(values)

            # format keypoint representation
            kpts_with_boxes = format_kpt_rep(item['keypoints'], item['bboxes_person'], args)
            bboxes_all_cats = item['bboxes_object']
            # format bbox representation
            bboxes_all_cats = format_box_rep(bboxes_all_cats, args)

            # format synopses [] * N candidate synopses
            if len(item['synopses']) == 1:
                synopses = []
                for i in range(len(item['synopses'][0])):
                    synopses.append([item['synopses'][0][i]])
            else:
                synopses = []
                for i in range(len(item['synopses'][0])):
                    temp = []
                    for j in range(len(item['synopses'])):
                        idx = random.randint(0, len(item['synopses'][j])) - 1
                        temp.append(item['synopses'][j][idx])
                    synopses.append(temp)

            if len(element_values) != 0:
                instruction = f'{args.prefix}; {inst.format(*element_values)};'
            else:
                instruction = f'{args.prefix}; {inst};'

            # split into n sequences
            # rules:
            # 1. within n shots, not split
            # 2. larger than n shots, split every n shots, conditional on one previous shot
            if len(kpts_with_boxes) <= args.max_frames:
                for i in range(len(synopses)):
                    if len(synopses[i]) != 1:
                        descriptions = ' '.join(synopses[i]).replace(';', '.')
                    else:
                        descriptions = synopses[i][0].replace(';', '.')

                    if random.random() > 0.5:
                        prompt = {
                            "objects": bboxes_all_cats,  # formatted in [box]
                            "main characters": kpts_with_boxes,  # formatted in [keypoint, box]
                        }
                    else:
                        prompt = {
                            "main characters": kpts_with_boxes,  # formatted in [keypoint, box]
                            "objects": bboxes_all_cats,  # formatted in [box]
                        }

                    if args.vis_storyboard:
                        prompt["resolution"] = item['resolution']
                        prompt["movie_id"] = item['movie_id']
                        prompt["key_frames"] = item['key_frames']

                    if args.instruct:
                        output.append(
                            instruction + f' {descriptions}; ' + str(prompt).replace("'[", "'").replace("]'", "'"))
                    else:
                        output.append(
                            f'{args.prefix}; {descriptions}; ' + str(prompt).replace("'[", "'").replace("]'", "'"))

            else:
                for j in range(0, len(kpts_with_boxes), args.max_frames):
                    if (len(kpts_with_boxes) - j) < args.max_frames:
                        j = j - (args.max_frames - (len(kpts_with_boxes) - j))

                    for k in range(len(synopses)):
                        if len(synopses[k]) != 1:
                            descriptions = ' '.join(synopses[k][j: j + args.max_frames]).replace(';', '.')
                        else:
                            descriptions = synopses[k][0].replace(';', '.')

                        if random.random() > 0.5:
                            prompt = {
                                "objects": bboxes_all_cats[j:j + args.max_frames],  # formatted in [box]
                                "main characters": kpts_with_boxes[j:j + args.max_frames],  # formatted in [keypoint, box]
                            }
                        else:
                            prompt = {
                                "main characters": kpts_with_boxes[j:j + args.max_frames],  # formatted in [keypoint, box]
                                "objects": bboxes_all_cats[j:j + args.max_frames],  # formatted in [box]
                            }

                        if args.vis_storyboard:
                            prompt["resolution"] = item['resolution']
                            prompt["movie_id"] = item['movie_id']
                            prompt["key_frames"] = item['key_frames'][j:j + args.max_frames]

                        if args.instruct:
                            output.append(
                                instruction + f' {descriptions}; ' + str(prompt).replace("'[", "'").replace("]'", "'"))
                        else:
                            output.append(
                                f'{args.prefix}; {descriptions}; ' + str(prompt).replace("'[", "'").replace("]'", "'"))

    if args.vis_storyboard:
        from utils import visualize_storyboard
        # not show all storyboards
        for i in tqdm(range(0, len(output), 100)):
            storyboard = eval(output[i].split('; ')[sidx])
            visualize_storyboard(storyboard, i, save_dir=args.vis_save_dir,
                                 data_root=args.data_root)

    return output


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-path", type=str, required=True, help='path to the raw json file')
    parser.add_argument("--output-dir", type=str, default='outputs/txt', help='directory to save the processed txt files')
    parser.add_argument("--vis-save-dir", type=str, default='outputs/debug', help='directory to save visualization')
    parser.add_argument("--data-root", type=str, default=None, help='data root of movie frames')
    parser.add_argument("--prefix", type=str, default='keypoint')
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--num-instructions", type=int, default=-1)
    parser.add_argument("--max-num-objects", type=int, default=8)
    parser.add_argument("--kpt-thr", type=float, default=0.6)
    parser.add_argument("--box-thr", type=float, default=0.1)
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--max-iou", type=float, default=0.15)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--add-only-box", action='store_true')
    parser.add_argument("--save-flag", type=str, default='')
    parser.add_argument("--instruct", action='store_true')
    parser.add_argument("--noise", action='store_false')
    parser.add_argument("--vis-storyboard", action='store_true')

    args = parser.parse_args()
    print('instruct:', args.instruct)
    print('visualization:', args.vis_storyboard)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.vis_save_dir):
        os.makedirs(args.vis_save_dir)

    with open(args.input_path) as f:
        data = json.load(f)

    data_str = to_sequences(data, args)
    if args.add_only_box:
        args.prefix = 'box'
        args.max_frames = 11
        data_str += to_sequences(data, args)

    if args.save_flag != '':
        save_path = args.input_path.split('/')[-1].split('.')[0] + f'_{args.save_flag}.txt'
    else:
        save_path = args.input_path.split('/')[-1].split('.')[0] + '.txt'

    if not args.vis_storyboard:
        with open(os.path.join(args.output_dir, save_path), 'w') as f:
            for l in data_str:
                f.write(l + '\n')


if __name__ == "__main__":
    main()
