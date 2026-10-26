# datasets/split_utils.py
import os
import json

def _class_from_rel_path(rel_path):
    """从 'class_name/img.jpg' 形式的相对路径提取类别名。"""
    return rel_path.replace('\\', '/').split('/')[0]


def load_fixed_split(split_dir, data_root, class_to_idx):
    """读取固定划分，返回 train/test 的图片绝对路径与标签。

    Args:
        split_dir: 含 train.json / test.json 的目录。
        data_root: 数据集根目录（类别子文件夹所在的目录）。
        class_to_idx: {类别名: 索引}，类别名应与数据目录保持一致。

    Returns:
        (train_paths, train_labels, test_paths, test_labels)；
        若 split_dir 无效、文件缺失或解析结果为空，返回 None。
    """
    train_json = os.path.join(split_dir, 'train.json')
    test_json = os.path.join(split_dir, 'test.json')
    if not (os.path.isfile(train_json) and os.path.isfile(test_json)):
        return None

    # 大小写不敏感回退映射，容忍 split 文件与数据目录之间的大小写差异
    lower_to_original = {k.lower(): k for k in class_to_idx}

    def _load(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        paths, labels = [], []
        for rel in data.get('image_paths', []):
            class_name = _class_from_rel_path(rel)
            if class_name not in class_to_idx:
                fallback = lower_to_original.get(class_name.lower())
                if fallback is None:
                    continue
                class_name = fallback
            paths.append(os.path.join(data_root, rel.replace('/', os.sep)))
            labels.append(class_to_idx[class_name])
        return paths, labels

    train_paths, train_labels = _load(train_json)
    test_paths, test_labels = _load(test_json)
    if not train_paths and not test_paths:
        return None
    return train_paths, train_labels, test_paths, test_labels
