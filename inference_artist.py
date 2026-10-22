"""
画师风格模型推理脚本
支持两种模式：
1. 聚类模式：提取特征向量用于聚类
2. 分类模式：直接输出分类结果
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from model_loading import (
    FEATURE_OUTPUTS, load_model_bundle, load_checkpoint_state, normalize_state_dict_keys, load_class_mapping,
)
from backend_lsnet.analysis import extract_tensor_batch, cache_bytes


def get_args_parser():
    parser = argparse.ArgumentParser('Artist Style Inference', add_help=False)

    # 模型参数
    parser.add_argument('--model', default=None, type=str,
                        help='Model architecture')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--num-classes', default=None, type=int,
                        help='Number of classes. If omitted, will try to infer from checkpoint or CSV mapping.')
    parser.add_argument('--feature-dim', default=None, type=int,
                        help='Feature dimension')
    parser.add_argument('--input-size', default=None, type=int,
                        help='Input image size')

    # 推理模式
    parser.add_argument('--mode', default='auto', type=str,
                        choices=['auto', 'classify', 'cluster', 'both'],
                        help='Inference mode: classify (with head), cluster (features only), or both')
    parser.add_argument('--output-type', choices=FEATURE_OUTPUTS, default='default')
    parser.add_argument('--layers', default='-1', help='Intermediate layer indices, comma separated')
    parser.add_argument('--no-intermediate-norm', action='store_true')

    # 输入输出
    parser.add_argument('--input', required=True, type=str,
                        help='Input image path or directory')
    parser.add_argument('--output', default='./output/inference', type=str,
                        help='Output directory')
    parser.add_argument('--class-csv', default=None, type=str,
                        help='Path to class mapping CSV exported during training')

    # 其他参数
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size for batch inference')
    parser.add_argument('--top-k', default=5, type=int,
                        help='Number of top predictions to show (default: 5)')
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='Probability threshold to filter predictions (default: 0.0)')

    return parser


def resolved_mode(model, mode):
    if mode not in ('auto', 'classify', 'cluster', 'both'):
        raise ValueError(f'Unknown inference mode: {mode}')
    if mode == 'auto':
        return 'classify' if model.has_classifier else 'cluster'
    if mode in ('classify', 'both') and not model.has_classifier:
        raise ValueError('Model has no classification head; use --mode auto or --mode cluster')
    return mode


def load_model(args, state_dict=None):
    bundle = load_model_bundle(checkpoint=args.checkpoint, device=args.device, model_name=args.model,
                               class_csv=args.class_csv, input_size=args.input_size)
    args.mode = resolved_mode(bundle['model'], args.mode)
    return bundle['model']


def preprocess_image(image_path, transform):
    """预处理单张图像"""
    with Image.open(image_path) as image:
        tensor = transform(image)
    return tensor.unsqueeze(0)


def classify_image(model, image_tensor, device, class_mapping: Optional[Dict[int, str]] = None, top_k=5, threshold=0.0):
    """对图像进行分类"""
    if not model.has_classifier:
        raise ValueError('This checkpoint has no classification head')
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        # 使用分类头
        logits = model(image_tensor, return_features=False)

        # 计算概率
        probs = F.softmax(logits.float(), dim=-1)

        # Top-K结果
        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)

        results = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            if prob >= threshold:
                class_name = class_mapping.get(int(idx), f"Class {idx}") if class_mapping else f"Class {idx}"
                results.append({
                    'class_id': int(idx),
                    'class_name': class_name,
                    'probability': float(prob)
                })

        # 如果过滤后结果少于 top_k，保持原样；否则取前 top_k
        if len(results) > top_k:
            results = results[:top_k]

        return results


def extract_features(model, image_tensor, device, output_type='default', layers='-1', intermediate_norm=True):
    """提取特征向量用于聚类"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        # 不使用分类头，直接返回特征
        features = extract_tensor_batch(model, image_tensor, output_type, layers, intermediate_norm)
        return features.float().cpu().numpy()


def process_single_image(args, model, transform, class_mapping: Optional[Dict[int, str]] = None):
    """处理单张图像"""
    image_path = Path(args.input)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    print(f"\nProcessing: {image_path.name}")

    # 预处理
    image_tensor = preprocess_image(image_path, transform)

    results = {}

    # 分类模式
    if args.mode in ['classify', 'both']:
        print("\n[Classification Results]")
        classification = classify_image(model, image_tensor, args.device, class_mapping, args.top_k, args.threshold)
        results['classification'] = classification

        for i, result in enumerate(classification, 1):
            print(f"{i}. {result['class_name']}: {result['probability']:.4f}")

    # 聚类模式（提取特征）
    if args.mode in ['cluster', 'both']:
        print("\n[Feature Extraction]")
        features = extract_features(model, image_tensor, args.device, args.output_type, args.layers, not args.no_intermediate_norm)
        results['features'] = features[0].tolist()
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature vector (first 10 dims): {features[0][:10]}")

    return results


def process_directory(args, model, transform, class_mapping: Optional[Dict[int, str]] = None):
    """批量处理目录中的图像"""
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: Directory not found: {input_dir}")
        return

    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = sorted(p for p in input_dir.glob('**/*') if p.suffix.lower() in image_extensions)

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images")

    all_results = {}

    # 批量处理
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]

        # 预处理批次
        batch_tensors = []
        valid_paths = []
        for path in batch_paths:
            try:
                tensor = preprocess_image(path, transform)
                batch_tensors.append(tensor)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path.name}: {e}")
                continue

        if not batch_tensors:
            continue

        batch_tensor = torch.cat(batch_tensors, dim=0)

        # 推理
        with torch.no_grad():
            batch_tensor = batch_tensor.to(args.device)

            if args.mode in ['classify', 'both']:
                logits = model(batch_tensor, return_features=False)
                probs = F.softmax(logits.float(), dim=-1)
                top_probs, top_indices = torch.topk(probs, k=min(args.top_k, probs.size(-1)), dim=-1)

            if args.mode in ['cluster', 'both']:
                features = extract_tensor_batch(model, batch_tensor, args.output_type, args.layers, not args.no_intermediate_norm)

        # 保存结果
        for j, path in enumerate(valid_paths):
            if j >= len(batch_tensors):
                continue

            result = {'image': path.name}

            if args.mode in ['classify', 'both']:
                # 获取该图像的 top-k 结果
                img_top_probs = top_probs[j].cpu().numpy()
                img_top_indices = top_indices[j].cpu().numpy()

                classifications = []
                for prob, idx in zip(img_top_probs, img_top_indices):
                    if prob >= args.threshold:
                        class_id = int(idx)
                        class_name = class_mapping.get(class_id, f"Class {class_id}") if class_mapping else f"Class {class_id}"
                        classifications.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'probability': float(prob)
                        })

                # 如果需要，取前 top_k
                if len(classifications) > args.top_k:
                    classifications = classifications[:args.top_k]

                result['classification'] = classifications

            if args.mode in ['cluster', 'both']:
                result['features'] = features[j].cpu().numpy().tolist()

            all_results[str(path.relative_to(input_dir))] = result

        print(f"Processed {min(i + args.batch_size, len(image_paths))}/{len(image_paths)} images")

    return all_results


def main(args):
    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError('batch-size and top-k must be positive')
    bundle = load_model_bundle(checkpoint=args.checkpoint, model_name=args.model, device=args.device,
                               class_csv=args.class_csv, input_size=args.input_size)
    model, transform, class_mapping = bundle['model'], bundle['transform'], bundle['class_mapping']
    args.mode = resolved_mode(model, args.mode)
    print(f"Loaded {bundle['model_type']}: classifier={bundle['has_classifier']}, "
          f"features={bundle['feature_dim']} ({bundle['feature_source']}), input={bundle['input_size']}")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 判断输入类型
    input_path = Path(args.input)

    if input_path.is_file():
        # 单张图像
        results = process_single_image(args, model, transform, class_mapping)

        # 保存结果
        output_file = output_dir / f"{input_path.stem}_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
        if results and 'features' in results:
            (output_dir / 'features.npz').write_bytes(cache_bytes(torch.tensor([results['features']]),
                [input_path.name], args.output_type, args.layers))

    elif input_path.is_dir():
        # 目录批量处理
        results = process_directory(args, model, transform, class_mapping)

        # 保存结果
        output_file = output_dir / "batch_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")

        # 如果是聚类模式，额外保存特征矩阵
        if args.mode in ['cluster', 'both']:
            features_list = []
            image_names = []
            for name, result in results.items():
                if 'features' in result:
                    features_list.append(result['features'])
                    image_names.append(name)

            if features_list:
                features_array = np.array(features_list)
                np.save(output_dir / "features.npy", features_array)
                (output_dir / 'features.npz').write_bytes(cache_bytes(torch.from_numpy(features_array),
                    image_names, args.output_type, args.layers))
                with open(output_dir / "image_names.txt", 'w') as f:
                    f.write('\n'.join(image_names))
                print(f"Feature matrix saved: {output_dir / 'features.npy'}")
                print(f"Feature matrix shape: {features_array.shape}")
    else:
        print(f"Error: Invalid input path: {input_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Artist Style Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
