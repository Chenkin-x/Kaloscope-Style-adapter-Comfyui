"""Compare a supplied temporal checkpoint against its original training encoder.

Run from the plugin directory:
python tests/smoke_checkpoint.py --model-dir ../model --source ../kaloscope-dinov3
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import types

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_loading import load_model_bundle, load_checkpoint_payload
from test_model_loading import load_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve()))
    from dinov3.finetune.temporal.model import Encoder
    bundle = load_model_bundle(args.model_dir, device=args.device)
    payload = load_checkpoint_payload(bundle["checkpoint"])
    original = Encoder(payload["model_config"], initialize=False)
    original.load_state_dict(payload["model"], strict=True)
    original.to(args.device).eval()
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (480, 640, 3), dtype=np.uint8))
    spatial = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512), transforms.PILToTensor()])
    raw = spatial(image).unsqueeze(0).to(args.device)
    batch = bundle["transform"](image).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        expected = original(raw)[:, original.feature_dim:]
        actual = bundle["model"](batch, return_features=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(actual).all()
    print(json.dumps({"architecture": bundle["model_type"], "has_classifier": bundle["has_classifier"],
                      "input_size": bundle["input_size"], "feature_source": bundle["feature_source"],
                      "features": list(actual.shape), "max_error_vs_training_encoder": (actual - expected).abs().max().item()}, indent=2), flush=True)
    with torch.inference_mode():
        tokens = original.backbone.forward_features(batch)
        cls, patches, storage = tokens['x_norm_clstoken'], tokens['x_norm_patchtokens'], tokens['x_storage_tokens']
        pooled = torch.cat((cls, patches.mean(1)), dim=1)
        references = {
            'default': actual, 'backbone': pooled, 'cls': cls, 'mean': patches.mean(1),
            'cls_mean': pooled, 'projector': actual, 'patch_tokens': patches,
            'patch_map': patches.transpose(1, 2).reshape(1, 768, 32, 32),
            'storage_tokens': storage, 'all_tokens': torch.cat((cls.unsqueeze(1), storage, patches), dim=1),
            'prenorm': tokens['x_prenorm'],
        }
        shapes = {}
        for kind, reference in references.items():
            result = bundle['model'].extract_tensor(batch, kind)
            torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)
            shapes[kind] = list(result.shape)
        native = original.backbone.get_intermediate_layers(batch, n=[8, 9, 10, 11],
            return_class_token=True, return_extra_tokens=True)
        for kind in ('cls', 'mean', 'cls_mean', 'patch_tokens', 'patch_map', 'storage_tokens', 'all_tokens'):
            expected_layers = []
            for patch, cls_layer, registers in native:
                if kind == 'cls':
                    value = cls_layer
                elif kind == 'mean':
                    value = patch.mean(1)
                elif kind == 'cls_mean':
                    value = torch.cat((cls_layer, patch.mean(1)), dim=1)
                elif kind == 'patch_tokens':
                    value = patch
                elif kind == 'patch_map':
                    value = patch.transpose(1, 2).reshape(1, 768, 32, 32)
                elif kind == 'storage_tokens':
                    value = registers
                else:
                    value = torch.cat((cls_layer.unsqueeze(1), registers, patch), dim=1)
                expected_layers.append(value)
            output_type = 'intermediate_' + kind
            result = bundle['model'].extract_tensor(batch, output_type, '-4,-3,-2,-1')
            torch.testing.assert_close(result, torch.stack(expected_layers, dim=1), rtol=1e-5, atol=1e-5)
            shapes[output_type] = list(result.shape)
        native_raw = original.backbone.get_intermediate_layers(batch, n=[8, 9, 10, 11], norm=False,
            return_class_token=True, return_extra_tokens=True)
        expected_raw = torch.stack([torch.cat((cls_layer.unsqueeze(1), registers, patch), dim=1)
                                   for patch, cls_layer, registers in native_raw], dim=1)
        result = bundle['model'].extract_tensor(batch, 'intermediate_prenorm', '8,9,10,11')
        torch.testing.assert_close(result, expected_raw, rtol=1e-5, atol=1e-5)
        shapes['intermediate_prenorm'] = list(result.shape)
        print('All feature outputs match original backbone:', json.dumps(shapes), flush=True)
    del original, payload
    nodes = load_nodes()
    image_tensor = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255
    tags, features_json = nodes.KaloscopeArtistInferenceNode().process(image_tensor, bundle, 5, 0.0)
    assert tags == "" and len(json.loads(features_json)["features"]) == 256
    features = nodes.KaloscopeExtractFeaturesNode().extract(image_tensor, bundle)[0]
    torch.testing.assert_close(features, actual.cpu(), rtol=1e-4, atol=1e-4)
    similarity = json.loads(nodes.KaloscopeArtistSimilarityNode().process(image_tensor, image_tensor, bundle)[0])
    assert abs(similarity["similarities"][0] - 1.0) < 1e-5
    print("ComfyUI inference, extraction and self-similarity passed", flush=True)
    from backend_lsnet.inference import process_image_from_pil
    output = process_image_from_pil(image, checkpoint=bundle["checkpoint"], device=args.device)
    np.testing.assert_allclose(output["features"], actual[0].cpu().numpy(), rtol=1e-4, atol=1e-4)
    print("Standalone/API backend output matches", flush=True)
    from inference_artist import get_args_parser, main as cli_main
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        image.save(path / "input.png")
        cli_args = get_args_parser().parse_args(["--checkpoint", bundle["checkpoint"], "--input", str(path / "input.png"),
            "--output", str(path / "output"), "--device", args.device])
        cli_main(cli_args)
        result = json.loads((path / "output/input_result.json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(result["features"], actual[0].cpu().numpy(), rtol=1e-4, atol=1e-4)
    print("CLI auto inference matches", flush=True)


if __name__ == "__main__":
    main()
