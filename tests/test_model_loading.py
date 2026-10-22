"""Functional tests with real, small DINOv3 networks and local checkpoints."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from aiohttp import web
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_loading import load_model_bundle, find_checkpoint, model_folders
from inference_artist import classify_image, resolved_mode
from kaloscope_dinov3.models.vision_transformer import DinoVisionTransformer


class ReferenceInferenceModel(torch.nn.Module):
    """Independent forward-only fixture for comparing loaded checkpoint outputs."""
    def __init__(self, config, num_classes):
        super().__init__()
        self.backbone = DinoVisionTransformer(**config["kwargs"])
        self.backbone.init_weights()
        self.feature_dim = 2 * self.backbone.embed_dim
        self.head = torch.nn.Linear(self.feature_dim, num_classes)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim), torch.nn.GELU(),
            torch.nn.Linear(self.feature_dim, config["projection_dim"]),
        )

    def forward(self, images, projections=False):
        tokens = self.backbone.forward_features(images)
        features = torch.cat((tokens["x_norm_clstoken"], tokens["x_norm_patchtokens"].mean(1)), dim=1)
        result = {"features": features, "logits": self.head(features)}
        if projections:
            result["projections"] = self.projector(features)
        return result


def load_nodes():
    fake = types.ModuleType("folder_paths")
    fake.models_dir = str(ROOT / "models")
    spec = importlib.util.spec_from_file_location("kaloscope_test_nodes", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    original = sys.modules.get("folder_paths")
    sys.modules["folder_paths"] = fake
    try:
        fake_server = types.ModuleType('server')
        fake_server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=web.RouteTableDef()))
        with patch.dict(sys.modules, {'server': fake_server}):
            spec.loader.exec_module(module)
    finally:
        if original is None:
            del sys.modules["folder_paths"]
        else:
            sys.modules["folder_paths"] = original
    return module


class ModelLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.options = {"name": "custom_vit", "pooling": "cls_mean", "projection_dim": 8,
                       "kwargs": {"embed_dim": 24, "depth": 2, "num_heads": 3, "patch_size": 16,
                                  "n_storage_tokens": 2,
                                  "pos_embed_rope_dtype": "fp32"}}
        torch.manual_seed(1)
        cls.original = ReferenceInferenceModel(cls.options, num_classes=3).eval()
        cls.nodes = load_nodes()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.config = {"model": "custom_vit", "input_size": 32}
        self.payload = {"model": self.original.state_dict(), "model_config": self.options,
                        "classes": ["a", "b", "c"]}
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.directory / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        torch.save(self.payload, self.directory / "best.pt")

    def bundle(self):
        return load_model_bundle(self.directory, device="cpu")

    def tensor(self, bundle):
        return bundle["transform"](Image.new("RGB", (48, 32), (70, 130, 190))).unsqueeze(0)

    def test_head_and_features_match_reference_model(self):
        bundle = self.bundle()
        batch = self.tensor(bundle)
        with torch.inference_mode():
            expected = self.original(batch)
            torch.testing.assert_close(bundle["model"](batch), expected["logits"])
            torch.testing.assert_close(bundle["model"](batch, return_features=True), expected["features"])
        self.assertTrue(bundle["has_classifier"])
        self.assertEqual(bundle["feature_dim"], 48)
        results = classify_image(bundle["model"], batch, "cpu", bundle["class_mapping"], top_k=20)
        self.assertEqual(len(results), 3)
        self.assertEqual({row["class_name"] for row in results}, {"a", "b", "c"})
        self.assertAlmostEqual(sum(row["probability"] for row in results), 1.0, places=6)

    def test_normalized_classifier_matches_training_and_preserves_features(self):
        # Exercise each supported config location, including safetensors-style
        # architecture objects that have no embedded checkpoint metadata.
        for location in ("top_level", "model", "embedded"):
            with self.subTest(location=location):
                self.config = {"model": "custom_vit", "input_size": 32}
                self.payload["model_config"] = dict(self.options)
                if location == "top_level":
                    self.config["classifier_input_normalization"] = "l2_sqrt_dim"
                elif location == "model":
                    self.config["model"] = dict(self.options, classifier_input_normalization="l2_sqrt_dim")
                else:
                    self.payload["model_config"]["classifier_input_normalization"] = "l2_sqrt_dim"
                self.config["feature_source"] = "projector"
                if location == "model":
                    self.config["model"]["feature_source"] = "projector"
                self.save()
                bundle = self.bundle()
                batch = self.tensor(bundle)
                with torch.inference_mode():
                    expected = self.original(batch, projections=True)
                    features = expected["features"]
                    logits = self.original.head(torch.nn.functional.normalize(features.float(), dim=-1)
                                                * features.shape[-1] ** 0.5)
                    output, actual = bundle["model"](batch, return_both=True)
                    torch.testing.assert_close(actual, logits)
                    torch.testing.assert_close(output, expected["projections"])
                    torch.testing.assert_close(bundle["model"](batch, return_features=True), output)
                    torch.testing.assert_close(bundle["model"].extract_tensor(batch, "backbone"), features)
                    self.assertFalse(torch.allclose(actual, expected["logits"]))

    def test_invalid_classifier_normalization_fails(self):
        self.config["classifier_input_normalization"] = "l2_typo"
        self.save()
        with self.assertRaisesRegex(ValueError, "classifier_input_normalization"):
            self.bundle()

    def test_no_head_is_feature_only_even_with_metadata_classes(self):
        self.payload["model"] = {k: v for k, v in self.payload["model"].items() if not k.startswith("head.")}
        self.save()
        bundle = self.bundle()
        self.assertFalse(bundle["has_classifier"])
        self.assertEqual(resolved_mode(bundle["model"], "auto"), "cluster")
        with self.assertRaisesRegex(ValueError, "no classification head"):
            classify_image(bundle["model"], self.tensor(bundle), "cpu")
        with torch.inference_mode():
            torch.testing.assert_close(bundle["model"](self.tensor(bundle), return_features=True),
                                       self.original(self.tensor(bundle))["features"])

    def test_temporal_projector_is_loaded_and_used(self):
        self.payload["model"] = {k: v for k, v in self.payload["model"].items() if not k.startswith("head.")}
        self.payload["model"].update(log_temperature=torch.tensor(1.0), bias=torch.tensor(-1.0))
        self.save()
        bundle = self.bundle()
        self.assertEqual(bundle["feature_source"], "projector")
        self.assertEqual(bundle["feature_dim"], 8)
        with torch.inference_mode():
            expected = self.original(self.tensor(bundle), projections=True)["projections"]
            torch.testing.assert_close(bundle["model"](self.tensor(bundle), return_features=True), expected)

    def test_raw_backbone_and_official_linear_head(self):
        self.payload = dict(self.original.backbone.state_dict())
        self.payload.update({"linear_head.weight": self.original.head.weight,
                             "linear_head.bias": self.original.head.bias})
        self.config["model"] = {"name": "custom_vit", "kwargs": self.options["kwargs"]}
        self.save()
        bundle = self.bundle()
        self.assertEqual(bundle["model"].pooling, "cls_mean")
        with torch.inference_mode():
            torch.testing.assert_close(bundle["model"](self.tensor(bundle)), self.original(self.tensor(bundle))["logits"])
        self.assertTrue(classify_image(bundle["model"], self.tensor(bundle), "cpu")[0]["class_name"].startswith("Class "))

    def test_raw_headless_safetensors(self):
        from safetensors.torch import save_file
        self.config["model"] = {"name": "custom_vit", "kwargs": self.options["kwargs"]}
        self.config["checkpoint"] = "backbone.safetensors"
        self.save()
        save_file(self.original.backbone.state_dict(), self.directory / "backbone.safetensors")
        bundle = self.bundle()
        self.assertFalse(bundle["has_classifier"])
        self.assertEqual(bundle["feature_dim"], 24)
        with torch.inference_mode():
            expected = self.original.backbone.forward_features(self.tensor(bundle))["x_norm_clstoken"]
            torch.testing.assert_close(bundle["model"](self.tensor(bundle), return_features=True), expected)

    def test_wrapped_state_dict_prefixes(self):
        self.payload = {"state_dict": {"module._orig_mod." + k: v for k, v in self.payload["model"].items()},
                        "model_config": self.options}
        self.save()
        self.assertTrue(self.bundle()["has_classifier"])

    def test_checkpoint_transform_path_works_without_finetune_module(self):
        from kaloscope_dinov3.preprocessing import lvd_transform
        image = Image.new("RGB", (48, 32), "blue")
        for prefix in ("dinov3.finetune.data.", "kaloscope_dinov3.finetune.data."):
            self.config["data"] = {"custom_transform": prefix + "lvd_transform"}
            self.save()
            torch.testing.assert_close(self.bundle()["transform"](image), lvd_transform(32)(image))

    def test_all_final_feature_outputs_match_backbone(self):
        bundle = self.bundle()
        model = bundle['model']
        batch = self.tensor(bundle)
        with torch.inference_mode():
            raw = self.original.backbone.forward_features(batch)
            cls, patches, storage = raw['x_norm_clstoken'], raw['x_norm_patchtokens'], raw['x_storage_tokens']
            pooled = torch.cat((cls, patches.mean(1)), dim=1)
            expected = {
                'default': pooled, 'backbone': pooled, 'cls': cls, 'mean': patches.mean(1),
                'cls_mean': pooled, 'patch_tokens': patches, 'storage_tokens': storage,
                'all_tokens': torch.cat((cls.unsqueeze(1), storage, patches), dim=1),
                'prenorm': raw['x_prenorm'], 'projector': self.original.projector(pooled),
                'patch_map': patches.transpose(1, 2).reshape(1, 24, 2, 2),
            }
            for kind, tensor in expected.items():
                with self.subTest(output=kind):
                    torch.testing.assert_close(model.extract_tensor(batch, kind), tensor)
        # Loading backbone as default must still retain trained projector for selection.
        self.assertEqual(bundle['feature_source'], 'backbone')
        self.assertIsNotNone(model.projector)

    def test_intermediate_features_keep_layer_order_and_shape(self):
        model = self.bundle()['model']
        batch = self.tensor(self.bundle())
        with torch.inference_mode():
            native = self.original.backbone.get_intermediate_layers(
                batch, n=[0, 1], return_class_token=True, return_extra_tokens=True)
            for kind in ('cls', 'mean', 'cls_mean', 'patch_tokens', 'patch_map', 'storage_tokens', 'all_tokens'):
                expected = []
                for patches, cls, storage in reversed(native):
                    if kind == 'cls':
                        tensor = cls
                    elif kind == 'mean':
                        tensor = patches.mean(1)
                    elif kind == 'cls_mean':
                        tensor = torch.cat((cls, patches.mean(1)), dim=1)
                    elif kind == 'patch_tokens':
                        tensor = patches
                    elif kind == 'patch_map':
                        tensor = patches.transpose(1, 2).reshape(1, 24, 2, 2)
                    elif kind == 'storage_tokens':
                        tensor = storage
                    else:
                        tensor = torch.cat((cls.unsqueeze(1), storage, patches), dim=1)
                    expected.append(tensor)
                with self.subTest(output=kind):
                    actual = model.extract_tensor(batch, 'intermediate_' + kind, '-1,0')
                    torch.testing.assert_close(actual, torch.stack(expected, dim=1))
            unnorm = self.original.backbone.get_intermediate_layers(
                batch, n=[0, 1], norm=False, return_class_token=True, return_extra_tokens=True)
            expected = torch.stack([torch.cat((cls.unsqueeze(1), storage, patches), dim=1)
                                    for patches, cls, storage in unnorm], dim=1)
            torch.testing.assert_close(model.extract_tensor(batch, 'intermediate_prenorm', '0,1'), expected)
            torch.testing.assert_close(model.extract_tensor(batch, 'intermediate_all_tokens', '0,1', False), expected)
        self.assertEqual(tuple(model.extract_tensor(batch, 'intermediate_patch_tokens').shape), (1, 1, 4, 24))

    def test_feature_selection_does_not_change_classifier(self):
        bundle = self.bundle()
        model = bundle['model']
        batch = self.tensor(bundle)
        with torch.inference_mode():
            before = model(batch)
            for kind in ('mean', 'projector', 'patch_tokens', 'cls'):
                model.extract_tensor(batch, kind)
            torch.testing.assert_close(model(batch), before)
        self.assertEqual(model.pooling, 'cls_mean')
        self.assertEqual(model.feature_source, 'backbone')

    def test_missing_projector_and_invalid_layers_fail_clearly(self):
        self.payload['model'] = {k: v for k, v in self.payload['model'].items() if not k.startswith('projector.')}
        self.save()
        bundle = self.bundle()
        batch = self.tensor(bundle)
        with self.assertRaisesRegex(ValueError, 'no supported projector'):
            bundle['model'].extract_tensor(batch, 'projector')
        for layers in ('2', '-3', '0,0', 'one', ''):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                bundle['model'].extract_tensor(batch, 'intermediate_cls', layers)

    def test_feature_node_returns_selected_tensor_for_image_batch(self):
        bundle = self.bundle()
        image = torch.full((2, 32, 48, 3), 0.5)
        node = self.nodes.KaloscopeExtractFeaturesNode()
        for kind, shape in {
            'cls': (2, 24), 'mean': (2, 24), 'cls_mean': (2, 48), 'projector': (2, 8),
            'patch_tokens': (2, 4, 24), 'patch_map': (2, 24, 2, 2),
            'storage_tokens': (2, 2, 24), 'all_tokens': (2, 7, 24), 'prenorm': (2, 7, 24),
            'intermediate_patch_tokens': (2, 2, 4, 24),
            'intermediate_patch_map': (2, 2, 24, 2, 2),
        }.items():
            with self.subTest(output=kind):
                result = node.extract(image, bundle, kind, '0,1')[0]
                self.assertEqual(tuple(result.shape), shape)
                self.assertEqual(result.device.type, 'cpu')
                self.assertFalse(result.requires_grad)
                self.assertTrue(torch.isfinite(result).all())

    def test_convnext_feature_outputs_and_variable_stages(self):
        from kaloscope_dinov3.models.convnext import ConvNeXt
        from model_loading import DinoInferenceModel
        backbone = ConvNeXt(depths=[1, 1, 1, 1], dims=[8, 16, 24, 32]).eval()
        backbone.init_weights()  # Custom LayerNorm parameters start uninitialized.
        model = DinoInferenceModel(backbone, 'cls').eval()
        images = torch.rand(2, 3, 64, 96)
        with torch.inference_mode():
            raw = backbone.forward_features(images)
            torch.testing.assert_close(model.extract_tensor(images, 'patch_tokens'), raw['x_norm_patchtokens'])
            patch_map = model.extract_tensor(images, 'patch_map')
            self.assertEqual(tuple(patch_map.shape), (2, 32, 2, 3))
            torch.testing.assert_close(patch_map.flatten(2).transpose(1, 2), raw['x_norm_patchtokens'])
            intermediate = model.extract_tensor(images, 'intermediate_patch_map', '1')
            self.assertEqual(tuple(intermediate.shape), (2, 1, 16, 8, 12))
            torch.testing.assert_close(model.extract_tensor(images, 'intermediate_prenorm', '-1')[:, 0], raw['x_prenorm'])
        with self.assertRaisesRegex(ValueError, 'no storage/register tokens'):
            model.extract_tensor(images, 'storage_tokens')
        with self.assertRaisesRegex(ValueError, 'different tensor shapes'):
            model.extract_tensor(images, 'intermediate_patch_tokens', '0,1')

    def test_lsnet_feature_node_preserves_default_and_rejects_dino_outputs(self):
        encoder = torch.nn.Identity()
        encoder.forward = lambda batch, return_features: batch.mean((2, 3))
        from kaloscope_dinov3.preprocessing import image_transform
        bundle = {'model': encoder, 'transform': image_transform(32), 'device': 'cpu'}
        node = self.nodes.KaloscopeExtractFeaturesNode()
        image = torch.ones(1, 32, 32, 3)
        self.assertEqual(tuple(node.extract(image, bundle)[0].shape), (1, 3))
        with self.assertRaisesRegex(ValueError, 'requires a DINOv3'):
            node.extract(image, bundle, 'patch_tokens')

    def test_bad_architecture_is_not_silently_ignored(self):
        self.config["model"] = "dinov3_typo"
        self.save()
        with self.assertRaises(ValueError):
            self.bundle()

    def test_incomplete_backbone_fails(self):
        del self.payload["model"]["backbone.cls_token"]
        self.save()
        with self.assertRaisesRegex(RuntimeError, "cls_token"):
            self.bundle()

    def test_class_mapping_must_match_head(self):
        (self.directory / "class_mapping.csv").write_text("class_id,class_name\n0,one\n2,two\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mapping IDs"):
            self.bundle()

    def test_model_discovery_only_uses_kaloscope(self):
        for folder in ("lsnet/shared", "kaloscope/shared", "lsnet/other"):
            (self.directory / folder).mkdir(parents=True)
        folders = model_folders(self.directory)
        self.assertEqual(folders, {"shared": self.directory / "kaloscope/shared"})

    def test_architecture_must_be_explicit(self):
        del self.config['model']
        self.save()
        with self.assertRaisesRegex(ValueError, 'model architecture in config.json'):
            self.bundle()
        bundle = load_model_bundle(self.directory, device='cpu', model_name='custom_vit')
        self.assertEqual(bundle['model_type'], 'custom_vit')

    def test_ambiguous_checkpoint_requires_selection(self):
        (self.directory / "best.pt").rename(self.directory / "one.pt")
        torch.save(self.payload, self.directory / "two.pth")
        with self.assertRaisesRegex(ValueError, "Expected one checkpoint"):
            find_checkpoint(self.directory)
        self.config["checkpoint"] = "two.pth"
        self.save()
        self.assertEqual(find_checkpoint(self.directory).name, "two.pth")

    def test_kaloscope_nodes_and_model_sockets(self):
        bundle = self.bundle()
        nodes = self.nodes
        self.assertEqual(len(nodes.NODE_CLASS_MAPPINGS), 19)
        self.assertEqual(set(nodes.NODE_CLASS_MAPPINGS), set(nodes.NODE_DISPLAY_NAME_MAPPINGS))
        self.assertEqual(nodes.KaloscopeModelLoader.RETURN_TYPES, ("KALOSCOPE_MODEL",))
        for name, node in nodes.NODE_CLASS_MAPPINGS.items():
            schema = node.INPUT_TYPES()
            self.assertTrue(name.startswith('Kaloscope'))
            if "model" in schema.get("required", {}):
                expected = 'MODEL' if name == 'KaloscopeZ3Apply' else 'KALOSCOPE_MODEL'
                self.assertEqual(schema["required"]["model"][0], expected)
            self.assertTrue(node.__name__.startswith("Kaloscope"))
            self.assertIn(node.CATEGORY, ("Kaloscope", "Kaloscope/Analysis", "Kaloscope/Style Transfer"))
        with patch.object(nodes, "model_folders", return_value={"sample": self.directory}):
            loaded = nodes.KaloscopeModelLoader().load("sample", "cpu")[0]
        self.assertTrue(loaded["has_classifier"])
        image = torch.full((2, 32, 48, 3), 0.5)
        features = nodes.KaloscopeExtractFeaturesNode().extract(image, bundle)[0]
        self.assertEqual(tuple(features.shape), (2, 48))
        tags, predictions = nodes.KaloscopeArtistInferenceNode().process(image, bundle, 5, 0.0)
        self.assertEqual(set(tags.split(",")), {"a", "b", "c"})
        self.assertEqual(len(json.loads(predictions)), 3)
        result, visualization = nodes.KaloscopeClusteringNode().cluster(
            "kmeans", 2, 0.5, 2, True, "pca", 5, group_1=torch.randn(4, 48))
        self.assertEqual(len(json.loads(result)["labels"]), 4)
        self.assertEqual(visualization.shape[-1], 3)

    def test_backend_auto_features_and_api(self):
        from backend_lsnet.inference import process_image_from_pil
        from backend_lsnet.api import Api
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        self.payload["model"] = {k: v for k, v in self.payload["model"].items() if not k.startswith("head.")}
        self.save()
        image = Image.new("RGB", (48, 32), "blue")
        result = process_image_from_pil(image, checkpoint=str(self.directory / "best.pt"), device="cpu")
        self.assertEqual(len(result["features"]), 48)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        app = FastAPI()
        api = Api(app)
        with patch("backend_lsnet.api.get_available_checkpoints", return_value=["best.pt"]), \
             patch("backend_lsnet.api.get_checkpoint_path", return_value=str(self.directory / "best.pt")), \
             patch("backend_lsnet.api.get_class_csv", return_value=None), TestClient(app) as client:
            response = client.post('/kaloscope/v1/infer', json={"input_image": base64.b64encode(buffer.getvalue()).decode(), "device": "cpu"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(response.json()["results"]["features"]), 48)
            self.assertTrue(all(route.path.startswith('/kaloscope/v1/') for route in app.routes
                                if hasattr(route, 'endpoint') and route.endpoint.__module__ == 'backend_lsnet.api'))
        api.executor.shutdown()


if __name__ == "__main__":
    unittest.main()
