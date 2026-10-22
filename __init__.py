import os
import sys
import json
import torch
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(__file__))

import folder_paths

from model_loading import FEATURE_OUTPUTS, load_model_bundle, model_folders
from inference_artist import classify_image, extract_features
from feature_analysis import CHART_TYPES, TENSOR_LAYOUTS, analyze_features
from backend_lsnet.analysis import extract_batch, output_layout

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class KaloscopeModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        subfolders = sorted(model_folders(folder_paths.models_dir))

        return {
            "required": {
                "model_folder": (subfolders, {"default": subfolders[0] if subfolders else ""}),
                "device": ("STRING", {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("KALOSCOPE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Kaloscope"

    def load(self, model_folder, device):
        folders = model_folders(folder_paths.models_dir)
        if model_folder not in folders:
            raise FileNotFoundError(f"Model folder not found: {model_folder}")
        return (load_model_bundle(folders[model_folder], device=device),)

class KaloscopeArtistInferenceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("KALOSCOPE_MODEL",),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 100}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tag_string", "json_output")
    FUNCTION = "process"
    CATEGORY = "Kaloscope"

    def process(self, image, model, top_k, threshold):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        class_mapping = model_bundle['class_mapping']
        device = model_bundle['device']

        if image.ndim == 4:
            image = image[0]
        image = (image * 255).clamp(0, 255).byte().cpu().numpy()
        pil_image = Image.fromarray(image)

        # Preprocess image
        image_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

        if not model_bundle['has_classifier']:
            features = extract_features(model, image_tensor, device)[0].tolist()
            return ('', json.dumps({'features': features, 'feature_dim': len(features),
                                    'feature_source': model_bundle['feature_source']}, ensure_ascii=False))
        results = classify_image(model, image_tensor, device, class_mapping, top_k, threshold)

        # Prepare outputs
        tags = [res['class_name'] for res in results]
        tag_string = ",".join(tags)
        tag_dict = {res['class_name']: res['probability'] for res in results}
        json_output = json.dumps(tag_dict, ensure_ascii=False)

        return (tag_string, json_output)

class KaloscopeArtistSimilarityNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "processed_image": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "model": ("KALOSCOPE_MODEL",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("similarity_json",)
    FUNCTION = "process"
    CATEGORY = "Kaloscope"

    def process(self, processed_image, reference_images, model):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        device = model_bundle['device']

        def image_to_tensor(img):
            if img.ndim == 4:
                img = img[0]
            img = (img * 255).clamp(0, 255).byte().cpu().numpy()
            pil_img = Image.fromarray(img)
            return transform(pil_img).unsqueeze(0)

        processed_tensor = image_to_tensor(processed_image)
        with torch.no_grad():
            processed_tensor = processed_tensor.to(device)
            processed_features = model(processed_tensor, return_features=True).cpu().numpy()[0]

        references = []
        similarities = []
        num_refs = reference_images.shape[0] if reference_images.ndim == 4 else 1
        for i in range(num_refs):
            ref_img = reference_images[i] if reference_images.ndim == 4 else reference_images
            ref_tensor = image_to_tensor(ref_img)
            with torch.no_grad():
                ref_tensor = ref_tensor.to(device)
                ref_features = model(ref_tensor, return_features=True).cpu().numpy()[0]
            references.append(ref_features.tolist())
            sim = np.dot(processed_features, ref_features) / (np.linalg.norm(processed_features) * np.linalg.norm(ref_features))
            similarities.append(float(sim))

        result = {
            "processed_features": processed_features.tolist(),
            "reference_features": references,
            "similarities": similarities
        }
        json_output = json.dumps(result, ensure_ascii=False)

        return (json_output,)

class KaloscopeCommonFeaturesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_images": ("IMAGE",),
                "model": ("KALOSCOPE_MODEL",),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("common_features",)
    FUNCTION = "process"
    CATEGORY = "Kaloscope"

    def process(self, reference_images, model):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        device = model_bundle['device']

        def image_to_tensor(img):
            if img.ndim == 4:
                img = img[0]
            img = (img * 255).clamp(0, 255).byte().cpu().numpy()
            pil_img = Image.fromarray(img)
            return transform(pil_img).unsqueeze(0)

        references = []
        num_refs = reference_images.shape[0] if reference_images.ndim == 4 else 1
        for i in range(num_refs):
            ref_img = reference_images[i] if reference_images.ndim == 4 else reference_images
            ref_tensor = image_to_tensor(ref_img)
            with torch.no_grad():
                ref_tensor = ref_tensor.to(device)
                ref_features = model(ref_tensor, return_features=True).cpu().numpy()[0]
            references.append(ref_features)

        if references:
            common_features = np.mean(np.array(references), axis=0)
        else:
            common_features = np.zeros(model_bundle['feature_dim'])
        return (torch.tensor(common_features),)

class KaloscopeClusteringNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (["kmeans", "dbscan", "hierarchical"], {"default": "kmeans"}),
                "n_clusters": ("INT", {"default": 10, "min": 2, "max": 100}),
                "eps": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0}),
                "min_samples": ("INT", {"default": 5, "min": 1, "max": 50}),
                "visualize": ("BOOLEAN", {"default": True}),
                "viz_method": (["tsne", "pca"], {"default": "tsne"}),
                "perplexity": ("INT", {"default": 30, "min": 5, "max": 100}),
            },
            "optional": {
                "group_1": ("TENSOR",),
                "group_2": ("TENSOR",),
                "group_3": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("clustering_json", "visualization")
    FUNCTION = "cluster"
    CATEGORY = "Kaloscope"

    def cluster(self, method, n_clusters, eps, min_samples, visualize, viz_method, perplexity, group_1=None, group_2=None, group_3=None):
        groups = []
        group_sizes = []
        for g in [group_1, group_2, group_3]:
            if g is not None:
                groups.append(g.cpu().numpy())
                group_sizes.append(g.shape[0])

        if not groups:
            return (json.dumps({"error": "No groups provided"}), torch.zeros(1, 64, 64, 3))

        features_np = np.vstack(groups)
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(features_np)
            centers = clusterer.cluster_centers_
        elif method == "dbscan":
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(features_np)
            centers = None
        elif method == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(features_np)
            centers = None

        result = {
            "method": method,
            "n_samples": len(features_np),
            "group_sizes": group_sizes,
            "labels": labels.tolist(),
        }
        if centers is not None:
            result["centers"] = centers.tolist()

        json_output = json.dumps(result, ensure_ascii=False)

        if visualize and len(features_np) > 1:
            if viz_method == "tsne":
                reducer = TSNE(n_components=2, perplexity=min(perplexity, len(features_np)-1), random_state=42)
            else:
                reducer = PCA(n_components=2, random_state=42)

            reduced_features = reducer.fit_transform(features_np)

            plt.figure(figsize=(10, 8))
            unique_labels = np.unique(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = labels == label
                plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1],
                           color=color, label=f'Cluster {label}', alpha=0.7)

            plt.title(f'{method.upper()} Clustering ({viz_method.upper()})')
            plt.legend()
            plt.tight_layout()

            fig = plt.gcf()
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            pil_image = Image.fromarray(img_array[:, :, :3])
            plt.close()

            viz_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
            if viz_tensor.ndim == 3:
                viz_tensor = viz_tensor.unsqueeze(0)
        else:
            viz_tensor = torch.zeros(1, 64, 64, 3)

        return (json_output, viz_tensor)

class KaloscopeFeatureComparisonNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("KALOSCOPE_MODEL",),
            },
            "optional": {
                "group_1": ("TENSOR",),
                "group_2": ("TENSOR",),
                "group_3": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("comparison_json",)
    FUNCTION = "compare"
    CATEGORY = "Kaloscope"

    def compare(self, image, model, group_1=None, group_2=None, group_3=None):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        device = model_bundle['device']

        if image.ndim == 4:
            image = image[0]
        image_np = (image * 255).clamp(0, 255).byte().cpu().numpy()
        pil_image = Image.fromarray(image_np)
        image_tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            query_features = model(image_tensor, return_features=True).cpu().numpy()[0]

        groups = []
        for g in [group_1, group_2, group_3]:
            if g is not None:
                groups.append(g.cpu().numpy())

        if not groups:
            return (json.dumps({"error": "No groups provided"}),)

        similarities = []
        for group_feat in groups:
            sim = np.dot(query_features, group_feat) / (np.linalg.norm(query_features) * np.linalg.norm(group_feat))
            similarities.append(float(sim))

        best_index = np.argmax(similarities)
        best_similarity = similarities[best_index]
        result = {
            "best_group_index": int(best_index),
            "best_similarity": best_similarity,
            "all_similarities": similarities
        }
        json_output = json.dumps(result, ensure_ascii=False)

        return (json_output,)

class KaloscopeArtistImageConnector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stacked_images",)
    FUNCTION = "connect"
    CATEGORY = "Kaloscope"

    def connect(self, image_1, image_2, image_3):
        def normalize_image(img):
            if img.ndim == 4:
                img = img[0]
            return img.unsqueeze(0)

        img1 = normalize_image(image_1)
        img2 = normalize_image(image_2)
        img3 = normalize_image(image_3)

        stacked = torch.cat([img1, img2, img3], dim=0)
        return (stacked,)

class KaloscopeExtractFeaturesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {'image': ('IMAGE',), 'model': ('KALOSCOPE_MODEL',)},
            'optional': {
                'output_type': (list(FEATURE_OUTPUTS), {'default': 'default'}),
                'layers': ('STRING', {'default': '-1', 'tooltip': 'Intermediate layer indices, e.g. -1 or 8,9,10,11; negative indices count from the end.'}),
                'intermediate_norm': ('BOOLEAN', {'default': True, 'tooltip': 'Apply model LayerNorm to intermediate features; prenorm always skips it.'}),
            },
        }

    RETURN_TYPES = ('TENSOR',)
    RETURN_NAMES = ('features',)
    FUNCTION = 'extract'
    CATEGORY = 'Kaloscope'

    @torch.inference_mode()
    def extract(self, image, model, output_type='default', layers='-1', intermediate_norm=True):
        images = image if image.ndim == 4 else image.unsqueeze(0)
        pil_images = [Image.fromarray((img * 255).clamp(0, 255).byte().cpu().numpy()) for img in images]
        return (extract_batch(pil_images, model, output_type, layers, intermediate_norm),)


class KaloscopeFeatureAnalysisNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {'features': ('TENSOR',), 'chart_type': (list(CHART_TYPES),)},
            'optional': {
                'metric': (['cosine', 'euclidean', 'manhattan'], {'default': 'cosine'}),
                'normalize': ('BOOLEAN', {'default': True, 'tooltip': 'Normalize each image vector to unit L2 norm before distance/clustering.'}),
                'cluster_method': (['kmeans', 'agglomerative', 'dbscan', 'none'], {'default': 'kmeans'}),
                'n_clusters': ('INT', {'default': 3, 'min': 1, 'max': 512}),
                'top_k': ('INT', {'default': 2, 'min': 1, 'max': 511, 'tooltip': 'Neighbor count for relation edges, neighbor ranking and isolation scores.'}),
                'reference_index': ('INT', {'default': 0, 'min': 0, 'max': 511}),
                'tensor_layout': (list(TENSOR_LAYOUTS), {'default': 'auto', 'tooltip': 'For 4D tensors select spatial [B,D,H,W] or layer_tokens [B,L,N,D] explicitly.'}),
                'layer_index': ('INT', {'default': -1, 'min': -128, 'max': 127}),
                'layer_pooling': (['selected', 'mean'], {'default': 'selected'}),
                'token_pooling': (['mean', 'flatten'], {'default': 'mean'}),
                'labels': ('STRING', {'default': '', 'multiline': True, 'tooltip': 'One image name per line or a JSON array, matching feature batch order.'}),
                'images': ('IMAGE', {'tooltip': 'Optional thumbnails only; never used for model inference.'}),
                'seed': ('INT', {'default': 42, 'min': 0, 'max': 2147483647}),
                'perplexity': ('FLOAT', {'default': 5.0, 'min': 0.5, 'max': 100.0}),
                'dbscan_eps': ('FLOAT', {'default': 0.35, 'min': 0.001, 'max': 100.0}),
                'dbscan_min_samples': ('INT', {'default': 2, 'min': 1, 'max': 512}),
                'max_dimensions': ('INT', {'default': 32, 'min': 1, 'max': 128}),
                'heatmap_order': (['cluster', 'input'], {'default': 'cluster'}),
                'grid_width': ('INT', {'default': 0, 'min': 0, 'max': 4096, 'tooltip': 'Patch grid columns; 0 infers a square grid. Spatial maps preserve their H,W.'}),
                'width': ('INT', {'default': 1400, 'min': 512, 'max': 4096, 'step': 64}),
                'height': ('INT', {'default': 1000, 'min': 512, 'max': 4096, 'step': 64}),
            },
        }

    RETURN_TYPES = ('IMAGE', 'STRING', 'TENSOR')
    RETURN_NAMES = ('visualization', 'analysis_json', 'distance_matrix')
    FUNCTION = 'analyze'
    CATEGORY = 'Kaloscope/Analysis'

    def analyze(self, features, chart_type='relationship_graph', **kwargs):
        return analyze_features(features, chart_type=chart_type, **kwargs)


class KaloscopeImageAnalysisNode:
    @classmethod
    def INPUT_TYPES(cls):
        schema = KaloscopeFeatureAnalysisNode.INPUT_TYPES()
        schema['required'].pop('features')
        schema['required'] = {'image': ('IMAGE',), 'model': ('KALOSCOPE_MODEL',), **schema['required']}
        schema['optional'].pop('images')
        schema['optional'].update(KaloscopeExtractFeaturesNode.INPUT_TYPES()['optional'])
        return schema

    RETURN_TYPES = ('IMAGE', 'STRING', 'TENSOR', 'TENSOR')
    RETURN_NAMES = ('visualization', 'analysis_json', 'features', 'distance_matrix')
    FUNCTION = 'analyze'
    CATEGORY = 'Kaloscope/Analysis'

    def analyze(self, image, model, chart_type='relationship_graph', output_type='default', layers='-1',
                intermediate_norm=True, **kwargs):
        features = KaloscopeExtractFeaturesNode().extract(image, model, output_type, layers, intermediate_norm)[0]
        images = image if image.ndim == 4 else image.unsqueeze(0)
        if kwargs.get('tensor_layout', 'auto') == 'auto':
            kwargs['tensor_layout'] = output_layout(output_type)
        visualization, report, distances = analyze_features(features, chart_type=chart_type, images=images, **kwargs)
        return visualization, report, features, distances


NODE_CLASS_MAPPINGS = {
    'KaloscopeModelLoader': KaloscopeModelLoader,
    'KaloscopeArtistInference': KaloscopeArtistInferenceNode,
    'KaloscopeArtistSimilarity': KaloscopeArtistSimilarityNode,
    'KaloscopeCommonFeatures': KaloscopeCommonFeaturesNode,
    'KaloscopeClustering': KaloscopeClusteringNode,
    'KaloscopeFeatureComparison': KaloscopeFeatureComparisonNode,
    'KaloscopeArtistImageConnector': KaloscopeArtistImageConnector,
    'KaloscopeExtractFeatures': KaloscopeExtractFeaturesNode,
    'KaloscopeFeatureAnalysis': KaloscopeFeatureAnalysisNode,
    'KaloscopeImageAnalysis': KaloscopeImageAnalysisNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'KaloscopeModelLoader': 'Kaloscope Model Loader',
    'KaloscopeArtistInference': 'Kaloscope Artist Inference',
    'KaloscopeArtistSimilarity': 'Kaloscope Artist Similarity',
    'KaloscopeCommonFeatures': 'Kaloscope Common Features',
    'KaloscopeClustering': 'Kaloscope Clustering',
    'KaloscopeFeatureComparison': 'Kaloscope Feature Comparison',
    'KaloscopeArtistImageConnector': 'Kaloscope Image Connector',
    'KaloscopeExtractFeatures': 'Kaloscope Extract Features',
    'KaloscopeFeatureAnalysis': 'Kaloscope Feature Analysis',
    'KaloscopeImageAnalysis': 'Kaloscope Image Analysis',
}

# Lazy Comfy imports inside the nodes preserve standalone/API entry points.
from kaloscope_style.nodes import NODE_CLASS_MAPPINGS as STYLE_NODES, NODE_DISPLAY_NAME_MAPPINGS as STYLE_NAMES
NODE_CLASS_MAPPINGS.update(STYLE_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(STYLE_NAMES)

WEB_DIRECTORY = './web'

from server import PromptServer
from kaloscope_style.nodes import artist_bank_paths
from kaloscope_style.routes import register_routes
register_routes(PromptServer.instance.routes, artist_bank_paths)
