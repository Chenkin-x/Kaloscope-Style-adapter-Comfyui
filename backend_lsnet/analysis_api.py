"""API schemas and handlers for the shared feature/analysis pipeline."""
import base64
from io import BytesIO
from typing import Optional, List

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import torch

from backend_lsnet.analysis import (extract_batch, cache_bytes, read_cache, thumbnail_batch,
                                   analyze_cached, feature_tools, output_layout)
from backend_lsnet.model_paths import get_checkpoint_path
from model_loading import load_model_bundle


class AnalysisOptions(BaseModel):
    metric: str = 'cosine'
    normalize: bool = True
    cluster_method: str = 'kmeans'
    n_clusters: int = Field(3, ge=1, le=512)
    top_k: int = Field(2, ge=1, le=511)
    reference_index: int = Field(0, ge=0, le=511)
    tensor_layout: str = 'auto'
    layer_index: int = -1
    layer_pooling: str = 'selected'
    token_pooling: str = 'mean'
    labels: List[str] = Field(default_factory=list)
    seed: int = Field(42, ge=0, le=2147483647)
    perplexity: float = Field(5., gt=0)
    dbscan_eps: float = Field(.35, gt=0)
    dbscan_min_samples: int = Field(2, ge=1)
    max_dimensions: int = Field(32, ge=1, le=128)
    heatmap_order: str = 'cluster'
    grid_width: int = Field(0, ge=0)
    width: int = Field(1400, ge=512, le=4096)
    height: int = Field(1000, ge=512, le=4096)


class FeaturesRequest(BaseModel):
    input_images: List[str] = Field(..., **({'min_length': 1, 'max_length': 512} if hasattr(BaseModel, 'model_dump')
                                         else {'min_items': 1, 'max_items': 512}))
    model_name: str = 'Kaloscope'
    device: str = 'cuda'
    output_type: str = 'default'
    layers: str = '-1'
    intermediate_norm: bool = True
    batch_size: int = Field(2, ge=1, le=512)
    labels: List[str] = Field(default_factory=list)


class CachedFeaturesRequest(BaseModel):
    cache_base64: Optional[str] = None
    features: Optional[list] = None
    output_type: str = 'default'
    labels: List[str] = Field(default_factory=list)


class AnalysisRequest(CachedFeaturesRequest):
    image_batch: Optional[FeaturesRequest] = None
    thumbnail_images: List[str] = Field(default_factory=list)
    chart_type: str = 'relationship_graph'
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class FeatureToolsRequest(CachedFeaturesRequest):
    operation: str = 'common_features'
    reference_index: int = Field(0, ge=0)
    groups: Optional[List[str]] = None
    tensor_layout: str = 'auto'
    layer_index: int = -1
    layer_pooling: str = 'selected'
    token_pooling: str = 'mean'


def values(model):
    return model.model_dump() if hasattr(model, 'model_dump') else model.dict()


def extract_request(req, decode):
    images = [decode(value) for value in req.input_images]
    bundle = load_model_bundle(checkpoint=get_checkpoint_path(req.model_name), device=req.device)
    features = extract_batch(images, bundle, req.output_type, req.layers, req.intermediate_norm, req.batch_size)
    labels = req.labels or [f'Image {index + 1:02d}' for index in range(len(images))]
    if len(labels) != len(images):
        raise ValueError('labels must match the image count')
    return {'features': features, 'labels': labels, 'output_type': req.output_type, 'layers': req.layers}, images


def cached_request(req):
    if (req.cache_base64 is None) == (req.features is None):
        raise ValueError('Provide exactly one of cache_base64 or features')
    if req.cache_base64 is not None:
        return read_cache(BytesIO(base64.b64decode(req.cache_base64, validate=True)))
    features = torch.tensor(req.features, dtype=torch.float32)
    if features.ndim < 2:
        raise ValueError('features must have an image batch dimension')
    labels = req.labels or [f'Image {index + 1:02d}' for index in range(len(features))]
    if len(labels) != len(features):
        raise ValueError('labels must match features')
    return {'features': features, 'labels': labels, 'output_type': req.output_type, 'layers': '-1'}


def serialized_cache(cached):
    return {'cache_base64': base64.b64encode(cache_bytes(cached['features'], cached['labels'], cached['output_type'], cached['layers'])).decode(),
            'shape': list(cached['features'].shape), 'labels': cached['labels'], 'output_type': cached['output_type']}


def analysis_request(req, decode):
    if req.image_batch is not None:
        if req.features is not None or req.cache_base64 is not None:
            raise ValueError('image_batch cannot be combined with features/cache_base64')
        cached, images = extract_request(req.image_batch, decode)
    else:
        cached = cached_request(req)
        images = [decode(value) for value in req.thumbnail_images]
    image, report, distances = analyze_cached(cached, req.chart_type,
                                             images=thumbnail_batch(images) if images else None, **values(req.options))
    png = BytesIO()
    Image.fromarray((image[0].numpy().clip(0, 1) * 255).round().astype(np.uint8)).save(png, format='PNG')
    import json
    return {'image_base64': base64.b64encode(png.getvalue()).decode(), 'analysis': json.loads(report),
            'distance_matrix': distances.tolist(), **serialized_cache(cached)}


def tools_request(req):
    cached = cached_request(req)
    layout = output_layout(cached['output_type']) if req.tensor_layout == 'auto' else req.tensor_layout
    return feature_tools(cached['features'], req.operation, req.reference_index, req.groups,
                         tensor_layout=layout, layer_index=req.layer_index, layer_pooling=req.layer_pooling, token_pooling=req.token_pooling)
