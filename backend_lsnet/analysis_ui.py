"""The same analysis tab is mounted by WebUI and the independent Gradio app."""
import json
from functools import wraps
from pathlib import Path
import tempfile

import gradio as gr
from PIL import Image

from backend_lsnet.analysis import (extract_batch, thumbnail_batch, cache_bytes, read_cache,
                                   analyze_cached, save_analysis, feature_tools, output_layout)
from backend_lsnet.model_paths import get_available_models, get_checkpoint_path
from feature_analysis import CHART_TYPES, TENSOR_LAYOUTS
from model_loading import FEATURE_OUTPUTS, load_model_bundle


def ui_errors(callback):
    @wraps(callback)
    def wrapped(*args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except (ValueError, FileNotFoundError, TypeError, KeyError) as error:
            raise gr.Error(str(error)) from error
    return wrapped


def file_path(value):
    return str(value) if isinstance(value, (str, Path)) else value.name


@ui_errors
def extract_uploaded(files, model_name, device, output_type, layers, intermediate_norm, batch_size):
    if not files:
        raise gr.Error('请先上传图片。')
    if len(files) > 512:
        raise gr.Error('最多支持 512 张图片。')
    images, labels = [], []
    for item in files:
        with Image.open(file_path(item)) as image:
            images.append(image.copy())
        labels.append(Path(file_path(item)).name)
    bundle = load_model_bundle(checkpoint=get_checkpoint_path(model_name), device=device)
    features = extract_batch(images, bundle, output_type, layers, intermediate_norm, int(batch_size))
    cached = {'features': features, 'labels': labels, 'output_type': output_type, 'layers': layers,
              'images': thumbnail_batch(images)}
    directory = Path(tempfile.mkdtemp(prefix='kaloscope-features-'))
    path = directory / 'features.npz'
    path.write_bytes(cache_bytes(features, labels, output_type, layers))
    return cached, f'已缓存 {len(images)} 张图片，TENSOR {list(features.shape)}。可反复制图，无需再推理。', str(path)


@ui_errors
def import_uploaded_cache(file):
    if not file:
        raise gr.Error('请选择 features.npz。')
    cached = read_cache(file_path(file))
    return cached, f'已导入 {len(cached["labels"])} 张图片的缓存，TENSOR {list(cached["features"].shape)}。未加载模型。'


@ui_errors
def plot_uploaded_cache(cached, chart_type, *values):
    if cached is None:
        raise gr.Error('先提取特征或导入缓存。')
    options = dict(zip(ANALYSIS_OPTION_NAMES, values))
    image, report, distances = analyze_cached(cached, chart_type, images=cached.get('images'), **options)
    directory = tempfile.mkdtemp(prefix='kaloscope-analysis-')
    files = save_analysis(directory, chart_type, image, report, distances)
    return image[0].numpy(), report, files


@ui_errors
def tools_uploaded_cache(cached, operation, reference_index, groups, tensor_layout, layer_index, layer_pooling, token_pooling):
    if cached is None:
        raise gr.Error('先提取特征或导入缓存。')
    layout = output_layout(cached['output_type']) if tensor_layout == 'auto' else tensor_layout
    result = feature_tools(cached['features'], operation, int(reference_index), groups.splitlines() if groups.strip() else None,
                           tensor_layout=layout, layer_index=int(layer_index), layer_pooling=layer_pooling, token_pooling=token_pooling)
    return json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)


ANALYSIS_OPTION_NAMES = (
    'metric', 'normalize', 'cluster_method', 'n_clusters', 'top_k', 'reference_index',
    'tensor_layout', 'layer_index', 'layer_pooling', 'token_pooling', 'labels', 'seed',
    'perplexity', 'dbscan_eps', 'dbscan_min_samples', 'max_dimensions', 'heatmap_order',
    'grid_width', 'width', 'height',
)


def build_analysis_tab():
    """Call within a Blocks/Tabs context; no ComfyUI or WebUI dependencies."""
    with gr.TabItem('Features & Analysis'):
        state = gr.State(value=None)
        gr.Markdown('### 批量特征与分析\n先提取一次特征，再选择图表反复制图。也可以导入 `.npz` 缓存，无需加载模型。')
        with gr.Row():
            with gr.Column():
                inputs = gr.File(label='图片批次（顺序以缓存 labels 为准）', file_count='multiple', file_types=['image'])
                available = get_available_models()
                model = gr.Dropdown(choices=available, value=(available or [None])[0], label='Model Folder')
                refresh = gr.Button('刷新模型列表')
                device = gr.Dropdown(['cuda', 'cpu'], value='cuda', label='Device')
                output_type = gr.Dropdown(list(FEATURE_OUTPUTS), value='default', label='Feature Output')
                layers = gr.Textbox(value='-1', label='Intermediate layers（逗号分隔 block 编号）')
                norm = gr.Checkbox(value=True, label='Intermediate LayerNorm')
                batch = gr.Number(value=2, precision=0, label='Inference batch size')
                extract = gr.Button('提取并缓存特征')
            with gr.Column():
                cache_input = gr.File(label='导入 features.npz', file_types=['.npz'])
                load = gr.Button('导入缓存（不推理）')
                status = gr.Textbox(label='缓存状态', interactive=False)
                cache_download = gr.File(label='下载特征缓存')
        chart = gr.Dropdown(list(CHART_TYPES), value='relationship_graph', label='Chart Type')
        with gr.Accordion('分析参数', open=False):
            with gr.Row():
                metric = gr.Dropdown(['cosine', 'euclidean', 'manhattan'], value='cosine', label='Distance metric')
                normalize = gr.Checkbox(value=True, label='L2 normalize')
                cluster = gr.Dropdown(['kmeans', 'agglomerative', 'dbscan', 'none'], value='kmeans', label='Clustering')
                clusters = gr.Number(value=3, precision=0, label='Cluster count')
            with gr.Row():
                top_k = gr.Number(value=2, precision=0, label='Neighbor count')
                reference = gr.Number(value=0, precision=0, label='Query image index (0-based)')
                layout = gr.Dropdown(list(TENSOR_LAYOUTS), value='auto', label='Tensor layout')
                layer_index = gr.Number(value=-1, precision=0, label='Input layer position')
            with gr.Row():
                layer_pooling = gr.Dropdown(['selected', 'mean'], value='selected', label='Layer pooling')
                token_pooling = gr.Dropdown(['mean', 'flatten'], value='mean', label='Token pooling')
                seed = gr.Number(value=42, precision=0, label='Seed')
                perplexity = gr.Number(value=5.0, label='t-SNE perplexity')
            with gr.Row():
                eps = gr.Number(value=.35, label='DBSCAN eps')
                min_samples = gr.Number(value=2, precision=0, label='DBSCAN min samples')
                dimensions = gr.Number(value=32, precision=0, label='Max dimensions')
                order = gr.Dropdown(['cluster', 'input'], value='cluster', label='Heatmap order')
            with gr.Row():
                grid_width = gr.Number(value=0, precision=0, label='Patch grid columns (0=auto)')
                width = gr.Number(value=1400, precision=0, label='Width (px)')
                height = gr.Number(value=1000, precision=0, label='Height (px)')
            labels = gr.Textbox(value='', lines=3, label='Labels（每行一个，留空使用文件名）')
        plot = gr.Button('从缓存生成图表')
        visualization = gr.Image(label='Visualization', interactive=False)
        report = gr.Textbox(label='Analysis JSON', lines=12, interactive=False)
        downloads = gr.File(label='下载 PNG / JSON / 距离矩阵 CSV', file_count='multiple')
        with gr.Accordion('共同特征 / 相似度 / 分组比较', open=False):
            operation = gr.Dropdown(['common_features', 'similarity', 'compare_groups'], value='common_features', label='Operation')
            groups = gr.Textbox(value='', lines=4, label='Group names（分组比较时，每张图片一个组名）')
            run_tools = gr.Button('分析缓存特征')
            tools_result = gr.Textbox(label='Feature result JSON', lines=8, interactive=False)
        controls = [metric, normalize, cluster, clusters, top_k, reference, layout, layer_index,
                    layer_pooling, token_pooling, labels, seed, perplexity, eps, min_samples, dimensions,
                    order, grid_width, width, height]
        extract.click(extract_uploaded, [inputs, model, device, output_type, layers, norm, batch], [state, status, cache_download])
        load.click(import_uploaded_cache, [cache_input], [state, status])
        plot.click(plot_uploaded_cache, [state, chart, *controls], [visualization, report, downloads])
        run_tools.click(tools_uploaded_cache, [state, operation, reference, groups, layout, layer_index, layer_pooling, token_pooling], [tools_result])
        refresh.click(lambda: gr.update(choices=get_available_models(), value=(get_available_models() or [None])[0]), [], [model])
