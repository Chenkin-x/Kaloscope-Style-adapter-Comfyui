"""CPU feature analysis and plotting. No model loading or inference is performed."""
import json
import math
from threading import RLock

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_samples
import torch

CHART_TYPES = (
    'relationship_graph', 'distance_heatmap', 'similarity_heatmap',
    'pca_scatter', 'mds_scatter', 'tsne_scatter', 'dendrogram', 'nearest_neighbors',
    'distance_distribution', 'silhouette', 'cluster_sizes', 'pca_variance',
    'feature_statistics', 'feature_heatmap', 'dimension_correlation',
    'cluster_centroid_heatmap', 'outlier_scores', 'patch_energy',
)
TENSOR_LAYOUTS = ('auto', 'vectors', 'tokens', 'spatial', 'layer_vectors',
                  'layer_tokens', 'layer_spatial', 'flatten')
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#666666')
MARKERS = ('o', '^', 's', 'D', 'P', 'X', 'v')
_PLOT_LOCK = RLock()


def prepare_features(features, tensor_layout='auto', layer_index=-1, layer_pooling='selected',
                     token_pooling='mean'):
    """Reduce tokens/maps explicitly; first dimension always identifies images."""
    if not isinstance(features, torch.Tensor) or features.ndim < 2:
        raise ValueError('features must be a TENSOR with a batch dimension, e.g. [B,D]')
    if any(size == 0 for size in features.shape):
        raise ValueError('Feature tensor contains an empty dimension')
    if features.shape[0] > 512:
        raise ValueError('Analysis supports at most 512 images per graph; split the batch explicitly')
    data = features.detach().float().cpu().numpy().astype(np.float64)
    if not np.isfinite(data).all():
        raise ValueError('Features contain NaN or infinity')
    original_shape = list(data.shape)
    layout = tensor_layout
    if layout == 'auto':
        if data.ndim == 4:
            raise ValueError('4D tensor is ambiguous: choose spatial [B,D,H,W] or layer_tokens [B,L,N,D]')
        layout = {2: 'vectors', 3: 'tokens', 5: 'layer_spatial'}.get(data.ndim)
        if layout is None:
            raise ValueError('Choose an explicit tensor_layout for this tensor')
    if layout not in TENSOR_LAYOUTS:
        raise ValueError(f'Unknown tensor layout: {layout}')
    ranks = {'vectors': 2, 'tokens': 3, 'spatial': 4, 'layer_vectors': 3,
             'layer_tokens': 4, 'layer_spatial': 5}
    if layout != 'flatten' and data.ndim != ranks[layout]:
        raise ValueError(f'{layout} requires {ranks[layout]} dimensions; received {data.shape}')
    transformations = [f'input layout: {layout}']
    if layout.startswith('layer_'):
        if layer_pooling == 'mean':
            data = data.mean(axis=1)
            transformations.append('mean over selected input layers')
        elif layer_pooling == 'selected':
            count = data.shape[1]
            index = layer_index + count if layer_index < 0 else layer_index
            if not 0 <= index < count:
                raise ValueError(f'layer_index is outside the input tensor with {count} layers')
            data = data[:, index]
            transformations.append(f'select input layer position {index}')
        else:
            raise ValueError('layer_pooling must be selected or mean')
        layout = layout.removeprefix('layer_')
    patches = None
    grid = None
    if layout == 'tokens':
        patches = data
    elif layout == 'spatial':
        grid = data.shape[-2:]
        patches = data.reshape(data.shape[0], data.shape[1], -1).transpose(0, 2, 1)
    if patches is not None:
        if token_pooling == 'mean':
            vectors = patches.mean(axis=1)
            transformations.append('mean over spatial/token positions')
        elif token_pooling == 'flatten':
            vectors = data.reshape(data.shape[0], -1)
            transformations.append('flatten spatial/token positions in original tensor order')
        else:
            raise ValueError('token_pooling must be mean or flatten')
    else:
        vectors = data.reshape(data.shape[0], -1)
    return vectors, patches, grid, {'input_shape': original_shape, 'tensor_layout': tensor_layout,
                                    'vector_shape': list(vectors.shape), 'transformations': transformations}


def parse_labels(labels, count):
    if not labels:
        return [f'Image {index + 1:02d}' for index in range(count)]
    if isinstance(labels, str):
        stripped = labels.strip()
        labels = json.loads(stripped) if stripped.startswith('[') else stripped.splitlines()
    if not isinstance(labels, (tuple, list)) or len(labels) != count:
        raise ValueError(f'labels must contain exactly {count} names, in feature batch order')
    return [str(label) for label in labels]


def _thumbnails(images, count):
    if images is None:
        return None
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] != count or images.shape[-1] not in (3, 4):
        raise ValueError('images must be a ComfyUI IMAGE batch [B,H,W,3/4] matching features')
    array = images.detach().float().cpu().numpy()
    if not np.isfinite(array).all():
        raise ValueError('Thumbnail images contain NaN or infinity')
    return [Image.fromarray((sample.clip(0, 1) * 255).astype(np.uint8)).convert('RGB') for sample in array]


def _classical_mds(distances):
    count = len(distances)
    center = np.eye(count) - np.ones((count, count)) / count
    values, vectors = np.linalg.eigh(-0.5 * center @ (distances ** 2) @ center)
    order = np.argsort(values)[::-1]
    positive = np.maximum(values[order[:2]], 0)
    coordinates = vectors[:, order[:2]] * np.sqrt(positive)
    coordinates = np.pad(coordinates, ((0, 0), (0, max(0, 2 - coordinates.shape[1]))))
    approximation = pairwise_distances(coordinates)
    denominator = np.square(distances).sum()
    stress = math.sqrt(np.square(approximation - distances).sum() / denominator) if denominator > 0 else 0.0
    return coordinates, {'method': 'classical MDS', 'normalized_distance_error': stress,
                         'negative_eigenvalue_mass': float(np.abs(values[values < -1e-10]).sum())}


def _pca(vectors):
    if len(vectors) < 2 or not np.any(np.std(vectors, axis=0) > 1e-12):
        return np.zeros((len(vectors), 2)), np.zeros(min(vectors.shape))
    estimator = PCA(n_components=min(vectors.shape), svd_solver='full')
    result = estimator.fit_transform(vectors)
    coordinates = np.pad(result[:, :2], ((0, 0), (0, max(0, 2 - result.shape[1]))))
    return coordinates, estimator.explained_variance_ratio_


def _clusters(vectors, distances, method, n_clusters, seed, eps, min_samples, warnings):
    count = len(vectors)
    if method == 'none':
        return np.zeros(count, dtype=int), {'method': 'none', 'n_clusters': 1}
    if method == 'dbscan':
        groups = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(distances)
        return groups, {'method': 'DBSCAN', 'metric': 'selected pairwise distance', 'eps': eps,
                        'min_samples': min_samples, 'noise_label': -1}
    distinct = len(np.unique(vectors, axis=0))
    actual = min(n_clusters, count, distinct)
    if actual != n_clusters:
        warnings.append(f'n_clusters reduced from {n_clusters} to {actual}: only {count} samples / {distinct} distinct vectors')
    if actual == 1:
        return np.zeros(count, dtype=int), {'method': method, 'n_clusters': 1}
    if method == 'kmeans':
        groups = KMeans(n_clusters=actual, random_state=seed, n_init=10).fit_predict(vectors)
        details = {'method': 'KMeans', 'n_clusters': actual, 'objective_metric': 'euclidean',
                   'note': 'KMeans uses Euclidean vectors even when graph distances use cosine/manhattan'}
    elif method == 'agglomerative':
        # Use sklearn to cut to exactly the requested cluster count.
        groups = AgglomerativeClustering(n_clusters=actual, metric='precomputed', linkage='average').fit_predict(distances)
        details = {'method': 'agglomerative', 'n_clusters': actual, 'metric': 'selected distance', 'linkage': 'average'}
    else:
        raise ValueError(f'Unknown clustering method: {method}')
    return groups, details


def _short(label):
    return label if len(label) <= 26 else label[:23] + '...'


def _color(group):
    return '#777777' if group < 0 else COLORS[group % len(COLORS)]


def _gallery(fig, grid, thumbnails, labels):
    cells = grid.subgridspec(math.ceil(len(thumbnails) / 2), 2, hspace=0.25, wspace=0.12)
    for index, thumbnail in enumerate(thumbnails):
        axis = fig.add_subplot(cells[index // 2, index % 2])
        preview = thumbnail.copy()
        preview.thumbnail((180, 150))
        axis.imshow(preview)
        title = _short(labels[index])
        if not title.startswith(f'{index + 1:02d}'):
            title = f'{index + 1:02d} ' + title
        axis.set_title(title, fontsize=8, pad=3)
        axis.axis('off')


def _main_axes(fig, thumbnails=None, labels=None):
    if thumbnails is not None and len(thumbnails) <= 16:
        grid = fig.add_gridspec(1, 2, width_ratios=[3.6, 1.4], wspace=0.12)
        _gallery(fig, grid[1], thumbnails, labels)
        return fig.add_subplot(grid[0])
    return fig.add_subplot(111)


def _style_axis(axis):
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    axis.grid(alpha=0.15)
    axis.set_axisbelow(True)


def _scatter(axis, coordinates, groups, labels):
    for group in sorted(set(groups)):
        selected = groups == group
        axis.scatter(coordinates[selected, 0], coordinates[selected, 1], s=110,
                     color=_color(group), marker=MARKERS[group % len(MARKERS)],
                     edgecolors='white', linewidths=1.0, label='Noise' if group < 0 else f'Cluster {group + 1}', zorder=3)
    for index, coordinate in enumerate(coordinates):
        axis.annotate(_short(labels[index]), coordinate, xytext=(7, 7), textcoords='offset points', fontsize=9,
                      bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none', 'pad': 1}, zorder=4)
    axis.legend(loc='best', frameon=False, fontsize=9)
    axis.margins(0.25)
    axis.set_aspect('equal', adjustable='datalim')
    _style_axis(axis)


def _matrix(fig, axis, values, labels, label, cmap, vmin=None, vmax=None):
    image = axis.imshow(values, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax, aspect='auto')
    axis.set_xticks(range(len(labels)), [_short(name) for name in labels], rotation=45, ha='right', fontsize=9)
    axis.set_yticks(range(len(labels)), [_short(name) for name in labels], fontsize=9)
    fig.colorbar(image, ax=axis, shrink=0.8, label=label)
    if len(labels) <= 14:
        lower, upper = image.get_clim()
        for row in range(len(labels)):
            for col in range(len(labels)):
                value = values[row, col]
                text_color = 'white' if (value - lower) / (upper - lower or 1) < 0.45 else '#152838'
                axis.text(col, row, f'{value:.2f}', ha='center', va='center', fontsize=8, color=text_color)


def create_analysis(features, chart_type='relationship_graph', metric='cosine', normalize=True,
                    cluster_method='kmeans', n_clusters=3, top_k=2, reference_index=0,
                    tensor_layout='auto', layer_index=-1, layer_pooling='selected', token_pooling='mean',
                    labels='', images=None, seed=42, perplexity=5.0, dbscan_eps=0.35, dbscan_min_samples=2,
                    max_dimensions=32, heatmap_order='cluster', grid_width=0, width=1400, height=1000):
    """Return a Figure, JSON-safe numeric report, and the original-space distance TENSOR."""
    if chart_type not in CHART_TYPES:
        raise ValueError(f'Unknown chart type: {chart_type}')
    if metric not in ('cosine', 'euclidean', 'manhattan'):
        raise ValueError(f'Unknown distance metric: {metric}')
    if top_k < 1 or n_clusters < 1 or max_dimensions < 1 or perplexity <= 0 or dbscan_eps <= 0 or dbscan_min_samples < 1:
        raise ValueError('Cluster count, top_k, dimensions, perplexity and DBSCAN parameters must be positive')
    if width < 512 or height < 512 or width > 4096 or height > 4096:
        raise ValueError('Plot dimensions must be between 512 and 4096 pixels')
    raw, patches, grid, preprocessing = prepare_features(features, tensor_layout, layer_index, layer_pooling, token_pooling)
    count = len(raw)
    if not 0 <= reference_index < count:
        raise ValueError(f'reference_index must be between 0 and {count - 1}')
    names = parse_labels(labels, count)
    thumbnails = _thumbnails(images, count)
    warnings = []
    norms = np.linalg.norm(raw, axis=1)
    if metric == 'cosine' and np.any(norms <= 1e-12):
        raise ValueError('Cosine distance is undefined for zero-norm features; choose Euclidean or fix the inputs')
    vectors = raw / np.maximum(norms[:, None], 1e-12) if normalize else raw.copy()
    distances = pairwise_distances(vectors, metric=metric)
    distances = np.maximum((distances + distances.T) / 2, 0)
    np.fill_diagonal(distances, 0)
    unit = raw / np.maximum(norms[:, None], 1e-12)
    similarities = (unit @ unit.T).clip(-1, 1)
    valid_cosine = np.outer(norms > 1e-12, norms > 1e-12)
    groups, clustering = _clusters(vectors, distances, cluster_method, n_clusters, seed, dbscan_eps, dbscan_min_samples, warnings)
    pca_coordinates, variance = _pca(vectors)
    mds_coordinates, mds_details = _classical_mds(distances)
    k = min(top_k, max(0, count - 1))
    neighbors = []
    for index in range(count):
        order = sorted((other for other in range(count) if other != index), key=lambda other: (distances[index, other], other))[:k]
        neighbors.append([{'index': other, 'label': names[other], 'distance': float(distances[index, other]),
                           'cosine_similarity': float(similarities[index, other]) if valid_cosine[index, other] else None} for other in order])
    offdiag = distances[np.triu_indices(count, 1)]
    centroid = vectors.mean(0)
    centered = vectors - centroid
    singular = np.linalg.svd(centered, compute_uv=False)
    mass = singular ** 2
    probabilities = mass / mass.sum() if mass.sum() > 1e-20 else np.zeros_like(mass)
    effective_rank = math.exp(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))) if np.any(probabilities) else 0.0
    ordering = np.arange(count)
    tree = linkage(squareform(distances, checks=False), method='average', optimal_ordering=True) if count > 1 else None
    if heatmap_order == 'cluster' and tree is not None:
        ordering = leaves_list(tree)
    elif heatmap_order not in ('input', 'cluster'):
        raise ValueError('heatmap_order must be input or cluster')
    non_noise = groups >= 0
    silhouette = None
    selected_groups = groups[non_noise]
    if 1 < len(set(selected_groups)) < len(selected_groups):
        silhouette = np.full(count, np.nan)
        silhouette[non_noise] = silhouette_samples(distances[np.ix_(non_noise, non_noise)], selected_groups, metric='precomputed')
    selected_dimensions = np.argsort(np.var(vectors, axis=0), kind='stable')[::-1][:max_dimensions]
    selected_dimensions.sort()
    report = {
        'chart_type': chart_type, 'sample_count': count, 'labels': names, 'preprocessing': preprocessing,
        'distance_metric': metric, 'normalize_vectors': bool(normalize), 'normalization': 'row L2' if normalize else 'none',
        'seed': seed, 'clustering': clustering, 'cluster_labels': groups.tolist(),
        'distances': distances.tolist(), 'cosine_similarities': [[float(similarities[i,j]) if valid_cosine[i,j] else None
            for j in range(count)] for i in range(count)], 'nearest_neighbors': neighbors,
        'summary': {'mean_pairwise_distance': float(offdiag.mean()) if offdiag.size else None,
                    'min_pairwise_distance': float(offdiag.min()) if offdiag.size else None,
                    'max_pairwise_distance': float(offdiag.max()) if offdiag.size else None,
                    'effective_rank_centered': effective_rank,
                    'effective_rank_definition': 'exp(entropy(normalized squared singular values of centered vectors))',
                    'raw_feature_norms': norms.tolist()},
        'pca_explained_variance_ratio': variance.tolist(), 'mds': mds_details,
        'silhouette_scores': [float(value) if np.isfinite(value) else None for value in silhouette] if silhouette is not None else None,
        'selected_dimension_indices': selected_dimensions.tolist(), 'heatmap_sample_order': ordering.tolist(), 'warnings': warnings,
    }
    plot_data = dict(raw=raw, vectors=vectors, patches=patches, grid=grid, names=names, thumbnails=thumbnails,
                     distances=distances, similarities=similarities, groups=groups, pca=pca_coordinates, mds=mds_coordinates,
                     variance=variance, ordering=ordering, tree=tree, silhouette=silhouette, dimensions=selected_dimensions,
                     neighbors=neighbors, offdiag=offdiag, norms=norms, valid_cosine=valid_cosine)
    with _PLOT_LOCK, mpl.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Microsoft YaHei', 'Noto Sans CJK SC'],
                                   'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                                   'axes.labelcolor': '#243746', 'text.color': '#243746', 'figure.facecolor': 'white',
                                   'axes.facecolor': 'white', 'savefig.facecolor': 'white'}):
        fig = Figure(figsize=(width / 120, height / 120), dpi=120, layout='constrained')
        FigureCanvasAgg(fig)
        _draw_chart(fig, chart_type, plot_data, report, reference_index, seed, perplexity, grid_width)
        subtitle = f'{count} images | {metric} distance | ' + ('L2-normalized vectors' if normalize else 'raw vectors')
        if chart_type in ('feature_statistics', 'patch_energy'):
            subtitle = f'{count} images | raw input statistics (before vector normalization)'
        fig.suptitle(chart_type.replace('_', ' ').title(), fontsize=20, fontweight='bold', x=0.03, ha='left')
        fig.supxlabel(subtitle, fontsize=10, color='#536575')
        fig.canvas.draw()
    return fig, report, torch.from_numpy(distances.astype(np.float32))


def figure_tensor(fig):
    """Render a figure directly into a ComfyUI IMAGE without a file roundtrip."""
    with _PLOT_LOCK:
        if fig.stale or not hasattr(fig.canvas, 'renderer'):
            fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    return torch.from_numpy(rgb).float().unsqueeze(0) / 255


def analyze_features(features, **kwargs):
    fig, report, distances = create_analysis(features, **kwargs)
    image = figure_tensor(fig)
    fig.clear()
    return image, json.dumps(report, ensure_ascii=False, allow_nan=False), distances


def _draw_chart(fig, chart, data, report, reference, seed, perplexity, grid_width):
    names, groups = data['names'], data['groups']
    count = len(names)
    distances = data['distances']
    ordered = data['ordering']
    if chart in ('relationship_graph', 'pca_scatter', 'mds_scatter', 'tsne_scatter'):
        axis = _main_axes(fig, data['thumbnails'], names)
        if chart == 'pca_scatter':
            coordinates = data['pca']
            ratio = np.pad(data['variance'], (0, max(0, 2 - len(data['variance']))))
            axis.set(xlabel=f'PC1 ({ratio[0]:.1%} variance)', ylabel=f'PC2 ({ratio[1]:.1%} variance)')
            axis.set_title('PCA projection; cluster labels computed in original feature space', fontsize=11)
            details = {'method': 'PCA', 'explained_variance_ratio_2d': ratio[:2].tolist()}
        elif chart == 'tsne_scatter':
            if count < 3:
                raise ValueError('t-SNE requires at least 3 images for this visualization')
            actual_perplexity = min(perplexity, count - 1)
            if actual_perplexity != perplexity:
                report['warnings'].append(f't-SNE perplexity reduced to {actual_perplexity} for {count} samples')
            if not distances.any():
                coordinates = np.zeros((count, 2))
                kl = 0.0
                report['warnings'].append('All feature distances are zero; t-SNE represented as coincident points')
            else:
                estimator = TSNE(n_components=2, perplexity=actual_perplexity, metric='precomputed',
                                 init='random', learning_rate='auto', random_state=seed)
                coordinates = estimator.fit_transform(distances)
                kl = float(estimator.kl_divergence_)
            details = {'method': 't-SNE', 'perplexity': actual_perplexity, 'kl_divergence': kl}
            axis.set(xlabel='t-SNE axis 1 (arbitrary units)', ylabel='t-SNE axis 2 (arbitrary units)')
            axis.set_title('Neighborhood visualization; 2D distances and cluster gaps are not original distances', fontsize=10)
        else:
            coordinates = data['mds']
            details = report['mds']
            axis.set(xlabel='MDS axis 1', ylabel='MDS axis 2')
            error = details['normalized_distance_error']
            axis.set_title(f'Approximate distance layout | relative distance error = {error:.3f}', fontsize=11)
        report['projection'] = {**details, 'coordinates': coordinates.tolist()}
        if chart == 'relationship_graph':
            edges = sorted({tuple(sorted((index, neighbor['index'])))
                            for index, row in enumerate(data['neighbors']) for neighbor in row})
            report['edges'] = [{'source': left, 'target': right, 'distance': float(distances[left, right])}
                               for left, right in edges]
            for left, right in edges:
                segment = coordinates[[left, right]]
                within = groups[left] == groups[right] and groups[left] >= 0
                delta = segment[1]-segment[0]
                curvature = 0.0
                if within and abs(delta[0]) > 5*abs(delta[1]):
                    members = sorted(np.flatnonzero(groups==groups[left]),key=lambda i:coordinates[i,0])
                    if abs(members.index(left)-members.index(right)) > 1:
                        curvature = 0.65
                edge = FancyArrowPatch(segment[0], segment[1], arrowstyle='-', connectionstyle=f'arc3,rad={curvature}',
                    color=_color(groups[left]) if within else '#98A4AD', linewidth=1.6, alpha=0.65,
                    linestyle='-' if within else '--', zorder=1)
                axis.add_patch(edge)
                if count <= 16:
                    midpoint = segment.mean(axis=0) + curvature/2*np.array([delta[1],-delta[0]])
                    axis.text(*midpoint, f'{distances[left,right]:.3f}', fontsize=8, color='#536575',
                              ha='center', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 1})
            axis.set_title(axis.get_title() + '\nUnion kNN graph; edge numbers are original-space distances', fontsize=10)
        plot_labels = [f'{index+1:02d}' for index in range(count)] if data['thumbnails'] is not None and count<=16 else names
        _scatter(axis, coordinates, groups, plot_labels)
    elif chart in ('distance_heatmap', 'similarity_heatmap'):
        axis = _main_axes(fig)
        if chart == 'similarity_heatmap':
            if not data['valid_cosine'].all():
                raise ValueError('Cosine similarity heatmap requires nonzero feature vectors')
            values = data['similarities'][np.ix_(ordered, ordered)]
            _matrix(fig, axis, values, [names[i] for i in ordered], 'Cosine similarity', 'coolwarm', -1, 1)
        else:
            values = distances[np.ix_(ordered, ordered)]
            _matrix(fig, axis, values, [names[i] for i in ordered], report['distance_metric'] + ' distance', 'viridis_r', 0)
        axis.set_title('Average-linkage order' if report['heatmap_sample_order'] != list(range(count)) else 'Input order', fontsize=11)
    elif chart == 'dendrogram':
        axis = _main_axes(fig)
        if data['tree'] is None:
            axis.text(0.5, 0.5, 'At least 2 images are required for a dendrogram', ha='center', transform=axis.transAxes)
            axis.axis('off')
        else:
            dendrogram(data['tree'], labels=[_short(name) for name in names], ax=axis,
                       leaf_rotation=35, leaf_font_size=9, above_threshold_color='#0072B2', color_threshold=0)
            report['linkage_matrix'] = data['tree'].tolist()
            axis.set(xlabel='Images', ylabel=report['distance_metric'] + ' distance')
            axis.set_title('Hierarchical relationships | average linkage', fontsize=12)
            _style_axis(axis)
    elif chart == 'nearest_neighbors':
        axis = _main_axes(fig, data['thumbnails'], names)
        row = data['neighbors'][reference]
        values = [item['distance'] for item in row]
        bars = axis.barh(range(len(row)), values, color=[_color(groups[item['index']]) for item in row])
        axis.set_yticks(range(len(row)), [_short(item['label']) for item in row])
        axis.invert_yaxis()
        for bar, value in zip(bars, values):
            axis.annotate(f'{value:.4f}', (value, bar.get_y() + bar.get_height()/2), xytext=(5, 0),
                          textcoords='offset points', va='center', fontsize=10)
        axis.set_xlim(0, max(values + [1e-9]) * 1.25)
        axis.set(xlabel=report['distance_metric'] + ' distance (lower = closer)', ylabel='Nearest images')
        axis.set_title('Query: ' + _short(names[reference]), fontsize=12)
        report['reference_index'] = reference
        _style_axis(axis)
    elif chart == 'distance_distribution':
        axis = _main_axes(fig)
        values = data['offdiag']
        if not len(values):
            raise ValueError('Pairwise distance distribution requires at least 2 images')
        bins = min(30, max(4, math.ceil(math.sqrt(len(values)))))
        counts, edges, _ = axis.hist(values, bins=bins, color='#0072B2', edgecolor='white', alpha=0.9)
        for pair, color, title in ((True, '#009E73', 'Within cluster'), (False, '#D55E00', 'Between clusters')):
            subset = [distances[i,j] for i in range(count) for j in range(i+1,count)
                      if groups[i] >= 0 and groups[j] >= 0 and (groups[i] == groups[j]) == pair]
            if subset:
                axis.axvline(np.mean(subset), color=color, linestyle='--' if pair else ':',
                             linewidth=2, label=f'{title} mean: {np.mean(subset):.3f}')
        report['histogram'] = {'bin_edges': edges.tolist(), 'counts': counts.astype(int).tolist(),
                               'pairwise_distances': values.tolist()}
        axis.legend(frameon=False)
        axis.set(xlabel=report['distance_metric'] + ' distance', ylabel='Number of unordered pairs')
        axis.set_title('Each unordered pair counted once; diagonal excluded', fontsize=11)
        _style_axis(axis)
    elif chart == 'silhouette':
        axis = _main_axes(fig)
        values = data['silhouette']
        if values is None:
            axis.text(0.5, 0.5, 'Silhouette unavailable\nRequires 2 to N-1 non-noise clusters',
                      transform=axis.transAxes, ha='center', va='center', fontsize=15)
            axis.axis('off')
            report['warnings'].append('Silhouette unavailable for this clustering')
        else:
            order = sorted((i for i in range(count) if np.isfinite(values[i])), key=lambda i: (groups[i], values[i]))
            axis.barh(range(len(order)), values[order], color=[_color(groups[i]) for i in order])
            axis.set_yticks(range(len(order)), [_short(names[i]) for i in order], fontsize=9)
            average = float(np.nanmean(values))
            report['summary']['mean_silhouette'] = average
            axis.axvline(average, color='#D55E00', linestyle='--', label=f'Mean: {average:.3f}')
            axis.axvline(0, color='#98A4AD', linewidth=1)
            axis.set(xlim=(-1,1), xlabel=f'Silhouette coefficient ({report["distance_metric"]})')
            axis.legend(frameon=False)
            _style_axis(axis)
    elif chart == 'cluster_sizes':
        axis = _main_axes(fig)
        unique, counts = np.unique(groups, return_counts=True)
        axis.bar(range(len(unique)), counts, color=[_color(group) for group in unique], edgecolor='white')
        axis.set_xticks(range(len(unique)), ['Noise' if group < 0 else f'Cluster {group+1}' for group in unique])
        for index, value in enumerate(counts):
            axis.text(index, value, str(value), ha='center', va='bottom', fontsize=12)
        report['cluster_sizes'] = {str(int(group)): int(size) for group,size in zip(unique,counts)}
        axis.set(ylabel='Image count', ylim=(0, max(counts)*1.2))
        _style_axis(axis)
    elif chart == 'pca_variance':
        axis = _main_axes(fig)
        variance = data['variance'][:32]
        indices = np.arange(1, len(variance)+1)
        axis.bar(indices, variance, color='#56B4E9', label='Per component')
        axis.plot(indices, np.cumsum(variance), color='#D55E00', marker='o', label='Cumulative')
        axis.set(xlabel='Principal component', ylabel='Fraction of variance', ylim=(0,1.04))
        axis.set_title(f'Centered effective rank: {report["summary"]["effective_rank_centered"]:.2f}', fontsize=12)
        axis.legend(frameon=False)
        _style_axis(axis)
    elif chart == 'feature_statistics':
        axes = fig.subplots(1,3)
        statistics = [('Raw L2 norm', data['norms']), ('Mean absolute value', np.abs(data['raw']).mean(1)),
                      ('Within-vector standard deviation', data['raw'].std(1))]
        report['raw_feature_statistics'] = {title: values.tolist() for title,values in statistics}
        for axis,(title,values) in zip(axes,statistics):
            axis.barh(range(count), values, color=[_color(group) for group in groups])
            axis.set_yticks(range(count), [_short(name) for name in names], fontsize=8)
            axis.invert_yaxis()
            axis.set(xlabel=title, xlim=(0, max(float(values.max())*1.15, 1e-9)))
            _style_axis(axis)
    elif chart in ('feature_heatmap', 'dimension_correlation', 'cluster_centroid_heatmap'):
        axis = _main_axes(fig)
        dimensions = data['dimensions']
        values = data['vectors'][:, dimensions]
        if chart == 'dimension_correlation':
            centered = values - values.mean(0)
            lengths = np.linalg.norm(centered, axis=0)
            scale = np.outer(lengths,lengths)
            correlation = np.divide(centered.T @ centered, scale, out=np.zeros_like(scale), where=scale>1e-20)
            # Undefined correlations (constant columns) stay visibly masked, not treated as zero.
            correlation = np.ma.masked_where(scale<=1e-20, correlation.clip(-1,1))
            cmap = mpl.colormaps['coolwarm'].with_extremes(bad='#D9DEE3')
            image = axis.imshow(correlation, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
            axis.set_xticks(range(len(dimensions)), [str(i) for i in dimensions], rotation=90, fontsize=8)
            axis.set_yticks(range(len(dimensions)), [str(i) for i in dimensions], fontsize=8)
            report['dimension_correlations'] = [[float(correlation[i,j]) if not np.ma.is_masked(correlation[i,j]) else None
                for j in range(len(dimensions))] for i in range(len(dimensions))]
            axis.set_title('Pearson correlation across images; gray = constant dimension', fontsize=11)
            fig.colorbar(image, ax=axis, label='Pearson correlation')
        else:
            if chart == 'cluster_centroid_heatmap':
                unique = [group for group in sorted(set(groups)) if group>=0]
                if not unique:
                    raise ValueError('Cluster centroid heatmap requires at least one non-noise cluster')
                values = np.stack([values[groups==group].mean(0) for group in unique])
                row_labels = [f'Cluster {group+1}' for group in unique]
                report['cluster_centroid_values'] = values.tolist()
            else:
                values = values[ordered]
                row_labels = [names[i] for i in ordered]
                report['displayed_feature_values'] = values.tolist()
            limit = max(float(np.abs(values).max()),1e-12)
            image = axis.imshow(values, cmap='coolwarm', interpolation='nearest', aspect='auto', vmin=-limit,vmax=limit)
            axis.set_xticks(range(len(dimensions)), [str(i) for i in dimensions], rotation=90, fontsize=8)
            axis.set_yticks(range(len(row_labels)), [_short(name) for name in row_labels], fontsize=9)
            fig.colorbar(image, ax=axis, label='Feature value after selected vector normalization')
            axis.set_title('Dimensions with highest across-image variance (ordered by dimension index)', fontsize=11)
        axis.set(xlabel='Feature dimension index')
    elif chart == 'outlier_scores':
        axis = _main_axes(fig)
        scores = np.array([np.mean([item['distance'] for item in row]) if row else 0.0 for row in data['neighbors']])
        order = np.argsort(scores)[::-1]
        axis.barh(range(count), scores[order], color=[_color(groups[i]) for i in order])
        axis.set_yticks(range(count), [_short(names[i]) for i in order], fontsize=9)
        axis.invert_yaxis()
        axis.set(xlabel='Mean distance to selected k nearest images (higher = more isolated)')
        axis.set_title('Exploratory isolation score; not a calibrated anomaly probability', fontsize=11)
        report['isolation_scores'] = scores.tolist()
        _style_axis(axis)
    elif chart == 'patch_energy':
        patches, grid = data['patches'], data['grid']
        if patches is None:
            raise ValueError('patch_energy requires patch_tokens/patch_map with tokens or spatial tensor_layout')
        if grid is None:
            n_tokens = patches.shape[1]
            columns = grid_width or math.isqrt(n_tokens)
            if columns < 1 or n_tokens % columns or (not grid_width and columns**2 != n_tokens):
                raise ValueError('Token count is not a square grid; specify grid_width for patch_energy')
            grid = (n_tokens//columns,columns)
        energy = np.linalg.norm(patches,axis=-1).reshape(count,*grid)
        report['patch_grid_shape'] = list(grid)
        report['patch_l2_norms'] = energy.tolist()
        columns = min(4,count)
        axes = fig.subplots(math.ceil(count/columns),columns,squeeze=False)
        lower,upper = float(energy.min()),float(energy.max())
        for index,axis in enumerate(axes.flat):
            if index>=count:
                axis.axis('off')
                continue
            image = axis.imshow(energy[index],cmap='viridis',vmin=lower,vmax=upper,interpolation='nearest')
            axis.set_title(_short(names[index]),fontsize=10)
            axis.set(xlabel='Patch column',ylabel='Patch row')
        fig.colorbar(image,ax=list(axes.flat),shrink=0.75,label='Raw patch feature L2 norm (not attention / importance)')
