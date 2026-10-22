import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_model_loading import load_nodes
from feature_analysis import CHART_TYPES, analyze_features, prepare_features


class FeatureAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.vectors=torch.tensor([[1.,0.,0.],[.9,.1,0.],[0.,1.,0.],[0.,.9,.1],[0.,0.,1.],[.1,0.,.9]])
        cls.nodes=load_nodes()

    def test_every_chart_returns_finite_image_and_original_distance_matrix(self):
        expected=1-torch.nn.functional.normalize(self.vectors,dim=1)@torch.nn.functional.normalize(self.vectors,dim=1).T
        expected.fill_diagonal_(0)
        for chart in CHART_TYPES:
            with self.subTest(chart=chart):
                features=self.vectors
                options={}
                if chart=='patch_energy':
                    features=self.vectors[:,None,:].repeat(1,4,1)
                    options['tensor_layout']='tokens'
                image,text,distances=analyze_features(features,chart_type=chart,width=800,height=600,**options)
                report=json.loads(text)
                self.assertEqual(tuple(image.shape),(1,600,800,3))
                self.assertTrue(torch.isfinite(image).all())
                self.assertGreater(float(image.max()-image.min()),0.5)
                torch.testing.assert_close(distances,expected,atol=2e-7,rtol=1e-6)
                self.assertEqual(report['chart_type'],chart)
                self.assertEqual(len(report['nearest_neighbors']),6)
                for index,row in enumerate(report['nearest_neighbors']):
                    self.assertNotIn(index,[item['index'] for item in row])

    def test_graph_edges_are_knn_union_and_use_original_distance(self):
        _,text,distances=analyze_features(self.vectors,top_k=1,width=800,height=600)
        report=json.loads(text)
        expected={tuple(sorted((i,row[0]['index']))) for i,row in enumerate(report['nearest_neighbors'])}
        actual={(edge['source'],edge['target']) for edge in report['edges']}
        self.assertEqual(actual,expected)
        for edge in report['edges']:
            self.assertAlmostEqual(edge['distance'],float(distances[edge['source'],edge['target']]),places=6)

    def test_distances_and_normalization_match_known_values(self):
        features=torch.tensor([[3.,4.],[0.,5.]])
        for metric,value in (('euclidean',np.sqrt(10)),('manhattan',4.0),('cosine',0.2)):
            _,_,dist=analyze_features(features,chart_type='distance_heatmap',metric=metric,normalize=False,width=800,height=600)
            self.assertAlmostEqual(float(dist[0,1]),value,places=6)
        _,text,dist=analyze_features(features,chart_type='distance_heatmap',metric='euclidean',normalize=True,width=800,height=600)
        self.assertAlmostEqual(float(dist[0,1]),np.sqrt(.4),places=6)
        self.assertEqual(json.loads(text)['normalization'],'row L2')

    def test_layout_reductions_preserve_image_batch(self):
        tokens=torch.arange(2*3*4,dtype=torch.float32).reshape(2,3,4)
        vector,patches,_,_=prepare_features(tokens,'tokens')
        np.testing.assert_allclose(vector,tokens.mean(1).numpy())
        self.assertEqual(patches.shape,(2,3,4))
        spatial=torch.arange(2*4*2*3,dtype=torch.float32).reshape(2,4,2,3)
        vector,patches,grid,_=prepare_features(spatial,'spatial')
        np.testing.assert_allclose(vector,spatial.mean((2,3)).numpy())
        self.assertEqual(grid,(2,3))
        layers=torch.stack((tokens,tokens+10),dim=1)
        vector,_,_,_=prepare_features(layers,'layer_tokens',layer_index=-1)
        np.testing.assert_allclose(vector,(tokens+10).mean(1).numpy())
        vector,_,_,_=prepare_features(layers,'layer_tokens',layer_pooling='mean')
        np.testing.assert_allclose(vector,(tokens+5).mean(1).numpy())
        vector,_,_,_=prepare_features(layers,'flatten')
        self.assertEqual(vector.shape,(2,24))
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            prepare_features(layers)

    def test_bad_inputs_and_undefined_cosine_fail_explicitly(self):
        for features in (torch.tensor([[0.,0.],[1.,0.]]),torch.tensor([[float('nan'),1.]])):
            with self.assertRaises(ValueError):
                analyze_features(features,width=800,height=600)
        with self.assertRaisesRegex(ValueError,'labels'):
            analyze_features(self.vectors,labels='one',width=800,height=600)
        with self.assertRaisesRegex(ValueError,'matching features'):
            analyze_features(self.vectors,images=torch.ones(1,32,32,3),width=800,height=600)
        with self.assertRaisesRegex(ValueError,'patch_energy requires'):
            analyze_features(self.vectors,chart_type='patch_energy',width=800,height=600)

    def test_degenerate_inputs_do_not_fabricate_silhouette_or_pca(self):
        identical=torch.ones(4,3)
        _,text,dist=analyze_features(identical,chart_type='silhouette',width=800,height=600)
        report=json.loads(text)
        self.assertIsNone(report['silhouette_scores'])
        self.assertEqual(report['pca_explained_variance_ratio'],[0.,0.,0.])
        self.assertTrue(report['warnings'])
        torch.testing.assert_close(dist,torch.zeros(4,4))
        _,text,_=analyze_features(torch.zeros(3,2),metric='euclidean',chart_type='dimension_correlation',width=800,height=600)
        self.assertEqual(json.loads(text)['dimension_correlations'],[[None,None],[None,None]])

    def test_feature_node_never_invokes_model_and_image_node_invokes_extraction_once(self):
        node=self.nodes.KaloscopeFeatureAnalysisNode()
        self.assertNotIn('model',node.INPUT_TYPES()['required'])
        with patch.object(self.nodes.KaloscopeExtractFeaturesNode,'extract',side_effect=AssertionError('unexpected inference')):
            image,_,matrix=node.analyze(self.vectors,chart_type='relationship_graph',width=800,height=600)
        self.assertEqual(matrix.shape,(6,6))
        with patch.object(self.nodes.KaloscopeExtractFeaturesNode,'extract',return_value=(self.vectors,)) as extraction:
            outputs=self.nodes.KaloscopeImageAnalysisNode().analyze(torch.ones(6,32,32,3),{},width=800,height=600)
        extraction.assert_called_once()
        torch.testing.assert_close(outputs[2],self.vectors)
        self.assertEqual(tuple(outputs[0].shape),(1,600,800,3))

    def test_clustering_options_and_seed_are_reproducible(self):
        for method in ('kmeans','agglomerative','dbscan','none'):
            _,text,_=analyze_features(self.vectors,chart_type='cluster_sizes',cluster_method=method,width=800,height=600)
            self.assertEqual(len(json.loads(text)['cluster_labels']),6)
        _,first,_=analyze_features(self.vectors,chart_type='pca_scatter',seed=7,width=800,height=600)
        _,second,_=analyze_features(self.vectors,chart_type='pca_scatter',seed=7,width=800,height=600)
        self.assertEqual(json.loads(first)['cluster_labels'],json.loads(second)['cluster_labels'])
        self.assertEqual(json.loads(first)['projection'],json.loads(second)['projection'])


if __name__=='__main__':
    unittest.main()
