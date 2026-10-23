"""MOSS checkpoint, image-only residual and three-branch CFG contracts."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kaloscope_style.adapter import StyleAdapter
from kaloscope_style.checkpoint import export_checkpoint,load_adapter
from kaloscope_style.features import normalized
with patch.dict(sys.modules,{'folder_paths':types.SimpleNamespace(models_dir='/models')}):
    from kaloscope_style.moss import MOSSAfterBlock,MOSSStyleGuidance,moss_cond_batch,KaloscopeMOSSBaseLoader


class MOSSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_moss_export_restores_exact_weights_and_mapping(self):
        config=dict(model='MOSS-Image-Base',feature_dim=256,adapter_width=256,style_tokens=8,adapter_banks=4)
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'step.pt';target=Path(temp)/'moss.safetensors'
            state={'adapter.'+k:v for k,v in StyleAdapter(hidden=3072).state_dict().items()}
            torch.save(dict(trainable=state,config=config,step=8500),source)
            self.assertEqual(export_checkpoint(source,target),8500)
            loaded=load_adapter(target)
            self.assertEqual(loaded['backend'],'moss');self.assertEqual(loaded['config']['injection_indices'],list(range(42)))
            self.assertEqual(loaded['config']['injection_slots'],[0]*11+[1]*10+[2]*11+[3]*10)
            self.assertEqual(sum(p.numel() for p in loaded['adapter'].parameters()),7017216)
            for k,v in loaded['adapter'].state_dict().items():torch.testing.assert_close(v,state['adapter.'+k],rtol=0,atol=0)
            config['injection_slots']=[i//7 for i in range(42)]
            torch.save(dict(trainable=state,config=config),source)
            with self.assertRaises(ValueError):load_adapter(source)

    def test_moss_patch_only_changes_image_tokens_and_honors_off(self):
        adapter=StyleAdapter(width=64,layers=1,hidden=32)
        torch.nn.init.normal_(adapter.out[0].weight,std=.01)
        style=normalized(torch.randn(1,256));x=torch.randn(2,9,32)
        patcher=MOSSAfterBlock(adapter,style,0,.7)
        args=dict(img=x,transformer_options={'kaloscope_style_enabled':True,'kaloscope_style_image_tokens':6})
        original=lambda a:dict(img=a['img']+.2)
        got=patcher(args,{'original_block':original})['img']
        expected=x+.2
        expected[:,:6]+=adapter.residual(expected[:,:6],adapter.encode(style).expand(2,-1,-1),0)*.7
        torch.testing.assert_close(got,expected)
        torch.testing.assert_close(got[:,6:],x[:,6:]+.2,rtol=0,atol=0)
        args['transformer_options']={}
        torch.testing.assert_close(patcher(args,{'original_block':original})['img'],x+.2,rtol=0,atol=0)

    def test_cfg_skips_only_negative_and_does_not_mutate_shared_options(self):
        original=object();options={'transformer_options':{'patches_replace':{'dit':{('single_block',22):original}}}}
        seen=[]
        def replace(opts,fn,*_):
            # Match ComfyUI's function: the outer options dict is mutated.
            opts['transformer_options']={'patches_replace':{'dit':{('single_block',22):fn}}};return opts
        def calc(model,conds,x,sigma,opts):
            seen.append(opts['transformer_options']['patches_replace']['dit'][('single_block',22)])
            return [torch.ones_like(x),torch.zeros_like(x)]
        mocks={'comfy.samplers':types.SimpleNamespace(calc_cond_batch=calc),
               'comfy.model_patcher':types.SimpleNamespace(set_model_options_patch_replace=replace)}
        with patch.dict(sys.modules,mocks):
            out=moss_cond_batch(dict(model=None,conds=['positive','negative'],input=torch.zeros(1,128,2,3),sigma=None,model_options=options))
        self.assertIs(seen[0],original);self.assertIsNot(seen[1],original)
        self.assertIs(options['transformer_options']['patches_replace']['dit'][('single_block',22)],original)
        self.assertEqual(len(out),2)

    def test_style_cfg_matches_velocity_formula_and_zero_is_base(self):
        x=torch.randn(1,128,2,3);sigma=.7
        vn,vt,vs=[torch.randn_like(x) for _ in range(3)]
        native=x-sigma*vt;negative=x-sigma*vn;styled=x-sigma*vs
        options={'transformer_options':{'keep':True}};seen=[]
        def calc(model,conds,z,s,opts):seen.append(opts);return [styled]
        args=dict(input=x,model=None,input_cond='positive',sigma=torch.tensor([sigma]),model_options=options,
                  cond_denoised=native,uncond_denoised=negative,cond_scale=3)
        with patch.dict(sys.modules,{'comfy.samplers':types.SimpleNamespace(calc_cond_batch=calc)}):
            actual=MOSSStyleGuidance(6)(args)/sigma
            torch.testing.assert_close(actual,vn+3*(vt-vn)+6*(vs-vt))
            torch.testing.assert_close(MOSSStyleGuidance(0)(args)/sigma,vn+3*(vt-vn))
        self.assertEqual(len(seen),1);self.assertEqual(seen[0]['transformer_options']['kaloscope_style_image_tokens'],6)
        self.assertEqual(options,{'transformer_options':{'keep':True}})

    def test_base_loader_defaults_to_base_and_explains_missing_dependency(self):
        self.assertEqual(KaloscopeMOSSBaseLoader.INPUT_TYPES()['required']['model_path'][1]['default'],'MOSS-Image-Base')
        with patch.dict(sys.modules,{'nodes':types.SimpleNamespace(NODE_CLASS_MAPPINGS={})}):
            with self.assertRaisesRegex(RuntimeError,'official MOSS'):KaloscopeMOSSBaseLoader().load()

if __name__=='__main__':unittest.main()
