"""Native MOSS inference must keep the buffer allocated before its malloc graph."""
import sys,types,unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kaloscope_style.adapter import StyleAdapter
with patch.dict(sys.modules,{'folder_paths':types.SimpleNamespace(models_dir='/models')}):
    from kaloscope_style.moss import MOSSAfterBlock,apply_moss
import test_moss

class MOSSAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_native_inference_keeps_storage_and_exact_residual(self):
        adapter=StyleAdapter(width=64,layers=8,hidden=32).eval()
        features=torch.nn.functional.normalize(torch.randn(1,256),dim=1)
        for dtype in [torch.float32,torch.bfloat16]:
            with self.subTest(dtype=dtype),torch.inference_mode():
                x=torch.randn(2,11,32,dtype=dtype);ptr=x.data_ptr()
                original=lambda a:dict(img=a['img'],untouched='metadata')
                for layer in range(42):
                    bank=layer*8//42
                    # Independent former out-of-place expression is the numerical baseline.
                    with torch.autocast('cpu',dtype=dtype) if dtype==torch.bfloat16 else nullcontext():
                        st=adapter.encode(features).expand(2,-1,-1)
                        delta=adapter.residual(x[:,:7],st,bank)
                    expected=torch.cat((x[:,:7]+(delta*.75).to(x.dtype),x[:,7:]),dim=1)
                    fn=MOSSAfterBlock(adapter,features,bank,.75,preserve_input_buffer=True)
                    out=fn(dict(img=x,transformer_options={'kaloscope_style_enabled':True,'kaloscope_style_image_tokens':7}),dict(original_block=original))
                    self.assertIs(out['img'],x);self.assertEqual(x.data_ptr(),ptr)
                    self.assertEqual(out['untouched'],'metadata')
                    torch.testing.assert_close(x,expected,rtol=0,atol=0)

    def test_autograd_keeps_out_of_place_path(self):
        adapter=StyleAdapter(width=64,layers=1,hidden=32)
        features=torch.randn(1,256);x=torch.randn(1,9,32,requires_grad=True);before=x.detach().clone()
        fn=MOSSAfterBlock(adapter,features,0,1.,preserve_input_buffer=True)
        out=fn(dict(img=x,transformer_options={'kaloscope_style_enabled':True,'kaloscope_style_image_tokens':6}),dict(original_block=lambda a:dict(img=a['img'])))['img']
        self.assertNotEqual(out.data_ptr(),x.data_ptr());torch.testing.assert_close(x,before,rtol=0,atol=0)
        out.sum().backward();self.assertIsNotNone(x.grad)

    def test_only_native_core_requests_stable_buffer(self):
        helper=test_moss.NativeMOSSApplyTests()
        for name,multiplier,expected in [('MOSSImage',1.,True),('MOSSImageTransformer',1000.,False)]:
            clone=apply_moss(helper.patcher(name,multiplier),helper.bundle(),torch.ones(1,256),6,1)
            self.assertEqual(len(clone.patches),42)
            self.assertTrue(all(p.preserve_input_buffer==expected for p in clone.patches.values()))

if __name__=='__main__':unittest.main()
