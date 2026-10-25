"""B8 export must preserve all trained banks and reject incompatible mappings."""
import sys,tempfile,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kaloscope_style.adapter import StyleAdapter
from kaloscope_style.checkpoint import export_checkpoint,load_adapter

class MOSSB8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_export_preserves_eight_banks_and_all_42_injections(self):
        cfg=dict(model='MOSS-Image-Base',feature_dim=256,adapter_width=256,style_tokens=8,adapter_banks=8)
        state={'adapter.'+k:v for k,v in StyleAdapter(layers=8,hidden=3072).state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'step.pt';target=Path(folder)/'step.safetensors'
            torch.save(dict(trainable=state,config=cfg,step=10000),source)
            self.assertEqual(export_checkpoint(source,target),10000)
            bundle=load_adapter(target)
            self.assertEqual(bundle['backend'],'moss')
            self.assertEqual(len(bundle['adapter'].query),8)
            self.assertEqual(bundle['config']['injection_indices'],list(range(42)))
            self.assertEqual(bundle['config']['injection_slots'],[0]*6+[1]*5+[2]*5+[3]*5+[4]*6+[5]*5+[6]*5+[7]*5)
            self.assertEqual(sum(p.numel() for p in bundle['adapter'].parameters()),13309696)
            for k,v in bundle['adapter'].state_dict().items():torch.testing.assert_close(v,state['adapter.'+k],rtol=0,atol=0)
            cfg['injection_slots']=[0]*11+[1]*10+[2]*11+[3]*10
            torch.save(dict(trainable=state,config=cfg),source)
            with self.assertRaises(ValueError):load_adapter(source)
            del cfg['injection_slots'];cfg['adapter_banks']=4
            torch.save(dict(trainable=state,config=cfg),source)
            with self.assertRaises(RuntimeError):load_adapter(source)

if __name__=='__main__':unittest.main()
