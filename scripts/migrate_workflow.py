"""Write a migrated copy; never overwrite the original workflow."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kaloscope_style.workflows import migrate_workflow
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('output');a=p.parse_args()
    src,dst=Path(a.input),Path(a.output)
    if src.resolve()==dst.resolve() or dst.exists():p.error('Choose a new output filename; source/existing files are preserved.')
    dst.write_text(json.dumps(migrate_workflow(json.loads(src.read_text(encoding='utf-8-sig'))),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
