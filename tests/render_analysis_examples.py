"""Infer each example once, cache features, then exercise every plotting node mode."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feature_analysis import CHART_TYPES
from kaloscope_dinov3.preprocessing import prepare_rgb
from model_loading import load_model_bundle
from test_model_loading import load_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../test')
    parser.add_argument('--model-dir', default='../model')
    parser.add_argument('--output', default='../outputs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--reuse-cache', action='store_true', help='Redraw charts from features.pt without loading the model')
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in source.iterdir() if path.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'})
    if not paths:
        raise ValueError(f'No images in {source}')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    cache_path=output/'features.pt'
    if args.reuse_cache:
        cached=torch.load(cache_path,map_location='cpu',weights_only=True)
        features,patch_features=cached['features'],cached['patch_tokens']
        labels,records=cached['labels'],cached['source_images']
        model_info=cached.get('model_info',{'model_type':'dinov3_vitb16','feature_source':'projector',
            'input_size':512,'checkpoint':str(Path(args.model_dir).resolve()/'best.pt')})
        inference_batches=cached.get('inference_batches',4)
        previews=[]
        for record in records:
            with Image.open(record['path']) as original:
                previews.append(ImageOps.pad(prepare_rgb(original),(240,200),color='white'))
        thumbnails=torch.stack([torch.from_numpy(np.array(image)).float()/255 for image in previews])
        print('Reusing cached features: zero new model forwards',flush=True)
    else:
        bundle = load_model_bundle(args.model_dir, device=args.device)
        model = bundle['model']
        model_info = {key: bundle[key] for key in ('model_type','feature_source','input_size','checkpoint')}
        if not hasattr(model, 'extract_tensor'):
            raise ValueError('This example suite requires a DINOv3 checkpoint for patch analysis')
        vectors, patches, previews = [], [], []
        labels, records = [], []
        inference_batches = 0
        with torch.inference_mode():
            for start in range(0,len(paths),args.batch_size):
                batch = []
                for index, path in enumerate(paths[start:start+args.batch_size], start):
                    with Image.open(path) as original:
                        image = prepare_rgb(original)
                        batch.append(bundle['transform'](image))
                        previews.append(ImageOps.pad(image, (240,200), color='white'))
                    artist = re.search(r'_drawn_by_(.+?)__[0-9a-f]+$', path.stem)
                    label = f'{index+1:02d} ' + (artist.group(1) if artist else path.stem[:22])
                    labels.append(label)
                    with path.open('rb') as stream:
                        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    records.append({'index': index, 'label': label, 'path': str(path), 'sha256': digest})
                tokens = model.backbone.forward_features(torch.stack(batch).to(args.device))
                pooled = model._pool(tokens['x_norm_clstoken'],tokens['x_norm_patchtokens'],model.pooling)
                feature = model.projector(pooled) if model.feature_source=='projector' else pooled
                vectors.append(feature.float().cpu())
                patches.append(tokens['x_norm_patchtokens'].float().cpu())
                inference_batches += 1
                print(f'Inferred images {start+1}-{min(start+args.batch_size,len(paths))} once',flush=True)
        features = torch.cat(vectors)
        patch_features = torch.cat(patches)
        thumbnails = torch.stack([torch.from_numpy(np.array(image)).float()/255 for image in previews])
        torch.save({'features':features, 'patch_tokens':patch_features, 'labels':labels, 'source_images':records,'model_info':model_info,'inference_batches':inference_batches},output/'features.pt')
        # Release GPU model; all charts below consume cached CPU tensors only.
        del model, bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    nodes = load_nodes()
    node = nodes.KaloscopeFeatureAnalysisNode()
    generated = []
    for chart in CHART_TYPES:
        tensor = patch_features if chart=='patch_energy' else features
        result, text, distances = node.analyze(tensor,chart_type=chart,labels=json.dumps(labels),images=thumbnails,
            tensor_layout='tokens' if chart=='patch_energy' else 'vectors',n_clusters=3,top_k=3,
            metric='cosine',normalize=True,width=1600,height=1100,perplexity=3.0)
        report = json.loads(text)
        report['source_images']=records
        report['feature_source']='patch_tokens' if chart=='patch_energy' else model_info['feature_source']
        array = (result[0].clamp(0,1).numpy()*255).round().astype(np.uint8)
        Image.fromarray(array).save(output/f'{chart}.png',dpi=(120,120))
        (output/f'{chart}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        if chart=='relationship_graph':
            with (output/'distance_matrix.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['image',*labels])
                writer.writerows([label,*row.tolist()] for label,row in zip(labels,distances))
        generated.append({'type':chart,'image':f'{chart}.png','report':f'{chart}.json'})
        print(f'Saved {chart}.png',flush=True)
    manifest={'input_directory':str(source),'checkpoint':model_info['checkpoint'],
              'architecture':model_info['model_type'],'model_info':model_info,'feature_shape':list(features.shape),'patch_shape':list(patch_features.shape),
              'source_images':records,'inference_batches':inference_batches,'image_forward_count':len(paths),
              'analysis_model_forward_count':0,'this_run_model_forward_count':0 if args.reuse_cache else len(paths),'seed':42,'charts':generated}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    body=''.join(f'<section><h2>{html.escape(item["type"])}</h2><a href="{item["image"]}"><img src="{item["image"]}" alt="{html.escape(item["type"])}"></a><p><a href="{item["report"]}">分析数据 JSON</a></p></section>' for item in generated)
    (output/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Kaloscope 分析图示例</title>
<style>body{font:16px system-ui;background:#eef2f5;color:#243746;max-width:1500px;margin:32px auto;padding:0 24px}section{background:white;margin:24px 0;padding:24px;border-radius:12px}img{max-width:100%;height:auto}a{color:#0072b2}</style>
<h1>Kaloscope 特征分析示例</h1><p>每张输入图片仅推理一次。18 类图表复用缓存特征；二维距离布局是近似结果，具体距离以矩阵和 JSON 为准。</p>
<p><a href="manifest.json">来源与运行记录</a> · <a href="distance_matrix.csv">距离矩阵 CSV</a></p>'''+body+'</html>',encoding='utf-8')
    contact = Image.new('RGB',(1600,math_rows(len(generated))*390),'#eef2f5')
    drawer=ImageDraw.Draw(contact)
    try:
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    except OSError:
        font=ImageFont.load_default()
    for index,item in enumerate(generated):
        x=(index%3)*530+10; y=(index//3)*390
        with Image.open(output/item['image']) as image:
            preview=ImageOps.contain(image,(510,350))
            contact.paste(preview,(x,y+32))
        drawer.text((x,y+5),item['type'],fill='#243746',font=font)
    contact.save(output/'overview.png')
    print(f'{len(generated)} chart types written to {output}; analysis performed zero model forwards',flush=True)


def math_rows(count):
    return (count+2)//3


if __name__=='__main__':
    main()
