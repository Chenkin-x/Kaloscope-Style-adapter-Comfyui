import sys
from pathlib import Path

# WebUI extensions and direct script execution do not put this plugin root on sys.path.
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import logging
import threading
from threading import Lock
from fastapi import FastAPI
from backend_lsnet.inference import *
from backend_lsnet.ui import *
from backend_lsnet.api import Api
import uvicorn
import gradio as gr
import os
import argparse

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Kaloscope Artist Inference WebUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--models-dir", type=str, default=None, help="Root containing kaloscope/ model folders")
    return parser.parse_args()

try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'opts': type('Opts', (), {
        'outdir_samples': '',
        'outdir_txt2img_samples': '',
        'outdir_img2img_samples': ''
    })})()

if IN_WEBUI:
    from backend_lsnet.api import on_app_started
    def on_ui_tabs():
        block = create_ui()
        return [(block, "Kaloscope", "kaloscope_tab")]
    script_callbacks.on_ui_tabs(on_ui_tabs)
    script_callbacks.on_app_started(on_app_started)
else:
    if __name__ == "__main__":
        args = parse_args()
        if args.models_dir:
            os.environ['KALOSCOPE_MODELS_DIR'] = str(Path(args.models_dir).resolve())
        # Create models directory
        from backend_lsnet.model_paths import models_root
        (models_root() / 'kaloscope').mkdir(parents=True, exist_ok=True)

        app = FastAPI(docs_url="/docs", openapi_url="/openapi.json")
        queue_lock = Lock()
        api = Api(app, queue_lock, prefix="/kaloscope/v1")
        logging.info("API 路由已挂载到 FastAPI 实例")

        block = create_ui()
        logging.info("Gradio 界面已创建")

        # Mount Gradio app to FastAPI
        app = gr.mount_gradio_app(app, block, path="")

        print(f"应用启动在：http://{args.host}:{args.port}")
        print(f"API 文档：http://{args.host}:{args.port}/docs")

        try:
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                log_level="info"
            )
        except Exception as e:
            logging.error(f"启动失败: {str(e)}")
