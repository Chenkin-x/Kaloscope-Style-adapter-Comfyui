import base64
import logging
from typing import Callable
from threading import Lock
from secrets import compare_digest
from io import BytesIO
import asyncio
import concurrent.futures
import os
import glob
import json

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
from backend_lsnet.inference import process_image_from_pil
from backend_lsnet.analysis_api import (FeaturesRequest, AnalysisRequest, FeatureToolsRequest,
                                       extract_request, serialized_cache, analysis_request, tools_request)

try:
    from modules import shared
    from modules.call_queue import queue_lock as webui_queue_lock
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'cmd_opts': type('CmdOpts', (), {'api_auth': None})()})()
    webui_queue_lock = None

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from backend_lsnet.model_paths import (
    get_available_checkpoints, get_available_csv, get_checkpoint_path, get_class_csv,
)

class InferenceRequest(BaseModel):
    input_image: str = Field(..., description="Input image as Base64 encoded string")
    model_name: str = Field('Kaloscope', description="Model name (subfolder in models/kaloscope/)")
    device: str = Field('cuda', description="Device to use")
    top_k: int = Field(5, ge=1, le=20, description="Number of top predictions")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Probability threshold")
    mode: str = 'auto'
    output_type: str = 'default'
    layers: str = '-1'
    intermediate_norm: bool = True

class InferenceResponse(BaseModel):
    results: dict = Field(..., description="Inference results")
    info: str = Field(..., description="Additional information")

class CancelResponse(BaseModel):
    info: str = Field(..., description="Cancel operation result")

class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock = None, prefix: str = "/kaloscope/v1"):
        self.app = app
        self.queue_lock = queue_lock or Lock()
        self.prefix = prefix
        self.credentials = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        if IN_WEBUI and shared.cmd_opts.api_auth:
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.add_api_route(
            "infer",
            self.endpoint_infer,
            methods=["POST"],
            response_model=InferenceResponse,
            summary="Perform artist style inference",
            description="Classify an image or extract features with LSNet or DINOv3."
        )
        self.add_api_route(
            "cancel",
            self.endpoint_cancel,
            methods=["POST"],
            response_model=CancelResponse,
            summary="Cancel the current inference task",
            description="Terminates the ongoing inference task."
        )
        self.add_api_route('features', self.endpoint_features, methods=['POST'], summary='Extract a batch once; return reusable NPZ cache')
        self.add_api_route('analyze', self.endpoint_analyze, methods=['POST'], summary='Render any analysis chart from features/cache or an image batch')
        self.add_api_route('feature-tools', self.endpoint_feature_tools, methods=['POST'], summary='Common features, similarity or group comparison without inference')
        self.add_api_route('models', self.endpoint_models, methods=['GET'], summary='Available model folders and supported chart/feature types')

    def auth(self, creds: HTTPBasicCredentials = Depends(HTTPBasic())):
        if not self.credentials:
            return True
        if creds.username in self.credentials:
            if compare_digest(creds.password, self.credentials[creds.username]):
                return True
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"}
        )

    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        route = f"{self.prefix}/{path}" if self.prefix else path
        dependencies = [Depends(self.auth)] if self.credentials else []
        self.app.add_api_route(route, endpoint, dependencies=dependencies, **kwargs)

    def decode_base64_image(self, base64_str: str) -> Image.Image:
        try:
            img_data = base64.b64decode(base64_str, validate=True)
            img = Image.open(BytesIO(img_data))
            img.load()
            return img
        except base64.binascii.Error:
            raise HTTPException(400, "Invalid Base64 string format")
        except Exception as e:
            raise HTTPException(400, f"Failed to decode image: {str(e)}")

    async def run_inference(self, image, **kwargs):
        """Run inference in a separate thread"""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(self.executor, lambda: process_image_from_pil(image, **kwargs))
        except Exception as e:
            logger.error(f"Inference execution failed: {str(e)}")
            raise

    async def endpoint_infer(self, req: InferenceRequest):
        logger.info(f"Received inference request: model_name={req.model_name}")
        try:
            with self.queue_lock:
                input_image = self.decode_base64_image(req.input_image)

                checkpoints = get_available_checkpoints(req.model_name)
                if not checkpoints:
                    raise HTTPException(400, f"No checkpoints found for model {req.model_name}")
                checkpoint = get_checkpoint_path(req.model_name)
                if not os.path.exists(checkpoint):
                    raise HTTPException(400, f"Checkpoint not found: {checkpoint}")

                class_csv = get_class_csv(req.model_name)
                infer_args = {
                    "checkpoint": checkpoint,
                    "mode": req.mode,
                    "output_type": req.output_type,
                    "layers": req.layers,
                    "intermediate_norm": req.intermediate_norm,
                    "device": req.device,
                    "top_k": req.top_k,
                    "threshold": req.threshold,
                    "class_csv": class_csv
                }

            # Run inference
            results = await self.run_inference(input_image, **infer_args)

            return InferenceResponse(results=results, info="Inference completed successfully")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise HTTPException(500, f"Inference failed: {str(e)}")

    async def endpoint_cancel(self):
        # For simplicity, just return a message since inference is quick
        return CancelResponse(info="No active inference to cancel")

    async def run_feature_job(self, callback):
        def locked_job():
            with self.queue_lock:
                return callback()
        try:
            return await asyncio.get_running_loop().run_in_executor(self.executor, locked_job)
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError, TypeError, KeyError) as error:
            raise HTTPException(400, str(error)) from error

    async def endpoint_features(self, req: FeaturesRequest):
        return await self.run_feature_job(lambda: serialized_cache(extract_request(req, self.decode_base64_image)[0]))

    async def endpoint_analyze(self, req: AnalysisRequest):
        return await self.run_feature_job(lambda: analysis_request(req, self.decode_base64_image))

    async def endpoint_feature_tools(self, req: FeatureToolsRequest):
        return await self.run_feature_job(lambda: tools_request(req))

    async def endpoint_models(self):
        from backend_lsnet.model_paths import get_available_models
        from feature_analysis import CHART_TYPES, TENSOR_LAYOUTS
        from model_loading import FEATURE_OUTPUTS
        return {'models': get_available_models(), 'feature_outputs': FEATURE_OUTPUTS,
                'chart_types': CHART_TYPES, 'tensor_layouts': TENSOR_LAYOUTS}

def on_app_started(demo, app):
    """Called when the webui app starts"""
    queue_lock = webui_queue_lock or Lock()
    api = Api(app, queue_lock)
    logger.info("Kaloscope API routes added to webui")
