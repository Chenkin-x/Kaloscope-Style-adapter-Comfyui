import gradio as gr
from backend_lsnet.inference import process_image_from_pil
from backend_lsnet.analysis_ui import build_analysis_tab
from model_loading import FEATURE_OUTPUTS
import os
import json

from backend_lsnet.model_paths import (
    get_available_models, get_available_checkpoints, get_available_csv,
    get_checkpoint_path, get_class_csv,
)

def create_ui():
    css = """
    .contain-image img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
        background: #222;
    }
    """

    block = gr.Blocks(css=css, analytics_enabled=False)
    with block:
        gr.Markdown('# Kaloscope Artist Inference')
        with gr.Tabs():
            with gr.TabItem("Inference"):
                with gr.Row():
                    with gr.Column():
                        image_kwargs = {'source': 'upload'} if int(gr.__version__.split('.')[0]) < 4 else {'sources': ['upload']}
                        input_image = gr.Image(**image_kwargs, type="pil", label="Input Image", height=320, elem_classes="contain-image")
                        model = gr.Dropdown(
                            choices=get_available_models(),
                            label="Model Folder", value=(get_available_models() or [None])[0]
                        )
                        device = gr.Dropdown(['cuda', 'cpu'], label="Device", value='cuda')
                        top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=5, step=1)
                        threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                        mode = gr.Dropdown(['auto', 'classify', 'cluster', 'both'], value='auto', label='Mode')
                        output_type = gr.Dropdown(list(FEATURE_OUTPUTS), value='default', label='Feature Output')
                        layers = gr.Textbox(value='-1', label='Intermediate layers')
                        norm = gr.Checkbox(value=True, label='Intermediate LayerNorm')
                        infer_button = gr.Button(value="Infer")
                    with gr.Column():
                        tag_string = gr.Textbox(label="Formatted Tags", lines=3, interactive=False)
                        result_json = gr.Textbox(label="JSON Results", lines=15, interactive=False)
                        error_message = gr.Markdown("", visible=False)
            build_analysis_tab()

        def infer(image, model, device, top_k, threshold, mode, output_type, layers, norm):
            if image is None:
                return "Please upload an image.", "", gr.update(visible=True)
            checkpoints = get_available_checkpoints(model)
            if not checkpoints:
                return f"No checkpoints found for model {model}.", "", gr.update(visible=True)
            try:
                checkpoint = get_checkpoint_path(model)
                class_csv = get_class_csv(model)
                kwargs = {
                    'checkpoint': checkpoint,
                    'mode': mode,
                    'output_type': output_type,
                    'layers': layers,
                    'intermediate_norm': norm,
                    'device': device,
                    'top_k': top_k,
                    'threshold': threshold,
                    'class_csv': class_csv
                }
                results = process_image_from_pil(image, **kwargs)
                tag_string = ",".join([r['class_name'] for r in results.get('classification', [])])
                output = results
                json_str = json.dumps(output, ensure_ascii=False)
                return tag_string, json_str, gr.update(visible=False)
            except Exception as e:
                return str(e), "", gr.update(visible=True)

        infer_button.click(
            infer,
            inputs=[input_image, model, device, top_k, threshold, mode, output_type, layers, norm],
            outputs=[tag_string, result_json, error_message]
        )

    return block
