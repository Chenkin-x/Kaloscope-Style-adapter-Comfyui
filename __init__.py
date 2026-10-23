"""ComfyUI generation-only entry point; no global module aliases or sys.path edits."""
from .kaloscope_style.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, artist_bank_paths
from .kaloscope_style.routes import register_routes
from server import PromptServer

WEB_DIRECTORY = './web'
register_routes(PromptServer.instance.routes, artist_bank_paths)
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
