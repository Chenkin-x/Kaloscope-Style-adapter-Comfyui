"""Migrate public workflow interfaces without altering prompts, seeds or numeric settings."""
from copy import deepcopy
from .compat import NODE_IDS, LEGACY_ADAPTER_SOCKET, public_asset_name, public_text

ENCODER = 'KaloscopeStyleEncoderLoader'
REFERENCE = 'KaloscopeStyleEncodeReference'
ASSET_INPUT = {
    ENCODER: 'model_folder',
    'KaloscopeStyleAdapterLoader': 'checkpoint',
    'KaloscopeStyleArtist': 'bank',
    'KaloscopeStyleArtistMix': 'bank',
}


def migrate_workflow(data):
    if not isinstance(data, dict):
        raise ValueError('Workflow must be a UI graph or API prompt object.')
    result = deepcopy(data)
    if isinstance(result.get('nodes'), list):
        nodes = {n['id']: n for n in result['nodes']}
        for node in nodes.values():
            kind = NODE_IDS.get(node.get('type'), node.get('type'))
            if kind == 'KaloscopeModelLoader':
                consumers = [NODE_IDS.get(nodes[link[3]]['type'], nodes[link[3]]['type'])
                             for link in result.get('links', []) if link[1] == node['id']]
                if consumers and all(t == REFERENCE for t in consumers): kind = ENCODER
            if kind not in ASSET_INPUT and kind not in NODE_IDS.values() and kind != 'KaloscopeMOSSBaseLoader':
                continue
            node['type'] = kind
            if 'title' in node: node['title'] = public_text(node['title'])
            node.setdefault('properties', {})['Node name for S&R'] = kind
            widgets = node.get('widgets_values')
            if kind in ASSET_INPUT and isinstance(widgets, list) and widgets and isinstance(widgets[0], str):
                widgets[0] = public_asset_name(widgets[0])
            for socket in node.get('inputs', []) + node.get('outputs', []):
                for key in ('name', 'type'):
                    if socket.get(key) == LEGACY_ADAPTER_SOCKET: socket[key] = 'KALOSCOPE_STYLE_ADAPTER'
        for link in result.get('links', []):
            if link[5] == LEGACY_ADAPTER_SOCKET: link[5] = 'KALOSCOPE_STYLE_ADAPTER'
        return result
    for nid, node in result.items():
        if not isinstance(node, dict) or 'class_type' not in node: continue
        kind = NODE_IDS.get(node['class_type'], node['class_type'])
        if kind == 'KaloscopeModelLoader':
            consumers = [NODE_IDS.get(v['class_type'], v['class_type']) for v in result.values()
                         if isinstance(v, dict) and 'class_type' in v
                         and any(isinstance(x, list) and len(x) == 2 and str(x[0]) == str(nid)
                                 for x in v.get('inputs', {}).values())]
            if consumers and all(t == REFERENCE for t in consumers): kind = ENCODER
        node['class_type'] = kind
        if kind in ASSET_INPUT:
            key = ASSET_INPUT[kind]
            val = node.get('inputs', {}).get(key)
            if isinstance(val, str): node['inputs'][key] = public_asset_name(val)
        if kind in NODE_IDS.values() and isinstance(node.get('_meta', {}).get('title'), str):
            node['_meta']['title'] = public_text(node['_meta']['title'])
    return result
