"""Migrate style-only encoder loaders while preserving mixed analysis workflows."""
from copy import deepcopy


def migrate_workflow(data):
    result = deepcopy(data)
    if isinstance(result.get('nodes'), list):
        nodes = {n['id']: n for n in result['nodes']}
        for node in nodes.values():
            if node.get('type') != 'KaloscopeModelLoader':
                continue
            consumers = [nodes[link[3]]['type'] for link in result.get('links', []) if link[1] == node['id']]
            if consumers and all(t == 'KaloscopeZ3EncodeReference' for t in consumers):
                node['type'] = 'KaloscopeZ3EncoderLoader'
                node['title'] = 'Kaloscope Adapter · Shared artist-v1 encoder'
                node.setdefault('properties', {})['Node name for S&R'] = node['type']
                # Socket stays KALOSCOPE_MODEL: links and original loader interoperability are unchanged.
        return result
    for nid, node in result.items():
        if not isinstance(node, dict) or node.get('class_type') != 'KaloscopeModelLoader':
            continue
        consumers = [v['class_type'] for v in result.values() if isinstance(v, dict) and 'class_type' in v
                     and any(isinstance(x, list) and len(x) == 2 and str(x[0]) == str(nid)
                             for x in v.get('inputs', {}).values())]
        if consumers and all(t == 'KaloscopeZ3EncodeReference' for t in consumers):
            node['class_type'] = 'KaloscopeZ3EncoderLoader'
            node['_meta'] = {'title': 'Kaloscope Adapter · Shared artist-v1 encoder'}
    return result
