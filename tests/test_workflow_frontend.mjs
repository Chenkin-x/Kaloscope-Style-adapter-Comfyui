// Run with Node; exercises the real extension hook without a browser or network.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const source = fs.readFileSync(new URL('../web/workflow_migration.js', import.meta.url), 'utf8').replace(/^import .*;\n/gm, '');
let extension;
let requests = 0;
const context = {
    app: { registerExtension(e) { extension = e; } },
    api: { async fetchApi(path, options) {
        requests++;
        assert.equal(path, '/kaloscope-adapter/migrate-workflow');
        assert.equal(options.method, 'POST');
        const graph = JSON.parse(options.body);
        graph.nodes[0].type = 'KaloscopeStyleAdapterLoader';
        return { ok: true, async json() { return graph; } };
    } },
    Set, Array, Object, JSON, Error
};
vm.runInNewContext(source, context);
const old = 'KaloscopeLegacyLoader';
const graph = {nodes: [{id: 1, type: old}], links: [], extra: {test: true}};
const missing = [old, 'UnrelatedMissingNode'];
await extension.beforeConfigureGraph(graph, missing);
assert.equal(graph.nodes[0].type, 'KaloscopeStyleAdapterLoader');
assert.deepEqual(missing, ['UnrelatedMissingNode']);
assert.equal(graph.extra.test, true);
await extension.beforeConfigureGraph({nodes: [{type: 'CLIPTextEncode'}]}, []);
assert.equal(requests, 1);
context.api.fetchApi = async () => ({ok: false});
await assert.rejects(() => extension.beforeConfigureGraph(graph, []), /migration failed/);
console.log('PASS: extension migration, missing-node cleanup, unrelated graph bypass, failure propagation');
