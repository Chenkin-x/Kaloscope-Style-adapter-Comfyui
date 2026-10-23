import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Kaloscope.Style.WorkflowMigration",
    async beforeConfigureGraph(graph, missingNodeTypes) {
        if (!Array.isArray(graph?.nodes) || !graph.nodes.some(n => n.type?.startsWith("Kaloscope"))) return;
        const response = await api.fetchApi("/kaloscope-adapter/migrate-workflow", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(graph)
        });
        if (!response.ok) throw new Error("Kaloscope workflow migration failed. Use the migration script or current examples.");
        const migrated = await response.json();
        const oldTypes = new Set(graph.nodes.map(n => n.type));
        const newTypes = new Set(migrated.nodes.map(n => n.type));
        Object.assign(graph, migrated);
        if (Array.isArray(missingNodeTypes)) {
            for (let i = missingNodeTypes.length - 1; i >= 0; i--) {
                if (oldTypes.has(missingNodeTypes[i]) && !newTypes.has(missingNodeTypes[i])) missingNodeTypes.splice(i, 1);
            }
        }
    }
});
