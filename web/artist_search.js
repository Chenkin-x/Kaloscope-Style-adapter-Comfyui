import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_INPUTS = { KaloscopeZ3ArtistStyle: "artist_name", KaloscopeZ3ArtistMix: "artist_tags" };
const MATCH_LABELS = { exact: "精确 / Exact", prefix: "前缀 / Prefix", substring: "包含 / Contains", fuzzy: "近似拼写 / Fuzzy", browse: "" };

function element(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
}

function openArtistSearch(node, inputName) {
    document.querySelector("dialog.kaloscope-artists")?.close();
    const widget = node.widgets?.find(w => w.name === inputName);
    if (!widget) return;
    const bank = node.widgets.find(w => w.name === "bank")?.value || "";
    const mix = inputName === "artist_tags";
    const dialog = element("dialog", "kaloscope-artists");
    dialog.setAttribute("aria-label", "查找画师 / Search artists");
    const header = element("div", "ks-header");
    header.append(element("h2", "", mix ? "添加画师 / Add artists" : "查找画师 / Search artists"));
    const close = element("button", "ks-close", "关闭 / Close");
    close.type = "button";
    close.onclick = () => dialog.close();
    header.append(close);
    const source = element("p", "ks-source", bank || "内置目录预览：请先安装并选择风格库 / Catalog preview: select a bank to generate");
    const input = element("input", "ks-query");
    input.type = "search";
    input.maxLength = 160;
    input.placeholder = "名称、别名、关键词或近似拼写 / Name, alias, keyword or typo";
    input.setAttribute("aria-label", "搜索画师 / Search artists");
    input.autocomplete = "off";
    input.spellcheck = false;
    input.value = mix ? "" : String(widget.value || "");
    const status = element("p", "ks-status");
    status.setAttribute("role", "status");
    const results = element("div", "ks-results");
    results.setAttribute("role", "listbox");
    results.setAttribute("aria-label", "画师搜索结果 / Artist results");
    input.setAttribute("aria-controls", "kaloscope-artist-results");
    results.id = "kaloscope-artist-results";
    const footer = element("div", "ks-footer");
    const download = element("a", "", "下载 CSV / Download CSV");
    download.href = api.apiURL("/kaloscope/style/artists/catalog.csv?" + new URLSearchParams({ bank }));
    download.download = "kaloscope_artists.csv";
    footer.append(download, element("span", "", "保留括号以区分同名画师 / Keep name suffixes"));
    dialog.append(header, source, input, status, results, footer);
    let timer, controller, active = -1, requestId = 0;
    let buttons = [];

    function setActive(index) {
        active = index;
        buttons.forEach((button, i) => {
            button.classList.toggle("ks-active", i === active);
            button.setAttribute("aria-selected", String(i === active));
        });
        if (buttons[active]) {
            input.setAttribute("aria-activedescendant", buttons[active].id);
            buttons[active].scrollIntoView({ block: "nearest" });
        } else input.removeAttribute("aria-activedescendant");
    }

    function choose(artist) {
        if (mix) {
            const escaped = artist.name.replace(/[\\(),:]/g, "\\$&");
            const previous = String(widget.value || "").trim();
            widget.value = previous ? previous + ", " + escaped : escaped;
        } else widget.value = artist.name;
        widget.callback?.(widget.value, app.canvas, node);
        node.graph?.change();
        node.setDirtyCanvas(true, true);
        if (mix) {
            status.textContent = `已添加 / Added: ${artist.name}`;
            input.focus();
        } else dialog.close();
    }

    async function search() {
        const id = ++requestId;
        controller?.abort();
        controller = new AbortController();
        buttons = [];
        results.replaceChildren();
        setActive(-1);
        status.textContent = "正在检索 / Searching…";
        try {
            const params = new URLSearchParams({ bank, q: input.value.trim(), limit: "40" });
            const response = await api.fetchApi("/kaloscope/style/artists?" + params, { signal: controller.signal });
            const data = await response.json();
            if (id !== requestId || !dialog.open) return;
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            status.textContent = `${data.total_artists.toLocaleString()} 位画师 · 显示 ${data.results.length} 条 / artists · results`;
            if (!data.results.length) status.textContent += " — 无匹配，请尝试更短关键词 / No matches";
            for (const [i, artist] of data.results.entries()) {
                const button = element("button", "ks-result");
                button.type = "button";
                button.id = `kaloscope-artist-result-${i}`;
                button.setAttribute("role", "option");
                button.append(element("strong", "", artist.name));
                button.append(element("span", "ks-detail", `${artist.families} 个原型作品族 / prototype works · ${MATCH_LABELS[artist.match]}`));
                if (artist.aliases.length) button.append(element("span", "ks-alias", "别名 / Alias: " + artist.aliases.join(", ")));
                button.onclick = () => choose(artist);
                buttons.push(button);
                results.append(button);
            }
            setActive(buttons.length ? 0 : -1);
        } catch (error) {
            if (id === requestId && error.name !== "AbortError") status.textContent = "检索失败 / Search failed: " + error.message;
        }
    }

    input.addEventListener("input", () => {
        clearTimeout(timer);
        ++requestId;
        controller?.abort();
        buttons = [];
        results.replaceChildren();
        setActive(-1);
        timer = setTimeout(search, 180);
    });
    input.addEventListener("keydown", event => {
        if (event.isComposing) return;
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            if (buttons.length) setActive((active + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length);
        } else if (event.key === "Enter") {
            event.preventDefault();
            buttons[active]?.click();
        }
    });
    dialog.addEventListener("close", () => {
        ++requestId;
        clearTimeout(timer);
        controller?.abort();
        dialog.remove();
    }, { once: true });
    document.body.append(dialog);
    dialog.showModal();
    input.focus();
    input.select();
    search();
}

app.registerExtension({
    name: "Kaloscope.Style.ArtistSearch",
    setup() {
        const link = element("link");
        link.rel = "stylesheet";
        link.href = new URL("./artist_search.css", import.meta.url).href;
        document.head.append(link);
    },
    nodeCreated(node) {
        const inputName = NODE_INPUTS[node.comfyClass];
        if (!inputName) return;
        node.addWidget("button", "查找画师 / Search artists", null, () => openArtistSearch(node, inputName), { serialize: false });
    },
});
