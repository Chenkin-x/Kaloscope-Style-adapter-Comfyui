# Kaloscope Style Adapter for ComfyUI

Krea2 Raw / Turbo 与 MOSS 风格适配节点，支持画师名称、加权多画师和参考图条件。
如需使用艺术家风格图像反推节点，请使用： [comfyui-kaloscope](https://github.com/spawner1145/comfyui-kaloscope)。

## 训练代码

适配器训练实现见 [Kaloscope Adapter Trainer](https://github.com/Chenkin-x/Kaloscope-adapter-trainer)，训练配置与运行方式请参考该仓库 README。

本项目的三个入口：

| 仓库 | 用途 |
| --- | --- |
| [模型权重与模型卡](https://huggingface.co/heathcliff01/Kaloscope-style-adapter) | 下载 Raw、Turbo、MOSS 适配器、配套风格资源与示例工作流 |
| [训练代码](https://github.com/Chenkin-x/Kaloscope-adapter-trainer) | 研究与扩展适配器训练流程 |
| [ComfyUI 推理节点](https://github.com/Chenkin-x/Kaloscope-Style-adapter-Comfyui) | 使用画师名称、多画师融合或参考图进行风格生成 |

希望直接体验生成效果，可以使用已发布模型与本仓库的推理节点；希望训练或扩展适配器，请查看训练代码仓库。模型、代码及其依赖分别适用对应的许可条款。


## 模型下载与效果展示

模型现已发布于 **[Hugging Face · Kaloscope Style Adapter](https://huggingface.co/heathcliff01/Kaloscope-style-adapter)**。本代码仓库继续保持私有，安装节点需要仓库访问权限；模型权重、配套资源和模型卡公开提供。

| 模型 | 底模 | 权重下载 | 模型卡 |
| --- | --- | --- | --- |
| Krea Kaloscope Raw | Krea-2-Raw | [下载](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/models/Krea-Kaloscope-Raw.safetensors?download=true) | [查看](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/blob/main/model_cards/raw/README.md) |
| Krea Kaloscope Turbo | Krea-2-Turbo BF16 | [下载](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/models/Krea-Kaloscope-Turbo.safetensors?download=true) | [查看](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/blob/main/model_cards/turbo/README.md) |
| Kaloscope MOSS | MOSS-Image-Base | [下载](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/models/Kaloscope-MOSS.safetensors?download=true) | [查看](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/blob/main/model_cards/moss/README.md) |

名称及名称融合需下载 [artist-v1 风格库](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/style-bank/artist-v1.safetensors?download=true)；参考图入口需下载 [artist-v1 编码器目录](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/tree/main/encoder/artist-v1) 内的配置、权重和许可。原版底模、文本编码器与 VAE 另行安装。

也可下载模型仓库后，用其安装脚本复制到兼容目录：

```bash
hf download heathcliff01/Kaloscope-style-adapter --local-dir Kaloscope-style-adapter
python Kaloscope-style-adapter/install_models.py --comfyui "/path/to/ComfyUI" --variant turbo --with-encoder
```

`--variant` 可选 `raw`、`turbo`、`moss` 或 `all`；纯名称入口可省略 `--with-encoder`。脚本只复制已下载的模型文件，校验权重，不安装节点或底模，不覆盖内容不同的已有文件。插件的内部目录和节点 ID 保留兼容名称，公开模型名称与此独立。

### 画师名称生成

![画师名称生成](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/images/artist-name.jpg)

### 多画师融合

![多画师融合](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/images/artist-mix.jpg)

### 参考图生成

![参考图生成](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/resolve/main/images/reference.jpg)

以上图墙直接拼接现有 ComfyUI 原图，仅保留生成方式与艺术家名称。名称和融合展示来自 Turbo，参考图展示包含 Raw 与 Turbo 输出，不作为 MOSS 效果或同条件基准。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Chenkin-x/Kaloscope-Style-adapter-Comfyui.git
# 使用运行 ComfyUI 的 Python；便携版请换成 python_embeded/python.exe
python -m pip install -r Kaloscope-Style-adapter-Comfyui/requirements.txt
```

重启 ComfyUI 并刷新浏览器。本包与原版 [comfyui-kaloscope](https://github.com/spawner1145/comfyui-kaloscope) 可同时安装：原版负责画师反推、特征分析，本包负责 adapter 生图。
若旧包是此前把 adapter 合并进去的扩展版，需先移除其中重复的 `KaloscopeZ3*` / `KaloscopeMOSS*` 注册和旧画师搜索前端，或将其备份后换回原版；具体见[迁移说明](docs/MIGRATION_ZH.md)。不要同时启用两个含相同 adapter 节点 ID 的版本。
ComfyUI 需要提供原生 Krea2 / MOSS 模型支持；本包不替代底模、文本编码器或 VAE。

```text
ComfyUI/models/
  kaloscope_z3/                 # 兼容目录；放置 Krea-Kaloscope-Raw/Turbo 或 Kaloscope-MOSS.safetensors
  kaloscope_style_banks/        # artist-v1.safetensors
  kaloscope/artist-v1/          # 参考图编码器：config.json + model.safetensors + LICENSE.md
  diffusion_models/            # Krea2 Raw / BF16 Turbo，或对应 MOSS 底模
  text_encoders/               # 对应底模的文本编码器
  vae/                         # 对应底模的 VAE
```

代码仓库包含名称 CSV，**不包含模型权重**。名称条件需要匹配的风格库；参考图条件需要匹配的 artist-v1 编码器。仅复制 CSV 不能替代它们。

## 与原插件共享 Kaloscope v3 权重

**同一份模型，无需改名、转换或再次下载。** 新加载器 `KaloscopeZ3EncoderLoader` 与原 `KaloscopeModelLoader` 都读取 `ComfyUI/models/kaloscope/<目录名>/`，沿用 `config.json` 的模型名称和 `checkpoint` 字段。若原插件已使用 `v1-artist-classifier`，直接在新加载器选择该目录，无需再复制一份到 `z3_artist_v1`。示例中的目录名只是默认值。

支持本项目训练匹配的 artist-v1 DINOv3 ViT-B/16 + 256维 projector：纯风格版与附带画师分类头的 v1 classifier 版均可。新加载器只读 backbone/projector，不加载分类头；不需要 `class_mapping.csv`。不支持的架构或缺失的 projector 会报错；同为256维但重新训练过的 projector 即使能加载，也**不保证风格空间适配**。

也可以把原版 **Kaloscope Model Loader** 的 `KALOSCOPE_MODEL` 输出直接接到本包 **Z3 Reference Encode**。在同时做画师反推和生图的工作流中，共用这一个加载器即可避免各自加载模型；使用新加载器则由 ComfyUI 管理编码器加载/卸载。新加载器的精简输出仅用于生图，不能反向接画师分类节点。

纯名称/加权名称生图只需要风格库和 adapter，不需要运行 DINO 编码器。代码不自动下载模型，也不创建另一套模型目录。

## 发布工作流（2026-10-23 独立包更新）

推荐从 [Hugging Face 的 workflows 目录](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/tree/main/workflows) 导入，已匹配本次公开权重名称和独立编码器加载节点。Raw、Turbo 各提供名称、参考图、加权画师三种入口；MOSS 提供名称与参考图，共 8 套 UI 工作流及对应 API JSON。本仓库 `examples/release/` 保留原始检查点文件名，使用公开权重时请在节点中重新选择相应文件；更早的示例保留历史参数。

Krea2底模和文本继续使用原生 `UNETLoader` / `CLIPLoader(type=krea2)`；采样、解码、保存复用通用节点。`KaloscopeZ3Sigmas`保留Raw分辨率调度与Turbo固定μ=1.15。

MOSS直接使用官方插件 `MOSSImageHFLoader`，将路径改为本机MOSS-Image-Base目录。参考[官方MOSS接口说明](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/README.md)与[官方Krea2 VAE说明](https://github.com/Comfy-Org/docs/blob/main/tutorials/image/krea/krea-2.mdx)。

## 画师快速检索

1. 添加 **Kaloscope Z3 Artist Style** 或 **Kaloscope Z3 Artist Mix**，选择 `bank`。
2. 点击 **查找画师 / Search artists**，输入名称、部分关键词、已验证别名或近似拼写。
3. 点击结果，或用 ↑/↓ 和 Enter 选择。单画师节点写入 `artist_name`；混合节点追加到 `artist_tags`，保留已有权重。
4. 混合权重格式为 `fkey, (rolua:0.8)`。括号用于消歧的完整名称会保留；含特殊字符的名字在追加时自动转义。

内置 [data/artists.csv](data/artists.csv) 含 **44,129 位画师、105 个已验证别名**，并附原型作品族数量和特征集中度。
数据来自当前 `z3_artist_v1_recent` 风格库；[来源与校验信息](data/artists.json) 随包提供。

- 搜索结果以**当前选中的风格库实际包含的画师**为准。换用其他风格库时不会把内置 CSV 的名字错误当成该库已支持的名字。
- 未选择风格库时可以浏览内置目录；生成仍需安装并选择风格库。
- 精确名称与前缀优先，再综合关键词覆盖率和拼写近似度排序，避免长昵称偶然包含几字母就压过明显的拼写候选。大小写与空格/下划线可兼容，已验证别名可直接用于节点。
- 模糊匹配只给候选，**不会在运行时静默替换成其他画师**。例如 `feiqizi_(fkey)` 是角色名，不会自动替换为 `fkey`。
- 不移除括号后缀；同名但不同后缀的画师单独展示。按名字搜索不是按画风相似度搜索。
- `families` 表示构建名称原型使用的作品族数量，**不是全部训练图像数量**。
- 弹窗可下载当前所选风格库的 CSV。全部搜索在本机完成，无需外部服务或新增搜索依赖。

![画师模糊检索界面](docs/artist-search.png)

## 推理节点

| 节点 | 用途 |
| --- | --- |
| `KaloscopeZ3EncoderLoader` | 从共享 `models/kaloscope/` 加载参考图编码器 |
| `KaloscopeZ3Loader` | 加载 Z3 adapter，输出检查点信息 |
| `KaloscopeZ3ArtistStyle` | 从风格库选择一个画师 |
| `KaloscopeZ3ArtistMix` | 多画师加权混合，支持搜索追加 |
| `KaloscopeZ3EncodeReference` | 用匹配的 DINOv3 artist-v1 编码参考图 |
| `KaloscopeZ3BlendFeatures` | 融合多组参考特征 |
| `KaloscopeZ3Apply` | 克隆 MODEL 并应用 Krea2 adapter |
| `KaloscopeZ3Sigmas` | 配合原生 `SamplerCustom` 使用对应采样时间表 |

另有 `KaloscopeMOSSBaseLoader` 与 `KaloscopeMOSSZ3Sigmas`，共10个生图节点；MOSS 细节见 [MOSS 使用说明](README_MOSS_Z3_ZH.md)。本包不注册画师分类、聚类和特征可视化节点，不包含 LSNet。
节点位于 `Kaloscope Adapter` 分类；原插件节点继续位于自己的分类。
旧版 Z3 示例和历史参数见 [README_Z3_ZH.md](README_Z3_ZH.md)，其早期 step12006 参数不是所有检查点的统一默认推荐。

本次 Raw / Turbo 配对验证所用设置，可作为同系列检查点的比较起点：

| 模型 | Euler 步数 | 文本 CFG | style_guidance | adapter_strength |
| --- | ---: | ---: | ---: | --- |
| Raw 63006 | 25 | 2 | 3.5 | 比较 1.0 与 0.8 |
| Turbo 从 63006 加训 | 8 | 1 | 1 | 比较 1.0 与 0.8 |

保持 seed、提示词、分辨率不变再比较强度。Fkey 的小样本测试中 0.8 常有更高风格余弦，但并非每个种子都提高；高余弦也不保证人体和构图更好。本次整理保留既有采样与残差注入计算。

## 本地接口与目录更新

```text
GET /kaloscope-adapter/artists?bank=z3_artist_v1_recent.safetensors&q=fky&limit=40
GET /kaloscope-adapter/artists/catalog.csv?bank=z3_artist_v1_recent.safetensors
```

省略 `bank` 返回内置目录；传入的 `bank` 必须存在于节点风格库列表。搜索最多返回 100 条，名称上限 160 字符。
名称索引按文件版本失效更新，不缓存模型权重。原有节点名称和输入输出保持兼容。

维护者可以从新风格库重新导出内置目录：

```bash
python scripts/export_artist_catalog.py /path/to/bank.safetensors --aliases /path/to/verified_aliases.json
```

别名 JSON 格式为 `{"mapping": {"old_name": "canonical_name"}}`；只应导入已核实的画师别名。

## 开发验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖名称与别名、模糊候选、同名消歧、角色名拒绝、风格库替换、实际支持列表、CSV 下载、非法路径与既有混合公式。同时覆盖独立导入、共享文件加载、原加载器 bundle、错误模型拒绝和工作流迁移。
前端按 [ComfyUI JavaScript 扩展接口](https://docs.comfy.org/custom-nodes/js/javascript_overview) 注册，搜索按钮不作为推理参数序列化。

## 许可

沿用上游 GPL-3.0，见 [LICENSE](LICENSE)。私有命名空间中的 DINOv3 源码保留其版权声明与[单独许可](kaloscope_style/_vendor/dinov3/LICENSE.md)；不将这些文件重新标为 GPL。来源与模块边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型权重按底模分别采用上游许可：Krea Kaloscope Raw / Turbo 使用 Krea 2 Community License，Kaloscope MOSS 使用 Apache-2.0，配套 DINOv3 编码器保留 DINOv3 License。具体授权、上游来源与归属声明见 [HF 组件许可](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/blob/main/LICENSE.md) 和 [NOTICE](https://huggingface.co/heathcliff01/Kaloscope-style-adapter/blob/main/NOTICE)，不把整个包统一改标为同一种许可。
