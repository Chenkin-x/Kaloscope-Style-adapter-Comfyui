# Z3 中间检查点：ComfyUI 风格迁移

本文保留 step12006 时期的实验设置和历史验证，不作为当前发布模型的默认参数。当前使用 Raw63006 / Turbo33000 / MOSS8500，优先阅读 [主说明](README.md) 并导入 `examples/release/`。

## 安装与共享编码器

按[独立包安装说明](README.md#安装)安装本包，可与原版画师反推插件并存。编码器复用 `models/kaloscope/<已有目录>/`，既可使用新 `KaloscopeZ3EncoderLoader`，也可将原 `KaloscopeModelLoader` 的输出直接连到参考图编码节点。无需复制模型。历史合并扩展版需按[迁移指南](docs/MIGRATION_ZH.md)去除重复注册。

DINO 必须匹配训练使用的 artist-v1；同为256维不代表特征空间兼容。VAE使用官方 `VAELoader` 选择 `qwen_image_vae.safetensors`，没有随adapter微调。

## 开始测试

将 `examples/z3/z3_img2img_workflow.json` 拖入 ComfyUI：

1. 在两个 Load Image 节点分别选择参考画风图和内容图。
2. 确认 Krea2、文本编码器、VAE、DINO 和 Z3 检查点文件选择正确。
3. 输入内容描述，通常不添加画师名称，negative 留空。
4. 运行队列。输出保存在 ComfyUI 的 `output/KaloscopeZ3/` 下。

默认：512×512、Euler 28 步、起始 sigma 0.85、文本 CFG 2、风格 guidance 5.5、adapter strength 1。
`Kaloscope Z3 Resolution Aware Sigmas` 按训练评估的时间表生成 29 个 sigma，接入原生 `SamplerCustom`。
固定 seed 方便横向比较；ComfyUI 与训练使用的随机数设备、BF16 运算和文本编码细节不同，不承诺同 seed 逐像素一致。

也可以把 Apply 节点输出接普通 `KSampler`，其 `cfg` 是**文本 CFG**。
普通 KSampler 的 denoise/scheduler 不等同于这里的精确起始 sigma 时间表。

仅凭参考图文生图：把内容图和 VAE Encode 分支替换为 `Empty Hunyuan Video 1.0 Latent`，设置
width=512、height=512、length=1、batch_size=1，并把起始 sigma 改成 1.0。
这是 Krea2 使用的 16 通道、单帧五维 latent；不要用普通 SD1.5 的四通道 Empty Latent。
文生图和其他分辨率可用于探索，当前训练评估基线仍是上述 512 图生图设置。

## 节点与参数

| 节点 | 用法 |
| --- | --- |
| Z3 Checkpoint Loader | 从 `models/kaloscope_z3/` 递归列出 `.safetensors/.pt/.pth`；输出 step 等信息。文件被替换后按修改时间和大小使缓存失效。 |
| Z3 Reference Encode | 接原有 Kaloscope Model Loader 的 DINO 和参考 IMAGE；输出每张图的 256 维归一化 projector 特征。采用训练匹配的中心裁方、PIL bicubic 512 缩放与 ImageNet 归一化。 |
| Z3 Blend Styles | 可接三组特征，拼接后按权重融合。`1,3` 表示两张参考的权重；留空为等权。多参考属于推理扩展，尚无专门的多画师融合质量保证。 |
| Z3 Apply Style | 将 adapter 应用到原生 Krea2 MODEL 的克隆，输出继续接 ComfyUI 采样器。输入多行特征时默认等权融合为一个风格，全批图使用该风格。 |
| Z3 Resolution Aware Sigmas | 复现训练评估的时间表，与原生 SamplerCustom 配合。 |

优先调 `style_guidance`，例如 3、5.5、8；越大越强调参考风格，也可能出现颜色或结构失真。
`adapter_strength` 改变每层残差注入量，建议先保持 1。
任一个参数为 0 即绕过风格 adapter，便于同 seed 比较基础模型。
多参考例：两个 Reference Encode 输出接 Blend 的 features/features_b，权重 `0.7,0.3`，再接 Apply。

画风编码器用 `cuda` 最接近训练计算方式；显存紧张时可选 `cpu`，速度和浮点结果会有区别。
原插件的模型加载节点保持其既有设备管理行为；新 adapter 使用 ComfyUI 的附加模型加载/卸载机制。

## 中间版本更新

可直接复制训练生成的 `checkpoint_stepXXXXXXXX.pt` 到 `models/kaloscope_z3/`，也支持包含完整训练状态的 `latest.pt`。
推荐导出仅含权重和架构元数据的 safetensors，避免携带优化器、随机数状态和多余体积：

```bash
python scripts/export_z3.py /path/to/checkpoint_step00012006.pt /path/to/ComfyUI/models/kaloscope_z3/z3_step12006.safetensors
```

导出与读取均不修改训练检查点。新文件放入目录后刷新节点文件列表或重启 ComfyUI，再在 Loader 中切换。
PyTorch 文件使用 `weights_only=True`；架构、28 层注入位置、4 个共享 bank、state dict 和权重有限值均严格检查。
不支持将 patch-token 实验、LoRA 或其他架构的检查点误作 Z3 adapter 加载。

## 与训练实现的对应关系

独立风格 CFG：`v = v_uncond + text_cfg*(v_text-v_uncond) + style_cfg*(v_style-v_text)`。
普通的单一 CFG 无法表达该公式；本节点额外计算关闭风格的文本分支，每步通常需要三次模型前向。
负向分支不注入风格；已有其他自定义 CFG 补丁时会明确报错，避免悄悄覆盖。

adapter 在每个完整 transformer block 后注入图像 token；保留 Krea2 原生的时间条件归一化、scale/shift/gate。
当前 Z3 检查点没有独立的 adapter AdaLN，因此节点也不添加未经训练的 AdaLN。
ComfyUI 图文 token 顺序不同于训练实现，使用原生 `img_slice` 定位图像，绝不修改文本 token。
不支持同时开启 Krea2 原生 reference-latent ICL，也不支持连续叠加两个 Apply 节点；请先融合特征。
BF16 原版 Krea2 是验证目标。FP8/GGUF、编译、第三方模型替换、ControlNet 等组合尚未验证。

## 验证与后续合并

数值与实际运行证据见 `evidence/z3/`。单步一致性不代表大规模风格质量评估结论。

- 5 项 Z3 数值测试、24 项原插件模型加载/节点/API 回归测试通过。
- step 12006 的 ComfyUI 与训练后端单步风格分支余弦相似度 0.99999845，相对 RMSE 0.2824%；BF16 运算路径存在细微差异。
- 配套 DINO 与训练编码器同图特征余弦相似度 0.999959；权重匹配，数值未宣称逐位一致。
- 撤销模型补丁后，基础模型输出与未注入分支逐位一致。
- 真实 ComfyUI HTTP 队列验证通过，完整工作流和关闭风格的基线均生成 512×512 图像；两者确实不同。
- 验证使用单张 H100 和 BF16 原版 Krea2，Windows、低显存 offload、量化模型组合未实机验证。

上述记录来自早期合并版。当前独立包使用 `python -m unittest discover -s tests -v` 验证；旧模型加载/分析测试随原插件维护，当前包不再包含这些模块。

## 2026-10-10：名称条件与分辨率采样

新增 `Kaloscope Z3 Artist Name Style`：选择 `z3_artist_v1_recent.safetensors`，输入画师规范名称，输出的风格特征接到 Apply Style。无需加载 DINO 或提供参考图。名称清单在 `models/kaloscope_style_banks/z3_artist_v1_recent.artists.txt`；不认识的名称会报错，可改用参考图。

这是训练作品原型查询，不是将分类头直接转换成生成模型，也不是 Krea2 文本编码器新增知识。原型库来自 v1 最近训练年份组的 44,129 位画师、1,313,015 个作品族，每族一幅代表作；先投影到 256 维并逐图归一化，再按画师求均值归一化。8,645 位画师不足 8 个作品族，风格稳定性需实际观察。缓存采用旧编码流水线的短边缩放/中心裁剪，与 Z3 的先裁剪/双三次缩放略有差异；此版用于验证名称条件。未将 dev/test 作品加入原型库。画师多时期风格可能被均值混合，后续可按年份/聚类提供多个原型。

`Kaloscope Z3 Resolution Aware Sigmas` 新增可选 `latent` 输入，连接采样器使用的同一 latent，按图像 token 数计算 raw 模型时间偏移。512×512 与旧时间表逐位一致；不连接时仍使用旧 512 设置，兼容已有工作流。新示例已连接。此修正不能保证未经训练的大尺寸和宽高比质量。

新增示例 `examples/z3/Kaloscope_Z3_artist_name_txt2img_step12006.json`；主模型、adapter 权重不变。当前数值/接口验证通过 8 项 Z3 测试和 24 项原节点回归测试。完整出图及质量结果以新增验证记录为准。

Turbo 调试：Sigmas 的 `model_variant` 选 `turbo`，使用固定 mu=1.15、8 步、start_sigma=1。普通 SamplerCustom CFG 先设 1，Apply Style 的 style_guidance 先设 1，adapter_strength 可比较 0.25/0.5/0.75/1。这只是迁移诊断配置，这是早期 Raw→Turbo 迁移的历史诊断记录；当前发布使用专门适配后的 Turbo33000，见主说明。Raw 默认无需更改。
