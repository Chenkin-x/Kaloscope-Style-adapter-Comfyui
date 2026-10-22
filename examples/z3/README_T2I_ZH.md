# 文生图示例

三份工作流均从空白 latent 和随机噪声生成新图，不需要内容图。

## 单张参考图 + 内容提示词

打开 `Kaloscope_Z3_reference_txt2img_step12006.json`。替换唯一的 Load Image 为画风参考图，在正向提示词中描述希望生成的内容。
Z3 用参考图的 DINO 特征控制风格，不需要写画师名称。

默认：512×512、Euler 28 步、文本 CFG 2、风格 guidance 5.5、adapter strength 1、起始 sigma 1.0。
原图生图示例的起始 sigma 为 0.85；这里改成 1.0，使生成从纯噪声开始。
若内容描述遵循不足，可先尝试文本 CFG 3–4；风格不足可试 style guidance 7–8，观察失真再调整。
这些调参值是探索起点，文生图尚未经过与图生图同规模的质量评测。

## 仅艺术家文本 + 内容提示词

打开 `Krea2_artist_text_txt2img.json`，在正向提示词中写内容和艺术家名字，例如：

```text
A portrait of a woman surrounded by flowers, decorative botanical border, in the style of Alphonse Mucha.
```

默认：512×512、Euler 28 步、CFG 5.5、denoise 1.0。不需要参考图，不加载 Z3 或 DINO。
效果来自 Krea2 基础模型原有的文本条件能力，取决于模型是否认识该艺术家，不能据此推断 Z3 学会了名字控制。
原有 Z3 训练没有名称输入。现已增加独立名称原型查询，使用下面的新工作流即可调用已收录画师的风格。

## 主权重名称

`ComfyUI/models/diffusion_models/krea2_raw.safetensors` 和原来的 `krea2_raw_z3.safetensors` 都链接到同一个原版 raw 文件。
不存在单独训练过的“Z3 主模型”。Z3 训练只更新独立 adapter，主模型始终冻结；安装时已核对主模型 SHA256 与训练版本完全相同。
新工作流采用不带 z3 后缀的名称。其他机器可在模型下拉框中选择自己的同版 raw 文件，文件名本身不影响推理。

为沿用训练条件，Z3 默认配合同版 Krea2 raw。冻结主模型不意味着 adapter 自动兼容所有底模；Turbo、其他微调或量化版本需要单独验证。
单帧空白 latent 使用 ComfyUI 原生 `Empty Hunyuan Video 1.0 Latent`，length 固定为 1，输出的是一张图。

工作流已通过本机 ComfyUI 原生输入/连接校验，并检查了空白 latent 和 sigma 输出；本次未重新执行完整文生图质量评测。

## Kaloscope 画师名称 + 内容提示词（新增）

打开 `Kaloscope_Z3_artist_name_txt2img_step12006.json`。在 Artist Name Style 节点输入规范画师名称，内容提示词只描述新图内容。此工作流无参考图、不加载 DINO，查询已安装的 44,129 位画师原型库并调用 Z3 adapter。名称清单位于 `models/kaloscope_style_banks/z3_artist_v1_recent.artists.txt`。这与基础模型自己的艺术家文本提示不同；不能据此声称底模已学会这些名字。

Z3 示例的 Sigmas 节点均已接入生成 latent，自适应分辨率时间偏移。旧工作流若要改变尺寸，需要补接该线；512 默认保持一致。尺寸自适应修正不等于多分辨率训练已完成。

## 多画师加权混合

导入 `Kaloscope_Z3_weighted_artists_txt2img.json`，使用 `Kaloscope Z3 Weighted Artist Mix` 节点。示例选择第三阶段 best `z3_stageC_best_step63006.safetensors`，style guidance 为 3.5。

`artist_tags` 示例：`rolua, (wanke:0.8), (nineo:0.5)`。裸名称权重为 1，支持英文/中文逗号和换行分隔；画师名自带括号可直接写，如 `chahan_(fried_rice0614)`，或加权写为 `(chahan_(fried_rice0614):0.8)`。支持转义括号。此处只支持名称和显式 `(名称:权重)`，不使用文本编码器的隐式括号强调规则。

权重为非负相对比例，0 表示不贡献；至少一个权重必须大于 0。各画师原型先归一化，加权相加后再次归一化，与多个 Artist Name Style 接 Blend Styles 等价。上述示例比例约为 43.48%、34.78%、21.74%；同时放大全部权重不会增强整体风格，整体强度在 Apply 节点调整。`artist_info` 输出解析后的名称、权重、占比及原型样本数。

将 `style_features` 接入现有 Apply Style 的同名输入。未找到的画师会明确报错，不会静默忽略或改成近似名字。混合在风格特征空间进行，不保证视觉效果严格按数值比例叠加。
