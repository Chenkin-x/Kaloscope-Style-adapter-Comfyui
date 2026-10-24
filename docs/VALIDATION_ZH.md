# 节点验证（2026-10-24）

- **56项Python回归通过**：Krea2/MOSS残差与独立CFG、模型导出回读、名称查询/混合/搜索、HTTP迁移、路径别名、旧UI/API转换、共享编码器接口、参考图精度与设备兼容、裁剪/缩放选项及公开命名检查。
- **前端迁移钩子测试通过**：调用实际扩展函数，验证图替换、已转换缺失节点清理、无关工作流跳过与错误传播；这是Node环境测试，未宣称真实浏览器完整运行。
- **10个节点接口**的注册ID、显示名称、输入/输出类型和本机模型下拉选项均不包含研究标签。
- **三份真实发布权重**均通过新名称加载，参数量与训练步数正确；风格库名称查询返回有限256维特征。文件来自已有模型目录，没有复制或重命名权重。见[模型验证](validation/public_model_validation.json)。
- **两份真实编码器**（纯风格版、v1 classifier版）各两张固定RGB图，经512中心裁剪，在CPU FP32下新旧加载器输出逐位一致，最大绝对误差0；原加载器bundle可直接输入新参考编码节点。见[编码一致性](validation/encoder_parity.json)。
- **真实ComfyUI ModelPatcher**在CPU完成编码器加载、参考编码与卸载。
- **8套UI和8套API**通过当前节点接口、模型选项、端口类型、双向链接与UI/API一致性检查。LoadImage中的 `reference.png` 是用户选图占位项，未当作已安装资源验证。
- 原反推包与新包按两种顺序导入，均为10+10节点且无ID冲突；编码器显示别名可不同，实际指向同一批模型目录，socket兼容。

本轮完成原生MOSS的GPU单步前向，未进行GPU完整出图、低显存压力测试或性能基准。两图CPU一致性不代表所有平台精度或人类偏好保证。更新本机文件后，需要重启ComfyUI并刷新页面才能启用新注册和迁移前端。

原始结果见 [validation/](validation/)。模型权重和数学计算保持不变；旧标识只在内部读取兼容层使用。

## 参考图编码精度兼容（3.1.1）

- 原路径对所有CUDA设备强制启用BF16 autocast。用户日志显示卷积实际收到BF16输入和FP32偏置；仅凭日志不能确定其Windows环境中是哪项后端或扩展导致自动转换失效。本机Linux正常路径未自然复现此错误。
- 修复按编码器实际设备检查BF16支持，将输入与存储精度对齐；仅针对已识别的混合精度类型冲突关闭autocast重试，其余图像沿用回退精度。标准编码器回退到FP32，不修改与反推节点共享的权重。OOM和形状错误仍直接抛出。
- 新增5项回归覆盖共享FP16/BF16权重、外层autocast隔离、卷积类型错误恢复、BF16不可用设备和非精度异常传播。CUDA用例已实际执行，未跳过。
- 使用真实artist-v1权重和发布参考图，在H100 / PyTorch 2.13.0上验证两张图的512裁剪编码：正常BF16输出与3.1.0逐位一致；向实际patch卷积注入与用户同类的精度冲突后，旧版报错，新版恢复，输出与显式FP32逐位一致。该检查不等于在用户Windows整合包中完成出图验证。

记录：[真实权重验证](validation/reference_precision.json)、[完整回归](validation/regression_tests.txt)。精度隔离处理参照[PyTorch AMP文档](https://docs.pytorch.org/docs/2.9/amp.html)，不依赖全局修改PyTorch或ComfyUI设置。

## 参考图完整缩放（3.2.0）

- `resize_mode` 为可选输入。`center_crop` 保持既有512中心裁剪；`resize` 将完整画面缩放到512×512，不裁剪、不补边，非正方形图片会改变比例。
- 新增3项回归验证横图/竖图边缘内容保留、节点选择实际影响特征且旧调用保持默认、非法模式拒绝。真实编码器在仅缩放模式下输出两组有限256维特征，默认模式与旧版保持逐位一致。

## 参考编码器部分卸载修复（3.2.1）

- 后续日志明确显示输入在CUDA、卷积权重在CPU。使用真实ComfyUI `ModelPatcher` 对发布编码器执行部分卸载，在 `NO_VRAM` 模式下复现了3.2.0的完整错误顺序：先是BF16输入与FP32偏置冲突，精度回退后出现CUDA输入与CPU权重冲突。之前仅处理精度的修复未解决该根因。
- DINO编码器采用普通PyTorch层，不能使用Comfy原生算子的按需权重搬运路径。现在请求完整加载，并检查全部参数与缓冲区；若低显存策略或部分加载缓存仍留下异设备张量，通过patcher完成加载，保持卸载和显存记账一致。没有全局禁用ComfyUI的低显存模式，也未修改共享权重精度。
- 新增4项回归覆盖分散的卷积/投影权重、忽略完整加载请求的缓存路径、单独留在CPU的缓冲区及CPU编码。全部CUDA用例已执行。
- 真实权重验证覆盖 `NO_VRAM` / `LOW_VRAM` × 中心裁剪 / 完整缩放，共4组；所有参数及缓冲区均在CUDA，输出与完整加载基线逐位一致。完整卸载后重新编码仍逐位一致，CPU模式输出有限，权重保持FP32。该检查使用真实内存管理器，不注入卷积故障，但未在用户Windows环境完成验证。

记录：[真实加载与卸载验证](validation/reference_devices.json)。对应官方实现：[加载策略](https://github.com/Comfy-Org/ComfyUI/blob/c194dd00cd42aa18d9dbf27d977bf6b85d9ea565/comfy/model_management.py)、[ModelPatcher](https://github.com/Comfy-Org/ComfyUI/blob/c194dd00cd42aa18d9dbf27d977bf6b85d9ea565/comfy/model_patcher.py)。

## MOSS 官方核心加载

- 示例使用内置UNETLoader、CLIPLoader（moss_image）和VAELoader，模型选项已核对官方文件清单。旧HF加载器标记为兼容节点。
- 当前核心与旧HF模型的真实代码，在三层微型网络、两种文本长度、三个时间步及底模/风格/跳层三种分支共18组CPU FP32比较中，最大绝对误差为1.70e-6。核心输入sigma，旧HF输入sigma×1000；节点检查这一区别，防止错误采样补丁导致静默失真。
- 真实MOSS Base和发布adapter权重经官方核心加载，在CUDA BF16下完成42层风格注入前向，输出有限；关闭风格与底模逐位一致，启用风格后平均绝对差为0.13606。
- 真实权重验证使用现有HF张量分片，经官方核心state-dict加载器读取；文本为随机特征，latent为8×8。该验证没有宣称完整CLIP文本编码、VAE解码、HTTP成图或视觉质量评估通过。

记录：[原生数值比较](validation/moss_native_parity.json)、[实际权重前向](validation/moss_native_weights.json)、[官方源码与模型文件](validation/moss_native_sources.json)。本机运行中的旧ComfyUI未升级或重启。
