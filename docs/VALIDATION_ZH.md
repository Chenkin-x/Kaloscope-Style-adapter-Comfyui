# 节点验证（2026-10-24）

- **49项Python回归通过**：Krea2/MOSS残差与独立CFG、模型导出回读、名称查询/混合/搜索、HTTP迁移、路径别名、旧UI/API转换、共享编码器接口、参考图精度兼容及公开命名检查。
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

## MOSS 官方核心加载

- 示例使用内置UNETLoader、CLIPLoader（moss_image）和VAELoader，模型选项已核对官方文件清单。旧HF加载器标记为兼容节点。
- 当前核心与旧HF模型的真实代码，在三层微型网络、两种文本长度、三个时间步及底模/风格/跳层三种分支共18组CPU FP32比较中，最大绝对误差为1.70e-6。核心输入sigma，旧HF输入sigma×1000；节点检查这一区别，防止错误采样补丁导致静默失真。
- 真实MOSS Base和发布adapter权重经官方核心加载，在CUDA BF16下完成42层风格注入前向，输出有限；关闭风格与底模逐位一致，启用风格后平均绝对差为0.13606。
- 真实权重验证使用现有HF张量分片，经官方核心state-dict加载器读取；文本为随机特征，latent为8×8。该验证没有宣称完整CLIP文本编码、VAE解码、HTTP成图或视觉质量评估通过。

记录：[原生数值比较](validation/moss_native_parity.json)、[实际权重前向](validation/moss_native_weights.json)、[官方源码与模型文件](validation/moss_native_sources.json)。本机运行中的旧ComfyUI未升级或重启。
