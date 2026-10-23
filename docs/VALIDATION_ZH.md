# 独立生图包验证（2026-10-23）

- 30项单元/接口回归通过：adapter公式、名称混合与搜索、共享目录/检查点、错误权重拒绝、工作流迁移、私有导入与路由。
- 两份真实encoder权重（纯风格版、v1 classifier版），各用两张固定RGB图，经512中心裁剪，在CPU FP32下新旧加载器输出逐位一致，最大绝对误差0；两份权重的风格输出也一致。原loader bundle直接输入新Reference Encode，无格式转换或文件复制。见 [encoder_parity.json](validation/encoder_parity.json)。
- 真实ComfyUI ModelPatcher在CPU完成加载、参考编码和卸载；输出为有限的归一化256维特征。
- 本机反推包与独立生图包分别按两种顺序导入，10+10节点无ID交集，模型目录列表与socket一致。
- 8套UI与8套API工作流通过required字段、文件选项、输出类型、双向链接、UI/API一致性检查，不再依赖旧VAE别名。

本轮未重启运行中的ComfyUI，也未重新做GPU完整出图或性能基准。早期出图与训练数值证据仍在统一发布包中，本次验证不扩大这些证据的适用范围。CPU两图一致性不构成所有平台精度、低显存卸载或人类偏好保证。

详细记录在 [validation/](validation/)。
