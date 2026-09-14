# 架构

[English](../en/architecture.md) | [中文](architecture.md)

`unisim-core` 拥有公共物理契约、后端能力、适配器工厂边界、引擎原生资源、一致性检查，以及预留的 benchmark case/result schema。它不依赖 UniLab、Hydra、Torch、Gymnasium、learner、runner 或任务代码。

UniLab 拥有 task/env/manager 生命周期、Hydra owner YAML、机器人资产、训练、checkpoint 和 sim2sim 策略 I/O。UniLab 将任务拥有的场景与随机化输入转换为 UniSim 契约。

引擎适配器使用包中立的 `SceneCfg` 和可选的向量化环境数量构造。模型元数据只在构造期间解析；状态和控制数组通过公共契约传递，引擎对象不会逃出适配器。IsaacGym 与 IsaacSim 共享子进程 IPC 组帧，并把它们的 Python 3.8 与 Kit worker 保留在核心 wheel 之外。

`unisim.ADAPTER_SPECS` 是所有已声明 UniLab 后端身份的唯一迁移清单。`available` 表示存在公共适配器和诊断；它不表示每台主机都安装了专有 SDK 或 GPU 运行时。运行时支持由适配器的 optional-extra 与 worker 冒烟测试确立。

所有资产和模型元数据解析都是冷路径职责。热路径 `step` 与 `reset` 代码接收校验过的数组和缓存标识符；适配器不得动态探测引擎私有属性。

区间域随机化是声明式的：UniSim 的 manager 从 `unisim.dr.interval` 定义的 `IntervalTermOp` 描述符构建 `IntervalRandomizationPlan.ops`（项名称、NumPy 载荷和可选 body ID；只使用标准库与 NumPy，因此计划在基于 spawn 的 collector 进程间保持可 pickle）。每个后端拥有自己的能力声明（`supported_interval_terms`）和冷路径构建的 `_interval_term_handlers()` 表；通用 `SimBackend.apply_interval_randomization` 分发会用内建项规格校验每个 op，路由到匹配 handler，并对后端未声明的项以 `NotImplementedError` 快速失败。自定义项是由注册后端拥有的自由字符串，只根据该后端的能力集校验。

重置时的模型字段写入使用 `ResetRandomizationPayload` 上的精选字段；调用者绝不独立提交几何包围盒等由编译器派生的字段。适配器通过 `SimBackend.get_reset_term_default(term)` 暴露权威默认值：单模型后端返回规范表，固定变体建立不同基线时返回逐环境表。

固定模型身份与重置随机化分离。任务在 `SceneCfg` 上携带 `FixedVariantPlan`，让引擎适配器在构造期间、首次 forward 之前以及 CUDA graph 捕获之前实现它。该计划包含最终只读赋值行、完整物化的 `ModelSourceDescriptor` 条目，以及公共布局保证（`same_layout` 或 `uniform_public_layout`）。域随机化能力声明适配器能实现的布局，以及播放是否暴露逐环境模型。计划与能力对象只使用标准库和 NumPy 类型，因此保持可 pickle；活跃 `MjSpec`、mjbatch 与 Warp 对象绝不跨越该边界。槽位合并、mesh/material 池化、逐世界数组、派生字段重算和播放表示都是适配器拥有的实现细节。

MuJoCo 适配器实现该契约，同时没有重新引入每个环境一个完整模型。在冷路径上，它独立编译每个物化 MJCF 作为数值/默认值 oracle，校验 same 或 uniform public layout，并把规范 mesh 池化委托给 mjbatch 的 `VariantPack`。适配器保留规范执行器模型、由编译器派生的变体行、紧凑默认值表和不可变赋值，而不是每个变体一个已编译 `MjModel`。运行时重置写入使用与规范模式相同的扩展模型字段视图。Same-layout mesh-geom 槽位可以逐世界禁用，而公共状态或控制拓扑变化会快速失败。播放按需从所选源编译分离的视觉 oracle，离线播放为每个被渲染环境保存一个自包含模型。

MJWarp 适配器在构造期间实现该计划。它独立编译每个 MJCF 源以获得 oracle 值，校验声明布局，把 mesh 与 material 池化到一个规范模型，并在 `put_model` 之后、首次 forward 与 CUDA graph 捕获之前安装逐世界 `geom_dataid`、`geom_matid` 和依赖 mesh 的模型字段。因此，重置镜像和 `get_reset_term_default()` 从每个世界分配到的变体开始。
