# SuperDex 适配器使用指南

[English](superdex.md)

本指南说明如何加载本地模型、通过 UniSim 运行 SuperDex、查看模型，以及验证适配器与原生 SDK
的结果是否一致。**物理计算由 SuperDex 完成；UniSim 提供统一接口，并转换模型、控制量和状态。**
UniLab 负责具体任务、动作归一化、奖励、观测、训练和策略 rollout。

中文版与英文版描述同一份实现。修改示例、支持范围或限制报告时，应同步更新两版。

## 目录

1. [安装并运行第一个模型](#quick-start)
2. [选择输入格式和执行模式](#inputs-and-modes)
3. [使用 Python API](#python-api)
4. [理解状态和控制量](#state-and-control)
5. [使用相机和控制器](#cameras-and-controllers)
6. [查看模型和比较结果](#tools)
7. [解析资产路径并验证文件](#assets)
8. [理解代码结构与公共接口](#architecture)
9. [查看当前限制与不支持的资产](#limitations)
10. [排查常见问题](#troubleshooting)
11. [验证修改与后续工作](#validation)

<a id="quick-start"></a>
## 1. 安装并运行第一个模型

以下命令均在 UniSim 仓库根目录运行。示例依赖本地资产包；克隆仓库不会下载被 Git 忽略的资产文件。

```bash
uv sync --python 3.12 --extra superdex --extra mujoco
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"

uv run --no-sync scripts/superdex_viewer.py \
    benchmarks/cart_pole/cart_pole.mochi_scene \
    --controlled-joints Cart --effort-limit 3.0
```

该命令通过 UniSim 适配器打开一个原生查看器窗口。`Cart` 选择小车的移动关节；`3.0` 是允许的
最大作用力，单位为牛顿，**不代表持续施加 3 N**。查看器会给出一个较小的演示控制量。
关闭窗口或按 Ctrl+C，会释放查看器和后端资源。

如只需运行有限帧数的渲染检查，不打开交互窗口：

```bash
uv run --no-sync scripts/superdex_viewer.py \
    benchmarks/cart_pole/cart_pole.mochi_scene \
    --controlled-joints Cart --effort-limit 3.0 --frames 9
```

默认不保存图片、视频或 JSON 报告。查看器可以在没有设置环境变量时找到仓库默认资产包；
下文的 Python 示例和资产验证命令显式设置该变量，以便明确依赖根目录。

### 运行环境要求

| 项目 | 当前接入范围 |
| --- | --- |
| Python | SuperDex wheel 使用 CPython 3.12 或 3.13 |
| SDK 包 | `superdex-physics-uni==1.0.0`、`superdex-robotics-uni==1.0.0` |
| 已验证环境 | Linux x86_64、CPU、FP32 |
| 其他平台 / FP64 | 本次接入尚未完成验证 |
| GPU 物理 | 未启用；测试使用的 wheel 不提供 CUDA 求解器 |
| UniSim 基础导入 | 不会因此导入物理引擎 SDK |

`superdex-uni` 是包含原生批量执行器的接入构建。它与上游包共用 `superdex/` 命名空间，不能同时
安装两套包。x86 构建要求 AVX2 等指令集。MuJoCo 3.11 用于受支持 MJCF 的加载解析，以及可选的
回放渲染；**它不负责 SuperDex 仿真的物理步进**。原生 bot/scene 加载不经过 MJCF 解析器。
适配器不依赖 SuperDex Lab、Gymnasium 或学习算法。

`uv run --no-sync` 使用现有环境，不先同步已安装的依赖，因此可以保留本地 SDK 或项目的 editable
安装。相邻 UniLab 仓库可使用 `uv pip install -e './[superdex,mujoco]' -e ../UniLab`；UniLab
的本地来源检查通过 `UNILAB_LOCAL_UNISIM` 指定确切的 UniSim 仓库。使用本地接入不需要发布 PyPI。
历史项目背景见 [UniLab roadmap #1533](https://github.com/Motphys/UniLab/issues/1533)。

<a id="inputs-and-modes"></a>
## 2. 选择输入格式和执行模式

**是否支持一个模型，取决于文件内容和执行模式，而不只是扩展名。**

| 输入 | 用途与当前支持范围 |
| --- | --- |
| `.superdex_bot` | 机器人定义；根关节可以是 HARD（固定）或 FREE（浮动），关节及组件必须处于支持范围内 |
| `.superdex_bot_archive` | 机器人及依赖打包文件；仅串行、单环境 |
| `.mochi_scene`、`.mochi_prefab` | 可以直接加载的场景输入；支持刚体、关节系统、嵌套 prefab、受支持的约束、接触过滤和场景自带姿态控制器 |
| MJCF `.xml` | 仅支持下述已审核子集，不保证任意 MuJoCo 模型均可加载 |
| `.urdf` | 不支持直接加载 |
| `.mochi.h5`、网格、纹理、CAD 文件 | 模型依赖的数据，不是独立仿真入口 |
| `.superdex_controller`、传感器配置文件 | 随兼容模型使用的配置，不是独立模型 |

例如 `rods/helix_with_visual.mochi.h5` 是可变形杆示例的形状数据，并不是查看器可直接仿真的刚体模型。

### 串行与批量执行

| 模式 | 行为 | 适用场景 |
| --- | --- | --- |
| `serial` | 在调用线程逐场景步进 | 原生查看器、调试器和高级模型 |
| `batch`（默认） | 通过原生 `SceneBatchExecutor` 和常驻 C++ 工作线程执行 | 支持批量模式的多环境仿真 |

普通受支持机器人可以使用两种模式。按当前审核范围，archive、球关节、闭环、耦合驱动、多关节系统、
约束和高级 link 跟踪控制器需要 **`serial` 且 `num_envs=1`**。原生交互回放也要求这一组合。
串行模式下支持某个特性，并不代表批量模式也支持。

原生调试器的 `DebugDraw` 有线程归属要求。适配器会拒绝调试器与 batch 后端的组合，避免原生线程
错误。连接调试器之前应设置 `superdex_execution_mode="serial"`。UniLab 对应配置为
`env.superdex_execution_mode=serial`，交互评估还需使用一个播放环境。

批量模式下，`superdex_num_workers=0` 表示自动选择当前进程可用的物理核心。串行模式应保持这个
外层场景工作线程参数为零；非零值会被拒绝。

`superdex_num_worker_threads` 用于在串行模式下配置进程级 SDK 内部求解器线程池：`-1` 表示由
SuperDex 自动选择，`0`（默认值）表示单线程，正整数表示请求对应数量的线程。批量模式会拒绝非零值，
避免两个线程池竞争。同一进程内所有同时存活的 SuperDex 后端必须使用相同值，因为它们共享 SDK
运行时。

### MJCF 的审核范围

支持一棵关节树、可选浮动根、hinge/slide 关节、标量无状态 motor 或线性 position 执行器、静态平面、
命名 keyframe 和受支持的片段。加载时转换质量、惯量/质心、坐标系/轴、armature、关节摩擦和执行器
限制。动态基本几何体会在加载时三角化，并生成 SDF 碰撞表示。

接触结果不保证与 MuJoCo 数值等价。扭转/滚动摩擦需要显式使用实验选项
`superdex_allow_contact_approximation=True`，并会警告仅保留滑动库仑摩擦。SDK 不支持逐碰撞对
摩擦覆盖：导入器会将滑动摩擦关系分解为 actor 系数，无法表示的组合会被拒绝，不会静默修改。

<a id="python-api"></a>
## 3. 使用 Python API

先按第 1 节设置 `SUPERDEX_ASSETS_PATH`。以下 Cart Pole、FR3 和控制器示例是**独立程序**。
不要在未关闭旧实例时，用同一个变量创建第二个后端。

### 加载场景、步进、读取状态和重置

```python
import os
from pathlib import Path

import numpy as np

from unisim import create_backend
from unisim.scene import SceneCfg

assets = Path(os.environ["SUPERDEX_ASSETS_PATH"]).expanduser().resolve()
backend = create_backend(
    "superdex",
    SceneCfg(str(assets / "benchmarks/cart_pole/cart_pole.mochi_scene")),
    num_envs=1,
    sim_dt=0.002,
    superdex_execution_mode="serial",
    superdex_controlled_joints=["Cart"],
    superdex_effort_limits=[3.0],
)
try:
    print(backend.get_actuator_names())  # ("Cart",)
    print(backend.get_model_info())
    command = np.array([[1.0]])  # One environment, one actuator; force in N.
    backend.step(command, nsteps=8)
    state = backend.get_state()
    print(state["qpos"], state["qvel"])
    backend.reset()
finally:
    backend.close()
```

`command` 的形状为 `(环境数, 执行器数)`。Cart Pole 有两个广义坐标，但这里只选择了一个执行器。
`sim_dt=0.002` 时，八个物理步对应 **0.016 秒仿真时间**，与实际计算耗时无关。
输入会按选定的作用力上限进行裁剪。

原生 scene/prefab 的主动控制需要有序的 `superdex_controlled_joints`，以及有限且为正的
`superdex_effort_limits`。传入 `[]` 表示被动场景，其控制数组形状为 `(环境数, 0)`。

### 加载两个独立的机器人环境

```python
import os
from pathlib import Path

import numpy as np

from unisim import create_backend
from unisim.scene import SceneCfg

assets = Path(os.environ["SUPERDEX_ASSETS_PATH"]).expanduser().resolve()
backend = create_backend(
    "superdex",
    SceneCfg(str(assets / "bots/arms/fr3_v2/fr3_v2.superdex_bot")),
    num_envs=2,
    sim_dt=0.002,
    superdex_effort_limits=[87, 87, 87, 87, 12, 12, 12],
)
try:
    backend.step(np.zeros((2, backend.num_actuators)), nsteps=8)
    state = backend.get_state()
    backend.set_state(np.array([0]), state["qpos"][[0]], state["qvel"][[0]])
    backend.reset(np.array([0]))  # Environment 1 is unchanged.
finally:
    backend.close()
```

力矩上限的顺序与执行器顺序一致。零力矩不会让机器人保持不动：重力和被动动力学仍然起作用。
此示例使用默认 batch 模式。`set_state()` 写入选定的状态行；`reset([0])` 将环境 0 恢复到模型
初始状态，环境 1 保持不变。

### 将机器人与刚体物体组合

使用 FR3 示例的导入和 `assets` 变量；关闭之前的后端后，独立执行以下代码：

```python
scene = SceneCfg(
    str(assets / "bots/arms/fr3_v2/fr3_v2.superdex_bot"),
    fragment_files=[str(assets / "prefabs/sphere/sphere.mochi_prefab")],
)
backend = create_backend(
    "superdex", scene, num_envs=1, sim_dt=0.002,
    superdex_execution_mode="serial",
    superdex_effort_limits=[87, 87, 87, 87, 12, 12, 12],
)
try:
    backend.step(np.zeros((1, backend.num_actuators)))
finally:
    backend.close()
```

片段保留文件中定义的变换。该操作不会把球固定到机器人上，也不会自动安排接触位置。
如需指定摆放方式，应通过 wrapper prefab 明确定义。

### 元数据、重力和资源释放

后端打开期间，可以通过 `get_model_info()` 查询关节系统、坐标名称/单位/表示方式、坐标分组和刚体
归属。使用 `get_actuator_names()`、`get_joint_state_qpos_indices()` 和
`get_joint_state_qvel_indices()` 确定动作与状态顺序，不要猜测数组索引。

`get_gravity()` 读取重力；`set_gravity([0, 0, -3])` 修改重力，且覆盖值在 reset 后仍然保留。
原生场景保留文件中的重力和求解器设置，缺省项使用 SDK 默认值。不自动添加地面，也不自动旋转坐标系。
嵌套场景显式指定的设置必须与根场景一致。时间步长由调用者的 `sim_dt` 决定；场景文件中的时间步长
字段会被拒绝，而不是被静默忽略。

| 已验证场景 | 有序控制关节 | 作用力/力矩上限 | 状态维数 |
| --- | --- | --- | --- |
| Cart Pole | `Cart` | 3 | `nq=nv=2` |
| Half Cheetah | `BackThigh`, `BackShin`, `BackFoot`, `FrontThigh`, `FrontShin`, `FrontFoot` | 120, 90, 60, 120, 60, 30 | `nq=nv=9` |

两个 benchmark 场景的重力均为 `[0, -9.8, 0]`；bot 加载使用负 Z 方向。

每个环境有独立的原生场景。UniSim 对进程级 SDK 运行时进行引用计数，因此关闭一个后端不会破坏其他
仍在使用的后端。应由 UniSim 管理初始化，不要把已创建的后端跨进程传递，并始终调用 `close()`
或公共生命周期接口 `cleanup_scene_assets()`。UniLab 的 `env.close()` 会调用该接口。

<a id="state-and-control"></a>
## 4. 理解状态和控制量

### 状态布局与坐标系

`get_state()` 返回独立的 NumPy 数组，包括 `qpos` 和 `qvel`，每个环境占一行。两者列数可能不同。

| 状态部分 | 公共 `qpos` | 公共 `qvel` |
| --- | --- | --- |
| 浮动根 | 世界系 xyz + **wxyz** 四元数，共 7 项 | 世界系刚体原点线速度 + **刚体系**角速度，共 6 项 |
| 转动关节 | 角度，rad | 角速度，rad/s |
| 移动关节 | 位移，m | 速度，m/s |
| 球关节 | 原生关节系旋转向量 XYZ，共 3 项 | 三个原生旋转速度坐标，不能直接理解为旋转向量的普通导数 |
| 动态刚体物体 | 世界系 xyz + wxyz，共 7 项 | 世界系刚体原点线速度 + 刚体系角速度，共 6 项 |

因此，一个浮动根加七个单自由度关节的机器人，`nq=14`、`nv=13`。静态物体有 body 信息，但没有
动态状态坐标。动态物体会追加自己的状态区间；复杂场景应使用元数据确定索引。
`get_root_state_layout(body_name)` 返回相应根状态的索引。名称以 `_w` 结尾的 body 查询返回世界系
量，不要把其角速度与 `qvel` 中的根刚体系角速度混淆。

SuperDex 内部使用旋转向量表示浮动根，并采用不同的原生速度约定。适配器负责转换这些表示、组合模型
中定义的 parent-joint/joint-link 变换，以及处理绕偏置关节旋转造成的速度差。
包含非平凡变换/Jacobian 的回归测试覆盖这些转换。不要直接把原生数组当作公共状态数组使用。

选择球关节名称时会展开三个坐标；`/x`、`/y`、`/z` 可选择单独坐标。约束和传动不会增加动作坐标。
实际归属应从元数据查询。

### 物理作用力、位置目标和子步

| 控制路径 | 输入含义 |
| --- | --- |
| 原生 bot 或已选择的原生场景关节：`step(ctrl)` | 物理作用力；移动关节为 N，转动关节为 N·m |
| 审核范围内的 MJCF motor | 遵循模型中 gear 和限制的电机输入 |
| 审核范围内的 MJCF position 执行器 | 位置目标，按模型增益和限制转换为作用力 |
| `step_controller(targets)` | 控制器对应的目标，每个物理子步重新计算输出 |

不能把所有后端和执行器的 `step()` 输入都理解为同一种物理量。任务动作归一化由 UniLab 负责。
`set_pre_step_control()` 回调也在每个物理子步执行。待施加的刚体外力与执行器作用力会合并提交，避免
一次原生外力写入覆盖另一个贡献。`apply_body_force()` 对动态刚体接收世界系质心力和力矩；静态对象
会被拒绝。

### 重置、接触与传感器

reset 恢复初始原生动态快照，清除控制量/待施加外力并刷新缓存。状态赋值恢复的是公共运动学状态，
不保证恢复全部隐藏的求解器历史；原生快照字节不是可跨环境搬运的 checkpoint。

命名关节/坐标系信号、陀螺仪和速度计数据根据原生状态重建。受支持的 MJCF 平面/几何体
`contact data="found" num="1"` 传感器使用原生接触点及真实 actor 对。接触数据描述的是
**最近一次完成的物理求解**：瞬移或 reset 后，应先执行正时间步，再读取新的接触结果。
零时长求解不会重建接触流形。

MJCF 审核路径能识别加速度计，但无法提供瞬时点加速度；请求或绑定时会抛出 `NotImplementedError`。
该范围内未使用的加速度计不会阻止模型加载。这与原生 bot 的组件检查是两件事：后者目前只接受
`SENSOR_CAMERA` 组件。

<a id="cameras-and-controllers"></a>
## 5. 使用相机和控制器

### 相机元数据与安装位姿

对已打开的、**带有相机定义的原生 bot/archive**，并已导入 NumPy 时：

```python
for name in backend.get_camera_names():
    settings = backend.get_camera_parameters(name)
    poses = backend.get_camera_poses(name)  # (num_envs, 7): xyz + wxyz
    selected = backend.get_camera_poses(name, np.array([0]))
    print(name, settings, poses, selected)
```

这些 API 提供元数据与位姿，**不返回图像像素**。相机列表允许为空，所以示例使用遍历，而不是直接
访问第一个元素。`get_sensor_data(camera_name)` 会提示调用相机专用接口。

配置以独立字典返回，字段为 SDK 的 snake_case：`name`、`image_width`、`image_height`、
`fov_vertical_deg`、`near_clip`、`far_clip`、`forward_axis`、`up_axis_local`、`offset_local`、
`look_at`、`look_distance`。尺寸单位是像素，视场角单位是度，距离单位是米。返回的位姿是传感器安装
坐标系，不是渲染器的光学视图矩阵；轴和偏移设置需要单独解释。步进、状态赋值和选择性 reset 后，
位姿会更新。创建实例时会校验相机清单、安装关系和设置。支持 bot 相机不代表支持原生 scene 相机。

### 内置控制器

| 控制器 | 公共目标类型 | 含义 |
| --- | --- | --- |
| `BASIC_JSC_PD` | `JointTarget` | 目标关节位置 |
| `BASIC_OSC_PD` | `CartesianTarget` | 在控制器约定坐标系下的目标末端位姿 |
| `MOCHI_ARTICULATED_POSE` | `ArticulationPoseTarget` | 目标根位姿和关节系统姿态 |

用 `get_controller_descriptions()` 查询当前模型的可用性。这些 API 要求兼容的原生 bot/archive；
前面的 Cart Pole scene 后端不能用于配置这些 bot 控制器。场景自带的姿态控制器属于另一条场景加载路径。

下面是完整的固定基座 FR3 示例，用关节控制器保持初始目标：

```python
import json
import os
from pathlib import Path

from unisim import JointTarget, create_backend
from unisim.scene import SceneCfg

assets = Path(os.environ["SUPERDEX_ASSETS_PATH"]).expanduser().resolve()
backend = create_backend(
    "superdex",
    SceneCfg(str(assets / "bots/arms/fr3_v2/fr3_v2.superdex_bot")),
    num_envs=1, sim_dt=0.002,
    superdex_execution_mode="serial",
    superdex_effort_limits=[87, 87, 87, 87, 12, 12, 12],
)
try:
    print(backend.get_controller_descriptions())
    n = backend.num_actuators
    backend.configure_controller(
        "BASIC_JSC_PD",
        param_args=json.dumps({
            "Kp": [10.0] * n,
            "Kd": [1.0] * n,
            "saturation": [2.0] * n,
            "deadband": [0.0] * n,
        }),
    )
    targets = [JointTarget(q.copy()) for q in backend.get_dof_pos()]
    backend.step_controller(targets, nsteps=8)
    backend.clear_controller()
finally:
    backend.close()
```

PD 控制中，`Kp` 乘以位置误差，`Kd` 提供阻尼，saturation 限制输出。为每个环境配置一个控制器后，
通过 `step_controller()` 执行；切换控制路径之前先清除控制器。预步进回调与已配置的控制器不能同时
启用。示例中的参数数组长度针对固定基座 FR3；原生浮动根配置可能需要根坐标占位项。
浮动基座 OSC 因 SDK 索引问题而明确不可用。

原生 SDK target 对象已弃用，应使用公共目标类型。配置文件包含 `type_name`、`param_args`、
`init_args`，后两者为 SDK JSON 字符串或文件路径。CLI 配置文件中的路径相对于该配置文件解析。
绝对目标 JSON 格式和 FR3 示例位于 [superdex-configs/](superdex-configs/)，CLI 目标必须匹配控制器类型。

<a id="tools"></a>
## 6. 查看模型和比较结果

### 查看器：检查几何和行为

```bash
uv run --no-sync scripts/superdex_viewer.py bots/arms/fr3_v2/fr3_v2.superdex_bot
uv run --no-sync scripts/superdex_viewer.py prefabs/sphere/sphere.mochi_prefab
uv run --no-sync scripts/superdex_viewer.py bots/arms/fr3_v2/fr3_v2.superdex_bot --compare
uv run --no-sync scripts/superdex_viewer.py --list-profiles
```

| 选项 / 模式 | 行为 |
| --- | --- |
| 普通查看器 | 通过适配器公共 API 加载和步进，由 `run_playback()` 打开原生查看器 |
| `--compare` | 增加独立的原生 SDK 仿真/窗口；两边同步推进，关闭任一窗口会一起结束 |
| 已注册机器人演示配置 | 演示初始姿态保持、有界运动和 reset |
| 未注册但受支持的模型 | 被动步进，不臆造控制配置 |
| `--controlled-joints Cart --effort-limit 3.0` | 选择 scene/prefab 的控制关节及标量作用力上限 |
| `--controller-config FILE --absolute-target FILE` | 为兼容 bot/archive 配置控制器和匹配的目标 |
| `--fragments FILE ...` | 为 bot/archive 添加刚体 prefab 片段 |
| `--no-gravity` | 在本次运行中将重力覆盖为零 |
| `--frames N` | 显式无窗口渲染冒烟检查；至少 6 帧 |
| `--out results/viewer.json` | 显式保存 JSON 验证信息 |

被动不等于静止：重力和模型自带动力学仍会起作用。冒烟检查覆盖启动、步进、reset 和清理，不能代替
人工确认模型外观。查看器不保存图片或视频。SDK 对照模式接受原生输入；审核范围内的 MJCF 保留
单适配器窗口和现有 pytest 验证路径。

后端另有**离线录制**路径，要求存在兼容的可视化 MJCF 模型；原生资产通过
`SceneCfg.visual_model_file` 指定。它使用共享 MuJoCo 渲染器，可在串行或批量模式下运行，既不是
相机像素读取，也不是查看器脚本的 `--frames` 模式。原生交互路径不支持 debug-overlay 和 `on_frame`
回调。

### 比较工具：验证适配器与 SDK 是否一致

```bash
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
uv run --no-sync scripts/superdex_compare.py bots/arms/fr3_v2/fr3_v2.superdex_bot
uv run --no-sync scripts/superdex_compare.py --all
uv run --no-sync scripts/superdex_compare.py --all --match hand
uv run --no-sync scripts/superdex_compare.py --fixtures
uv run --no-sync scripts/superdex_compare.py --all --out results/superdex
uv run --no-sync scripts/superdex_compare.py bots/arms/fr3_v2/fr3_v2.superdex_bot \
    --controller-config docs/superdex-configs/fr3-jsc.json \
    --absolute-target docs/superdex-configs/fr3-jsc-absolute.json
```

两边使用的都是 **SuperDex 物理引擎**。一边通过 UniSim 加载和步进，另一边通过 SDK 独立创建。
比较时匹配初始状态、控制量、时间步长、重力和求解器设置。它检查的是适配器转换是否忠实，不是
SuperDex 与其他物理引擎是否等价。

`--all` 从仓库本地资产根目录发现模型，`--roots DIR ...` 可指定根目录。`--fixtures` 临时生成
回归模型。每个模型在独立子进程中执行，原生崩溃或超时会被记录，后续模型仍继续执行。
检查覆盖控制路由/裁剪、原生与公共状态、刚体位姿/速度、接触、相机/控制器、重置、清理/重建和
已验证的批量模式。给一个 bot 加上 `--composition FRAGMENT ...` 可验证刚体片段组合。

| 结果 | 含义 |
| --- | --- |
| Passed | 该配置实际执行的检查满足容差 |
| Blocked / unsupported | 指定特性阻止该配置；绝不计为通过 |
| Failed | 断言、数值检查或其他执行错误失败 |
| Crashed / timeout | 子进程异常退出或超过时间预算 |
| Unverified | 缺少所需运行时，没有完成成功验证 |

预期的静态限制不会让其余检查成功的整轮扫描失败。数值错误、非预期的运行时拒绝、缺少运行时、崩溃和
超时会产生非零退出码。因此应查看分类数量，而不只看退出码。
`--out` 在指定目录写入 `comparison.json`，包含检查、限制和可复现性指纹。应使用被忽略的 `results/`，
不要提交生成结果；不传 `--out` 就不保存持久报告。

bot/控制器默认运行 1040 个物理步，scene/组合默认 1000 步。bot/控制器的 `--steps` 按完整八步区间
执行，可缩短验证长度。嵌套机器人接触 fixture 保留 1 ms 步长和 1000 步。
缩短运行不能代表完整基线，零重力验证通过也不能代表重力条件下通过。

<a id="assets"></a>
## 7. 解析资产路径并验证文件

主要本地资产包位于 `assets/superdex`，物理示例位于 `assets/superdex-physics`。
资产文件及许可证保留在本地，不进入发布包。维护使用 `scripts/copy_superdex_assets.py`；
验证过程不得读取 SDK 源码仓库中的资产，也不得修改资产内容。

查看器按 `--assets ROOT`、`SUPERDEX_ASSETS_PATH`、仓库默认目录的优先级选择根目录。
已有的显式模型路径直接使用，否则在选定根目录下查找。该根目录会传给适配器和对照子进程。
外部模型在未指定根目录或环境变量时，保留以自身目录为依赖根目录的回退行为。

找到模型文件与解析其依赖是两个步骤。附近的 `.superdex_root` 定义模型中的根引用；scene 中的
`./...` 相对于所在文档，普通相对路径则相对于解析得到的资产根目录。原生 bot 资产还可声明带标签的
依赖根目录。应保留文件中的引用关系，不要把单个模型从依赖中单独移走。
archive 必须包含 `.mochi_bot_archive_metadata`，指定包内 bot，并包含完整依赖闭包。

被跟踪的 [JSON 清单](superdex-assets-inventory.json) 记录来源、哈希和依赖，也包括 archive 内的依赖。
**清单中的 candidate 不代表运行验证通过。** 适配器不会在每次加载时重新计算整个资产包的哈希。

按现有清单验证本地资产：

```bash
uv run --no-sync python - <<'PYTHON'
from pathlib import Path
from unisim.backend.superdex.assets import verify_asset_bundle

print(verify_asset_bundle(
    Path("assets/superdex"), Path("docs/superdex-assets-inventory.json")
))
PYTHON
```

有意修改本地资产包之后，才使用以下命令刷新清单：

```bash
uv run --no-sync scripts/copy_superdex_assets.py --inventory-only
```

该命令只读取本地资产并写入清单 JSON，不是修复意外丢失资产的方法。
资产覆盖范围变化时，应同步维护本指南中的限制报告；不再维护自动生成的 Markdown 兼容性表。

<a id="architecture"></a>
## 8. 理解代码结构与公共接口

```text
Application / UniLab / normal viewer
    -> create_backend("superdex", ...)
    -> SuperDexBackend implementing SimBackend
    -> SuperDex Physics / Robotics SDK
```

[工厂](../src/unisim/factory.py) 负责选择适配器；
[SimBackend](../src/unisim/backend/base.py) 定义统一公共接口。
当前 SuperDex 的所有公共方法名都属于这个接口。但公共**声明**不等于每个引擎均支持该操作：
默认实现可能抛出 `NotImplementedError`，具体实现也可能限制模型或执行模式。

| `src/unisim/backend/superdex/` 模块 | 职责 |
| --- | --- |
| `__init__.py` | 导出 `SuperDexBackend` 和 `SuperDexDependencyError` |
| `backend.py` | 主类；生命周期、步进、状态、外力和回放 |
| `components.py` | 相机校验与通过 `BuiltinAPI` 继承的相机/控制器接口 |
| `model.py` | 状态布局、坐标/根变换及公共元数据 |
| `runtime.py` | SDK 查找、运行时所有权和 CPU 拓扑 |
| `materialization.py` | bot/archive、审核 MJCF 和几何体的创建 |
| `scenes.py` | scene/prefab 校验、加载与组合 |
| `assets.py` | 清单、依赖闭包和完整性验证 |

`SuperDexBackend(BuiltinAPI, SimBackend)` 从 `components.py` 继承相机/控制器方法。
因此，虽然实现位于另一个文件，`backend.get_camera_names()` 仍是公共适配器调用。

以下三种调用说明了公共声明与具体实现的区别：

| 调用 | 声明位置 | SuperDex 实际使用的实现 |
| --- | --- | --- |
| `backend.step(...)` | `SimBackend` | `backend.py` 中的覆盖实现 |
| `backend.reset(...)` | `SimBackend` | 继承公共实现，再调用 SuperDex 的 `set_state()` |
| `backend.get_camera_names()` | `SimBackend` | 从 `components.py` 的 `BuiltinAPI` 继承覆盖实现 |

因此，相机/控制器操作可以属于共享接口，即使目前只有 SuperDex 实现它们。
`_refresh()` 等内部辅助函数不是公共接口方法，不要求在 `SimBackend` 中声明。

这些模块没有被强制隐藏。比较工具会直接使用资产、运行时、场景辅助函数及内部 actor 句柄，以检查
适配器行为；查看器的对照子进程也直接使用运行时辅助函数创建 SDK 场景；资产维护调用者直接使用
`assets.py`。普通仿真应用应使用公共后端接口，私有辅助函数和句柄不属于兼容性承诺。

<a id="limitations"></a>
## 9. 当前限制与不支持的资产

本报告列出目前无法加载的**直接模型输入**，不把辅助几何、纹理、CAD 或配置文件算作缺少的模型入口。
以下十项均由**静态校验**发现，包括递归检查 recipe 依赖；并不表示这些模型曾成功运行 SDK。
这些是适配器限制；关于 SDK 本身的结论需要独立的直接验证。

| 模型（相对于 `assets/`） | 格式 | 不支持的特性 | 限制来源 | 检查层级 | 所需能力 |
| --- | --- | --- | --- | --- | --- |
| `superdex/test/urdf/fr3v2_1_urdf/robots/fr3v2_1_franka_hand.urdf` | URDF | 直接 URDF 模型加载 | 适配器限制 | 静态校验 | URDF 导入，或转换为审核范围内的 MJCF；另需复核 SDK 历史几何丢失问题 |
| `superdex/bots/fun/example_bot_2dof/example_bot_2dof.superdex_bot` | bot | 自定义组件：`MY_VELOCITY_SERVO_ACTUATOR`、`MY_CONTACT_FORCE_SENSOR` | 适配器限制 | 静态校验 | 自定义组件转换；当前只支持 `SENSOR_CAMERA` |
| `superdex/bots/sensors/dg5f_seed/dg5f_seed.superdex_bot` | bot | seed 传感器：`SENSOR_SEED_V1_MLP` | 适配器限制 | 静态校验 | learned 传感器转换 |
| `superdex/bots/hands/dg5f_long_seed/left/dg5f_long_seed_left.superdex_bot` | bot | 通过 `AttachBot` recipe 继承的五个 seed 传感器 | 适配器限制 | 静态校验（recipe 依赖闭包） | learned 传感器转换 |
| `superdex/bots/hands/dg5f_long_seed/right/dg5f_long_seed_right.superdex_bot` | bot | 通过 `AttachBot` recipe 继承的五个 seed 传感器 | 适配器限制 | 静态校验（recipe 依赖闭包） | learned 传感器转换 |
| `superdex/bots/hands/dg5f_short_seed/left/dg5f_short_seed_left.superdex_bot` | bot | 通过 `AttachBot` recipe 继承的五个 seed 传感器 | 适配器限制 | 静态校验（recipe 依赖闭包） | learned 传感器转换 |
| `superdex/bots/hands/dg5f_short_seed/right/dg5f_short_seed_right.superdex_bot` | bot | 通过 `AttachBot` recipe 继承的五个 seed 传感器 | 适配器限制 | 静态校验（recipe 依赖闭包） | learned 传感器转换 |
| `superdex/bots/arm_hand_combos/fr3_dg5f_short_seed/right/fr3_dg5f_short_seed_right.superdex_bot` | bot | 通过附加的 seed 手部间接继承的传感器 | 适配器限制 | 静态校验（recipe 依赖闭包） | learned 传感器转换 |
| `superdex/prefabs/duck_lamp/duck_lamp_recumbent.mochi_prefab` | prefab | 可变形 `soft` actor，neoHookean 材料 | 适配器限制 | 静态校验 | 审核场景范围内的软体支持 |
| `superdex-physics/samples/articulations_soft_skinned_double_pendulum.mochi_scene` | scene | 可变形 `softSkinned` actor | 适配器限制 | 静态校验 | soft-skinned actor 支持 |

recipe 可以附加带有不支持组件的 bot，即使最外层文件没有直接声明该组件，限制也会传递到手部或
机械臂-手组合。目前原生 bot 接受 `SENSOR_CAMERA`，自定义执行器和 learned seed 传感器仍不支持。

### 不会阻止模型所有用法的限制

- 浮动基座 OSC 因 SDK 的 bot-space/actor-space 力矩索引不一致而不可用；其他受支持控制路径仍可使用。
- 近乎无质量的球关节手部在 debug SDK 中对重力和探测力矩敏感；比较配置在两边均使用零重力及按
  armature 缩放的探测力矩。
- 深度自碰撞历史可能触发原生接触查询断言；穿透接触检查使用新建的参考场景。
- 被动场景没有选定的动作空间，但仍可加载和仿真。
- 当前范围不提供模型域随机化、ROM/可变形/触觉状态、GPU 批量物理，也不提供基础接口中的独立图像
  捕获 API。回放支持单独见第 6 节。

这十项是本地资产包的维护基线，不保证未来所有资产都受支持。发现新限制时需更新两种语言版本。
SDK 历史上存在的 URDF 基本几何丢失问题，仍需在设计忠实导入器前独立复核。

<a id="troubleshooting"></a>
## 10. 排查常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 找不到模型 | 检查本地资产、工作目录、`--assets` 和环境变量 |
| 网格路径重复出现 `benchmarks/cart_pole` | 模型查找与依赖加载使用了不同根目录；查看器现已传递统一根目录。直接 API 调用应导出正确的 `SUPERDEX_ASSETS_PATH` |
| 要求显式有限作用力上限 | 按执行器顺序提供正的标量或逐坐标限制 |
| 要求显式控制关节 | 给场景传有序关节列表，或用 `[]` 被动加载 |
| batch 与调试器/高级模型不兼容 | 使用串行；高级模型和交互查看还需要单环境 |
| 控制器配置被拒绝 | 检查 `get_controller_descriptions()`、模型类型、目标类型和原生参数维数 |
| 没有相机图像 | 相机接口提供元数据/位姿，不提供像素；可视化使用受支持的回放路径 |
| reset 后接触标志为零 | 先推进正时间步；接触反映最近一次完成的求解 |
| 其他加载器能打开，但这里被拒绝 | 检查不支持的字段/组件；UniSim 会拒绝无法忠实转换的内容 |

<a id="validation"></a>
## 11. 验证修改与后续工作

```bash
export SUPERDEX_ASSETS_PATH="$PWD/assets/superdex"
uv run --no-sync scripts/superdex_compare.py --all --fixtures
UV_NO_SYNC=1 make check
make package
```


依赖元数据变化时执行 `uv lock --check`。SDK 测试需要可选运行时，资产测试通过
`SUPERDEX_ASSETS_PATH` 找到资产。核心契约/导入测试可以在没有原生运行时的环境中执行。
打包应排除资产文件和生成结果。执行验证命令不代表发布或提交代码。

| 检查 | 能证明什么 |
| --- | --- |
| 清单校验 | 文件及依赖的完整性 |
| `make check` | lint 与回归测试通过 |
| 比较工具 | 所测试条件下与直接 SDK 的结果一致 |
| 查看器 `--frames` | 自动原生渲染冒烟检查通过 |
| 人工交互查看 | 人对几何和行为作出视觉判断 |
| `make package` | 发布包构建成功 |

两个工具整合的是验证/查看流程，不是把全部单元测试压成两个文件。`tests/` 中仍保留专门的 pytest 覆盖：

| 保留的检查类别 | 比较函数 / 回归位置 |
| --- | --- |
| FR3、机器人路由/裁剪、PD、接触、重置、批量/串行 | `check_bot`；bot/FR3 qualification 测试 |
| FREE 根和带平移/旋转的模型参考系 | `check_bot`；`test_superdex_native_floating.py` |
| 刚体场景、多关节系统、嵌套 prefab/控制器 | `check_scene`；scene/rigid 测试 |
| 机器人与嵌套球/插板、实际机器人接触和隔离 | `check_prefab_contact`，由 `--fixtures` 运行；prefab 测试 |
| 相机、控制器子步、外力、选择性重置、link 目标 | `check_controller`；component/shared API 测试 |
| archive、球关节、tendon、耦合驱动 | 合成 fixture；rigid/shared API/inventory 测试 |
| 审核 MJCF 与延迟导入 | materialization/contract/import 测试 |

后续工作包括 learned/自定义组件、soft/soft-skinned 物体、忠实 URDF 加载、高级关节树的批量支持，以及
已验证场景的人工视觉确认。训练、任务 rollout 和 sim2sim 策略验证由 UniLab 负责。
