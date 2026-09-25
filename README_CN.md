# RoboSTD

[English](README.md) | **中文**

RoboSTD 从单臂示范构造伪双臂监督数据。当前实现按照论文分为两个阶段，同时保留原有的视频、图像、ACT-HDF5 和 LeRobot 预处理工具。

## 方法全流程

<p align="center">
  <img src="asserts/methods.svg" alt="RoboSTD 方法全流程" width="100%">
</p>

## 方法实现情况

### Stage 1：矢状面镜像

Stage 1 会同步处理 observation、state 和 action：

- 水平镜像独立相机；
- 镜像并交换左右腕部相机；
- 使用同一套平台规则变换 state 与 action；
- 保留原有 ALOHA/AgileX 表驱动工具；
- 支持根据 URDF 关节轴和安装几何推导、核验关节符号。

#### 实际案例：原始轨迹与镜像轨迹

| 原始轨迹 | 矢状面镜像轨迹 |
|:--:|:--:|
| <img src="asserts/origin.png" alt="原始机器人轨迹" width="100%"> | <img src="asserts/mirrored.png" alt="矢状面镜像机器人轨迹" width="100%"> |

若左右臂位于同一个 URDF 根坐标系，可生成经过几何推导的规则：

```bash
python scripts/derive_mirror_rules.py \
  --urdf /path/to/robot.urdf \
  --left-joints left_j1,left_j2,left_j3 \
  --right-joints right_j1,right_j2,right_j3 \
  --left-fields left_joint_1,left_joint_2,left_joint_3 \
  --right-fields right_joint_1,right_joint_2,right_joint_3 \
  --plane-normal 0,1,0 \
  --rule-name my_robot \
  --output configs/my_robot_mirror.yaml
```

工具计算 `S = I - 2nn^T`，在 URDF 根坐标系的零位姿下解析关节轴，并根据论文公式（7）推导对角符号。若夹爪不是成对的旋转关节，需要在输出 YAML 中补充夹爪规则。

原来的 Stage 1 入口保持不变：

```bash
# 含 MP4 观测的 LeRobot 数据集
python scripts/main.py \
  --input /data/single_arm \
  --output /data/single_arm_mirrored \
  --config configs/default.yaml \
  --rule aloha16 \
  --ops mirror

# ACT 单 episode HDF5
python scripts/act_hdf5_mirror.py \
  --input_dir /data/act \
  --output_dir /data/act_mirrored \
  --joint_mode aloha14
```

训练前应使用真实机器人的零位姿核验表驱动规则。仅凭数据字段名称无法恢复缺失的安装标定。

### Stage 2：LLM 引导的时空重建

Stage 2 将 `G = (rho, E_pre, E_conf)` 真正用于调度：

- `rho` 选择原始或镜像技能并分配执行手臂；
- `E_pre` 作为有向无环图决定技能先后顺序；
- `E_conf` 阻止冲突技能同时执行；
- 同一手臂上的技能串行执行，不冲突的异侧技能可以并行；
- position action 下，非活动手臂重复当前位置目标；delta action 下使用零动作；
- 两臂序列被放到同一个重建时间轴；
- 每一帧都带有子任务、手臂角色和时间阶段标注；
- 输出保存增强后的语言条件和逐帧来源信息。

代码支持按 `[左臂, 右臂]` 排列的任意偶数维 state/action，包括 ALOHA-14 和 16 维 LeRobot 数据。

单个 parquet 推荐显式提供语义边界：

```bash
python scripts/robostd_reconstruct.py \
  --input /data/original/data/chunk-000/episode_000000.parquet \
  --mirror /data/mirrored/data/chunk-000/episode_000000.parquet \
  --output /data/reconstructed/episode_000000.parquet \
  --source-arm R \
  --unit-boundaries 0,35,70,105,140 \
  --unit-names grasp,lift,transfer,place \
  --language "collect the cups" \
  --objects "four cups and one container" \
  --workspace "avoid the shared center workspace" \
  --planner openai \
  --action-mode position
```

完整 LeRobot 数据集的重建会同步生成 parquet、视频和 metadata：

```bash
export OPENAI_API_KEY="..."
python scripts/main.py \
  --input /data/original \
  --mirror-input /data/mirrored \
  --output /data/robostd_bimanual \
  --ops reconstruct \
  --source-arm R \
  --n-units 4 \
  --unit-names grasp,lift,transfer,place \
  --language "collect the cups" \
  --objects "cups and target container" \
  --workspace "shared center region; avoid simultaneous entry" \
  --planner openai \
  --action-mode position
```

默认配置使用 GPT-4.1。如果 API 或结构化输出不可用，程序会明确报错，避免悄悄变成消融版本。`--planner default` 明确运行论文中的 **RoboSTD w/o LLM**；只有传入 `--allow-planner-fallback` 时，LLM 失败才会回退。

也可以使用人工审核过的约束 JSON，避免调用 API：

```json
{
  "rho": {"0": "L", "1": "R", "2": "L", "3": "R"},
  "e_pre": [[0, 1], [1, 2], [2, 3]],
  "e_conf": [[1, 2]]
}
```

通过 `--constraints-json constraints.json` 传入。验证器会拒绝缺失分配、未知 unit、重复边、自环、非法手臂和有环的先后关系。

## 输出格式

重建后的 parquet 保留原始列，并增加：

- `robostd.stage_id`
- `robostd.active_unit_left` / `robostd.active_unit_right`
- `robostd.source_frame_left` / `robostd.source_frame_right`
- `robostd.observation_source`
- `robostd.stage_annotation`
- `robostd.language`

完整数据集模式还会写入 `meta/robostd_plans/episode_XXXXXX.json`，更新 episode 长度和 task index，并将视频对齐到重建时间轴。当原始技能和镜像技能同时执行时，单张真实相机图像无法同时包含两次源执行；程序使用源手臂视频作为主观测，并把这些帧明确标记为 `mixed`，便于训练时过滤或接入平台专用的视觉合成器。

## 安装与测试

建议使用 Python 3.10 或更高版本，并安装 FFmpeg：

```bash
pip install -r requirements.txt
pytest
```

仓库不再跟踪 `data/` 目录。请从外部提供数据集，并把生成结果保存在源码目录之外。

## 主要目录

- `src/robostd_stage2.py`：约束验证、调度、重建和语言标注
- `src/dataset_reconstruction.py`：LeRobot parquet、视频和 metadata 输出
- `src/geometry_mirror.py`：URDF 反射与关节符号推导
- `src/act_hdf5_mirror.py`：原有 ACT-HDF5 Stage 1
- `scripts/`：命令行入口
- `configs/`：Stage 1 映射和 Stage 2 默认配置
- `tools/tests/`：单元测试与集成测试

## License

MIT
