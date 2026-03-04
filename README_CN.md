# RoboSTD Unified 项目

## 简介
RoboSTD Unified 是一个专为处理机器人数据集（特别是 CoRobot 格式）而设计的综合工具包。它集成了视频处理（裁剪、镜像、拼接等）与同步关节数据变换功能。

## 目录结构
- `src/`: 核心逻辑模块 (`video_*`, `joint_*`)。
- `scripts/`: 可执行脚本 (`main.py`, `corobot_mirror_tool.py`)。
- `configs/`: 配置文件 (`default.yaml`)。
- `tools/`: 验证与测试工具。
- `data/`: 数据集存储 (input/output)。

## 安装说明
1. 克隆仓库。
2. 安装系统依赖（必须安装 FFmpeg 以支持 AV1 视频）：
   ```bash
   sudo apt-get install ffmpeg
   ```
3. 安装 Python 依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 快速开始
### 1. CoRobot 镜像工具 (推荐)
这是一个专门用于镜像 CoRobot 数据集视频的独立工具，支持自动交换左/右视角文件夹。

**使用方法：**
```bash
python scripts/corobot_mirror_tool.py --root /path/to/videos
```
**功能特性：**
- **镜像处理**：自动水平翻转 `high` 或 `head` 文件夹中的视频。
- **镜像并交换**：对 `left` 和 `right` 文件夹中的视频进行镜像处理，并自动**交换内容**（即左视角的镜像视频存入右视角文件夹，反之亦然）。
- **自动备份**：处理前会自动备份原始目录，确保数据安全。
- **回滚机制**：如果发生严重错误，工具会自动从备份中恢复。

### 2. 交互式模式 (主流程)
运行主脚本不带参数即可启动交互式向导，按提示选择操作：
```bash
python scripts/main.py
```

### 3. 命令行模式 (高级)
通过参数指定输入输出路径及操作：
```bash
python scripts/main.py \
  --input data/raw \
  --output data/output \
  --ops mirror,reverse \
  --rule aloha
```

## 配置文件
`configs/default.yaml` 文件定义了：
- **关节镜像规则**：定义了左/右臂关节数据的映射关系（例如 `aloha`, `realman` 机器人）。
- **视频设置**：默认的分辨率和帧率设置。

## 常见问题与解决方案
- **AV1 视频无法打开？**
  - 本项目已内置 AV1 转码支持。程序会自动检测 AV1 编码，处理时转为 H.264，并在最终输出时还原为原始编码或兼容性更好的 H.264 格式。请确保系统已安装 `ffmpeg`。
- **校验和不匹配 (Checksum Mismatch)？**
  - 如果视频帧数与 Parquet 数据行数不一致，系统会发出警告。这通常是原始数据的问题，但不会中断处理流程。

## 贡献指南
1. Fork 本仓库。
2. 创建特性分支。
3. 提交 Pull Request。
4. 确保通过 `tools/verify_project.sh` 测试。

## 许可证
MIT License
