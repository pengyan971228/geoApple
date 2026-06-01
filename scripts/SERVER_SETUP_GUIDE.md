# Linux 服务器环境搭建指南（新手版）

> 适用于：全新 Linux 服务器 + NVIDIA RTX 5090 显卡
> 目标：一键搭建深度学习训练环境（Python + PyTorch + YOLO）

---

## 0. 你需要准备什么

| 项目 | 说明 |
|------|------|
| 一台 Linux 服务器 | 推荐 Ubuntu 22.04 或 24.04 |
| RTX 5090 显卡 | 已物理安装到服务器上 |
| SSH 登录方式 | 用户名 + 密码，或 SSH 密钥 |
| 本地电脑 | Mac / Windows / Linux 均可 |

---

## 1. 连接到服务器

打开你电脑的终端（Mac: Terminal, Windows: PowerShell 或 CMD）：

```bash
# 把 user 换成你的用户名，把 1.2.3.4 换成服务器 IP
ssh user@1.2.3.4
```

第一次连接会问你 `Are you sure you want to continue connecting?`，输入 `yes` 回车。

然后输入密码（输入时屏幕不会显示任何字符，这是正常的）。

> 如果你用的是密钥登录：
> ```bash
> ssh -i ~/.ssh/your_key.pem user@1.2.3.4
> ```

---

## 2. 上传安装脚本到服务器

### 方法 A：从本地上传（推荐）

在**你的电脑**上（不是服务器），打开一个新终端：

```bash
# 把路径、用户名、IP 换成你自己的
scp /path/to/setup_server.sh user@1.2.3.4:~/
```

Mac 用户实际命令示例：

```bash
scp ~/path/to/geoApple/scripts/setup_server.sh user@SERVER_IP:~/
```

### 方法 B：在服务器上直接创建

如果上传不方便，可以在服务器上直接用 `vim` 或 `nano` 创建：

```bash
# 在服务器上执行
nano ~/setup_server.sh
# 粘贴脚本内容，然后按 Ctrl+X，按 Y，按 Enter 保存
```

---

## 3. 运行安装脚本

```bash
# 给脚本执行权限
chmod +x ~/setup_server.sh

# 用 sudo 运行（需要管理员权限）
sudo ~/setup_server.sh
```

脚本会自动安装以下内容（大约需要 15-30 分钟，取决于网速）：

```
Step 1/7: 系统基础包（git, cmake, 编译工具等）
Step 2/7: NVIDIA 显卡驱动（570+）
Step 3/7: CUDA 12.8（GPU 计算框架）
Step 4/7: Miniconda（Python 环境管理器）
Step 5/7: uv（快速 Python 包管理器）
Step 6/7: 创建 conda 环境 + 安装 PyTorch、YOLO 等
Step 7/7: 验证所有安装是否成功
```

> 安装过程中如果断网怎么办？重新运行脚本即可，它会跳过已安装的部分。

---

## 4. 重启服务器（重要！）

如果脚本提示安装了新的 NVIDIA 驱动，**必须重启**：

```bash
sudo reboot
```

重启后重新 SSH 连接：

```bash
ssh user@1.2.3.4
```

---

## 5. 验证安装

重启后，逐步检查每个组件：

### 5.1 检查显卡驱动

```bash
nvidia-smi
```

正常输出应该类似：

```
+---------------------------+
| NVIDIA-SMI 570.x.x       |
| Driver Version: 570.x.x  |
| CUDA Version: 12.8       |
|---------------------------+
| GPU  Name        Mem      |
| 0    RTX 5090    32GB     |
+---------------------------+
```

### 5.2 激活 conda 环境

```bash
conda activate geo_apple
```

> 如果提示 `conda: command not found`，先执行：
> ```bash
> source ~/miniconda3/etc/profile.d/conda.sh
> conda activate geo_apple
> ```

### 5.3 检查 Python

```bash
python --version
# 应该显示: Python 3.12.x
```

### 5.4 检查 PyTorch 能否使用 GPU

```bash
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'GPU 名称: {torch.cuda.get_device_name(0)}')
print(f'GPU 显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB')
"
```

正常输出：

```
PyTorch 版本: 2.x.x
CUDA 可用: True
GPU 名称: NVIDIA GeForce RTX 5090
GPU 显存: 32 GB
```

### 5.5 检查 YOLO

```bash
yolo checks
```

---

## 6. 日常使用

### 每次 SSH 登录后，先激活环境

```bash
conda activate geo_apple
```

### 运行训练

```bash
# 进入项目目录
cd ~/geo_apple_detection

# 训练 YOLO 模型（示例）
yolo segment train model=yolo11n-seg.pt data=dataset.yaml epochs=100 imgsz=640
```

### 查看 GPU 使用情况

```bash
# 实时监控（每 1 秒刷新）
watch -n 1 nvidia-smi
# 按 Ctrl+C 退出
```

### 后台训练（断开 SSH 后继续运行）

```bash
# 方法 1：使用 tmux（推荐）
tmux new -s train              # 创建一个名为 train 的会话
conda activate geo_apple       # 激活环境
python train.py                # 开始训练
# 按 Ctrl+B 然后按 D 脱离会话（训练继续在后台运行）

# 重新连接到会话
tmux attach -t train

# 方法 2：使用 nohup
nohup python train.py > train.log 2>&1 &
# 查看日志
tail -f train.log
```

---

## 7. 常见问题

### Q: `nvidia-smi` 显示 `command not found`

重启后再试。如果还不行：

```bash
# 检查驱动是否安装
dpkg -l | grep nvidia-driver
# 如果没有，手动安装
sudo apt install nvidia-driver-570
sudo reboot
```

### Q: `conda activate` 提示 `command not found`

```bash
# 初始化 conda
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda activate geo_apple
```

### Q: PyTorch 显示 `CUDA 可用: False`

```bash
# 检查 CUDA 版本是否匹配
nvcc --version
python -c "import torch; print(torch.version.cuda)"
# 两个版本号应该一致（都是 12.8）

# 如果不一致，重新安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Q: 训练时显示 `CUDA out of memory`

```bash
# 减小 batch size
yolo train ... batch=8   # 把数字改小

# 或者清理 GPU 显存
nvidia-smi                              # 看看哪个进程占用了 GPU
kill -9 <PID>                           # 杀掉占用 GPU 的进程
```

### Q: 安装包时很慢

使用国内镜像源加速：

```bash
# pip 使用清华源
pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple

# conda 使用清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --set show_channel_urls yes
```

### Q: SSH 断开后训练就停了

一定要使用 `tmux` 或 `nohup`（见第 6 节）。直接在 SSH 里运行，断开连接后进程会被杀掉。

---

## 8. 常用命令速查表

| 目的 | 命令 |
|------|------|
| 连接服务器 | `ssh user@1.2.3.4` |
| 激活环境 | `conda activate geo_apple` |
| 退出环境 | `conda deactivate` |
| 查看 GPU | `nvidia-smi` |
| 实时监控 GPU | `watch -n 1 nvidia-smi` |
| 查看磁盘空间 | `df -h` |
| 查看内存 | `free -h` |
| 查看 CPU 使用 | `htop` |
| 后台训练 | `tmux new -s train` |
| 回到后台会话 | `tmux attach -t train` |
| 查看所有会话 | `tmux ls` |
| 上传文件到服务器 | `scp local_file user@ip:~/remote_path/` |
| 从服务器下载文件 | `scp user@ip:~/remote_file ./local_path/` |
| 上传文件夹 | `scp -r local_dir user@ip:~/` |

---

## 9. 文件传输进阶

### 上传整个项目到服务器

```bash
# 在你的电脑上执行（排除大文件和数据集）
rsync -avz --progress \
    --exclude='*.zip' \
    --exclude='data/' \
    --exclude='runs/' \
    --exclude='.git/' \
    ~/ldxy/ldxy_ml/geo_apple_detection/ \
    user@1.2.3.4:~/geo_apple_detection/
```

### 从服务器下载训练结果

```bash
# 下载训练结果到本地
scp -r user@1.2.3.4:~/geo_apple_detection/runs/ ./runs/
```

---

> 安装日志保存在服务器的 `/tmp/setup_server_*.log`，遇到问题可以查看。
