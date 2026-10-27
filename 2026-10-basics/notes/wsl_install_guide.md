# WSL2 安装指南（为 W19 的 vLLM / SGLang 提前铺路）

> 本机实测数据：2026-10-27。硬件若有变化（换机、换盘），先重跑文末的「环境体检」再照着执行。
> 本文所有命令都指向 **F 盘**，原因是 C 盘只剩 12.4GB，装不下 WSL。**跳过这一步会把系统盘挤爆。**

---

## 0. 为什么现在就要装

日历把 vLLM 排在 W19（2027.02），但：

- `vLLM` / `SGLang` **不支持 Windows 原生**，只能在 Linux 下跑
- Windows 上唯一可行方案是 **WSL2（Windows Subsystem for Linux 2，Windows 的 Linux 子系统，第二代走轻量虚拟机）**
- 装 WSL2 会踩：虚拟化开关、驱动直通、磁盘位置、内存分配四类坑。**这些坑在 W19 撞上会卡一整周，现在装好只用半小时**

装完不用管，W19 直接用。

---

## 1. 本机现状（2026-10-27 实测）

| 项目 | 值 | 影响 |
|---|---|---|
| 系统 | Windows 11 家庭版 | 没有 Hyper-V 管理器，属正常，**不需要 Hyper-V** |
| CPU | 虚拟化已在 BIOS 开启（`VirtualizationFirmwareEnabled=True`） | ✅ 不用进 BIOS 折腾 |
| 显卡 | NVIDIA GeForce RTX 3060 Laptop GPU，**6144 MiB = 6GB** | 见第 6 节「能跑什么」 |
| 驱动 | 566.07 | **WSL 复用这个驱动，不要在里面另装** |
| 内存 | 15.6 GB 总（当时可用仅 2.5GB） | 必须用 `.wslconfig` 限内存，否则 WSL 默认抢一半 |
| C 盘 | 177.8 GB 总，**剩余 12.4 GB** | ⚠️ **绝对不能装 C 盘** |
| F 盘 | 1907.7 GB 总，剩余 1029.6 GB | ✅ 装这里 |
| `.wslconfig` | 已创建：`C:\Users\lenovo\.wslconfig` | 限 6GB 内存 / 8 核 / swap 放 F 盘 |
| `F:\WSL` 目录 | 已创建 | 存放发行版与 swap |

---

## 2. 完整步骤（6 步）

### ① 以管理员身份打开 PowerShell 7

开始菜单搜 **PowerShell 7** → 右键 → **以管理员身份运行**。

窗口标题栏必须带「管理员」字样。当前权限不是管理员的话，`wsl --install` 会静默失败或报拒绝访问。

### ② 只装 WSL 本体，先不装发行版

```powershell
wsl --install --no-distribution
```

- `--no-distribution`：只装 WSL 引擎，不自动下载 Ubuntu（默认行为会把 Ubuntu 装到 C 盘）
- 这一步会启用两个 Windows 功能：`Microsoft-Windows-Subsystem-Linux`、`VirtualMachinePlatform`

### ③ 重启电脑（必须）

Windows 可选功能要重启才生效。**不重启直接做 ④ 会报错。**

### ④ 重启后，再次以管理员打开 PowerShell 7，把 Ubuntu 装到 F 盘

```powershell
wsl --install -d Ubuntu-22.04 --location "F:\WSL\Ubuntu2204"
```

- 选 **22.04** 不选 24.04：vLLM / SGLang 在 22.04 上验证最充分，Python 3.10 生态最稳
- `--location` 是 WSL 2.0+ 才有的参数。**报「未知参数」就走 ⑤**

### ⑤ 备选方案（若 ④ 报「未知参数 --location」）

先装再迁：

```powershell
wsl --install -d Ubuntu-22.04
wsl --shutdown
wsl --export Ubuntu-22.04 F:\WSL\ubuntu2204.tar
wsl --unregister Ubuntu-22.04
wsl --import Ubuntu2204 F:\WSL\Ubuntu2204 F:\WSL\ubuntu2204.tar --version 2
```

迁完确认位置（应显示挂在 `/dev/sdX` 且总量对应 F 盘）：

```powershell
wsl -d Ubuntu2204 -- df -h /
```

### ⑥ 首次启动，设置 UNIX 用户名密码

```powershell
wsl -d Ubuntu-22.04
```

- 提示输入用户名：**小写英文**（如 `lenovo`），**不要用 root**
- 密码输入时**屏幕不显示任何字符**，这是正常的，照打回车
- 若 ④ 用了 `--location`，发行版名就是 `Ubuntu-22.04`；若走了 ⑤ 的 `--import`，名字是 `Ubuntu2204`

---

## 3. 装完的 3 条验证（依次通过才算成功）

```powershell
# 1. WSL 版本：要 ≥ 2.0
wsl --version

# 2. 发行版列表：STATE 应为 Running，VERSION 应为 2
wsl -l -v

# 3. GPU 直通（最关键）
wsl -d Ubuntu-22.04 -- nvidia-smi
```

**第 3 条要看到这一行才算通：**

```
NVIDIA GeForce RTX 3060 Laptop GPU   6144MiB
```

看到 6144MiB 就说明 W19 的 vLLM 有路了。看不到就看第 7 节排查。

---

## 4. 四个必须提前知道的坑

### 坑 1：绝对不要在 WSL 里装 NVIDIA 驱动 ⚠️ 最严重

进去以后**不要**执行：

```bash
sudo apt install nvidia-driver-*      # ❌ 禁止
```

**原因**：WSL 的 GPU 是复用 Windows 侧驱动的（你装的 566.07），WSL 会自动把驱动挂到 `/usr/lib/wsl/lib/`。在 Linux 里再装一遍驱动会**破坏直通**，恢复起来要卸内核模块、重装系统级组件，非常麻烦。

**要装只装 CUDA Toolkit**（用于编译），不装驱动：

```bash
# 需要时才装，不需要就先不装
# 用 NVIDIA 官方 WSL-Ubuntu 源，不要用 Ubuntu 自带源
```

### 坑 2：C 盘只剩 12.4GB，别让 WSL 回落到 C

WSL 默认装 C 盘。Ubuntu（约 2GB）+ CUDA Toolkit（约 5GB）+ PyTorch（约 3GB）+ 模型权重（1.5B fp16 约 3GB）→ **至少 13GB，直接撑爆**。

装完必须验：

```powershell
wsl -d Ubuntu-22.04 -- df -h /
```

如果总量和 C 盘对得上（约 177GB 的样子不对，C 盘现剩 12.4GB 就更跑不起来），说明装错了，重走 ⑤ 迁移。

### 坑 3：内存 15.6GB，WSL 默认抢一半

`C:\Users\lenovo\.wslconfig` 已写好：

```ini
[wsl2]
memory=6GB
processors=8
swap=4GB
swapFile=F:\\WSL\\swap.vhdx

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
```

- **改完要 `wsl --shutdown` 再启动才生效**
- 如果 `wsl --version` 显示低于 2.0，把 `[experimental]` 整段删掉（旧版不认这两个参数）
- `memory=6GB` 是刻意的：给 Windows 留 9GB，同时装完 CUDA 后训练时也不会把系统拖死。真要训练时再临时调到 10GB

### 坑 4：不要给家庭版装 Hyper-V

Windows 11 家庭版没有 Hyper-V 管理器，**这是正常的**。WSL2 走的是 `VirtualMachinePlatform`，不需要 Hyper-V 图形界面。

网上「家庭版怎么启用 Hyper-V」的教程是绕远路，装了反而可能和 WSL2 抢资源。

---

## 5. 验证 GPU 直通的完整自查（`nvidia-smi` 失败的排查顺序）

按概率从高到低：

| 现象 | 原因 | 处理 |
|---|---|---|
| `nvidia-smi: command not found` | WSL 没拿到 Windows 侧驱动挂载点 | 在 WSL 里执行 `ls /usr/lib/wsl/lib/` 看有没有 `nvidia-smi`；有就用 `/usr/lib/wsl/lib/nvidia-smi` 全路径 |
| `ls /usr/lib/wsl/lib/` 也空 | Windows 驱动太旧（WSL GPU 直通需要 ≥ 470 系） | 你当前 566.07，应该没问题；若换机后出现，升级 Windows 侧驱动 |
| `Failed to initialize NVML: GPU access blocked by the operating system` | Windows 侧驱动状态异常 | Windows 里重启一次，或 `wsl --shutdown` 后重开 |
| 能看到卡但显存显示 4096MiB 之类 | 笔记本有独显/核显切换 | 在 NVIDIA 控制面板里把 `wsl.exe` 设为「高性能 NVIDIA 处理器」 |
| 完全没输出 | 虚拟化被关（BIOS 恢复默认过） | 任务管理器 → 性能 → CPU → 看「虚拟化：已启用」 |

---

## 6. 6GB 显存在 WSL 里能跑什么（决定 W19 的方案）

| 任务 | 需要显存 | 6GB 能跑 |
|---|---|---|
| vLLM 推理 0.5B / 1.5B（fp16） | 2—4GB + KV cache | ✅ |
| vLLM 推理 7B（fp16） | 14GB | ❌ |
| QLoRA 微调 1.5B（4bit） | 约 3GB | ✅ |
| LoRA 微调 0.5B | 3—4GB | ✅ |
| 全参微调 0.5B | 8GB+ | ❌（W10 租卡跑一次） |
| 7B 的 4bit 推理（AWQ/GPTQ） | 4.5—5.5GB | ⚠️ 极限，要压 `max_model_len` 和 `gpu_memory_utilization` |

**结论：W19—W22 的 vLLM 对照实验统一用 1.5B，7B 不上本地。** 别为了「跑个 7B」折腾两天，租卡 2 小时约 6 元更划算。

---

## 7. 常见错误速查

| 报错 | 含义 | 处理 |
|---|---|---|
| `WslRegisterDistribution failed with error: 0x80370102` | 虚拟化未启用 / 功能未生效 | 确认 `VirtualMachinePlatform` 已启用并重启 |
| `error: 0x8007019e` | 没装 WSL 可选功能 | 回到第 ② 步 |
| `Please enable the Virtual Machine Platform Windows feature` | 缺功能 | `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart` 后重启 |
| `--location` 未知参数 | WSL 版本旧 | 走第 ⑤ 步的 export/import |
| `wsl --install` 卡在 0% | 网络问题（微软源被墙） | 挂代理重试，或手动下载 `.wsl` 包离线安装 |
| 磁盘空间不足 | 装到 C 盘了 | 重走第 ⑤ 步迁移到 `F:\WSL` |

---

## 8. 装完之后（W19 之前都不用管）

- **不要**在 WSL 里装 Python 环境的包，也不要和 Windows 侧的 `D:\anaconda3\envs\aigc` 混用
- W19 才需要：`pip install vllm` / `pip install sglang`，届时单独开虚拟环境
- 想随时停掉 WSL 释放内存：`wsl --shutdown`

---

## 9. 环境体检（换机或出问题时重跑）

```powershell
# 系统与虚拟化
(Get-CimInstance Win32_OperatingSystem) | Select-Object Caption, Version, BuildNumber
Get-CimInstance Win32_Processor | Select-Object Name, VirtualizationFirmwareEnabled
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux, VirtualMachinePlatform |
    Select-Object FeatureName, State

# 磁盘与内存
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" |
    Select-Object DeviceID, @{n='TotalGB';e={[math]::Round($_.Size/1GB,1)}}, @{n='FreeGB';e={[math]::Round($_.FreeSpace/1GB,1)}}
$os = Get-CimInstance Win32_OperatingSystem
"总内存 GB: " + [math]::Round($os.TotalVisibleMemorySize/1MB,1)

# 显卡
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# WSL 状态
wsl --version
wsl -l -v
```

---

## 10. 一句话总结

**管理员 PowerShell → `wsl --install --no-distribution` → 重启 → `wsl --install -d Ubuntu-22.04 --location "F:\WSL\Ubuntu2204"` → 设用户名 → 验 `nvidia-smi` 看到 6144MiB。**

四条铁律：**只装 F 盘、不装 Linux 驱动、改完 `.wslconfig` 要 `wsl --shutdown`、不用管 Hyper-V。**
