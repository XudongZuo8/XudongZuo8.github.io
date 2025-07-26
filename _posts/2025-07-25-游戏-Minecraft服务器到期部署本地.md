本博客主要分享在和女朋友玩Minecraft时遇到的一些问题和解决方案

### 一、基本介绍

**server基本命令：**

请查看[Minecraft维基百科](https://minecraft.fandom.com/zh/wiki/Server.properties#Java%E7%89%88)

**1.如何启动游戏**

`nohup java -Xmx2048M -Xms1024M -jar minecraft_server.1.16.5.jar nogui &`

**2.一些小技巧**

用服务器搭建minecraft，我们肯定希望这个服务器是能够随时访问的，而不是服主关闭自己的笔记本，项目就停下运行。

我们可以使用tmux来创建一个在后台运行的进程，以便我们可以随时访问游戏。

**3.服务器内存分配注意**

>[root@iZbp18scv7r5s0p0ipu4p8Z ~]# free -h
>                total        used        free      shared  buff/cache   available
>Mem:  1.7G        1.4G         67M        660K        300M        212M
>Swap:   0B          0B          0B

我的内存使用情况比较极限，其中Swap为零，这意味着在物理内存耗尽时，我的mc就崩掉了。

**内存分配问题**

- 当前的物理内存非常紧张，仅剩 **67MB 空闲**，且即便考虑缓存也只有 **212MB 可用**。

- 没有配置 Swap 分区，在物理内存耗尽时，系统可能会强制终止一些程序（OOM: Out of Memory）

**解决方案**

1.增加swap分区

```markdown
# 创建一个2GB的Swap文件
mkdir -p ~/myswapfile
sudo dd if=/dev/zero of=~/myswapfile/swapfile bs=1M count=2048

# 设置Swap文件权限
sudo chmod 600 /swapfile

# 启用新的 swapfile
sudo mkswap ~/myswapfile/swapfile
sudo swapon ~/myswapfile/swapfile

# 启用Swap
sudo swapon /swapfile

# 确保开机自动加载
echo '/root/myswapfile/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

详见这篇文站：https://blog.csdn.net/Datapad/article/details/115965665

**为什么需要 mkswap？**

在你创建一个文件（例如用 dd 命令生成一个固定大小的文件）后，操作系统并不知道该文件的用途。通过 mkswap 命令，你告诉操作系统这个文件是一个交换文件，随后可以通过 swapon 激活它。

检测命令是否生效：

```markdown
sudo swapoff -a
sudo swapon -a
```

**4.mc内部的命令**

Java 版mc搭建完毕以后，想要使用命令一般无需更改文件设置。

>只需要在服务器端创建世界以后，在**服务器的控制台上**输入**op 用户名**，即可在游戏内使用所有的指令，但是在设置op之前，**游戏内输入op 用户名是没有任何用的**。另外注意，输入的用户名不需要加中括号。

### 二、配置过程

下面主要是因为买的服务器是2G内存，需要配置swap防止系统卡死。

![image-2024122501](/images/images_for_blog/image-2024122501.png)

![image-2024122502](/images/images_for_blog/image-2024122502.png)

到这里就完成虚拟内存的配置了，实际效果如何等晚上打游戏再说。

### 三、本地部署（这里才是今天做的事情）

 在服务器上有时会网络不稳定，而且到期续费要花钱，也可以选择本地部署，即开即玩。

下面是从服务器转移到本地必须下载的文件

```bash
# 核心文件
minecraft_server.1.16.5.jar    # 服务器程序
eula.txt                        # 许可协议
server.properties              # 服务器配置

# 游戏数据（最重要！）
world/                         # 游戏世界数据（包含所有建筑、进度）
banned-ips.json               # 封禁IP列表
banned-players.json           # 封禁玩家列表
ops.json                      # 管理员列表
whitelist.json                # 白名单
usercache.json                # 用户缓存
```

（1）首先确保Java环境

```bash
java -version
```

Minecraft 1.16.5 需要 Java 8 或更高版本。

(2)启动服务器

文件都下载到本地以后，新建一文件夹用存储这些文件，在此文件夹打开terminal。

```bash
java -Xmx2048M -Xms1024M -jar minecraft_server.1.16.5.jar nogui
```

其中Xmx是游戏占用的最大内存，Xms是最小的内存，最小内存不能占用过多，否则也会造成卡顿。

（3）连接游戏

- 打开Minecraft客户端
- 选择”多人游戏“
- 添加服务器，地址为：`localhost`或者`127.0.0.1`
- 端口默认为25565

接着打开本地的minecraft启动台，添加服务器，服务器名随便取，服务器地址输入：127.0.01

（4）内网穿透

```bash
# 下载ngrok后运行
ngrok tcp 25565 
# 会给一个类似0.tcp.ngrok.io:12345的地址，用这个连接
```

![image-2025072501](/images/images_for_blog/image-2025072501.png)

我的是mac所以用homebrew下载的ngrok，也可以使用`frp`、`花生壳`、`cpolar`等其他内网穿透工具。

![image-20250725232121715](/images/images_for_blog/image-2025072502.png)

现在把forwarding前面那个输入就行了，大功告成。

ps：若处于一个局域网下，课使用内网IP连接。

### 附录：

**1.什么是swap交换内存？**

![img](https://i-blog.csdnimg.cn/blog_migrate/55a8145924bef2d832c3c5b2a188dc45.png)

**2.ngrok注册激活教程（需要绑定银行卡）**

**步骤1：注册账号** 访问：https://dashboard.ngrok.com/signup

**步骤2：获取 authtoken** 登录后访问：https://dashboard.ngrok.com/get-started/your-authtoken

**步骤3：配置 authtoken**

```bash
ngrok config add-authtoken 你的token
```

**步骤4：启动隧道**

```bash
ngrok tcp 25565
```



