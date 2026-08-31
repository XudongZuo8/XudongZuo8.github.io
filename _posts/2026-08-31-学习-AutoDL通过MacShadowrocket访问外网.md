---
title: "让 AutoDL 借用 Mac 的 Shadowrocket：用 SSH RemoteForward + autossh 搭一条稳定隧道"
date: 2026-08-31
excerpt: "把开发连接与代理隧道解耦，让 AutoDL 按需复用 Mac 上的 Shadowrocket。"
tags:
  - 服务器
  - SSH
  - AutoDL
  - macOS
---

今天给 AutoDL 配了一条“借用 Mac 出口”的代理链路：服务器上的程序访问外网时，流量通过 SSH 反向端口转发回到 Mac，再交给 Mac 上的 Shadowrocket。真正值得记录的不是某一条命令，而是一个连接生命周期的设计取舍。

## 先说结论：两条 SSH 连接，各司其职

最初我把 <code>RemoteForward</code> 塞进 VS Code Remote-SSH 的配置里。这样确实能用，但代理隧道会跟着 VS Code 的重连一起消失，最后表现为 AutoDL 上的 <code>Connection refused</code>。

更稳妥的拆法是：

- VS Code Remote-SSH：只负责编辑代码、打开终端和调试。
- <code>autossh</code>：单独维护一条只做端口转发的 SSH 连接。

两条连接互不依赖。VS Code 重连时，代理隧道不会被顺手带走。

<div class="blog-diagram diagram-architecture" role="img" aria-label="AutoDL 与 Mac 之间的最终架构，开发连接和代理隧道是两条独立连接">
  <span class="diagram-kicker">最终架构</span>
  <div class="diagram-caption">开发连接与代理隧道，各自拥有独立的生命周期</div>
  <div class="architecture-grid">
    <div class="architecture-node node-autodl">
      <span class="node-label">Server</span>
      <strong>AutoDL</strong>
      <span class="node-detail">Shell · Python · npm · pip</span>
      <span class="node-port">入口：127.0.0.1:17897</span>
    </div>
    <div class="architecture-lanes">
      <div class="architecture-lane lane-dev">
        <span class="lane-tag">开发连接</span>
        <span class="lane-arrow">←</span>
        <span class="lane-track"></span>
        <span class="lane-note">VS Code SSH</span>
      </div>
      <div class="architecture-lane lane-proxy">
        <span class="lane-tag">代理流量</span>
        <span class="lane-track"></span>
        <span class="lane-arrow">→</span>
        <span class="lane-note">RemoteForward</span>
      </div>
    </div>
    <div class="architecture-node node-mac">
      <span class="node-label">Client</span>
      <strong>Mac</strong>
      <span class="node-detail">VS Code · autossh</span>
      <div class="node-stack">
        <span class="node-chip">Shadowrocket：127.0.0.1:1082</span>
        <span class="node-chip node-chip-accent">代理出口 → Internet</span>
      </div>
    </div>
  </div>
  <p class="diagram-note">左侧的 AutoDL 请求经过 <code>:17897</code> 到达 Mac 的 <code>:1082</code>；上方的开发连接只是另一条 SSH 会话。</p>
</div>

这里刻意用了两个端口：<code>17897</code> 是 AutoDL 上给应用使用的代理入口，<code>1082</code> 是 Mac 上 Shadowrocket 的实际监听端口。端口不同，排错时不容易把“服务器入口”和“本地出口”混在一起。

## RemoteForward 到底把什么转给了谁

Mac 的 <code>~/.ssh/config</code> 中，为这条长期隧道单独写一个 Host：

~~~ssh
Host autodl-proxy
  HostName connect.bjb2.seetacloud.com
  Port 55000
  User root

  IdentityFile ~/.ssh/autodl_ed25519
  IdentitiesOnly yes

  RemoteForward 127.0.0.1:17897 127.0.0.1:1082
  ExitOnForwardFailure yes
  ServerAliveInterval 20
  ServerAliveCountMax 3
  TCPKeepAlive yes
~~~

最关键的是：

~~~ssh
RemoteForward 127.0.0.1:17897 127.0.0.1:1082
~~~

它的含义是：SSH 登录到 AutoDL 后，在 AutoDL 上监听 <code>127.0.0.1:17897</code>；当 AutoDL 上的程序连接这个端口时，数据通过 SSH 隧道回到 Mac，再交给 Mac 的 <code>127.0.0.1:1082</code>。

也就是：

~~~text
AutoDL application
      ↓
AutoDL:17897
      ↓  SSH RemoteForward
Mac:1082
      ↓
Shadowrocket → Internet
~~~

<code>ExitOnForwardFailure yes</code> 也很重要：如果 <code>17897</code> 没有成功监听，SSH 直接退出，<code>autossh</code> 才能判断“隧道没有真正建起来”。否则可能出现 SSH 看起来在线，但代理端口实际不可用的假状态。

## 为什么一定要 SSH Key

<code>autossh</code> 只能自动重启 SSH，不能在凌晨断线后替你输入密码。因此这条连接必须能够免密重连：

~~~ssh
IdentityFile ~/.ssh/autodl_ed25519
IdentitiesOnly yes
~~~

私钥不要写进博客或提交到仓库；这里的文件名只是示例，按自己的密钥路径替换即可。

## autossh 不是另一种 SSH

普通 SSH 先跑通，再交给 <code>autossh</code> 守护：

~~~bash
# 先测试：终端没有输出并一直等待，是正常的
ssh -N autodl-proxy

# 测试通过后，再放到后台持续维护
AUTOSSH_GATETIME=0 autossh -f -M 0 -N autodl-proxy
~~~

- <code>-N</code>：只建立转发，不打开远端 Shell。
- <code>-f</code>：放到后台运行。
- <code>-M 0</code>：不额外使用 autossh monitor port，主要依赖 SSH 的存活检测。
- <code>AUTOSSH_GATETIME=0</code>：初始连接失败时也继续尝试重连。

它的关系可以压缩成一句话：<code>autossh</code> 启动并监控 <code>ssh</code>，SSH 再读取配置、使用密钥认证并建立 <code>RemoteForward</code>。

<div class="blog-diagram diagram-stack" role="img" aria-label="autossh 监控并重启 ssh，ssh 负责读取配置、密钥认证、存活检测和端口转发">
  <span class="diagram-kicker">进程关系</span>
  <div class="diagram-caption">autossh 不替代 ssh，它只负责让 ssh 活得久一点</div>
  <div class="stack-box primary"><strong>autossh</strong><span>monitor / restart</span></div>
  <span class="stack-arrow">↓</span>
  <div class="stack-box"><strong>ssh autodl-proxy</strong><span>真正建立 SSH 连接</span></div>
  <span class="stack-arrow">↓</span>
  <div class="stack-tools">
    <span class="stack-tool">~/.ssh/config</span>
    <span class="stack-tool">SSH Key</span>
    <span class="stack-tool">ServerAlive</span>
    <span class="stack-tool">RemoteForward</span>
  </div>
</div>

## 一个容易踩的坑：Fake-IP 不该接管 SSH 控制连接

配置过程中遇到过类似：

~~~text
Connection closed by 198.18.0.27 port 55000
~~~

<code>198.18.x.x</code> 往往是 Shadowrocket TUN/Fake-IP 模式返回的虚拟地址，不是云服务器的真实公网地址。如果 SSH 域名被解析成 Fake-IP，SSH 自己的控制连接就可能被错误地送进代理规则。

正确的职责划分应该是：

- Mac → AutoDL 的 SSH 控制连接：使用真实 IP，并对云厂商 SSH 域名设置 <code>DIRECT</code>。
- AutoDL → Mac:1082 的 HTTP/HTTPS 流量：再交给 Shadowrocket 代理出去。

<div class="blog-diagram diagram-fakeip" role="img" aria-label="错误的 Fake-IP SSH 链路与正确的 REAL-IP DIRECT SSH 链路对比">
  <span class="diagram-kicker">网络职责</span>
  <div class="diagram-caption">SSH 控制流量直连；AutoDL 的外网请求才进入 Shadowrocket</div>
  <div class="fakeip-grid">
    <div class="fakeip-card is-wrong">
      <span class="compare-badge">容易出问题</span>
      <strong>SSH 被 Fake-IP 接管</strong>
      <div class="fakeip-step"><code>connect.bjb2...</code></div>
      <div class="fakeip-step">↓ Shadowrocket DNS</div>
      <div class="fakeip-step"><code>198.18.x.x</code> 虚拟地址</div>
      <div class="fakeip-step">TUN / 代理规则 → SSH 不稳定</div>
    </div>
    <div class="fakeip-card is-right">
      <span class="compare-badge">推荐</span>
      <strong>控制连接与代理流量分工</strong>
      <div class="fakeip-step"><code>connect.bjb2...</code></div>
      <div class="fakeip-step">↓ REAL-IP + DIRECT</div>
      <div class="fakeip-step">SSH RemoteForward</div>
      <div class="fakeip-step">AutoDL 外网请求 → Shadowrocket</div>
    </div>
  </div>
</div>

## AutoDL 端按需使用代理

隧道保持在线，不代表所有请求都必须走代理。需要访问外网时再显式指定：

~~~bash
curl -I -x http://127.0.0.1:17897 https://github.com
~~~

也可以临时设置环境变量：

~~~bash
export http_proxy=http://127.0.0.1:17897
export https_proxy=http://127.0.0.1:17897
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
~~~

不建议把它无脑写进所有服务器环境：国内源、内网服务和不需要代理的请求，没必要绕一圈回 Mac；而且 Mac 睡眠或关机后，这条出口必然不可用。

## 验证顺序：先证明链路，再检查守护进程

以后代理失效时，按下面顺序排查，能少走很多弯路：

<div class="blog-diagram diagram-diagnostics" role="img" aria-label="AutoDL 代理不可用时的五步排查流程：Shadowrocket、普通 SSH、RemoteForward、autossh">
  <span class="diagram-kicker">排错流程</span>
  <div class="diagram-caption">不要一上来重装 autossh；先确认它下面依赖的每一层</div>
  <ol class="diagnostic-flow">
    <li class="diagnostic-step">
      <span class="step-number">1</span>
      <div class="step-content">
        <strong>Mac 的 Shadowrocket :1082 正常吗？</strong>
        <span>先确认本地代理出口存在。</span>
        <span class="step-branch">否 → 先修 Shadowrocket / 代理规则</span>
      </div>
    </li>
    <li class="diagnostic-step">
      <span class="step-number">2</span>
      <div class="step-content">
        <strong><code>ssh -N autodl-proxy</code> 能免密连上吗？</strong>
        <span>若看到 198.18.x.x，优先检查 Fake-IP、DNS 和 DIRECT 规则。</span>
        <span class="step-branch">否 → 检查 Host、Port、SSH Key 和 DNS</span>
      </div>
    </li>
    <li class="diagnostic-step">
      <span class="step-number">3</span>
      <div class="step-content">
        <strong>AutoDL 的 :17897 能访问外网吗？</strong>
        <span>使用 <code>curl -x http://127.0.0.1:17897 ...</code> 验证整条转发链路。</span>
        <span class="step-branch">否 → 检查 RemoteForward 和 ExitOnForwardFailure</span>
      </div>
    </li>
    <li class="diagnostic-step">
      <span class="step-number">4</span>
      <div class="step-content">
        <strong>Mac 上的 autossh 进程还在吗？</strong>
        <span>普通 SSH 已经验证通过后，再检查后台守护进程。</span>
        <span class="step-branch">否 → 重新启动 autossh</span>
      </div>
    </li>
    <li class="diagnostic-step">
      <span class="step-number">✓</span>
      <div class="step-content">
        <strong>代理链路恢复</strong>
        <span>AutoDL:17897 → Mac:1082 → Shadowrocket → Internet</span>
      </div>
    </li>
  </ol>
</div>

## 最后的边界

这套方案能自动恢复的是“短暂的网络抖动”和“SSH 会话断开”，不能让关机的 Mac 继续提供出口：

~~~text
Mac 在线 + Shadowrocket 在线 + autossh 在线
→ AutoDL 代理可用

Mac 睡眠 / 关机 / 断网
→ 代理出口不可用
~~~

如果 AutoDL 换了实例，通常只需要更新 SSH Host 中的 <code>HostName</code> 和 <code>Port</code>，<code>RemoteForward</code>、<code>autossh</code> 和 Shadowrocket 的整体架构不需要重做。

这次配置最后留下的工程经验很简单：

> 开发工具的连接，服务于人的操作；长期基础设施连接，应该由独立的守护进程维护。
