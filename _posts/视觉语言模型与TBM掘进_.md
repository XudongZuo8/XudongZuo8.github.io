---
title: Gemini调研的如何
date: 2025-07-26
excerpt: "VLM在TBM自主掘进的应用前景"
tags:
  - 自主掘进
---

# **感知型TBM：视觉语言模型在自主掘进中的集成应用调研报告** 

## **执行摘要**

本报告旨在对视觉语言模型（Vision Language Models, VLM）与全断面隧道掘进机（Tunnel Boring Machine, TBM）自主掘进技术的融合潜力进行深度调研与前瞻性分析。核心论点在于，VLM作为人工智能领域的前沿技术，其强大的视觉理解与语言推理能力，有望解决长期制约TBM实现更高水平自主化的感知、解译与决策瓶颈。

报告的关键发现涵盖了多个高价值应用领域。首先，在**实时地质智能**方面，VLM能够通过分析刀盘摄像头捕捉的掌子面图像，实现岩体质量的自动分类和地质异常（如断层、涌水）的实时预警，将TBM从“盲掘”状态转变为“感知掘进”。其次，在**预测性维护**领域，VLM可通过对刀具和输送带渣料的持续视觉监控，实现对刀具磨损状态的精准评估和对掘进效率的动态优化，从而显著降低停机风险与运营成本。再者，VLM在**安全保障**中的作用体现在其卓越的异常检测能力，能够及时发现塌方、卡机等地质灾害的早期视觉征兆，为操作人员提供关键的预警窗口。

为实现上述功能，本报告提出了一套VLM集成TBM的系统架构。该架构的核心是将VLM作为“认知引擎”，融合来自高分辨率摄像头、传统传感器（扭矩、推力、振动等）以及地质勘测的多模态数据流。通过板载边缘计算平台进行实时推理，并通过自然语言界面（NLI）将复杂的机器状态与地质环境以直观、可交互的方式呈现给操作员，最终实现自动化的每日运营与地质报告生成。

最后，报告勾勒出一条从试点研究到全面部署的战略路线图。该路线图以分阶段的方式推进，从初始的数据采集与模型验证，到“影子模式”下的在线测试，再到人机协作的“智能副驾”，最终迈向特定场景下的闭环控制与“自驱动”掘进。报告同时指出了实施过程中面临的主要挑战，包括领域专用数据集的构建、模型鲁棒性的保证以及安全认证体系的建立。总体而言，VLM技术的引入预示着TBM将从高度机械化的施工装备，向具备环境感知、自主推理与决策能力的智能化系统演进，为隧道工程行业带来一场深刻的范式革命。

---

## **第一部分 技术范式革命：视觉语言模型（VLM）导论**

本章旨在为非人工智能领域的工程技术专家，建立一个关于视觉语言模型（VLM）技术的坚实且技术上稳健的理解。其目标是揭开VLM的神秘面纱，并将其能力置于复杂工业问题的背景下进行阐述，为后续将其应用于TBM领域奠定理论基础。

### **1.1 核心架构：解构视觉与语言的融合**

#### **1.1.1 基本定义**

视觉语言模型（VLM）是一种先进的人工智能模型，它深度融合了计算机视觉（Computer Vision）与自然语言处理（Natural Language Processing, NLP）两大功能 1。VLM的核心任务是学习并映射文本数据与视觉数据（如图像或视频）之间的复杂关系。这种能力使得模型不仅能“看懂”视觉输入的内容，还能基于这些视觉信息生成描述性文本、回答相关问题，或是在自然语言指令的引导下进行视觉信息处理 1。作为一种典型的多模态人工智能系统，VLM能够同时接收图像/视频和文本作为输入，并以文本形式产出分析结果，例如生成图像标题、回答关于图像细节的问题，或识别视频中的特定对象与事件 1。

#### **1.1.2 两大支柱：编码器**

VLM的架构通常建立在两个关键组件之上：视觉编码器和语言编码器 1。

* **视觉编码器（Visual Encoder）**：其核心功能是从图像或视频输入中提取关键的视觉特征，如颜色、形状、纹理和空间关系，并将这些复杂的视觉信息转换成机器可以处理的数值表示，即向量嵌入（Vector Embeddings）1。在技术演进路线上，早期的VLM多采用卷积神经网络（Convolutional Neural Networks, CNN）进行特征提取。然而，更现代的VLM普遍采用视觉转换器（Vision Transformer, ViT）架构。ViT借鉴了语言模型中Transformer的成功经验，创新性地将图像分割成一系列固定大小的图块（Patches），并将这些图块视为一个序列，类似于语言模型处理句子中的单词（Tokens）1。这种基于图块的序列化处理方式，对于后续分析隧道掌子面或渣料堆的特定区域至关重要。  
* **语言编码器（Language Encoder）**：其主要职责是捕捉输入文本中单词和短语之间的语义及上下文关联，同样将其转换为文本嵌入 1。当今绝大多数VLM采用基于Transformer的特定神经网络架构作为其语言编码器，例如谷歌的BERT或OpenAI的GPT系列模型 1。Transformer架构通过其自注意力机制（Self-Attention Mechanism），能够动态地评估输入序列中不同词元的重要性，从而更深刻地理解语言的细微差别和长距离依赖关系。

#### **1.1.3 融合之桥：多模态融合**

VLM的魔力在于其能够将来自视觉和语言这两个独立数据流的信息有效对齐和融合。模型通过特定的训练策略，学习如何将图像的视觉嵌入与文本的语义嵌入关联起来。实现这一目标的关键技术之一是交叉注意力机制（Cross-Attention Mechanism），它允许模型在处理一种模态的信息时，“关注”另一种模态中的相关部分 1。举例来说，当VLM接收到指令“在隧道掌子面图像中寻找裂隙带”时，交叉注意力机制会引导模型在分析图像的视觉特征时，重点关注那些与“裂隙带”这个语言概念在语义上最相关的图像图块。这种深度融合是VLM能够执行复杂视觉推理任务的基础。

### **1.2 从识别到推理：VLM在工业环境中的关键能力**

VLM的能力并非凭空而来，而是通过精巧的训练策略和海量数据学习到的。这些策略催生了从简单识别到复杂推理的一系列关键能力。

#### **1.2.1 训练策略与涌现能力**

* **对比学习（Contrastive Learning）**：以CLIP（Contrastive Language-Image Pre-training）模型为代表，这种策略的核心思想是，在一个共享的嵌入空间中，将匹配的图像-文本对的表示拉近，同时将不匹配的对推远 1。CLIP通过在从互联网搜集的4亿个图像-文本对上进行训练，获得了强大的“零样本（Zero-shot）”分类能力。这意味着模型能够识别和分类它在训练中从未明确见过的对象或概念 1。对于可能遭遇未知地质构造的TBM而言，这种泛化能力至关重要。  
* **掩蔽建模（Masked Modeling）**：以FLAVA（Foundation Language and Vision Alignment）模型为例，该技术通过随机遮蔽输入文本或图像的一部分，并训练模型来预测被遮蔽的内容，从而迫使模型学习更深层次的上下文依赖关系 1。  
* **指令微调（Instruction Tuning）**：以LLaVA（Large Language and Vision Assistant）为代表，这是将一个基础模型转变为一个交互式“助手”的关键步骤。通过在一个包含大量指令-响应对（例如，“这张图片里有什么？”-“图片里有一只猫坐在沙发上”）的数据集上进行微调，模型学会了遵循人类指令并以对话方式进行交互 6。

#### **1.2.2 核心能力**

* **视觉问答（Visual Question Answering, VQA）**：这是VLM最基础也是最核心的能力之一，即能够用自然语言回答关于图像内容的问题 1。这为TBM操作员与机器的交互提供了全新的可能性，例如直接提问：“掌子面左上角是否存在涌水迹象？”  
* **图像描述与字幕生成（Image Captioning & Description）**：VLM能够自动为图像生成详尽的文本描述 2。这一能力是实现TBM掘进日报自动化的基础。  
* **结构化推理（Structured Reasoning）**：以LLaVA-o1等新一代模型为代表，VLM正在发展出更复杂的、类似人类的结构化推理能力。模型在给出最终结论前，会经历一个包括“总结问题、描述视觉信息、进行逻辑推理、生成结论”的多阶段内部思考过程 7。这种能力与地质工程师或设备工程师分析复杂故障的思维过程高度相似，对于TBM的复杂诊断任务极具价值。  
* **异常检测（Anomaly Detection）**：VLM能够学习一个场景的“正常”状态，并识别出与这种正常状态的偏差。这种偏差可以是结构性的（如一个破损的刀具），也可以是逻辑性的（如一个缺失的部件）8。对于高度关注安全的TBM操作而言，这可能是VLM最关键的应用能力之一。

### **1.3 前沿VLM框架对比分析**

VLM领域技术迭代迅速，涌现出多个具有代表性的模型框架，它们在架构、性能和应用模式上各有侧重。

* **GPT-4V / GPT-4o**：作为由OpenAI开发的闭源通用大模型，GPT-4V及其后续版本GPT-4o在通用推理、手写文字与图表数据解读、甚至从草图生成代码等方面展现了顶尖性能 11。其强大的能力已在“Be My Eyes”等真实世界应用中得到验证，帮助视障人士理解周围环境，展示了其作为通用视觉助手的巨大潜力 4。其架构基于Transformer，包含视觉编码器、多模态融合模块和解码器，可通过微软Azure等云服务API进行调用 3。  
* **LLaVA及其变体**：作为开源社区的领军模型，LLaVA系列开创性地利用GPT-4生成高质量的多模态指令微调数据，成功地为开源模型的发展“自举”了训练资源 6。LLaVA家族的快速发展体现了VLM领域的一个重要趋势：向专用化演进。针对不同应用场景，已衍生出多个变体，例如：  
  * **LLaVA-UHD**：专注于处理任意宽高比和高分辨率图像 15。  
  * **LLaVA-Mini**：追求高效推理和视频理解 16。  
  * **LLaVA-ST**：增强对时空细粒度信息的理解能力 17。  
  * **LLaVA-o1**：专为结构化、多步骤推理任务设计 7。  
  * **LLaVA-Read**：强化对图像内嵌文本的读取与理解 18。

    这一系列模型的发展路径表明，为TBM这类特定工业领域打造一个高度定制化的“TBM-VLM”不仅是可行的，而且是技术发展的必然方向。  
* **其他关键模型**：除上述两者外，市场上还存在众多其他重要的VLM，如谷歌的Gemini、Meta的Llama系列、阿里巴巴的Qwen-VL系列等，它们共同构成了当前VLM技术蓬勃发展的生态系统 1。

从VLM的技术演进中可以观察到两个深刻的趋势，这对T-BM的应用具有指导意义。

首先，**VLM领域的“专用化趋势”为构建“TBM-VLM”铺平了道路**。早期的VLM，如CLIP，是基于海量通用互联网数据训练的“通才”1。然而，近期的发展清晰地展示了一条从通用到专用的演进路径。LLaVA家族针对高分辨率图像、视频、文本读取等特定任务的细分 15，以及专为道路异常检测设计的URA-VLM 19，都印证了这一趋势。这与其他人工智能领域的发展历程如出一辙：强大的基础模型被微调和适配到金融、医疗、法律等高价值的垂直领域。因此，开发一个“TBM-VLM”并非天马行空的猜想，而是遵循了既有技术轨迹的逻辑延伸。其核心挑战已不再是“能否实现”，而是“如何”有效地收集和标注TBM领域的专用数据（如刀盘图像、渣料视频、传感器日志），并定义出符合工程需求的专业推理任务（如地质分类、磨损评估、灾害预测）以进行模型微调。LLaVA等开源模型的存在，为这类定制化开发提供了直接、可控的蓝图 6。

其次，**从“分类”到“对话式推理”的转变，是解锁TBM自主能力的关键**。传统的工业计算机视觉主要集中于分类（例如，判断岩体为III级或IV级）或目标检测 20，这些任务仅能提供孤立的数据点。而现代VLM通过指令微调，已经超越了简单的分类，实现了可解释的、对话式的交互和多步骤推理 2。它不仅能判断“这可能是IV级围岩”，更能解释其判断依据：“

**因为**我在图像右上象限观察到高密度的节理和涌水迹象”。TBM的运营决策并非基于单一数据点，而是依赖于对复杂、动态环境的综合判断，这正是人类专家的价值所在 22。因此，VLM在TBM中的真正价值，不仅仅是自动化“看”的过程，更是自动化其后“思考”的过程。VLM可以扮演一个认知副驾驶的角色，解释其推理过程，提供模仿经验丰富的专家的决策支持。这将推动TBM从简单的自动化（如根据扭矩调整转速）向真正的自主化（理解环境并提出整体掘进策略）迈进。

| 表1：面向TBM应用的前沿VLM框架对比分析 |  |  |  |  |  |  |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **模型名称** | **基础模型 (LLM/视觉编码器)** | **可用性** | **关键训练方法** | **核心优势** | **TBM任务适用性** | **关键信息来源** |
| GPT-4o / GPT-4V | OpenAI专有模型 / Transformer架构 | 闭源API | 指令微调, 多模态端到端训练 | 强大的通用推理能力、文本与数据图表解读、多模态交互 | 地质解译、自然语言交互、自动报告生成 | 4 |
| LLaVA 家族 (v1.6, UHD, o1, etc.) | Vicuna/Llama / CLIP ViT-L/14 | 开源 | GPT辅助的指令数据生成、指令微调 | 高度可定制、针对特定任务的变体（高分、推理）、社区活跃 | 可定制开发TBM专用模型，用于地质、渣料、磨损、安全的全方位分析 | 6 |
| Qwen-VL (通义千问VL) | Qwen LLM / ViT | 开源/API | 指令微调 | 强大的中英文理解能力、长视频理解、桌面/手机界面理解 | 地质视频分析、操作界面理解、中文报告生成 | 1 |
| Google Gemini | Google专有模型 | 闭源API | 多模态端到端训练 | 跨音频、图像、文本、视频的本地多模态能力 | 综合多源信息（振动声、视频）进行地质灾害预警 | 1 |
| Meta Llama 3.2-VLM | Llama 3.1 LLM / ViT | 开源 | 交叉注意力机制、指令微调 | 强大的开源LLM基础、视频适配器 | 可作为开发TBM-VLM的强大开源基座 | 1 |
| CLIP | ViT / Transformer | 开源 | 对比学习 | 卓越的零样本分类能力、强大的视觉-语言对齐基础 | 作为TBM-VLM视觉编码器的基础，用于识别未见过的地质特征 | 1 |

---

## **第二部分 运营前沿：自主TBM掘进的普遍挑战**

本章将详细阐述现代TBM运营中普遍存在的“痛点”。其目的在于将后续章节中关于VLM解决方案的讨论，牢固地建立在真实世界中高风险的工程问题之上，以期与目标读者的直接经验产生共鸣。

### **2.1 “盲目推进”：当前感知与数据解译的局限**

#### **2.1.1 地质不确定性原则**

TBM运营的核心困境在于：设备对前方地质条件的变化极其敏感，然而其提前、精确感知这些变化的能力却非常有限 23。TBM的掘进过程常被形容为“盲掘”状态 22，即在很大程度上依赖于事后反应而非事前预知。

#### **2.1.2 现有“超前探测”技术的局限**

目前，行业内已应用多种“超前地质预报”技术，如隧道地震波探测（TSP），它能够探测到前方中长距离的异常地质体，但其分辨率和信噪比通常较低，难以提供精细的结构信息 24。其他地球物理方法，如地质雷达或瞬变电磁法（BEAM），同样存在探测范围与分辨率之间的权衡问题 25。这些技术的共同局限在于，它们提供的是一个相对模糊的、远距离的概览，而非TBM掘进决策最需要的、关于掌子面即时状况的高分辨率、精确图像。

#### **2.1.3 数据过载与洞察力不足的矛盾**

现代TBM装备了大量传感器，能够实时记录海量的运营数据，包括总推力、刀盘扭矩、转速、贯入度等 26。然而，这些数据与前方地质条件之间的关系是高度非线性的、复杂的。目前，对这些数据的解译仍然严重依赖于操作员的个人经验，或是基于传统机器学习模型（如人工神经网络ANN、支持向量机SVM）的辅助预测 23。这导致了一个普遍的矛盾：一方面是数据的海量过载，另一方面是可指导行动的深刻洞察力严重不足。数据采集与 actionable intelligence 之间存在巨大的鸿沟。

### **2.2 高风险挑战：地质灾害与设备故障**

#### **2.2.1 灾难性风险**

TBM掘进过程中面临的风险具有高后果性。在高地应力的软弱围岩中，围岩的挤压大变形可能导致“卡机”，即TBM被岩体卡住无法动弹。统计数据显示，卡机是TBM重大事故中占比最高的类型（占37%），一次严重的卡机事故可能导致数月的工期延误和巨大的经济损失 22。其他重大地质灾害还包括遭遇未预见的断层破碎带、掌子面失稳坍塌、突发性的大量涌水或突泥等 22。

#### **2.2.2 复合地层的挑战**

TBM在均质、中硬岩地层中掘进效率最高 22。然而，在煤矿等工程环境中，地层通常由软硬相间的岩层组成（如煤层、泥岩、砂岩），形成所谓的“软硬复合地层”。在这种地层中掘进，刀盘会承受极不均匀的载荷，导致掘进姿态难以控制，并可能对刀盘本身及关键的主轴承系统造成严重冲击和损坏 22。

#### **2.2.3 环境与后勤制约**

特定的施工环境，如煤矿，还带来了额外的挑战。高瓦斯（甲烷）环境要求所有设备必须满足严苛的防爆标准；深部巷道空间狭小，给TBM的井下组装、拆解、维护和物料运输带来了巨大困难；同时，有效的通风和粉尘抑制系统也是保障安全和正常运营的先决条件 22。这些因素使得任何形式的人工检查和干预都变得更加危险、耗时和昂贵。

### **2.3 渣料与刀具磨损分析的数据解译瓶颈**

#### **2.3.1 渣料分析作为代理指标**

一个重要的工程原理是：TBM掘进排出的渣料（muck），其尺寸、形状和体积等物理特性，是岩体与刀盘相互作用的直接产物，因此蕴含了关于掌子面岩体状况和破岩效率的丰富信息 33。例如，在完整的岩体中，产生尺寸较大、形状扁平的岩屑通常意味着较高的破岩效率 33。

#### **2.3.2 当前渣料分析的局限**

传统的渣料分析方法，如人工筛分，不仅耗时耗力，而且提供的是滞后的数据，无法用于实时决策 33。尽管已有研究尝试使用基础的机器视觉系统来拍摄输送带上的渣料图像，但由于现场光照条件差、粉尘干扰严重、颗粒相互重叠以及尺寸变化范围大等问题，要实现精确的图像分割和特征分析极为困难 21。

#### **2.3.3 刀具磨损监控**

刀具磨损是TBM运营中的一项主要成本，也是导致停机维护的主要原因之一 36。位于刀盘外周的刮刀（gauge cutters）由于运动路径最长，磨损速度最快 37。对刀具的检查，特别是在需要加压维持掌子面稳定的密闭式TBM（如土压平衡或泥水盾构）中，是一项高风险、高成本且必须中断掘进的操作 37。

#### **2.3.4 当前磨损监控的局限**

现有的在线磨损监控系统主要依赖间接测量。一些系统通过安装在刀具上的传感器来监测振动、温度和转速，从而推断磨损状态 38。另一些系统则采用更直接的测量方法，如涡流传感器、磁传感器或激光传感器来测量刀具边缘与传感器之间的距离变化 36。这些方法各有其局限性，例如传感器可能很脆弱、测量范围有限，或者只能提供一个单一的磨损量数值，而无法提供关于磨损类型的直观信息（例如，是均匀的磨耗还是灾难性的崩刃）。

在深入分析TBM运营的诸多挑战后，可以发现两个根本性的问题模式。

其一，**TBM的运营普遍陷入一个“被动反应的恶性循环”**。由于对前方即时地质情况的感知能力有限，TBM在很大程度上处于“盲目”掘进状态 22。因此，无论是自动控制系统还是人类操作员，其决策模式主要是对已经发生的变化做出反应——这些变化通过刀盘扭矩、推力或振动等参数的波动表现出来 28。这种被动的姿态意味着，当一个问题（如进入一个未预见的断层带）被充分识别时，TBM往往已经处于一个次优甚至危险的工况之中。其后果是严重的：掘进效率低下、刀具过度磨损，在最坏的情况下，甚至发生灾难性的卡机事故 22。现有的补救性措施，如分析已排出的渣料或停机进行掌子面检查 33，本质上是“滞后指标”，它们确认的是一个已经发生的问题。因此，行业的根本挑战在于打破这个反应式循环，转向一种

**主动预测的运营模式**。这迫切需要一种能够提供关于“即时未来”（掌子面状况）和“即时过去”（掘进产生的渣料）的高分辨率、实时、可解译信息的感知技术。这正是基于VLM的视觉分析技术所要填补的精确空白。

其二，**TBM自身是最佳但利用最不足的地质探针**。目前，大量的研发投入被用于各种“超前”地球物理技术（如地震波、电阻率法），以期预测TBM前方较远距离的地质情况 24。然而，关于地质状况最直接、保真度最高的信息，恰恰是在掘进的瞬间产生的。这些信息被编码在三个主要来源中：掌子面岩体的

**视觉外观** 42、被掘下渣料的

**物理特性** 34，以及机器与岩体相互作用时的

**实时动态响应**（如振动、扭矩）38。当前的问题在于，这些宝贵的数据流要么被孤立地分析，要么存在严重的时间延迟（如渣料分析），要么可见性极差（如掌子面检查），要么只能通过间接推断来解读（如扭矩值）。缺乏一个能够将它们进行整体融合的系统。因此，行业最大的未开发机遇，或许不在于寻找一种新的、神奇的远程传感器，而在于构建一个能够智能地融合和解译TBM

**已经产生**的丰富多模态数据的“认知引擎”。VLM，凭借其天生的、融合视觉与文本进行推理的能力，正是这个认知引擎的理想候选者。它有潜力同时观察掌子面图像、分析对应的渣料图像，并将这两者与该时刻的传感器数据流进行关联，从而形成一个前所未有的、全面的地质与工况理解。

---

## **第三部分 协同增效：利用VLM应用打造“会思考、会看见”的TBM**

本章是报告的核心，将第一部分介绍的VLM能力与第二部分详述的TBM挑战直接对接。每个小节都将深入探讨一个具体的、具有高价值的VLM应用场景，展示技术融合的巨大潜力。

### **3.1 应用场景一：实时地质智能与灾害探测**

#### **3.1.1 基于刀盘影像的岩体质量自动分类**

* **愿景**：将目前依赖人工通过狭窄人孔进行的有限、间断的掌子面检查 42，升级为连续、自动化的实时分析。通过在刀盘上安装坚固耐用的高分辨率摄像头（已有研究项目进行过原型测试 42），可以获得持续的掌子面视频流。  
* **VLM的角色**：一个经过专门训练的TBM-VLM将实时分析这些图像。其训练数据集将包含大量掌子面图像，并由地质专家标注对应的地质参数，如岩石质量指标（RQD）、岩体分级（RMR）、岩石类型、节理组数、蚀变程度等 29。  
* **超越分类的语义理解**：VLM的价值远不止于输出一个简单的分类结果，如“IV级围岩”。它能利用其强大的推理能力 7，提供一种描述性的、可解释的分析报告，例如：“在右上象限检测到高度破碎的岩体，符合IV级围岩特征。观察到擦痕面和涌水迹象，存在块状岩体掉落的高风险。” 这种能力源于VLM能够识别和描述传统计算机视觉难以语境化的细微地质特征 20。

#### **3.1.2 VLM驱动的异常检测与主动安全预警**

* **愿景**：将VLM部署为一个不知疲倦的“哨兵”，持续将实时的掌子面视觉信息与一个基于当前地质单元学习到的“正常”或“预期”状态模型进行比对。  
* **VLM的角色**：当VLM检测到显著的偏差——例如岩石纹理的突然变化、断层线的出现、意外的渗水，或掌子面失稳的早期迹象——它会立即触发警报。这充分利用了VLM在上下文感知异常检测方面的优势 8，这里的“上下文”即包括先前观察到的地质情况，也包括项目的整体地质模型。  
* **与其他传感器的融合**：VLM的视觉异常检测结果将与其他数据源进行交叉验证，以提高预警的准确性。例如，一个关于“可能存在空洞”的视觉警报，可以与TBM的振动数据（当刀具撞击空洞时会产生冲击信号 38）和刀盘扭矩的下降进行关联。这种多模态信息的融合（VLM的核心理念之一）能够提供高置信度的灾害预警，有效减少误报 50。

### **3.2 应用场景二：面向性能优化的智能渣料分析**

* **愿景**：用一个实时的、自动化的分析系统取代滞后的、人工的渣料分析，为掘进参数的优化提供一个持续的反馈闭环。一套摄像系统将被安装在TBM的主输送带上方 33。  
* **VLM的角色**：  
  * **分割与测量**：首先，VLM将对渣料图像执行先进的实例分割，即使在光照不佳、颗粒重叠的情况下，也能准确识别出单个岩屑。这是深度学习方法远超传统图像处理方法的领域 21。  
  * **特征表征**：对每一个被分割出的岩屑，VLM将提取其关键的形态学属性，包括尺寸分布（粒度）、形状（如扁平度、棱角性）以及颜色/纹理 35。  
  * **关联与推理**：最关键的一步是，VLM会将这些实时的视觉特征与该时刻TBM的运营参数（推力、转速、扭矩）进行关联。通过在历史数据上进行训练，VLM可以学习到复杂的岩石-机器相互作用规律，例如，“在这种花岗岩中，较低的推力会产生过细的渣料，这表明发生了低效的研磨而非有效的切削”，或者“大量棱角状大块岩屑的突然出现，与进入预期的节理发育带相符” 33。这为实现掘进性能的最大化，提供了一种直接的、数据驱动的调优依据。

### **3.3 应用场景三：VLM驱动的预测性维护与设备健康管理**

#### **3.3.1 刀盘刀具磨损与损伤的直接视觉评估**

* **愿景**：用连续、自动化的视觉监控，增强甚至替代间歇性的、高风险的人工刀具检查。摄像头将被部署在能够捕捉到刀具高清图像的位置，特别是在拼装管片等非掘进时段 37。  
* **VLM的角色**：一个专门的TBM-VLM将被训练来执行多项任务：  
  1. **磨损测量**：自动识别每个刀盘的切削刃，并与基线状态进行比较，从而直接测量其直径的减少量，即磨损量。这类似于激光或涡流传感器的目标，但提供了丰富的视觉上下文 36。  
  2. **损伤分类**：超越简单的磨损量测量，对损伤的**类型**进行分类。是均匀的磨料磨损，还是灾难性的崩刃，或是异常的偏磨？这是一个典型的细粒度视觉分类任务，非常适合VLM，其提供的信息远比单一的传感器数值丰富 36。  
  3. **逻辑异常检测**：识别“逻辑”层面的问题，例如一个刀具缺失或被岩石卡住，这将被VLM识别为与预期的刀盘配置不符的异常状态 10。

#### **3.3.2 融合视觉与传感数据的整体状态诊断**

* **愿景**：通过融合VLM的视觉分析与现有的传感器数据，为刀盘创建一个全面的“健康监控”系统。  
* **VLM的角色**：VLM在此扮演数据融合引擎的角色。例如，如果某个特定刀具的振动传感器显示出高频冲击信号 38，同时其温度传感器读数上升，VLM可以将这些传感器数据与其视觉流进行关联。如果视觉上没有发现明显的损伤，VLM可能会推断是轴承即将失效。反之，如果VLM在视觉上观察到该刀盘上有一个大的崩刃，它就可以确认传感器数据的异常是由冲击损伤引起的。这种将“传感器感觉到的”与“摄像头看到的”进行融合，能够提供比任何单一数据流都更准确、更具可操作性的诊断。

将VLM的能力应用于TBM的挑战，揭示了两个深层次的转变。

其一，**VLM扮演着不同数据模态之间的“翻译官”角色，从而创造出一个关于TBM状态的、统一且易于理解的叙事**。一台TBM会产生多种互不相通的数据流：视觉的（图像）、时间序列的（扭矩、转速）和事件驱动的（振动峰值）28。人类专家的工作，正是在大脑中将这些数据流进行融合。他们观察渣料，倾听机器的声音，感受振动，并查看控制面板上的仪表，最终形成一个整体的判断。VLM的核心能力之一就是实现视觉和语言之间的转换 1。我们可以将这个概念进一步扩展：VLM可以学习将传感器数据的“语言”也翻译到与视觉数据相同的语义空间中。因此，VLM的角色不仅仅是分析图像，更是创造一个统一的叙事。它能够生成一段自然语言摘要，综合所有可用的信息：“掌子面视觉分析显示，地质正向块状岩体过渡。刀盘振动增加15%和扭矩的无规律波动证实了这一点。相应的渣料显示出更高比例的大尺寸、棱角状岩屑。建议将转速降低10%以减小冲击载荷。” 在这个过程中，VLM成为了终极的数据融合引擎和“故事讲述者”。

其二，**VLM的应用将用于模型训练的“地面真实（ground truth）”来源，从依赖外部的、稀疏的实验室测试，转变为依赖内部的、连续的运营数据**。传统的TBM岩土模型和机器学习预测器，其训练的“真实标签”主要来源于钻孔取样和岩石样本的室内试验结果（如单轴抗压强度UCS、RMR分级）28。这些数据不仅稀疏，而且可能无法完全代表TBM实际遇到的地质情况。而一个基于VLM的系统，其训练方式截然不同。它通过关联它所

**看到**的（掌子面/渣料图像）与机器所**做**的（运营参数）以及最终所**发生**的（灾害、磨损、掘进速度）来进行学习。对于一张给定的掌子面图像，其“标签”不再仅仅是“IV级围岩”。它的标签变成了一个极其丰富的、并发的数据集合：{扭矩: X, 推力: Y, 振动: Z, 渣料尺寸分布: \[...\], 刀具磨损率: W, 灾害事件: 无}。这就创建了一个自我强化的、持续学习的系统。TBM自身的运营历史成为了其训练数据集，而模型的预测则由后续的实际掘进表现来验证。这种模式摆脱了对施工前地质模型的过度依赖，转向一个能够实时学习项目特定“岩-机”相互作用动态的系统。这是隧道岩土工程表征方法论上的一次根本性转变。

| 表2：TBM挑战与VLM解决方案映射 |  |  |  |  |  |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **TBM挑战** | **当前局限性** | **VLM解决方案** | **所需数据输入** | **核心收益** | **关键信息来源** |
| **地质灾害风险 (如卡机、塌方)** | 反应式应对，依赖推力突增等滞后指标 | 基于掌子面图像的实时视觉异常检测，融合振动数据进行高置信度预警 | 刀盘摄像头实时视频流、加速度计数据、扭矩/推力数据 | 主动预警，在灾害发生前提供调整参数的时间窗口，提升安全性 | 8 |
| **岩体质量快速变化导致效率低下** | 依赖稀疏的地勘资料和操作员经验，无法实时适应 | 掌子面图像自动岩体质量分类（RMR/RQD），并提供可解释的语义描述 | 刀盘摄像头视频流、标注的地质图像数据集 | 连续、客观的地质评估，为实时优化掘进参数提供依据 | 20 |
| **渣料分析滞后，无法指导掘进** | 人工筛分耗时耗力，传统图像分割在恶劣环境下效果差 | 基于输送带图像的实时渣料智能分析（分割、尺寸、形状），并与运营参数关联 | 输送带摄像头视频流、TBM运营参数（PLC数据） | 建立掘进参数与破岩效率的实时反馈闭环，实现性能最优化 | 21 |
| **刀具磨损检查危险、耗时、成本高** | 需停机进行人工检查，或依赖间接、信息量少的传感器数据 | 对刀具进行直接视觉评估，实现磨损量测量和损伤类型（崩刃、偏磨）分类 | 刀盘摄像头高清图像/视频（尤其在停机间隙） | 无需停机即可获得精确、丰富的刀具健康状况，实现预测性维护 | 36 |
| **多源数据孤立，操作员认知负荷大** | 操作员需同时解读数十个仪表和图表，难以形成整体判断 | VLM作为“认知引擎”，融合视觉、传感器和运营数据，通过自然语言界面提供综合性洞察 | 所有可用数据流（视觉、传感器、PLC） | 降低操作员认知负荷，将复杂数据转化为直观、可操作的智能建议 | 52 |

---

## **第四部分 VLM集成TBM的系统架构设计**

本章将从“是什么”和“为什么”转向“如何做”。它将概述将VLM驱动的TBM变为现实所需的实际工程和系统集成工作，旨在满足读者作为工程和实施决策者的思维模式。

### **4.1 感知系统：板载摄像头、照明与数据基础设施**

#### **4.1.1 硬件选型**

视觉传感器的选择是整个系统的基础。所有部署在TBM上的摄像头必须是工业级的坚固耐用型产品，能够抵抗强烈的冲击、振动、高湿度和粉尘侵蚀 42。

#### **4.1.2 摄像头布局与照明**

* **刀盘区域**：根据已有研究项目的经验，摄像头可以安装在刀盘的人孔或刀具座内 42。由于刀盘在掘进时高速旋转，为避免运动模糊并克服不稳定的环境光，必须配备与相机快门同步的脉冲式高强度LED照明系统 43。  
* **输送带区域**：摄像头应安装在主输送带的正上方，并配备稳定、均匀的照明光源，以确保获取的渣料图像质量一致，便于后续分析 33。

#### **4.1.3 数据传输**

从旋转的刀盘上可靠地传输高清视频数据是一个技术挑战。可行的方案包括使用高度耐用的高速有线连接（例如，7类工业以太网电缆通过滑环传输）43，或采用无线发射器。然而，在密闭式TBM的加压环境中，无线信号的可靠性可能会受到影响 39。无论采用何种方式，数据都需要被实时、稳定地传输到一个中央处理单元。

### **4.2 认知核心：模型部署与边缘计算策略**

#### **4.2.1 边缘计算的必要性**

对于需要即时响应的应用，如地质灾害预警，VLM的推理过程必须在“边缘端”——即在TBM本体上——完成。由于隧道内网络连接的延迟和不稳定性，依赖云端进行计算是不可行的 55。

#### **4.2.2 推理硬件**

这意味着TBM需要装备强大且坚固的边缘计算硬件，例如内置GPU的工业级服务器。这些服务器负责运行VLM模型并处理来自多个传感器的数据流。

#### **4.2.3 模型部署与持续学习**

部署在TBM上的TBM-VLM模型，其开发路径应为：首先在一个通用的、大规模的数据集上进行预训练，然后利用在特定TBM项目或地质条件下收集的专用数据集进行微调。为了在边缘硬件上高效运行，模型需要经过优化以加速推理，例如可以借鉴LLaVA-Mini等模型中使用的效率提升技术 16。

一个理想的架构应支持**持续学习闭环**：TBM在运营过程中不断收集新的、带有事实标签（如实际掘进效率、发生的事件）的数据。这些数据在网络连接可用时，被定期上传到云端服务器，用于重新训练和迭代优化VLM模型。更新后的模型再被推送回TBM的边缘计算机。这种模式可以实现“群体学习”，即整个TBM机队的经验都可以用来提升单个机器的智能水平。

### **4.3 操作员的副驾：TBM运营的自然语言界面**

#### **4.3.1 概念**

引入自然语言数据界面（Natural Language Data Interface, NLDI）作为人类操作员与VLM赋能的TBM系统交互的主要方式 52。操作员不再需要费力解读数十个原始的仪表盘和数据图表，而是可以用日常语言直接向系统提问。

#### **4.3.2 查询多模态数据库**

NLDI会将操作员的自然语言问题（例如，“显示刮刀行磨损最严重的三个刀具，并对比它们过去一小时的振动模式”）自动翻译成一个结构化的查询指令，该指令能够从VLM和其它传感器融合后的多模态数据库中检索信息并生成答案 53。

#### **4.3.3 收益**

这种交互方式极大地降低了操作的技术门槛，减轻了操作员的认知负荷，并使得数据探索和决策过程更加快速和直观 52。它有效地弥合了复杂的机器数据与人类直观理解之间的鸿沟，解决了当前系统中一个主要的瓶颈问题 53。

### **4.4 自动化智能：基于多模态数据生成每日进度报告**

#### **4.4.1 愿景**

将繁琐但至关重要的TBM每日掘进报告的编写工作完全自动化。

#### **4.4.2 VLM的角色**

VLM系统将扮演一个自动报告生成代理的角色。在每个班次或工作日结束时，它会自动综合处理过的所有数据，生成一份全面的报告：

* **视觉摘要**：系统会自动挑选出具有代表性的掌子面和渣料图像，并为其生成描述性标题（例如，“第543环掌子面状况”，“风化花岗岩地层典型渣料”）2。  
* **地质日志**：基于其对掌子面的实时分类结果，生成一段关于当天所遇地质条件的文本摘要（例如，“今日掘进15米，穿越IV级砂岩，随后遭遇一个3米宽的断层破碎带，伴有明显涌水”）。  
* **性能指标**：整合来自TBM可编程逻辑控制器（PLC）的量化数据，如掘进速度、设备利用率、平均扭矩等。  
* 维护建议：报告检测到的刀具磨损情况，并为维护团队标记出任何异常（例如，“G5号刀具出现加速磨损，建议进行目视检查”）。  
  这个过程与VLM在医疗领域的应用——从多视角医学影像自动生成诊断报告——有着直接的类比，该领域的技术已证明是成功和有效的 59。

VLM-TBM的系统架构设计，必然要求TBM数据基础设施发生根本性的转变。它必须从当前各系统**孤立记录数据的模式，演变为一个统一的、高带宽的、实时的“神经网络系统”**。目前的TBM数据系统通常是孤岛式的：PLC记录运营参数，一个独立的系统可能记录振动数据，而视觉数据（如果采集的话）往往缺乏系统性 28。然而，本报告中提出的所有VLM应用（地质、渣料、磨损分析）都依赖于高分辨率、高帧率的视频流 34。这所产生的数据量比传统的传感器日志要高出几个数量级。更重要的是，为了实现实时推理和数据融合，所有数据流必须被精确地时间同步，并能被一个中央处理单元低延迟地访问。因此，集成VLM不仅仅是一个软件升级，它要求对TBM的内部数据架构进行彻底的重新设计。这需要一个高带宽、高鲁棒性的内部网络，大量的板载存储空间，以及强大的边缘计算能力。TBM必须从一个由独立子系统组成的集合体，被重新设计为一个单一的、高度集成的网络物理系统（Cyber-Physical System），其复杂性和集成度将向现代自动驾驶汽车看齐。

| 表3：VLM集成TBM系统架构组件 |  |  |  |  |
| :---- | :---- | :---- | :---- | :---- |
| **子系统** | **组件** | **规格/要求** | **在系统中的用途** | **关键研究依据** |
| **I. 感知系统** | 坚固型工业相机 | 高分辨率（≥2MP），GigE Vision接口，IP67防护等级，抗振动/冲击 | 采集刀盘掌子面、输送带渣料、刀具磨损的高清图像/视频 | 42 |
|  | 高强度LED照明系统 | 脉冲式，\>10,000流明，与相机快门同步 | 在高速旋转和恶劣光照条件下提供无运动模糊的清晰成像 | 43 |
|  | 多模态传感器 | 加速度计、温度传感器、PLC数据接口等 | 提供与视觉数据融合的振动、温度、扭矩、推力等信息 | 38 |
| **II. 板载数据网络** | 高速工业以太网 | ≥1Gbps，7类屏蔽电缆，通过高性能滑环传输 | 保证从旋转刀盘到处理单元的高带宽、低延迟视频数据传输 | 43 |
|  | 时间同步协议 | PTP (Precision Time Protocol) 或同等技术 | 确保所有视觉和传感器数据流具有统一的时间戳，是数据融合的基础 | \- |
| **III. 边缘计算核心** | 工业级边缘服务器 | 内置高性能GPU（如NVIDIA Jetson AGX或更高级别），宽温工作范围，抗振设计 | 在TBM上实时运行VLM推理模型，执行地质分类、异常检测等任务 | 16 |
| **IV. 软件与模型** | 定制化TBM-VLM | 基于开源模型（如LLaVA）微调，针对TBM数据进行优化 | 系统的“大脑”，负责解译所有多模态数据，生成洞察 | 6 |
|  | 数据融合与管理平台 | 负责收集、存储、同步和管理所有数据流 | 为VLM和NLI提供统一的数据访问接口 | \- |
|  | 持续学习框架 | 负责将板载数据上传至云端进行模型再训练，并将更新后的模型部署回边缘端 | 实现模型的持续迭代和性能提升 | \- |
| **V. 人机界面** | 自然语言界面 (NLI) | 基于Web的交互式仪表盘，支持自然语言输入和多模态输出 | 允许操作员通过对话方式查询TBM状态、接收智能警报和建议 | 52 |
|  | 自动报告生成模块 | \- | 每日自动汇总所有分析结果，生成结构化的掘进报告 | 59 |

---

## **第五部分 战略路线图与结论**

本章将提供一个前瞻性的战略视角，勾勒出一条通向实际应用的路径，并探讨关键的风险与挑战，最后以对未来隧道工程的展望作为结尾。

### **5.1 分阶段实施计划：从试点项目到机队范围的集成**

实现VLM与TBM的深度融合是一项复杂的系统工程，应采用循序渐进的策略。建议的实施路线图分为五个阶段：

* **第一阶段：数据采集与可行性研究（第一年）**：此阶段的核心目标是构建一个高质量、领域专用的多模态数据集。选择一台TBM作为试点，为其安装所需的摄像头和数据记录系统。系统地记录来自刀盘和输送带的视频，并与所有PLC和传感器数据进行精确的时间同步。同时，组织地质和设备工程师团队对采集到的数据进行细致的人工标注（例如，标注地质特征、岩体等级、渣料类型、刀具磨损状况等）。  
* **第二阶段：离线模型开发与验证（第二年）**：利用第一阶段收集的数据集，对一个基础VLM（如一个开源的LLaVA变体）进行微调。在离线环境中开发和测试用于地质分类、渣料分析和磨损检测的核心算法模型。将模型的预测结果与人工标注的“地面真实”进行严格比对，以验证其准确性和可靠性。  
* **第三阶段：板载试点部署（影子模式）（第三年）**：将经过验证的模型部署到一台正在作业的TBM的边缘计算机上，以“影子模式”运行。在此模式下，系统会进行实时的预测和分析，但其输出结果仅供观察和记录，并不会直接干预机器的控制。其目的是将模型的预测与人类操作员的决策及实际掘进结果进行对比，从而进一步优化模型，并逐步建立操作人员对系统性能的信任。  
* **第四阶段：交互式副驾驶集成（第四年）**：正式启用自然语言界面。此时，VLM系统开始扮演一个主动的“智能副驾驶”角色，能够回答操作员的提问，并提供主动的警报和操作建议。人类操作员仍然拥有完全的控制权，但其决策过程得到了人工智能的有力增强。  
* **第五阶段：闭环控制与群体学习（第五年及以后）**：对于一些已被充分理解和验证的任务（例如，根据渣料分析结果自动优化推力），可以授权系统进行有限的闭环控制。同时，将该系统扩展到整个TBM机队，所有机器都将运营数据贡献到一个中央云数据库，用于模型的持续改进，实现“群体智能”的进化。这一愿景与“TBM自动驾驶”和“智能掘进”的行业目标相契合 23。

### **5.2 应对关键障碍：数据稀缺、模型鲁棒性与安全保证**

尽管前景广阔，但实现VLM驱动的TBM仍需克服几个重大障碍：

* **数据瓶颈**：最大的初始障碍是缺乏大规模、公开可用的、经过良好标注的TBM运营数据集 55。因此，第一阶段的数据采集工作至关重要且需要大量资源投入。  
* **模型的鲁棒性与“幻觉”问题**：VLM与所有大型语言模型一样，存在产生“幻觉”（即生成看似合理但实际上是错误的信息）的风险 10。这要求在模型开发和部署过程中进行极其严格的测试和验证。系统设计时必须能够表达其预测的不确定性。采用多模态数据融合（例如，用传感器数据交叉验证视觉分析结果）是缓解这一问题的关键策略。  
* **安全保证与认证**：将一个基于人工智能的系统应用于高风险、任务关键型的工业环境中，其安全认证将是一个巨大的挑战。这需要与监管机构和安全评估机构紧密合作，共同探索和建立全新的保证流程和标准 55。本报告提出的分阶段、人机协同的实施路径，正是为了在实践中逐步构建起必要的安全案例。

### **5.3 未来展望：迈向全自主、自驱动的隧道掘进系统**

视觉语言模型的集成，是通向真正意义上自主TBM的关键赋能技术。最终的愿景是打造一台“自驱动”的TBM——它能够像经验丰富的工程师一样，实时地感知和解译其所处的复杂环境，智能地做出决策，并动态调整其掘进策略。

这一技术飞跃不仅将极大地提升隧道工程的效率和安全性，更有可能从根本上改变地下工程的经济性，使那些目前因技术难度或风险过高而被认为不可行的项目成为可能。它标志着隧道工程正在从一个传统的重工业过程，向一门由数据驱动、以智能为核心的高科技科学演进。

#### **Works cited**

1. 什么是视觉语言模型(VLM)？| IBM, accessed July 26, 2025, [https://www.ibm.com/cn-zh/think/topics/vision-language-models](https://www.ibm.com/cn-zh/think/topics/vision-language-models)  
2. 视觉语言模型解析 \- Ultralytics, accessed July 26, 2025, [https://www.ultralytics.com/zh/blog/understanding-vision-language-models-and-their-applications](https://www.ultralytics.com/zh/blog/understanding-vision-language-models-and-their-applications)  
3. GPT-4 Vision Explained: Overview, Applications, and Use Cases \- Upcore Technologies, accessed July 26, 2025, [https://www.upcoretech.com/insights/gpt-4-vision-explained-applications-use-cases/](https://www.upcoretech.com/insights/gpt-4-vision-explained-applications-use-cases/)  
4. What's new in GPT-4: Architecture and Capabilities | Medium, accessed July 26, 2025, [https://medium.com/@amol-wagh/whats-new-in-gpt-4-an-overview-of-the-gpt-4-architecture-and-capabilities-of-next-generation-ai-900c445d5ffe](https://medium.com/@amol-wagh/whats-new-in-gpt-4-an-overview-of-the-gpt-4-architecture-and-capabilities-of-next-generation-ai-900c445d5ffe)  
5. GPT-4 Vision's Impact on Visual Understanding and Text Interaction \- Reveation Labs, accessed July 26, 2025, [https://www.reveation.io/blog/gpt4-vision-impact](https://www.reveation.io/blog/gpt4-vision-impact)  
6. Multimodal GPT-4 and LLaVA integration of advanced image ..., accessed July 26, 2025, [https://ai-scholar.tech/zh/articles/computer-vision/LLaVA](https://ai-scholar.tech/zh/articles/computer-vision/LLaVA)  
7. LLaVA-o1: Let Vision Language Models Reason Step-by-Step \- arXiv, accessed July 26, 2025, [https://arxiv.org/html/2411.10440v1](https://arxiv.org/html/2411.10440v1)  
8. A VLM-based Method for Visual Anomaly Detection in Robotic Scientific Laboratories \- arXiv, accessed July 26, 2025, [https://arxiv.org/html/2506.05405v1/](https://arxiv.org/html/2506.05405v1/)  
9. A VLM-based Method for Visual Anomaly Detection in Robotic Scientific Laboratories, accessed July 26, 2025, [https://www.researchgate.net/publication/392515240\_A\_VLM-based\_Method\_for\_Visual\_Anomaly\_Detection\_in\_Robotic\_Scientific\_Laboratories](https://www.researchgate.net/publication/392515240_A_VLM-based_Method_for_Visual_Anomaly_Detection_in_Robotic_Scientific_Laboratories)  
10. LogicAD: Explainable Anomaly Detection via VLM-based Text Feature Extraction \- arXiv, accessed July 26, 2025, [https://arxiv.org/html/2501.01767v1](https://arxiv.org/html/2501.01767v1)  
11. GPT-4 Vision: A Comprehensive Guide for Beginners \- DataCamp, accessed July 26, 2025, [https://www.datacamp.com/tutorial/gpt-4-vision-comprehensive-guide](https://www.datacamp.com/tutorial/gpt-4-vision-comprehensive-guide)  
12. GPT-4 Vision Deep Dive. Get started with GPT-4 Vision on Azure \- Medium, accessed July 26, 2025, [https://medium.com/@o.anonthanasap/gpt-4-vision-deep-dive-c5de36689fa0](https://medium.com/@o.anonthanasap/gpt-4-vision-deep-dive-c5de36689fa0)  
13. How to use vision-enabled chat models \- Azure OpenAI in Azure AI Foundry Models | Microsoft Learn, accessed July 26, 2025, [https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/gpt-with-vision](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/gpt-with-vision)  
14. \[2304.08485\] Visual Instruction Tuning \- arXiv, accessed July 26, 2025, [https://arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)  
15. LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images \- arXiv, accessed July 26, 2025, [https://arxiv.org/abs/2403.11703](https://arxiv.org/abs/2403.11703)  
16. \[2501.03895\] LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token \- arXiv, accessed July 26, 2025, [https://arxiv.org/abs/2501.03895](https://arxiv.org/abs/2501.03895)  
17. \[2501.08282\] LLaVA-ST: A Multimodal Large Language Model for Fine-Grained Spatial-Temporal Understanding \- arXiv, accessed July 26, 2025, [https://arxiv.org/abs/2501.08282](https://arxiv.org/abs/2501.08282)  
18. \[2407.19185\] LLaVA-Read: Enhancing Reading Ability of Multimodal Language Models, accessed July 26, 2025, [https://arxiv.org/abs/2407.19185](https://arxiv.org/abs/2407.19185)  
19. Urban Road Anomaly Monitoring Using Vision–Language Models for Enhanced Safety Management \- MDPI, accessed July 26, 2025, [https://www.mdpi.com/2076-3417/15/5/2517](https://www.mdpi.com/2076-3417/15/5/2517)  
20. (PDF) Automated rock mass condition assessment during TBM tunnel excavation using deep learning \- ResearchGate, accessed July 26, 2025, [https://www.researchgate.net/publication/358282199\_Automated\_rock\_mass\_condition\_assessment\_during\_TBM\_tunnel\_excavation\_using\_deep\_learning](https://www.researchgate.net/publication/358282199_Automated_rock_mass_condition_assessment_during_TBM_tunnel_excavation_using_deep_learning)  
21. Automatic segmentation of TBM muck images via a deep-learning approach to estimate the size and shape of rock chips | Request PDF \- ResearchGate, accessed July 26, 2025, [https://www.researchgate.net/publication/350528411\_Automatic\_segmentation\_of\_TBM\_muck\_images\_via\_a\_deep-learning\_approach\_to\_estimate\_the\_size\_and\_shape\_of\_rock\_chips](https://www.researchgate.net/publication/350528411_Automatic_segmentation_of_TBM_muck_images_via_a_deep-learning_approach_to_estimate_the_size_and_shape_of_rock_chips)  
22. TBM在煤矿巷道掘进中的技术应用和研究进展 \- 煤炭科学技术, accessed July 26, 2025, [https://www.mtkxjs.com.cn/cn/article/pdf/preview/10.13199/j.cnki.cst.2022-2253.pdf](https://www.mtkxjs.com.cn/cn/article/pdf/preview/10.13199/j.cnki.cst.2022-2253.pdf)  
23. Intelligent tunnelling robot system for deep-buried long tunnels \- Frontiers, accessed July 26, 2025, [https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1135948/full](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1135948/full)  
24. Tunnel Seismic Detection for Tunnel Boring Machine by Joint Active and Passive Source Method and Imaging Advanced Prediction | Lithosphere | GeoScienceWorld, accessed July 26, 2025, [https://pubs.geoscienceworld.org/gsw/lithosphere/article/2024/4/lithosphere\_2024\_170/650048/Tunnel-Seismic-Detection-for-Tunnel-Boring-Machine](https://pubs.geoscienceworld.org/gsw/lithosphere/article/2024/4/lithosphere_2024_170/650048/Tunnel-Seismic-Detection-for-Tunnel-Boring-Machine)  
25. NAT 2012 1 Real-Time Tunnel Boring Machine Monitoring \- Colorado School of Mines, accessed July 26, 2025, [https://www.mines.edu/underground/wp-content/uploads/sites/183/2018/07/RealTimeTBMMonitoring.pdf](https://www.mines.edu/underground/wp-content/uploads/sites/183/2018/07/RealTimeTBMMonitoring.pdf)  
26. 关于硬岩隧道掘进机刀盘设计的进一步研究, accessed July 26, 2025, [https://www.engineering.org.cn/engi/CN/10.1016/j.eng.2017.12.009](https://www.engineering.org.cn/engi/CN/10.1016/j.eng.2017.12.009)  
27. \[1809.06688\] Geology prediction based on operation data of TBM: comparison between deep neural network and statistical learning methods \- arXiv, accessed July 26, 2025, [https://arxiv.org/abs/1809.06688](https://arxiv.org/abs/1809.06688)  
28. Collection of TBM Operating Data and Geological Information \- ResearchGate, accessed July 26, 2025, [https://www.researchgate.net/figure/Collection-of-TBM-Operating-Data-and-Geological-Information\_fig1\_333376015](https://www.researchgate.net/figure/Collection-of-TBM-Operating-Data-and-Geological-Information_fig1_333376015)  
29. Optimized Random Forest Models for Rock Mass Classification in Tunnel Construction, accessed July 26, 2025, [https://www.mdpi.com/2076-3263/15/2/47](https://www.mdpi.com/2076-3263/15/2/47)  
30. Data-driven real-time advanced geological prediction in tunnel construction using a hybrid deep learning approach | Request PDF \- ResearchGate, accessed July 26, 2025, [https://www.researchgate.net/publication/365764686\_Data-driven\_real-time\_advanced\_geological\_prediction\_in\_tunnel\_construction\_using\_a\_hybrid\_deep\_learning\_approach](https://www.researchgate.net/publication/365764686_Data-driven_real-time_advanced_geological_prediction_in_tunnel_construction_using_a_hybrid_deep_learning_approach)  
31. 全斷面隧道鑽掘工法(TBM) 類號： SDS-P-046, accessed July 26, 2025, [https://www.ilosh.gov.tw/media/kren2giz/f1402386147133.pdf](https://www.ilosh.gov.tw/media/kren2giz/f1402386147133.pdf)  
32. 煤矿岩巷TBM 适应性与新技术发展, accessed July 26, 2025, [http://www.mtkxjs.com.cn/cn/article/pdf/preview/10.13199/j.cnki.cst.2022-1404.pdf](http://www.mtkxjs.com.cn/cn/article/pdf/preview/10.13199/j.cnki.cst.2022-1404.pdf)  
33. Development of a real-time muck analysis system for assistant intelligence TBM tunnelling \- Sci-Hub, accessed July 26, 2025, [https://sci-hub.se/downloads/2020-11-28/03/gong2021.pdf](https://sci-hub.se/downloads/2020-11-28/03/gong2021.pdf)  
34. Development of a real-time muck analysis system for assistant intelligence TBM tunnelling, accessed July 26, 2025, [https://www.researchgate.net/publication/346358300\_Development\_of\_a\_real-time\_muck\_analysis\_system\_for\_assistant\_intelligence\_TBM\_tunnelling](https://www.researchgate.net/publication/346358300_Development_of_a_real-time_muck_analysis_system_for_assistant_intelligence_TBM_tunnelling)  
35. Evaluation of rock muck using image analysis and its application in the TBM tunneling, accessed July 26, 2025, [https://www.researchgate.net/publication/351367988\_Evaluation\_of\_rock\_muck\_using\_image\_analysis\_and\_its\_application\_in\_the\_TBM\_tunneling](https://www.researchgate.net/publication/351367988_Evaluation_of_rock_muck_using_image_analysis_and_its_application_in_the_TBM_tunneling)  
36. A New Strategy for Disc Cutter Wear Status Perception Using Vibration Detection and Machine Learning, accessed July 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9459918/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9459918/)  
37. Periodic inspection of gauge cutter wear on EPB TBMs using cone penetration testing \- Colorado School of Mines, accessed July 26, 2025, [https://www.mines.edu/underground/wp-content/uploads/sites/183/2018/07/GaugeCutterInspection\_CPT\_2013.pdf](https://www.mines.edu/underground/wp-content/uploads/sites/183/2018/07/GaugeCutterInspection_CPT_2013.pdf)  
38. Cutter Instrumentation System for Tunnel Boring Machines \- The Robbins Company, accessed July 26, 2025, [https://www.robbinstbm.com/wp-content/uploads/2010/09/CutterMonitoring\_NAT\_2010.pdf](https://www.robbinstbm.com/wp-content/uploads/2010/09/CutterMonitoring_NAT_2010.pdf)  
39. Cutter Monitoring \- Robbins, accessed July 26, 2025, [https://www.robbinstbm.com/about/advancements/cutter-monitoring/](https://www.robbinstbm.com/about/advancements/cutter-monitoring/)  
40. Study on the cutter wear based on the cutterhead working status monitoring system in TBM tunneling \- ResearchGate, accessed July 26, 2025, [https://www.researchgate.net/publication/370547817\_Study\_on\_the\_cutter\_wear\_based\_on\_the\_cutterhead\_working\_status\_monitoring\_system\_in\_TBM\_tunneling](https://www.researchgate.net/publication/370547817_Study_on_the_cutter_wear_based_on_the_cutterhead_working_status_monitoring_system_in_TBM_tunneling)  
41. Detection for Disc Cutter Wear of TBM Using Magnetic Force \- MDPI, accessed July 26, 2025, [https://www.mdpi.com/2075-1702/11/3/388](https://www.mdpi.com/2075-1702/11/3/388)  
42. Photographic documentation of the face of a TBM-driven tunnel \- ResearchGate, accessed July 26, 2025, [https://www.researchgate.net/publication/300376224\_Photographic\_documentation\_of\_the\_face\_of\_a\_TBM-driven\_tunnel](https://www.researchgate.net/publication/300376224_Photographic_documentation_of_the_face_of_a_TBM-driven_tunnel)  
43. Brenner Base Tunnel machines upgraded with cameras for geological mapping, accessed July 26, 2025, [https://www.imveurope.com/news/brenner-base-tunnel-machines-upgraded-cameras-geological-mapping](https://www.imveurope.com/news/brenner-base-tunnel-machines-upgraded-cameras-geological-mapping)  
44. Tunnel face rock mass class rapid identification based on TBM cutterhead vibration monitoring and deep learning model \- PubMed, accessed July 26, 2025, [https://pubmed.ncbi.nlm.nih.gov/40186002/](https://pubmed.ncbi.nlm.nih.gov/40186002/)  
45. Photographic Documentation of the face of a TBM-Driven Tunnel | ISRM EUROCK, accessed July 26, 2025, [https://onepetro.org/ISRMEUROCK/proceedings/EUROCK14/All-EUROCK14/ISRM-EUROCK-2014-150/40155](https://onepetro.org/ISRMEUROCK/proceedings/EUROCK14/All-EUROCK14/ISRM-EUROCK-2014-150/40155)  
46. Engineering Classification of Rock Masses for the Design of Tunnel Support \- ResearchGate, accessed July 26, 2025, [https://www.researchgate.net/publication/226039636\_Engineering\_Classification\_of\_Rock\_Masses\_for\_the\_Design\_of\_Tunnel\_Support](https://www.researchgate.net/publication/226039636_Engineering_Classification_of_Rock_Masses_for_the_Design_of_Tunnel_Support)  
47. Rock mass classification \- Rocscience, accessed July 26, 2025, [https://www.rocscience.com/assets/resources/learning/hoek/Practical-Rock-Engineering-Chapter-3-Rock-Mass-Classification.pdf](https://www.rocscience.com/assets/resources/learning/hoek/Practical-Rock-Engineering-Chapter-3-Rock-Mass-Classification.pdf)  
48. Classification and Prediction of Rock Mass Boreability Based on Daily Advancement during TBM Tunneling \- MDPI, accessed July 26, 2025, [https://www.mdpi.com/2075-5309/14/7/1893](https://www.mdpi.com/2075-5309/14/7/1893)  
49. MONITORING THE VIBRATION RESPONSE OF A TUNNEL BORING MACHINE: APPLICATION TO REAL TIME BOULDER DETECTION \- Colorado School of Mines, accessed July 26, 2025, [https://www.mines.edu/underground/wp-content/uploads/sites/183/2018/07/buckley\_thesis\_final.pdf](https://www.mines.edu/underground/wp-content/uploads/sites/183/2018/07/buckley_thesis_final.pdf)  
50. Exploration of Anomaly detection through CCTV Cameras: Computer Vision \- CS229, accessed July 26, 2025, [https://cs229.stanford.edu/proj2019aut/data/assignment\_308832\_raw/26608523.pdf](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26608523.pdf)  
51. Characterization of TBM Muck for Construction Applications \- MDPI, accessed July 26, 2025, [https://www.mdpi.com/2076-3417/11/18/8623](https://www.mdpi.com/2076-3417/11/18/8623)  
52. Natural Language Data Interfaces: Benefits, Use Cases & How to Get Started \- Alation, accessed July 26, 2025, [https://www.alation.com/blog/natural-language-data-interfaces-guide/](https://www.alation.com/blog/natural-language-data-interfaces-guide/)  
53. Constructing an Interactive Natural Language Interface for Relational Databases∗ \- VLDB Endowment, accessed July 26, 2025, [https://www.vldb.org/pvldb/vol8/p73-li.pdf](https://www.vldb.org/pvldb/vol8/p73-li.pdf)  
54. Remote Disc Cutter Monitoring in Tunnelling, accessed July 26, 2025, [https://www.tunnel-online.info/en/artikel/tunnel\_2011-08\_Remote\_Disc\_Cutter\_Monitoring\_in\_Tunnelling-1300454.html](https://www.tunnel-online.info/en/artikel/tunnel_2011-08_Remote_Disc_Cutter_Monitoring_in_Tunnelling-1300454.html)  
55. Future of tunnelling: high level review of emerging technologies (annex) \- GOV.UK, accessed July 26, 2025, [https://www.gov.uk/government/publications/future-of-the-subsurface-report/future-of-tunnelling-high-level-review-of-emerging-technologies-annex](https://www.gov.uk/government/publications/future-of-the-subsurface-report/future-of-tunnelling-high-level-review-of-emerging-technologies-annex)  
56. Natural Language Interfaces for Tabular Data Querying and Visualization: A Survey \- arXiv, accessed July 26, 2025, [https://arxiv.org/html/2310.17894v3](https://arxiv.org/html/2310.17894v3)  
57. Natural Language Interfaces for Structured Query Generation in IoD Platforms \- MDPI, accessed July 26, 2025, [https://www.mdpi.com/2504-446X/9/6/444](https://www.mdpi.com/2504-446X/9/6/444)  
58. NLI4DB: A Systematic Review of Natural Language Interfaces for Databases \- arXiv, accessed July 26, 2025, [https://arxiv.org/abs/2503.02435](https://arxiv.org/abs/2503.02435)  
59. Automatic Medical Report Generation Based on Cross-View Attention and Visual-Semantic Long Short Term Memorys, accessed July 26, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10451690/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10451690/)  
60. 渝城论“道”，肇启新篇向未来——地下洞室多场景TBM技术创新应用研讨会侧记, accessed July 26, 2025, [https://www.powerchina.cn/col7450/art/2024/art\_7450\_2023414.html](https://www.powerchina.cn/col7450/art/2024/art_7450_2023414.html)