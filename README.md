# SAIL

SAIL-Lab统一代码库

## Motivation

创建这个项目的动机最早来源于实验室组内成员相互Debug代码的时候遇到的麻烦。由于整个小组内所做的课题非常类似，而借鉴的代码都是来自不同的人写的，尽管大部分的时候都使用Pytorch，但是每个人写的特征计算流程、数据预处理流程、模型训练流程、模型推理流程、性能评估流程千差万别，这非常不利于组内的科研交流，因此萌生了写一套完整的、统一的代码库，同时也便于实验室后续的同学继续研究相关课题。

除此之外，还有下列的一些原因帮助推动该代码库持续不断的完善和更新。

1. 实验室内部对于特征计算以及模型性能评估的理解有所差异甚至出现错误，并且延续了多个年级的研究生。
2. 当希望尝试加入新的计算特征时发现要对原始代码进行大规模的改动。
3. 组内所研究的药物分子、晶体分子以及蛋白质分子属于规模复杂程度不同的图，但是很多对于药物分子有效的工程技巧未必适用于另外两者。
4. 无法避免的batch size=1的问题。

该项目作为我的硕士毕业论文的展望与延伸部分，会持续进行更新，并且在实验室允许的范围内进行开源。

## TODO

下面展示的模型代码复现主要方向为SAIL-Lab药物分子、晶体分子以及蛋白质分子相关研究的模型。

### CMPNN
- [ ] 根据[Moleculenet](https://moleculenet.org/datasets-1)中收集的数据集，增加Quantum Mechanics系列和Biophysics系列中提到的数据集结果。
- [ ] 根据[OGB](https://ogb.stanford.edu/)中收集的数据集，增加结果。

### CoMPT
- [ ] 代码进度0%

### CrystalNet
- [ ] 检查重现代码的性能与原始实现性能相差较大的问题。
- [ ] 检查由于GRU带来的晶胞图与超晶胞图性能不一致问题。

### DeepANIS
- [ ] 代码进度0%

### GraphPPIS
- [x] 增加性能结果日志。
- [ ] 调参研究性能差距。

### GraphSite
- [ ] 代码进度60%

### GraphSol
- [ ] 增加S.cerevisiae数据集的结果。
- [ ] 增加DeepSol数据集的结果。

### SPROF
- [ ] 代码进度95%
- [ ] 加速Paper中超大模型的运行时间，减少消耗的显存。

## Thank you
如果你觉得本项目对你的科研进展有所帮助的话，请给本项目一个star，谢谢！

