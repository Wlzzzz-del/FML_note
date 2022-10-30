# 研究目标
异质异构联邦元学习，主要涉及学习过程的*收敛性*、*隐私性*
# FML面临的挑战
+ 参与设备的数量可能有成千上万台，一个低效的设备选择算法会导致全局模型收敛速度降低。
(12-14)有几篇文章提出了联邦学习收敛的设想，但不能直接应用于FML。
+ 面对真实情况，FML与设备的一些时间有强关联，比如计算时间、通信时间等。
+ 由于FML在本地训练中涉及高阶信息（可能是梯度）和有偏差的随机梯度下降，因此现有的FL加速方法很难应用在FML中。
# FML的数学分析
+ “A collaborative learning framework via federated meta-learning,”
+ “Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach,”
+ “Inexact-ADMM based federated meta-learning for fast and continual edge learning,”
# 设备选择研究paper
## Client selection for federated learning with heterogeneous resources in mobile edge

## Fast-convergent federated learning

## A joint learning and communications framework for federated learning over wireless networks

## J. Ren, G. Yu, and G. Ding Accelerating DNN training in wireless federated edge learning systems
引入batchsize选择设备的方法来加速联邦学习过程
## [29]Adaptive Federated Learning in Resource Constrained Edge Computing Systems
0. 作者提出一种控制算法，在给定一个已知的资源预算的情况下，利用该控制算法在本地更新和全局参数聚合之间作出最优权衡(就是在每轮算法中实时调整tao的值),该权衡的目的是最小化损失函数。
1. tao是指两次全局之间的本地迭代轮次

## EFFICIENT FEDERATED META-LEARNING OVER MULTI-ACCESS WIRELESS NETWORKS
0. 作者首先通过FML的几个假设，证明了通过三个批次数据来对损失函数的一阶、二阶梯度进行估计得到的元函数梯度估计是一个有偏估计。
1. 作者提出一种非随机的设备选择算法NUFM，来提高原始联邦元学习的收敛效率，在设备选择算法中每个设备需要计算贡献度，把贡献度传回server进行排序，选择贡献度大的设备参与计算，设备贡献度的计算涉及某些未知参数的估计，作者参考文章[29]在训练过程中对参数进行估计。
2. 基于NUFM提出一种资源分配策略URAL，在真实无线网络场景下联合地优化收敛速度、设备钟点时间、能量消耗。
3. 更详细的，作者量化了每个设备对于全局模型收敛的贡献，通过*得出在一轮中全局损失减少的下界*。
4. 老板更关注异步性，比如***贡献度的消息如果异步到达，会如何处理***？

### 想法
假设有几个贡献度一直很大的client，总是参与了参与训练的过程，那么全局模型就朝着该些client期望的方向前进，而较低贡献度的模型没有得到想要的优化效果，这似乎不符合FL的初衷(所以是否需要引入概率？--drop out.)

# 想法
+ 异质异构是指每个client的模型层数、参数数量可能不同，有没有办法找到一种能够衡量模型的方法（包括层数、参数数量）
