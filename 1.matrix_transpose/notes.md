## 1.题目：Bounding the Effect of Partition Camping in GPU kernels

## 2.做的是什么:
本文聚焦在NVIDIA GPU的一个重要但是常被忽略的问题: Partition Camping。Partition 是在可用的memory partitions的子集的内存访问导致的，这一问题可以降GPU kernel performance降低七倍。
### Execution configuration of a kernel
threads排列在grid的block中，cuda thread 调度器会在SMs中调度blocks。当在SM执行一个block时，cuda会将block分成32一组的warps,warps里所有thread执行同一个instruction。cuda 在SMs里分批次调度blocks，但是由于register和shared memory的限制，不是同时全部调度。在当前scheduled batch里的blocks叫做active blocks（or warps）per SM。CUDA thread scheduler 将一个SM里的active blocks 作为一个统一的活跃的warps的集合等待执行。这样，cuda通过调度另一个活跃的warp来隐藏当前warps的读取时间。任意数量的blocks的performance会被其set of active blocks（active warps）的performance。在本文中，选择active warps per SM 来描述kernel的执行配置。
### Global memory transactions
一个half-warp的所有threads的global memory 的访存请求可以合并(coalesced)为尽量少的内存指令。根据数据类型的需要，指令可以是32-byte、64-byte、128-byte。指令size会被reduce。eg：如果一个half-warp里面的所有threads访问4-byte的words，这个transaction size就是16x4=64bytes。如果只有上半或者下半half被使用，则transcation size就会reduce成32 bytes。transaction types可以读和写，每个transaction types有3个transaction size（32-、64-、128-bytes），意味着每一个GPU上跑的kernel 有6种可能的memory transactions。
### Partition Camping Problem
![](https://raw.githubusercontent.com/liuxubit/picgo/partition_camping/1.png)
global memory 被分成 256 bytes 宽的 6 partitions或 8 partitions，partition camping problem和shared memory bank conflicts相似，但是在宏观上是在这样的场景下：所有active warps是在partitions的一个子集进行执行的，所以导致了active warps在部分partitions里面排队等待执行，但是剩余的partitions却被闲置。
### Designing the Micro-BenchmarksW
设计了一个micro-benchmark来研究不同memory access的效果，然后展示micro-benchmars是如何影响partition camping effect的，在sec4 展示了同样的benchmarks对真实应用的影响。尤其是，设计了一个可能的执行时间范围，来藐视partition camping存在的程度。设计时考虑两种极端情况：1）所有的可用的partition共同使用（Without Partition Camping）；2）只有一个memory partition被访问（With Partition Camping）。每组benchmarks都测试了不同的memory transaction types（reads and wirtes），不同的transaction sizes（32-、64- and 128-bytes），benchmars为12种。如下图，partition camping可以达到七倍的降效，这一结果来自于跑一个简单的64-byte 的micro-kernel read。
![](https://raw.githubusercontent.com/liuxubit/picgo/partition_camping/2.png)
- 不带Partition Camping的benchmark
![](https://raw.githubusercontent.com/liuxubit/picgo/partition_camping/3.png)
- 带有Partition Camping的benchmark
![](https://raw.githubusercontent.com/liuxubit/picgo/partition_camping/4.png)
可以模拟partition camping effect的benchmarks(figure 5)仅从一个partition里访问内存，写和读相似。修改TYPE为2-byte、4-byte、8-byte以触发32-、64-、128-byte memory transaction to global memory。
### performance range prediction
首先设计了精确的模型来预测在GPU的kernel里的partition camping的影响
- Performance Model
在研究partition camping effect的时候，将memory read 和 memory write 分开，将with partition camping和without partition camping分开，因此一共分4种情况。使用几个独立的参数来研究partition camping：1）每个SM的活跃的warps；2）每个线程读和写的线程字的长度。字长只有三种类型：2-、4-、8-byte，文中将其作为group variable（b）。因此，首先将data-type分成两种：$b_2$ 和 $b_4$，他们的相关系数为0或1。如果都不设置，则使用8bytes。使用$\alpha_i$来表示不同变量的作用，$\beta$是常量偏置，其公式为：
$t = \alpha_1 * \omega + \alpha_2 * b_2  + \alpha_3 * b_4 + \beta$
然后使用SAS来分析，得出结果。
### APPLICATION SUITE
选择了三个实现来验证模型：1）GEM（Gaussian Electrostatic Model）；2）Clique-Counter；3）Matrix Transpose。这里只介绍Matrix Transpose。
Matrix Transpose 选择了两个不同的版本：1)Transpose Coalesced;2)Transpose Diagonal。二者的区别是global memory access pattern，8a展示了coalesced pattern，8b展示了diagonal pattern。
![](https://raw.githubusercontent.com/liuxubit/picgo/partition_camping/5.png)


