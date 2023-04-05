
## 1.题目：
https://developer.download.nvidia.cn/assets/cuda/files/MatrixTranspose.pdf

## 2.做的是什么:
矩阵转置优化,具体包括三个技术：
- to和from global memory的合并访存
- shared memory bank conflicts
- partition camping


## 3.怎么做的:

