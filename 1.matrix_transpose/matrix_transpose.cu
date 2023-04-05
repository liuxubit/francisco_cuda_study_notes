#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM    (32)
#define BLOCK_ROWS  (8)
#define NUM_REPS    (100)

__global__ void copy(float *odata, float *idata, int width, int height, int nreps)
{
    int x_index = blockIdx.x * tile_dim + threadIdx.x;
    int y_index = blockIdx.y * tile_dim + threadIdx.y;

    int index = width * y_index + x_index;

    for(int r = 0; r < nreps; ++r){
        for(int i = 0; i < tile_dim; ++i){
            odata[index + i * width] = idata[index + i * width];
        }
    }
}

/*
    use blocks of 32x8 threads on a 32x32 matrix tiles
    odata: pt to output
    idata: pt to input
    width: 
    height:
    nreps: how many times the loop over data movement between matrices are performed
*/
__global__ void transpose_naive(float *odata, float *idata, int width, int height, int nreps)
{
    int x_index = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_index = blockIdx.y * TILE_DIM + threadIdx.x;

    int index_in = y_index * width + x_index;
    int index_out = x_index * height + y_index;

    for(int r = 0; r < nreps; ++r)
    {
        for(int i = 0; i < TILE_DIM; ++i)
        {
            odata[index_out + i] = idata[index_in + i * width];
        }
    }

}

__global__ void transpose_coalesced(float *odata, float *idata, int width, int height, int nreps)
{

    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x_index = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_index = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in = y_index * width + x_index;

    x_index = blockIdx.y * TILE_DIM + threadIdx.x;
    y_index = blockIdx.x * TILE_DIM + threadIdx.y;

    int index_out = y_index * height + x_index;

    for(int r = 0; r < nreps; ++r)
    {
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
        }
        __syncthreads();
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
        }
    }

}

__global__ void copy_shared_mem(float *odata, float *idata, int width, int height, int nreps)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x_index = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_index = blockIdx.y * TILE_DIM + threadIdx.y;

    int index = y_index * width + x_index;

    for (int r = 0; r < nreps; ++r)
    {
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            tile[threadIdx.y + i][threadIdx.x] = idata[index + i * width];
        }
        __syncthreads();
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            odata[index + i * width] = tile[threadIdx.y + i][threadIdx.x];
        }
    }

}

__global__ void transpose_coalesced_no_bank_conflits(float *odata, float *idata, int width, int height, int nreps)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int index_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int index_y = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in = index_y * width + index_x;

    index_x = blockIdx.y * TILE_DIM + threadIdx.x;
    index_y = blockIdx.x * TILE_DIM + threadIdx.y;

    int index_out = index_y * height + index_x;

    for (int r = 0; r < nreps; ++r)
    {
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
        }
        __syncthreads();
        for(int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

__global__ void transpose_diagonal(float *odata, float *idata, int width, int height, int nreps)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int blockIdx_x, blockIdx_y;

    // diagonal reordering
    if (width == height)
    {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    }
    else {
        int bid = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    int x_index = blockIdx_x * TILE_DIM + threadIdx.x;
    int y_index = blockIdx_y * TILE_DIM + threadIdx.y;
    int index_in = x_index + y_index * width;

    x_index = blockIdx_y * TILE_DIM + threadIdx.x;
    y_index = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = x_index + y_index * height;

    for (int r = 0; r < nreps; ++r)
    {
        for (int i = 0; i < TILE_DIM; ++i)
        {
            tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
        }
        __syncthreads();
        for (int i = 0; i < TILE_DIM; ++i)
        {
            odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
        }
    }

}


设计了一个micro-benchmark来研究不同memory access的效果，然后展示micro-benchmars是如何影响partition camping effect的，在sec4 展示了同样的benchmarks对真实应用的影响。尤其是，设计了一个可能的执行时间范围，来藐视partition camping存在的程度。设计时考虑两种极端情况：1）所有的可用的partition共同使用（Without Partition Camping）；2）只有一个memory partition被访问（With Partition Camping）。每组benchmarks都测试了不同的memory transaction types（reads and wirtes），不同的transaction sizes（32-、64- and 128-bytes），benchmars为12种。如下图，partition camping可以达到七倍的降效，这一结果来自于跑一个简单的64-byte 的micro-kernel read。
![](https://raw.githubusercontent.com/liuxubit/picgo/partition_camping/2.png)























