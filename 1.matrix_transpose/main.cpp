#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM    (32)
#define BLOCK_ROWS  (8)
#define NUM_REPS    (100)

int main()
{

    // set matrix size
    const int size_x = 2048, size_y = 2048;

    // kernel pointer and descriptor
    void (*kernel) (float *, float *, int, int, int);
    char *kernel_name;

    // execution configuration parameters
    dim3 grid(size_x / TILE_DIM, size_y / TILE_DIM);
    dim3 threads(TILE_DIM, BLOCK_ROWS);

    // CUDA envents
    cudaEvent_t start, stop;

    // size of mempry required to store the matrix
    const int mem_size = sizeof(float) * size_x * size_y;

    // allocate host memory
    float *h_idata = (float *)malloc(mem_size);
    float *h_odata = (float *)malloc(mem_size);
    float *transposed_gold = (float *)malloc(mem_size);
    float *gold;
    
    // allocate device memory
    float *d_idata, *d_odata;
    cudaMalloc((void **) &d_idata, mem_size);
    cudaMalloc((void **) &d_odata, mem_size);

    // initialize host data
    for (int i = 0; i < (size_x * size_y); ++i){
        h_idata[i] = (float)i;
    }

    // copy host data to device
    cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

    // compute reference transpose solution
    computeTransposed(transposed_gold, h_idata, size_x, size_y);

    // print out common data for all kernels
    printf("\nMatrix size: %dx%d, tile: %dx%d, block: %dx%d\n\n", 
            size_x, size_y, TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);
    
    printf("Kernel\t\tLoop over kernel\tLoop within kernel\n");
    printf("******\t\t****************\t******************\n");

    // 
    //  Loop over different kernels
    // 
    for (int k = 0; k < 8; ++k){
        // set kernel pointer
        switch (k){
            case 0:
                kernel = &copy;
                kernel_name = "simple copy                      ";
                break;
            case 1:
                kernel = &copy_shared_mem;
                kernel_name = "shared memory copy               ";
                break;
            case 2:
                kernel = &transpose_naive;
                kernel_name = "shared memory copy               ";
                break;
            case 3:
                kernel = &transpose_coalesced;
                kernel_name = "coalesced transpose              ";
                break;
            case 4:
                kernel = &transpose_no_bank_conflicts;
                kernel_name = "no bank confict trans            ";
                break;
            case 5:
                kernel = &transpose_coarse_grained;
                kernel_name = "coarse-grained                   ";
                break;
            case 6:
                kernel = &transpose_fine_grained;
                kernel_name = "fine-grained                     ";
                break;
            case 7:
                kernel = &transpose_diagonal;
                kernel_name = "diagonal                         ";
                break;
        }

        // set reference solution
        // NB: fine and coarse grained kernels are not full transposed, so bypass check
        if (kernel == &copy || kernel == &copy_shared_mem)
        {
            gold = h_idata;
        }
        else if (kernel == &transpose_coarse_grained || kernel == &transpose_fine_grained)
        {
            gold = h_odata;
        }
        else
        {
            gold = transposed_gold;
        }

        // initialize events, EC parameters, 
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // warmup to avoid timing startup
        kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y, 1);

        // take measurements for loop over kernel launches
        cudaEventRecord(start, 0);
        cudaEventSynchronize(stop);
        float outer_time;
        cudaEventElapsedTime(&outer_time, start, stop);

        cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
        int res = comparef(gold, h_odata, size_x * size_y);
        if (1 != res){
            printf("*** %s kernel FAILED ***\n", kernel_name);
        }

        // take measurements for loop inside kernel
        cudaEventRecord(start, 0);
        kernel <<<grid, threads>>>(d_odata, d_idata, size_x, size_y, NUM_REPS);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float inner_time = 0;
        cudaEventElapsedTime(&inner_time, start, stop);

        cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
        res = comparef(gold, h_odata, size_x * size_y);
        if (1 != res){
            printf("*** %s kernel FAILED ***\n", kernel_name);
        }

        // report effective bandwidths
        float outer_bandwidth = 2.0 * 1000 * mem_size / (1<<30) / (outer_time / NUM_REPS);
        float inner_bandwidth = 2.0 * 1000 * mem_size / (1<<30) / (inner_time / NUM_REPS);

        printf("%s\t5.2f GB/s\t\t%5.2f GB/s\n", kernel_name, outer_bandwidth, inner_bandwidth);

    }
    // clean up
    free(h_idata);
    free(h_odata);
    free(transposed_gold);

    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}



















