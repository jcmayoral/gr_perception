#include <cuda_runtime.h>
#include <stdio.h>

extern "C"{
    __device__
    int get_globalIdx_1D_1D(){
      return blockIdx.x *blockDim.x + threadIdx.x;
    }
    __global__
    void registration_kernel(int *x, int* t){
      //__shared__ int s[256];
      //int idx = blockIdx.x * blockDim.x + threadIdx.x;
      //int index = blockIdx.x * blockDim.x + threadIdx.x;
      //int stride = blockDim.x * gridDim.x;
      int index = get_globalIdx_1D_1D();

      //printf("Value %d %d \n", index,index, x[index]);
      t[x[index]] = t[x[index]] + 1;
    }

    void stop_cuda_stuff(int *x, int *t){
      cudaFree(x);
      cudaFree(t);
    }

    int call_registration(int* o_x, int size){
      // initialize x array on the host
      int * x;
      int * bin_counts;
      int * cpu_results;
      int bin_number = 1000;

      cudaMallocManaged(&x, size*sizeof(int));
      cudaMemcpy(x, o_x, size*sizeof(int), cudaMemcpyHostToDevice);
      bin_counts = static_cast<int*>(malloc(sizeof(int) * bin_number));
      memset(bin_counts, 0, bin_number*sizeof(int));
      cudaMallocManaged(&bin_counts, bin_number*sizeof(int));
    
//      for (int i =0; i< bin_number; i++){
  //       printf("index %d", bin_counts[i]);
    //  }
      int ngrid = 32;
      dim3 grid (ngrid);
      int nblocks = ceil((size+ngrid -1)/ngrid);

      registration_kernel<<<nblocks,grid>>>(x,bin_counts);
      cudaDeviceSynchronize(); // to print results

      cpu_results = (int *) malloc(bin_number * sizeof(int));
      cudaMemcpy(cpu_results, bin_counts, bin_number*sizeof(int), cudaMemcpyDeviceToHost);

      cudaFree(x);
      cudaFree(bin_counts);
      
      int max_value = -1;
      int max_index = -1;
      cpu_results[0] = 0;

      int counting = 0;
      for (int i =0; i< bin_number; i++){
       //  printf("index %d %d /n", counting, cpu_results[i]);
        if (cpu_results[i] > max_value){
          max_value = cpu_results[i];
          max_index = i;
        }
        counting = counting + 1;
      }
      printf("Max index %d", max_index);
      return max_index;
    }
}
