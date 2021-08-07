#include <cuda.h>
#include <stdio.h>

#include <iomanip>
#include <iostream>
#include <random>

// const int kSize = 5; // 5000 * 5000 matrix;
const int kSize = 5000; // 5000 * 5000 matrix;
// const int kKernelSize = 3; // 13 * 13 kernal
const int kKernelSize = 13; // 13 * 13 kernal

#define InitRandom()                         \
  std::random_device r;                      \
  std::default_random_engine generator(r()); \
  std::uniform_real_distribution<float> distribution(0, 1e3);

void Conv(const float *const a, const float *const w, float *const b);

void Generate(float *const a, float *const w)
{
#pragma omp parallel for
  for (int i = 0; i < kSize; ++i)
  {
    InitRandom();
    const int j_upperbound = (i + 1) * kSize;
    for (int j = i * kSize; j < j_upperbound; ++j)
      a[j] = distribution(generator);
  }
  {
    InitRandom();
    for (int i = 0; i < kKernelSize * kKernelSize; ++i)
      w[i] = distribution(generator);
  }
}

void Check(const float *const a, const float *const w, float *const b)
{
  auto b_std = new float[kSize * kSize];
  Conv(a, w, b_std);
  for (int i = 0; i < kSize * kSize; ++i)
  {
    if (abs(b[i] / b_std[i] - 1) > 1e-3 || isnanf(b[i]) || isinff(b[i]))
    {
      std::cout << "\x1b[31m"
                   "Wrong Answer"
                   "\x1b[0m"
                   " at "
                << i << std::endl;
      std::cout << "expected " << b_std[i] << " but found " << b[i]
                << std::endl;
      delete[] b_std;
      return;
    }
  }
  std::cout << "\x1b[32m"
               "Correct"
               "\x1b[0m"
            << std::endl;

  delete[] b_std;
}

void Output(const float *const a, const float *const w, const float *const b)
{
  for (int i = 0; i < kSize; ++i)
  {
    for (int j = 0; j < kSize; ++j)
      std::cout << std::setw(2) << a[i * kSize + j] << ' ';
    std::cout << std::endl;
  }

  for (int i = 0; i < kKernelSize; ++i)
  {
    for (int j = 0; j < kKernelSize; ++j)
      std::cout << std::setw(2) << w[i * kKernelSize + j] << ' ';
    std::cout << std::endl;
  }

  for (int i = 0; i < kSize; ++i)
  {
    for (int j = 0; j < kSize; ++j)
      std::cout << std::setw(2) << b[i * kSize + j] << ' ';
    std::cout << std::endl;
  }
}

void PrintMatrix(const float *const a, int x, int y)
{
  for (int i = 0; i < x; i++)
  {
    for (int j = 0; j < y; j++)
    {
      printf("%lf ", a[i * x + j]);
    }
    printf("\n");
  }
}

// a = original matrix ; w = convolution kernal ; b = result.
void Conv(const float *const a, const float *const w, float *const b)
{
  // #pragma omp parallel for
  for (int i = 0; i < kSize; ++i)
  {
    for (int j = 0; j < kSize; ++j)
    {
      float conv = 0;
      int x = i - kKernelSize / 2, y = j - kKernelSize / 2;
      // int xBase = x, yBase = y;
      // int xt = x, yt = y;
      for (int k = 0; k < kKernelSize; ++k)
      {
        // xt = xBase + k;
        for (int l = 0; l < kKernelSize; ++l)
        {
          // int yt = yBase + l + 1;
          // printf("%d,%d ", x, y);
          if (!(x < 0 || x >= kSize || y < 0 || y >= kSize))
          {
            // int adder = a[x * kSize + y] * w[k * kKernelSize + l];
            // printf("%d ", adder);
            float aValue = a[x * kSize + y];
            float wValue = w[k * kKernelSize + l];
            float dconv = aValue * wValue;
            // printf("a[%d]_=%d,w[%d]=%d_conv=%d \n", x * kSize + y, a[x * kSize + y], k * kKernelSize + l, w[k * kKernelSize + l], dconv);
            // printf("a[%d]=%d, w[%d] = %d\n", x * kSize + y, a[x * kSize + y], k * kKernelSize + l, w[k * kKernelSize + l]);
            conv = conv + dconv;
          }
          y++;
          // printf("%d ", conv);
          // printf("(x,xt):(%d,%d)", x, xt);
          // printf("(y,yt):(%d,%d) ", y, yt);
        }
        x++;
        y -= kKernelSize;
        // printf("\n");
      }
      b[i * kSize + j] = conv;
      // printf("%ld ", b[i * kSize + j]);
      // }
      // printf("\n");
      // }

      // printf("\n-----------------------\n");
      // for (int i = 0; i < kSize; ++i)
      // {
      //   for (int j = 0; j < kSize; ++j)
      //   {
      //     float conv = 0;
      //     int x = i - kKernelSize / 2, y = j - kKernelSize / 2;
      //     int xt = x, yt = y;
      //     for (int k = 0; k < kKernelSize; ++k)
      //     {
      //       xt = x + k;
      //       for (int l = 0; l < kKernelSize; ++l)
      //       {
      //         yt = y + l;
      //         // printf("%d,%d ", xt, yt);
      //         if (!(xt < 0 || xt >= kSize || yt < 0 || yt >= kSize))
      //         {
      //           int adder = a[xt * kSize + yt] * w[k * kKernelSize + l];
      //           printf("%d ", adder);
      //           conv += a[xt * kSize + yt] * w[k * kKernelSize + l];
      //         }
      //       }
      //     }
      //     // b[i * kSize + j] = conv;
      //     printf("[%d] ", conv);
    }
    // printf("\n");
  }
}

__global__ void ConvGPU(float *ad, float *wd, float *bd, float *tmp)
{
  /*
    blockIdx.x = i
    blockIdx.y = j
    // ?~M?~C?确?~Z blockDim ?~Z~D?~P??~I?~@~B GridDim
    blockDim.x = blockDim.y = kKernalSize
    threadIdx.x = k
    threadIdx.y = l
  */

  float conv = 0;
  int x = blockIdx.x - kKernelSize / 2 + threadIdx.x;
  int y = blockIdx.y - kKernelSize / 2 + threadIdx.y;
  // printf("x, y = (%d, %d)", x, y);

  if (!(x < 0 || x >= kSize || y < 0 || y >= kSize))
  {
    int adi = x * kSize + y;
    int wdi = threadIdx.x * kKernelSize + threadIdx.y;
    conv = ad[adi] * wd[wdi];
  }

  unsigned int tidInBlk = threadIdx.x * kKernelSize + threadIdx.y;
  if (tidInBlk >= kKernelSize * kKernelSize)
  {
    return;
  }

  float *idata = tmp + (blockIdx.x * kSize + blockIdx.y) * kKernelSize * kKernelSize;
  idata[tidInBlk] = conv;
  __syncthreads();
  if (tidInBlk == 0)
  {
    for (int i = 1; i < kKernelSize * kKernelSize; i++)
    {
      idata[0] += idata[i];
    }
  }
  __syncthreads();
  bd[blockIdx.x * kSize + blockIdx.y] += idata[0];
  // printf("%ld ", conv);

  // printf("%ld ", bd[blockIdx.x * kSize + blockIdx.y]);

  // if (blockIdx.y == 4 && threadIdx.x == 0 && threadIdx.y == 0)
  // {
  //   // printf("\n");
  // }

  // if (blockIdx.y == 4 && threadIdx.x == 0 && threadIdx.y == 0)
  // {
  //   printf("(%d,%d)\n", x, y);
  // }
  // else if (threadIdx.x == 0 && threadIdx.y == 0)
  // {
  //   printf("(%d,%d) ", x, y);
  // }

  // printf("Thread[%d]: running here1\n", blockIdx.x * blockDim.x + threadIdx.x);
  // float conv = 0;

  // // Every thread has to execute this. Optimize it?
  // if (!(x < 0 || x >= kSize || y < 0 || y >= kSize))
  // {
  //   conv += a[x * kSize + y] * w[threadIdx.x * kKernelSize + threadIdx.y];
  // }
  // int index = blockIdx.y * blockDim.x + blockIdx.x;
}

int main()
{
  // generate original data from host.
  auto a = new float[kSize * kSize];
  auto w = new float[kKernelSize * kKernelSize];
  auto b = new float[kSize * kSize];
  // float a[25] = {323, 133, 234, 645, 567,
  //                345, 665, 453, 124, 535,
  //                235, 675, 234, 645, 234,
  //                627, 346, 346, 264, 756,
  //                254, 668, 671, 374, 432};
  // float w[9] = {67,
  //               562,
  //               782,
  //               567,
  //               782,
  //               43,
  //               435,
  //               12,
  //               768};
  Generate(a, w);

  float *bx = NULL;
  bx = (float *)malloc(sizeof(float) * kSize * kSize);

  // cudaEvent_t start_e, stop_e;
  // cudaEventCreate(&start_e);
  // cudaEventCreate(&stop_e);

  // // cuda start timing
  // cudaEventRecord(start_e);

  int gridSize = kSize * kSize;               // how many blocks.
  int kernalSize = kKernelSize * kKernelSize; // how many threads in a block.

  // initialize data in device memory.
  float *ad = NULL, *tmp = NULL,
        *wd = NULL, *bd = NULL;
  cudaMalloc(&ad, gridSize * sizeof(float));
  cudaMalloc(&wd, kernalSize * sizeof(float));
  cudaMalloc(&bd, gridSize * sizeof(float));
  cudaMalloc(&tmp, gridSize * kernalSize * sizeof(float));

  cudaDeviceSynchronize();

  cudaMemcpy(ad, a, gridSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(wd, w, kernalSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(bd, b, gridSize * sizeof(float), cudaMemcpyHostToDevice);

  /*
    optimism strategy? transfer some of the data and start to run. when calculating
    continue and finish the transferring.
    [todo] test the time consumption and choose wether to optimize it.
    [easy] there are few more functions for 2D or 3D malloc and transfusion.
    [hard] it's hard to evaluate the disposal speed and transfer speed. 
    [solution] make use of some synchronize methods? 
  */

  cudaDeviceSynchronize();

  // the promoted GPU version.
  dim3 grid(kSize, kSize, 1);
  dim3 block(kKernelSize, kKernelSize, 1);
  // printf("kernalSize:%d kSize:%d\n", kKernelSize, kSize);
  // printf("block: %d, %d, %d",block.x, block.y, block.z);

  // the basic solution with CPU loops.
  Conv(a, w, b);
  PrintMatrix(b, kSize, kSize);
  ConvGPU<<<grid, block>>>(ad, wd, bd, tmp);
  printf("\n------------\n");
  cudaDeviceSynchronize();

  cudaMemcpy(bx, bd, kSize * kSize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  PrintMatrix(bx, kSize, kSize);
  // cudaEventRecord(stop_e);
  // cudaEventSynchronize(stop_e);

  // Check(a, w, b);

  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start_e, stop_e);
  // std::cout << milliseconds << " milliseconds" << std::endl;
  // cudaEventDestroy(start_e);
  // cudaEventDestroy(stop_e);

  // Output(a, w, b);

  // delete[] a;
  // delete[] w;
  // delete[] b;
  return 0;
}
