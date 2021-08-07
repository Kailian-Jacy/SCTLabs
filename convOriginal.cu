#include <cuda.h>

#include <iomanip>
#include <iostream>
#include <random>

// const int kSize = 5000;
const int kSize = 25;
const int kKernelSize = 13; // odd

#define InitRandom()                         \
  std::random_device r;                      \
  std::default_random_engine generator(r()); \
  std::uniform_real_distribution<float> distribution(0, 1e3);

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

void Conv(const float *const a, const float *const w, float *const b)
{
#pragma omp parallel for
  for (int i = 0; i < kSize; ++i)
  {
    for (int j = 0; j < kSize; ++j)
    {
      float conv = 0;
      int x = i - kKernelSize / 2, y = j - kKernelSize / 2;
      for (int k = 0; k < kKernelSize; ++k)
      {
        for (int l = 0; l < kKernelSize; ++l)
        {
          if (!(x < 0 || x >= kSize || y < 0 || y >= kSize))
            conv += a[x * kSize + y] * w[k * kKernelSize + l];
          y++;
        }
        x++;
        y -= kKernelSize;
      }
      b[i * kSize + j] = conv;
    }
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

// void TraverseGrandMatrix()
// {
//   int skSize = 20;
// }

int main()
{
  auto a = new float[kSize * kSize];
  auto w = new float[kKernelSize * kKernelSize];
  auto b = new float[kSize * kSize];
  Generate(a, w);

  cudaEvent_t start_e, stop_e;
  cudaEventCreate(&start_e);
  cudaEventCreate(&stop_e);

  cudaEventRecord(start_e);

  // initialize data in device memory.
  float *ad = NULL, *tmp = NULL,
        *wd = NULL, *bd = NULL;
  cudaMalloc(&ad, kSize * kSize * sizeof(float));
  cudaMalloc(&wd, kKernelSize * kKernelSize * sizeof(float));
  cudaMalloc(&bd, kSize * kSize * sizeof(float));
  cudaMalloc(&tmp, kSize * kSize * kKernelSize * kKernelSize * sizeof(float));

  cudaDeviceSynchronize();

  cudaMemcpy(ad, a, kSize * kSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(wd, w, kKernelSize * kKernelSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(bd, b, kSize * kSize * sizeof(float), cudaMemcpyHostToDevice);

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

  ConvGPU<<<grid, block>>>(ad, wd, bd, tmp);
  cudaDeviceSynchronize();

  cudaMemcpy(b, bd, kSize * kSize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaEventRecord(stop_e);
  cudaEventSynchronize(stop_e);

  Check(a, w, b);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start_e, stop_e);
  std::cout << milliseconds << " milliseconds" << std::endl;
  cudaEventDestroy(start_e);
  cudaEventDestroy(stop_e);

  // Output(a, w, b);

  delete[] a;
  delete[] w;
  delete[] b;
  return 0;
}
