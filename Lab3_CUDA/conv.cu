#include <cuda.h>

#include <iomanip>
#include <iostream>
#include <random>

const int kSize = 5000;
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

__global__ void ConvGPU(const float *const ad, const float *const wd, float *bd, float *tmp, int const roundX, int const roundY)
{

  int RoundStartingIndex = 50 * roundY + 50 * roundX * 5000;

  float conv = 0;
  int x = blockIdx.x - 13 / 2 + threadIdx.x;
  int y = blockIdx.y - 13 / 2 + threadIdx.y;

  if (!((x + 50 * roundX < 0) || (x + 50 * roundX >= 5000) || (y + 50 * roundY < 0) || (y + 50 * roundY >= 5000)))
  {
    conv = ad[RoundStartingIndex + x * 5000 + y] * wd[threadIdx.x * 13 + threadIdx.y];
  }

  unsigned int tidInBlk = threadIdx.x * 13 + threadIdx.y;
  if (tidInBlk >= 13 * 13)
  {
    return;
  }

  float *idata = tmp + (blockIdx.x * 50 + blockIdx.y) * 13 * 13;
  idata[tidInBlk] = conv;
  __syncthreads();
  if (tidInBlk == 0)
  {
    for (int i = 1; i < 13 * 13; i++)
    {
      idata[0] += idata[i];
    }
  }
  bd[RoundStartingIndex + blockIdx.x * 5000 + blockIdx.y] = idata[0];
}

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

  int tmpKsize = 50;

  // initialize data in device memory.
  float *ad = NULL, *tmp = NULL,
        *wd = NULL, *bd = NULL;
  cudaMalloc(&ad, kSize * kSize * sizeof(float));
  cudaMalloc(&wd, kKernelSize * kKernelSize * sizeof(float));
  cudaMalloc(&bd, kSize * kSize * sizeof(float));
  cudaMalloc(&tmp, tmpKsize * tmpKsize * kKernelSize * kKernelSize * sizeof(float));

  cudaDeviceSynchronize();

  cudaMemcpy(ad, a, kSize * kSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(wd, w, kKernelSize * kKernelSize * sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  // the promoted GPU version.
  dim3 grid(tmpKsize, tmpKsize, 1);
  dim3 block(kKernelSize, kKernelSize, 1);

  for (int i = 0; i < kSize / tmpKsize; i++)
  {
    for (int j = 0; j < kSize / tmpKsize; j++)
    {
      ConvGPU<<<grid, block>>>(ad, wd, bd, tmp, i, j); // for efficiency concern, tmpKsize is hard-coded.
    }
  }
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
