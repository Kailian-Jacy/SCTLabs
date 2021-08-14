#include "hw.h"
#include "mpi.h"

#define N __X_N
#define L1Size 8
// #define L2Size 8
#define Threads 4

int matA[N * N], matB[N * N],
    matCm[N * N], matCm2[N * N];

int *matC, *matC2, n;

void Calculate()
{
    for (int k = 0; k < n; ++k)
    {
#pragma vector temporal ivdep
#pragma GOMP_CPU_AFFINITY = "0-4"
#pragma omp parallel for num_threads(Threads)
        for (int i = 0; i < N * N; ++i)
            matA[i] += matB[i];
        int threadSize = N / Threads;
#pragma vector temporal ivdep
#pragma GOMP_CPU_AFFINITY = "0-4"
#pragma omp parallel for num_threads(Threads)
        for (int j = 0; j < N - threadSize; j += threadSize)
        {
            int t = 0;
            for (t = 0; t < N - L1Size; t += L1Size)
            {
                for (int i = 0; i < N; ++i)
                    for (int jj = 0; jj < threadSize; jj++)
                    {
                        int sum = 0;
                        for (int tt = 0; tt < L1Size; tt++)
                        {
                            sum += matC[i * N + tt + t] * matA[jj + j + (tt + t) * N];
                        }
                        matC2[i * N + jj + j] += sum;
                    }
            }
            for (; t < N; t++)
            {
                for (int i = 0; i < N; ++i)
                    for (int jj = 0; jj < threadSize; jj++)
                    {
                        // matC2[i * N + jj] += matC[i * N + t] * matA[jj * N + t];
                        int sum = 0;
                        sum += matC[i * N + t] * matA[jj + j + t * N];
                        matC2[i * N + jj + j] += sum;
                    }
            }
        }
#pragma vector temporal ivdep
#pragma GOMP_CPU_AFFINITY = "0-4"
#pragma omp parallel for num_threads(Threads)
        for (int jj = N - (N - 1) % threadSize - 1; jj < N; jj++)
        {
            int t = 0;
            for (t = 0; t < N - L1Size; t += L1Size)
            {
                for (int i = 0; i < N; ++i)
                {
                    int sum = 0;
                    for (int tt = 0; tt < L1Size; tt++)
                    {
                        sum += matC[i * N + tt + t] * matA[jj + N * (tt + t)];
                        // sum += matC[i * N + tt + t] * matA[jj + (tt + t) * N];
                    }
                    matC2[i * N + jj] += sum;
                }
            }
            for (; t < N; t++)
            {
                for (int i = 0; i < N; ++i)
                {
                    int sum = 0;
                    sum += matC[i * N + t] * matA[jj + t * N];
                    matC2[i * N + jj] += sum;
                    // matC2[i * N + jj] += matC[i * N + t] * matA[jj + t * N];
                }
            }
        }
        int *tmp = matC;
        matC = matC2;
        matC2 = tmp;
    }
}

int main()
{
    input(matA, matB);
    matC = matCm;
    matC2 = matCm2;
    n = 1;
    memcpy(matC, matA, sizeof(int[N * N]));

    Calculate();
    output(matC, n);
}
