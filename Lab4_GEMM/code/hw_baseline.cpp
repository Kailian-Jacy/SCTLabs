#include "hw.h"

#define N __X_N
#define L1Size 8
// #define L2Size 8
#define Threads 4

int matA[N * N], matB[N * N],
    matCm[N * N], matCm2[N * N];

int *matC, *matC2, n;

void Orig()
{
    //#pragma omp parallel for schedule(guided, 100)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < i; ++j)
        {
            int t = matA[i * N + j];
            matA[i * N + j] = matA[j * N + i];
            matA[j * N + i] = t;
        }
    //#pragma omp parallel for schedule(guided, 100)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < i; ++j)
        {
            int t = matB[i * N + j];
            matB[i * N + j] = matB[j * N + i];
            matB[j * N + i] = t;
        }
    for (int k = 0; k < n; ++k)
    {
#pragma omp parallel for
        for (int i = 0; i < N * N; ++i)
            matA[i] += matB[i];
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
            {
                int sum = 0;
                for (int k = 0; k < N; ++k)
                    sum += matC[i * N + k] * matA[j * N + k];
                matC2[i * N + j] = sum;
            }
        int *t = matC;
        matC = matC2;
        matC2 = t;
    }
}

void Divide()
{
    for (int k = 0; k < n; ++k)
    {
        // omp_set_num_threads(64);
        // printf("starting adding.\n");
#pragma vector temporal ivdep
#pragma GOMP_CPU_AFFINITY = "0-4"
#pragma omp parallel for num_threads(Threads)
        for (int i = 0; i < N * N; ++i)
            matA[i] += matB[i];
        // printf("starting calculating.\n");
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
                            // sum += matC[i * N + tt + t] * matA[jj * N + (tt + t)];
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
    // printf("starting..\n");
    input(matA, matB);
    matC = matCm;
    matC2 = matCm2;
    n = 1;
    // printf("starting Memcpy");
    memcpy(matC, matA, sizeof(int[N * N]));

    // Orig();
    Divide();
    output(matC, n);
}
