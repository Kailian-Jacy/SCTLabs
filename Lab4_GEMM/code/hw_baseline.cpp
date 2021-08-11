#include "hw.h"

#define N __X_N

int matA[N * N], matB[N * N],
    matCm[N * N], matCm2[N * N];

int *matC, *matC2, n;

/*
    这是经过验证版本的最初基准函数。
    可以发现，取消转置之后，运行速度就降到了 200Mops (N=1024数据量) 
    原先转置的时候是 400 左右。
*/

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

void Divide(int L1Size, int L2Size)
{
    // #pragma omp parallel for schedule(guided, 100)
    // for (int i = 0; i < N; ++i)
    //     for (int j = 0; j < i; ++j)
    //     {
    //         int t = matA[i * N + j];
    //         matA[i * N + j] = matA[j * N + i];
    //         matA[j * N + i] = t;
    //     }
    // //#pragma omp parallel for schedule(guided, 100)
    // for (int i = 0; i < N; ++i)
    //     for (int j = 0; j < i; ++j)
    //     {
    //         int t = matB[i * N + j];
    //         matB[i * N + j] = matB[j * N + i];
    //         matB[j * N + i] = t;
    //     }
    for (int k = 0; k < n; ++k)
    {
#pragma omp parallel for
        for (int i = 0; i < N * N; ++i)
            matA[i] += matB[i];
        int j = 0;
#pragma omp parallel for
        for (j = 0; j < N - L2Size; j += L2Size)
        {
            int t = 0;
            for (t = 0; t < N - L1Size; t += L1Size)
            {
                for (int i = 0; i < N; ++i)
                    for (int jj = 0; jj < L2Size; jj++)
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
                    for (int jj = 0; jj < L2Size; jj++)
                    {
                        // matC2[i * N + jj] += matC[i * N + t] * matA[jj * N + t];
                        int sum = 0;
                        sum += matC[i * N + t] * matA[jj + j + t * N];
                        matC2[i * N + jj + j] += sum;
                    }
            }
        }
#pragma omp parallel for
        for (; j < N; j++)
        {
            int t = 0;
            for (t = 0; t < N - L1Size; t += L1Size)
            {
                for (int i = 0; i < N; ++i)
                {
                    int sum = 0;
                    for (int tt = 0; tt < L1Size; tt++)
                    {
                        sum += matC[i * N + tt + t] * matA[j + N * (tt + t)];
                        // sum += matC[i * N + tt + t] * matA[j + (tt + t) * N];
                    }
                    matC2[i * N + j] += sum;
                }
            }
            for (; t < N; t++)
            {
                for (int i = 0; i < N; ++i)
                {
                    int sum = 0;
                    sum += matC[i * N + t] * matA[j + t * N];
                    matC2[i * N + j] += sum;
                    // matC2[i * N + j] += matC[i * N + t] * matA[j + t * N];
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
    // int rate[] = {10,  // 345.036
    //               20,  // 324
    //               40,  // 335
    //               50,
    //               60,  // 349
    //               70,
    //               90,
    //               100,
    //               110,
    //               150,
    //               200,
    //               250,
    //               400};
    input(matA, matB);
    matC = matCm;
    matC2 = matCm2;
    n = 1;
    memcpy(matC, matA, sizeof(int[N * N]));

    // Orig();
    Divide(8, 512);
    output(matC, n);
}
