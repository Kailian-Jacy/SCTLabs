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

void Divide()
{
    //#pragma omp parallel for schedule(guided, 100)
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
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
            {
                int sum = 0;
                for (int k = 0; k < N; ++k)
                    sum += matC[i * N + k] * matA[j + k * N];
                matC2[i * N + j] = sum;
            }
        int *t = matC;
        matC = matC2;
        matC2 = t;
    }
}

int main()
{
    input(matA, matB);
    matC = matCm;
    matC2 = matCm2;
    n = 1;
    memcpy(matC, matA, sizeof(int[N * N]));

    // Orig();
    Divide();

    output(matC, n);
}
