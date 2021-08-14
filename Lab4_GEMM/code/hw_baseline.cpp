#include "hw.h"

#define N __X_N
#define L1Size 8
#define Threads 4

// #define TotalWidth N
// #define TotalHeight N

int matA[N * N], matB[N * N],
    matCm[N * N], matCm2[N * N];

int *matC, *matC2, n, myid;

int StartWidth[4] = {0, (N + 1) / 2, 0, (N + 1) / 2};
int EndWidth[4] = {(N + 1) / 2, N, (N + 1) / 2, N};
int StartHeight[4] = {0, N / 2, 0, N / 2};
int EndHeight[4] = {N / 2, N, N / 2, N};
int Count[4];

// struct message
// {
//     int width;
//     int height;
// };

void MainProcess(int StartWidth, int EndWidth, int StartHeight, int EndHeight)
{
    printf("[Processing.]Process %d: \nsw: %d, nw: %d\n sh: %d, ew: %d\n=========", myid, StartWidth, EndWidth, StartHeight, EndHeight);
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
        for (int j = StartWidth; j < EndWidth - threadSize; j += threadSize)
        {
            int t = 0;
            for (t = 0; t < N - L1Size; t += L1Size)
            {
                for (int i = StartHeight; i < EndHeight; ++i)
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
                for (int i = StartHeight; i < EndHeight; ++i)
                    for (int jj = 0; jj < threadSize; jj++)
                    {
                        int sum = 0;
                        sum += matC[i * N + t] * matA[jj + j + t * N];
                        matC2[i * N + jj + j] += sum;
                    }
            }
        }
        int blockWidth = EndWidth - StartWidth;
#pragma vector temporal ivdep
#pragma GOMP_CPU_AFFINITY = "0-4"
#pragma omp parallel for num_threads(Threads)
        for (int jj = EndWidth - (blockWidth - 1) % threadSize - 1; jj < EndWidth; jj++)
        {
            int t = 0;
            for (t = 0; t < N - L1Size; t += L1Size)
            {
                for (int i = StartHeight; i < EndHeight; ++i)
                {
                    int sum = 0;
                    for (int tt = 0; tt < L1Size; tt++)
                    {
                        sum += matC[i * N + tt + t] * matA[jj + N * (tt + t)];
                    }
                    matC2[i * N + jj] += sum;
                }
            }
            for (; t < N; t++)
            {
                for (int i = StartHeight; i < EndHeight; ++i)
                {
                    int sum = 0;
                    sum += matC[i * N + t] * matA[jj + t * N];
                }
            }
        }
        int *tmp = matC;
        matC = matC2;
        matC2 = tmp;
    }
    printf("[process] process %d done.\n", myid);
}

void Merge(int i)
{
#pragma omp parallel for num_threads(Threads)
    for (int j = StartWidth[i]; j < EndWidth[i]; j++)
    {
        for (int i = StartHeight[i]; i < EndHeight[i]; i++)
        {
            matC[i * N + j] = matC2[i * N + j];
        }
    }
}

int main(int argc, char **argv)
{
    printf("[mpi]: SendingMatrix.\n ");

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0)
    {
        input(matA, matB);
        matC = matCm;
        matC2 = matCm2;
        n = 1;
        memcpy(matC, matA, sizeof(int[N * N]));
    }

    MPI_Bcast(matA, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matB, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(matC, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    printf("[mpi]: MatrixSended.\n ");

    if (myid == 0)
    {
        MainProcess(StartWidth[0], EndWidth[0], StartHeight[0], EndHeight[0]);
        MPI_Status stats[4];
        for (int i = 0; i < 4; i++)
        {
            Count[i] = (EndWidth[i] - StartWidth[i] - 1) * (EndHeight[i] - StartHeight[i] - 1);
        }
        for (int i = 1; i < 4; i++)
        {
            printf("Receiving result from process: %d\n", i);
            MPI_Recv(matC2, Count[i], MPI_INT, i, 0, MPI_COMM_WORLD, &(stats[0]));
            printf("error code: %d\n Starting Merging...\n", stats[i].MPI_ERROR);
            Merge(i);
            printf("Merging Done\n ");
        }

        output(matC, n);
    }
    if (myid == 1)
    {
        MainProcess(StartWidth[myid], EndWidth[myid], StartHeight[myid], EndHeight[myid]);
        printf("[node1]Sending Result to process 0\n");
        MPI_Send(matC, Count[1], MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf("[node1]Sending Done\n");
    }
    if (myid == 2)
    {
        MainProcess(StartWidth[myid], EndWidth[myid], StartHeight[myid], EndHeight[myid]);
        printf("[node2]Sending Result to process 0\n");
        MPI_Send(matC, Count[2], MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf("[node2]Sending Done\n");
    }
    if (myid == 3)
    {
        MainProcess(StartWidth[myid], EndWidth[myid], StartHeight[myid], EndHeight[myid]);
        printf("[node3]Sending Result to process 0\n");
        MPI_Send(matC, Count[3], MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf("[node3]Sending Done\n");
    }

    MPI_Finalize();
}

// Test1: The size of blocks. The usage of each node.

// 在 mpi run 的时候如何绑定进程至主机 ip 地址。
// 格式化整理打印调试页面。