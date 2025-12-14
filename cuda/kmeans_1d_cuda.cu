/* kmeans_1d_cuda.cu
   K-means 1D com paralelização CUDA (GPU):
   - Assignment: paralelizado na GPU (1 thread por ponto)
   - Update: feito na CPU (opção A simples)
   - Medições de tempo com CUDA events

   Compilar: nvcc -O2 kmeans_1d_cuda.cu -o kmeans_1d_cuda
   Uso:      ./kmeans_1d_cuda dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [block_size=256] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

/* ---------- Macro para verificação de erros CUDA ---------- */
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error em %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* ---------- util CSV 1D ---------- */
static int count_rows(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        exit(1);
    }
    int rows = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f))
    {
        int only_ws = 1;
        for (char *p = line; *p; p++)
        {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            {
                only_ws = 0;
                break;
            }
        }
        if (!only_ws)
            rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out)
{
    int R = count_rows(path);
    if (R <= 0)
    {
        fprintf(stderr, "Arquivo vazio: %s\n", path);
        exit(1);
    }
    double *A = (double *)malloc((size_t)R * sizeof(double));
    if (!A)
    {
        fprintf(stderr, "Sem memoria para %d linhas\n", R);
        exit(1);
    }

    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        free(A);
        exit(1);
    }

    char line[8192];
    int r = 0;
    while (fgets(line, sizeof(line), f))
    {
        int only_ws = 1;
        for (char *p = line; *p; p++)
        {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            {
                only_ws = 0;
                break;
            }
        }
        if (only_ws)
            continue;

        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if (!tok)
        {
            fprintf(stderr, "Linha %d sem valor em %s\n", r + 1, path);
            free(A);
            fclose(f);
            exit(1);
        }
        A[r] = atof(tok);
        r++;
        if (r > R)
            break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N)
{
    if (!path)
        return;
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K)
{
    if (!path)
        return;
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int c = 0; c < K; c++)
        fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- Kernel CUDA para Assignment ---------- */
__global__ void assignment_kernel(const double *X, const double *C,
                                   int *assign, double *errors,
                                   int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        int best = -1;
        double bestd = 1e300;

        for (int c = 0; c < K; c++)
        {
            double diff = X[i] - C[c];
            double d = diff * diff;
            if (d < bestd)
            {
                bestd = d;
                best = c;
            }
        }

        assign[i] = best;
        errors[i] = bestd;
    }
}

/* ---------- Update step (CPU) ---------- */
static void update_step_1d(const double *X, double *C, const int *assign, int N, int K)
{
    double *sum = (double *)calloc((size_t)K, sizeof(double));
    int *cnt = (int *)calloc((size_t)K, sizeof(int));
    if (!sum || !cnt)
    {
        fprintf(stderr, "Sem memoria no update\n");
        exit(1);
    }

    for (int i = 0; i < N; i++)
    {
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }
    for (int c = 0; c < K; c++)
    {
        if (cnt[c] > 0)
            C[c] = sum[c] / (double)cnt[c];
        else
            C[c] = X[0];
    }
    free(sum);
    free(cnt);
}

/* ---------- K-means com CUDA ---------- */
static void kmeans_1d_cuda(const double *X, double *C, int *assign,
                            int N, int K, int max_iter, double eps,
                            int block_size, int *iters_out, double *sse_out,
                            double *kernel_time_out, double *total_time_out)
{
    double *X_dev, *C_dev, *errors_dev;
    int *assign_dev;

    CUDA_CHECK(cudaMalloc(&X_dev, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&C_dev, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&assign_dev, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&errors_dev, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(X_dev, X, N * sizeof(double), cudaMemcpyHostToDevice));

    int grid_size = (N + block_size - 1) / block_size;

    double *errors_host = (double *)malloc(N * sizeof(double));

    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_total));

    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    float total_kernel_ms = 0.0f;

    printf("\n--- SSE por iteração (CUDA) ---\n");

    for (it = 0; it < max_iter; it++)
    {
        CUDA_CHECK(cudaMemcpy(C_dev, C, K * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(start_kernel));
        assignment_kernel<<<grid_size, block_size>>>(X_dev, C_dev, assign_dev, errors_dev, N, K);
        CUDA_CHECK(cudaEventRecord(stop_kernel));
        CUDA_CHECK(cudaEventSynchronize(stop_kernel));

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
        total_kernel_ms += kernel_ms;

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(assign, assign_dev, N * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(errors_host, errors_dev, N * sizeof(double), cudaMemcpyDeviceToHost));

        sse = 0.0;
        for (int i = 0; i < N; i++)
        {
            sse += errors_host[i];
        }

        printf("Iteração %d: SSE = %.6f\n", it + 1, sse);

        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps)
        {
            printf("Convergiu (variação relativa < %.6f)\n", eps);
            it++;
            break;
        }

        update_step_1d(X, C, assign, N, K);
        prev_sse = sse;
    }

    printf("----------------------------\n\n");

    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start_total, stop_total));

    *iters_out = it;
    *sse_out = sse;
    *kernel_time_out = (double)total_kernel_ms;
    *total_time_out = (double)total_ms;

    free(errors_host);
    CUDA_CHECK(cudaFree(X_dev));
    CUDA_CHECK(cudaFree(C_dev));
    CUDA_CHECK(cudaFree(assign_dev));
    CUDA_CHECK(cudaFree(errors_dev));
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
}

/* ---------- main ---------- */
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [block_size=256] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    int block_size = (argc > 5) ? atoi(argv[5]) : 256;
    const char *outAssign = (argc > 6) ? argv[6] : NULL;
    const char *outCentroid = (argc > 7) ? argv[7] : NULL;

    if (max_iter <= 0 || eps <= 0.0 || block_size <= 0)
    {
        fprintf(stderr, "Parâmetros inválidos: max_iter>0, eps>0, block_size>0\n");
        return 1;
    }

    int N = 0, K = 0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int *)malloc((size_t)N * sizeof(int));
    if (!assign)
    {
        fprintf(stderr, "Sem memoria para assign\n");
        free(X);
        free(C);
        return 1;
    }

    int iters = 0;
    double sse = 0.0;
    double kernel_time = 0.0;
    double total_time = 0.0;

    kmeans_1d_cuda(X, C, assign, N, K, max_iter, eps, block_size, &iters, &sse, &kernel_time, &total_time);

    printf("=== K-means 1D (CUDA - GPU) ===\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Block size: %d | Grid size: %d\n", block_size, (N + block_size - 1) / block_size);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, total_time);
    printf("Tempo kernel: %.1f ms (%.1f%%)\n", kernel_time, 100.0 * kernel_time / total_time);
    printf("===============================\n");

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign);
    free(X);
    free(C);
    return 0;
}
