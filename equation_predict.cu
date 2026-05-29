/// equation_predict.cu
/// CUDA SDK source code script >> Heavy mathematical predictive model runs on GPU for extreme performance
/// Project GitHub Repos: (xxx)
/// Developer Name   : ELYES DAGDAGUI 
/// Developer Email  : dagdaguielyes50@gmail.com
/// Developer Github : https://github.com/elyes-dagdagui

// Includes
#include <iostream>
#include <stdbool.h>
#include <math.h>
#include <cuda_runtime.h>

// CONSTANTS (NOTE:VALUES ARE SENSITIVE)
#define OFFSET 0.0001   
#define EPSILON 0.99
#define EQVARCOUNT 3
#define DEFAULT_MINRANGE -20
#define DEFAULT_MAXRANGE 22
#define DEFAULT_XMIN 65
#define DEFAULT_XMAX 122
#define DEFAULT_YMIN 4096
#define DEFAULT_YMAX 65535

// Function to run on GPU
__global__ void complexpredict(double x1, double x2, double y1, double y2, double minVar, double maxVar, double *vars, int *predictions)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double a = minVar + (i * OFFSET);
    double b = 0, c = 0;
    if (a < maxVar)
    {
        bool predict = false;
        for (b = minVar; b < maxVar && !predict; b = b + OFFSET)
        {
            for (c = minVar; c < maxVar && !predict; c = c + OFFSET)
            {
                if (fabs(((a * x1 * x1) + (b * x1) + c) - y1) < EPSILON && fabs(((a * x2 * x2) + (b * x2) + c) - y2) < EPSILON)
                {
                    predict = true;
                    atomicAdd(predictions, 1);
                }
            }
        }
        if (predict)
        {
            vars[0] = a;
            vars[1] = b - OFFSET;
            vars[2] = c - OFFSET;
        }
    }
};

// Program entry-point
int main()
{
    char userEntry;
    const int range = DEFAULT_MAXRANGE - (DEFAULT_MINRANGE);
    int N = range / OFFSET;
    size_t size = EQVARCOUNT * sizeof(double);

    // 1. RESET FIRST
    cudaDeviceReset();

    //  2. NOW ALLOCATE
    double *h_vars = (double *)malloc(size);
    int *h_predictions = (int *)malloc(sizeof(int));
    double *d_vars;
    int *d_predictions;

    cudaMalloc(&d_vars, size);
    cudaMalloc(&d_predictions, sizeof(int));

    // 3. INITIALIZE & COPY
    h_vars[0] = 0;
    h_vars[1] = 0;
    h_vars[2] = 0;
    *h_predictions = 0;
    cudaMemcpy(d_vars, h_vars, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_predictions, h_predictions, sizeof(int), cudaMemcpyHostToDevice);

    // 4. EXECUTION
    complexpredict<<<(N+255)/256, 256>>>(DEFAULT_XMIN, DEFAULT_XMAX, DEFAULT_YMIN, DEFAULT_YMAX, DEFAULT_MINRANGE, DEFAULT_MAXRANGE, d_vars, d_predictions);

    // 5. SYNC & ERROR CHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    // Copy back to CPU
    cudaMemcpy(h_vars, d_vars, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_predictions, d_predictions, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\033[32m" << h_vars[0] << "\033[0m" << std::endl;
    std::cout << "\033[31m" << h_vars[1] << "\033[0m" << std::endl;
    std::cout << "\033[34m" << h_vars[2] << "\033[0m" << std::endl;
    std::cout << *(h_predictions) << std::endl;

    // WAIT FOR USER'S SIGNAL TO QUIT THE PROGRAM
    std::cin >> userEntry;

    // Finally Cleanup
    cudaFree(d_vars);
    cudaFree(d_predictions);
    free(h_vars);
    free(h_predictions);

    return 0;
}