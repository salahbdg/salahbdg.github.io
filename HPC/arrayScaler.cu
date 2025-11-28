#include <stdio.h>

__global__ void ArrayScaler(float *data, int numElements, float factor) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        data[i] *= factor;
    }
}

int main() {
    int numElements = 10;
    float factor = 3.0f;

    // Allocate host memory
    float h_data[numElements];
    for (int i = 0; i < numElements; i++) h_data[i] = i + 1;

    // Allocate device memory
    float *d_data;
    cudaMalloc((void**)&d_data, numElements * sizeof(float));

    // Copy host → device
    cudaMemcpy(d_data, h_data, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel (1 block of 256 threads is enough for 10 elements)
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    ArrayScaler<<<numBlocks, blockSize>>>(d_data, numElements, factor);

    // Copy device → host
    cudaMemcpy(h_data, d_data, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Scaled array:\\n");
    for (int i = 0; i < numElements; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("\\n");

    // Free device memory
    cudaFree(d_data);

    return 0;
}
