/*
 ============================================================================
 Name        : image-denoise.cu
 Author      : jzheadley
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <stdlib.h>
#include <stdio.h>
#define DEBUG 0
__global__ void denoise(int *r, int *g, int *b, int width, int numElements) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid % width == 0) {
		if (tid == 0) { // top left corner
			r[tid] = (r[tid] + r[tid + 1] + r[tid + width]) / 3;
		} else if (tid == (numElements - width)) { // bottom left corner
			r[tid] = (r[tid] + r[tid - width] + r[tid + 1]) / 3;
		} else { // middle left edge
			r[tid] = (r[tid] + r[tid - width] + r[tid + width] + r[tid + 1])
					/ 4;
		}
	} else if (tid < width) {
		if (tid == (width - 1)) { //top right corner
			r[tid] = (r[tid] + r[tid - 1] + r[tid + width]) / 3;
		} else { // middle first row
			r[tid] = (r[tid] + r[tid + 1] + r[tid - 1] + r[tid + width]) / 4;
		}
	} else if (tid % width == (width - 1)) {
		if (tid == numElements - 1) { //bottom right corner
			r[tid] = (r[tid] + r[tid - 1] + r[tid - width]) / 3;
		} else { //middle right edge
			r[tid] = (r[tid] + r[tid - width] + r[tid + width] + r[tid - 1])
					/ 4;
		}
	} else if (tid > numElements - width) { //bottom middle
		r[tid] = (r[tid] + r[tid - width] + r[tid + 1] + r[tid - 1]) / 4;
	} else { // middle of the matrix
		r[tid] = ((r[tid] + r[tid - width] + r[tid + width] + r[tid - 1]
				+ r[tid + 1]) / 5);
	}

}

__global__ void naiveDenoise(int *r, int *g, int *b, int width,
		int numElements) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// naive no denoise done or edges or corners
	if (tid % width == 0 || tid % width == (width - 1) || tid < width
			|| tid > numElements - width) {
	} else {
		r[tid] = ((r[tid] + r[tid - width] + r[tid + width] + r[tid - 1]
				+ r[tid + 1]) / 5);
		g[tid] = ((g[tid] + g[tid - width] + g[tid + width] + g[tid - 1]
				+ g[tid + 1]) / 5);
		b[tid] = ((b[tid] + b[tid - width] + b[tid + width] + b[tid - 1]
				+ b[tid + 1]) / 5);
	}
}
void printMatrix(int numElements, int width, int* matrix) {

	for (int i = 0; i < numElements; ++i) {
		if (i % width == 0 && i != 0) {
			printf("\n");
		}
		printf("%i\t", matrix[i]);
	}
}

int main(void) {
	int width = 2048;
	int numElements = width * width;
	time_t t;

	srand((unsigned) time(&t));

	// Allocate host memory
	int *h_r = (int *) malloc(numElements * sizeof(int));
	int *h_g = (int *) malloc(numElements * sizeof(int));
	int *h_b = (int *) malloc(numElements * sizeof(int));

	// Initialize the host arrays of 'colors'
	for (int i = 0; i < numElements; ++i) {
		h_r[i] = rand() % 255;
		h_g[i] = rand() % 255;
		h_b[i] = rand() % 255;
	}

	// Allocate the device arrays of 'colors'
	int *d_r, *d_g, *d_b;

	cudaMalloc(&d_r, numElements * sizeof(int));
	cudaMalloc(&d_g, numElements * sizeof(int));
	cudaMalloc(&d_b, numElements * sizeof(int));

	// Copy the host input vectors A and B in host memory to the device input vectors in
	cudaMemcpy(d_r, h_r, numElements * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_g, h_g, numElements * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, numElements * sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	if (DEBUG) {
		printf("original is:\n");
		printMatrix(numElements, width, h_r);
		printf("\n\n");
	}
	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	cudaEventRecord(start);

	denoise<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_g, d_b, width,
			numElements);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaError_t cudaError = cudaGetLastError();

	// Copy the device result matrix in device memory to the host result matrix
	cudaMemcpy(h_r, d_r, numElements * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g, d_g, numElements * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_b, numElements * sizeof(int), cudaMemcpyDeviceToHost);

	if (cudaError != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError,
				cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}
	if (DEBUG) {
		printf("result is:\n");
		printMatrix(numElements, width, h_r);
		printf("\n");
	}
	printf("GPU time to denoise the image %f ms\n", milliseconds);

	cudaFree(d_r);
	cudaFree(d_g);
	cudaFree(d_b);
	free(h_r);
	free(h_g);
	free(h_b);

	printf("All done\n");
	exit(0);
}


