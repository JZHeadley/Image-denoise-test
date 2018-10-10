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
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

#define DEBUG 0

__global__ void denoise(char *r, char*g, char*b, int width, int numElements) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	int column = tid % width;
	int column = tid - ((tid / width) * width); // should be a mod operator
	int row = tid / width;
//	int row = blockDim.x * blockIdx.x + threadIdx.x;
//	int column = blockDim.x * blockIdx.x + threadIdx.y;
	int height = numElements / width;
	int sumR = 0, countR = 0, sumG = 0, countG = 0, sumB = 0, countB = 0;
	for (int i = row - 1; i <= row + 1; i++) {
		for (int j = column - 1; j <= column + 1; j++) {

			if (i > 0 && i < height && j > 0 && j < width) {
				sumR += r[i * width + j];
				countR++;
				sumG += g[i * width + j];
				countG++;
				sumB += b[i * width + j];
				countB++;
			}
		}
	}
	r[row * width + column] = (sumR / countR);
	g[row * width + column] = (sumG / countG);
	b[row * width + column] = (sumB / countB);

}

void printMatrix(int numElements, int width, char* matrix) {

	for (int i = 0; i < numElements; ++i) {
		if (i % width == 0 && i != 0) {
			printf("\n");
		}
		printf("%i\t", matrix[i]);
	}
}

int main(void) {

	time_t t;
	srand((unsigned) time(&t));
	String imageName("../src/noisy-man.png");
	Mat3b image = imread(imageName, IMREAD_COLOR);

	if (!image.data) {
		cout << "Could not open or find the image" << endl;
	}
	int width = image.cols;
	int numElements = image.total();
	printf("There are %i pixels in the image\n", numElements);
	// Allocate host memory
	vector<char> blue;
	vector<char> green;
	vector<char> red;
	Vec3b pixel;

	for (int i = 0; i < numElements; ++i) {
		pixel = image(i);
		blue.push_back(pixel.val[0]);
		green.push_back(pixel.val[1]);
		red.push_back(pixel.val[2]);
	}
	char*h_r = &red[0];
	char*h_g = &green[0];
	char*h_b = &blue[0];

	// Allocate the device arrays of 'colors'
	char *d_r, *d_g, *d_b;
	cudaMalloc(&d_r, red.size() * sizeof(char));
	cudaMalloc(&d_g, green.size() * sizeof(char));
	cudaMalloc(&d_b, blue.size() * sizeof(char));

	// Copy the host input vectors A and B in host memory to the device input vectors in
	cudaMemcpy(d_r, h_r, red.size() * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_g, h_g, green.size() * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, blue.size() * sizeof(char), cudaMemcpyHostToDevice);

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
//	dim3 threadsPerBlock(width, width);
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
//	dim3 blocksPerGrid(width / threadsPerBlock.x, width / threadsPerBlock.y);
	cudaEventRecord(start);

	denoise<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_g, d_b, width,
			numElements);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaError_t cudaError = cudaGetLastError();

// Copy the device result matrix in device memory to the host result matrix
	cudaMemcpy(h_r, d_r, red.size() * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_g, d_g, green.size() * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_b, blue.size() * sizeof(char), cudaMemcpyDeviceToHost);

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
	// reassembling the image
	Vec3b newPixel;
	for (int i = 0; i < numElements; ++i) {
		newPixel = image(i);
		newPixel[0] = h_b[i];
		newPixel[1] = h_g[i];
		newPixel[2] = h_r[i];
		image.at<Vec3b>(i) = newPixel;
	}

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	try {
		imwrite("denoised.png", image, compression_params);
	} catch (exception& ex) {
		fprintf(stderr, "Exception converting image ot PNG format: %s\n",
				ex.what());
		return 1;
	}
	cudaFree(d_r);
	cudaFree(d_g);
	cudaFree(d_b);
	free(h_r);
	free(h_g);
	free(h_b);

	printf("All done\n");
	exit(0);
}


