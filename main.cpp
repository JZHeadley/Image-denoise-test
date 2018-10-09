#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>

int *r;
int *g;
int *b;
int *rr;
int *gr;
int *br;
int width;
int numElements;

void denoise(int *r, int *g, int *b, int *rr, int *rg, int *rb, int width, int numElements)
{
    for (int i = 0; i < numElements; i++)
    {
        if (i % width == 0 || i % width == (width - 1) || i < width || i > numElements - width)
        {
            // this if tells us if we're at any edge of the matrix
            // skipping these so we don't go out of bounds with our array
            // set the edge to its original value
            rr[i] = r[i];
            rg[i] = g[i];
            rb[i] = b[i];
            continue;
        }
        rr[i] = ((r[i] + r[i - width] + r[i + width] + r[i - 1] + r[i + 1]) / 5);
        rg[i] = ((g[i] + g[i - width] + g[i + width] + g[i - 1] + g[i + 1]) / 5);
        rb[i] = ((b[i] + b[i - width] + b[i + width] + b[i - 1] + b[i + 1]) / 5);
    }
}

void printMatrix(int numElements, int width, int *matrix)
{

    for (int i = 0; i < numElements; ++i)
    {
        if (i % width == 0 && i != 0)
        {
            printf("\n");
        }
        printf("%i\t", matrix[i]);
    }
}

void *threadDenoise(void *args)
{
    int threadId = *((int *)args);
    if (threadId % width == 0)
    { // left edge
        if (threadId == 0)
        {
            // top left corner
            rr[threadId] = (r[threadId] + r[threadId + 1] + r[threadId + width]) / 3;
            return (void *)NULL;
        }
        else if (threadId == (numElements - width))
        { // bottom left corner
            rr[threadId] = (r[threadId] + r[threadId - width] + r[threadId + 1]) / 3;
            return (void *)NULL;
        }
        rr[threadId] = (r[threadId] + r[threadId - width] + r[threadId + width] + r[threadId + 1]) / 4;
        return (void *)NULL;
    }
    if (threadId < width)
    {                                // first row
        if (threadId == (width - 1)) //top right corner
        {
            rr[threadId] = (r[threadId] + r[threadId - 1] + r[threadId + width]) / 3;
            return (void *)NULL;
        }
        rr[threadId] = (r[threadId] + r[threadId + 1] + r[threadId - 1] + r[threadId + width]) / 4;
        return (void *)NULL;
    }
    if (threadId % width == (width - 1)) // right edge
    {
        if (threadId == numElements - 1)
        {
            //bottom right corner
            rr[threadId] = (r[threadId] + r[threadId - 1] + r[threadId - width]) / 3;
            return (void *)NULL;
        }
        rr[threadId] = (r[threadId] + r[threadId - width] + r[threadId + width] + r[threadId - 1]) / 4;
        return (void *)NULL;
    }

    if (threadId > numElements - width) // bottom
    {
        //bottom middle
        rr[threadId] = (r[threadId] + r[threadId - width] + r[threadId + 1] + r[threadId - 1]) / 4;
        return (void *)NULL;
    }
    rr[threadId] = ((r[threadId] + r[threadId - width] + r[threadId + width] + r[threadId - 1] + r[threadId + 1]) / 5);
}

int main(int argc, char *argv[])
{
    int matrixSize = 5;
    width = matrixSize;
    numElements = matrixSize * matrixSize;

    r = (int *)calloc(numElements, sizeof(int));
    g = (int *)calloc(numElements, sizeof(int));
    b = (int *)calloc(numElements, sizeof(int));

    for (int i = 0; i < numElements; ++i)
    {
        r[i] = rand() % 255;
        g[i] = rand() % 255;
        b[i] = rand() % 255;
    }

    rr = (int *)calloc(numElements, sizeof(int));
    gr = (int *)calloc(numElements, sizeof(int));
    br = (int *)calloc(numElements, sizeof(int));
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    // denoise(r, g, b, rr, gr, br, matrixSize, numElements);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t seqDiff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    pthread_t *threads = (pthread_t *)malloc(numElements * sizeof(pthread_t));
    int *threadIds = (int *)malloc(numElements * sizeof(int));

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    for (int i = 0; i < numElements; i++)
        threadIds[i] = i;

    for (int i = 0; i < numElements; i++)
    {
        int status = pthread_create(&threads[i], NULL, threadDenoise, (void *)&threadIds[i]);
    }

    for (int i = 0; i < numElements; i++)
    {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t parDiff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("original is:\n");
    printMatrix(numElements, matrixSize, r);
    printf("\n\n");
    printf("result is:\n");
    printMatrix(numElements, matrixSize, rr);
    printf("\n");

    printf("Took %lu ms to denoise the 'image' sequentially\n", seqDiff);
    printf("Took %lu ms to denoise the 'image' in parallel\n", parDiff);
    free(r);
    free(g);
    free(b);
    free(rr);
    free(gr);
    free(br);
}
