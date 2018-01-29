#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#define epsilon 0.00015
#define sigma 1
#define BLOCK_SIZE 10

void getData(char* const filePath, int** const dim, float*** const data);
__global__ void cudaMeanshift1(float* y, float* x, float* denom, float* numer, dim3 dim);
__global__ void newYvCalculation(float* numer, float* denom, float* y, float* meanshift, dim3 dim);
__device__ __noinline__ float gaussian(float x, float h);


int main(int argc , char** argv){
	//Host and device variables and matrices
	float ** X;
	float * dXv, * dYv;
	float * dNumerator, * dDenominator;
	float meanshift, * dMeanshift;
	float cudaTotalTime = 0.0f, cudaIterTime;
	int* dim, iter = 0;
	//tic, toc used for overall program time and dTic, dToc for total device time
	clock_t tic, toc;
	cudaEvent_t dTic, dToc;
	cudaEventCreate(&dTic);
	cudaEventCreate(&dToc);

	if (argc < 2){
		printf("Please provide a path for the data.");
		return 1;
	}

	tic = clock();

	//Read data file for X matrix
	getData(argv[1], &dim, &X);

	//Device memory allocation
	cudaMalloc(&dXv, dim[0] * dim[1] * sizeof(float));
	cudaMalloc(&dYv, dim[0] * dim[1] * sizeof(float));

	cudaMalloc(&dNumerator, dim[0] * dim[1] * sizeof(float));
	cudaMalloc(&dDenominator, dim[0] * sizeof(float));
	cudaMalloc(&dMeanshift, sizeof(float));

	//Copy memory from host to device and release it from host
	for(int i = 0; i < dim[0]; i++){
		cudaMemcpy(dXv + i * dim[1], X[i], dim[1] * sizeof(float), cudaMemcpyHostToDevice);
		free(X[i]);
	}
	free(X);
	//Initiallization of matrix Y = X
	cudaMemcpy(dYv, dXv, dim[0] * dim[1] * sizeof(float), cudaMemcpyDeviceToDevice);

	do {
		iter++;

		//Set numerator, denominator and meanshift to zero since they are calculated as sums
		cudaMemset(dNumerator, 0, dim[0] * dim[1] * sizeof(float));
		cudaMemset(dDenominator, 0, dim[0] * sizeof(float));
		cudaMemset(dMeanshift, 0, sizeof(float));

		//For the first kernel we need N x N threads in total
		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 numBlocks((int) ceilf((float) dim[0] / threadsPerBlock.x), (int) ceil((float) dim[0] / threadsPerBlock.y));
		dim3 dimensions(dim[0], dim[0], dim[1]); //N x N x d

		cudaEventRecord(dTic);

		//Meanshift first phase of parallelization
		cudaMeanshift1<<<numBlocks, threadsPerBlock, threadsPerBlock.x * threadsPerBlock.y * dimensions.z * sizeof(float)>>>
				(dYv, dXv, dDenominator, dNumerator, dimensions);
		//Make sure every thread is over before continueing
		cudaDeviceSynchronize();

		//For the second kernel we need N x d threads in total
		numBlocks.y = (int) ceilf((float) dim[1] / threadsPerBlock.y);
		dim3 YDim(dim[0], dim[1]);
		//Meanshift second phase of parallelization
		newYvCalculation<<<numBlocks, threadsPerBlock>>>(dNumerator, dDenominator, dYv, dMeanshift, YDim);
		//Make sure every thread is over before going on to the next iteration
		cudaDeviceSynchronize();

		cudaEventRecord(dToc);
		cudaEventSynchronize(dToc);
		cudaEventElapsedTime(&cudaIterTime, dTic, dToc);
		cudaTotalTime += cudaIterTime;

		//Copy the squared error calculated from device to host
		cudaMemcpy(&meanshift, dMeanshift, sizeof(float), cudaMemcpyDeviceToHost);
		//The error in the loop condition: ||m(y)||
		meanshift = sqrtf(meanshift);

		printf("Iteration %3d error: %f\n", iter, meanshift);
	} while(meanshift > epsilon);
	toc = clock();

	printf("\nRan %d iterations with error %f\n", iter, meanshift);
	printf("Total time (Host & Device): %f secs\n", (float) (toc - tic) / CLOCKS_PER_SEC);
	printf("Total calculation time (Device): %f secs\n", cudaTotalTime / 1000.0f);
	printf("Total memcpy time (approximately): %lf secs\n", (float) (toc - tic) / CLOCKS_PER_SEC - cudaTotalTime / 1000.0f);

	//Memory deallocation
	cudaFree(dXv);
	cudaFree(dYv);
	cudaFree(dNumerator);
	cudaFree(dDenominator);
	cudaFree(dMeanshift);
	free(dim);

	return 0;
}

/*
 * Calculates the denominator and numerator of the new Y. Every thread is matched to a certain pair of i and j.
 * Uses the minimum possible ammount of local thread memory (registers).
 * Shared memory is used for the parts of the X matrix that correspond to each thread of a block.
 * Commented code uses shared memory for all matrices, but is much slower and needs too muc memory.
 */
//Shared memory
extern __shared__ float data[];
__global__ void cudaMeanshift1(float* y, float* x, float* denom, float* numer, dim3 dim){
	//Find indexes of Z, X and Y matrices
	unsigned short sharedMemLength = blockDim.x * blockDim.y * dim.z;
	unsigned short colZGlobal = (blockDim.x * blockIdx.x + threadIdx.x);
	unsigned short rowZGlobal = (blockDim.y * blockIdx.y + threadIdx.y);
	unsigned short rowXShared = (colZGlobal * dim.z) % sharedMemLength;
	//Squared distance of y[i] from x[j]
	float difYiXj = 0.0f;

	float* sharedX = data;

	/*float* sharedY = data + threadsNum;
	float* sharedDenom = data + threadsNum * 2;
	if (threadIdx.x == 0 && threadIdx.y == 0)
		memset(sharedDenom, 0, threadsNum * sizeof(float));
	__syncthreads();*/

	//Check if the indexes are valid (some threads will do nothing because of the asymmetry between block size
	//and data set size
	if (colZGlobal < dim.x && rowZGlobal < dim.y){
		//Calculate difYiXj
		for (short i = 0; i < dim.z; i++){
			sharedX[rowXShared + i] = x[colZGlobal * dim.z + i];

			/*sharedY[rowYShared + i] = y[rowZGlobal * dim.z + i];
			__syncthreads();*/

			difYiXj += powf(y[rowZGlobal * dim.z + i] - sharedX[rowXShared + i], 2.0f);
		}

		//If the distance between y[i] and x[j] is less than σ^2, calculate the contribution of current
		//thread in numerator and denominator according to the formula used
		if (sqrtf(difYiXj) <= sigma * sigma){
			float gauss = gaussian(difYiXj, sigma);
			/* atomicAdd(sharedDenom + rowZShared, gauss); */
			//We don't want two threads to access a position in memory simultaneously (race condition),
			//so we need to make sure atomic operations are used for each action
			atomicAdd(denom + rowZGlobal, gauss);
			for (short i = 0; i < dim.z; i++){
				atomicAdd(numer + rowZGlobal * dim.z + i, gauss * sharedX[rowXShared + i]);
				/* atomicAdd(sharedNumer + rowZShared * dim.z + i, gauss * sharedX[rowXShared + i]); */
			}
		}

		/*__syncthreads();
		if (colZGlobal % blockDim.y == 0){
			atomicAdd(denom + rowZGlobal, sharedDenom[rowZShared]);
		}*/
	}
}

/*
 * Calculates the new values of Y matrix and the meanshift error.
 * Uses the minimum possible ammount of local thread memory (registers).
 */
__global__ void newYvCalculation(float* numer, float* denom, float* y, float* meanshift, dim3 dim){
	//Find indexes of Y matrix
	unsigned short row = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned short col = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned short indexY = row * dim.y + col;
	//Used in order to avoid the global storage of both the old and new value of Y
	float temp;

	//Check if the indexes are valid (some threads will do nothing because of the asymmetry between block size
	//and data set size
	if (row < dim.x && col < dim.y){
		//If denominator is zero, then the numerator will be zero, too. So make denominator equals one to avoid
		//a division with zero.
		temp = numer[indexY] / (!denom[row] ? 1.0f : denom[row]);
		//Again make sure no race condition occurs
		atomicAdd(meanshift, powf(temp - y[indexY], 2.0f));
		//Store new value of Y
		y[indexY] = temp;
	}
}

/*
 * Calculates the gaussian value of for h.
 */
__device__ __noinline__ float gaussian(float x, float h){
	return expf(-x / (2 * h * h));
}


//Reads a 2-D array from a text file in filePath and copies the result to data
void getData(char* const filePath, int** const dim, float*** const data){
	FILE* fp;
	*dim = (int*) calloc(2, sizeof(int));
	char* line = NULL;
	size_t len = 0;

	fp = fopen(filePath, "r");

	getline(&line, &len, fp);
	(*dim)[0] = atoi(strtok(line, "\t "));
	(*dim)[1] = atoi(strtok(NULL, "\t "));

	*data = (float**) calloc((*dim)[0], sizeof(float*));
	for (int i = 0; i < (*dim)[0]; i++)
		(*data)[i] = (float*) calloc((*dim)[1], sizeof(float));

	int i = 0, j = 0;
	char * token = NULL;
	while(getline(&line, &len, fp) != -1){
		(*data)[i][j++] = atof(strtok(line, "\t "));
		while((token = strtok(NULL, "\t ")))
			(*data)[i][j++] = atof(token);
		i++;
		j = 0;
	}

	fclose(fp);
	return;
}
