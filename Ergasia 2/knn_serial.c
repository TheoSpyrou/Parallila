#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

//Structure for knn algorith result
struct knnRes{
	//Array of the indexes of the k nearest neighbors of each point
	double** IDX;
	//Array of the distances of the k nearest neighbors of each point
	double** D;
};

typedef struct knnRes knnRes;

void getData(char* const filePath, int** const dim, double*** const data);
knnRes knnSearch(double** X, int* dimX, double** Y, int* dimY, int k);
int mode(int* vector, int length);

int K = 0;

int main(int argc , char** argv){
	//Get k from user
	if (argc < 2){
		printf("Too few arguments (specify k)\n");
		return -1;
	}
	else
		K = atoi(argv[1]);

	int* dim = NULL, * dimL = NULL;
	double** X = NULL, ** L = NULL;
	int LnnRow[K], MnnRow;
	clock_t tic, toc, tic_total, toc_total;

	tic_total = clock();

	//Get data
	getData("data/train_X.txt", &dim, &X);
	getData("data/train_labels.txt", &dimL, &L);

	//Perform knn
	tic = clock();
	knnRes knn = knnSearch(X, dim, X, dim, K + 1);
	toc = clock();

	//Find the match percentage by counting the points their flags were
	//determined correctly
	int matchesCount = 0;
	for(int i = 0; i < dim[0]; i++){
		for(int j = 1; j < K+1; j++)
			LnnRow[j-1] = (int) L[(int)knn.IDX[i][j]][0];

		MnnRow = mode(LnnRow, K);
		matchesCount += MnnRow == L[i][0];
	}

	toc_total = clock();

	//Print the results
	printf("(Serial) Match percentage: %lf%%\n", 100.0 * matchesCount / dim[0]);
	printf("(Serial) Total knn time: %.1f sec\n", (float) (toc - tic) / CLOCKS_PER_SEC);
	printf("(Serial) Total knn time: %.1f sec\n\n\n", (float) (toc_total - tic_total) / CLOCKS_PER_SEC);

	return 0;
}

/*
 * Algorith to find the k nearest neighbors of Y in X.
 */
knnRes knnSearch(double** X, int* dimX, double** Y, int* dimY, int k){
	knnRes result = { .IDX = NULL, .D = NULL };

	if (dimX[1] != dimY[1]){
		printf("Error: X and Y matrices must have same dimension points.\n");
		return result;
	}

	//Allocate memory and initialize variables
	result.IDX = (double**) calloc(dimY[0], sizeof(double*));
	result.D = (double**) calloc(dimY[0], sizeof(double*));
	for (int i = 0; i < dimY[0]; i++){
		result.IDX[i] = (double*) calloc(k, sizeof(double));
		result.D[i] = (double*) calloc(k, sizeof(double));
		for (int j = 0; j < k; j++){
			result.IDX[i][j] = DBL_MAX;
			result.D[i][j] = DBL_MAX;
		}
	}

	//Calculate distances of every point in Y with every point in X
	//Insertion sort
	double dist;
	int d;
	for(int i = 0; i < dimY[0]; i++){
		for(int idx = 0; idx < dimX[0]; idx++){
			dist = 0.0;
			for(int l = 0; l < dimX[1]; l++)
				dist += (Y[i][l] - X[idx][l]) * (Y[i][l] - X[idx][l]);
			dist = sqrt(dist);

			d = 0;
			while(d < k && result.D[i][d] < dist)
				d++;

			if (d < k) {
				for(int m = k-1; m > d; m--){
					result.D[i][m] = result.D[i][m-1];
					result.IDX[i][m] = result.IDX[i][m-1];
				}
				result.D[i][d] = dist;
				result.IDX[i][d] = idx;
			}
		}
	}

	return result;
}

/*
 * It reads the knn data file
 */
void getData(char* const filePath, int** const dim, double*** const data){
	FILE* fp;
	*dim = calloc(2, sizeof(int));
	char* line = NULL;
	size_t len = 0;
	size_t read;

	fp = fopen(filePath, "r");
	
	read = getline(&line, &len, fp);
	(*dim)[0] = atoi(strtok(line, "\t "));
	(*dim)[1] = atoi(strtok(NULL, "\t "));

	*data = (double**) calloc((*dim)[0], sizeof(double*));
	for (int i = 0; i < (*dim)[0]; i++)
		(*data)[i] = (double*) calloc((*dim)[1], sizeof(double));

	int i = 0, j = 0;
	char * token = NULL;
	while((read = getline(&line, &len, fp)) != -1){
		(*data)[i][j++] = atof(strtok(line, "\t "));
		while(token = strtok(NULL, "\t "))
			(*data)[i][j++] = atof(token);
		i++;
		j = 0;
	}

	fclose(fp);
	return;
}

/*
 * Finds the most frequent value in an array
 */
int mode(int* vector, int length){
	int maxValue = -1;
	for(int i = 0; i < length; i++)
		if (vector[i] > maxValue)
			maxValue = vector[i];

	int freq[maxValue];
	int maxFreq = 0, mostFreq = -2;

	for(int i = 0; i < maxValue; i++)
		freq[i] = 0;

	for (int i = 0; i < length; i++)
		freq[vector[i] - 1]++;
	
	for (int i = 0; i < maxValue; i++)
		if (freq[i] > maxFreq){
			maxFreq = freq[i];
			mostFreq = i + 1;
		}

	return mostFreq;
}







