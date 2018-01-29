#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define epsilon 0.0001
#define sigma 1

void getData(char* const filePath, int** const dim, double*** const data);
double gaussian(double x, double h);
double norm2(double* vector, int length);
void vectorDif(double* a, double* b, double* c, int length);
void vectorSum(double* a, double* b, double* c, int length);
void vectorScalarMult(double* v, double* u, int length, double s);

int main(int argc , char** argv){
	//Needed matrices for meanshift
	//mV is a vector representaion of the m array
	double** X, ** Y, **new_Y, * mV;
	int* dim, iter = 0;
	double meanshift = DBL_MAX;
	clock_t tic, toc;

	if (argc < 2){
		printf("Please provide a path for the data.");
		return 1;
	}

	//Read data file for X matrix
	getData(argv[1], &dim, &X);

	//Initiallization of matrices - Y (=X)
	Y = calloc(dim[0], sizeof(double*));
	new_Y = calloc(dim[0], sizeof(double*));
	mV = calloc(dim[0] * dim[1], sizeof(double));
	for (int i = 0; i < dim[0]; i++){
		Y[i] = calloc(dim[1], sizeof(double));
		new_Y[i] = calloc(dim[1], sizeof(double));

		memcpy(Y[i], X[i], dim[1] * sizeof(double));
	}

	//Allocate memory for temporary vectors needed in new_Y calculation
	double coef, distYiXj, denominator = 0.0;;
	double* vec_dif = calloc(dim[1], sizeof(double));
	double * vec_mult = calloc(dim[1], sizeof(double));
	double* numerator = calloc(dim[1], sizeof(double));

	tic = clock();
	while(meanshift > epsilon){
		iter++;

		for (int i = 0; i < dim[0]; i++){
			
			//Set sum variables to 0
			for (int p = 0; p < dim[1]; p++){
				numerator[p] = 0.0;
				denominator = 0.0;
			}

			//Implementation of the meanshift formula
			for (int j = 0; j < dim[0]; j++){
				//vec_diff = Y[i] - X[j]
				vectorDif(Y[i], X[j], vec_dif, dim[1]);
				//distYiXj = ||vec_dil||
				distYiXj = norm2(vec_dif, dim[1]);

				if (distYiXj <= sigma * sigma){
					//coef = k(distYiXj ^ 2)
					coef = gaussian(distYiXj * distYiXj, sigma);
					//vec_mult = coef * X[j]
					vectorScalarMult(X[j], vec_mult, dim[1], coef);
					//numerator += vec_mult
					vectorSum(numerator, vec_mult, numerator, dim[1]);

					denominator += coef;
				}

			}
			//new_Y[i] = numerator / denominator
			//If denominator is zero, numerator will be (0, 0), too.
			//So we would have new_Y[i] = (0, 0) * 1 / 1 = 0. This way a possible
			//division with zero is avoided
			vectorScalarMult(numerator, new_Y[i], dim[1], 1.0 / (!denominator ? 1.0 : denominator));
			//m[i] = new_Y[i] - Y[i] (m[i] = mV[i*2])
			vectorDif(new_Y[i], Y[i], mV + i*dim[1], dim[1]);

			//Y[i] = new_Y[i]
			for (int p = 0; p < dim[1]; p++)
				Y[i][p] = new_Y[i][p];
		}

		//meanshift = ||m||
		//The Frobenius norm is used to calculate the error, instead of the induced one.
		//The result will be more accurate, since Frobenius norm is always greater than
		//the induced one for a certain matrix.
		meanshift = norm2(mV, dim[0] * dim[1]);

		printf("Iteration %3d error: %lf\n", iter, meanshift);
	}
	toc = clock();

	printf("\nRan %d iterations with error %lf in %f secs\n", iter, meanshift, (float) (toc - tic) / CLOCKS_PER_SEC);

	//Memory deallocation
	for (int i = 0; i < dim[0]; i++){
		free(X[i]);
		free(Y[i]);
		free(new_Y[i]);
	}
	free(X);
	free(Y);
	free(new_Y);
	free(mV);
	free(vec_dif);
	free(vec_mult);
	free(numerator);
	
	return 0;
}

//Multiplies a vector with a scalar value
//u = v * s
inline void vectorScalarMult(double* v, double* u, int length, double s){
	for (int i = 0; i < length; i++)
		u[i] = s * v[i];

	return;
}

//Sums two vectors
//c = a + b
inline void vectorSum(double* a, double* b, double* c, int length){
	for (int i = 0; i < length; i++)
		c[i] = a[i] + b[i];
	
	return;
}

//Substracts two vectors
//c = a - b
inline void vectorDif(double* a, double* b, double* c, int length){
	for (int i = 0; i < length; i++)
		c[i] = a[i] - b[i];
	
	return;
}

//Calculates the gaussian of x with sigma = h
inline double gaussian(double x, double h){
	return exp(-x / (2 * h * h));
}

//Calculates the norm 2 of a vector
inline double norm2(double* vector, int length){
	double squareSum = 0.0;
	
	for (int i = 0; i < length; i++)
		squareSum += vector[i] * vector[i];

	return sqrt(squareSum);
}

//Reads a 2-D array from a text file in filePath and copies the result to data
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
		while((token = strtok(NULL, "\t ")))
			(*data)[i][j++] = atof(token);
		i++;
		j = 0;
	}

	fclose(fp);
	return;
}
