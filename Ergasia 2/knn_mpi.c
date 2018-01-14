#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <mpi.h>

//Structure for knn algorith result
struct knnRes{
	//Array of the indexes of the k nearest neighbors of each point
	double** IDX;
	//Array of the distances of the k nearest neighbors of each point
	double** D;
};

typedef struct knnRes knnRes;

void getTaskData(char* const filePath, int** const dim, double*** const data);
knnRes knnSearch(double** X, int* dimX, double** Y, int* dimY, int k);
int mode(int* vector, int length);
void blockingTask();
void nonBlockingTask();
void serializeArray(double** array2D, int* dim, double** array1D);
void deserializeArray(double* array1D, int* dim, double*** array2D);
void validateSourceAndTarget();

int K = 0, source, target;
int SelfTID, NumTasks;

int main(int argc , char** argv){
	clock_t tic, toc;
	//Get k from user
	if (argc < 2){
		printf("Too few arguments (specify k)\n");
		return -1;
	}
	else
		K = atoi(argv[1]);

	//Mpi initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

	tic = clock();
	//Call of the blocking method
	target = SelfTID + 1;
	source = SelfTID;
	validateSourceAndTarget();
	blockingTask();
	MPI_Barrier(MPI_COMM_WORLD);
	toc = clock();
	if (SelfTID == 0)
		printf("(Blocking) Total time: %.2f secs\n\n", (float) (toc - tic) / CLOCKS_PER_SEC);

	tic = clock();
	//Call of the non-blocking method
	nonBlockingTask();
	MPI_Barrier(MPI_COMM_WORLD);
	toc = clock();
	if (SelfTID == 0)
		printf("(Non-Blocking) Total time: %.2f secs\n\n\n", (float) (toc - tic) / CLOCKS_PER_SEC);

	MPI_Finalize();
	return 0;
}

/*
 * In the blocking method the messages are sent and received with the MPI_Sendrecv function.
*/
void blockingTask(){
	//Needed variables declaration
	int* dim = NULL, * dimL = NULL, ** Lnn, * Mnn;
	double** X = NULL, ** Xr, * X_MPI, * Xr_MPI;
	double** L = NULL, **Lr, * L_MPI, * Lr_MPI;
	clock_t tic, toc, task_time = 0, max_time = 0;
	MPI_Status mpistat;
	knnRes res, partialRes;

	//Get the block of the points for the current task only
	getTaskData("data/train_X.txt", &dim, &X);

	if (K >= dim[0]){
		if(SelfTID == 0)
			printf("Please provide a k not larger or equal to the maximum number of points per block.\n");
		exit(3);
	}

	//Perform the first knn search to find the k nearest neighbors of the same block
	tic = clock();
	res = knnSearch(X, dim, X, dim, K + 1);
	toc = clock();
	task_time += toc - tic;

	//Convert the X block array from 2D to 1D in order to send it via mpi
	serializeArray(X, dim, &X_MPI);
	Xr_MPI = calloc(dim[0] * dim[1], sizeof(double));

	//Initialize source and target variables according to the current task id
	source = SelfTID - 1;
	target = SelfTID + 1;
	for (int t = 0; t < NumTasks; target++, source--, t++){
		validateSourceAndTarget();

		if (target == SelfTID || source == SelfTID)
			continue;

		//In each iteration send the current task's block to target task and receive a new block from the source task
		//The MPI_Sendrecv function sends a message and posts a receive before blocking. This way a deadlock is avoided
		MPI_Sendrecv(X_MPI, dim[0] * dim[1], MPI_DOUBLE, target, t,
				Xr_MPI, dim[0] * dim[1], MPI_DOUBLE, source, t, MPI_COMM_WORLD, &mpistat);

		//Convert the received block from 1D to 2D array
		deserializeArray(Xr_MPI, dim, &Xr);

		//Perform knn search with the received block
		tic = clock();
		partialRes = knnSearch(X, dim, Xr, dim, K + 1);
		toc = clock();
		task_time += toc - tic;

		//Update the knn's result
		for (int i = 0; i < dim[0]; i++){
			double* idxRow = malloc((K + 1) * sizeof(double));
			double* distRow = malloc((K+ 1) * sizeof(double));

			//Both the res.D and the partialRes.D are already sorted in ascending order.
			//So only take the first k + 1 smallest distances from both arrays and store them in
			//a temporary array. When done change the new rows with the ones in res structure.
			for (int k = 0, a = 0, b = 0; k < K + 1; k++){
				if (partialRes.D[i][a] < res.D[i][b]){
					idxRow[k] = partialRes.IDX[i][a];
					distRow[k] = partialRes.D[i][a];
					a++;
				}
				else{
					idxRow[k] = res.IDX[i][b];
					distRow[k] = res.D[i][b];
					b++;
				}
			}

			//Release unneeded memory and update content inside res
			free(res.D[i]);
			free(res.IDX[i]);
			res.D[i] = distRow;
			res.IDX[i] = idxRow;

			free(partialRes.D[i]);
			free(partialRes.IDX[i]);
		}

		for (int i = 0; i < dim[0]; i++)
			free(Xr[i]);
	}

	//Release unneeded memory
	for (int i = 0; i < dim[0]; i++)
		free(X[i]);
	free(X_MPI);
	free(Xr_MPI);

	//Get the labels of the points for the current task only
	getTaskData("data/train_labels.txt", &dimL, &L);

	//Allocate memory
	Lnn = calloc(dimL[0], sizeof(double*));
	for (int i = 0; i < dimL[0]; i++)
		Lnn[i] = calloc(K, sizeof(double));

	//Concentrate neighbors' labels for each point in the current task to Lnn array
	for(int i = 0; i < dimL[0]; i++){
		for(int j = 1, l_index; j < K+1; j++){
			l_index = (int) res.IDX[i][j];

			if (l_index >= dimL[0] * SelfTID && l_index < dimL[0] * (SelfTID + 1))
				Lnn[i][j-1] = (int) L[l_index - dimL[0] * SelfTID][0];
		}
	}

	//Convert Labels array from 2D to 1D in order to comply with mpi sending
	serializeArray(L, dimL, &L_MPI);
	Lr_MPI = calloc(dimL[0] * dimL[1], sizeof(double));

	//Initialize target and source variables
	target = SelfTID + 1;
	source = SelfTID - 1;
	for (int t = 0; t < NumTasks; target++, source--, t++){
		validateSourceAndTarget();

		if (target == SelfTID || source == SelfTID)
			continue;

		//In each iteration send the current task's labels to target task and receive new labels from the source task
		//The MPI_Sendrecv function sends a message and posts a receive before blocking. This way a deadlock is avoided
		MPI_Sendrecv(L_MPI, dimL[0] * dimL[1], MPI_DOUBLE, target, t,
				Lr_MPI, dimL[0] * dimL[1], MPI_DOUBLE, source, t, MPI_COMM_WORLD, &mpistat);

		//Convert the received labels array from 1D to 2D
		deserializeArray(Lr_MPI, dimL, &Lr);

		//Same as before, concentrate neighbors' labels for each point for each task to Lnn array
		for(int i = 0; i < dimL[0]; i++){
			for(int j = 1, l_index; j < K+1; j++){
				l_index = (int) res.IDX[i][j];

				if (l_index >= dimL[0] * source && l_index < dimL[0] * (source + 1))
					Lnn[i][j-1] = (int) Lr[l_index - dimL[0] * source][0];
			}
		}
		for (int i = 0; i < dimL[0]; i++)
			free(Lr[i]);
	}

	//Free up unneeded memory
	free(L_MPI);
	free(Lr_MPI);

	Mnn = calloc(dimL[0], sizeof(int));

	int partialMatches = 0, totalMatches = 0;
	for(int i = 0; i < dimL[0]; i++){
		//Set the label for each point of the current task to the most frequent one according to point's
		//nearest neighbors
		Mnn[i] = mode(Lnn[i], K);
		//Count how many of the labels calculated agree with the original ones
		partialMatches += Mnn[i] == L[i][0];
	}

	//Sum all the matches per block to find the total match percentage
	MPI_Reduce(&partialMatches, &totalMatches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	//Get the maximum time knn algorith needed across all tasks
	MPI_Reduce(&task_time, &max_time, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

	//Print the results
	if (SelfTID == 0){
		printf("(Blocking) Match percentage: %.2lf%%\n", 100.0 * totalMatches / (dim[0] * NumTasks));
		printf("(Blocking) Maximum knn time: %.2f secs\n", (float) max_time / CLOCKS_PER_SEC);
	}

	//Release memory
	for (int i = 0; i < dim[0]; i++){
		free(L[i]);
		free(res.D[i]);
		free(res.IDX[i]);
	}
}

/*
 * In the non-blocking method the messages are sent and received with the
 * MPI_Isend() and MPI_Irecv() functions correspondingly.
*/
void nonBlockingTask(){
	//Needed variables declaration
	int* dim = NULL, * dimL = NULL, ** Lnn, * Mnn;
	double** X = NULL, ** Xr, * X_MPI, ** Xr_MPI;
	double** L = NULL, **Lr, * L_MPI, ** Lr_MPI;
	clock_t tic, toc, task_time = 0, max_time = 0;
	MPI_Status mpistat;
	MPI_Request sendXreqs[NumTasks - 1], recvXreqs[NumTasks - 1];
	MPI_Request sendLreqs[NumTasks - 1], recvLreqs[NumTasks - 1];
	knnRes res, partialRes;

	//Get the block of the points and their labels for the current task only
	getTaskData("data/train_X.txt", &dim, &X);
	getTaskData("data/train_labels.txt", &dimL, &L);

	if (K >= dim[0]){
		if(SelfTID == 0)
			printf("Please provide a k not larger or equal to the maximum number of points per block.\n");
		exit(3);
	}

	//Convert the X block array from 2D to 1D in order to send it via mpi
	//Allocate memory for each receive buffer for the labels to be received
	serializeArray(X, dim, &X_MPI);
	Xr_MPI = calloc(NumTasks - 1, sizeof(double*));
	for (int r = 0; r < NumTasks - 1; r++)
		Xr_MPI[r] = calloc(dim[0] * dim[1], sizeof(double));

	//Convert the L block array from 2D to 1D in order to send it via mpi
	//Allocate memory for each receive buffer for the labels to be received
	serializeArray(L, dimL, &L_MPI);
	Lr_MPI = calloc(NumTasks - 1, sizeof(double*));
	for (int r = 0; r < NumTasks - 1; r++)
		Lr_MPI[r] = calloc(dim[0] * dim[1], sizeof(double));

	//Initialize source and target variables according to the current task id
	source = SelfTID - 1;
	target = SelfTID + 1;
	for (int r = 0; r < NumTasks - 1; target++, source--, r++){
		validateSourceAndTarget();

		//Send the current task's block without waiting
		MPI_Isend(X_MPI, dim[0] * dim[1], MPI_DOUBLE, target, r, MPI_COMM_WORLD, sendXreqs + r);

		//Send the current task's block labels without waiting
		MPI_Isend(L_MPI, dimL[0] * dimL[1], MPI_DOUBLE, target, r + NumTasks, MPI_COMM_WORLD, sendLreqs + r);
	}

	source = SelfTID;
	target = SelfTID + 1;
	validateSourceAndTarget();
	//Perform the first knn search to find the k nearest neighbors of the same block
	tic = clock();
	res = knnSearch(X, dim, X, dim, K + 1);
	toc = clock();
	task_time += toc - tic;

	//Start receiving after knnSearch in order to hide the network latency
	source = SelfTID - 1;
	target = SelfTID + 1;
	for (int r = 0; r < NumTasks - 1; target++, source--, r++){
		validateSourceAndTarget();

		//Set a receive flag for the blocks to be received from the other tasks
		//and start receiving when something has been sent
		MPI_Irecv(Xr_MPI[r], dim[0] * dim[1], MPI_DOUBLE, source, r, MPI_COMM_WORLD, recvXreqs + r);
	}

	for(int t = 0; t < NumTasks - 1; t++){
		int r = 0;
		//Wait until a block arrives
		MPI_Waitany(NumTasks - 1, recvXreqs, &r, &mpistat);
		source = mpistat.MPI_SOURCE;

		//Convert the received block from 1D to 2D array
		deserializeArray(Xr_MPI[r], dim, &Xr);

		//Perform knn search with the received block
		tic = clock();
		partialRes = knnSearch(X, dim, Xr, dim, K + 1);
		toc = clock();
		task_time += toc - tic;

		//Update the knn's result
		for (int i = 0; i < dim[0]; i++){
			double* idxRow = malloc((K + 1) * sizeof(double));
			double* distRow = malloc((K+ 1) * sizeof(double));

			//Both the res.D and the partialRes.D are already sorted in ascending order.
			//So only take the first k + 1 smallest distances from both arrays and store them in
			//a temporary array. When done change the new rows with the ones in res structure.
			for (int k = 0, a = 0, b = 0; k < K + 1; k++){
				if (partialRes.D[i][a] < res.D[i][b]){
					idxRow[k] = partialRes.IDX[i][a];
					distRow[k] = partialRes.D[i][a];
					a++;
				}
				else{
					idxRow[k] = res.IDX[i][b];
					distRow[k] = res.D[i][b];
					b++;
				}
			}

			//Release unneeded memory and update content inside res
			free(res.D[i]);
			free(res.IDX[i]);
			res.D[i] = distRow;
			res.IDX[i] = idxRow;

			free(partialRes.D[i]);
			free(partialRes.IDX[i]);
		}

		for (int i = 0; i < dim[0]; i++)
			free(Xr[i]);
	}

	//Release unneeded memory
	for (int i = 0; i < dim[0]; i++)
		free(X[i]);
	free(X_MPI);
	for (int r = 0; r < NumTasks - 1; r++){
		free(Xr_MPI[r]);
		//Release the mpi send requests
		MPI_Request_free(sendXreqs + r);

		//Start receiving labels
		MPI_Irecv(Lr_MPI[r], dimL[0] * dimL[1], MPI_DOUBLE, MPI_ANY_SOURCE, r + NumTasks, MPI_COMM_WORLD, recvLreqs + r);
	}

	//Allocate memory
	Lnn = calloc(dimL[0], sizeof(double*));
	for (int i = 0; i < dimL[0]; i++)
		Lnn[i] = calloc(K, sizeof(double));

	//Concentrate neighbors' labels for each point in the current task to Lnn array
	for(int i = 0; i < dimL[0]; i++){
		for(int j = 1, l_index; j < K+1; j++){
			l_index = (int) res.IDX[i][j];

			if (l_index >= dimL[0] * SelfTID && l_index < dimL[0] * (SelfTID + 1))
				Lnn[i][j-1] = (int) L[l_index - dimL[0] * SelfTID][0];
		}
	}

	//Set a barrier for all tasks to get synchronised before they start determing
	//points' labels
	MPI_Barrier(MPI_COMM_WORLD);

	for(int t = 0; t < NumTasks - 1; t++){
		int r = 0;
		//Wait until a block of labels arrives
		MPI_Waitany(NumTasks - 1, recvLreqs, &r, &mpistat);
		source = mpistat.MPI_SOURCE;

		//Convert the received labels array from 1D to 2D
		deserializeArray(Lr_MPI[r], dimL, &Lr);

		//Same as before, concentrate neighbors' labels for each point for each task to Lnn array
		for(int i = 0; i < dimL[0]; i++){
			for(int j = 1, l_index; j < K+1; j++){
				l_index = (int) res.IDX[i][j];

				if (l_index >= dimL[0] * source && l_index < dimL[0] * (source + 1))
					Lnn[i][j-1] = (int) Lr[l_index - dimL[0] * source][0];
			}
		}

		for (int i = 0; i < dimL[0]; i++)
			free(Lr[i]);
	}

	//Free up unneeded memory and release the send requests for the labels
	free(L_MPI);
	for (int r = 0; r < NumTasks - 1; r++){
		free(Lr_MPI[r]);
		MPI_Request_free(sendLreqs + r);
	}

	Mnn = calloc(dimL[0], sizeof(int));

	int partialMatches = 0, totalMatches = 0;
	for(int i = 0; i < dimL[0]; i++){
		//Set the label for each point of the current task to the most frequent one according to point's
		//nearest neighbors
		Mnn[i] = mode(Lnn[i], K);
		//Count how many of the labels calculated agree with the original ones
		partialMatches += Mnn[i] == L[i][0];
	}

	//Sum all the matches per block to find the total match percentage
	MPI_Reduce(&partialMatches, &totalMatches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	//Get the maximum time knn algorith needed across all tasks
	MPI_Reduce(&task_time, &max_time, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

	//Print the results
	if (SelfTID == 0){
		printf("(Non-Blocking) Match percentage: %.2lf%%\n", 100.0 * totalMatches / (dim[0] * NumTasks));
		printf("(Non-Blocking) Maximum knn time: %.2f secs\n", (float) max_time / CLOCKS_PER_SEC);
	}

	//Release memory
	for (int i = 0; i < dim[0]; i++){
		free(L[i]);
		free(res.D[i]);
		free(res.IDX[i]);
	}
}

/*
 * Algorith to find the k nearest neighbors of Y in X.
 */
knnRes knnSearch(double** Y, int* dimY, double** X, int* dimX, int k){
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
				result.IDX[i][d] = idx + dimX[0] * source;
			}
		}
	}

	return result;
}

/*
 * It reads the part of the knn data file for the current task
 */
void getTaskData(char* const filePath, int** const dim, double*** const data){
	FILE* fp;
	*dim = calloc(2, sizeof(int));
	char* line = NULL;
	size_t len = 0;
	size_t read;

	fp = fopen(filePath, "r");

	read = getline(&line, &len, fp);
	(*dim)[0] = atof(strtok(line, "\t ")) / NumTasks;
	(*dim)[1] = atoi(strtok(NULL, "\t "));

	if (SelfTID == NumTasks - 1)
		(*dim)[0] += atoi(strtok(line, "\t ")) % NumTasks;


	*data = (double**) calloc((*dim)[0], sizeof(double*));
	for (int i = 0; i < (*dim)[0]; i++)
		(*data)[i] = (double*) calloc((*dim)[1], sizeof(double));

	char * token = NULL;
	for (int i = 0; i < (*dim)[0] * SelfTID; i++)
		read = getline(&line, &len, fp);

	for (int i = 0; i < (*dim)[0]; i++){
		if ((read = getline(&line, &len, fp)) == -1)
			exit(1);

		(*data)[i][0] = atof(strtok(line, "\t "));
		for (int j = 1; j < (*dim)[1]; j++){
			if (!(token = strtok(NULL, "\t ")))
				break;

			(*data)[i][j] = atof(token);
		}
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

/*
 * Converts a 2D array to an 1D one
 */
void serializeArray(double** array2D, int* dim, double** array1D){
	*array1D = malloc(dim[0] * dim[1] * sizeof(double));
	for (int i = 0; i < dim[0]; i++)
		memcpy((*array1D) + dim[1] * i, array2D[i], dim[1] * sizeof(double));
}

/*
 * Converts an 1D array to a 2D one
 */
void deserializeArray(double* array1D, int* dim, double*** array2D){
	*array2D = malloc(dim[0] * sizeof(double*));
	for (int i = 0; i < dim[0]; i++){
		(*array2D)[i] = malloc(dim[1] * sizeof(double));
		memcpy((*array2D)[i], array1D + dim[1] * i, dim[1] * sizeof(double));
	}
}

/*
 * Ensures that the target and source variables are inside the right
 * boundaries for mpi send and receive operations
 */
void validateSourceAndTarget(){
	if (target >= NumTasks)
		target = 0;
	if (source < 0)
		source = NumTasks - 1;
}


