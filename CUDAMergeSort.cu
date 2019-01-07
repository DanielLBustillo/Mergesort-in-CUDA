//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Libreries for use CUDA FRAMEWORK	to run in parallel an implementation of Mergesort algorithm using c language
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#define N1 1
#define N10 10
#define N100 100
#define N1000 1000

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//		Function to combine the sublists generated. We use the flag __device__ because it is called using the function CudaMergeSort which is working on the gpu. 
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

__device__ void CudaMerge(int* values, int* results, int l, int r, int u)
{
	int i, j, k;
	i = l; j = r; k = l;
	while (i < r && j < u) {
		if (values[i] <= values[j]) { results[k] = values[i]; i++; }
		else { results[k] = values[j]; j++; }
		k++;
	}

	while (i < r) {
		results[k] = values[i]; i++; k++;
	}

	while (j < u) {
		results[k] = values[j]; j++; k++;
	}
	for (k = l; k < u; k++) {
		values[k] = results[k];
	}
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//		Function to generate the sublists of the array to sort. It use flag __global__ because the function is called from the main. It is also call Kernel and use a specific call. 
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ static void CudaMergeSort(int * values, int* results, int dim)
{
	extern __shared__ int shared[];



	const unsigned int tid = threadIdx.x;
	int k, u, i;
	shared[tid] = values[tid];


	__syncthreads();
	k = 1;
	while (k <= dim)
	{
		i = 0;
		while (i + k < dim)
		{
			u = i + k * 2;;
			if (u > dim)
			{
				u = dim + 1;
			}
			CudaMerge(shared, results, i, i + k, u);
			i = i + k * 2;
		}
		k = k * 2;

		__syncthreads();
	}

	values[tid] = shared[tid];
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Declaration of the function MergeSort implemented in C, to study the different times of execution
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void mergeSort(int arr[], int p, int q);


int main(int argc, char** argv)
{	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		VARIABLES USED DURING EXECUTION
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		TIMING CUDA OPERATIONS USING EVENTS
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	float elapsed1, elapsed10, elapsed100, elapsed1000 = 0;
	cudaEvent_t start1, stop1, start10, stop10, start100, stop100, start1000, stop1000;
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		CREATING THE EVENTS
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	cudaEventCreate(&start1);
	cudaEventCreate(&start10);
	cudaEventCreate(&start100);
	cudaEventCreate(&start1000);
	cudaEventCreate(&stop1);
	cudaEventCreate(&stop10);
	cudaEventCreate(&stop100);
	cudaEventCreate(&stop1000);
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		ARRAYS TO STUDY
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	int values1[N1], values2[N10], values3[N100], values4[N1000];
	// Creating copies for the study in sequential. 
	int values1s[N1], values2s[N10], values3s[N100], values4s[N1000];
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		POINTERS TO USE ARRAYS IN GPU OPERATIONS
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	int* dvalues, *results;
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//																									MAIN CODE
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	printf("\n\t\t ---------------------- PARALLEL IMPLEMENTATION ---------------------- ");
	//		THE PROGRAM CREATES ARRAYS FOR 1, 10, 100 AND 1000 ELEMENTS WITH RANDOM VALUES
	printf("\nCreating vector for N = 1, 10, 100 and 1000 with random values.\n");
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		ARRAY WITH 1 ELEMENT
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Creating the array
	printf("\nElements for N = 1:\n");
	for (int i = 0; i < N1; i++)
	{
		values1[i] = rand();
		values1s[i] = values1[i];
		printf("%d ", values1[i]);
	}
	printf("\n");
	//Start monitorizing of cuda operations
	cudaEventRecord(start1, 0);
	//Generation cuda variables ables to work and copying the variables from host to device
	cudaMalloc((void**)&dvalues, sizeof(int) * N1);
	cudaMemcpy(dvalues, values1, sizeof(int) * N1, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&results, sizeof(int) * N1);
	cudaMemcpy(results, values1, sizeof(int)* N1, cudaMemcpyHostToDevice);
	//Calling algorithm MergeSort
	CudaMergeSort << <1, N1, sizeof(int) * N1 * 2 >> > (dvalues, results, N1);
	// Freeing memory space used
	cudaFree(dvalues);
	cudaMemcpy(values1, results, sizeof(int)*N1, cudaMemcpyDeviceToHost);
	cudaFree(results);
	//Stopping time monitoring
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	//Calculating the total time of execution
	cudaEventElapsedTime(&elapsed1, start1, stop1);
	// Freeing the events created before
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	//Showing sorted elements
	printf("\nSorted Elements:\n");
	for (int i = 0; i < N1; i++)
	{
		printf("%d ", values1[i]);
	}
	//Showing the time of execution
	printf("\n\t||| The elapsed time in gpu was %.2f ms |||", elapsed1);
	printf("\n");
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		ARRAY WITH 10 ELEMENT
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Creating the array
	printf("\nElements for N = 10:\n");
	for (int i = 0; i < N10; i++)
	{
		values2[i] = rand();
		values2s[i] = values2[i];
		printf("%d ", values2[i]);
	}
	printf("\n");
	//Start monitorizing of cuda operations
	cudaEventRecord(start10, 0);
	//Generation cuda variables ables to work and copying the variables from host to device
	cudaMalloc((void**)&dvalues, sizeof(int) * N10);
	cudaMemcpy(dvalues, values2, sizeof(int) * N10, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&results, sizeof(int) * N10);
	cudaMemcpy(results, values2, sizeof(int)* N10, cudaMemcpyHostToDevice);
	//Calling algorithm MergeSort
	CudaMergeSort << <1, N10, sizeof(int) * N10 * 2 >> > (dvalues, results, N10);
	// Freeing memory space used
	cudaFree(dvalues);
	cudaMemcpy(values2, results, sizeof(int)*N10, cudaMemcpyDeviceToHost);
	cudaFree(results);
	//Stopping time monitoring
	cudaEventRecord(stop10, 0);
	cudaEventSynchronize(stop10);
	//Calculating the total time of execution
	cudaEventElapsedTime(&elapsed10, start10, stop10);
	// Freeing the events created before
	cudaEventDestroy(start10);
	cudaEventDestroy(stop10);
	//Showing sorted elements
	printf("\nSorted Elements:\n");
	for (int i = 0; i < N10; i++)
	{
		printf("%d ", values2[i]);
	}
	//Showing the time of execution
	printf("\n\t||| The elapsed time in gpu was %.2f ms |||", elapsed10);
	printf("\n");

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		ARRAY WITH 100 ELEMENT
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Creating the array
	printf("\nElements for N = 100:\n");
	for (int i = 0; i < N100; i++)
	{
		values3[i] = rand();
		values3s[i] = values3[i];
		printf("%d ", values3[i]);
	}
	printf("\n");
	//Start monitorizing of cuda operations
	cudaEventRecord(start100, 0);
	//Generation cuda variables ables to work and copying the variables from host to device
	cudaMalloc((void**)&dvalues, sizeof(int) * N100);
	cudaMemcpy(dvalues, values3, sizeof(int) * N100, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&results, sizeof(int) * N100);
	cudaMemcpy(results, values3, sizeof(int)* N100, cudaMemcpyHostToDevice);
	//Calling algorithm MergeSort
	CudaMergeSort << <1, N100, sizeof(int) * N100 * 2 >> > (dvalues, results, N100);
	// Freeing memory space used
	cudaFree(dvalues);
	cudaMemcpy(values3, results, sizeof(int)*N100, cudaMemcpyDeviceToHost);
	cudaFree(results);
	//Stopping time monitoring
	cudaEventRecord(stop100, 0);
	cudaEventSynchronize(stop100);
	//Calculating the total time of execution
	cudaEventElapsedTime(&elapsed100, start100, stop100);
	// Freeing the events created before
	cudaEventDestroy(start100);
	cudaEventDestroy(stop100);
	//Showing sorted elements
	printf("\nSorted Elements:\n");
	for (int i = 0; i < N100; i++)
	{
		printf("%d ", values3[i]);
	}
	//Showing the time of execution
	printf("\n\t||| The elapsed time in gpu was %.2f ms |||", elapsed100);
	printf("\n");

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		ARRAY WITH 1000 ELEMENT
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Creating the array
	printf("\nElements for N = 1000:\n");
	for (int i = 0; i < N1000; i++)
	{
		values4[i] = rand();
		values4s[i] = values4[i];
		printf("%d ", values4[i]);
	}
	printf("\n");
	//Start monitorizing of cuda operations
	cudaEventRecord(start1000, 0);
	//Generation cuda variables ables to work and copying the variables from host to device
	cudaMalloc((void**)&dvalues, sizeof(int) * N1000);
	cudaMemcpy(dvalues, values4, sizeof(int) * N1000, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&results, sizeof(int) * N1000);
	cudaMemcpy(results, values4, sizeof(int)* N1000, cudaMemcpyHostToDevice);
	//Calling algorithm MergeSort
	CudaMergeSort << <1, N1000, sizeof(int) * N1000 * 2 >> > (dvalues, results, N1000);
	// Freeing memory space used and returning values sortered
	cudaFree(dvalues);
	cudaMemcpy(values4, results, sizeof(int)*N1000, cudaMemcpyDeviceToHost);
	cudaFree(results);
	//Stopping time monitoring
	cudaEventRecord(stop1000, 0);
	cudaEventSynchronize(stop1000);
	//Calculating the total time of execution
	cudaEventElapsedTime(&elapsed1000, start1000, stop1000);
	// Freeing the events created before
	cudaEventDestroy(start1000);
	cudaEventDestroy(stop1000);
	//Showing sorted elements
	printf("\nSorted Elements:\n");
	for (int i = 0; i < N1000; i++)
	{
		printf("%d ", values4[i]);
	}
	//Showing the time of execution
	printf("\n\t||| The elapsed time in gpu was %.2f ms |||", elapsed1000);
	printf("\n");

	cudaDeviceReset();
	cudaThreadExit();

	printf("\n\t\t ---------------------- SEQUENTIAL IMPLEMENTATION ---------------------- ");

	float elapsed1s, elapsed10s, elapsed100s, elapsed1000s = 0;
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		ARRAY WITH 1 ELEMENT
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//Mergesort for array with one random value
	printf("\nElements for N = 1:\n");
	for (int i = 0; i < N1; i++) {
		printf("%d ", values1s[i]);
	}
	printf("\n\nSorted Elements:");
	//Calling mergesort in c
	clock_t start = clock();
	mergeSort(values1s, 0, N1);
	elapsed1s = ((((double)clock() - start) / CLOCKS_PER_SEC) * 1000000);
	printf("\n");

	for (int i = 0; i < N1; i++) {
		printf("%d ", values1s[i]);
	}
	printf("\n\t||| The elapsed time in cpu was %.2f ms |||", elapsed1s);

	printf("\n");
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		ARRAY WITH 10 ELEMENT
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//Mergesort for array with one random value
	printf("\nElements for N = 10:\n");
	for (int i = 0; i < N10; i++) {
		printf("%d ", values2s[i]);
	}
	printf("\n\nSorted Elements:");
	//Calling mergesort in c
	mergeSort(values2s, 0, N10 - 1);
	elapsed10s = ((((double)clock() - start - elapsed1) / CLOCKS_PER_SEC) * 1000000);
	printf("\n");

	for (int i = 0; i < N10; i++) {
		printf("%d ", values2s[i]);
	}
	printf("\n\t||| The elapsed time in cpu was %.2f ms |||", elapsed10s);

	printf("\n");

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		ARRAY WITH 100 ELEMENT
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//Mergesort for array with one random value
	printf("\nElements for N = 100:\n");
	for (int i = 0; i < N100; i++) {
		printf("%d ", values3s[i]);
	}
	printf("\n\nSorted Elements:");
	//Calling mergesort in c
	mergeSort(values3s, 0, N100 - 1);
	elapsed100s = ((((double)clock() - start - elapsed10) / CLOCKS_PER_SEC) * 1000000);
	printf("\n");

	for (int i = 0; i < N100; i++) {
		printf("%d ", values3s[i]);
	}
	printf("\n\t||| The elapsed time in cpu was %.2f ms |||", elapsed100s);

	printf("\n");

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//		ARRAY WITH 1000 ELEMENT
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//Mergesort for array with one random value
	printf("\nElements for N = 1000:\n");
	for (int i = 0; i < N1000; i++) {
		printf("%d ", values4s[i]);
	}
	printf("\n\nSorted Elements:");
	//Calling mergesort in c
	mergeSort(values4s, 0, N1000 - 1);
	elapsed1000s = ((((double)clock() - start - elapsed100) / CLOCKS_PER_SEC) * 1000000);
	printf("\n");

	for (int i = 0; i < N1000; i++) {
		printf("%d ", values4s[i]);
	}
	printf("\n\t||| The elapsed time in cpu was %.2f ms |||", elapsed1000s);

	printf("\n");

	return 0;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//		Function to combine the different sublists generated and create the array sortered.
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void merge(int arr[], int p, int q, int r) {

	int i, j, k;
	int n1 = q - p + 1;
	int n2 = r - q;

	//arrs temporales
	int L[n1], M[n2];

	for (int i = 0; i < n1; i++)
	{
		L[i] = arr[p + i];
	}

	for (int j = 0; j < n2; j++)
	{
		M[j] = arr[q + 1 + j];
	}

	i = 0;
	j = 0;
	k = p;

	while (i < n1 && j < n2)
	{
		if (L[i] <= M[j])
		{
			arr[k] = L[i];
			i++;
		}
		else
		{
			arr[k] = M[j];
			j++;
		}
		k++;
	}

	while (i < n1)
	{
		arr[k] = L[i];
		i++;
		k++;
	}

	while (j < n2)
	{
		arr[k] = M[j];
		j++;
		k++;
	}
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//		Function which subdivide the array in sublist to be ordered separetly. It use recursivity and it is based in the divide and conquer paradigm. 
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void mergeSort(int arr[], int p, int q) {

	if (p < q)
	{
		int mitad = (p + q) / 2;

		mergeSort(arr, p, mitad);
		mergeSort(arr, mitad + 1, q);
		merge(arr, p, mitad, q);
	}
}
