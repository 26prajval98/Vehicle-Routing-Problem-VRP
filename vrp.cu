#include<iostream>
#include<string.h>
#include "cuda_runtime.h"
#include <thrust/sort.h>

#define X_THREADS 32
#define Y_THREADS 32
#define SIZE 32*32

using namespace std;

typedef struct node{
    int node,
    int x,
    int y,
    int d
} Node;

typedef struct savings{
    int start,
    int end,
    int s_between
} Savings;

typedef struct route{
    int nodes[1000],
    int no
} Route;

struct Demand{
	int node;
	int d;
};

__global__ void
calculateSavings(int* costMatrix, int* savingsMatrix, int rows, int columns)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
	if (y < rows && x < columns) {		
		int valx = costMatrix[x];
		int valy = costMatrix[y];
		int valxy = costMatrix[x + y * rows];
		*(savingsMatrix + x + y * rows) = (x != y && x > y) ? valx + valy - valxy : 0;
    }
}

struct OBCmp {
    __host__ __device__
    bool operator()(const Savings& o1, const Savings& o2) {
        return o1.s_between < o2.s_between;
    }
};

void
sortSavings(Savings * obs, int N){
    thrust::sort(obs, obs+N, OBCmp)
}

__global__ void
getCostMatrix(struct Node* nodeInfos, int *costMatrix, int rows, int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < rows & y < columns) {
		Node tempNode0 = nodeInfos[x];
        Node tempNode1 = nodeInfos[y];

        // Cuda math functions
        // refer https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__CAST.html
		*(costMatrix + x*columns + y) = (x == y) ? 0 : __float2uint_ru(__fsqrt_ru((float)(((tempNode0.xCoOrd-tempNode1.xCoOrd)*(tempNode0.xCoOrd - tempNode1.xCoOrd))+((tempNode0.yCoOrd - tempNode1.yCoOrd)*(tempNode0.yCoOrd - tempNode1.yCoOrd)))));
	}
}



int main(){

}