#include <stdio.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <bits/stdc++.h>
#include <cstdlib>
#include<iostream>
#include <fstream>
#include <vector>
#include <chrono>


#include <opencv2/opencv.hpp>

using namespace std;
using namespace std::chrono;

#define SIZE 32
#define B_SIZE 50000



float median(std::vector<float> a)
{
    int n = a.size();
    if (n % 2 == 0) {
        std::nth_element(a.begin(),a.begin() + n / 2,a.end());
        std::nth_element(a.begin(),a.begin() + (n - 1) / 2,a.end());
  
        return (float)(a[(n - 1) / 2]+ a[n / 2])/ 2.0;
    }
    else {
     std::nth_element(a.begin(),a.begin() + n / 2,a.end());
        return (float)a[n / 2];
    }
}

std::string flindMedianAndReturnBitStr(std::vector<float> flattenVect){

	float md = median(flattenVect);

	std::string bitStr = "";
    for(int a=0;a<64;a++){
		if(flattenVect[a] > md) bitStr += to_string(1);
		else bitStr += to_string(0);
    }

	return bitStr;
}

std::string binary_string_to_hex(std::string &bitStr){

  unsigned long long value = std::stoull(bitStr, 0, 2); 
  std::ostringstream ss;
  ss <<std::setw(16)<<setfill('0') << std::hex << value; 
  std::string result = ss.str();
  return result;

}




//CUDA Constant bariables
__device__ float PRE_COMPUTE_COSINE[SIZE][SIZE];


__constant__ float pi = 3.14159265358979323846;
#define G_BATCH_SIZE B_SIZE
#define G_SIZE SIZE

__device__ float arr[G_BATCH_SIZE][G_SIZE][G_SIZE];


__global__ void preComputeCosineKernel(){
	/* preComputeCosineKernel */
	
	int tx = threadIdx.x;
	for(int i=0;i<SIZE;i++){
		PRE_COMPUTE_COSINE[i][tx] = cosf((2 * i + 1) * tx * pi / (2 * SIZE));
	}
}

__global__ void dctKernel(float *d_a, float *d_o, int s, int bs)
{
	/* Discrete Cosine Transform */

	float ci, cj, sum, dct1;
	
	int bx = blockIdx.x;  //int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	arr[bx][ty][tx] = d_a[(ty * s + tx)+ (bx * s * s)];
	__syncthreads();

	ci = 0;
	cj = 0;
	if (ty == 0)
        ci = 0.17677669529663687;
    else
        ci = 0.25;
    if (tx == 0)
        cj = 0.17677669529663687;
    else
        cj = 0.25;

	sum = 0.0;
	dct1 = 0.0;
	for (int k = 0; k < s; k++) {
		for (int l = 0; l < s; l++) {
			dct1 = arr[bx][k][l] * PRE_COMPUTE_COSINE[k][ty] *  PRE_COMPUTE_COSINE[l][tx];
			sum = sum + dct1;
		}
	}

	d_o[ (ty * s + tx)+ (bx * s * s) ] = ci * cj * sum;

}


int main()
{

	float *d_a,*d_o;
	float *c, *a;


	a = (float*)malloc(B_SIZE * (SIZE * SIZE) * sizeof(float));
	c = (float*)malloc(B_SIZE * (SIZE * SIZE) * sizeof(float));


	////////////////input values///////////////
	vector<cv::Mat> images;
	vector<string> imgFiles;
	for(int i=0;i<1;i++){
		imgFiles.push_back("./cropsFiles/c1.jpg");
		imgFiles.push_back("./cropsFiles/c2.jpg");
		imgFiles.push_back("./cropsFiles/p1.jpg");
		imgFiles.push_back("./cropsFiles/p2.jpg");
		imgFiles.push_back("./cropsFiles/p1.jpg");
		imgFiles.push_back("./cropsFiles/p2.jpg");
	}
	
			


	for(string f : imgFiles){
		cv::Mat img = cv::imread(f);
		cv::Mat greyMat;
		cv::cvtColor(img, greyMat, cv::COLOR_BGR2GRAY);
		cv::Rect myROI(0,0,112,112);
		cv::Mat croppedImage = greyMat(myROI);
		cv::Mat resizedImage, beforeDCT;
		cv::resize(croppedImage, resizedImage, cv::Size(SIZE, SIZE), 0, 0, cv::INTER_AREA);
		resizedImage.convertTo(beforeDCT,CV_32F);
		images.push_back(beforeDCT);
	}
	
	




	int CROP_B_SIZE = images.size();

	//If crop of images batch size more than max batch size in this assert will execute.
	assert(CROP_B_SIZE <= B_SIZE);

	std::cout << " Assigning file data " << std::endl;
	for(int b=0;b<CROP_B_SIZE; b++){
		for(int i=0;i<SIZE;i++){
			for(int j=0;j<SIZE;j++){
				a[(i*SIZE + j) + (b * SIZE * SIZE)] = images[b].at<float>(i * SIZE + j);
			}
		}
	}
	std::cout << " -------------End---------- " << std::endl;

	printf("CUDA memory allocation: %d \n", B_SIZE * (SIZE * SIZE) * sizeof(float));

	//CUDA Memory allocation
	cudaMalloc((void**)&d_a, B_SIZE * (SIZE * SIZE) * sizeof(float));
	cudaMalloc((void**)&d_o, B_SIZE * (SIZE * SIZE) * sizeof(float));
	cudaMemcpy(d_a,a, B_SIZE * (SIZE * SIZE) * sizeof(float),cudaMemcpyHostToDevice);

	//Size initializing for CUDA BLOCKS, THREADS
	dim3 dimBlock(SIZE,SIZE);

	//Calling pre computing consine kernel
	preComputeCosineKernel<<<1,SIZE>>>();

	//Benchmarking and calling dctKernel
	auto start = std::chrono::system_clock::now();
	dctKernel<<<CROP_B_SIZE,dimBlock>>>(d_a, d_o, SIZE, B_SIZE);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout<<"Time taken for dctKernel: "<<elapsed_seconds.count()<<"\n";

	//Synchronizing all CUDA threads
	cudaDeviceSynchronize();

	//insted of copying full matrix only taking 8x8 values.
	cudaMemcpy(c,d_o, B_SIZE * (SIZE * SIZE) * sizeof(float),cudaMemcpyDeviceToHost);


	// //croping top left 8x8 from all btach
	vector<vector<float>> batchLevelFlattenVect;
	for(int b=0;b<CROP_B_SIZE;b++){
		vector<float> flattenVect;
		for(int x=0;x<8;x++){
			for(int y=0;y<8;y++){
				flattenVect.emplace_back(c[(x*SIZE + y) + (b * SIZE * SIZE)]);
			}
		}
		batchLevelFlattenVect.emplace_back(flattenVect);
	}
	
	//Computing binary to hex codes.
	for(int b=0;b<CROP_B_SIZE;b++){
		std::string bitStr = flindMedianAndReturnBitStr(batchLevelFlattenVect[b]);
		std::string dctHex = binary_string_to_hex(bitStr);
		std::cout<<"DCT hex: "<<dctHex<<"\n";
	}


	//Free up all GPU and CPU memorys.
	cudaFree(d_a);
	cudaFree(d_o);
	free(a);
	free(c);

	return 0;
}











