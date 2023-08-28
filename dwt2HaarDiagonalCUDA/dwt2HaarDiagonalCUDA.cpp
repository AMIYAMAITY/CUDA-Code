#define KERNEL_RUN 0

/////////////////// Kernel Imp ////////////////////

#if KERNEL_RUN == 1

    #include "opencv2/opencv.hpp"
    #include<iostream>
    #include<math.h>
    #include <iomanip>
    #include <chrono>
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
    #include "cuda_utils.h"


    #define MAX_IMAGE_INPUT_SIZE_THRESH 1000 * 1000

    // Pointer of cuda stream to be used 
    static cudaStream_t stream;
    using namespace std;
    using namespace cv;

    static uchar* dInCpuBuffer = nullptr;
    static float* dOutCpuBuffer = nullptr;
    static uchar* dInGpuBuffer = nullptr;
    static float* tempInGpuBuffer = nullptr;
    static float* dOutGpuBuffer = nullptr;

    ///////////////////////////////Kernel code start //////////////////////////////
    __global__ void dwtHaarKernelFirst(unsigned char* dInGpuBuffer, int inpH, int inpW, float *dOutGpuBuffer){

        uint rcnt = (blockIdx.x * blockDim.x) + threadIdx.x;
        uint ccnt = (blockIdx.y * blockDim.y) + threadIdx.y;

        if(rcnt % 2 != 0 || rcnt >= inpH || ccnt >= inpW) return;

        float a = (float)dInGpuBuffer[inpW * rcnt + ccnt];
        float b = (float)dInGpuBuffer[inpW * (rcnt +1 ) + ccnt];

        float d=(a-b)*0.707;
        int _rcnt = _rcnt=rcnt/2;

        dOutGpuBuffer[inpW * _rcnt + ccnt] = d;

    }


    __global__ void dwtHaarKernelSecond(float* dInGpuBuffer, int inpH, int inpW, float *dOutGpuBuffer){

        uint rcnt = (blockIdx.x * blockDim.x) + threadIdx.x;
        uint ccnt = (blockIdx.y * blockDim.y) + threadIdx.y;

        if(ccnt % 2 != 0 || rcnt >= inpH || ccnt >= inpW) return;

        float a = (float)dInGpuBuffer[inpW * rcnt + ccnt];
        float b = dInGpuBuffer[inpW * rcnt + (ccnt + 1)];

        int _ccnt=ccnt/2;
        float d=(a-b)*0.707;

        dOutGpuBuffer[(inpW/2) * rcnt + _ccnt] = fabs(d);
    }

    void dwtHaarGPU(unsigned char* &dInCpuBuffer, float* &dOutCpuBuffer, unsigned char* &dInGpuBuffer, float* &dOutGpuBuffer, float* &tempInGpuBuffer,  int  inpH, int inpW, int inpC,  int Outsize, cudaStream_t stream){

        dim3 threadsPerBlock(22,22); // 484 threads
        dim3 dimGrid(inpH/threadsPerBlock.x + 1, inpW/threadsPerBlock.y + 1);

        cudaMemcpy(dInGpuBuffer,dInCpuBuffer, inpH * inpW * inpC * sizeof(unsigned char),cudaMemcpyHostToDevice);

        dwtHaarKernelFirst<<<dimGrid,threadsPerBlock, 0, stream>>>(dInGpuBuffer, inpH, inpW,tempInGpuBuffer);
        cudaStreamSynchronize(stream);

        dim3 dimGrid2((inpH/2)/threadsPerBlock.x + 1, inpW/threadsPerBlock.y + 1);
        dwtHaarKernelSecond<<<dimGrid2,threadsPerBlock, 0, stream>>>(tempInGpuBuffer, inpH/2, inpW,dOutGpuBuffer);

        cudaStreamSynchronize(stream);
        cudaMemcpy(dOutCpuBuffer,dOutGpuBuffer,Outsize *sizeof(float),cudaMemcpyDeviceToHost);


    }   


    ///////////////////////////////Kernel code End ////////////////////////////// 


    void memoryInit(){
        CUDA_CHECK(cudaMallocHost((void**)&dInCpuBuffer,  MAX_IMAGE_INPUT_SIZE_THRESH * sizeof(uchar)));
        CUDA_CHECK(cudaMallocHost((void**)&dOutCpuBuffer, MAX_IMAGE_INPUT_SIZE_THRESH * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&dInGpuBuffer,  MAX_IMAGE_INPUT_SIZE_THRESH * sizeof(uchar)));
        CUDA_CHECK(cudaMalloc((void**)&tempInGpuBuffer,  MAX_IMAGE_INPUT_SIZE_THRESH * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&dOutGpuBuffer, MAX_IMAGE_INPUT_SIZE_THRESH * sizeof(float)));
        
        CUDA_CHECK(cudaStreamCreate(&stream));
    }



    // Function to compare two integers for qsort
    int compare(const void* a, const void* b)
    {
        const float* x = (float*) a;
        const float* y = (float*) b;

        if (*x > *y)
            return 1;
        else if (*x < *y)
            return -1;

        return 0;
    }

    double findMedianUnSortedArray(float* &a, int n)
    {
        // First we sort the array
        qsort(a, n, sizeof(float), compare);

    
        // check for even case
        if (n % 2 != 0)
            return (double)a[n / 2];
    
        return (double)(a[(n - 1) / 2] + a[n / 2]) / 2.0;
    }


    double getDwtMadNoiseEst(cv::Mat &im){

        cv::Mat grayScaleIm;
        if ((int)im.channels()  == 3)
            cvtColor(im, grayScaleIm, cv::COLOR_BGRA2GRAY);
        else
            grayScaleIm = im;


        int size = grayScaleIm.rows/2 * grayScaleIm.cols/2;

        dInCpuBuffer = grayScaleIm.data;

        auto start2 = std::chrono::system_clock::now();

        //Sincce we are not computing other axis like cV,cH, etc, so that need bo be pass tempInGpuBuffer, otherwise memcopy need to be done.
        dwtHaarGPU(dInCpuBuffer, dOutCpuBuffer, dInGpuBuffer, dOutGpuBuffer,tempInGpuBuffer,  grayScaleIm.rows, grayScaleIm.cols, 1 ,  size, stream);

        auto end2 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
        printf("TimeTaken for dwt2Haar GPU: %f .ms\n", elapsed_seconds2.count() * 1000);


        auto start = std::chrono::system_clock::now();
        double med_value = findMedianUnSortedArray(dOutCpuBuffer, size);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        printf("TimeTaken for findMedianUnSortedArray: %f .ms\n\n", elapsed_seconds.count() * 1000);
        double noiseLevel = med_value / 0.6745;
        return noiseLevel;
    }



    int main()
    {
        memoryInit();
        cv::Mat im;
        double noiseLevel;
        std::chrono::duration<double> elapsed_seconds;


        // std::vector<std::string> images = { "./images/1.jpg", "./images/2.jpg",  "./images/Black_Screen.jpg", "./images/warpedFace_cpp.jpg"};

        vector<cv::String> images;
        glob("./shreejal_images/*.png", images, false);
        sort(images.begin(), images.end());

        for(std::string name: images){
            im=cv::imread(name); //Load image in Gray Scale
            noiseLevel = getDwtMadNoiseEst(im);
            // std::cout<<name<<"  noiseLevel: "<< std::fixed << std::setw(30) << std::setprecision(26)<<(double)noiseLevel<<std::endl;
            printf("GPU: %s  -> noiseLevel: %lf \n",name.c_str(), (double)noiseLevel);
            //  sleep(5);
        }


        //Timing profile
        im=cv::imread("./images/warpedFace_cpp.jpg");
        printf("Input Image size: H: %d  W:%d  C:%d",im.rows, im.cols, im.channels());


        auto start = std::chrono::system_clock::now();
         noiseLevel = getDwtMadNoiseEst(im);
        auto end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        printf("\n End-to-End TimeTaken for getDwtMadNoiseEst: %f .ms\n", elapsed_seconds.count() * 1000);
        printf("Noise Level: %f\n ", noiseLevel);



        return 0;

    }
















#else





    // /////////////////////////////// CPP code /////////////////////////


    #include "opencv2/opencv.hpp"
    #include<iostream>
    #include<math.h>
    #include <iomanip>
    #include <chrono>

    using namespace std;
    using namespace cv;
    cv::Mat im2, im6; //im1,im2,im3,im4,im5,im6;


    float* cD_abs_values = (float*)malloc(1000 * 1000 * sizeof(float));

    void  dwt2Haar(cv::Mat &im){

        float a,b,c,d;
        im.convertTo(im,CV_32F,1.0,0.0);

        // im1=Mat::zeros(im.rows/2,im.cols,CV_32F);
        im2=Mat::zeros(im.rows/2,im.cols,CV_32F);
        // im3=Mat::zeros(im.rows/2,im.cols/2,CV_32F);
        // im4=Mat::zeros(im.rows/2,im.cols/2,CV_32F);
        // im5=Mat::zeros(im.rows/2,im.cols/2,CV_32F);
        im6=Mat::zeros(im.rows/2,im.cols/2,CV_32F);

        //--------------Decomposition-------------------
        for(int rcnt=0;rcnt<im.rows;rcnt+=2)
        {
            for(int ccnt=0;ccnt<im.cols;ccnt++)
            {
                a=im.at<float>(rcnt,ccnt);
                b=im.at<float>(rcnt+1,ccnt);

                // c=(a+b)*0.707;
                d=(a-b)*0.707;
                int _rcnt=rcnt/2;
                // im1.at<float>(_rcnt,ccnt)=c;
                im2.at<float>(_rcnt,ccnt)=d;
            }
        }


        // for(int i=0; i< (im.rows / 2) * im.cols; i++){
        //     printf("\n dataComp: CPUim2 %f", (float)im2.data[i]);
        // }

        // for(int rcnt=0;rcnt<im.rows/2;rcnt++)
        // {
        //     for(int ccnt=0;ccnt<im.cols;ccnt+=2)
        //     {

        //         a=im1.at<float>(rcnt,ccnt);
        //         b=im1.at<float>(rcnt,ccnt+1);
        //         c=(a+b)*0.707;
        //         d=(a-b)*0.707;
        //         int _ccnt=ccnt/2;
        //         im3.at<float>(rcnt,_ccnt)=c;
        //         im4.at<float>(rcnt,_ccnt)=d;
        //     }
        // }

        int i=0;
        for(int rcnt=0;rcnt<im.rows/2;rcnt++)
        {
            for(int ccnt=0;ccnt<im.cols;ccnt+=2)
            {

                a=im2.at<float>(rcnt,ccnt);
                b=im2.at<float>(rcnt,ccnt+1);

                // c=(a+b)*0.707;
                d=(a-b)*0.707;
                int _ccnt=ccnt/2;
                // im5.at<float>(rcnt,_ccnt)=c;
                // im6.at<float>(rcnt,_ccnt)=d;

                cD_abs_values[i] = (float)abs(d);
                i++;

            }
        }
    }


    // Function to compare two integers for qsort
    int compare(const void* a, const void* b)
    {
        const float* x = (float*) a;
        const float* y = (float*) b;

        if (*x > *y)
            return 1;
        else if (*x < *y)
            return -1;

        return 0;
    }

    double findMedianUnSortedArray(float* &a, int n)
    {
        // First we sort the array
        qsort(a, n, sizeof(float), compare);
        // thrust::sort(thrust::host, a, a + n, thrust::greater<float>());
    
        // check for even case
        if (n % 2 != 0)
            return (double)a[n / 2];
    
        return (double)(a[(n - 1) / 2] + a[n / 2]) / 2.0;
    }


    double getDwtMadNoiseEst(cv::Mat &im){

        cv::Mat grayScaleIm;
        if ((int)im.channels()  == 3)
            cvtColor(im, grayScaleIm, cv::COLOR_BGRA2GRAY);
        else
            grayScaleIm = im;


        int size = grayScaleIm.rows/2 * grayScaleIm.cols/2;


        auto start1 = std::chrono::system_clock::now();
        dwt2Haar(grayScaleIm);
        auto end1 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds1 = end1 - start1;
        printf("TimeTaken for dwt2Haar CPU: %f .ms\n", elapsed_seconds1.count() * 1000);

        auto start = std::chrono::system_clock::now();
        double med_value = findMedianUnSortedArray(cD_abs_values, size);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        printf("TimeTaken for findMedianUnSortedArray: %f .ms\n\n", elapsed_seconds.count() * 1000);
        double noiseLevel = med_value / 0.6745;
        return noiseLevel;
    }



    int main()
    {
        cv::Mat im;
        double noiseLevel;
        std::chrono::duration<double> elapsed_seconds;


        // std::vector<std::string> images = { "./images/1.jpg", "./images/2.jpg",  "./images/Black_Screen.jpg", "./images/warpedFace_cpp.jpg"};
        // std::vector<std::string> images = { "./images/Black_Screen.jpg","./images/warpedFace_cpp.jpg",  "./images/warpedFace_cpp.jpg"};

        vector<cv::String> images;
        glob("./shreejal_images/*.png", images, false);
        sort(images.begin(), images.end());

        for(std::string name: images){
            im=cv::imread(name); //Load image in Gray Scale
            noiseLevel = getDwtMadNoiseEst(im);
            printf("CPU: %s  -> noiseLevel: %lf \n",name.c_str(), (double)noiseLevel);
        }


        //Timing profile
        im=cv::imread("./images/warpedFace_cpp.jpg");
        printf("Input Image size: H: %d  W:%d  C:%d",im.rows, im.cols, im.channels());

        auto start = std::chrono::system_clock::now();
        noiseLevel = getDwtMadNoiseEst(im);
        auto end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        printf("\n End-to-End TimeTaken for getDwtMadNoiseEst: %f .ms\n", elapsed_seconds.count() * 1000);
        printf("Noise Level: %f\n ", noiseLevel);


        return 0;

    }



#endif
