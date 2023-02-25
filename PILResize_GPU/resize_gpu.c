#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include<opencv2/opencv.hpp>
#include <chrono>
#include <ctime>
#include <iostream>


using namespace cv;
using namespace std;

#define ROUND_UP(f) ((int) ((f) >= 0.0 ? (f) + 0.5F : (f) - 0.5F))
#define UINT8 unsigned char
#define INT32 int
#define INT_MAX 0x7fffffff

/* pixel types */
#define IMAGING_TYPE_UINT8 0
#define IMAGING_TYPE_INT32 1
#define IMAGING_TYPE_FLOAT32 2
#define IMAGING_TYPE_SPECIAL 3 /* check mode for details */

#define IMAGING_MODE_LENGTH 6+1 /* Band names ("1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "BGR;xy") */


/* standard transforms */
#define IMAGING_TRANSFORM_AFFINE 0
#define IMAGING_TRANSFORM_PERSPECTIVE 2
#define IMAGING_TRANSFORM_QUAD 3


/* standard filters */
#define IMAGING_TRANSFORM_NEAREST 0
#define IMAGING_TRANSFORM_BOX 4
#define IMAGING_TRANSFORM_BILINEAR 2
#define IMAGING_TRANSFORM_HAMMING 5
#define IMAGING_TRANSFORM_BICUBIC 3
#define IMAGING_TRANSFORM_LANCZOS 1

typedef void (*ResampleFunction)(unsigned char *pOut, unsigned char *pIn, int offset,
                               int ksize, int *bounds, double *prekk, int inpWd, int inpHt, int inpStride, int outWd, int outHt, int outStride, int imType, int channels, int letterBox, int pad_top, int pad_left);
struct filter {
    double (*filter)(double x);
    double support;
};

static inline double box_filter(double x)
{
    if (x >= -0.5 && x < 0.5)
        return 1.0;
    return 0.0;
}

static inline double bilinear_filter(double x)
{
    if (x < 0.0)
        x = -x;
    if (x < 1.0)
        return 1.0-x;
    return 0.0;
}

static inline double hamming_filter(double x)
{
    if (x < 0.0)
        x = -x;
    if (x == 0.0)
        return 1.0;
    if (x >= 1.0)
        return 0.0;
    x = x * M_PI;
    return sin(x) / x * (0.54f + 0.46f * cos(x));
}

static inline double bicubic_filter(double x)
{
    /* https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm */
#define a -0.5
    if (x < 0.0)
        x = -x;
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x*x + 1;
    if (x < 2.0)
        return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
#undef a
}

static inline double sinc_filter(double x)
{
    if (x == 0.0)
        return 1.0;
    x = x * M_PI;
    return sin(x) / x;
}

static inline double lanczos_filter(double x)
{
    /* truncated sinc */
    if (-3.0 <= x && x < 3.0)
        return sinc_filter(x) * sinc_filter(x/3);
    return 0.0;
}

static struct filter BOX = { box_filter, 0.5 };
static struct filter BILINEAR = { bilinear_filter, 1.0 };
static struct filter HAMMING = { hamming_filter, 1.0 };
static struct filter BICUBIC = { bicubic_filter, 2.0 };
static struct filter LANCZOS = { lanczos_filter, 3.0 };


/* 8 bits for result. Filter can have negative areas.
   In one cases the sum of the coefficients will be negative,
   in the other it will be more than 1.0. That is why we need
   two extra bits for overflow and int type. */
#define PRECISION_BITS (32 - 8 - 2)


/* Handles values form -640 to 639. */
UINT8 _clip8_lookups[1280] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};


//Global variables
unsigned char *pIn_gpu, *pOut_gpu, *lpOut_gpu, *pTemp;
int *bounds_gpu;
double *kk_gpu;
UINT8 *clip8_lookups_gpu;
unsigned char* transpose_gpu;
unsigned char *pIn2;


void memory_initialization_gpu(){
    /*
    memory_initialization_gpu
    */
    int MAX_OUTPUT_HW_SIZE = 1024;
    int MAX_INPUT_HW_SIZE = 3000;
    int MAX_K_SIZE = 2024;
    int CHANNELS = 3;

    cudaMalloc((void**)&bounds_gpu, MAX_OUTPUT_HW_SIZE * 2 * sizeof(int));
    cudaMalloc((void**)&kk_gpu, MAX_OUTPUT_HW_SIZE * MAX_K_SIZE * sizeof(double));
    cudaMalloc((void**)&pIn_gpu, MAX_INPUT_HW_SIZE * MAX_INPUT_HW_SIZE * CHANNELS * sizeof(unsigned char));
    cudaMalloc((void**)&pOut_gpu, MAX_OUTPUT_HW_SIZE * MAX_OUTPUT_HW_SIZE * CHANNELS * sizeof(unsigned char));
    cudaMalloc((void**)&lpOut_gpu, MAX_OUTPUT_HW_SIZE * MAX_OUTPUT_HW_SIZE * CHANNELS * sizeof(unsigned char));
    cudaMalloc((void**)&pTemp, MAX_INPUT_HW_SIZE * MAX_INPUT_HW_SIZE * CHANNELS * sizeof(unsigned char));
    
    //lookup
    UINT8 *clip8_lookups_cpu = &_clip8_lookups[640];
    cudaMalloc((void**)&clip8_lookups_gpu,  1280 * sizeof(UINT8) );
    cudaMemcpy(clip8_lookups_gpu, clip8_lookups_cpu , 1280 * sizeof(UINT8) ,cudaMemcpyHostToDevice);
}


__global__ void horizontalKernel(unsigned char *pOut, unsigned char *pIn, 
                                int *bounds,  double* kk, 
                                int dst_width, int dst_height, 
                                int ksize, 
                                int inpStride, int outStride,
                                int inpWd,
                                int inpHt,
                                int channels,
                                UINT8 *clip8_lookups_gpu){
    

    uint xx = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint yy = (blockIdx.y * blockDim.y) + threadIdx.y;

    if(xx >= dst_width || yy >= dst_height) return; //safe

    int xmin = bounds[xx * 2 + 0];
    int xmax = bounds[xx * 2 + 1];
    double *k = &kk[xx * ksize];

    for (int c = 0; c < channels; c++){
        double ss0 = 1U << (PRECISION_BITS - 1U);
        for (int x = 0; x < xmax ; x++)
        {
            ss0 += pIn[yy  * (inpWd*channels) + (x + xmin) * channels + c] * k[x]; //working code for RGB
        }

        int index= yy* (dst_width*channels) + (channels*xx) + c;
        pOut[index] = (unsigned char)clip8_lookups_gpu[ (int)ss0 >> PRECISION_BITS]; //clip8(ss0); // working code RGB

    }

} 


__global__ void verticalKernel( unsigned char *pOut, unsigned char *pIn, 
                                int *bounds,  double* kk, 
                                int dst_width, int dst_height, 
                                int ksize, int inpStride, int outStride,
                                int inpWd,
                                int inpHt,
                                int channels,
                                UINT8 *clip8_lookups_gpu){

    uint xx = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint yy = (blockIdx.y * blockDim.y) + threadIdx.y;

    if(xx >= dst_width || yy >= dst_height) return;

    double ss0;
    double *k = &kk[xx * ksize];
    int ymin = bounds[xx * 2 + 0];
    int ymax = bounds[xx * 2 + 1];

    for (int c = 0; c < channels; c++){
        ss0 = 1U << (PRECISION_BITS - 1U);
        for (int y = 0; y < ymax; y++)
        {   
            int idx = yy  * (inpWd*channels) + (y + ymin) * channels + c;
            ss0 += pIn[idx] * k[y]; //working code for RGB  
        }

        int index = yy* (dst_width*channels) + (channels*xx) + c;
        pOut[index] = (unsigned char)(clip8_lookups_gpu[ (int)ss0 >> PRECISION_BITS]);

    }
}

__global__ void transposeKernel(unsigned char* inputData, unsigned char* transpose,unsigned char* lpOut_gpu, int width, int height, int channels, int letterBox, int pad_top, int pad_left){
	
	uint xx = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint yy = (blockIdx.y * blockDim.y) + threadIdx.y;

    if(xx >= width || yy >= height) return; //safe

	for(int c = 0; c < channels; c++){
		transpose[xx* (height*channels) + (channels*yy) + c] = inputData[yy* (width*channels) + (channels*xx) + c];

        if(letterBox){
            int targetHeight = ( width + (pad_top * 2));
            int targetWidth = (height + (pad_left * 2));
            lpOut_gpu[(xx+ pad_top)* (targetWidth*channels) + (channels*(yy + pad_left)) + c] = transpose[xx* (height*channels) + (channels*yy) + c];
        }
	}
}


UINT8 *clip8_lookups = &_clip8_lookups[640];

static inline UINT8 clip8(int in)
{
    //printf("%d\n", in);
    return clip8_lookups[in >> PRECISION_BITS];
}


int
precompute_coeffs(int inSize, float in0, float in1, int outSize,
                  struct filter *filterp, int **boundsp, double **kkp) {
    double support, scale, filterscale;
    double center, ww, ss;
    int xx, x, ksize, xmin, xmax;
    int *bounds;
    double *kk, *k;

    /* prepare for horizontal stretch */
    //printf("outsize = %d :: in1 = %f :: in0 = %f \n", outSize, in1, in0);
    filterscale = scale = (double) (in1 - in0) / outSize;
    if (filterscale < 1.0) {
        filterscale = 1.0;
    }

    /* determine support size (length of resampling filter) */
    support = filterp->support * filterscale;

    /* maximum number of coeffs */
    ksize = (int) ceil(support) * 2 + 1;
    //printf("ksize = %d\n", ksize);
    // check for overflow
    if (outSize > INT_MAX / (ksize * sizeof(double))) {
        return 0;
    }

    /* coefficient buffer */
    /* malloc check ok, overflow checked above */
    kk = (double *) malloc(outSize * ksize * sizeof(double));
    if ( ! kk) {
        return 0;
    }

    /* malloc check ok, ksize*sizeof(double) > 2*sizeof(int) */
    bounds = (int *) malloc(outSize * 2 * sizeof(int));
    if ( ! bounds) {
        free(kk);
        return 0;
    }

    for (xx = 0; xx < outSize; xx++) {
        center = in0 + (xx + 0.5) * scale;
        ww = 0.0;
        ss = 1.0 / filterscale;
        // Round the value
        xmin = (int) (center - support + 0.5);
        if (xmin < 0)
            xmin = 0;
        // Round the value
        xmax = (int) (center + support + 0.5);
        if (xmax > inSize)
            xmax = inSize;
        xmax -= xmin;
        k = &kk[xx * ksize];
        for (x = 0; x < xmax; x++) {
            double w = filterp->filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++) {
            if (ww != 0.0)
                k[x] /= ww;
        }
	
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < ksize; x++) {
            k[x] = 0;
        }
	/*printf("xmin = %d :: xmax = %d\n", xmin, xmax);
	for (x = 0; x < xmax; x++)
	    printf("%f ", k[x]);
	printf("\n");*/
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    *boundsp = bounds;
    *kkp = kk;
    return ksize;
}


void
normalize_coeffs_8bpc_old(int outSize, int ksize, double *prekk)
{
    int x;
    INT32 *kk;

    // use the same buffer for normalized coefficients
    kk = (INT32 *) prekk;

    for (x = 0; x < outSize * ksize; x++) {
        if (prekk[x] < 0) {
            kk[x] = (int) (-0.5 + prekk[x] * (1 << PRECISION_BITS));
        } else {
            kk[x] = (int) (0.5 + prekk[x] * (1 << PRECISION_BITS));
        }
    }
}

double *normalize_coeffs_8bpc(int outSize, int ksize, double *prekk)
{
    int x;
    double *kk;
    kk = (double *)malloc((ksize * outSize * sizeof(double)));
    // use the same buffer for normalized coefficients
    // kk = (INT32 *) prekk;

    for (x = 0; x < outSize * ksize; x++) {
        if (prekk[x] < 0) {
            kk[x] = static_cast<int> (-0.5 + prekk[x] * (1 << PRECISION_BITS));
        } else {
            kk[x] = static_cast<int> (0.5 + prekk[x] * (1 << PRECISION_BITS));
        }
    }
    return kk;
}


void
ImagingResampleVertical_8bpc(unsigned char *pOut, unsigned char *pIn, int offset,
                             int ksize, int *bounds, double *prekk, int inpWd, int inpHt, int inpStride, int outWd, int outHt, int outStride, int imType, int channels, int letterBox, int pad_top, int pad_left)
{


    double *kk = normalize_coeffs_8bpc(outHt, ksize, prekk); //CPU

    ///////////////////////////////////////////////////////kernel call start////////////////////////////////////////////////////////////

    int outSize = outHt;
    // int xsize = outHt;


    //KERNEL CALLS FOR TRANSPOSE
    cudaMalloc((void**)&transpose_gpu, inpWd * inpHt * channels * sizeof(unsigned char));
    dim3 threadsPerBlockForTranspose(30, 30);
	dim3 numBlocksForTranspose(inpWd/threadsPerBlockForTranspose.x + 1, inpHt/threadsPerBlockForTranspose.y + 1); 
	transposeKernel<<<numBlocksForTranspose,threadsPerBlockForTranspose>>>(pIn, transpose_gpu, lpOut_gpu,inpWd, inpHt, channels, 0,0,0);
	cudaDeviceSynchronize();


    cudaMemcpy( bounds_gpu , bounds ,outSize * 2 * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy( kk_gpu , kk ,outSize * ksize * sizeof(double),cudaMemcpyHostToDevice);
    free(kk);


    //CALL VERTICAL;
    dim3 threadsPerBlockForVertical(30,30);
	dim3 numBlocksForVertical(outHt/threadsPerBlockForVertical.x + 1, outWd/threadsPerBlockForVertical.y+1); 
    verticalKernel<<<numBlocksForVertical, threadsPerBlockForVertical>>>(pOut_gpu, transpose_gpu, bounds_gpu, kk_gpu, outHt,outWd, ksize, inpStride, outStride, inpHt, inpWd, channels,clip8_lookups_gpu);
    cudaDeviceSynchronize();


    //Different size of transpose size memory allocation and deallocation.
    cudaFree(transpose_gpu);
    cudaMalloc((void**)&transpose_gpu, outWd * outHt * channels * sizeof(unsigned char));

    //LetterBox
    int toutWd = outWd + (pad_left * 2);
    int toutHt = outHt + (pad_top * 2);

    if(letterBox)
        cudaMemset(lpOut_gpu, 0, (toutWd * toutHt * channels * sizeof(unsigned char)));

    //KERNEL CALLS FOR TRANSPOSE
    dim3 threadsPerBlock(30, 30);
	dim3 numBlocks(outHt/threadsPerBlock.x + 1, outWd/threadsPerBlock.y + 1); 
	transposeKernel<<<numBlocks,threadsPerBlock>>>(pOut_gpu, transpose_gpu, lpOut_gpu, outHt, outWd ,channels, letterBox, pad_top, pad_left);
	cudaDeviceSynchronize();


    if(!letterBox)
        cudaMemcpy( pOut_gpu , transpose_gpu ,outWd*outHt * channels * sizeof(unsigned char),cudaMemcpyDeviceToDevice);
    else
        cudaMemcpy( pOut_gpu , lpOut_gpu ,(toutWd * toutHt * channels * sizeof(unsigned char)),cudaMemcpyDeviceToDevice);

    cudaFree(transpose_gpu);

    ///////////////////////////////////////////////////////kernel call end////////////////////////////////////////////////////////////
}

        
void
ImagingResampleHorizontal_8bpc(unsigned char *pOut, unsigned char *pIn, int offset,
                               int ksize, int *bounds, double *prekk, int inpWd, int inpHt, int inpStride, int outWd, int outHt, int outStride, int imType, int channels, int lt, int pt, int pl)
{


    double *kk = normalize_coeffs_8bpc(outWd, ksize, prekk); //CPU
    int outSize = outWd;
    int xsize = outHt;


    ///////////////////////////////////////////////////////kernel call start////////////////////////////////////////////////////////////

    cudaMemcpy( bounds_gpu , bounds ,outSize * 2 * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy( kk_gpu , kk ,outSize * ksize * sizeof(double),cudaMemcpyHostToDevice);
    free(kk);



    //KERNEL CALL
	dim3 threadsPerBlock(30, 30);
	dim3 numBlocks(outWd/threadsPerBlock.x + 1, outHt/threadsPerBlock.y + 1); 

    horizontalKernel<<<numBlocks, threadsPerBlock>>>(pOut_gpu, pIn_gpu, 
                                            bounds_gpu, kk_gpu, 
                                            outWd, outHt, ksize, 
                                            inpStride, outStride, 
                                            inpWd, 
                                            inpHt,
                                            channels, 
                                            clip8_lookups_gpu);
    cudaDeviceSynchronize();
    ///////////////////////////////////////////////////////kernel call end////////////////////////////////////////////////////////////
}

int ImagingResampleInner(unsigned char *pIn, unsigned char *pOut, int inpWd, int inpHt, int inpStride, int xsize, int ysize, int outStride,
                     struct filter *filterp, float box[4],
                     ResampleFunction ResampleHorizontal,
                     ResampleFunction ResampleVertical, int imType, int channels, int letterBox, int pad_top, int pad_left)
{

    unsigned char *pImTemp = NULL;
    int i, need_horizontal, need_vertical;
    int ybox_first, ybox_last;
    int ksize_horiz, ksize_vert;
    int *bounds_horiz, *bounds_vert;
    double *kk_horiz, *kk_vert;

    need_horizontal = xsize != inpWd || box[0] || box[2] != xsize;
    need_vertical = ysize != inpHt || box[1] || box[3] != ysize;
    
    ksize_horiz = precompute_coeffs(inpWd, box[0], box[2], xsize,
                                    filterp, &bounds_horiz, &kk_horiz);
    if ( ! ksize_horiz) {
        return -1;
    }

    ksize_vert = precompute_coeffs(inpHt, box[1], box[3], ysize,
                                   filterp, &bounds_vert, &kk_vert);

    if ( ! ksize_vert) {
        free(bounds_horiz);
        free(kk_horiz);
        free(bounds_vert);
        free(kk_vert);
        return -1;
    }

    // First used row in the source image
    ybox_first = bounds_vert[0];
    // Last used row in the source image
    ybox_last = bounds_vert[ysize*2 - 2] + bounds_vert[ysize*2 - 1];


    /* two-pass resize, horizontal pass */
    if (need_horizontal) {
        int stride;
        // Shift bounds for vertical pass
        for (i = 0; i < ysize; i++) {
            bounds_vert[i * 2] -= ybox_first;
        }
    if (need_vertical){
        // pImTemp = pTemp; //(unsigned char *)malloc(xsize * inpHt * channels * 4);
    }
    else pTemp = pOut_gpu;

    stride = need_vertical?xsize:outStride;
    if (pTemp) {
        ResampleHorizontal(pTemp, pIn_gpu, ybox_first,ksize_horiz, bounds_horiz, kk_horiz, inpWd, inpHt, inpStride, xsize, inpHt, stride, imType, channels, letterBox, pad_top, pad_left);
        pTemp = pOut_gpu;
    }
    free(bounds_horiz);
    free(kk_horiz);
    if ( ! pTemp) {
        free(bounds_vert);
        free(kk_vert);
        return -1;
    }

    } else {
        // Free in any case
        free(bounds_horiz);
        free(kk_horiz);
    }

    /* vertical pass */
    if (need_vertical) {
        // unsigned char *pIn2;
        int wd;
        int stride = need_horizontal?xsize:inpStride;
        pIn2 = need_horizontal?pTemp:pIn_gpu;
        if (1) {
            /* imIn can be the original image or horizontally resampled one */
            ResampleVertical(pOut_gpu, pIn2, 0,ksize_vert, bounds_vert, kk_vert, xsize, inpHt, stride, xsize, ysize, outStride, imType, channels, letterBox, pad_top, pad_left);
        }

        /* it's safe to call ImagingDelete with empty value
           if previous step was not performed. */
        free(pImTemp); 
        free(bounds_vert);
        free(kk_vert);
        return 0;
        /*if ( ! imOut) {
            return NULL;
        }*/
    } else {
        // Free in any case
        free(bounds_vert);
        free(kk_vert);
    }

    // /* none of the previous steps are performed, copying */
    // if ( ! (need_horizontal || need_vertical)) {
    // //printf("memcpy only\n");
    //     //memcpy(pOut, pIn, xsize*ysize*((imType == IMAGING_TYPE_UINT8)?1:4)*channels);
    // int i;
    // for (i = 0; i < ysize; i++)
    //     memcpy(pOut + i*outStride, pIn + i*inpStride, xsize*channels*((imType == IMAGING_TYPE_UINT8)?1:4));
    // }

    return 0;
}


int ImagingResample(unsigned char *pIn, unsigned char *pOut, int inpWd, int inpHt, int inpStride, int xsize, int ysize, int outStride, int filter, float box[4], int imType, int channels, int letterBox, int pad_top, int pad_left)
{
    struct filter *filterp;
    ResampleFunction ResampleHorizontal;
    ResampleFunction ResampleVertical;


    
        switch(imType) {
            case IMAGING_TYPE_UINT8:
                ResampleHorizontal = ImagingResampleHorizontal_8bpc;
                ResampleVertical = ImagingResampleVertical_8bpc;
                break;
            case IMAGING_TYPE_INT32:
            case IMAGING_TYPE_FLOAT32:
                //ResampleHorizontal = ImagingResampleHorizontal_32bpc;
                //ResampleVertical = ImagingResampleVertical_32bpc;
                //break;
            default:
                return -1;
        }

    /* check filter */
    switch (filter) {
    case IMAGING_TRANSFORM_BOX:
        filterp = &BOX;
        break;
    case IMAGING_TRANSFORM_BILINEAR:
        filterp = &BILINEAR;
        break;
    case IMAGING_TRANSFORM_HAMMING:
        filterp = &HAMMING;
        break;
    case IMAGING_TRANSFORM_BICUBIC:
        filterp = &BICUBIC;
        break;
    case IMAGING_TRANSFORM_LANCZOS:
        filterp = &LANCZOS;
        break;
    default:
        return -1;
    }

    return ImagingResampleInner(pIn, pOut,inpWd, inpHt, inpStride, xsize, ysize, outStride, filterp, box,ResampleHorizontal, ResampleVertical, imType, channels, letterBox, pad_top, pad_left);
}


// modified resize routine
int resizeModPIL(unsigned char *pIn, unsigned char *pOut, int inpWd, int inpHt, int inpStride, int outWd, int outHt, int outStride, int channels, int letterBox, int pad_top, int pad_left)
{
    //Imaging imIn;
    //Imaging imOut;

    int xsize, ysize;
    int filter = IMAGING_TRANSFORM_BICUBIC; //IMAGING_TRANSFORM_LANCZOS;
    float box[4] = {0, 0, 0, 0};
    int imType = IMAGING_TYPE_UINT8;
    //imIn = self->image;
    box[2] = inpWd;
    box[3] = inpHt;
    
    xsize = outWd;
    ysize = outHt;
    
    
    if (xsize < 1 || ysize < 1) {
        return -1;//ImagingError_ValueError("height and width must be > 0");
    }

    if (box[0] < 0 || box[1] < 0) {
        return -1;//ImagingError_ValueError("box offset can't be negative");
    }

    if (box[2] > inpWd || box[3] > inpHt) {
        return -1;//ImagingError_ValueError("box can't exceed original image size");
    }

    if (box[2] - box[0] < 0 || box[3] - box[1] < 0) {
        return -1;//ImagingError_ValueError("box can't be empty");
    }

    // If box's coordinates are int and box size matches requested size
    if (0)/*(box[0] - (int) box[0] == 0 && box[2] - box[0] == xsize
            && box[1] - (int) box[1] == 0 && box[3] - box[1] == ysize) */{
        //imOut = ImagingCrop(imIn, box[0], box[1], box[2], box[3]);
    }
    else if (filter == IMAGING_TRANSFORM_NEAREST) {
        double a[6];

        memset(a, 0, sizeof a);
        a[0] = (double) (box[2] - box[0]) / xsize;
        a[4] = (double) (box[3] - box[1]) / ysize;
        a[2] = box[0];
        a[5] = box[1];

        /*imOut = ImagingNewDirty(imIn->mode, xsize, ysize);

        imOut = ImagingTransform(
            imOut, imIn, IMAGING_TRANSFORM_AFFINE,
            0, 0, xsize, ysize,
            a, filter, 1);*/
    }
    else {
        return ImagingResample(pIn, pOut,inpWd, inpHt, inpStride, xsize, ysize, outStride, filter, box, imType, channels, letterBox, pad_top, pad_left);
    }

    return 0;
}


int main(int argc, char *argv[])
{

    unsigned char *pIn, *pOut;
    int ret, i,inpWd,inpHt,inpStride,outWd,outHt,outStride,nCh, targetOutWd, targetOutHt, letterBox, pad_top, pad_left;
    cv::Mat pInImage;
    unsigned char *outData;
    memory_initialization_gpu();

    ///////////////////////////////////////// Test Cases RUN ///////////////////////////////////////

    //Test case-1 - Different size resize without Letterboxing
    //Input Information
    inpWd = 1881;
    inpHt = 926;
    inpStride = 1881;

    //Target size and channels
    targetOutWd = 222;
    targetOutHt = 112;
    nCh = 3;

    //LetterBox
    letterBox = 0;
    pad_top = 5;
    pad_left = 5;


    if(letterBox){
        outWd = targetOutWd - (pad_left * 2);
        outHt = targetOutHt - (pad_top * 2);
        outStride = outWd;
    }else{
        outWd = targetOutWd;
        outHt = targetOutHt;
        outStride = outWd;
    }

    if(nCh == 1){
        pInImage = imread("./LM_crop.jpg", IMREAD_GRAYSCALE);
    }
    else{
        pInImage = imread("./LM_crop.jpg");
    }

    cudaMemcpy( pIn_gpu , pInImage.data ,inpHt * inpWd * nCh * sizeof(unsigned char),cudaMemcpyHostToDevice);
    ret = resizeModPIL(pIn_gpu, pOut_gpu,inpWd, inpHt, inpStride, outWd, outHt, outStride, nCh,letterBox, pad_top, pad_left);
    printf("return status = %d\n", ret);

    if(!letterBox){
        targetOutWd = outWd;
        targetOutHt = outHt;
    }

    outData = (unsigned char*)malloc(targetOutWd * targetOutHt * nCh * sizeof(unsigned char));
    cudaMemcpy( outData , pOut_gpu ,targetOutWd * targetOutHt * nCh * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    if(nCh == 1){
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC1, outData);
        imwrite("./TestCase1.jpg", outImage);
    }
    else{
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC3, outData);
        imwrite("./TestCase1.jpg", outImage);
    }




    //Test case -2 - Smaller and different size resize without letterbox
    //Input Information
    inpWd = 1280;
    inpHt = 853;
    inpStride = 1280;

    //Target size and channels
    targetOutWd = 22;
    targetOutHt = 32;
    nCh =3;

    //LetterBox
    letterBox = 0;
    pad_top = 5;
    pad_left = 5;


    if(letterBox){
        outWd = targetOutWd - (pad_left * 2);
        outHt = targetOutHt - (pad_top * 2);
        outStride = outWd;
    }else{
        outWd = targetOutWd;
        outHt = targetOutHt;
        outStride = outWd;
    }

    cv::Mat pInImage2;
    if(nCh == 1){
        pInImage2 = imread("./carImage.jpg", IMREAD_GRAYSCALE);
    }
    else{
        pInImage2 = imread("./carImage.jpg");
    }

    cudaMemcpy( pIn_gpu , pInImage2.data ,inpHt * inpWd * nCh * sizeof(unsigned char),cudaMemcpyHostToDevice);
    ret = resizeModPIL(pIn_gpu, pOut_gpu,inpWd, inpHt, inpStride, outWd, outHt, outStride, nCh, letterBox, pad_top, pad_left);
    printf("return status = %d\n", ret);

    if(!letterBox){
        targetOutWd = outWd;
        targetOutHt = outHt;
    }

    unsigned char *outData2 = (unsigned char*)malloc(targetOutWd * targetOutHt * nCh * sizeof(unsigned char));
    cudaMemcpy( outData2 , pOut_gpu ,targetOutWd * targetOutHt * nCh * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    if(nCh == 1){
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC1, outData2);
        imwrite("./TestCase2.jpg", outImage);
    }
    else{
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC3, outData2);
        imwrite("./TestCase2.jpg", outImage);
    }



    //Test case-3 - Smaller size image to upsample and Letterboxing with RGB image
    //Input Information
    inpWd = 22;
    inpHt = 32;
    inpStride = 22;

    //Target size and channels
    targetOutWd = 41;
    targetOutHt = 35;
    nCh = 3;

    //LetterBox
    letterBox = 1;
    pad_top = 5;
    pad_left = 5;


    if(letterBox){
        outWd = targetOutWd - (pad_left * 2);
        outHt = targetOutHt - (pad_top * 2);
        outStride = outWd;
    }else{
        outWd = targetOutWd;
        outHt = targetOutHt;
        outStride = outWd;
    }


    if(nCh == 1){
        pInImage = imread("./TestCase2.jpg", IMREAD_GRAYSCALE);
    }
    else{
        pInImage = imread("./TestCase2.jpg");
    }

    cudaMemcpy( pIn_gpu , pInImage.data ,inpHt * inpWd * nCh * sizeof(unsigned char),cudaMemcpyHostToDevice);
    ret = resizeModPIL(pIn_gpu, pOut_gpu,inpWd, inpHt, inpStride, outWd, outHt, outStride, nCh,letterBox, pad_top, pad_left);
    printf("return status = %d\n", ret);

    if(!letterBox){
        targetOutWd = outWd;
        targetOutHt = outHt;
    }

    outData = (unsigned char*)malloc(targetOutWd * targetOutHt * nCh * sizeof(unsigned char));
    cudaMemcpy( outData , pOut_gpu ,targetOutWd * targetOutHt * nCh * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    if(nCh == 1){
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC1, outData);
        imwrite("./TestCase3.jpg", outImage);
    }
    else{
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC3, outData);
        imwrite("./TestCase3.jpg", outImage);
    }



    //Test case-4 - Letterboxing with RGB image and different pad size
    //Input Information
    inpWd = 1280;
    inpHt = 853;
    inpStride = 1280;

    //Target size and channels
    targetOutWd = 160;
    targetOutHt = 60;
    nCh = 3;

    //LetterBox
    letterBox = 1;
    pad_top = 25;
    pad_left = 15;


    if(letterBox){
        outWd = targetOutWd - (pad_left * 2);
        outHt = targetOutHt - (pad_top * 2);
        outStride = outWd;
    }else{
        outWd = targetOutWd;
        outHt = targetOutHt;
        outStride = outWd;
    }


    if(nCh == 1){
        pInImage = imread("./carImage.jpg", IMREAD_GRAYSCALE);
    }
    else{
        pInImage = imread("./carImage.jpg");
    }

    cudaMemcpy( pIn_gpu , pInImage.data ,inpHt * inpWd * nCh * sizeof(unsigned char),cudaMemcpyHostToDevice);
    ret = resizeModPIL(pIn_gpu, pOut_gpu,inpWd, inpHt, inpStride, outWd, outHt, outStride, nCh,letterBox, pad_top, pad_left);
    printf("return status = %d\n", ret);

    if(!letterBox){
        targetOutWd = outWd;
        targetOutHt = outHt;
    }

    outData = (unsigned char*)malloc(targetOutWd * targetOutHt * nCh * sizeof(unsigned char));
    cudaMemcpy( outData , pOut_gpu ,targetOutWd * targetOutHt * nCh * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    if(nCh == 1){
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC1, outData);
        imwrite("./TestCase4.jpg", outImage);
    }
    else{
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC3, outData);
        imwrite("./TestCase4.jpg", outImage);
    }



    //Test case-5 - Without letterboxing  RGB image and different pad size
    //Input Information
    inpWd = 1280;
    inpHt = 853;
    inpStride = 1280;

    //Target size and channels
    targetOutWd = 160;
    targetOutHt = 60;
    nCh = 3;

    //LetterBox
    letterBox = 0;
    pad_top = 25;
    pad_left = 15;


    if(letterBox){
        outWd = targetOutWd - (pad_left * 2);
        outHt = targetOutHt - (pad_top * 2);
        outStride = outWd;
    }else{
        outWd = targetOutWd;
        outHt = targetOutHt;
        outStride = outWd;
    }


    if(nCh == 1){
        pInImage = imread("./carImage.jpg", IMREAD_GRAYSCALE);
    }
    else{
        pInImage = imread("./carImage.jpg");
    }

    cudaMemcpy( pIn_gpu , pInImage.data ,inpHt * inpWd * nCh * sizeof(unsigned char),cudaMemcpyHostToDevice);
    ret = resizeModPIL(pIn_gpu, pOut_gpu,inpWd, inpHt, inpStride, outWd, outHt, outStride, nCh,letterBox, pad_top, pad_left);
    printf("return status = %d\n", ret);

    if(!letterBox){
        targetOutWd = outWd;
        targetOutHt = outHt;
    }

    outData = (unsigned char*)malloc(targetOutWd * targetOutHt * nCh * sizeof(unsigned char));
    cudaMemcpy( outData , pOut_gpu ,targetOutWd * targetOutHt * nCh * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    if(nCh == 1){
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC1, outData);
        imwrite("./TestCase5.jpg", outImage);
    }
    else{
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC3, outData);
        imwrite("./TestCase5.jpg", outImage);
    }


    //Test case-6 - With letterboxing  Gray scale image and different pad size
    //Input Information
    inpWd = 1280;
    inpHt = 853;
    inpStride = 1280;

    //Target size and channels
    targetOutWd = 160;
    targetOutHt = 60;
    nCh = 1;

    //LetterBox
    letterBox = 1;
    pad_top = 25;
    pad_left = 15;


    if(letterBox){
        outWd = targetOutWd - (pad_left * 2);
        outHt = targetOutHt - (pad_top * 2);
        outStride = outWd;
    }else{
        outWd = targetOutWd;
        outHt = targetOutHt;
        outStride = outWd;
    }


    if(nCh == 1){
        pInImage = imread("./carImage.jpg", IMREAD_GRAYSCALE);
    }
    else{
        pInImage = imread("./carImage.jpg");
    }

    cudaMemcpy( pIn_gpu , pInImage.data ,inpHt * inpWd * nCh * sizeof(unsigned char),cudaMemcpyHostToDevice);
    ret = resizeModPIL(pIn_gpu, pOut_gpu,inpWd, inpHt, inpStride, outWd, outHt, outStride, nCh,letterBox, pad_top, pad_left);
    printf("return status = %d\n", ret);

    if(!letterBox){
        targetOutWd = outWd;
        targetOutHt = outHt;
    }

    outData = (unsigned char*)malloc(targetOutWd * targetOutHt * nCh * sizeof(unsigned char));
    cudaMemcpy( outData , pOut_gpu ,targetOutWd * targetOutHt * nCh * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    if(nCh == 1){
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC1, outData);
        imwrite("./TestCase6.jpg", outImage);
    }
    else{
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC3, outData);
        imwrite("./TestCase6.jpg", outImage);
    }


    //Test case-7 - Without letterBoxing gray scale image
    //Input Information
    inpWd = 1881;
    inpHt = 926;
    inpStride = 1881;

    //Target size and channels
    targetOutWd = 222;
    targetOutHt = 112;
    nCh = 1;

    //LetterBox
    letterBox = 0;
    pad_top = 5;
    pad_left = 5;


    if(letterBox){
        outWd = targetOutWd - (pad_left * 2);
        outHt = targetOutHt - (pad_top * 2);
        outStride = outWd;
    }else{
        outWd = targetOutWd;
        outHt = targetOutHt;
        outStride = outWd;
    }

    if(nCh == 1){
        pInImage = imread("./LM_crop.jpg", IMREAD_GRAYSCALE);
    }
    else{
        pInImage = imread("./LM_crop.jpg");
    }

    cudaMemcpy( pIn_gpu , pInImage.data ,inpHt * inpWd * nCh * sizeof(unsigned char),cudaMemcpyHostToDevice);
    ret = resizeModPIL(pIn_gpu, pOut_gpu,inpWd, inpHt, inpStride, outWd, outHt, outStride, nCh,letterBox, pad_top, pad_left);
    printf("return status = %d\n", ret);

    if(!letterBox){
        targetOutWd = outWd;
        targetOutHt = outHt;
    }

    outData = (unsigned char*)malloc(targetOutWd * targetOutHt * nCh * sizeof(unsigned char));
    cudaMemcpy( outData , pOut_gpu ,targetOutWd * targetOutHt * nCh * sizeof(unsigned char),cudaMemcpyDeviceToHost);

    if(nCh == 1){
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC1, outData);
        imwrite("./TestCase7.jpg", outImage);
    }
    else{
        cv::Mat outImage =  cv::Mat(Size(targetOutWd, targetOutHt), CV_8UC3, outData);
        imwrite("./TestCase7.jpg", outImage);
    }

    ///////////////////////////////////////// Test Cases End///////////////////////////////////////

    return 0;
}
