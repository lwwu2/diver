#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"


#define NUM_BLOCKS 1024
#define MAX_HIT 8
#define PARAM_SIZE 3168


__device__ __forceinline__ float sigmoid (float x)
{
    return 1.0 / (1.0 + __expf (-x));
}

static texture <int, 3, cudaReadModeElementType> c_voxel_map; // voxel index map
__device__ __constant__ float params_cache[PARAM_SIZE]; // mlp weights


/**
 * Performs mlp evaluation of single pass
 * @param batch_size number of pixels to evaluate
 * @param coord buffer that stores the [entry, exit] location of a voxel intersection
 * @param voxels 1D array of feature vectors
 * @param v viewing direction buffer
 * @param rgba rbga output buffer
 * @param mask mask of pixels that have finished the evaluation
 */
__global__ void mlp_eval_kernel(
            int batch_size,
            const float *__restrict__ coord,
            const float *__restrict__ voxels,
            const float *__restrict__ v,
            float *__restrict__ rgba,
            bool* __restrict__ mask
        ) {
    
    constexpr float frequency_bands[4] 
        = {3.141592653589793f,6.283185307179586f,12.566370614359172f,25.132741228718345f};
    
    int b_idx = blockIdx.x;
    int tx = ((b_idx%50)*blockDim.x) + threadIdx.x;
    int ty = ((b_idx/50)*blockDim.y) + threadIdx.y;

    int batch_index = 800*ty + tx; 
    
    while (batch_index < batch_size) {
        if (mask[batch_index]) { // check if current pixel need to be evaluated
            b_idx += NUM_BLOCKS;
            tx = ((b_idx%50)*blockDim.x) + threadIdx.x;
            ty = ((b_idx/50)*blockDim.y) + threadIdx.y;
            batch_index = 800*ty + tx;
            continue;
        }
        for (int hit=0; hit < MAX_HIT; hit++) { // sequantially evaluate all the hits
            int mask_idx = batch_index*6 + hit*batch_size*6;
    
            float x0 = coord[mask_idx];
            float y0 = coord[mask_idx+1];
            float z0 = coord[mask_idx+2];
            float x1 = coord[mask_idx+3];
            float y1 = coord[mask_idx+4];
            float z1 = coord[mask_idx+5];
            if (x0 < 0) { // we use x0 = -1 as an inicator of the traversal is over
                mask[batch_index] = true;
                break;
            }
    
            int cx = min(int(x0+1e-4f),int(x1+1e-4f)); // accurate boxel index calculation
            int cy = min(int(y0+1e-4f),int(y1+1e-4f));
            int cz = min(int(z0+1e-4f),int(z1+1e-4f));
            x0 -= cx;
            x1 -= cx;
            y0 -= cy;
            y1 -= cy;
            z0 -= cz;
            z1 -= cz;

            // calculate weights for eight vertices
            float z01 = z0 + z1;
            float y01 = y0 + y1;
            float x01 = x0 + x1;
            float w[8];
            w[7]=(2*x0*y0*z0+2*x1*y1*z1+x01*y01*z01)/12;
            float w2=(y01*z01+y0*z0+y1*z1)/6;
            w[5]=(x01*z01+x0*z0+x1*z1)/6-w[7];
            float w4=(x01*y01+x0*y0+x1*y1)/6;
    
            x01 *= 0.5f;
            y01 *= 0.5f;
            z01 *= 0.5f;
    
            w[4] = z01-w[5]-w2;
            w[0] = 1-w[4]-y01-x01+w4;
            w[6] = w2-w[7];
            w[2] = y01-w[6]-w4;
            w[1] = x01 -w[5] - w4;
            w[3] = -w[7] +w4;
    
            float buffer1[32];
            int param_offset = 0;
            // load features
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                buffer1[i] = params_cache[param_offset];
                param_offset += 1;
            }

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int dx = j%2+cx;
                int dy = (j/2)%2+cy;
                int dz = j/4+cz;
                
                int v_idx = tex3D(c_voxel_map,dx,dy,dz);
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    buffer1[i] += w[j]*voxels[v_idx*32+i];
                }
            }
    
            // first mlp evaluation
            float buffer2[32];
            #pragma unroll
            for (int i=0; i < 32; i++) {
                buffer2[i] = params_cache[param_offset];
                param_offset += 1;
            }
    
            #pragma unroll
            for (int i=0; i < 32; i++) {
                #pragma unroll
                for (int j=0; j < 32; j++) {
                    buffer2[j] += params_cache[param_offset]*fmaxf(buffer1[i],0.0f);
                    param_offset += 1;
                }
            }
    
            // density mlp evaluation
            float sigma = params_cache[param_offset];
            param_offset += 1;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                sigma += params_cache[param_offset]*fmaxf(buffer2[i],0.0f);
                param_offset += 1;
            }
            sigma = -fmaxf(sigma,0.0f);
            sigma = 1.0f - __expf(sigma);
            
            if (sigma < 1e-2f) { // if alpha is low, no need to calculate color
                continue;
            }
    
            // second mlp evaluation
            float buffer3[32];
            #pragma unroll
            for (int i =0; i < 32; i++) {
                buffer3[i] = params_cache[param_offset];
                param_offset += 1;
            }
            #pragma unroll
            for (int i =0; i < 32; i++) {
                #pragma unroll
                for (int j =0; j < 32; j++) {
                    buffer3[j] += params_cache[param_offset]*fmaxf(buffer2[i],0.0f);
                    param_offset += 1;
                }
            }
            // cat view dependency
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                float input_elem = v[batch_index*3+i];
                #pragma unroll
                for (int e =0; e <9; e++) {
                    float embedded_input_elem;
                    if (e == 0) {
                        embedded_input_elem = input_elem;
                    } else if (e < 5) {
                        embedded_input_elem = __sinf(frequency_bands[e-1]*input_elem);
                    } else {
                        embedded_input_elem = __cosf(frequency_bands[e-5]*input_elem);
                    }
    
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        buffer3[j] += params_cache[param_offset]*embedded_input_elem;
                        param_offset += 1;
                    }
                }
            }
    
            // last mlp evaluation
            float buffer4[3];
            #pragma unroll
            for (int i=0; i < 3; i++ ) {
                buffer4[i] = params_cache[param_offset];
                param_offset += 1;
            }
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                #pragma unroll
                for (int j =0; j < 3; j++) {
                    buffer4[j] += params_cache[param_offset]*fmaxf(buffer3[i],0.0f);
                    param_offset += 1;
                }
            }
    
            // update rgba buffer
            float acc_sigma = rgba[batch_index*4+3];
            float new_sigma = acc_sigma*sigma;
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                rgba[batch_index*4+i] += new_sigma*sigmoid(buffer4[i]);
            }
            new_sigma = acc_sigma*(1-sigma);
            rgba[batch_index*4+3] = new_sigma;
            if (new_sigma < 1e-2f) { //early ray termination criteria checking
                mask[batch_index] = true;
                break;
            }
        }

        b_idx += NUM_BLOCKS;
        tx = ((b_idx%50)*blockDim.x) + threadIdx.x;
        ty = ((b_idx/50)*blockDim.y) + threadIdx.y;
        batch_index = 800*ty + tx;
    }
}


void mlp_eval_wrapper(
  int batch_size,
  const float* coord,
  const float* voxels,
  const float* v,
  float* rgba,
  bool* mask
){

  dim3 block(16,24);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  mlp_eval_kernel<<<NUM_BLOCKS, block,0,stream>>>(
      batch_size,
      coord,
      voxels,
      v,
      rgba,mask);
  
  CUDA_CHECK_ERRORS();
  cudaDeviceSynchronize();
}



/**
* upload mlp weights and voxel index map
* @param device_id cuda device id
* @param map_size size of the voxel index map
* @param voxel_map voxel index map
*/
void upload_weight_wrapper(
    int device_id,
    int map_size,
    const float* params,
    const int* voxel_map
) {
    cudaSetDevice(device_id);



    // allocate map
    cudaChannelFormatDesc cf = cudaCreateChannelDesc<int>();
    cudaArray * voxel_map_array = 0;
    cudaMalloc3DArray(&voxel_map_array,&cf,make_cudaExtent(map_size,map_size,map_size),0);
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)voxel_map,map_size*sizeof(int),map_size,map_size);
    copyParams.dstArray = voxel_map_array;
    copyParams.extent = make_cudaExtent(map_size,map_size,map_size);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    
    c_voxel_map.normalized = 0;
    c_voxel_map.filterMode = cudaFilterModePoint;
    c_voxel_map.addressMode[0] = cudaAddressModeClamp;
    c_voxel_map.addressMode[1] = cudaAddressModeClamp;
    c_voxel_map.addressMode[2] = cudaAddressModeClamp;
    cudaBindTextureToArray(c_voxel_map,voxel_map_array,cf); 
    

    cudaMemcpyToSymbol(params_cache,params,PARAM_SIZE*sizeof(float),0,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

