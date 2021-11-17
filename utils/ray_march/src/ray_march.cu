#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"
#define NUM_THREADS 1024
#define NUM_BLOCKS 256
#define MAX_HIT 8

__device__ __constant__ float c_center[3];

__device__ __forceinline__ int sign(float x) { 

	int t = x<0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

/**
 * the ray marching code is similar to the one included in the real-time application, except 
 * the rotation matrix is calculated outside the CUDA routine.
 * Better commented version is provided in the diver-rt repo
 */
template<int voxel_num>
__global__ void aabb_intersect_kernel(
            const int batch_size,
            float *__restrict__ o,
            const float *__restrict__ v,
            bool* __restrict__ finish,
            float* __restrict__ rgba,
            const bool* __restrict__ octrees) {

    int batch_index = (blockIdx.x * blockDim.x) + threadIdx.x; 

    if(batch_index >= batch_size){
        return;
    }

    // set up index
    o += batch_size*6*(MAX_HIT-1) + batch_index*6 + 3;
    v += batch_index*3;
    finish += batch_index;
    rgba += batch_index*4;
    rgba[0] = 0.0f;
    rgba[1] = 0.0f;
    rgba[2] = 0.0f;
    rgba[3] = 1.0f;
    finish[0] = false;

    float ox = c_center[0];
    float oy = c_center[1];
    float oz = c_center[2];


    float3 dir = make_float3(v[0],v[1],v[2]);
    float3 ori = make_float3(ox,oy,oz);

    bool is_inside = (ori.x >= 0) & (ori.y >= 0) & (ori.z >= 0) 
                & (ori.x <= voxel_num) & (ori.y <= voxel_num) & (ori.z <= voxel_num);

    if (is_inside==false) {
        // ray bounding volume intersection
        float t0 = (-ori.x)/dir.x;
        float t1 = (voxel_num-ori.x)/dir.x;
        float tmin = fminf(t0,t1);
        float tmax = fmaxf(t0,t1);

        t0 = (-ori.y)/dir.y;
        t1 = (voxel_num-ori.y)/dir.y;
        tmin = fmaxf(tmin, fminf(t0,t1));
        tmax = fminf(tmax, fmaxf(t0,t1));

        t0 = (-ori.z)/dir.z;
        t1 = (voxel_num-ori.z)/dir.z;
        tmin = fmaxf(tmin, fminf(t0,t1));
        tmax = fminf(tmax, fmaxf(t0,t1));

        ori.x = clamp(ori.x+dir.x*tmin, 0.0f, float(voxel_num));
        ori.y = clamp(ori.y+dir.y*tmin, 0.0f, float(voxel_num));
        ori.z = clamp(ori.z+dir.z*tmin, 0.0f, float(voxel_num));

        // a miss, exit
        if (tmin > tmax) {
            o[0] = -1.0f;
            finish[0] = true;
            return;
        }
    }
    ori /= 64.0f;
    float3 ori_last;
    float3 step = make_float3(sign(dir.x), sign(dir.y), sign(dir.z));
    float3 bound;
    float tx;
    float ty;
    float tz;
    int mask_size = voxel_num/64;
    int octree_offset = 0;

    while (mask_size < voxel_num) {
        ori_last.x = ori.x;
        ori_last.y = ori.y;
        ori_last.z = ori.z;

        bound = floor(ori_last*step+1.0f)*step;
        //bound = floor(ori_last*step+step_scale)*step;
        tx = (bound.x-ori_last.x) / dir.x;
        ty = (bound.y-ori_last.y) / dir.y;
        tz = (bound.z-ori_last.z) / dir.z;

        float tnext = fminf(tx, fminf(ty,tz));
        ori = ori_last + (dir*tnext);

        // check which planes has been hitted
        if (tnext == tx) {
            ori.x = bound.x;
        } else if (tnext ==ty) {
            ori.y = bound.y;
        } else {
            ori.z = bound.z;
        }

        // check whether go out of volume
        if (ori.x <0 | ori.y <0 | ori.z < 0 |
            ori.x > mask_size | ori.y > mask_size | ori.z > mask_size) {
            // boundary with zero padding
            o[0] = -1.0f;
            finish[0] = true;
            return;
        }

        // check if hit an empty space
        float3 corner = fminf(ori_last+1e-4f,ori+1e-4f);
        int corner_index = int(corner.z)*mask_size*mask_size
                         + int(corner.y)*mask_size 
                         + int(corner.x);

        if (octrees[octree_offset+corner_index]) {
            octree_offset += mask_size*mask_size*mask_size;
            ori_last *= 4.0f;
            ori.x = ori_last.x;
            ori.y = ori_last.y;
            ori.z = ori_last.z;
            mask_size *= 4;
        }
    }
    o[0] = ori_last.x;
    o[1] = ori_last.y;
    o[2] = ori_last.z;
}

template<int voxel_num>
__global__ void ray_march_kernel(
    int batch_size,
    float* __restrict__ o,
    const float* __restrict__ v,
    const bool* __restrict__ mask,
    bool* __restrict__ finish
) {
    int batch_index = (blockIdx.x * blockDim.x) + threadIdx.x; 
    
    while (batch_index < batch_size) {
        if (finish[batch_index]) {
            batch_index += NUM_THREADS*NUM_BLOCKS;
            continue;
        }

        int last_idx = (MAX_HIT-1)*6*batch_size+batch_index*6+3;
        float3 ori_last = make_float3(o[last_idx],o[last_idx+1],o[last_idx+2]);

        o[batch_index*6] = -1.0f;
        if (ori_last.x < 0.0f) {
            break;
        }
        
        float3 dir = make_float3(v[batch_index*3],v[batch_index*3+1],v[batch_index*3+2]);
        float3 step = make_float3(sign(dir.x), sign(dir.y), sign(dir.z));
        float3 bound;
        float3 ori;
        float tx;
        float ty;
        float tz;

        int hit_num = 0;
        while (hit_num < MAX_HIT) {
            bound = floor(ori_last*step+1.0f)*step;
            tx = (bound.x-ori_last.x) / dir.x;
            ty = (bound.y-ori_last.y) / dir.y;
            tz = (bound.z-ori_last.z) / dir.z;

            float tnext = fminf(tx, fminf(ty,tz));
            ori = ori_last + (dir*tnext);

            // check which planes has been hitted
            if (tnext == tx) {
                ori.x = bound.x;
            } else if (tnext ==ty) {
                ori.y = bound.y;
            } else {
                ori.z = bound.z;
            }

            // check whether go out of volume
            if (ori.x <0 | ori.y <0 | ori.z < 0 |
                ori.x >= voxel_num | ori.y >= voxel_num | ori.z >= voxel_num) {
                // boundary with zero padding
                o[batch_index*6+hit_num*6*batch_size] = -1.0f;
                break;
            }

            // check if hit an empty space
            float3 corner = fminf(ori_last+1e-4f,ori+1e-4f);
            int corner_index = int(corner.z)*voxel_num*voxel_num+int(corner.y)*voxel_num+int(corner.x);

            if (mask[corner_index]) {
                int rec_idx = batch_index*6+hit_num*6*batch_size;
                o[rec_idx] = ori_last.x;
                o[rec_idx+1] = ori_last.y;
                o[rec_idx+2] = ori_last.z;
                o[rec_idx+3] = ori.x;
                o[rec_idx+4] = ori.y;
                o[rec_idx+5] = ori.z;
                hit_num += 1;
            }
            ori_last.x = ori.x;
            ori_last.y = ori.y;
            ori_last.z = ori.z;
        }
        batch_index += NUM_BLOCKS*NUM_THREADS;
    }
}






void aabb_intersect_wrapper(
    const int batch_size, 
    float* o, const float* v, const float* center,
    bool* finish, float* rgba, 
    const bool* octrees,
    int voxel_num
){
    cudaMemcpyToSymbol(c_center, center, 3*sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    switch (voxel_num) {
        case 256:
            aabb_intersect_kernel<256><<<ceil(batch_size*1.0 / NUM_THREADS), NUM_THREADS,0,stream>>>(
                batch_size,
                o,v, finish, rgba,
                octrees
            );
            break;
       case 128:
           aabb_intersect_kernel<128><<<ceil(batch_size*1.0 / NUM_THREADS), NUM_THREADS,0,stream>>>(
                batch_size,
                o,v, finish, rgba,
                octrees
            );
            break;
    }
    
    CUDA_CHECK_ERRORS();
    cudaDeviceSynchronize();
  }

void ray_march_wrapper(
    const int batch_size, 
    float* o, const float* v,
    const bool* mask, bool* finish,
    int voxel_num
){

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (voxel_num) {
        case 256:
            ray_march_kernel<256><<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(
                batch_size,
                o,v,
                mask, finish
            );
            break;
       case 128:
           ray_march_kernel<128><<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(
                batch_size,
                o,v,
                mask, finish
            );
            break;
    }


    CUDA_CHECK_ERRORS();
    cudaDeviceSynchronize();
}
