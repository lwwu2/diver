#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"  // required for float3 vector math
#define NUM_THREADS 1024

__device__ int sign(float x) { 

	int t = x<0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

/**
 * ray marching
 * @param batch_size number of pixels
 * @param max_n maximum possible intersections
 * @param xyzmin min volume bound
 * @param xyzmax max volume bound
 * @param voxel_num voxel grid size
 * @param voxel_size size of each voxel
 * @param o ray origins
 * @param v ray directions
 * @param intersection ray-voxel intersection buffer
 * @param intersection_num number of intersections for each ray
 * @param tns distance between ray origin and intersected location
 */
__global__ void ray_voxel_intersect_kernel(
            int batch_size, int max_n,
            const float xyzmin, const float xyzmax, 
            const float voxel_num, const float voxel_size,
            const float *__restrict__ o,
            const float *__restrict__ v,
            float *__restrict__ intersection,
            int *__restrict__ intersect_num,
            float *__restrict__ tns) {

    int batch_index = (blockIdx.x * blockDim.x) + threadIdx.x; 

    if(batch_index >= batch_size){
        return;
    }

    // set up index
    o += batch_index*3;
    v += batch_index*3;
    intersection += batch_index*max_n*3;
    intersect_num += batch_index;
    tns += batch_index*max_n;

    // assume size of each voxel to be one
    float ox = (o[0]-xyzmin) / voxel_size;
    float oy = (o[1]-xyzmin) / voxel_size;
    float oz = (o[2]-xyzmin) / voxel_size;


    float3 dir = make_float3(v[0],v[1],v[2]);
    float3 ori = make_float3(ox,oy,oz);

    bool is_inside = (ox >= 0) & (oy >= 0) & (oz >= 0) 
                & (ox <= voxel_num) & (oy <= voxel_num) & (oz <= voxel_num);

    if (is_inside) { // if inside the volume
        intersection[0] = ori.x;
        intersection[1] = ori.y;
        intersection[2] = ori.z;
        tns[0] = 0.0f;
    } else {
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

        ori.x = clamp(ori.x+dir.x*tmin, 0.0f, voxel_num);
        ori.y = clamp(ori.y+dir.y*tmin, 0.0f, voxel_num);
        ori.z = clamp(ori.z+dir.z*tmin, 0.0f, voxel_num);

        intersection[0] = ori.x;
        intersection[1] = ori.y;
        intersection[2] = ori.z;

        // a miss, exit
        if (tmin > tmax) {
            return;
        } else {
            tns[0] = tmin;
        }
    }

    float t_now = tns[0];
    tns[0] *= voxel_size;
    intersect_num[0] += 1;

    intersection += 3;
    tns += 1;

    float3 step = make_float3(sign(dir.x), sign(dir.y), sign(dir.z));
    float3 bound;

    float tx;
    float ty;
    float tz;
    float tnext;


    while (true) {
        bound = floor(ori*step+1.0f)*step; // get candidate bounds for next intersection
        tx = (bound.x-ori.x) / dir.x;
        ty = (bound.y-ori.y) / dir.y;
        tz = (bound.z-ori.z) / dir.z;

        tnext = fminf(tx, fminf(ty,tz));
        ori += (dir*tnext);
        t_now += tnext;

        // enforce the point to be at the hitted plane (drifting error introduced) 
        if (tnext == tx) {
            ori.x = bound.x;
        } else if (tnext == ty) {
            ori.y = bound.y;
        } else { 
            ori.z = bound.z;
        }

        if (ori.x < 0 | ori.y < 0 | ori.z < 0 | 
            ori.x > voxel_num | ori.y > voxel_num | ori.z > voxel_num) {
            return;
        }

        intersection[0] = ori.x;
        intersection[1] = ori.y;
        intersection[2] = ori.z;
        intersect_num[0] += 1;
        tns[0] = t_now*voxel_size;

        intersection += 3;
        tns += 1;
    }
}


void ray_voxel_intersect_wrapper(
  int device_id,
  int batch_size, int max_n,
  const float xyzmin, const float xyzmax, 
  const float voxel_num, const float voxel_size,
  const float *o, const float *v, 
  float *intersection, int *intersect_num, float *tns){

  cudaSetDevice(device_id);

  ray_voxel_intersect_kernel<<<ceil(batch_size*1.0 / NUM_THREADS), NUM_THREADS>>>(
      batch_size, max_n,
      xyzmin, xyzmax, voxel_num, voxel_size,
      o, v, 
      intersection, intersect_num, tns);
  
  CUDA_CHECK_ERRORS();
  cudaDeviceSynchronize();
}



/**
 * ray marching with occupancy mask
 * @param batch_size number of pixels
 * @param max_n maximum possible intersections
 * @param xyzmin min volume bound
 * @param xyzmax max volume bound
 * @param voxel_num voxel grid size
 * @param voxel_size size of each voxel
 * @param mask_scale relative scale of the mask in respect to the voxel grid
 * @param o ray origins
 * @param v ray directions
 * @param mask occupancy mask
 * @param intersection ray-voxel intersection buffer
 * @param intersection_num number of intersections for each ray
 * @param tns distance between ray origin and intersected location
 */
 __global__ void masked_intersect_kernel(
    int batch_size, int max_n,
    const float xyzmin, const float xyzmax, 
    const float voxel_num, const float voxel_size, const float mask_scale,
    const float *__restrict__ o,
    const float *__restrict__ v,
    const bool *__restrict__ mask,
    float *__restrict__ intersection,
    int *__restrict__ intersect_num,
    float *__restrict__ tns) {

    int batch_index = (blockIdx.x * blockDim.x) + threadIdx.x; 
    
    if(batch_index >= batch_size){
        return;
    }
    
    // set up index
    o += batch_index*3;
    v += batch_index*3;
    intersection += batch_index*max_n*6;
    intersect_num += batch_index;
    tns += batch_index*max_n;

    float ox = (o[0]-xyzmin) / voxel_size;
    float oy = (o[1]-xyzmin) / voxel_size;
    float oz = (o[2]-xyzmin) / voxel_size;

    
    float3 dir = make_float3(v[0],v[1],v[2]);
    float3 ori = make_float3(ox,oy,oz);
    float3 ori_last;
    float t_now;

    bool is_inside = (ox >= 0) & (oy >= 0) & (oz >= 0) 
                    & (ox <= voxel_num) & (oy <= voxel_num) & (oz <= voxel_num);

    if (is_inside) {
        ori_last = make_float3(ori.x, ori.y, ori.z);
        t_now = 0.0f;
    } else {
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

        ori.x = clamp(ori.x+dir.x*tmin, 0.0f, voxel_num);
        ori.y = clamp(ori.y+dir.y*tmin, 0.0f, voxel_num);
        ori.z = clamp(ori.z+dir.z*tmin, 0.0f, voxel_num);
        ori_last = make_float3(ori.x,ori.y,ori.z);
        // a miss, exit
        if (tmin > tmax) {
            return;
        } else {
            t_now = tmin;
        }
    }

    float3 step = make_float3(sign(dir.x), sign(dir.y), sign(dir.z));
    float3 bound;

    float tx;
    float ty;
    float tz;
    float tnext;
    int mask_size = int(voxel_num * mask_scale);


    while (true) {
        bound = floor(ori_last*step+1.0f)*step; // get candidate bounds for next intersection
        tx = (bound.x-ori_last.x) / dir.x;
        ty = (bound.y-ori_last.y) / dir.y;
        tz = (bound.z-ori_last.z) / dir.z;

        tnext = fminf(tx, fminf(ty,tz));
        ori = ori_last + (dir*tnext);
        t_now += tnext;

        // enforce the point to be at the hitted plane (drifting error introduced) 
        if (tnext == tx) {
            ori.x = bound.x;
        } else if (tnext == ty) {
            ori.y = bound.y;
        } else { 
            ori.z = bound.z;
        }

        // check if exceed the boundary
        if (ori.x < 0 | ori.y < 0 | ori.z < 0 | 
            ori.x > voxel_num | ori.y > voxel_num | ori.z > voxel_num) {
            return;
        }

        // accurate voxel corner calculation
        float3 corner = fminf(ori_last+1e-4f,ori+1e-4f)*mask_scale;
        
        int corner_index = int(corner.z)*mask_size*mask_size + int(corner.y)*mask_size + int(corner.x);
        
        if (mask[corner_index]) {// check whether the voxel is an empty space
            intersection[0] = ori_last.x;
            intersection[1] = ori_last.y;
            intersection[2] = ori_last.z;
            intersection[3] = ori.x;
            intersection[4] = ori.y;
            intersection[5] = ori.z;
            intersect_num[0] += 1;
            tns[0] = t_now*voxel_size;

            intersection += 6;
            tns += 1;
        }

        ori_last.x = ori.x;
        ori_last.y = ori.y;
        ori_last.z = ori.z;
    }
}



void masked_intersect_wrapper(
    int device_id,
    int batch_size, int max_n,
    const float xyzmin, const float xyzmax, 
    const float voxel_num, const float voxel_size, const float mask_scale,
    const float *o, const float *v, const bool *mask,
    float *intersection, int *intersect_num, float *tns){

    cudaSetDevice(device_id);

    masked_intersect_kernel<<<ceil(batch_size*1.0 / NUM_THREADS), NUM_THREADS>>>(
    batch_size, max_n,
    xyzmin, xyzmax, voxel_num, voxel_size, mask_scale,
    o, v, mask, 
    intersection, intersect_num, tns);

    CUDA_CHECK_ERRORS();
    cudaDeviceSynchronize();
}
