/*
 *
 *	Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 * 
 *	This file is part of nPhysImage library.
 *
 *	nPhysImage is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU Lesser General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	(at your option) any later version.
 *
 *	nPhysImage is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public License
 *	along with neutrino.  If not, see <http://www.gnu.org/licenses/>.
 *
 *	Contact Information: 
 *	Alessandro Flacco <alessandro.flacco@polytechnique.edu>
 *	Tommaso Vinci <tommaso.vinci@polytechnique.edu>
 *
 */
#include "nCuda.h"
#include "math_constants.h"

#define BLOCKSIZE 256

__global__ void gabor_kernel(cufftComplex* in, cufftComplex* out, int dx, int dy,  int dx_2, int dy_2, float sr, float cr, float lambda_norm, float damp_norm, float thick_norm) {
	int l = blockDim.x * blockIdx.x + threadIdx.x; 
	if (l < dx*dy) {
		int i,j;
		j=l/dx;
		i=l%dx; //swap and center
		if (i>=dx_2) i-=dx;
		if (j>=dy_2) j-=dy;
		float xr=i*cr-j*sr; //rotate
		float yr=i*sr+j*cr; 
		float gauss= expf(-pow(damp_norm*(xr*lambda_norm-1.0f),2))*expf(-pow(yr*thick_norm,2));						
		out[l].x = gauss*in[l].x; 
		out[l].y = gauss*in[l].y; 
	}
}

void gabor (cufftComplex* in, cufftComplex* out, int dx, int dy, float arad, float lambda, float damp, float thickness) {
	dim3 grid(ceil(dx*dy/BLOCKSIZE));
	dim3 block(BLOCKSIZE);
	float sr,cr;
	sincos(arad,&sr,&cr);

	float damp_norm=damp*CUDART_PI_F;

// 	float thick_norm=thickness*CUDART_PI_F/dy;
// 	float lambda_norm=lambda/dx;
	float thick_norm=thickness*CUDART_PI_F/sqrt(pow(sr*dx,2)+pow(cr*dy,2));
	float lambda_norm=lambda/sqrt(pow(cr*dx,2)+pow(sr*dy,2));
	gabor_kernel<<< grid, block >>>(in, out, dx, dy, dx/2, dy/2, sr, cr, lambda_norm, damp_norm, thick_norm);
}


__global__ void fase_kernel(cufftComplex* in,cufftComplex* ref, cufftComplex* out, int dxdy, float adeg, float lambda) {
	int l = blockDim.x * blockIdx.x + threadIdx.x; 
	if (l < dxdy) {
		float intensity=pow(in[l].x,2)+pow(in[l].y,2);
		if (ref[l].x<intensity){
			ref[l].x = intensity;
			ref[l].y = atan2(in[l].y,in[l].x);
			out[l].x = adeg;
			out[l].y = lambda;
		} 
	}
}

void fase (cufftComplex* in, cufftComplex* ref,  cufftComplex* out, int dxdy, float adeg, float lambda) {
	dim3 grid(ceil(dxdy/BLOCKSIZE));
	dim3 block(BLOCKSIZE);
	fase_kernel<<< grid, block >>>(in, ref, out, dxdy,  adeg, lambda);
}

// __global__ void blur_kernel(cufftComplex* in, cufftComplex* out, int dx, int dy,  int dx_2, int dy_2, float sx, float sy)
// {
// 	int l = blockDim.x * blockIdx.x + threadIdx.x; 
// 	if (l < dx*dy) {
//   	int i,j;
// 		j=l/dx;
// 		i=l%dx; //swap and center
// 		if (i>=dx_2) i-=dx;
// 		if (j>=dy_2) j-=dy;
// 		
// 		float blur= expf(-(i*i)/sx)*expf(-(j*j)/sy);						
// 		out[l].x = blur*in[l].x; 
// 		out[l].y = blur*in[l].y; 
// 	}
// }
// 
// void blur (cufftComplex* in, cufftComplex* out, int dx, int dy, float radius) {
// 	dim3 grid(ceil(dx*dy/BLOCKSIZE));
// 	dim3 block(BLOCKSIZE);
// 
// 
// 	float sx=pow(dx/(radius*CUDART_PI_F),2)/2.0;
// 	float sy=pow(dy/(radius*CUDART_PI_F),2)/2.0;
// 
// 	blur_kernel<<< grid, block >>>(in, out, dx, dy, dx/2, dy/2, sx, sy);
// }
// 
