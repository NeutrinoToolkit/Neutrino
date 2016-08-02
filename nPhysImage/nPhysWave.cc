/*
 *
 *    Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 * 
 *    This file is part of nPhysImage library.
 *
 *    nPhysImage is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    nPhysImage is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public License
 *    along with neutrino.  If not, see <http://www.gnu.org/licenses/>.
 *
 *    Contact Information: 
 *	Alessandro Flacco <alessandro.flacco@polytechnique.edu>
 *	Tommaso Vinci <tommaso.vinci@polytechnique.edu>
 *
 */
#include "nPhysWave.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "nCuda.h"
#endif

#include "unwrapping/unwrap_simple.h"
#include "unwrapping/unwrap_goldstein.h"
#include "unwrapping/unwrap_quality.h"

using namespace std;

/*! \addtogroup nPhysWave
 * @{
 */ 

void phys_wavelet_field_2D_morlet(wavelet_params &params)
{
	DEBUG(5,"start");
            
            
	if (params.data && (params.n_angles > 0) && (params.n_lambdas > 0) && (params.data->getSurf() != 0)) {

        params.olist.clear();
        
		int dx=params.data->getW();
		int dy=params.data->getH();

		vector<int> xx(dx), yy(dy);
		
		nPhysImageF<mcomplex> zz_morlet("zz_morlet");
	
		zz_morlet.resize(dx,dy);
		
		for (int i=0;i<dx;i++) xx[i]=(i+(dx+1)/2)%dx-(dx+1)/2; // swap and center
		for (int i=0;i<dy;i++) yy[i]=(i+(dy+1)/2)%dy-(dy+1)/2;
	
		nPhysD *qmap, *wphase, *lambda, *angle, *intensity;
		qmap = new nPhysD(dx, dy, 0.0, "quality");
		
		wphase = new nPhysD("phase/2pi");
		wphase->resize(dx, dy);
		lambda = new nPhysD("lambda");
		lambda->resize(dx, dy);
		angle = new nPhysD("angle");
		angle->resize(dx, dy);
		intensity = new nPhysD("intensity");
		intensity->resize(dx, dy);


        vector<double> angles(params.n_angles), lambdas(params.n_lambdas);
        if (params.n_angles==1) {
			angles[0]=0.5*(params.end_angle+params.init_angle);
		} else {
			for (size_t i=0; i<params.n_angles; i++)
				angles[i] = params.init_angle + i*(params.end_angle-params.init_angle)/(params.n_angles-1);
		}	
		if (params.n_lambdas==1) {
			lambdas[0]=0.5*(params.end_lambda+params.init_lambda);
		} else {
			for (size_t i=0; i<params.n_lambdas; i++)
				lambdas[i] = params.init_lambda + i*(params.end_lambda-params.init_lambda)/(params.n_lambdas-1);
		}
		
		nPhysImageF<mcomplex> Fmain_window=params.data->ft2(PHYS_FORWARD);
		
		params.iter=0;
		*params.iter_ptr=0;
		
		double damp_norm=params.damp*M_PI;
		for (size_t i=0; i<lambdas.size(); i++) {
			for (size_t j=0; j<angles.size(); j++) {
	
				if ((*params.iter_ptr)==-1) {
				    DEBUG("Aborting");
				    break;
				}
				params.iter++;
				(*params.iter_ptr)++;
                DEBUG(11,(100.*params.iter)/(angles.size()*lambdas.size())<<"\% lam "<<lambdas[i]<<", ang "<<angles[j]);
				double cr = cos(angles[j] * _phys_deg); 
				double sr = sin(angles[j] * _phys_deg);

				double thick_norm=params.thickness*M_PI/sqrt(pow(sr*dx,2)+pow(cr*dy,2));
				double lambda_norm=lambdas[i]/sqrt(pow(cr*dx,2)+pow(sr*dy,2));
//				double thick_norm=wave_params.thickness*M_PI/dy;
//				double lambda_norm=lambdas[i]/dx;
                size_t x,y;
#pragma omp parallel for collapse(2)
				for (x=0;x<zz_morlet.getW();x++) {
					for (y=0;y<zz_morlet.getH();y++) {
						double xr = xx[x]*cr - yy[y]*sr; //rotate
						double yr = xx[x]*sr + yy[y]*cr;
			
						double e_x = -pow(damp_norm*(xr*lambda_norm-1.0), 2.);
						double e_y = -pow(yr*thick_norm, 2.);
			
						double gauss = exp(e_x)*exp(e_y);
			
						zz_morlet.Timg_matrix[y][x]=Fmain_window.Timg_matrix[y][x]*gauss;
			
					}
				}
				nPhysImageF<mcomplex> zz_convolve = zz_morlet.ft2(PHYS_BACKWARD);

				// decision
#pragma omp parallel for
				for (size_t k=0; k<params.data->getSurf(); k++) {
					double qmap_local=zz_convolve.Timg_buffer[k].mcabs();
					if ( qmap_local > qmap->Timg_buffer[k]) {
						qmap->Timg_buffer[k] = qmap_local;
						wphase->Timg_buffer[k] = zz_convolve.Timg_buffer[k].arg();
						lambda->Timg_buffer[k] = lambdas[i];
						angle->Timg_buffer[k] = angles[j];
					}
				}
				
			}
			//! todo: this is awful: add exception?
            if ((*params.iter_ptr)==-1) {
                DEBUG("aborting");
                break;
            }
		}
		
		if ((*params.iter_ptr)!=-1) {
		
#pragma omp parallel for
            for (size_t k=0; k<params.data->getSurf(); k++) {
				qmap->Timg_buffer[k]=sqrt(qmap->Timg_buffer[k])/(dx*dy);
				intensity->Timg_buffer[k] = params.data->Timg_buffer[k] - 2.0*qmap->Timg_buffer[k]*cos(wphase->Timg_buffer[k]);
				wphase->Timg_buffer[k]/=2*M_PI;
			}
			
			phys_fast_gaussian_blur(*intensity,params.thickness/2.0);

			params.olist["phase_2pi"] = wphase;
			params.olist["contrast"] = qmap;
			params.olist["lambda"] = lambda;
			params.olist["angle"] = angle;
			params.olist["intensity"] = intensity;
			
			map<string, nPhysD *>::const_iterator itr;
			for(itr = params.olist.begin(); itr != params.olist.end(); ++itr) {
				itr->second->TscanBrightness();
				itr->second->set_origin(params.data->get_origin());
				itr->second->set_scale(params.data->get_scale());
				itr->second->setFromName(params.data->getFromName());
				itr->second->setShortName(itr->first);
				itr->second->setName(itr->first+ " "+params.data->getName());
#pragma omp parallel for
                for (size_t k=0; k<params.data->getSurf(); k++) {
                    if(!std::isfinite(params.data->Timg_buffer[k])){
                        itr->second->Timg_buffer[k]   = numeric_limits<double>::quiet_NaN();
                    }
                }
            }
		}
		
	}
	DEBUG("Out of here");
}

bool cudaEnabled() {
#ifdef HAVE_CUDA
    DEBUG("HAVE_CUDA");
	int device_count = 0;
	cudaGetDeviceCount( &device_count );
	DEBUG("cuda device count "<< device_count);
	if (device_count>0) return true;
#endif
	return false;
}

void phys_wavelet_field_2D_morlet_cuda(wavelet_params &params) {
#ifdef HAVE_CUDA
	if (params.data && (params.n_angles > 0) && (params.n_lambdas > 0) && (params.data->getSurf() != 0)) {

        params.olist.clear();

        params.iter=0;
        *params.iter_ptr = 0;

        DEBUG(5,"start");

        int device_count = 0;
        cudaGetDeviceCount( &device_count );
        if (device_count<1) {
            WARNING("Problem To use CUDA you need an NVIDIA card. You also need to install the driver\nwww.nvidia.com/page/drivers.html");
            return;
        }
    
    
        cudaDeviceProp device_properties;
        int max_gflops_device = 0;
        int max_gflops = 0;
    
        int current_device = 0;
        cudaGetDeviceProperties( &device_properties, current_device );
        max_gflops = device_properties.multiProcessorCount * device_properties.clockRate;
        ++current_device;
    
        while( current_device < device_count )
        {
            cudaGetDeviceProperties( &device_properties, current_device );
            int gflops = device_properties.multiProcessorCount * device_properties.clockRate;
            if( gflops > max_gflops )
            {
                max_gflops        = gflops;
                max_gflops_device = current_device;
            
                if (device_properties.major>=1 && device_properties.minor>=3) {
                    DEBUG("This graphic card could do double precision... please contact developer");
                }

            } 
            ++current_device;
        }	
        cudaSetDevice( max_gflops_device );

        int cubuf_size = sizeof(cufftComplex)*params.data->getSurf();
    
        cufftHandle plan;
        cufftComplex *cub1, *cub2, *cub3, *cuc1, *cuc2;
    
        vector<cufftComplex> b1(params.data->getSurf());
        vector<cufftComplex> b2(params.data->getSurf());
    
        for (size_t j = 0; j < params.data->getH(); j++){
            for (size_t i = 0; i < params.data->getW(); i++) {
                b1[i+j*params.data->getW()].x = std::isfinite(params.data->Timg_matrix[j][i]) ? params.data->Timg_matrix[j][i]:0;
                b1[i+j*params.data->getW()].y=0.0;
            }
        }
    
        cudaMalloc((void**)&cub1, cubuf_size);
        if (cudaGetLastError()!=cudaSuccess) {
            WARNING("cannot allocate");
            return;
        }
        cudaMalloc((void**)&cub2, cubuf_size);
        if (cudaGetLastError()!=cudaSuccess) {
            WARNING("cannot allocate");
            return;
        }
        cudaMalloc((void**)&cub3, cubuf_size);
        if (cudaGetLastError()!=cudaSuccess) {
            WARNING("cannot allocate");
            return;
        }
        cudaMalloc((void**)&cuc1, cubuf_size);
        if (cudaGetLastError()!=cudaSuccess) {
            WARNING("cannot allocate");
            return;
        }
        cudaMalloc((void**)&cuc2, cubuf_size);
        if (cudaGetLastError()!=cudaSuccess) {
            WARNING("cannot allocate");
            return;
        }
    
        cudaMemset(cub1, 0, cubuf_size );
        cudaMemset(cub2, 0, cubuf_size );
        cudaMemset(cub3, 0, cubuf_size );
        cudaMemset(cuc1, 0, cubuf_size );
        cudaMemset(cuc2, 0, cubuf_size );
    
        // Create a 2D FFT plan.  
        cufftPlan2d(&plan, params.data->getH(), params.data->getW(), CUFFT_C2C);
        if (cudaGetLastError()!=cudaSuccess) {
            WARNING("cannot create plan");
            return;
        }

        cudaMemcpy(cub1, &b1[0], cubuf_size, cudaMemcpyHostToDevice);

        cufftExecC2C(plan, cub1, cub2, CUFFT_FORWARD);
        if (cudaGetLastError()!=cudaSuccess) {
            WARNING("cannot do FFT");
            return;
        }
    //	cudaThreadSynchronize();
    
        if ((params.n_angles > 0) && (params.n_lambdas > 0) && (params.data->getSurf() != 0)) {

            int dx=params.data->getW();
            int dy=params.data->getH();

            vector<double> angles(params.n_angles), lambdas(params.n_lambdas);

            nPhysD *qmap, *wphase, *lambda, *angle, *intensity;
            qmap = new nPhysD(dx, dy, 0.0, "quality");
        
            wphase = new nPhysD("phase/2pi");
            wphase->resize(dx, dy);
            lambda = new nPhysD("lambda");
            lambda->resize(dx, dy);
            angle = new nPhysD("angle");
            angle->resize(dx, dy);
            intensity = new nPhysD("intensity");
            intensity->resize(dx, dy);


            if (params.n_angles==1) {
                angles[0]=0.5*(params.end_angle+params.init_angle);
            } else {
                for (size_t i=0; i<params.n_angles; i++)
                    angles[i] = params.init_angle + i*(params.end_angle-params.init_angle)/(params.n_angles-1);
            }	
            if (params.n_lambdas==1) {
                lambdas[0]=0.5*(params.end_lambda+params.init_lambda);
            } else {
                for (size_t i=0; i<params.n_lambdas; i++)
                    lambdas[i] = params.init_lambda + i*(params.end_lambda-params.init_lambda)/(params.n_lambdas-1);
            }
        
            for (size_t i=0; i<lambdas.size(); i++) {
                for (size_t j=0; j<angles.size(); j++) {
    
                    if ((*params.iter_ptr)==-1) {
                        DEBUG("aborting");
                        break;
                    }
                    params.iter++;
                    (*params.iter_ptr)++;
                    DEBUG((100.*params.iter)/(angles.size()*lambdas.size())<<"\% lam "<<lambdas[i]<<", ang "<<angles[j]);
            
                    gabor(cub2, cub1, dx, dy, angles[j]/180.*M_PI, lambdas[i], (float)params.damp, (float)params.thickness);
                    cufftExecC2C(plan, cub1, cub3, CUFFT_INVERSE);
                    fase(cub3, cuc1, cuc2, params.data->getSurf(), angles[j], lambdas[i]);
                    cudaThreadSynchronize();
            
                }
                //! todo: this is awful: add exception?
                if ((*params.iter_ptr)==-1) {
                    DEBUG("aborting");
                    break;
                }
            }
            DEBUG(PRINTVAR(*params.iter_ptr));
            cudaThreadSynchronize();

            if ((*params.iter_ptr)!=-1) {
                cudaMemcpy(&b1[0], cuc1, cubuf_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(&b2[0], cuc2, cubuf_size, cudaMemcpyDeviceToHost);
                cudaThreadSynchronize();
                        
                phys_fast_gaussian_blur(*intensity,params.thickness/2.0);
                        
                wphase->property["unitsCB"]="2pi";
                params.olist["phase_2pi"] = wphase;
                params.olist["contrast"] = qmap;
                lambda->property["unitsCB"]="px";
                params.olist["lambda"] = lambda;
                angle->property["unitsCB"]="deg";
                params.olist["angle"] = angle;
                params.olist["intensity"] = intensity;
            
                map<string, nPhysD *>::const_iterator itr;
                for(itr = params.olist.begin(); itr != params.olist.end(); ++itr) {
                    itr->second->TscanBrightness();
                    itr->second->set_origin(params.data->get_origin());
                    itr->second->set_scale(params.data->get_scale());
                    itr->second->setFromName(params.data->getFromName());
                    itr->second->setShortName(itr->first);
                    itr->second->setName(itr->first+ " "+params.data->getName());
                    for (size_t k=0; k<params.data->getSurf(); k++) {
                        if(!std::isfinite(params.data->Timg_buffer[k])){
                            itr->second->Timg_buffer[k]   = numeric_limits<double>::quiet_NaN();
                        }
                    }
                }
            }
            
            cudaFree(cub1);
            cudaFree(cub2);
            cudaFree(cub3);
            cudaFree(cuc1);
            cudaFree(cuc2);
    
            cufftDestroy (plan);
            cudaThreadSynchronize();
            cudaThreadExit();

        }
    }
#endif
    DEBUG("Out of here");
}

unsigned int opencl_closest_size(unsigned int num) {
    unsigned int closest=2*num;
    // REMEMBER to use clfft 2.12 to support radix 11 and 13!
    for (unsigned int i13=0; i13<=log(num)/log(13); i13++ ) {
        for (unsigned int i11=0; i11<=log(num)/log(11); i11++ ) {
            for (unsigned int i7=0; i7<=log(num)/log(7); i7++ ) {
                for (unsigned int i5=0; i5<=log(num)/log(5); i5++ ) {
                    for (unsigned int i3=0; i3<=log(num)/log(3); i3++ ) {
                        for (unsigned int i2=0; i2<=log(num)/log(2); i2++ ) {
                            unsigned int test_val=pow(2,i2)*pow(3,i3)*pow(5,i5)*pow(7,i7)*pow(11,i11)*pow(13,i13);
                            if (test_val>=num && test_val<closest) {
                                closest=test_val;
                                if (closest==num) return num;
                            }
                        }
                    }
                }
            }
        }
    }
    return closest;
}

vec2 opencl_closest_size(vec2 num){
    return vec2(opencl_closest_size(num.x()),opencl_closest_size(num.y()));
}

int openclEnabled() {
    int found_GPU=0;
#ifdef HAVE_LIBCLFFT
    // get all platforms
    cl_uint platformCount;
    clGetPlatformIDs(0, NULL, &platformCount);
    vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, &platforms[0], NULL);
    for (unsigned int i = 0; i < platformCount; i++) {
        cl_uint deviceCount;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        if (deviceCount>0) {
            found_GPU++;
        }
    }
#endif
    return found_GPU;
}

#ifdef HAVE_LIBCLFFT
pair<cl_platform_id,cl_device_id> get_platform_device_opencl(int num) {
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    int found_GPU=0;
    // get all platforms
    cl_uint platformCount;
    clGetPlatformIDs(0, NULL, &platformCount);
    vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, &platforms[0], NULL);

    for (unsigned int i = 0; i < platformCount; i++) {

        // get all devices
        cl_uint deviceCount;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, &devices[0], NULL);

        DEBUG(deviceCount);
        // for each device print critical attributes
        for (int j = 0; j < (int) deviceCount; j++) {
            found_GPU++;
            if (found_GPU==num) {
                platform=platforms[i];
                device=devices[j];
            }
        }
    }
    DEBUG("HERE");
    return make_pair(platform,device);
}
#endif

void phys_wavelet_field_2D_morlet_opencl(wavelet_params &params) {
#ifdef HAVE_LIBCLFFT

    if (params.opencl_unit>0) {

        params.iter=0;
        *params.iter_ptr=0;

        vec2 newSize(opencl_closest_size(params.data->getSize()));

        double mean=params.data->sum()/params.data->getSurf();

        nPhysD padded(newSize.x(), newSize.y(), mean);
        bidimvec<int> offset=(newSize-params.data->get_size())/2;
        DEBUG("padding offset : " << offset);
        padded.set_origin(params.data->get_origin()+offset);

#pragma omp parallel for collapse(2)
        for (size_t j=0; j<params.data->getH(); j++) {
            for (size_t i=0; i<params.data->getW(); i++) {
                padded.set(i+offset.x(),j+offset.y(),params.data->getPoint(i,j));
            }
        }

        unsigned int dx = padded.getW();
        unsigned int dy = padded.getH();

        cl_int err;

        pair<cl_platform_id,cl_device_id> my_pair = get_platform_device_opencl(params.opencl_unit);
        cl_platform_id platform = my_pair.first;
        cl_device_id device=my_pair.second;

        /* Setup OpenCL environment. */

        cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
        cl_context ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
        check_opencl_error(err, "clCreateContext");
        cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
        check_opencl_error(err, "clCreateCommandQueue");


        // KERNEL
        string textkernel=
                "__kernel void gabor(__global float *inReal, __global float *inImag, __global float *outReal, __global float *outImag, const unsigned int dx, const unsigned int dy,  const float sr,  const float cr, const float lambda_norm, const float damp_norm,  const float thick_norm){\n"
                "    size_t id = get_global_id(0);\n"
                "    int i = id%dx;\n"
                "    int j = id/dx;\n"
                "    if (i>=(int)dx/2) i-=dx;\n"
                "    if (j>=(int)dy/2) j-=dy;\n"
                "    float xr=i*cr-j*sr;\n"
                "    float yr=i*sr+j*cr;\n"
                "    float gauss=native_exp(-pown(damp_norm*(xr*lambda_norm-1.0f),2))*native_exp(-pown(yr*thick_norm,2));\n"
                "    outReal[id] = gauss * inReal[id];\n"
                "    outImag[id] = gauss * inImag[id];\n"
                "}\n"
                "__kernel void best(__global float *inReal, __global float *inImag, __global float *outQual, __global float *outPhase, __global unsigned int *outLambdaAngle, const unsigned int nlambdaangle){\n"
                "    size_t id = get_global_id(0);\n"
                "    float quality=pown(inReal[id],2)+pown(inImag[id],2);\n"
                "    if (quality>outQual[id]) {\n"
                "        outQual[id]=quality;\n"
                "        outPhase[id]=atan2pi(inImag[id],inReal[id]);\n"
                "        outLambdaAngle[id]=nlambdaangle;\n"
                "    }\n"
                "}\n";

        DEBUG("KERNEL:\n" << textkernel << "\nEND KERNEL") ;

        const char *source=textkernel.c_str();

        cl_program program = clCreateProgramWithSource(ctx,1,&source, NULL, &err);
        check_opencl_error(err, "clCreateProgramWithSource");
        err=clBuildProgram(program, 1, &device, "-Werror -cl-fast-relaxed-math", NULL, NULL);
        check_opencl_error(err, "clBuildProgram");

        if (err == CL_BUILD_PROGRAM_FAILURE) {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            // Allocate memory for the log
            std::string mylog(log_size, ' ');

            // Get the log
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &mylog[0], NULL);

            // Print the log
            WARNING(mylog);
        }

        /* Setup clFFT. */
        clfftSetupData fftSetup;
        err = clfftInitSetupData(&fftSetup);
        check_opencl_error(err, "clfftInitSetupData");
        err = clfftSetup(&fftSetup);
        check_opencl_error(err, "clfftSetup");

        /* Create a default plan for a complex FFT. */
        clfftPlanHandle planHandle;
        vector<size_t> clLengths =  {(size_t)dx, (size_t)dy};
        err = clfftCreateDefaultPlan(&planHandle, ctx, CLFFT_2D, &clLengths[0]);
        check_opencl_error(err, "clfftCreateDefaultPlan");

        /* Set plan parameters. */
        err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
        check_opencl_error(err, "clfftSetPlanPrecision");
        err = clfftSetLayout(planHandle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
        check_opencl_error(err, "clfftSetLayout");
        err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);
        check_opencl_error(err, "clfftSetResultLocation");

        /* Bake the plan. */
        err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
        check_opencl_error(err, "clfftBakePlan");

        /* Real and Imaginary arrays. */
        cl_uint N = dx*dy;
        vector<cl_float> inReal(N,0);
        vector<cl_float> inImag(N,0);

        /* Initialization of inReal*/
        for(cl_uint j=0; j<dy; j++) {
          for(cl_uint i=0; i<dx; i++) {
            inReal[j*dx+i] = padded.point(i,j);
          }
        }

        /* Size of temp buffer. */
        size_t tmpBufferSize = 0;
        err = clfftGetTmpBufSize(planHandle, &tmpBufferSize);
        check_opencl_error(err, "clfftGetTmpBufSize");

        /* Temporary buffer. */
        cl_mem tmpBuffer = 0;
        if ((err == 0) && (tmpBufferSize > 0)) {
          tmpBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, tmpBufferSize, 0, &err);
          check_opencl_error(err, "tmpBuffer clCreateBuffer " << tmpBufferSize);
          DEBUG("intermediate buffer needed");
        }

        cl_mem buffersIn[2]  = {0, 0};
        /* Prepare OpenCL memory objects : create buffer for input. */
        buffersIn[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), &inReal[0], &err);
        check_opencl_error(err, "inReal buffersIn[0] clCreateBuffer");

        /* Enqueue write inReal array into buffersIn[0]. */
        err = clEnqueueWriteBuffer(queue, buffersIn[0], CL_TRUE, 0, N * sizeof(float), &inReal[0], 0, NULL, NULL);
        check_opencl_error(err, "inReal buffersIn[0] clEnqueueWriteBuffer");

        /* Prepare OpenCL memory objects : create buffer for input. */
        buffersIn[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), &inImag[0], &err);
        check_opencl_error(err, "inImag buffersIn[1] clCreateBuffer");

        /* Enqueue write inImag array into buffersIn[1]. */
        err = clEnqueueWriteBuffer(queue, buffersIn[1], CL_TRUE, 0, N * sizeof(float), &inImag[0], 0, NULL, NULL);
        check_opencl_error(err, "inImag buffersIn[1] clEnqueueWriteBuffer");

        cl_mem buffersOut[2] = {0, 0};
        /* Prepare OpenCL memory objects : create buffer for output. */
        buffersOut[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(cl_float), NULL, &err);
        check_opencl_error(err, "buffersOut[0] clCreateBuffer");

        buffersOut[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(cl_float), NULL, &err);
        check_opencl_error(err, "buffersOut[1] clCreateBuffer");

        cl_mem best[3] = {0, 0, 0};
//        best[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(cl_float),  NULL, &err);
//        check_opencl_error(err, "best[0] clCreateBuffer");
//        best[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(cl_float),  NULL, &err);
//        check_opencl_error(err, "best[1] clCreateBuffer");
//        best[2] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(unsigned char),  NULL, &err);
//        check_opencl_error(err, "best[1] clCreateBuffer");
//        best[3] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(unsigned char),  NULL, &err);
//        check_opencl_error(err, "best[1] clCreateBuffer");


        /* Prepare OpenCL memory objects : create buffer for quality. */
        best[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), &inImag[0], &err);
        check_opencl_error(err, "inReal best[0] clCreateBuffer");
        err = clEnqueueWriteBuffer(queue, best[0], CL_TRUE, 0, N * sizeof(float), &inImag[0], 0, NULL, NULL);
        check_opencl_error(err, "inReal best[0] clEnqueueWriteBuffer");

        best[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(cl_float),  NULL, &err);
        check_opencl_error(err, "best[1] clCreateBuffer");
        best[2] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(cl_uint),  NULL, &err);
        check_opencl_error(err, "best[2] clCreateBuffer");

        /* Execute Forward FFT. */
        err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, buffersIn, buffersOut, tmpBuffer);
        check_opencl_error(err, "clfftEnqueueTransform");

        /* Wait for calculations to be finished. */
        err = clFinish(queue);
        check_opencl_error(err, "clFinish");

        err=clReleaseMemObject(tmpBuffer);
        check_opencl_error(err, "clReleaseMemObject tmpBuffer");

        /* Release the plan. */
        err = clfftDestroyPlan(&planHandle );
        check_opencl_error(err, "clfftDestroyPlan");


//            make new plan
        err = clfftCreateDefaultPlan(&planHandle, ctx, CLFFT_2D, &clLengths[0]);
        check_opencl_error(err, "clfftCreateDefaultPlan");

        /* Set plan parameters. */
        err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
        check_opencl_error(err, "clfftSetPlanPrecision");
        err = clfftSetLayout(planHandle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
        check_opencl_error(err, "clfftSetLayout");
        err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
        check_opencl_error(err, "clfftSetResultLocation");

        /* Bake the plan. */
        err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
        check_opencl_error(err, "clfftBakePlan");

        /* Size of temp buffer. */
        err = clfftGetTmpBufSize(planHandle, &tmpBufferSize);
        check_opencl_error(err, "clfftGetTmpBufSize");

        /* Temporary buffer. */
        if ((err == 0) && (tmpBufferSize > 0)) {
          tmpBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, tmpBufferSize, 0, &err);
          check_opencl_error(err, "tmpBuffer clCreateBuffer " << tmpBufferSize);
          DEBUG("intermediate buffer needed");
        }


        cl_kernel kernelGabor = clCreateKernel(program, "gabor", &err);
        check_opencl_error(err, "clCreateKernel");

        clSetKernelArg(kernelGabor, 0, sizeof(cl_mem), &buffersOut[0]); //out because fft: in->out and gabor: out->in
        clSetKernelArg(kernelGabor, 1, sizeof(cl_mem), &buffersOut[1]);
        clSetKernelArg(kernelGabor, 2, sizeof(cl_mem), &buffersIn[0]);
        clSetKernelArg(kernelGabor, 3, sizeof(cl_mem), &buffersIn[1]);
        clSetKernelArg(kernelGabor, 4, sizeof(unsigned int), &dx);
        clSetKernelArg(kernelGabor, 5, sizeof(unsigned int), &dy);
        float damp_norm= params.damp * M_PI;
        clSetKernelArg(kernelGabor, 9, sizeof(float), &damp_norm);

        cl_kernel kernelBest = clCreateKernel(program, "best", &err);
        check_opencl_error(err, "clCreateKernel best");

        clSetKernelArg(kernelBest, 0, sizeof(cl_mem), &buffersIn[0]); //out because fft: in->out and gabor: out->in
        clSetKernelArg(kernelBest, 1, sizeof(cl_mem), &buffersIn[1]);
        clSetKernelArg(kernelBest, 2, sizeof(cl_mem), &best[0]);
        clSetKernelArg(kernelBest, 3, sizeof(cl_mem), &best[1]);
        clSetKernelArg(kernelBest, 4, sizeof(cl_mem), &best[2]);



        vector<double> angles(params.n_angles), lambdas(params.n_lambdas);
        if (params.n_angles==1) {
            angles[0]=0.5*(params.end_angle+params.init_angle);
        } else {
            for (size_t nangle=0; nangle<params.n_angles; nangle++)
                angles[nangle] = params.init_angle + nangle*(params.end_angle-params.init_angle)/(params.n_angles-1);
        }
        if (params.n_lambdas==1) {
            lambdas[0]=0.5*(params.end_lambda+params.init_lambda);
        } else {
            for (size_t nlambda=0; nlambda<params.n_lambdas; nlambda++)
                lambdas[nlambda] = params.init_lambda + nlambda*(params.end_lambda-params.init_lambda)/(params.n_lambdas-1);
        }


        size_t totalJobs=N;

        for (unsigned char nangle=0; nangle <params.n_angles; nangle++) {

            for (unsigned char nlambda=0; nlambda <params.n_lambdas; nlambda++) {

                if ((*params.iter_ptr)==-1) {
                    DEBUG("Aborting");
                    break;
                }

                DEBUG("Angle: " << (int)nlambda << " " << angles[nangle] << " Lambda: " << (int)nangle << " " << lambdas[nlambda] );


                float sr=sin(angles[nangle]*_phys_deg);
                float cr=cos(angles[nangle]*_phys_deg);
                err=clSetKernelArg(kernelGabor, 6, sizeof(float), &sr);
                check_opencl_error(err, "clSetKernelArg");
                err=clSetKernelArg(kernelGabor, 7, sizeof(float), &cr);
                check_opencl_error(err, "clSetKernelArg");
                float lambda_norm=lambdas[nlambda]/sqrt(pow(cr*dx,2)+pow(sr*dy,2));
                err=clSetKernelArg(kernelGabor, 8, sizeof(float), &lambda_norm);
                check_opencl_error(err, "clSetKernelArg");

                float thick_norm=params.thickness * M_PI/sqrt(pow(sr*dx,2)+pow(cr*dy,2));
                err=clSetKernelArg(kernelGabor,10, sizeof(float), &thick_norm);
                check_opencl_error(err, "clSetKernelArg");


                clEnqueueNDRangeKernel(queue, kernelGabor, 1, NULL, &totalJobs, NULL, 0, NULL, NULL);
                err = clFinish(queue);
                check_opencl_error(err, "clFinish");


                /* Execute Backward FFT. */
                err = clfftEnqueueTransform(planHandle, CLFFT_BACKWARD, 1, &queue, 0, NULL, NULL, buffersIn, NULL, tmpBuffer);
                check_opencl_error(err, "clfftEnqueueTransform ");

                /* Wait for calculations to be finished. */
                err = clFinish(queue);
                check_opencl_error(err, "clFinish");

                unsigned int iter=params.iter;
                err=clSetKernelArg(kernelBest, 5, sizeof(unsigned int), &iter);
                check_opencl_error(err, "clSetKernelArg");

                clEnqueueNDRangeKernel(queue, kernelBest, 1, NULL, &totalJobs, NULL, 0, NULL, NULL);
                err = clFinish(queue);
                check_opencl_error(err, "clFinish");
                params.iter++;
                (*params.iter_ptr)++;

            }
        }

        vector<float> quality_sqr(N,0);
        err = clEnqueueReadBuffer(queue, best[0], CL_TRUE, 0, N * sizeof(float), &quality_sqr[0], 0, NULL, NULL);
        check_opencl_error(err, "clEnqueueReadBuffer");

        vector<float> phase(N,0);
        err = clEnqueueReadBuffer(queue, best[1], CL_TRUE, 0, N * sizeof(float), &phase[0], 0, NULL, NULL);
        check_opencl_error(err, "clEnqueueReadBuffer");

        vector<unsigned int> lambdaangle(N,0);
        err = clEnqueueReadBuffer(queue, best[2], CL_TRUE, 0, N * sizeof(unsigned int), &lambdaangle[0], 0, NULL, NULL);
        check_opencl_error(err, "clEnqueueReadBuffer");

        err = clFinish(queue);
        check_opencl_error(err, "clFinish");

        /* Release OpenCL memory objects. */
        clReleaseMemObject(buffersIn[0]);
        clReleaseMemObject(buffersIn[1]);
        clReleaseMemObject(buffersOut[0]);
        clReleaseMemObject(buffersOut[1]);
        clReleaseMemObject(best[0]);
        clReleaseMemObject(best[1]);
        clReleaseMemObject(best[2]);
        clReleaseMemObject(tmpBuffer);

        /* Release the plan. */
        err = clfftDestroyPlan(&planHandle );
        check_opencl_error(err, "clfftDestroyPlan");

        /* Release clFFT library. */
        clfftTeardown( );

        /* Release OpenCL working objects. */
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);

        params.olist.clear();

        nPhysD *nQuality = new nPhysD(params.data->getW(),params.data->getH(),0,"Quality");
        nPhysD *nPhase = new nPhysD(params.data->getW(),params.data->getH(),0,"Phase");
        nPhysD *nIntensity = new nPhysD(params.data->getW(),params.data->getH(),0,"Intensity");

        for (size_t j=0; j<params.data->getH(); j++) {
            for (size_t i=0; i<params.data->getW(); i++) {
                unsigned int k=(j+offset.y())*dx+i+offset.x();
                nQuality->set(i,j,sqrt(quality_sqr[k]));
                nPhase->set(i,j,phase[k]/2.0);
                nIntensity->set(i,j,params.data->point(i,j) - 2.0*nQuality->point(i,j)*cos(M_PI*phase[k]));
            }
        }

        phys_fast_gaussian_blur(*nIntensity,params.thickness/2.0);

        params.olist["phase_2pi"] = nPhase;
        params.olist["contrast"] = nQuality;
        params.olist["intensity"] = nIntensity;


        if (params.n_angles>1) {
            nPhysD *nAngle = new nPhysD(params.data->getW(),params.data->getH(),0,"Angle");
            for (size_t j=0; j<params.data->getH(); j++) {
                for (size_t i=0; i<params.data->getW(); i++) {
                    unsigned int k=(j+offset.y())*dx+i+offset.x();
                    nAngle->set(i,j,angles[lambdaangle[k]%params.n_angles]);
                }
            }
            params.olist["angle"] = nAngle;
        }

        if (params.n_lambdas>1) {
            nPhysD *nLambda = new nPhysD(params.data->getW(),params.data->getH(),0,"Lambda");
            for (size_t j=0; j<params.data->getH(); j++) {
                for (size_t i=0; i<params.data->getW(); i++) {
                    unsigned int k=(j+offset.y())*dx+i+offset.x();
                    nLambda->set(i,j,lambdas[lambdaangle[k]/params.n_angles]);
                }
            }
            params.olist["lambda"] = nLambda;
        }


        map<string, nPhysD *>::const_iterator itr;
        for(itr = params.olist.begin(); itr != params.olist.end(); ++itr) {
            itr->second->TscanBrightness();
            itr->second->set_origin(params.data->get_origin());
            itr->second->set_scale(params.data->get_scale());
            itr->second->setFromName(params.data->getFromName());
            itr->second->setShortName(itr->first);
            itr->second->setName(itr->first+ " "+params.data->getName());
#pragma omp parallel for
            for (size_t k=0; k<params.data->getSurf(); k++) {
                if (isnan(params.data->point(k))) {
                    itr->second->set(k,numeric_limits<double>::quiet_NaN());
                }
            }
        }

    }
#endif
}



// ---------------------- thread transport functions ------------------------

void phys_wavelet_trasl_cuda(void *params, int &iter) {
    DEBUG("Enter here");
	((wavelet_params *)params)->iter_ptr = &iter;
	phys_wavelet_field_2D_morlet_cuda(*((wavelet_params *)params));
}

void phys_wavelet_trasl_opencl(void *params, int &iter) {
    DEBUG("Enter here");
    ((wavelet_params *)params)->iter_ptr = &iter;
    phys_wavelet_field_2D_morlet_opencl(*((wavelet_params *)params));
}

void phys_wavelet_trasl_nocuda(void *params, int &iter) {
    DEBUG("Enter here");
	((wavelet_params *)params)->iter_ptr = &iter;
	phys_wavelet_field_2D_morlet(*((wavelet_params *)params));
}


// unwrap methods
nPhysD *
phys_phase_unwrap(nPhysD &wphase, nPhysD &quality, enum unwrap_strategy strategy)
{

	if (wphase.getSurf() == 0)
		return NULL;

	nPhysD *uphase;
	uphase = new nPhysD (wphase.getW(), wphase.getH(), 0., "unwrap");
	uphase->set_origin(wphase.get_origin());
	uphase->set_scale(wphase.get_scale());
	uphase->setName("Unwrap "+wphase.getName());

	switch (strategy) {
		case SIMPLE_HV :
			unwrap_simple_h(&wphase, uphase);
 			unwrap_simple_v(&wphase, uphase);
			break;

		case SIMPLE_VH :
 			unwrap_simple_v(&wphase, uphase);
			unwrap_simple_h(&wphase, uphase);
			break;

		case GOLDSTEIN :
			unwrap_goldstein(&wphase, uphase);
			break;

		case QUALITY :
			unwrap_quality(&wphase, uphase, &quality);
 			break;
 			
		case MIGUEL :
			unwrap_miguel(&wphase, uphase);
 			break;

		case MIGUEL_QUALITY :
			unwrap_miguel_quality(&wphase, uphase, &quality);
 			break;
			
	}
    DEBUG("here");
    uphase->TscanBrightness();
	return uphase;
}

void phys_synthetic_interferogram (nPhysImageF<double> &synthetic, nPhysImageF<double> *phase_over_2pi, nPhysImageF<double> *quality){
    if (phase_over_2pi && quality) {
        if (phase_over_2pi->getW()==quality->getW() && phase_over_2pi->getH()==quality->getH()) {
            synthetic.resize(phase_over_2pi->getW(),phase_over_2pi->getH());
#pragma omp parallel for
            for (size_t ii=0; ii<phase_over_2pi->getSurf(); ii++) {
                synthetic.set(ii,M_PI*quality->point(ii)*(1.0+cos(phase_over_2pi->point(ii)*2*M_PI)));
            }
            synthetic.property=phase_over_2pi->property;
            synthetic.setShortName("synthetic");
            synthetic.setName("synthetic("+phase_over_2pi->getName()+","+quality->getName()+")");
            synthetic.TscanBrightness();
        }
    }
}

void
phys_subtract_carrier(nPhysD &iphys, double kx, double ky)
{

#pragma omp parallel for collapse(2)
    for (size_t ii=0; ii<iphys.getW(); ii++) {
        for (size_t jj=0; jj<iphys.getH(); jj++) {
			iphys.Timg_matrix[jj][ii] -= ii*kx + jj*ky;
		}
	}
	iphys.TscanBrightness();
}

//! this function returns the carrier bidimvec<double(angle[deg],interfringe[px])>
bidimvec<double>
phys_guess_carrier(nPhysD &phys, double weight)
{
	size_t dx=phys.getW();
	size_t dy=phys.getH();
	
    fftw_complex *myData=fftw_alloc_complex(dy*(dx/2+1));
    fftw_plan plan=fftw_plan_dft_r2c_2d(dy,dx, phys.Timg_buffer, myData, FFTW_ESTIMATE);
    fftw_execute(plan);

    double valmax=0.0;
    int imax=0,jmax=0;
    for (size_t i=0; i<dx/2+1; i++) {
        for (size_t j=0; j<dy; j++) {
            int j1=(dy/2+1)-(j+dy/2+1)%dy;
            double r=sqrt(i*i+j1*j1);
            size_t k=i+j*(dx/2+1);
            double val=pow(r,weight)*vec2f(myData[k][0],myData[k][1]).mod();
            if (val>valmax && (i>0||j>0)) {
                valmax=val;
                imax=i;
                jmax=j1;
            }
        }
    }

	fftw_free(myData);
	fftw_destroy_plan(plan);

	bidimvec<double> retCarrier;

	if (imax!=0 || jmax!=0) {
		mcomplex freq(imax/((double)dx),jmax/((double)dy));
		retCarrier = bidimvec<double>(1.0/freq.mod(),fmod(freq.arg()*180.0/M_PI+360.0,180.0));
	} else {
		retCarrier = bidimvec<double>(0.0,0.0);
	}
	
	return retCarrier;
}

// --------------------------------------------------------------------- integral inversions --

void
phys_apply_inversion_gas(nPhysD &invimage, double probe_wl, double res, double molar_refr)
{
	// TODO: accendere candelina al dio dei define che rendono il codice leggibile come 'sta ceppa di cazzo
	//double kappa = M_2_PI/probe_wl;
	double kappa = 2*M_PI/probe_wl;
	//double mult = _phys_avogadro / (3*kappa*kappa*molar_refr);
	double mult = _phys_avogadro / (3*molar_refr);
    for (size_t ii=0; ii<invimage.getSurf(); ii++) {
		double the_point = invimage.point(ii)/res;
		//invimage.set(ii, - mult * (the_point*the_point+2*kappa*the_point));
		invimage.set(ii, mult*(pow(the_point/kappa + 1,2.)-1) );
	}
    invimage.property["unitsCB"]="m-3";
	invimage.TscanBrightness();
}

void
phys_apply_inversion_plasma(nPhysD &invimage, double probe_wl, double res)
{
	double kappa = 2*M_PI/probe_wl;
	double mult = _phys_emass*_phys_vacuum_eps*_phys_cspeed*_phys_cspeed/(_phys_echarge*_phys_echarge);
	DEBUG(5,"resolution: "<< res << ", probe: " << probe_wl << ", mult: " << mult);
    for (size_t ii=0; ii<invimage.getSurf(); ii++) {
		double the_point = invimage.point(ii)/res;
		invimage.set(ii, - mult * (the_point*the_point+2*kappa*the_point));
	}
    invimage.property["unitsCB"]="m-3";
	invimage.TscanBrightness();
}

void
phys_apply_inversion_protons(nPhysD &invimage, double energy, double res, double distance, double magnification)
{
	double mult = (2.0*_phys_vacuum_eps*magnification*energy)/(distance*res);
    phys_multiply(invimage,mult);
	invimage.set_scale(res*1e2,res*1e2);
    invimage.property["unitsX"]="cm";
    invimage.property["unitsY"]="cm";
    invimage.property["unitsCB"]="C/m-3";
}


//! General function for Abel inversion
void phys_invert_abel(abel_params &params)
{
	
	std::vector<phys_point> iaxis = params.iaxis; // TODO: passa a bidimvec
	phys_direction idir = params.idir;
	inversion_algo ialgo = params.ialgo;
//	inversion_physics iphysics = params.iphysics;


	if (params.iimage->getSurf() == 0)
		return;

	params.oimage = new nPhysD (params.iimage->getW(), params.iimage->getH(),numeric_limits<double>::quiet_NaN(),"Inverted");
	params.oimage->set_origin(params.iimage->get_origin());
	params.oimage->set_scale(params.iimage->get_scale());

    
	// adorabile ridondanza qui..
	
	// 1. set direction indexes
	enum phys_direction sym_idx, inv_idx;
	switch (idir) {
		case PHYS_Y:
			sym_idx = PHYS_Y; // y
			inv_idx = PHYS_X; // x
			break;

		case PHYS_X:
		default:
			sym_idx = PHYS_X; // x
			inv_idx = PHYS_Y; // y
			break;
	}

	DEBUG(5,"after direction allocation");
	
	// 2. switch on algo
	size_t axe_point[2];
// 	size_t longitudinal_size = params.iimage->getSizeByIndex(sym_idx);
	size_t integral_size = params.iimage->getSizeByIndex(inv_idx);
	double *copy_buffer, *out_buffer;
	copy_buffer = new double[integral_size];
	out_buffer = new double[integral_size];
	
//     params.rimage.resize(params.iimage->getW(), params.iimage->getH());
// 	vector<double> rbuffer_pos(integral_size);
// 	for (int j=0; j<integral_size; j++) {
// 	    rbuffer_pos[j]=j;
//     }
    
	// .alex. old version
	/*switch (idir) {
		case PHYS_Y:
			params.oimage->set(iaxis[ii].x,iaxis[ii].y, 
					0.5*(params.oimage->point(iaxis[ii].x-1,iaxis[ii].y)+params.oimage->point(iaxis[ii].x+1,iaxis[ii].y)));
			break;
		
		case PHYS_X:
		default:
			params.oimage->set(iaxis[ii].x,iaxis[ii].y, 
					0.5*(params.oimage->point(iaxis[ii].x,iaxis[ii].y-1)+params.oimage->point(iaxis[ii].x,iaxis[ii].y+1)));
			break;
	}*/

	*params.iter_ptr = 0;
	
	if (ialgo == ABEL) {
		DEBUG(1, "Plain ABEL inversion");
        for (size_t ii = 0; ii<iaxis.size(); ii++) {
			if ((*params.iter_ptr)==-1) {
				    DEBUG("aborting");
				    break;
			}
			(*params.iter_ptr)++;
			
			axe_point[0] = iaxis[ii].x;
			axe_point[1] = iaxis[ii].y;
			//cerr << axe_point[0]  << " , " << axe_point[1] << endl;
			int copied = params.iimage->get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], copy_buffer, integral_size, PHYS_NEG);

// 			params.rimage.set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &rbuffer_pos[0], integral_size, PHYS_NEG);
            
			for (size_t j=copied; j<integral_size; j++)
				copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization
	
			phys_invert_abel_1D(copy_buffer, out_buffer, integral_size);
			
			params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], out_buffer, integral_size, PHYS_NEG);
			

			copied = params.iimage->get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], copy_buffer, integral_size, PHYS_POS);

// 			params.rimage.set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &rbuffer_pos[0], integral_size, PHYS_POS);

			for (size_t j=copied; j<integral_size; j++)
				copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization

			phys_invert_abel_1D(copy_buffer, out_buffer, integral_size);
			params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], out_buffer, integral_size, PHYS_POS);

			//FIXME: pretty sure there is a better way!! ALEX!!!!
			// .alex. : fixed in some way. To be checked
			params.oimage->set(iaxis[ii].x, iaxis[ii].y, 
					0.5*(params.oimage->point(iaxis[ii].x-(idir),iaxis[ii].y+(idir-1))+params.oimage->point(iaxis[ii].x+idir,iaxis[ii].y+(1-idir))));

		}

		params.oimage->setName(string("ABEL ")+params.oimage->getName());
		params.oimage->setShortName("ABEL");

		//params.oimage->setName(string("Inverted (ABEL) ")+string(params.iimage->getShortName()));
	} else if (ialgo == ABEL_HF) {
		DEBUG(1, "Hankel-Fourier implementation of ABEL inversion");
		// testing purposes only! Many problems in this one:
		// 1. code copy
		// 2. lut optimization not working properly
		// 3. should convert H_0 to FT
		bessel_alloc_t my_lut;

		//! by fixing integral_size as a single vector size for transformation, results in padding
		//! hence in a modification of the image resolution

		/*if (sym_idx == PHYS_X) {
			params.oimage->resize(params.iimage->getW(), 3*integral_size);
		} else {
			params.oimage->resize(3*integral_size, params.iimage->getH());
		}*/

		int axe_inv_mean[2];
		axe_inv_mean[0] = 0;
		axe_inv_mean[1] = 0;
        for (size_t ii=0; ii<iaxis.size(); ii++) {
			axe_inv_mean[0] += iaxis[ii].x;
			axe_inv_mean[1] += iaxis[ii].y;
		}

        DEBUG(5, "Axe average: "<<(double)axe_inv_mean[inv_idx]/iaxis.size());

        for (size_t ii = 0; ii<iaxis.size(); ii++) {
			if ((*params.iter_ptr)==-1) {
				DEBUG("aborting");
				break;
			}
			(*params.iter_ptr)++;

			axe_point[0] = iaxis[ii].x;
			axe_point[1] = iaxis[ii].y;
			//cerr << axe_point[0]  << " , " << axe_point[1] << endl;
			int copied = params.iimage->get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], copy_buffer, integral_size, PHYS_NEG);
			
			for (size_t j=copied; j<integral_size; j++)
				copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization
	
			
			phys_invert_abelHF_1D(copy_buffer, out_buffer, integral_size, &my_lut);
			//phys_invert_abelHF_1D(copy_buffer, out_buffer, copied, &my_lut);
			
			params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], out_buffer, integral_size, PHYS_NEG);
			//params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx]-axe_average+1.5*integral_size, out_buffer, integral_size, PHYS_NEG);
		
			double upper_axe_point = out_buffer[0];
			

			copied = params.iimage->get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], copy_buffer, integral_size, PHYS_POS);
			for (size_t j=copied; j<integral_size; j++)
				copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization

			phys_invert_abelHF_1D(copy_buffer, out_buffer, integral_size, &my_lut);
			//phys_invert_abelHF_1D(copy_buffer, out_buffer, copied, &my_lut);
			
			params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], out_buffer, integral_size, PHYS_POS);
			//params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx]-axe_average+1.5*integral_size, out_buffer, integral_size, PHYS_POS);

			//FIXME: pretty sure there is a better way!! ALEX!!!!
			// .alex. : fixed in some way. To be checked
			
			// me lo ricordo mica cosa fa questa riga..
			//params.oimage->set(iaxis[ii].x, iaxis[ii].y, 
			//		0.5*(params.oimage->point(iaxis[ii].x-(idir),iaxis[ii].y+(idir-1))+params.oimage->point(iaxis[ii].x+idir,iaxis[ii].y+(1-idir))));

			params.oimage->set(iaxis[ii].x, iaxis[ii].y, 0.5*out_buffer[0]+0.5*upper_axe_point); 

			DEBUG(10,"step: "<<ii);
		}
		params.oimage->setName(string("ABEL")+params.oimage->getName());
		params.oimage->setShortName("ABEL");
	} else {
		DEBUG(1, "Unknown inversion type: "<<(int)ialgo);
	}

	delete copy_buffer;
	delete out_buffer;

	params.oimage->TscanBrightness();

    DEBUG((*params.iter_ptr)); 
}



/*!
 * @}
 */
