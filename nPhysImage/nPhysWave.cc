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

#include "unwrap/unwrap.h"

/*! \addtogroup nPhysWave
 * @{
 */

void physWave::phys_wavelet_field_2D_morlet(wavelet_params &params)
{
    DEBUG(5,"start");


    if (params.data && (params.n_angles > 0) && (params.n_lambdas > 0) && (params.data->getSurf() != 0)) {

        params.olist.clear();

        unsigned int dx=params.data->getW();
        unsigned int dy=params.data->getH();

        unsigned int surf=dx*dy;

        std::vector<int> xx(dx), yy(dy);

        physC zz_morlet("zz_morlet");

        zz_morlet.resize(dx,dy);

        physD *qmap = new physD(dx, dy, 0.0, "quality");
        physD *wphase = new physD(dx,dy,0.0,"phase_2pi");
        physD *lambda = new physD(dx,dy,0.0,"lambda");
        physD *angle = new physD(dx,dy,0.0,"angle");
        physD *thick = new physD(dx,dy,0.0,"thick");
        physD *intensity = new physD(dx,dy,0.0,"intensity");


        std::vector<double> angles(params.n_angles), lambdas(params.n_lambdas), thickness(params.n_thicks);
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
        if (params.n_thicks==1) {
            thickness[0]=0.5*(params.end_thick+params.init_thick);
        } else {
            for (size_t nthick=0; nthick<params.n_thicks; nthick++)
                thickness[nthick] = params.init_thick + nthick*(params.end_thick-params.init_thick)/(params.n_thicks-1);
        }

        params.iter=0;
        *params.iter_ptr=0;


        fftw_complex *t = fftw_alloc_complex(surf);
        fftw_complex *Ft = fftw_alloc_complex(surf);
        fftw_complex *Ftmorlet = fftw_alloc_complex(surf);

        fftw_plan plan_fwd = fftw_plan_dft_2d(dy, dx, t, Ft, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan plan_bwd = fftw_plan_dft_2d(dy, dx, t, Ftmorlet, FFTW_BACKWARD, FFTW_ESTIMATE);

#pragma omp parallel for
        for (size_t i = 0; i < surf; i++) {
            t[i][0] = params.data->Timg_buffer[i];
            t[i][1] = 0;
        }

        fftw_execute(plan_fwd);

        double damp_norm=params.damp*M_PI;

        for (size_t l=0; l<thickness.size(); l++) {
            for (size_t i=0; i<lambdas.size(); i++) {
                for (size_t j=0; j<angles.size(); j++) {

                    if ((*params.iter_ptr)==-1) {
                        DEBUG("Aborting");
                        break;
                    }
                    params.iter++;
                    (*params.iter_ptr)++;
                    DEBUG(11,(100.*params.iter)/(angles.size()*lambdas.size())<<" lam "<<lambdas[i]<<", ang "<<angles[j]);
                    double angle_rad=angles[j]*_phys_deg;
                    double cr = cos(angle_rad);
                    double sr = sin(angle_rad);

                    double thick_norm=thickness[l]*M_PI/sqrt(pow(sr*dx,2)+pow(cr*dy,2));
                    double lambda_norm=lambdas[i]/sqrt(pow(cr*dx,2)+pow(sr*dy,2));

#pragma omp parallel for
                    for (unsigned int k=0;k<surf;k++) {
                        unsigned int x=k%dx;
                        unsigned int y=k/dx;

                        int xx=(x+(dx+1)/2)%dx-(dx+1)/2;
                        int yy=(y+(dy+1)/2)%dy-(dy+1)/2;

                        double xr = xx*cr - yy*sr; //rotate
                        double yr = xx*sr + yy*cr;

                        double e_x = -pow(damp_norm*(xr*lambda_norm-1.0), 2.);
                        double e_y = -pow(yr*thick_norm, 2.);

                        double gauss = exp(e_x)*exp(e_y);

                        t[k][0]=Ft[k][0]*gauss;
                        t[k][1]=Ft[k][1]*gauss;

                    }

                    fftw_execute(plan_bwd);

                    // decision
#pragma omp parallel for
                    for (size_t k=0; k<surf; k++) {
                        double qmap_local=pow(Ftmorlet[k][0],2)+pow(Ftmorlet[k][1],2);
                        if ( qmap_local > qmap->Timg_buffer[k]) {
                            qmap->Timg_buffer[k] = qmap_local;
                            wphase->Timg_buffer[k] = atan2(Ftmorlet[k][1], Ftmorlet[k][0]);
                            lambda->Timg_buffer[k] = lambdas[i];
                            angle->Timg_buffer[k] = angles[j];
                            thick->Timg_buffer[k] = thickness[l];
                        }
                    }

                }
                //! todo: this is awful: add exception?
                if ((*params.iter_ptr)==-1) {
                    DEBUG("aborting");
                    break;
                }
            }
        }

        fftw_free(t);
        fftw_free(Ft);
        fftw_free(Ftmorlet);
        fftw_destroy_plan(plan_fwd);
        fftw_destroy_plan(plan_bwd);

        if ((*params.iter_ptr)!=-1) {

#pragma omp parallel for
            for (size_t k=0; k<params.data->getSurf(); k++) {
                qmap->Timg_buffer[k]=sqrt(qmap->Timg_buffer[k])/surf;
                intensity->Timg_buffer[k] = params.data->Timg_buffer[k] - 2.0 * qmap->Timg_buffer[k]*cos(wphase->Timg_buffer[k]);
                wphase->Timg_buffer[k]/=2*M_PI;
            }

            physMath::phys_fast_gaussian_blur(*intensity,params.end_thick/2.0);

            params.olist["phase_2pi"] = wphase;
            params.olist["contrast"] = qmap;
            params.olist["lambda"] = lambda;
            params.olist["angle"] = angle;
            params.olist["intensity"] = intensity;
            params.olist["thick"] = thick;

            std::map<std::string, physD *>::const_iterator itr;
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
                        itr->second->Timg_buffer[k]   = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            }
        }

    }
    DEBUG("Out of here");
}


unsigned int physWave::opencl_closest_size(unsigned int num) {
    unsigned int closest=2*num;
    unsigned int i2,i3,i5,i7,i11,i13;
    i2=i3=i5=i7=i11=i13=0;
    // REMEMBER to use clfft 2.12 to support radix 11 and 13!
//    for (i13=0; i13<=log(num)/log(13); i13++ ) {
//        for (i11=0; i11<=log(num)/log(11); i11++ ) {
//            for (i7=0; i7<=log(num)/log(7); i7++ ) {
//                for (i5=0; i5<=log(num)/log(5); i5++ ) {
//                    for (i3=0; i3<=log(num)/log(3); i3++ ) {
//                        for (i2=0; i2<=log(num)/log(2); i2++ ) {
//                            unsigned int test_val=pow(2,i2)*pow(3,i3)*pow(5,i5)*pow(7,i7)*pow(11,i11)*pow(13,i13);
//                            if (test_val>=num && test_val<closest) {
//                                closest=test_val;
//                                if (closest==num) return num;
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
    for (i7=0; i7<=log(num)/log(7); i7++ ) {
        for (i5=0; i5<=log(num)/log(5); i5++ ) {
            for (i3=0; i3<=log(num)/log(3); i3++ ) {
                for (i2=0; i2<=log(num)/log(2); i2++ ) {
                    unsigned int test_val=pow(2,i2)*pow(3,i3)*pow(5,i5)*pow(7,i7);
                    if (test_val>=num && test_val<closest) {
                        closest=test_val;
                        if (closest==num) return num;
                    }
                }
            }
        }
    }


    DEBUG("Factorization (" << closest << ") " << num << " : " << i2 << " " << i3 << " " <<  i5 << " " <<  i7 << " " <<  i11 << " " <<  i13);
    return closest;
}

vec2i physWave::opencl_closest_size(vec2i num){
    return vec2i(opencl_closest_size(num.x()),opencl_closest_size(num.y()));
}

#define NEUTRINO_OPENCL CL_DEVICE_TYPE_DEFAULT
int physWave::openclEnabled() {
    int found_GPU=0;
#ifdef HAVE_LIBCLFFT
    // get all platforms
    cl_uint platformCount;
    clGetPlatformIDs(0, NULL, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    DEBUG("OpenCL heads found: "<<platformCount );
    clGetPlatformIDs(platformCount, &platforms[0], NULL);
    for (unsigned int i = 0; i < platformCount; i++) {
        cl_uint deviceCount=0;
        clGetDeviceIDs(platforms[i], NEUTRINO_OPENCL, 0, NULL, &deviceCount);
        DEBUG("OpenCL device count: "<<deviceCount);
        found_GPU+=deviceCount;
    }
#endif
    return found_GPU;
}

#ifdef HAVE_LIBCLFFT
std::pair<cl_platform_id,cl_device_id> physWave::get_platform_device_opencl(int num) {
    int found_GPU=0;

    cl_uint platformCount=0;
    clGetPlatformIDs(0, NULL, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, &platforms[0], NULL);

    for (unsigned int i = 0; i < platformCount; i++) {
        cl_uint deviceCount=0;
        clGetDeviceIDs(platforms[i], NEUTRINO_OPENCL, 0, NULL, &deviceCount);
        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platforms[i], NEUTRINO_OPENCL, deviceCount, &devices[0], NULL);

        for (int j = 0; j < (int) deviceCount; j++) {
            found_GPU++;
            if (found_GPU==num) {
                return std::make_pair(platforms[i],devices[j]);
            }
        }
    }
    return std::make_pair(nullptr,nullptr);
}

std::string physWave::get_platform_device_info_opencl(int num){
    std::string desc;
    cl_device_id device=get_platform_device_opencl(num).second;

    size_t valueSize;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    std::string value;

    value.resize(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, &value[0], NULL);
    desc+="Device : "+value;

    // print hardware device version
    clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &valueSize);
    value.resize(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, valueSize, &value[0], NULL);
    desc+="\nHardware version : "+value;

    // print software driver version
    clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &valueSize);
    value.resize(valueSize);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, valueSize, &value[0], NULL);
    desc+="\nSoftware version : "+value;

    // print c version supported by compiler for device
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
    value.resize(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, valueSize, &value[0], NULL);
    desc+="\nOpenCL C version : "+value;

    // print parallel compute units
    cl_uint int_val;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(int_val), &int_val, NULL);
    desc+="\nParallel compute units : "+std::to_string(int_val);

    clGetDeviceInfo(device,  CL_DEVICE_MAX_CLOCK_FREQUENCY ,sizeof(int_val), &int_val, NULL);
    desc+="\nClock frequency : "+std::to_string(int_val);

    cl_ulong ulong_val;
    clGetDeviceInfo(device,  CL_DEVICE_MAX_MEM_ALLOC_SIZE ,sizeof(ulong_val), &ulong_val, NULL);
    desc+="\nAllocatable Memory : "+std::to_string(ulong_val) +"bytes";
    DEBUG(desc);

    clGetDeviceInfo( device, CL_DEVICE_EXTENSIONS, 0, NULL, &valueSize );
    value.resize(valueSize);
    clGetDeviceInfo( device, CL_DEVICE_EXTENSIONS, valueSize, &value[0], NULL );
    desc+="\nExtensions : "+value;

    desc+="\nDouble support : ";
    desc+=((value.find("cl_khr_fp64") != std::string::npos) ? "Yes":"No");

    cl_device_fp_config cfg;
    clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cfg), &cfg, NULL);
    desc+="\nCL_DEVICE_DOUBLE_FP_CONFIG : "+std::to_string(cfg);


    return desc;
}

std::string physWave::CHECK_OPENCL_ERROR(cl_int err) {
    if (err != CL_SUCCESS) {
        switch (err) {
            case CL_DEVICE_NOT_FOUND:                           return "CL_DEVICE_NOT_FOUND";
            case CL_DEVICE_NOT_AVAILABLE:                       return "CL_DEVICE_NOT_AVAILABLE";
            case CL_COMPILER_NOT_AVAILABLE:                     return "CL_COMPILER_NOT_AVAILABLE";
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:              return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case CL_OUT_OF_RESOURCES:                           return "CL_OUT_OF_RESOURCES";
            case CL_OUT_OF_HOST_MEMORY:                         return "CL_OUT_OF_HOST_MEMORY";
            case CL_PROFILING_INFO_NOT_AVAILABLE:               return "CL_PROFILING_INFO_NOT_AVAILABLE";
            case CL_MEM_COPY_OVERLAP:                           return "CL_MEM_COPY_OVERLAP";
            case CL_IMAGE_FORMAT_MISMATCH:                      return "CL_IMAGE_FORMAT_MISMATCH";
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:                 return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case CL_BUILD_PROGRAM_FAILURE:                      return "CL_BUILD_PROGRAM_FAILURE";
            case CL_MAP_FAILURE:                                return "CL_MAP_FAILURE";
            case CL_MISALIGNED_SUB_BUFFER_OFFSET:               return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:  return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            case CL_COMPILE_PROGRAM_FAILURE:                    return "CL_COMPILE_PROGRAM_FAILURE";
            case CL_LINKER_NOT_AVAILABLE:                       return "CL_LINKER_NOT_AVAILABLE";
            case CL_LINK_PROGRAM_FAILURE:                       return "CL_LINK_PROGRAM_FAILURE";
            case CL_DEVICE_PARTITION_FAILED:                    return "CL_DEVICE_PARTITION_FAILED";
            case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:              return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            case CL_INVALID_VALUE:                              return "CL_INVALID_VALUE";
            case CL_INVALID_DEVICE_TYPE:                        return "CL_INVALID_DEVICE_TYPE";
            case CL_INVALID_PLATFORM:                           return "CL_INVALID_PLATFORM";
            case CL_INVALID_DEVICE:                             return "CL_INVALID_DEVICE";
            case CL_INVALID_CONTEXT:                            return "CL_INVALID_CONTEXT";
            case CL_INVALID_QUEUE_PROPERTIES:                   return "CL_INVALID_QUEUE_PROPERTIES";
            case CL_INVALID_COMMAND_QUEUE:                      return "CL_INVALID_COMMAND_QUEUE";
            case CL_INVALID_HOST_PTR:                           return "CL_INVALID_HOST_PTR";
            case CL_INVALID_MEM_OBJECT:                         return "CL_INVALID_MEM_OBJECT";
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case CL_INVALID_IMAGE_SIZE:                         return "CL_INVALID_IMAGE_SIZE";
            case CL_INVALID_SAMPLER:                            return "CL_INVALID_SAMPLER";
            case CL_INVALID_BINARY:                             return "CL_INVALID_BINARY";
            case CL_INVALID_BUILD_OPTIONS:                      return "CL_INVALID_BUILD_OPTIONS";
            case CL_INVALID_PROGRAM:                            return "CL_INVALID_PROGRAM";
            case CL_INVALID_PROGRAM_EXECUTABLE:                 return "CL_INVALID_PROGRAM_EXECUTABLE";
            case CL_INVALID_KERNEL_NAME:                        return "CL_INVALID_KERNEL_NAME";
            case CL_INVALID_KERNEL_DEFINITION:                  return "CL_INVALID_KERNEL_DEFINITION";
            case CL_INVALID_KERNEL:                             return "CL_INVALID_KERNEL";
            case CL_INVALID_ARG_INDEX:                          return "CL_INVALID_ARG_INDEX";
            case CL_INVALID_ARG_VALUE:                          return "CL_INVALID_ARG_VALUE";
            case CL_INVALID_ARG_SIZE:                           return "CL_INVALID_ARG_SIZE";
            case CL_INVALID_KERNEL_ARGS:                        return "CL_INVALID_KERNEL_ARGS";
            case CL_INVALID_WORK_DIMENSION:                     return "CL_INVALID_WORK_DIMENSION";
            case CL_INVALID_WORK_GROUP_SIZE:                    return "CL_INVALID_WORK_GROUP_SIZE";
            case CL_INVALID_WORK_ITEM_SIZE:                     return "CL_INVALID_WORK_ITEM_SIZE";
            case CL_INVALID_GLOBAL_OFFSET:                      return "CL_INVALID_GLOBAL_OFFSET";
            case CL_INVALID_EVENT_WAIT_LIST:                    return "CL_INVALID_EVENT_WAIT_LIST";
            case CL_INVALID_EVENT:                              return "CL_INVALID_EVENT";
            case CL_INVALID_OPERATION:                          return "CL_INVALID_OPERATION";
            case CL_INVALID_GL_OBJECT:                          return "CL_INVALID_GL_OBJECT";
            case CL_INVALID_BUFFER_SIZE:                        return "CL_INVALID_BUFFER_SIZE";
            case CL_INVALID_MIP_LEVEL:                          return "CL_INVALID_MIP_LEVEL";
            case CL_INVALID_GLOBAL_WORK_SIZE:                   return "CL_INVALID_GLOBAL_WORK_SIZE";
            case CL_INVALID_PROPERTY:                           return "CL_INVALID_PROPERTY";
            case CL_INVALID_IMAGE_DESCRIPTOR:                   return "CL_INVALID_IMAGE_DESCRIPTOR";
            case CL_INVALID_COMPILER_OPTIONS:                   return "CL_INVALID_COMPILER_OPTIONS";
            case CL_INVALID_LINKER_OPTIONS:                     return "CL_INVALID_LINKER_OPTIONS";
            case CL_INVALID_DEVICE_PARTITION_COUNT:             return "CL_INVALID_DEVICE_PARTITION_COUNT";
            default: return "Unknown error";
        }
    }
    return std::string();
}

#endif


void physWave::phys_wavelet_field_2D_morlet_opencl(wavelet_params &params) {
    DEBUG(">>>>>>>>>>>>>>>>>>>>>here " << params.opencl_unit);
#ifdef HAVE_LIBCLFFT

    DEBUG(">>>>>>>>>>>>>>>>>>>>>here " << params.opencl_unit);
    if (params.opencl_unit>0) {
        DEBUG(">>>>>>>>>>>>>>>>>>>>>here \n" << get_platform_device_info_opencl(params.opencl_unit));
        params.iter=0;
        *params.iter_ptr=0;

        vec2i newSize(opencl_closest_size(params.data->getSize()));

        double mean=params.data->sum()/params.data->getSurf();

        physD padded(newSize.x(), newSize.y(), mean);
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

        std::pair<cl_platform_id,cl_device_id> my_pair = get_platform_device_opencl(params.opencl_unit);
        cl_platform_id platform = my_pair.first;
        cl_device_id device=my_pair.second;

        /* Setup OpenCL environment. */

        cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
        cl_context ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
        check_opencl_error(err, "clCreateContext");
        cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
        check_opencl_error(err, "clCreateCommandQueue");


        // KERNEL
        std::string textkernel=
                "__kernel void gabor(__global float *inReal, __global float *inImag, __global float *outReal, __global float *outImag, const unsigned int dx, const unsigned int dy, const float damp_norm,  const float thick_norm,  const float angle, const float lambda_norm){\n"
                "    size_t id = get_global_id(0);\n"
                "    int i = id%dx;\n"
                "    int j = id/dx;\n"
                "    if (i>=(int)dx/2) i-=dx;\n"
                "    if (j>=(int)dy/2) j-=dy;\n"
                "    float cr,sr;\n"
                "    sr=sincos(angle,&cr);\n"
                "    float xr=i*cr-j*sr;\n"
                "    float yr=i*sr+j*cr;\n"
                "    float gauss=native_exp(-pown(damp_norm*(xr*lambda_norm-1.0f),2))*native_exp(-pown(yr*thick_norm,2));\n"
                "    outReal[id] = gauss * inReal[id];\n"
                "    outImag[id] = gauss * inImag[id];\n"
                "}\n"
                "__kernel void best(__global float *inReal, __global float *inImag, __global float *outQual, __global float *outPhase, __global unsigned int *outLambdaAngle, const unsigned int nlambdaangle){\n"
                "    size_t id = get_global_id(0);\n"
                "    float quality=pown(inReal[id],2)+pown(inImag[id],2);\n"
                "    if (quality>=outQual[id]) {\n"
                "        outQual[id]=quality;\n"
                "        outPhase[id]=atan2(inImag[id],inReal[id]);\n"
                "        outLambdaAngle[id]=nlambdaangle;\n"
                "    }\n"
                "}\n";

        DEBUG("KERNEL:\n" << textkernel << "\nEND KERNEL") ;

        const char *source=textkernel.c_str();

        cl_program program = clCreateProgramWithSource(ctx,1,&source, NULL, &err);
        check_opencl_error(err, "clCreateProgramWithSource");
        err=clBuildProgram(program, 1, &device, "-Werror -cl-finite-math-only", NULL, NULL);
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
        std::vector<size_t> clLengths =  {(size_t)dx, (size_t)dy};
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
        cl_float *inReal = new cl_float[N];
        cl_float *inImag = new cl_float[N];

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
            check_opencl_error(err, "tmpBuffer clCreateBuffer ");
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

        cl_mem buffersOut[2] = {0, nullptr};
        /* Prepare OpenCL memory objects : create buffer for output. */
        buffersOut[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(cl_float), NULL, &err);
        check_opencl_error(err, "buffersOut[0] clCreateBuffer");

        buffersOut[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(cl_float), NULL, &err);
        check_opencl_error(err, "buffersOut[1] clCreateBuffer");

        std::array<cl_mem,3> best = {{0,0,0}};

        /* Prepare OpenCL memory objects : create buffer for quality. */
        best[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), &inImag[0], &err);
        check_opencl_error(err, "inReal best[0] clCreateBuffer");
        err = clEnqueueWriteBuffer(queue, best[0], CL_TRUE, 0, N * sizeof(float), &inImag[0], 0, NULL, NULL);
        check_opencl_error(err, "inReal best[0] clEnqueueWriteBuffer");

        best[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(cl_float),  NULL, &err);
        check_opencl_error(err, "best[1] clCreateBuffer");
        best[2] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, N * sizeof(cl_uint),  NULL, &err);
        check_opencl_error(err, "best[2] clCreateBuffer");

        std::vector<unsigned int> lambdaangle(N,0);
        err = clEnqueueWriteBuffer(queue, best[2], CL_TRUE, 0, N * sizeof(unsigned int), &lambdaangle[0], 0, NULL, NULL);
        check_opencl_error(err, "best[2] clEnqueueWriteBuffer");


        /* Execute Forward FFT. */
        err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, buffersIn, buffersOut, tmpBuffer);
        check_opencl_error(err, "clfftEnqueueTransform");

        /* Wait for calculations to be finished. */
        err = clFinish(queue);
        check_opencl_error(err, "clFinish");

        if (tmpBufferSize > 0) {
            err=clReleaseMemObject(tmpBuffer);
            check_opencl_error(err, "clReleaseMemObject tmpBuffer");
        }

        /* Release the plan. */
        err = clfftDestroyPlan(&planHandle);
        check_opencl_error(err, "clfftDestroyPlan");


        // make new plan
        err = clfftCreateDefaultPlan(&planHandle, ctx, CLFFT_2D, &clLengths[0]);
        check_opencl_error(err, "clfftCreateDefaultPlan");

        /* Set plan parameters. */
        err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
        check_opencl_error(err, "clfftSetPlanPrecision")
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
            check_opencl_error(err, "tmpBuffer clCreateBuffer ");
            DEBUG("intermediate buffer needed");
        }

        DEBUG("kernel Gabor");
        cl_kernel kernelGabor = clCreateKernel(program, "gabor", &err);
        check_opencl_error(err, "clCreateKernel");

        clSetKernelArg(kernelGabor, 0, sizeof(cl_mem), &buffersOut[0]); //out because fft: in->out and gabor: out->in
        clSetKernelArg(kernelGabor, 1, sizeof(cl_mem), &buffersOut[1]);
        clSetKernelArg(kernelGabor, 2, sizeof(cl_mem), &buffersIn[0]);
        clSetKernelArg(kernelGabor, 3, sizeof(cl_mem), &buffersIn[1]);
        clSetKernelArg(kernelGabor, 4, sizeof(unsigned int), &dx);
        clSetKernelArg(kernelGabor, 5, sizeof(unsigned int), &dy);
        float damp_norm= params.damp * M_PI;
        clSetKernelArg(kernelGabor, 6, sizeof(float), &damp_norm);
        check_opencl_error(err, "clSetKernelArg");

        cl_kernel kernelBest = clCreateKernel(program, "best", &err);
        check_opencl_error(err, "clCreateKernel best");

        clSetKernelArg(kernelBest, 0, sizeof(cl_mem), &buffersIn[0]); //out because fft: in->out and gabor: out->in
        clSetKernelArg(kernelBest, 1, sizeof(cl_mem), &buffersIn[1]);
        clSetKernelArg(kernelBest, 2, sizeof(cl_mem), &best[0]);
        clSetKernelArg(kernelBest, 3, sizeof(cl_mem), &best[1]);
        clSetKernelArg(kernelBest, 4, sizeof(cl_mem), &best[2]);



        std::vector<double> angles(params.n_angles), lambdas(params.n_lambdas), thickness(params.n_thicks);
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
        if (params.n_thicks==1) {
            thickness[0]=0.5*(params.end_thick+params.init_thick);
        } else {
            for (size_t nthick=0; nthick<params.n_thicks; nthick++)
                thickness[nthick] = params.init_thick + nthick*(params.end_thick-params.init_thick)/(params.n_thicks-1);
        }


        size_t totalJobs=N;

        unsigned int iter=0;

        for (size_t nthick=0; nthick <params.n_thicks; nthick++) {

            for (size_t nangle=0; nangle <params.n_angles; nangle++) {

                for (size_t nlambda=0; nlambda <params.n_lambdas; nlambda++) {

                    if ((*params.iter_ptr)==-1) {
                        DEBUG("Aborting");
                        break;
                    } else {
                        (*params.iter_ptr)++;
                    }

                    DEBUG("Angle: " << (int)nlambda << " " << angles[nangle] << " Lambda: " << (int)nangle << " " << lambdas[nlambda] << " Thick: " << (int)nthick << " " << thickness[nthick] );

                    float angle_rad=angles[nangle]*_phys_deg;
                    float sr=sin(angle_rad);
                    float cr=cos(angle_rad);

                    float thick_norm=thickness[nthick] * M_PI/sqrt(pow(sr*dx,2)+pow(cr*dy,2));
                    err=clSetKernelArg(kernelGabor,7, sizeof(float), &thick_norm);

                    err=clSetKernelArg(kernelGabor, 8, sizeof(float), &angle_rad);
                    check_opencl_error(err, "clSetKernelArg");

                    float lambda_norm=lambdas[nlambda]/sqrt(pow(cr*dx,2)+pow(sr*dy,2));
                    err=clSetKernelArg(kernelGabor, 9, sizeof(float), &lambda_norm);
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

                    err=clSetKernelArg(kernelBest, 5, sizeof(unsigned int), &iter);
                    check_opencl_error(err, "clSetKernelArg");

                    clEnqueueNDRangeKernel(queue, kernelBest, 1, NULL, &totalJobs, NULL, 0, NULL, NULL);
                    err = clFinish(queue);
                    check_opencl_error(err, "clFinish");
                    iter++;
                    params.iter++;

                }
            }
        }

        std::vector<float> quality_sqr(N,0);
        err = clEnqueueReadBuffer(queue, best[0], CL_TRUE, 0, N * sizeof(float), &quality_sqr[0], 0, NULL, NULL);
        check_opencl_error(err, "clEnqueueReadBuffer");

        std::vector<float> phase(N,0);
        err = clEnqueueReadBuffer(queue, best[1], CL_TRUE, 0, N * sizeof(float), &phase[0], 0, NULL, NULL);
        check_opencl_error(err, "clEnqueueReadBuffer");

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
        clReleaseKernel(kernelBest);
        clReleaseKernel(kernelGabor);
        clReleaseProgram(program);

        delete [] inReal;
        delete [] inImag;
        /* Release the plan. */
        err = clfftDestroyPlan(&planHandle );
        check_opencl_error(err, "clfftDestroyPlan");

        /* Release clFFT library. */
        clfftTeardown( );

        /* Release OpenCL working objects. */
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);

        params.olist.clear();

        physD *nQuality = new physD(params.data->getW(),params.data->getH(),0,"Quality");
        physD *nPhase = new physD(params.data->getW(),params.data->getH(),0,"Phase");
        physD *nIntensity = new physD(params.data->getW(),params.data->getH(),0,"Intensity");

        for (size_t j=0; j<params.data->getH(); j++) {
            for (size_t i=0; i<params.data->getW(); i++) {
                unsigned int k=(j+offset.y())*dx+i+offset.x();
                nQuality->set(i,j,sqrt(quality_sqr[k]));
                nPhase->set(i,j,phase[k]/(2*M_PI));
                nIntensity->set(i,j,params.data->point(i,j) - 2.0*nQuality->point(i,j)*cos(phase[k]));
            }
        }

//            physMath::phys_fast_gaussian_blur(*nIntensity,params.thickness/2.0);

        params.olist["phase_2pi"] = nPhase;
        params.olist["contrast"] = nQuality;
        params.olist["intensity"] = nIntensity;


        if (params.n_thicks>1) {
            physD *nThick = new physD(params.data->getW(),params.data->getH(),0,"Thick");
            for (size_t j=0; j<params.data->getH(); j++) {
                for (size_t i=0; i<params.data->getW(); i++) {
                    unsigned int k=j*dx+i;
                    unsigned int val=lambdaangle[k]/params.n_lambdas/params.n_angles;
                    nThick->set(i,j,thickness[val]);
                }
            }
            params.olist["thick"] = nThick;
        }

        if (params.n_angles>1) {
            physD *nAngle = new physD(params.data->getW(),params.data->getH(),0,"Angle");
            for (size_t j=0; j<params.data->getH(); j++) {
                for (size_t i=0; i<params.data->getW(); i++) {
                    unsigned int k=j*dx+i;
                    unsigned int val=lambdaangle[k]/params.n_lambdas%params.n_angles;
                    if (val >= angles.size()) {
                        DEBUG("ERROR \n" << i << " " << j << " " << val << " " << angles.size() << " " << k << " " << lambdaangle[k] << " " << params.n_lambdas << " " << dx );
                    } else {
                        nAngle->set(i,j,angles[val]);
                    }
                }
            }
            params.olist["angle"] = nAngle;
        }

        if (params.n_lambdas>1) {
            physD *nLambda = new physD(params.data->getW(),params.data->getH(),0,"Lambda");
            for (size_t j=0; j<params.data->getH(); j++) {
                for (size_t i=0; i<params.data->getW(); i++) {
                    unsigned int k=j*dx+i;
                    unsigned int val=lambdaangle[k]%params.n_lambdas;
                    nLambda->set(i,j,lambdas[val]);
                }
            }
            params.olist["lambda"] = nLambda;
        }

        for(std::map<std::string, physD *>::const_iterator itr = params.olist.begin(); itr != params.olist.end(); ++itr) {
            itr->second->TscanBrightness();
            itr->second->set_origin(params.data->get_origin());
            itr->second->set_scale(params.data->get_scale());
            itr->second->setFromName(params.data->getFromName());
            itr->second->setShortName(itr->first);
            itr->second->setName(itr->first+ " "+params.data->getName());
#pragma omp parallel for
            for (size_t k=0; k<params.data->getSurf(); k++) {
                if (std::isnan(params.data->point(k))) {
                    itr->second->set(k,std::numeric_limits<double>::quiet_NaN());
                }
            }
        }
    }
#endif
}



// ---------------------- thread transport functions ------------------------

void physWave::phys_wavelet_trasl_opencl(void *params, int &iter) {
    DEBUG("Enter here");
    ((wavelet_params *)params)->iter_ptr = &iter;
    physWave::phys_wavelet_field_2D_morlet_opencl(*((wavelet_params *)params));
}

void physWave::phys_wavelet_trasl_cpu(void *params, int &iter) {
    DEBUG("Enter here");
    ((wavelet_params *)params)->iter_ptr = &iter;
    physWave::phys_wavelet_field_2D_morlet(*((wavelet_params *)params));
}


// unwrap methods
void physWave::phys_phase_unwrap(physD &wphase, physD &quality, enum unwrap_strategy strategy, physD &uphase)
{
    uphase.resize(wphase.getW(), wphase.getH());
    if (wphase.getSurf()) {
        uphase.set_origin(wphase.get_origin());
        uphase.set_scale(wphase.get_scale());
        uphase.setName("Unwrap "+wphase.getName());
        uphase.setFromName(wphase.getFromName());

        switch (strategy) {
            case SIMPLE_HV :
                unwrap::simple_h(wphase, uphase);
                unwrap::simple_v(wphase, uphase);
                break;

            case SIMPLE_VH :
                unwrap::simple_v(wphase, uphase);
                unwrap::simple_h(wphase, uphase);
                break;

            case GOLDSTEIN :
                unwrap::goldstein(wphase, uphase);
                break;

            case QUALITY :
                unwrap::quality(wphase, uphase, quality);
                break;

            case MIGUEL :
                unwrap::miguel(wphase, uphase);
                break;

            case MIGUEL_QUALITY :
                unwrap::miguel_quality(wphase, uphase, quality);
                break;

        }
        DEBUG("here");
        uphase.TscanBrightness();
    }
}

physD physWave::phys_synthetic_interferogram (physD *phase_over_2pi, physD *quality){
    physD synthetic;
    if (phase_over_2pi && quality) {
        if (phase_over_2pi->getW()==quality->getW() && phase_over_2pi->getH()==quality->getH()) {
            synthetic.resize(phase_over_2pi->getW(),phase_over_2pi->getH());
#pragma omp parallel for
            for (size_t ii=0; ii<phase_over_2pi->getSurf(); ii++) {
                synthetic.set(ii,M_PI*quality->point(ii)*(1.0+cos(phase_over_2pi->point(ii)*2*M_PI)));
            }
            synthetic.prop=phase_over_2pi->prop;
            synthetic.setShortName("synthetic");
            synthetic.setName("synthetic("+phase_over_2pi->getName()+","+quality->getName()+")");
            synthetic.TscanBrightness();
        }
    }
    return synthetic;
}

void
physWave::phys_subtract_carrier(physD &iphys, double alpha, double lambda)
{
    double kx = cos(alpha*_phys_deg)/lambda;
    double ky = -sin(alpha*_phys_deg)/lambda;

#pragma omp parallel for collapse(2)
    for (size_t ii=0; ii<iphys.getW(); ii++) {
        for (size_t jj=0; jj<iphys.getH(); jj++) {
            iphys.Timg_matrix[jj][ii] -= ii*kx + jj*ky;
        }
    }
    iphys.TscanBrightness();
}

vec2f lambda_angle (int imax, int jmax, int dx, int dy) {
    mcomplex freq(imax/((double)dx),jmax/((double)dy));
    return vec2f(1.0/freq.mod(),fmod(freq.arg()*180.0/M_PI,180.0));
}

//! this function returns the carrier vec2f(angle[deg],interfringe[px])
std::vector<vec2f>
physWave::phys_guess_carrier(physD &phys, double weight)
{
    std::vector<vec2f> retlist;
    size_t dx=phys.getW();
    size_t dy=phys.getH();

    int imax=0,jmax=0;

    if (dx!=0 && dy!=0) {
        fftw_complex *myData=fftw_alloc_complex(dy*(dx/2+1));
        fftw_plan plan=fftw_plan_dft_r2c_2d(dy,dx, phys.Timg_buffer, myData, FFTW_ESTIMATE);
        fftw_execute(plan);

        double valmax=0.0;
        std::vector<std::pair<vec2f,double>> my_vec;

        for (int i=0; i<dx/2+1; i++) {
            for (int j=0; j<dy; j++) {
                int j1=(dy/2+1)-(j+dy/2+1)%dy;
                double r=sqrt(i*i+j1*j1);
                int k=i+j*(dx/2+1);
                double val=pow(r,weight)*vec2f(myData[k][0],myData[k][1]).mod();
                if (val>valmax && (i>0||j>0)) {
                    valmax=val;
                    imax=i;
                    jmax=j1;
                    DEBUG(">>>" << i << " " << j << " " << j1 << " " << k << "  =  " << valmax);
                    my_vec.clear();
                    for (int ii=i-1;ii<=i+1;ii++) {
                        DEBUG("here"<< j-1 << " " << j+1);
                        for (int jj=j-1;jj<=j+1;jj++) {
                            DEBUG("here2");
                            int iiclean=(ii+dx/2+1)%(dx/2+1);
                            int jjclean=(jj+dy)%(dy);
                            int j1clean=(dy/2+1)-(jjclean+dy/2+1)%dy;
                            int kk=k=iiclean+jjclean*(dx/2+1);
                            double rclean=sqrt(iiclean*iiclean+j1clean*j1clean);
                            vec2f my_l_a=lambda_angle(iiclean,j1clean,dx,dy);
                            double my_weight=pow(rclean,weight)*vec2f(myData[kk][0],myData[kk][1]).mod();
                            DEBUG(my_l_a << " " << my_weight);
                            my_vec.push_back(std::make_pair(my_l_a, my_weight));
                        }
                    }
                }
            }
        }
        DEBUG(dx << " " << dy << " " << dx/2+1 << " " << dy/2+1);
        vec2f a_l_mean(0,0);
        double sum_weights=0;
        for (auto &a: my_vec) {
            a_l_mean += a.first*a.second;
            sum_weights += a.second;
        }
        a_l_mean/=sum_weights;
        DEBUG(a_l_mean);

        fftw_free(myData);
        fftw_destroy_plan(plan);
    }

    if (imax!=0 || jmax!=0) {
        vec2f retCarrier=lambda_angle(imax,jmax,dx,dy);
        vec2f carr_m10=lambda_angle(imax-1,jmax,dx,dy);
        vec2f carr_p10=lambda_angle(imax+1,jmax,dx,dy);
        vec2f carr_0m1=lambda_angle(imax,jmax-1,dx,dy);
        vec2f carr_0p1=lambda_angle(imax,jmax+1,dx,dy);
        std::stringstream ss;
        ss << "carrier:" << retCarrier << "; i-" << carr_m10 << " i+" << carr_p10 << "; j-" << carr_0m1 << " j+" << carr_0p1 << std::endl;
        DEBUG("-------------------------");
        DEBUG(ss.str());
        DEBUG("-------------------------");
        retlist.push_back(retCarrier);
        retlist.push_back(carr_m10);
        retlist.push_back(carr_p10);
        retlist.push_back(carr_0m1);
        retlist.push_back(carr_0p1);
    }

    return retlist;
}

// --------------------------------------------------------------------- integral inversions --

void
physWave::phys_apply_inversion_gas(physD &invimage, double probe_wl, double res, double molar_refr)
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
    invimage.prop["unitsCB"]="m-3";
    invimage.TscanBrightness();
}

void
physWave::phys_apply_inversion_plasma(physD &invimage, double probe_wl, double res)
{
    double kappa = 2*M_PI/probe_wl; // unit: 1/m
    double mult = _phys_emass*_phys_vacuum_eps*_phys_cspeed*_phys_cspeed/(_phys_echarge*_phys_echarge); // unit: 1/m
    DEBUG(5,"resolution: "<< res << ", probe: " << probe_wl << ", mult: " << mult);
    for (size_t ii=0; ii<invimage.getSurf(); ii++) {
        double the_point = invimage.point(ii)/res;
//        invimage.set(ii, - mult * (the_point*the_point+2*kappa*the_point));
        invimage.set(ii, - mult * 2*kappa*the_point);
    }
    invimage.prop["unitsCB"]="m-3";
    invimage.TscanBrightness();
}

void
physWave::phys_apply_inversion_protons(physD &invimage, double energy, double res, double distance, double magnification)
{
    double mult = (2.0*_phys_vacuum_eps*magnification*energy)/(distance*res);
    physMath::phys_multiply(invimage,mult);
    invimage.set_scale(res*1e2,res*1e2);
    invimage.prop["unitsX"]="cm";
    invimage.prop["unitsY"]="cm";
    invimage.prop["unitsCB"]="C/m-3";
}


//! General function for Abel inversion
void physWave::phys_invert_abel(abel_params &params)
{

    if (params.iimage->getSurf() == 0) {
        return;
    }

    params.oimage = new physD (params.iimage->getW(), params.iimage->getH(),/*std::numeric_limits<double>::quiet_NaN()*/ 0.0,"Inverted");
    params.oimage->set_origin(params.iimage->get_origin());
    params.oimage->set_scale(params.iimage->get_scale());

    
    // adorabile ridondanza qui..

    // 1. set direction indexes
    enum phys_direction sym_idx, inv_idx;
    switch (params.idir) {
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

    size_t integral_size = params.iimage->getSizeByIndex(inv_idx);

    *params.iter_ptr = 0;
    // 2. switch on algo

    if (params.ialgo == ABEL) {
        DEBUG(1, "Plain ABEL inversion");
#pragma omp parallel for shared (params)
        for (size_t ii = 0; ii<params.iaxis.size(); ii++) {
            if ((*params.iter_ptr)!=-1) {

                (*params.iter_ptr)++;

                size_t axe_point[2];

                std::vector<double> copy_buffer(integral_size), out_buffer(integral_size);
                axe_point[0] = params.iaxis[ii].x();
                axe_point[1] = params.iaxis[ii].y();

                int copied=0;

                copied = params.iimage->get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &copy_buffer[0], integral_size, PHYS_NEG);

                for (size_t j=copied; j<integral_size; j++)
                    copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization

                phys_invert_abel_1D(copy_buffer, out_buffer);

                double axis_val=0.5*out_buffer[1];

                params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &out_buffer[0], integral_size, PHYS_NEG);

                copied = params.iimage->get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &copy_buffer[0], integral_size, PHYS_POS);

                for (size_t j=copied; j<integral_size; j++)
                    copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization

                phys_invert_abel_1D(copy_buffer, out_buffer);

                axis_val+=0.5*out_buffer[1];

                params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &out_buffer[0], integral_size, PHYS_POS);

                // set on axis value
                params.oimage->set(params.iaxis[ii].x(), params.iaxis[ii].y(),axis_val);
            }
        }

        params.oimage->setName(std::string("ABEL ")+params.oimage->getName());
        params.oimage->setShortName("ABEL");

        //params.oimage->setName(std::string("Inverted (ABEL) ")+std::string(params.iimage->getShortName()));
    } else if (params.ialgo == ABEL_HF) {

        DEBUG(1, "Hankel-Fourier implementation of ABEL inversion");
        // testing purposes only! Many problems in this one:
        // 1. code copy
        // 2. lut optimization not working properly
        // 3. should convert H_0 to FT
        std::vector<double> my_lut;

        //! by fixing integral_size as a single vector size for transformation, results in padding
        //! hence in a modification of the image resolution

        /*if (sym_idx == PHYS_X) {
            params.oimage->resize(params.iimage->getW(), 3*integral_size);
        } else {
            params.oimage->resize(3*integral_size, params.iimage->getH());
        }*/

        //		int axe_inv_mean[2];
        //		axe_inv_mean[0] = 0;
        //		axe_inv_mean[1] = 0;
        //		for (size_t ii=0; ii<params.iaxis.size(); ii++) {
        //			axe_inv_mean[0] += params.iaxis[ii].x();
        //			axe_inv_mean[1] += params.iaxis[ii].y();
        //		}

        //		DEBUG(5, "Axe average: "<<(double)axe_inv_mean[inv_idx]/params.iaxis.size());

        std::vector<double> copy_buffer(integral_size), out_buffer(integral_size);

        std::vector<double> Fivec;
        fftw_plan r2rplan=nullptr;

        for (size_t ii = 0; ii<params.iaxis.size(); ii++) {
            if ((*params.iter_ptr)!=-1) {
                (*params.iter_ptr)++;

                int axe_point[2] = { params.iaxis[ii].x(), params.iaxis[ii].y()};

                //cerr << axe_point[0]  << " , " << axe_point[1] << endl;
                int copied = params.iimage->get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &copy_buffer[0], integral_size, PHYS_NEG);

                for (size_t j=copied; j<integral_size; j++)
                    copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization


                phys_invert_abelHF_1D(copy_buffer, out_buffer, my_lut, Fivec, r2rplan);
                //phys_invert_abelHF_1D(&copy_buffer[0], out_buffer, copied, &my_lut);
                double upper_axe_point = 0.5*out_buffer[0];

                params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &out_buffer[0], integral_size, PHYS_NEG);
                //params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx]-axe_average+1.5*integral_size, &out_buffer[0], integral_size, PHYS_NEG);

                copied = params.iimage->get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &copy_buffer[0], integral_size, PHYS_POS);
                for (size_t j=copied; j<integral_size; j++)
                    copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization

                phys_invert_abelHF_1D(copy_buffer, out_buffer, my_lut, Fivec, r2rplan);
                //phys_invert_abelHF_1D(copy_buffer, &out_buffer[0], copied, &my_lut);

                upper_axe_point+=0.5*out_buffer[0];

                params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &out_buffer[0], integral_size, PHYS_POS);
                //params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx]-axe_average+1.5*integral_size, &out_buffer[0], integral_size, PHYS_POS);

                params.oimage->set(params.iaxis[ii].x(), params.iaxis[ii].y(), upper_axe_point);

                DEBUG(10,"step: "<<ii);
            }
        }

        fftw_destroy_plan(r2rplan);

        params.oimage->setName(std::string("ABEL")+params.oimage->getName());
        params.oimage->setShortName("ABEL");
    } else {
        DEBUG(1, "Unknown inversion type: "<<(int)params.ialgo);
    }

    params.oimage->TscanBrightness();

    DEBUG((*params.iter_ptr));
}



/*!
 * @}
 */
