/*
 *
 *    Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 *
 *    This file is part of neutrino.
 *
 *    Neutrino is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Neutrino is distributed in the hope that it will be useful,
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
#include "nOpenCL.h"
#include "neutrino.h"
#include "clFFT.h"
#include "nPhysWave.h"
#include "nRect.h"

nOpenCL::nOpenCL(neutrino *nparent, QString winname)
    : nGenericPan(nparent, winname) , physCL(NULL)
{
    my_w.setupUi(this);
    region =  new nRect(nparent);
    region->setParentPan(panName,1);
    region->setRect(QRectF(100,100,100,100));
    connect(my_w.actionRegion, SIGNAL(triggered()), region, SLOT(togglePadella()));

    connect(my_w.weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(on_actionGuess_Carrier_triggered()));

    decorate();
}

void nOpenCL::on_actionGuess_Carrier_triggered() {
    nPhysD *image=getPhysFromCombo(my_w.image);
    if (image) {
        QRect geom2=region->getRect(image);
        nPhysD datamatrix;
        datamatrix = image->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());

        vec2f vecCarr=phys_guess_carrier(datamatrix, my_w.weightCarrier->value());

        if (vecCarr.first()==0) {
            my_w.statusbar->showMessage(tr("ERROR: Problem finding the carrier"), 5000);
        } else {
            my_w.statusbar->showMessage(tr("Carrier: ")+QString::number(vecCarr.first())+"px "+QString::number(vecCarr.second())+"deg", 5000);
            my_w.interfringeCarrier->setValue(vecCarr.first());
            my_w.angleCarrier->setValue(vecCarr.second());
        }
    }
}

void nOpenCL::on_doIt_released () {
    nPhysD *image=getPhysFromCombo(my_w.image);
    if (image) {

        saveDefaults();
        QRect geom2=region->getRect(image);
        nPhysD datamatrix;
        datamatrix = image->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());
        QProgressDialog progress("OpenCL Wavelet",QString(), 0, my_w.numAngle->value()*my_w.numStretch->value(), this);
        progress.setWindowModality(Qt::WindowModal);
        progress.show();

        QTime timer;
        timer.start();

        vec2 newSize(opencl_closest_size(datamatrix.getSize()));

        double mean=datamatrix.sum()/datamatrix.getSurf();

        nPhysD padded(newSize.x(), newSize.y(), mean);
        bidimvec<int> offset=(newSize-datamatrix.get_size())/2;
        DEBUG("padding offset : " << offset);
        padded.set_origin(datamatrix.get_origin()+offset);

        for (size_t j=0; j<datamatrix.getH(); j++) {
            for (size_t i=0; i<datamatrix.getW(); i++) {
                padded.set(i+offset.x(),j+offset.y(),datamatrix.getPoint(i,j));
            }
        }

        unsigned int dx = padded.getW();
        unsigned int dy = padded.getH();

        cl_int err;

        QSettings settings("neutrino","");
        settings.beginGroup("Preferences");
        int num_device = settings.value("openclUnit").toInt();

        if (num_device>0) {
            pair<cl_platform_id,cl_device_id> my_pair = get_platform_device_opencl(num_device);
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
                    "    float gauss=exp(-pown(damp_norm*(xr*lambda_norm-1.0f),2))*exp(-pown(yr*thick_norm,2));\n"
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
            err=clBuildProgram(program, 1, &device, "-Werror", NULL, NULL);
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
            float damp_norm= my_w.damp->value() * M_PI;
            clSetKernelArg(kernelGabor, 9, sizeof(float), &damp_norm);

            cl_kernel kernelBest = clCreateKernel(program, "best", &err);
            check_opencl_error(err, "clCreateKernel best");

            clSetKernelArg(kernelBest, 0, sizeof(cl_mem), &buffersIn[0]); //out because fft: in->out and gabor: out->in
            clSetKernelArg(kernelBest, 1, sizeof(cl_mem), &buffersIn[1]);
            clSetKernelArg(kernelBest, 2, sizeof(cl_mem), &best[0]);
            clSetKernelArg(kernelBest, 3, sizeof(cl_mem), &best[1]);
            clSetKernelArg(kernelBest, 4, sizeof(cl_mem), &best[2]);


            size_t totalJobs=N;


            unsigned int iter=0;
            for (unsigned char nangle=0; nangle <my_w.numAngle->value(); nangle++) {

                float angle;
                if (my_w.numAngle->value()==1) {
                    angle=my_w.angleCarrier->value() + 0.5*(my_w.maxAngle->value()+my_w.minAngle->value());
                } else {
                    angle=my_w.angleCarrier->value() + my_w.minAngle->value() + nangle*(my_w.maxAngle->value()-my_w.minAngle->value())/(my_w.numAngle->value()-1);
                }
                angle=angle* M_PI/180.0;

                for (unsigned char nlambda=0; nlambda <my_w.numStretch->value(); nlambda++) {

                    float lambda;
                    if (my_w.numStretch->value()==1) {
                        lambda=my_w.interfringeCarrier->value() * 0.5*(my_w.maxStretch->value()+my_w.minStretch->value());
                    } else {
                        lambda=my_w.interfringeCarrier->value() * (my_w.minStretch->value() + nlambda*(my_w.maxStretch->value()-my_w.minStretch->value())/(my_w.numStretch->value()-1));
                    }
                    DEBUG("Angle: " << (int)nlambda << " " << angle << " Lambda: " << (int)nangle << " " << lambda );


                    float sr=sin(angle);
                    float cr=cos(angle);
                    err=clSetKernelArg(kernelGabor, 6, sizeof(float), &sr);
                    check_opencl_error(err, "clSetKernelArg");
                    err=clSetKernelArg(kernelGabor, 7, sizeof(float), &cr);
                    check_opencl_error(err, "clSetKernelArg");
                    float lambda_norm=lambda/sqrt(pow(cr*dx,2)+pow(sr*dy,2));
                    err=clSetKernelArg(kernelGabor, 8, sizeof(float), &lambda_norm);
                    check_opencl_error(err, "clSetKernelArg");

                    float thick_norm=my_w.thickness->value() * M_PI/sqrt(pow(sr*dx,2)+pow(cr*dy,2));
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

                    err=clSetKernelArg(kernelBest, 5, sizeof(unsigned int), &iter);
                    check_opencl_error(err, "clSetKernelArg");

                    clEnqueueNDRangeKernel(queue, kernelBest, 1, NULL, &totalJobs, NULL, 0, NULL, NULL);
                    err = clFinish(queue);
                    check_opencl_error(err, "clFinish");
                    progress.setValue(++iter);

                }
            }

            /* Fetch results of calculations : Real and Imaginary. */
            vector<float> quality(N,0);
            err = clEnqueueReadBuffer(queue, best[0], CL_TRUE, 0, N * sizeof(float), &quality[0], 0, NULL, NULL);
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

            nPhysD *nQuality = new nPhysD(datamatrix.getW(),datamatrix.getH(),0,"Quality");
            nPhysD *nPhase = new nPhysD(datamatrix.getW(),datamatrix.getH(),0,"Phase");

            for (size_t j=0; j<datamatrix.getH(); j++) {
                for (size_t i=0; i<datamatrix.getW(); i++) {
                    unsigned int k=(j+offset.y())*dx+i+offset.x();
                    nQuality->set(i,j,quality[k]);
                    nPhase->set(i,j,phase[k]);
                }
            }

            nQuality->TscanBrightness();
            nQuality->set_origin(datamatrix.get_origin());
            nparent->addPhys(nQuality);
            nPhase->TscanBrightness();
            nPhase->set_origin(datamatrix.get_origin());
            nparent->addPhys(nPhase);

            if (my_w.numAngle->value()>1) {
                nPhysD *nAngle = new nPhysD(datamatrix.getW(),datamatrix.getH(),0,"Angle");
                for (size_t j=0; j<datamatrix.getH(); j++) {
                    for (size_t i=0; i<datamatrix.getW(); i++) {
                        unsigned int k=(j+offset.y())*dx+i+offset.x();
                        float angle=my_w.minAngle->value() + (lambdaangle[k]%my_w.numAngle->value())*(my_w.maxAngle->value()-my_w.minAngle->value())/(my_w.numAngle->value()-1);
                        nAngle->set(i,j,angle);
                    }
                }
                nAngle->TscanBrightness();
                nAngle->set_origin(datamatrix.get_origin());
                nparent->addPhys(nAngle);
            }

            if (my_w.numStretch->value()>1) {
                nPhysD *nLambda = new nPhysD(datamatrix.getW(),datamatrix.getH(),0,"Lambda");
                for (size_t j=0; j<datamatrix.getH(); j++) {
                    for (size_t i=0; i<datamatrix.getW(); i++) {
                        unsigned int k=(j+offset.y())*dx+i+offset.x();
                        float lambda=my_w.minStretch->value() + (lambdaangle[k]/my_w.numAngle->value())*(my_w.maxStretch->value()-my_w.minStretch->value())/(my_w.numStretch->value()-1);
                        nLambda->set(i,j,lambda);
                    }
                }
                nLambda->TscanBrightness();
                nLambda->set_origin(datamatrix.get_origin());
                nparent->addPhys(nLambda);
            }


            QString out;
            out.sprintf("OpenCL: %.2f sec, %.2f Mpx/s",1.0e-3*timer.elapsed(), 1.0e-3*my_w.numAngle->value()*my_w.numStretch->value()*datamatrix.getSurf()/timer.elapsed());
            my_w.statusbar->showMessage(out, 50000);


        }

        progress.close();

    }
}




