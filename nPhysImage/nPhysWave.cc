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
#include "unwrapping/unwrap_miguel.h"
#include "unwrapping/unwrap_goldstein.h"
#include "unwrapping/unwrap_quality.h"

using namespace std;

/*! \addtogroup nPhysWave
 * @{
 */ 

//! this is nomore used
list<nPhysD *> *
phys_wavelet_field_2D(nPhysD &ifg, wavelet_params &wave_params)
{
	DEBUG(5,"start");
	list<nPhysD *> *olist = new list<nPhysD *>;

	if ((wave_params.n_angles > 0) && (wave_params.n_lambdas > 0) && (ifg.getSurf() != 0)) {

		double hwF = ifg.getW()/2.;
		double hhF = ifg.getH()/2.;
		int surf=ifg.getSurf();

		nPhysD xx("xx"), yy("yy"), dbuffer; 
		nPhysImageF<mcomplex> zz_morlet("zz_morlet"), zz_convolve("zz_convolve");
		nPhysD zz_Fmorlet("Fmorlet");
	
		meshgrid_data mesh_params = {-hwF, hwF, -hhF, hhF, ifg.getW(), ifg.getH()};
		phys_generate_meshgrid(&mesh_params, xx, yy);
	
		vector<double> angles(wave_params.n_angles), lambdas(wave_params.n_lambdas);
		int n_iter = angles.size()*lambdas.size();
		
		nPhysD *qmap, *wphase, *lambda, *angle;
		qmap = new nPhysD(xx.getW(), xx.getH(), 0.0, "quality map");
		
		wphase = new nPhysD("wrap phase");
		wphase->resize(xx.getW(), xx.getH());
		lambda = new nPhysD("lambda");
		lambda->resize(xx.getW(), xx.getH());
		angle = new nPhysD("angle");
		angle->resize(xx.getW(), xx.getH());


		if (wave_params.n_angles==1) {
			angles[0]=0.5*(wave_params.end_angle+wave_params.init_angle);
		} else {
			for (size_t i=0; i<wave_params.n_angles; i++)
				angles[i] = wave_params.init_angle + i*(wave_params.end_angle-wave_params.init_angle)/(wave_params.n_angles-1);
		}	
		if (wave_params.n_lambdas==1) {
			lambdas[0]=0.5*(wave_params.end_lambda+wave_params.init_lambda);
		} else {
			for (size_t i=0; i<wave_params.n_lambdas; i++)
				lambdas[i] = wave_params.init_lambda + i*(wave_params.end_lambda-wave_params.init_lambda)/(wave_params.n_lambdas-1);
		}

		nPhysImageF<mcomplex> Fmain_window=ifg.ft2(PHYS_FORWARD);

		wave_params.iter=0;
		
#ifndef __winz
//		clock_t initTime=clock();
#endif
		for (size_t i=0; i<lambdas.size(); i++) {
			for (size_t j=0; j<angles.size(); j++) {
	
				if (wave_params.iter==-1) break; 
				wave_params.iter++;
				DEBUG(5,(100.*wave_params.iter)/n_iter<<"\% lam "<<lambdas[i]<<", ang "<<angles[j]);
	
				// morlet generation (to be optimized)
				morlet_data morlet_params = {lambdas[i], angles[j] * _phys_deg, wave_params.thickness, wave_params.damp};
				
				phys_generate_Fmorlet(&morlet_params, xx, yy, zz_morlet);
				DEBUG(10,"morlet generated");
				
				zz_morlet.fftshift();
				DEBUG(10,"fft shifted");
				phys_convolve_Fm1_Fm2(Fmain_window, zz_morlet, zz_convolve);
				DEBUG(10,"convolved");
	
				// decision
				for (size_t k=0; k<xx.getSurf(); k++) {
					double qmap_local=zz_convolve.Timg_buffer[k].mcabs()/surf;
					if ( qmap_local > qmap->Timg_buffer[k]) {
						qmap->Timg_buffer[k] = qmap_local;
						wphase->Timg_buffer[k] = zz_convolve.Timg_buffer[k].arg()/(2*M_PI);
						
						lambda->Timg_buffer[k] = lambdas[i];
						angle->Timg_buffer[k] = angles[j];
					}
				}
	
			}
		}
// #ifndef __winz
// 		cerr<<"Calculation end (n_iter "<<wave_params.iter<<"): "<<clock()-initTime<<endl;	
// #endif
// 
		olist->push_back(wphase);
		olist->push_back(qmap);
		olist->push_back(lambda);
		olist->push_back(angle);

		list<nPhysD *>::const_iterator itr;
		for(itr = olist->begin(); itr != olist->end(); ++itr) {
			(*itr)->TscanBrightness();
			(*itr)->set_origin(ifg.get_origin());
			(*itr)->set_scale(ifg.get_scale());
			(*itr)->setFromName(ifg.getFromName());
			(*itr)->setShortName((*itr)->getName());
			(*itr)->setName((*itr)->getShortName()+ " "+ifg.getName());
		}

		return olist;
	}
	DEBUG(5,"end");
	return olist;
}

// list<nPhysD *> *
// phys_wavelet_field_2D(nPhysD &ifg, wavelet_params &wave_params)
// {
// 	cout<<"[phys_wave] starting wavelet analysis "<< ifg.getSurf() << endl;
// 	list<nPhysD *> *olist = new list<nPhysD *>;
// 
// 	if ((wave_params.n_angles > 0) && (wave_params.n_lambdas > 0) && (ifg.getSurf() != 0)) {
// 
// 		double hwF = ifg.getW()/2.;
// 		double hhF = ifg.getH()/2.;
// 		int surf=ifg.getSurf();
// 
// 		nPhysD xx("xx"), yy("yy"), dbuffer; 
// 		nPhysImageF<mcomplex> zz_morlet("zz_morlet"), zz_convolve("zz_convolve"), Fmain_window("Fmain-window"), *Fptr;
// 		nPhysD zz_Fmorlet("Fmorlet");
// 	
// 		meshgrid_data mesh_params = {-hwF, hwF, -hhF, hhF, ifg.getW(), ifg.getH()};
// 		phys_generate_meshgrid(&mesh_params, xx, yy);
// 	
// 		vector<double> angles(wave_params.n_angles), lambdas(wave_params.n_lambdas);
// 		int n_iter = angles.size()*lambdas.size();
// 		
// 		nPhysD *qmap, *wphase, *lambda, *angle;
// 		qmap = new nPhysD(xx.getW(), xx.getH(), 0.0, "quality map");
// 		
// 		wphase = new nPhysD("wrap phase");
// 		wphase->resize(xx.getW(), xx.getH());
// 		lambda = new nPhysD("lambda");
// 		lambda->resize(xx.getW(), xx.getH());
// 		angle = new nPhysD("angle");
// 		angle->resize(xx.getW(), xx.getH());
// 
// 
// 		if (wave_params.n_angles==1) {
// 			angles[0]=0.5*(wave_params.end_angle+wave_params.init_angle);
// 		} else {
// 			for (size_t i=0; i<wave_params.n_angles; i++)
// 				angles[i] = wave_params.init_angle + i*(wave_params.end_angle-wave_params.init_angle)/(wave_params.n_angles-1);
// 		}	
// 		if (wave_params.n_lambdas==1) {
// 			lambdas[0]=0.5*(wave_params.end_lambda+wave_params.init_lambda);
// 		} else {
// 			for (size_t i=0; i<wave_params.n_lambdas; i++)
// 				lambdas[i] = wave_params.init_lambda + i*(wave_params.end_lambda-wave_params.init_lambda)/(wave_params.n_lambdas-1);
// 		}
// 
// 		Fptr = ifg.getFFT(1); 
// 
// 		Fmain_window = *Fptr;
// 		delete Fptr;
// 		
// 		wave_params.iter=0;
// 		
// #ifndef __winz
// 		clock_t initTime=clock();
// #endif
// 		for (size_t i=0; i<lambdas.size(); i++) {
// 			for (size_t j=0; j<angles.size(); j++) {
// 	
// 				if (wave_params.iter==-1) break; 
// 				wave_params.iter++;
// 				cout<<"\r"<<(100.*wave_params.iter)/n_iter<<"\% -- lambda: "<<lambdas[i]<<", angle: "<<angles[j] << endl;
// 	
// 				// morlet generation (to be optimized)
// 				morlet_data morlet_params = {lambdas[i], angles[j] * _phys_deg, wave_params.thickness, wave_params.damp};
// 				
// 				phys_generate_Fmorlet(&morlet_params, xx, yy, zz_morlet);
// 				cout<<"morlet generated"<<endl;
// 				
// 				zz_morlet.fftshift();
// 				cout<<"fft shifted"<<endl;
// 				phys_convolve_Fm1_Fm2(Fmain_window, zz_morlet, zz_convolve);
// 				cout<<"convolved"<<endl;
// 	
// 				// decision
// 				for (size_t k=0; k<xx.getSurf(); k++) {
// 					double qmap_local=zz_convolve.Timg_buffer[k].mcabs()/surf;
// 					if ( qmap_local > qmap->Timg_buffer[k]) {
// 						qmap->Timg_buffer[k] = qmap_local;
// 						wphase->Timg_buffer[k] = zz_convolve.Timg_buffer[k].arg()/(2*M_PI);
// 						
// 						lambda->Timg_buffer[k] = lambdas[i];
// 						angle->Timg_buffer[k] = angles[j];
// 					}
// 				}
// 	
// 			}
// 		}
// #ifndef __winz
// 		cerr<<"Calculation end (n_iter "<<wave_params.iter<<"): "<<clock()-initTime<<endl;	
// #endif
// 
// 		olist->push_back(wphase);
// 		olist->push_back(qmap);
// 		olist->push_back(lambda);
// 		olist->push_back(angle);
// 
// 		list<nPhysD *>::const_iterator itr;
// 		for(itr = olist->begin(); itr != olist->end(); ++itr) {
// 			(*itr)->TscanBrightness();
// 			(*itr)->set_origin(ifg.get_origin());
// 			(*itr)->set_scale(ifg.get_scale());
// 			(*itr)->setFromName(ifg.getFromName());
// 			(*itr)->setShortName((*itr)->getName());
// 			(*itr)->setName((*itr)->getShortName()+ " "+ifg.getName());
// 		}
// 
// 		return olist;
// 	}
// 	cout<<"[phys_wave] ending wavelet analysis"<<endl;
// 	return olist;
// }

list<nPhysD *>
phys_wavelet_field_2D_morlet(nPhysD &ifg, wavelet_params &wave_params)
{
	DEBUG(5,"start");
	list<nPhysD *> olist;

	if ((wave_params.n_angles > 0) && (wave_params.n_lambdas > 0) && (ifg.getSurf() != 0)) {

		int dx=ifg.getW();
		int dy=ifg.getH();

		int xx[dx], yy[dy];
		
		nPhysImageF<mcomplex> zz_morlet("zz_morlet");
	
		zz_morlet.resize(dx,dy);
		
		for (int i=0;i<dx;i++) xx[i]=(i+(dx+1)/2)%dx-(dx+1)/2; // swap and center
		for (int i=0;i<dy;i++) yy[i]=(i+(dy+1)/2)%dy-(dy+1)/2;
	
		vector<double> angles(wave_params.n_angles), lambdas(wave_params.n_lambdas);
		int n_iter = angles.size()*lambdas.size();
		
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


		if (wave_params.n_angles==1) {
			angles[0]=0.5*(wave_params.end_angle+wave_params.init_angle);
		} else {
			for (size_t i=0; i<wave_params.n_angles; i++)
				angles[i] = wave_params.init_angle + i*(wave_params.end_angle-wave_params.init_angle)/(wave_params.n_angles-1);
		}	
		if (wave_params.n_lambdas==1) {
			lambdas[0]=0.5*(wave_params.end_lambda+wave_params.init_lambda);
		} else {
			for (size_t i=0; i<wave_params.n_lambdas; i++)
				lambdas[i] = wave_params.init_lambda + i*(wave_params.end_lambda-wave_params.init_lambda)/(wave_params.n_lambdas-1);
		}
		
		nPhysImageF<mcomplex> Fmain_window=ifg.ft2(PHYS_FORWARD);
		
		wave_params.iter=0;
		*wave_params.iter_ptr=0;
		
		double damp_norm=wave_params.damp*M_PI;
		for (size_t i=0; i<lambdas.size(); i++) {
			for (size_t j=0; j<angles.size(); j++) {
	
				if ((*wave_params.iter_ptr)==-1) {
				    DEBUG("Aborting");
				    break;
				}
				wave_params.iter++;
				(*wave_params.iter_ptr)++;
				DEBUG(11,(100.*wave_params.iter)/n_iter<<"\% lam "<<lambdas[i]<<", ang "<<angles[j]);
				double cr = cos(angles[j] * _phys_deg); 
				double sr = sin(angles[j] * _phys_deg);

				double thick_norm=wave_params.thickness*M_PI/sqrt(pow(sr*dx,2)+pow(cr*dy,2));
				double lambda_norm=lambdas[i]/sqrt(pow(cr*dx,2)+pow(sr*dy,2));
//				double thick_norm=wave_params.thickness*M_PI/dy;
//				double lambda_norm=lambdas[i]/dx;
				for (size_t x=0;x<zz_morlet.getW();x++) {
					for (size_t y=0;y<zz_morlet.getH();y++) {
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
				for (size_t k=0; k<ifg.getSurf(); k++) {
// 					double a0=angles[j];
// 					double a1=angle->Timg_buffer[k];
// 					double angdist=1.0;
//					if(i!=0||j!=0) angdist=std::min(abs(a0-a1),abs(a0-a1+360-a1))/360.0;
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
            if ((*wave_params.iter_ptr)==-1) {
                DEBUG("aborting");
                break;
            }
		}
		
		if ((*wave_params.iter_ptr)!=-1) {
		
			for (size_t k=0; k<ifg.getSurf(); k++) {
				qmap->Timg_buffer[k]=sqrt(qmap->Timg_buffer[k])/(dx*dy);
				intensity->Timg_buffer[k] = ifg.Timg_buffer[k] - 2.0*qmap->Timg_buffer[k]*cos(wphase->Timg_buffer[k]);
				wphase->Timg_buffer[k]/=2*M_PI;
			}
			
			phys_fast_gaussian_blur(*intensity,wave_params.thickness/2.0);
			
			for (size_t k=0; k<ifg.getSurf(); k++) {
				if(!std::isfinite(ifg.Timg_buffer[k])){
					qmap->Timg_buffer[k]   = numeric_limits<double>::quiet_NaN();
					wphase->Timg_buffer[k] = numeric_limits<double>::quiet_NaN();
					angle->Timg_buffer[k]  = numeric_limits<double>::quiet_NaN();
					lambda->Timg_buffer[k] = numeric_limits<double>::quiet_NaN();
					intensity->Timg_buffer[k] = numeric_limits<double>::quiet_NaN();
				}
			}
			
			olist.push_back(wphase);
			olist.push_back(qmap);
			olist.push_back(lambda);
			olist.push_back(angle);
			olist.push_back(intensity);
			
			list<nPhysD *>::const_iterator itr;
			for(itr = olist.begin(); itr != olist.end(); ++itr) {
				(*itr)->TscanBrightness();
				(*itr)->set_origin(ifg.get_origin());
				(*itr)->set_scale(ifg.get_scale());
				(*itr)->setFromName(ifg.getFromName());
				(*itr)->setShortName((*itr)->getName());
				(*itr)->setName((*itr)->getShortName()+ " "+ifg.getName());
			}
		}
		
	}
	DEBUG("Out of here");
	return olist;
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

#ifdef HAVE_CUDA
list<nPhysD *>
phys_wavelet_field_2D_morlet_cuda(nPhysD &ifg, wavelet_params &wave_params) {

    wave_params.iter=0;
    *wave_params.iter_ptr = 0;

	list<nPhysD *> olist;

	DEBUG(5,"start");

	int device_count = 0;
	cudaGetDeviceCount( &device_count );
	if (device_count<1) {
		WARNING("Problem To use CUDA you need an NVIDIA card. You also need to install the driver\nwww.nvidia.com/page/drivers.html");
		return olist;
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

	int cubuf_size = sizeof(cufftComplex)*ifg.getSurf();
	
	cufftHandle plan;
	cufftComplex *b1,*b2;
	cufftComplex *cub1, *cub2, *cub3, *cuc1, *cuc2;
	
	b1 = new cufftComplex [ifg.getSurf()];
	b2 = new cufftComplex [ifg.getSurf()];
	
	for (size_t j = 0; j < ifg.getH(); j++){
		for (size_t i = 0; i < ifg.getW(); i++) {
			b1[i+j*ifg.getW()].x = std::isfinite(ifg.Timg_matrix[j][i]) ? ifg.Timg_matrix[j][i]:0;
			b1[i+j*ifg.getW()].y=0.0;
		}
	}
	
	cudaMalloc((void**)&cub1, cubuf_size);
	if (cudaGetLastError()!=cudaSuccess) {
		WARNING("cannot allocate");
		return olist;
	}
	cudaMalloc((void**)&cub2, cubuf_size);
	if (cudaGetLastError()!=cudaSuccess) {
		WARNING("cannot allocate");
		return olist;
	}
	cudaMalloc((void**)&cub3, cubuf_size);
	if (cudaGetLastError()!=cudaSuccess) {
		WARNING("cannot allocate");
		return olist;
	}
	cudaMalloc((void**)&cuc1, cubuf_size);
	if (cudaGetLastError()!=cudaSuccess) {
		WARNING("cannot allocate");
		return olist;
	}
	cudaMalloc((void**)&cuc2, cubuf_size);
	if (cudaGetLastError()!=cudaSuccess) {
		WARNING("cannot allocate");
		return olist;
	}
	
	cudaMemset(cub1, 0, cubuf_size );
	cudaMemset(cub2, 0, cubuf_size );
	cudaMemset(cub3, 0, cubuf_size );
	cudaMemset(cuc1, 0, cubuf_size );
	cudaMemset(cuc2, 0, cubuf_size );
	
	// Create a 2D FFT plan.  
	cufftPlan2d(&plan, ifg.getH(), ifg.getW(), CUFFT_C2C);
	if (cudaGetLastError()!=cudaSuccess) {
		WARNING("cannot create plan");
		return olist;
	}

	cudaMemcpy(cub1, b1, cubuf_size, cudaMemcpyHostToDevice);

	cufftExecC2C(plan, cub1, cub2, CUFFT_FORWARD);
	if (cudaGetLastError()!=cudaSuccess) {
		WARNING("cannot do FFT");
		return olist;
	}
//	cudaThreadSynchronize();
	
	if ((wave_params.n_angles > 0) && (wave_params.n_lambdas > 0) && (ifg.getSurf() != 0)) {

		int dx=ifg.getW();
		int dy=ifg.getH();

		vector<double> angles(wave_params.n_angles), lambdas(wave_params.n_lambdas);
		int n_iter = angles.size()*lambdas.size();
		
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


		if (wave_params.n_angles==1) {
			angles[0]=0.5*(wave_params.end_angle+wave_params.init_angle);
		} else {
			for (size_t i=0; i<wave_params.n_angles; i++)
				angles[i] = wave_params.init_angle + i*(wave_params.end_angle-wave_params.init_angle)/(wave_params.n_angles-1);
		}	
		if (wave_params.n_lambdas==1) {
			lambdas[0]=0.5*(wave_params.end_lambda+wave_params.init_lambda);
		} else {
			for (size_t i=0; i<wave_params.n_lambdas; i++)
				lambdas[i] = wave_params.init_lambda + i*(wave_params.end_lambda-wave_params.init_lambda)/(wave_params.n_lambdas-1);
		}
		
		for (size_t i=0; i<lambdas.size(); i++) {
			for (size_t j=0; j<angles.size(); j++) {
	
				if ((*wave_params.iter_ptr)==-1) {
				    DEBUG("aborting");
				    break;
				}
				wave_params.iter++;
				(*wave_params.iter_ptr)++;
				DEBUG((100.*wave_params.iter)/n_iter<<"\% lam "<<lambdas[i]<<", ang "<<angles[j]);
			
				gabor(cub2, cub1, dx, dy, angles[j]/180.*M_PI, lambdas[i], (float)wave_params.damp, (float)wave_params.thickness);
				cufftExecC2C(plan, cub1, cub3, CUFFT_INVERSE);
				fase(cub3, cuc1, cuc2, ifg.getSurf(), angles[j], lambdas[i]);
				cudaThreadSynchronize();
			
			}
			//! todo: this is awful: add exception?
            if ((*wave_params.iter_ptr)==-1) {
                DEBUG("aborting");
                break;
            }
		}
		DEBUG("*wave_params.iter_ptr" <<  *wave_params.iter_ptr);
		cudaThreadSynchronize();

		if ((*wave_params.iter_ptr)!=-1) {
			cudaMemcpy(b1, cuc1, cubuf_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(b2, cuc2, cubuf_size, cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();
			
			for (size_t k=0; k<ifg.getSurf(); k++) {
				if(std::isfinite(ifg.Timg_buffer[k])){
					qmap->Timg_buffer[k] = sqrt(b1[k].x)/(dx*dy);
					wphase->Timg_buffer[k] = b1[k].y/(2*M_PI);
					angle->Timg_buffer[k] = b2[k].x;
					lambda->Timg_buffer[k] = b2[k].y;
					intensity->Timg_buffer[k] = ifg.Timg_buffer[k] - 2.0*qmap->Timg_buffer[k]*cos(b1[k].y);
				} else {
					qmap->Timg_buffer[k]   = numeric_limits<double>::quiet_NaN();
					wphase->Timg_buffer[k] = numeric_limits<double>::quiet_NaN();
					angle->Timg_buffer[k]  = numeric_limits<double>::quiet_NaN();
					lambda->Timg_buffer[k] = numeric_limits<double>::quiet_NaN();
					intensity->Timg_buffer[k] = numeric_limits<double>::quiet_NaN();
				}
				
			}
			
			
			phys_fast_gaussian_blur(*intensity,wave_params.thickness/2.0);
			
			olist.push_back(wphase);
			olist.push_back(qmap);
			olist.push_back(lambda);
			olist.push_back(angle);
			olist.push_back(intensity);
			
			list<nPhysD *>::const_iterator itr;
			for(itr = olist.begin(); itr != olist.end(); ++itr) {
				(*itr)->TscanBrightness();
				(*itr)->set_origin(ifg.get_origin());
				(*itr)->set_scale(ifg.get_scale());
				(*itr)->setFromName(ifg.getFromName());
				(*itr)->setShortName((*itr)->getName());
				(*itr)->setName((*itr)->getShortName()+ " "+ifg.getName());
			}
		}
		
		delete [] b1;
		delete [] b2;
	
		cudaFree(cub1);
		cudaFree(cub2);
		cudaFree(cub3);
		cudaFree(cuc1);
		cudaFree(cuc2);
	
		cufftDestroy (plan);
	    cudaThreadSynchronize();
		cudaThreadExit();

	}
	DEBUG("Out of here");
	return olist;	
}
#else
list<nPhysD *>
phys_wavelet_field_2D_morlet_cuda(nPhysD &ifg, wavelet_params &wave_params) {
    WARNING("We should never go here...");
}
#endif

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
            DEBUG("here");
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

void phys_synthetic_interferogram (nPhysImageF<double> &synthetic, nPhysImageF<double> &phase_over_2pi, nPhysImageF<double> &quality){
    
    if (phase_over_2pi.getW()==quality.getW() && phase_over_2pi.getH()==quality.getH()) {
        synthetic.resize(phase_over_2pi.getW(),phase_over_2pi.getH());
        for (size_t ii=0; ii<phase_over_2pi.getSurf(); ii++) {
            synthetic.set(ii,quality.point(ii)*(1.0+cos(phase_over_2pi.point(ii)*2*M_PI)));
        }
        synthetic.setShortName("synthetic");
        synthetic.setName("synthetic("+phase_over_2pi.getName()+","+quality.getName()+")");
        synthetic.TscanBrightness();
    }

}

void
phys_subtract_carrier(nPhysD &iphys, double kx, double ky)
{

	for (register size_t ii=0; ii<iphys.getW(); ii++) {
		for (register size_t jj=0; jj<iphys.getH(); jj++) {
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
	
	fftw_complex *myData=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*dy*(dx/2+1));
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
	for (register size_t ii=0; ii<invimage.getSurf(); ii++) {
		double the_point = invimage.point(ii)/res;
		//invimage.set(ii, - mult * (the_point*the_point+2*kappa*the_point));
		invimage.set(ii, mult*(pow(the_point/kappa + 1,2.)-1) );
	}
	invimage.TscanBrightness();
}

void
phys_apply_inversion_plasma(nPhysD &invimage, double probe_wl, double res)
{
	double kappa = 2*M_PI/probe_wl;
	double mult = _phys_emass*_phys_vacuum_eps*_phys_cspeed*_phys_cspeed/(_phys_echarge*_phys_echarge);
	DEBUG(5,"resolution: "<< res << ", probe: " << probe_wl << ", mult: " << mult);
	for (register size_t ii=0; ii<invimage.getSurf(); ii++) {
		double the_point = invimage.point(ii)/res;
		invimage.set(ii, - mult * (the_point*the_point+2*kappa*the_point));
	}
	invimage.TscanBrightness();
}


//! General function for Abel inversion
nPhysD * phys_invert_abel(nPhysD &iimage, abel_params &params)
{
	
	std::vector<phys_point> iaxis = params.iaxis; // TODO: passa a bidimvec
	phys_direction idir = params.idir;
	inversion_algo ialgo = params.ialgo;
//	inversion_physics iphysics = params.iphysics;


	if (iimage.getSurf() == 0)
		return NULL;

	nPhysD *oimage;
	oimage = new nPhysD (iimage.getW(), iimage.getH(),numeric_limits<double>::quiet_NaN(),"Inverted");
	oimage->set_origin(iimage.get_origin());
	oimage->set_scale(iimage.get_scale());

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
// 	size_t longitudinal_size = iimage.getSizeByIndex(sym_idx);
	size_t integral_size = iimage.getSizeByIndex(inv_idx);
	double *copy_buffer, *out_buffer;
	copy_buffer = new double[integral_size];
	out_buffer = new double[integral_size];

	// .alex. old version
	/*switch (idir) {
		case PHYS_Y:
			oimage->set(iaxis[ii].x,iaxis[ii].y, 
					0.5*(oimage->point(iaxis[ii].x-1,iaxis[ii].y)+oimage->point(iaxis[ii].x+1,iaxis[ii].y)));
			break;
		
		case PHYS_X:
		default:
			oimage->set(iaxis[ii].x,iaxis[ii].y, 
					0.5*(oimage->point(iaxis[ii].x,iaxis[ii].y-1)+oimage->point(iaxis[ii].x,iaxis[ii].y+1)));
			break;
	}*/

	*params.iter_ptr = 0;
	
	if (ialgo == ABEL) {
		DEBUG(1, "Plain ABEL inversion");
		for (register size_t ii = 0; ii<iaxis.size(); ii++) {
			if ((*params.iter_ptr)==-1) {
				    DEBUG("aborting");
				    break;
			}
			(*params.iter_ptr)++;
			
			axe_point[0] = iaxis[ii].x;
			axe_point[1] = iaxis[ii].y;
			//cerr << axe_point[0]  << " , " << axe_point[1] << endl;
			int copied = iimage.get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], copy_buffer, integral_size, PHYS_NEG);
			
			for (size_t j=copied; j<integral_size; j++)
				copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization
	
			
			phys_invert_abel_1D(copy_buffer, out_buffer, integral_size);
			
			oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], out_buffer, integral_size, PHYS_NEG);
			
			

			copied = iimage.get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], copy_buffer, integral_size, PHYS_POS);
			for (size_t j=copied; j<integral_size; j++)
				copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization

			phys_invert_abel_1D(copy_buffer, out_buffer, integral_size);
			oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], out_buffer, integral_size, PHYS_POS);

			//FIXME: pretty sure there is a better way!! ALEX!!!!
			// .alex. : fixed in some way. To be checked
			oimage->set(iaxis[ii].x, iaxis[ii].y, 
					0.5*(oimage->point(iaxis[ii].x-(idir),iaxis[ii].y+(idir-1))+oimage->point(iaxis[ii].x+idir,iaxis[ii].y+(1-idir))));

		}
		oimage->setName(string("ABEL ")+oimage->getName());
		oimage->setShortName("ABEL");

		//oimage->setName(string("Inverted (ABEL) ")+string(iimage.getShortName()));
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
			oimage->resize(iimage.getW(), 3*integral_size);
		} else {
			oimage->resize(3*integral_size, iimage.getH());
		}*/

		int axe_inv_mean[2];
		axe_inv_mean[0] = 0;
		axe_inv_mean[1] = 0;
		for (register size_t ii=0; ii<iaxis.size(); ii++) {
			axe_inv_mean[0] += iaxis[ii].x;
			axe_inv_mean[1] += iaxis[ii].y;
		}
		int axe_average = (double)axe_inv_mean[inv_idx]/iaxis.size();
		DEBUG(5, "Axe average: "<<axe_average);

		for (register size_t ii = 0; ii<iaxis.size(); ii++) {
			if ((*params.iter_ptr)==-1) {
				DEBUG("aborting");
				break;
			}
			(*params.iter_ptr)++;

			axe_point[0] = iaxis[ii].x;
			axe_point[1] = iaxis[ii].y;
			//cerr << axe_point[0]  << " , " << axe_point[1] << endl;
			int copied = iimage.get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], copy_buffer, integral_size, PHYS_NEG);
			
			for (size_t j=copied; j<integral_size; j++)
				copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization
	
			
			phys_invert_abelHF_1D(copy_buffer, out_buffer, integral_size, &my_lut);
			//phys_invert_abelHF_1D(copy_buffer, out_buffer, copied, &my_lut);
			
			oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], out_buffer, integral_size, PHYS_NEG);
			//oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx]-axe_average+1.5*integral_size, out_buffer, integral_size, PHYS_NEG);
		
			double upper_axe_point = out_buffer[0];
			

			copied = iimage.get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], copy_buffer, integral_size, PHYS_POS);
			for (size_t j=copied; j<integral_size; j++)
				copy_buffer[j] = copy_buffer[copied-1];	// boundary normalization

			phys_invert_abelHF_1D(copy_buffer, out_buffer, integral_size, &my_lut);
			//phys_invert_abelHF_1D(copy_buffer, out_buffer, copied, &my_lut);
			
			oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], out_buffer, integral_size, PHYS_POS);
			//oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx]-axe_average+1.5*integral_size, out_buffer, integral_size, PHYS_POS);

			//FIXME: pretty sure there is a better way!! ALEX!!!!
			// .alex. : fixed in some way. To be checked
			
			// me lo ricordo mica cosa fa questa riga..
			//oimage->set(iaxis[ii].x, iaxis[ii].y, 
			//		0.5*(oimage->point(iaxis[ii].x-(idir),iaxis[ii].y+(idir-1))+oimage->point(iaxis[ii].x+idir,iaxis[ii].y+(1-idir))));

			oimage->set(iaxis[ii].x, iaxis[ii].y, 0.5*out_buffer[0]+0.5*upper_axe_point); 

			DEBUG(10,"step: "<<ii);
		}
		oimage->setName(string("ABEL")+oimage->getName());
		oimage->setShortName("ABEL");

	} else {
		DEBUG(1, "Unknown inversion type: "<<(int)ialgo);
	}

	delete copy_buffer;
	delete out_buffer;

	oimage->TscanBrightness();

    DEBUG((*params.iter_ptr)); 
	return oimage;
}



/*!
 * @}
 */
