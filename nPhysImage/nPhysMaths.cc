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
#include "nPhysMaths.h"
#include <time.h>

using namespace std;

/*! \addtogroup nPhysMaths
 * @{
 */ 

inline void planeFit(nPhysImageF<double> *pi, double *coeffs)
{
	int fit_n = pi->getW()*pi->getH();	// grossino...
	int fit_p = 2;				// plane fit

	double chi_sq;
	gsl_matrix *X = gsl_matrix_alloc(fit_n, fit_p);
	gsl_matrix *cov = gsl_matrix_alloc(fit_p, fit_p);
	gsl_vector *y = gsl_vector_alloc(fit_n);
	gsl_vector *c = gsl_vector_alloc(fit_p);

	// data translation and copy
	for (size_t i=0; i<pi->getH(); i++) {
		for (size_t j=0; j<pi->getW(); j++) {
			(y->data)[i*pi->getW()+j] = pi->Timg_buffer[i*pi->getW()+j] - pi->Timg_buffer[0];
			X->data[(i*pi->getW()+j)*X->tda + 0] = j;		// row=datapoint index, cols 1/2
			X->data[(i*pi->getW()+j)*X->tda + 1] = i;
		}
	}

	// calc
	gsl_multifit_linear_workspace *work =  gsl_multifit_linear_alloc (fit_n, fit_p);
	gsl_multifit_linear (X, y, c, cov, &chi_sq, work);

	// e supponendo che abbia davvero fatto qualcosa.. - MIND THE INVERSION!! -
	if (coeffs != NULL) {
		coeffs[0] = c->data[0];
		coeffs[1] = c->data[1];
	}

	// free all
	gsl_matrix_free(X);
	gsl_vector_free(y);
	gsl_vector_free(c);
	gsl_multifit_linear_free(work);

}

// ------------------ general purpose functions for wavelet analysis ------------------------
inline mcomplex 
morlet(double lambda, double tau, double x)
{
	double kappa = 2.*3.141592/lambda;
	return mcomplex(exp(-pow((x/tau),2.))*(cos(kappa*x)),- exp(-pow((x/tau),2.))*(sin(kappa*x)));
}



inline mcomplex 
morlet(double lambda, double tau_l, double tau_v, double x, double y, double rotation)
{
	double kappa = 2.*3.141592/lambda;
	double cr = cos(rotation); double sr = sin(rotation);
	double rx = cr*x - sr*y; double ry = sr*x + cr*y;
	double mult = exp(-pow((ry/tau_v),2.))*exp(-pow((rx/tau_l),2.));
	return mcomplex(mult*cos(kappa*rx), -mult*sin(kappa*rx));
}


inline void
phys_generate_meshgrid(int x1, int x2, int y1, int y2, nPhysImageF<int> &xx, nPhysImageF<int> &yy)
{
	int xsize = x2-x1+1;
	int ysize = y2-y1+1;

	xx.resize(xsize, ysize);
	yy.resize(xsize, ysize);	

	for (int i=0; i<xsize; i++) {
		for (int j=0; j<ysize; j++) {
			xx.set(i, j, x1+i);
			yy.set(i, j, y1+j);
		}
	}

}



// via function pointers (via meshgrids)

inline void 
phys_generate_morlet(morlet_data *md, nPhysImageF<double> &xx, nPhysImageF<double> &yy, nPhysImageF<mcomplex> &zz)
{

	if ((xx.getW() != yy.getW()) || (xx.getH() != yy.getH())) {
		WARNING("size mismatch: op1 is: "<<xx.getW()<<"x"<<xx.getH()<<", op2 is: "<<yy.getW()<<"x"<<yy.getH());
		return;
	}

//	double alpha = 2.*3.141592/(md->lambda*md->sigma);
	double kappa = 2.*3.141592/md->lambda;

	double cr = cos(md->angle); double sr = sin(md->angle);
//	double cs = (1/sqrt(1+exp(-pow(md->sigma, 2.))-2*exp(-3*pow(md->sigma, 2.)/4))) * pow(3.1415, -0.25) * alpha;
	zz.resize(xx.getW(), xx.getH());

	for (size_t i=0; i<zz.getW(); i++) {
		for (size_t j=0; j<zz.getH(); j++) {
			double x = xx.Timg_matrix[j][i];
			double y = yy.Timg_matrix[j][i];
			double rx = (cr*x - sr*y); double ry = (sr*x + cr*y);
			double mult = exp(-pow((ry/md->thickness),2.))*exp(-pow((rx/(md->damp*md->lambda)),2.));
			//double mult = cs*exp(-pow((ry/md->tau_dump),2.))*exp(-pow(rx,2.)/2);
			zz.Timg_matrix[j][i] = mcomplex(mult*cos(kappa*rx), -mult*sin(kappa*rx));
//			zz.Timg_matrix[j][i] = mcomplex(mult*cos(md->sigma*rx), -mult*sin(md->sigma*rx));

		}
	}
}


inline void phys_reverse_vector(double *buf, int size)
{
	int inv = size/2; // integer division
	double reg;
	if (size%2 == 0) {
		for (int i=0; i<inv; i++) {
			reg = buf[inv-i-1];
			buf[inv-i-1] = buf[inv+i];
			buf[inv+i] = reg;
		}
	} else {
		for (int i=0; i<inv; i++) {
			reg = buf[inv-i-1];
			buf[inv-i-1] = buf[inv+i+1];
			buf[inv+i+1] = reg;
		}
	}

}

// some nice filters


template<> void
nPhysImageF<mcomplex>::TscanBrightness() {
	return;
}

void 
phys_add(nPhysImageF<double> &iimage, double val) { 
	if (val!=0.0) {
		for (register size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)+val); 
		iimage.TscanBrightness();
	}
	std::ostringstream ostr;
	ostr << val;
	iimage.setName("("+iimage.getName()+")+"+ostr.str());
}

void 
phys_subtract(nPhysImageF<double> &iimage, double val) {
	if (val!=0.0) {
		for (register size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)-val); 
		iimage.TscanBrightness();
	}
	std::ostringstream ostr;
	ostr << val;
	iimage.setName("("+iimage.getName()+")-"+ostr.str());
}

void 
phys_multiply(nPhysImageF<double> &iimage, double val) {
	if (val!=1.0) {
		for (register size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)*val); 
		iimage.TscanBrightness();
	}
	std::ostringstream ostr;
	ostr << val;
	iimage.setName("("+iimage.getName()+")*"+ostr.str());
}

void 
phys_divide(nPhysImageF<double> &iimage, double val) {
	if (val!=1.0) {
		for (register size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)/val); 
		iimage.TscanBrightness();
	}
	std::ostringstream ostr;
	ostr << val;
	iimage.setName("("+iimage.getName()+")/"+ostr.str());

}

void 
phys_point_add(nPhysImageF<double> &iimage, nPhysImageF<double> &iimage2) {
	if (iimage.getSurf()==iimage2.getSurf()) {
		for (register size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)+iimage2.point(ii)); 
		iimage.TscanBrightness();
		iimage.setName("("+iimage.getName()+")+("+iimage2.getName()+")");
	}
}

void 
phys_point_subtract(nPhysImageF<double> &iimage, nPhysImageF<double> &iimage2) {
	if (iimage.getSurf()==iimage2.getSurf()) {
		for (register size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)-iimage2.point(ii)); 
		iimage.TscanBrightness();
		iimage.setName("("+iimage.getName()+")-("+iimage2.getName()+")");
	}
}

void 
phys_point_multiply(nPhysImageF<double> &iimage, nPhysImageF<double> &iimage2) {
	if (iimage.getSurf()==iimage2.getSurf()) {
		for (register size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)*iimage2.point(ii)); 
		iimage.TscanBrightness();
		iimage.setName("("+iimage.getName()+")*("+iimage2.getName()+")");
	}
}

void 
phys_point_divide(nPhysImageF<double> &iimage, nPhysImageF<double> &iimage2) {
	if (iimage.getSurf()==iimage2.getSurf()) {
		for (register size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)/iimage2.point(ii)); 
		iimage.TscanBrightness();
		iimage.setName("("+iimage.getName()+")/("+iimage2.getName()+")");
	}
}

double phys_sum_points(nPhysImageF<double> &iimage) {
	double retVal=0.0;
	for (register size_t ii=0; ii<iimage.getSurf(); ii++) {
		retVal+=iimage.point(ii);
	}
	return retVal;
}

double phys_sum_square_points(nPhysImageF<double> &iimage) {
	double retVal=0.0;
	for (register size_t ii=0; ii<iimage.getSurf(); ii++) {
		retVal+=iimage.point(ii)*iimage.point(ii);
	}
	return retVal;
}

void phys_opposite(nPhysImageF<double> &iimage) {
	for (register size_t ii=0; ii<iimage.getSurf(); ii++) {
		iimage.set(ii,-iimage.point(ii));
	}
	iimage.TscanBrightness();
	iimage.setName("-("+iimage.getName()+")");
}


void phys_inverse(nPhysImageF<double> &iimage) {
	for (register size_t ii=0; ii<iimage.getSurf(); ii++) {
		iimage.set(ii,1.0/iimage.point(ii));
	}
	iimage.TscanBrightness();
	iimage.setName("1/("+iimage.getName()+")");
}

void phys_replace(nPhysImageF<double> &iimage, double oldval, double newval) {
	for (register size_t ii=0; ii<iimage.getSurf(); ii++) {
		if (iimage.point(ii)==oldval) iimage.set(ii,newval);
	}
	iimage.TscanBrightness();
	std::ostringstream ostr;
	ostr << "replace(" << iimage.getName() << "," << oldval << "," << newval << ")";
	iimage.setName(ostr.str());
}

void phys_replace_NaN(nPhysImageF<double> &iimage, double newval) {
	for (register size_t ii=0; ii<iimage.getSurf(); ii++) {
		if (!isfinite(iimage.point(ii))) iimage.set(ii,newval);
	}
	iimage.TscanBrightness();
	
	std::ostringstream ostr;
	ostr << "replaceNaN(" << iimage.getName() << "," << newval << ")";
	iimage.setName(ostr.str());
}

void phys_cutoff(nPhysImageF<double> &iimage, double minval, double maxval) {
    iimage.setShortName("IntensityCutoff");
	for (register size_t ii=0; ii<iimage.getSurf(); ii++) {
        if (isfinite(iimage.point(ii))) iimage.set(ii,std::min(std::max(iimage.point(ii),minval),maxval));
    }
	iimage.TscanBrightness();
	std::ostringstream ostr;
	ostr << "min_max(" << iimage.getName() << "," << minval << "," << maxval << ")";
	iimage.setName(ostr.str());
}


void 
phys_add_noise(nPhysImageF<double> &iimage, double vMax=1.0)
{
	for (register size_t ii=0; ii<iimage.getSurf(); ii++) {
		iimage.set(ii, iimage.point(ii) + vMax*((double)rand() / RAND_MAX));
	}
	iimage.TscanBrightness();
	std::ostringstream ostr;
	ostr << vMax;
	iimage.setName("("+iimage.getName()+")+rand("+ostr.str()+")");
}


// gaussian blur
void
phys_gaussian_blur(nPhysImageF<double> &m1, double radius)
{
	DEBUG(5,"radius: "<<radius);
	if (!(radius > 0) )
		return;

	nPhysImageF<double> xx, yy, gauss;
	nPhysImageF<mcomplex> out;

//FIXME: this is probably wrong for odd matrices
	meshgrid_data md = {-m1.getW()/2., m1.getW()/2, -m1.getH()/2., m1.getH()/2., m1.getW(), m1.getH()};
	phys_generate_meshgrid(&md, xx, yy);

	gauss.resize(xx.getW(), xx.getH());
	double mult = 1/(pow(radius, 2.)*2*M_PI);
	for (size_t i=0; i<xx.getW(); i++) {
		for (size_t j=0; j<xx.getH(); j++) {
			gauss.Timg_matrix[j][i] = mult*exp( -(pow(xx.Timg_matrix[j][i],2)+pow(yy.Timg_matrix[j][i],2))/(2.*pow(radius, 2)) );
		}
	}

	phys_convolve(m1, gauss, out);
	out.fftshift();

	for (size_t i=0; i<xx.getW(); i++) {
		for (size_t j=0; j<xx.getH(); j++) {
			m1.Timg_matrix[j][i] = (out.Timg_matrix[j][i].real())/double(xx.getSurf());
		}
	}

	m1.TscanBrightness();
	std::ostringstream ostr;
	ostr << radius;
	m1.setName("blur("+m1.getName()+","+ostr.str()+")");
}

void
phys_sin(nPhysImageF<double> &m1)
{
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,sin(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("sin("+m1.getName()+")");
}

void
phys_cos(nPhysImageF<double> &m1)
{
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,cos(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("cos("+m1.getName()+")");
}

void
phys_tan(nPhysImageF<double> &m1)
{
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,tan(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("tan("+m1.getName()+")");
}

void
phys_pow(nPhysImageF<double> &m1, double exponent)
{
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,pow(m1.point(i),exponent));
	}
	m1.TscanBrightness();
	std::ostringstream ostr;
	ostr << exponent;
	m1.setName("("+m1.getName()+")^"+ostr.str());
}

void
phys_square(nPhysImageF<double> &m1)
{
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,pow(m1.point(i),2));
	}
	m1.TscanBrightness();
	m1.setName("("+m1.getName()+")^2");
}

void
phys_sqrt(nPhysImageF<double> &m1)
{
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,sqrt(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("sqrt("+m1.getName()+")");
}

void
phys_abs(nPhysImageF<double> &m1)
{
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,abs(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("abs("+m1.getName()+")");
}

void
phys_log(nPhysImageF<double> &m1)
{
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,log(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("ln("+m1.getName()+")");
}

void
phys_log10(nPhysImageF<double> &m1)
{
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,log10(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("log("+m1.getName()+")");
}

void
phys_fast_gaussian_blur(nPhysImageF<double> &m1, double radius)
{
	vector<double> nan_free_phys(m1.getSurf());
	for (size_t i=0; i< m1.getSurf(); i++) {
		if (std::isfinite(m1.point(i))) {
			nan_free_phys[i]=m1.point(i);
		} else {
			nan_free_phys[i]=m1.Tminimum_value;
		}
	}
	fftw_complex *b2 = fftw_alloc_complex(m1.getH()*(m1.getW()/2+1));
	
	fftw_plan fb = fftw_plan_dft_r2c_2d(m1.getH(),m1.getW(),&nan_free_phys[0],b2,FFTW_ESTIMATE);
	fftw_plan bb = fftw_plan_dft_c2r_2d(m1.getH(),m1.getW(),b2,&nan_free_phys[0],FFTW_ESTIMATE);
	
	fftw_execute(fb);

 	double sx=pow(m1.getW()/(radius),2)/2.0;
 	double sy=pow(m1.getH()/(radius),2)/2.0;

 	for (size_t j = 0 ; j < m1.getH(); j++) {
		for (size_t i = 0 ; i < m1.getW()/2+1 ; i++) {
			double blur=exp(-((i*i)/sx+(j-m1.getH()/2)*(j-m1.getH()/2)/sy))/m1.getSurf();
			int k=i+((j+m1.getH()/2+1)%m1.getH())*(m1.getW()/2+1);
			b2[k][0]*=blur;
			b2[k][1]*=blur;
		}
	}
	fftw_execute(bb);
	for (size_t i=0; i< m1.getSurf(); i++) {
		if (std::isfinite(m1.point(i))) {
			m1.set(i,nan_free_phys[i]);
		}
	}
	m1.TscanBrightness();
	fftw_destroy_plan(fb);
	fftw_destroy_plan(bb);
	fftw_free(b2);
}

void
phys_gaussian_subtraction(nPhysImageF<double> &m1, double radius1, double radius2)
{
	vector<double> nan_free_phys(m1.getSurf());
	for (size_t i=0; i< m1.getSurf(); i++) {
		if (std::isfinite(m1.point(i))) {
			nan_free_phys[i]=m1.point(i);
		} else {
			nan_free_phys[i]=m1.Tminimum_value;
		}
	}
	fftw_complex *b2 = fftw_alloc_complex(m1.getH()*(m1.getW()/2+1));
	
	fftw_plan fb = fftw_plan_dft_r2c_2d(m1.getH(),m1.getW(),&nan_free_phys[0],b2,FFTW_ESTIMATE);
	fftw_plan bb = fftw_plan_dft_c2r_2d(m1.getH(),m1.getW(),b2,&nan_free_phys[0],FFTW_ESTIMATE);
	
	fftw_execute(fb);

 	double sx1=pow(m1.getW()/(radius1*M_PI),2)/2.0;
 	double sy1=pow(m1.getH()/(radius1*M_PI),2)/2.0;
 	double sx2=pow(m1.getW()/(radius2*M_PI),2)/2.0;
 	double sy2=pow(m1.getH()/(radius2*M_PI),2)/2.0;

 	for (size_t j = 0 ; j < m1.getH(); j++) {
		for (size_t i = 0 ; i < m1.getW()/2+1 ; i++) {
			double blur1=exp(-((i*i)/sx1+(j-m1.getH()/2)*(j-m1.getH()/2)/sy1))/m1.getSurf();
			double blur2=exp(-((i*i)/sx2+(j-m1.getH()/2)*(j-m1.getH()/2)/sy2))/m1.getSurf();
			int k=i+((j+m1.getH()/2+1)%m1.getH())*(m1.getW()/2+1);
			b2[k][0]*=blur1-blur2;
			b2[k][1]*=blur1-blur2;
		}
	}
	fftw_execute(bb);	
	for (size_t i=0; i< m1.getSurf(); i++) {
		if (std::isfinite(m1.point(i))) {
			m1.set(i,nan_free_phys[i]);
		}
	}
	m1.TscanBrightness();
	fftw_destroy_plan(fb);
	fftw_destroy_plan(bb);
	fftw_free(b2);
}

pair<double, bidimvec<int> > phys_cross_correlate(nPhysImageF<double>* img1, nPhysImageF<double>* img2) {
	size_t dx=img1->getW();
	size_t dy=img1->getH();
	double maxValue=-1.0;	
	bidimvec<int> maxP(0,0);

	if (dx == img2->getW() && dy== img2->getH()) {
		nPhysD *rPhys=new nPhysD(dx,dy,0.0,"Result");
		
		fftw_complex *myData1C=fftw_alloc_complex(dy*(dx/2+1));
		fftw_complex *myData2C=fftw_alloc_complex(dy*(dx/2+1));
		
		fftw_plan plan1R2C=fftw_plan_dft_r2c_2d(dy,dx, img1->Timg_buffer, myData1C, FFTW_ESTIMATE);
		fftw_plan plan2R2C=fftw_plan_dft_r2c_2d(dy,dx, img2->Timg_buffer, myData2C, FFTW_ESTIMATE);
		
		fftw_plan planC2R=fftw_plan_dft_c2r_2d(dy,dx, myData1C, rPhys->Timg_buffer, FFTW_ESTIMATE);
		
		fftw_execute(plan1R2C);
		fftw_execute(plan2R2C);
		
		for (size_t i=0;i<dy*(dx/2+1);i++) {
			double a[2]={myData1C[i][0],myData1C[i][1]};
			
			myData1C[i][0]=(a[0]*myData2C[i][0] + a[1]*myData2C[i][1])/img1->getSurf();
			myData1C[i][1]=(a[1]*myData2C[i][0] - a[0]*myData2C[i][1])/img1->getSurf();
		}
		fftw_execute(planC2R);
		
		for (size_t i=0;i<dx;i++) {
			for (size_t j=0;j<dy;j++) {
				if (rPhys->point(i,j) > rPhys->point(maxP.x(),maxP.y())) {
					maxP=vec2f(i,j);
				}
			}
		}
		maxValue=rPhys->point(maxP.x(),maxP.y());
		bidimvec<int> shift(dx/2+1,dy/2+1);
		maxP+=shift;
		maxP=vec2f(maxP.x()%dx,maxP.y()%dy)-shift;

		DEBUG(5,"max corr " << maxP.x() << " " << maxP.y() << " " << maxValue);
		
		delete rPhys;
		
		fftw_free(myData1C);
		fftw_free(myData2C);
		fftw_destroy_plan(plan1R2C);
		fftw_destroy_plan(plan2R2C);
		fftw_destroy_plan(planC2R);
	}
	
	return make_pair(maxValue, maxP);

}
void phys_get_vec_brightness(const double *ivec, size_t vlen, double &vmin, double &vmax)
{
	if (ivec==NULL || vlen<1) {
		vmin = 0;
		vmax = 0;
		return;
	}

	vmin = std::isfinite(ivec[0]) ? ivec[0]:0;
	vmax = std::isfinite(ivec[0]) ? ivec[0]:0;

	for (register size_t ii=0; ii<vlen; ii++) {
		if (std::isfinite(ivec[ii])) {
			if (ivec[ii] < vmin)
				vmin = ivec[ii];
			else if (ivec[ii] > vmax)
				vmax = ivec[ii];
		}
	}
}	

bidimvec<size_t> phys_max_p(nPhysImageF<double> &image) {
	bidimvec<size_t> p(0,0);
	for (size_t i=0;i<image.getW();i++) {
		for (size_t j=0;j<image.getH();j++) {
			if (image.point(i,j) > image.point(p.x(),p.y())) {
				p= bidimvec<size_t>(i,j);
			}
		}
	}
	return p;
	
}


// complex functions

//! split mcomplex matrix on polar representation
std::map<std::string, nPhysImageF<double> > to_polar(nPhysImageF<mcomplex> &iphys) {
	nPhysImageF<double> rho, theta;
	std::map<std::string, nPhysImageF<double> > omap;

	rho.resize(iphys.getW(), iphys.getH());
	theta.resize(iphys.getW(), iphys.getH());

	for (register size_t ii=0; ii<iphys.getSurf(); ii++) {
		mcomplex pt = iphys.point(ii);
		rho.set(ii, pt.mod() );
		theta.set(ii, pt.arg() );
	}

	omap[std::string("rho")] = rho;
	omap[std::string("theta")] = theta;
	return omap;
}

//! split mcomplex matrix on rectangular representation
std::map<std::string, nPhysImageF<double> > to_rect(const nPhysImageF<mcomplex> &iphys) {
	nPhysImageF<double> re, im;
	std::map<std::string, nPhysImageF<double> > omap;

	re.resize(iphys.getW(), iphys.getH());
	im.resize(iphys.getW(), iphys.getH());

	for (register size_t ii=0; ii<iphys.getSurf(); ii++) {
		mcomplex pt = iphys.point(ii);
		re.set(ii, pt.real() );
		im.set(ii, pt.imag() );
	}

	omap[std::string("real")] = re;
	omap[std::string("imag")] = im;
	return omap;
}

//! split mcomplex matrix on power spectrum, representation
std::map<std::string, nPhysImageF<double> > to_powersp(nPhysImageF<mcomplex> &iphys) {
	nPhysImageF<double> psp;
	std::map<std::string, nPhysImageF<double> > omap;

	psp.resize(iphys.getW(), iphys.getH());

	for (register size_t ii=0; ii<iphys.getSurf(); ii++) {
		psp.set(ii, log10(iphys.point(ii).mcabs()) );
	}

	omap[std::string("power spectrum")] = psp;
	return omap;
}

/*!
 * @}
 */
