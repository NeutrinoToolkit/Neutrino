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

inline void planeFit(nPhysD *pi, double *coeffs)
{
#ifdef HAVE_LIBGSL
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
#endif
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
phys_generate_morlet(morlet_data *md, nPhysD &xx, nPhysD &yy, nPhysC &zz)
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

//TODO:
template<> void
nPhysC::TscanBrightness() {
	return;
}

void 
phys_add(nPhysD &iimage, double val) { 
	if (val!=0.0) {
#pragma omp parallel for
		for (size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)+val); 
		iimage.TscanBrightness();
	}
	ostringstream ostr;
	ostr << val;
	iimage.setName("("+iimage.getName()+")+"+ostr.str());
}

void 
phys_subtract(nPhysD &iimage, double val) {
	if (val!=0.0) {
#pragma omp parallel for
		for (size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)-val); 
		iimage.TscanBrightness();
	}
	ostringstream ostr;
	ostr << val;
	iimage.setName("("+iimage.getName()+")-"+ostr.str());
}

void 
phys_multiply(nPhysD &iimage, double val) {
	if (val!=1.0) {
#pragma omp parallel for
		for (size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)*val); 
		iimage.TscanBrightness();
	}
	ostringstream ostr;
	ostr << val;
	iimage.setName("("+iimage.getName()+")*"+ostr.str());
}

void 
phys_divide(nPhysD &iimage, double val) {
	if (val!=1.0) {
#pragma omp parallel for
		for (size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)/val); 
		iimage.TscanBrightness();
	}
	ostringstream ostr;
	ostr << val;
	iimage.setName("("+iimage.getName()+")/"+ostr.str());

}

void
phys_divide(nPhysC &iimage, double val) {
    if (val!=1.0) {
#pragma omp parallel for
        for (size_t ii=0; ii<iimage.getSurf(); ii++)
            iimage.set(ii, iimage.point(ii)/val);
        iimage.TscanBrightness();
    }
    ostringstream ostr;
    ostr << val;
    iimage.setName("("+iimage.getName()+")/"+ostr.str());

}

void 
phys_point_add(nPhysD &iimage, nPhysD &iimage2) {
	if (iimage.getSurf()==iimage2.getSurf()) {
#pragma omp parallel for
		for (size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)+iimage2.point(ii)); 
		iimage.TscanBrightness();
		iimage.setName("("+iimage.getName()+")+("+iimage2.getName()+")");
	}
}

void 
phys_point_subtract(nPhysD &iimage, nPhysD &iimage2) {
	if (iimage.getSurf()==iimage2.getSurf()) {
#pragma omp parallel for
		for (size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)-iimage2.point(ii)); 
		iimage.TscanBrightness();
		iimage.setName("("+iimage.getName()+")-("+iimage2.getName()+")");
	}
}

void 
phys_point_multiply(nPhysD &iimage, nPhysD &iimage2) {
	if (iimage.getSurf()==iimage2.getSurf()) {
#pragma omp parallel for
		for (size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)*iimage2.point(ii)); 
		iimage.TscanBrightness();
		iimage.setName("("+iimage.getName()+")*("+iimage2.getName()+")");
	}
}

void 
phys_point_divide(nPhysD &iimage, nPhysD &iimage2) {
	if (iimage.getSurf()==iimage2.getSurf()) {
#pragma omp parallel for
		for (size_t ii=0; ii<iimage.getSurf(); ii++) 
			iimage.set(ii, iimage.point(ii)/iimage2.point(ii)); 
		iimage.TscanBrightness();
		iimage.setName("("+iimage.getName()+")/("+iimage2.getName()+")");
	}
}

double phys_sum_points(nPhysD &iimage) {
	double retVal=0.0;
#pragma omp parallel for reduction(+:retVal)
	for (size_t ii=0; ii<iimage.getSurf(); ii++) {
		retVal+=iimage.point(ii);
	}
	return retVal;
}

double phys_sum_square_points(nPhysD &iimage) {
	double retVal=0.0;
#pragma omp parallel for reduction(+:retVal)
	for (size_t ii=0; ii<iimage.getSurf(); ii++) {
		retVal+=iimage.point(ii)*iimage.point(ii);
	}
	return retVal;
}

void phys_opposite(nPhysD &iimage) {
#pragma omp parallel for
	for (size_t ii=0; ii<iimage.getSurf(); ii++) {
		iimage.set(ii,-iimage.point(ii));
	}
	iimage.TscanBrightness();
	iimage.setName("-("+iimage.getName()+")");
}


void phys_inverse(nPhysD &iimage) {
#pragma omp parallel for
	for (size_t ii=0; ii<iimage.getSurf(); ii++) {
		iimage.set(ii,1.0/iimage.point(ii));
	}
	iimage.TscanBrightness();
	iimage.setName("1/("+iimage.getName()+")");
}

void phys_replace(nPhysD &iimage, double oldval, double newval) {
#pragma omp parallel for
	for (size_t ii=0; ii<iimage.getSurf(); ii++) {
		if (iimage.point(ii)==oldval) iimage.set(ii,newval);
	}
	iimage.TscanBrightness();
	ostringstream ostr;
	ostr << "replace(" << iimage.getName() << "," << oldval << "," << newval << ")";
	iimage.setName(ostr.str());
}

void phys_replace_NaN(nPhysD &iimage, double newval) {
#pragma omp parallel for
	for (size_t ii=0; ii<iimage.getSurf(); ii++) {
		if (!isfinite(iimage.point(ii))) iimage.set(ii,newval);
	}
	iimage.TscanBrightness();
	
	ostringstream ostr;
	ostr << "replaceNaN(" << iimage.getName() << "," << newval << ")";
	iimage.setName(ostr.str());
}

void phys_cutoff(nPhysD &iimage, double minval, double maxval) {
    iimage.setShortName("IntensityCutoff");
#pragma omp parallel for
	for (size_t ii=0; ii<iimage.getSurf(); ii++) {
		double val=iimage.point(ii);
        if (isfinite(val)) iimage.set(ii,min(max(val,minval),maxval));
    }
	iimage.TscanBrightness();
	ostringstream ostr;
	ostr << "min_max(" << iimage.getName() << "," << minval << "," << maxval << ")";
	iimage.setName(ostr.str());
}

void 
phys_add_noise(nPhysD &iimage, double vMax=1.0)
{
#pragma omp parallel for
	for (size_t ii=0; ii<iimage.getSurf(); ii++) {
		iimage.set(ii, iimage.point(ii) + vMax*((double)rand() / RAND_MAX));
	}
	iimage.TscanBrightness();
	ostringstream ostr;
	ostr << vMax;
	iimage.setName("("+iimage.getName()+")+rand("+ostr.str()+")");
}


// gaussian blur
void
phys_gaussian_blur(nPhysD &m1, double radius)
{
	DEBUG(5,"radius: "<<radius);
	if (!(radius > 0) )
		return;

	nPhysD xx, yy, gauss;
    nPhysC out;

//FIXME: this is probably wrong for odd matrices
    meshgrid_data md = {-m1.getW()/2., m1.getW()/2., -m1.getH()/2., m1.getH()/2., (int) (m1.getW()), (int) m1.getH()};
	phys_generate_meshgrid(&md, xx, yy);

	gauss.resize(xx.getW(), xx.getH());
	double mult = 1/(pow(radius, 2.)*2*M_PI);
	for (size_t i=0; i<xx.getW(); i++) {
        size_t j;
		for (j=0; j<xx.getH(); j++) {
			gauss.Timg_matrix[j][i] = mult*exp( -(pow(xx.Timg_matrix[j][i],2)+pow(yy.Timg_matrix[j][i],2))/(2.*pow(radius, 2)) );
		}
	}

	phys_convolve(m1, gauss, out);
	out.fftshift();

	for (size_t i=0; i<xx.getW(); i++) {
        size_t j;
		for (j=0; j<xx.getH(); j++) {
			m1.Timg_matrix[j][i] = (out.Timg_matrix[j][i].real())/double(xx.getSurf());
		}
	}

	m1.TscanBrightness();
	ostringstream ostr;
	ostr << radius;
	m1.setName("blur("+m1.getName()+","+ostr.str()+")");
}

void
phys_sin(nPhysD &m1)
{
#pragma omp parallel for
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,sin(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("sin("+m1.getName()+")");
}

void
phys_cos(nPhysD &m1)
{
#pragma omp parallel for
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,cos(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("cos("+m1.getName()+")");
}

void
phys_tan(nPhysD &m1)
{
#pragma omp parallel for
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,tan(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("tan("+m1.getName()+")");
}

void
phys_pow(nPhysD &m1, double exponent)
{
#pragma omp parallel for
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,pow(m1.point(i),exponent));
	}
	m1.TscanBrightness();
	ostringstream ostr;
	ostr << exponent;
	m1.setName("("+m1.getName()+")^"+ostr.str());
}

void
phys_square(nPhysD &m1)
{
#pragma omp parallel for
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,pow(m1.point(i),2));
	}
	m1.TscanBrightness();
	m1.setName("("+m1.getName()+")^2");
}

void
phys_sqrt(nPhysD &m1)
{
#pragma omp parallel for
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,sqrt(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("sqrt("+m1.getName()+")");
}

void
phys_abs(nPhysD &m1)
{
#pragma omp parallel for
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,abs(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("abs("+m1.getName()+")");
}

void
phys_log(nPhysD &m1)
{
#pragma omp parallel for
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,log(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("ln("+m1.getName()+")");
}

void
phys_log10(nPhysD &m1)
{
#pragma omp parallel for
	for (size_t i=0; i< m1.getSurf(); i++) {
		m1.set(i,log10(m1.point(i)));
	}
	m1.TscanBrightness();
	m1.setName("log("+m1.getName()+")");
}

void
phys_fast_gaussian_blur(nPhysD &m1, double radius)
{
	phys_fast_gaussian_blur(m1,radius,radius);
}

void phys_laplace(nPhysD &image) {
    nPhysD my_copy=image;
    image.setShortName("Laplace");
    image.setName("Laplace "+image.getName());
    image.setFromName(image.getFromName());
    double Lap[9];
    Lap[0] = 1.0; Lap[1] = 1.0; Lap[2] = 1.0;
    Lap[3] = 1.0; Lap[4] =-8.0; Lap[5] = 1.0;
    Lap[6] = 1.0; Lap[7] = 1.0; Lap[8] = 1.0;

    for(size_t i = 0 ; i < my_copy.getW() ; i++) {
        for(size_t j = 0 ; j < my_copy.getH(); j++) {
            double val_lap = 0.0;
            for(size_t k = 0 ; k < 3 ; k++) {
                for(size_t l = 0 ; l < 3 ; l++) {
                    val_lap += Lap[l * 3 + k] * my_copy.point((i+1)+(1-k),(j+1)+(1-l));
                }
            }
            image.set(i,j,val_lap);
        }
    }
    image.TscanBrightness();
}

void phys_gauss_laplace(nPhysD &image, double radius) {
    phys_fast_gaussian_blur(image, radius);
    phys_laplace(image);
}


// get sobel matrix
void phys_sobel(nPhysD &image) {
    nPhysD my_copy=image;
    image.setShortName("Sobel");
    image.setName("Sobel "+image.getName());
    image.setFromName(image.getFromName());
    double Gx[9];
    Gx[0] = 1.0; Gx[1] = 0.0; Gx[2] = -1.0;
    Gx[3] = 2.0; Gx[4] = 0.0; Gx[5] = -2.0;
    Gx[6] = 1.0; Gx[7] = 0.0; Gx[8] = -1.0;

    double Gy[9];
    Gy[0] =-1.0; Gy[1] =-2.0; Gy[2] = -1.0;
    Gy[3] = 0.0; Gy[4] = 0.0; Gy[5] =  0.0;
    Gy[6] = 1.0; Gy[7] = 2.0; Gy[8] =  1.0;

    for(size_t i = 0 ; i < my_copy.getW() ; i++) {
        for(size_t j = 0 ; j < my_copy.getH(); j++) {
            double value_gx = 0.0;
            double value_gy = 0.0;
            for(size_t k = 0 ; k < 3 ; k++) {
                for(size_t l = 0 ; l < 3 ; l++) {
                    value_gx += Gx[l * 3 + k] * my_copy.point((i+1)+(1-k),(j+1)+(1-l));
                    value_gy += Gy[l * 3 + k] * my_copy.point((i+1)+(1-k),(j+1)+(1-l));
                }
            }
            image.set(i,j,sqrt(value_gx*value_gx + value_gy*value_gy));
        }
    }
    image.TscanBrightness();
}

void phys_gauss_sobel(nPhysD &image, double radius) {
    phys_fast_gaussian_blur(image, radius);
    phys_sobel(image);
}

void phys_median_filter(nPhysD& image, unsigned int N){
    nPhysD my_copy=image;
    int median_pos=(N*N)/2;
#pragma omp parallel for collapse(2)
    for(size_t i = 0 ; i < image.getW(); i++) {
        for(size_t j = 0 ; j < image.getH(); j++) {
            vector<double> mat(N*N);
            int k=0;
            for(size_t ii = 0 ; ii < N ; ii++) {
               for(size_t jj = 0 ; jj < N; jj++) {
                    mat[k++]=image.point(i+ii,j+jj);
               }
            }
            nth_element(mat.begin(), mat.begin()+median_pos, mat.end());
            my_copy.set(i,j,mat[median_pos]);
        }
    }
#pragma omp parallel for collapse(2)
    for(size_t i = 0 ; i < image.getW()-N ; i++) {
        for(size_t j = 0 ; j < image.getH()-N; j++) {
            image.set(i+N/2,j+N/2,my_copy.point(i,j));
        }
    }
    image.setShortName("median");
    image.setName("median("+image.getName()+")");
    image.TscanBrightness();
}

void
phys_fast_gaussian_blur(nPhysD &image, double radiusX, double radiusY)
{
    vector<double> nan_free_phys(image.getSurf());
#pragma omp parallel for
    for (size_t i=0; i< image.getSurf(); i++) {
        nan_free_phys[i]=isfinite(image.point(i))? image.point(i):image.get_min();
	}
    fftw_complex *b2 = fftw_alloc_complex(image.getH()*((image.getW()/2+1)));
	
    fftw_plan fb = fftw_plan_dft_r2c_2d(image.getH(),image.getW(),&nan_free_phys[0],b2,FFTW_ESTIMATE);
    fftw_plan bb = fftw_plan_dft_c2r_2d(image.getH(),image.getW(),b2,&nan_free_phys[0],FFTW_ESTIMATE);
	
	fftw_execute(fb);

    double sx=pow(image.getW()/(radiusX),2)/2.0;
    double sy=pow(image.getH()/(radiusY),2)/2.0;

#pragma omp parallel for collapse(2)
    for (size_t j = 0 ; j < image.getH(); j++) {
        for (size_t i = 0 ; i < image.getW()/2+1 ; i++) {
            double blur=exp(-((i*i)/sx+(j-image.getH()/2)*(j-image.getH()/2)/sy));
            int k=i+((j+image.getH()/2+1)%image.getH())*(image.getW()/2+1);
			b2[k][0]*=blur;
			b2[k][1]*=blur;
		}
	}
    fftw_execute(bb);

#pragma omp parallel for
    for (size_t i=0; i< image.getSurf(); i++) {
        if (isfinite(image.point(i))) {
            image.set(i,nan_free_phys[i]/image.getSurf());
		}
	}
    image.TscanBrightness();
	fftw_destroy_plan(fb);
	fftw_destroy_plan(bb);
	fftw_free(b2);
}

pair<double, bidimvec<int> > phys_cross_correlate(nPhysD* img1, nPhysD* img2) {
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

	vmin = isfinite(ivec[0]) ? ivec[0]:0;
	vmax = isfinite(ivec[0]) ? ivec[0]:0;

    for (size_t ii=0; ii<vlen; ii++) {
		if (isfinite(ivec[ii])) {
			if (ivec[ii] < vmin)
				vmin = ivec[ii];
			else if (ivec[ii] > vmax)
				vmax = ivec[ii];
		}
	}
}	

bidimvec<size_t> phys_max_p(nPhysD &image) {
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
map<string, nPhysD > to_polar(nPhysC &iphys) {
	nPhysD rho, theta;
	map<string, nPhysD > omap;

	rho.resize(iphys.getW(), iphys.getH());
	theta.resize(iphys.getW(), iphys.getH());

    for (size_t ii=0; ii<iphys.getSurf(); ii++) {
		mcomplex pt = iphys.point(ii);
		rho.set(ii, pt.mod() );
		theta.set(ii, pt.arg() );
	}

	omap[string("rho")] = rho;
	omap[string("theta")] = theta;
	return omap;
}

//! split mcomplex matrix on rectangular representation
map<string, nPhysD > to_rect(const nPhysC &iphys) {
	nPhysD re, im;
	map<string, nPhysD > omap;

	re.resize(iphys.getW(), iphys.getH());
	im.resize(iphys.getW(), iphys.getH());

    for (size_t ii=0; ii<iphys.getSurf(); ii++) {
		mcomplex pt = iphys.point(ii);
		re.set(ii, pt.real() );
		im.set(ii, pt.imag() );
	}

	omap[string("real")] = re;
	omap[string("imag")] = im;
	return omap;
}

//! split mcomplex matrix on power spectrum, representation
map<string, nPhysD > to_powersp(nPhysC &iphys, bool doLog) {
	nPhysD psp;
	map<string, nPhysD > omap;

	psp.resize(iphys.getW(), iphys.getH());

    if (doLog) {
        for (size_t ii=0; ii<iphys.getSurf(); ii++) {
            psp.set(ii, log10(iphys.point(ii).mcabs()) );
        }
    } else {
        for (size_t ii=0; ii<iphys.getSurf(); ii++) {
            psp.set(ii, iphys.point(ii).mcabs() );
        }
    }
    psp.TscanBrightness();
	omap[string("power spectrum")] = psp;
	return omap;
}

// 2 real matrix to complex fftw
nPhysC from_real_imaginary (nPhysD& real, nPhysD&imag) {
    nPhysC ret;
    if (real.getSize() == imag.getSize()) {
          ret.resize(real.getW(),real.getH());
#pragma omp parallel for
          for (size_t ii=0; ii<real.getSurf(); ii++) {
              ret.set(ii, mcomplex(real.point(ii),imag.point(ii)));
          }
    }

    return ret;
}

nPhysC from_real (nPhysD&real, double val){
    nPhysC ret;
    ret.resize(real.getW(),real.getH());
#pragma omp parallel for
    for (size_t ii=0; ii<real.getSurf(); ii++) {
        ret.set(ii, mcomplex(real.point(ii),val));
    }
    return ret;
}

// contour functions
//
void contour_trace(nPhysD &iimage, std::list<vec2> &contour, float level, bool blur, float blur_radius)
{
	// marching squares algorithm
	
	contour.clear();
	bool contour_ok = false;

	nPhysD wimage(iimage); // work image
	if (blur) {
		DEBUG(5, "Blurring image, radius "<<blur_radius);
		phys_fast_gaussian_blur(wimage, blur_radius);
	}

	// 0. find centroid if not supplied
	vec2 centr;
	if (wimage.get_origin() == vec2(0,0)) {
        wimage.set_origin(wimage.max_Tv);
        iimage.set_origin(wimage.max_Tv);
    } else {
		centr = wimage.get_origin();
    }


	// 1. generate boolean map
    vec2 orig = wimage.get_origin();
//	double c_value = wimage.point(orig.x(),orig.y());

	nPhysImageF<short> bmap(wimage.getW(), wimage.getH(), 0);
	for (size_t ii=0; ii<wimage.getSurf(); ii++)
		if (wimage.point(ii) > level) 
			bmap.set(ii, 1);

	// 2. cell map
	nPhysImageF<short> cmap(wimage.getW()-1, wimage.getH()-1, 0);
    for (size_t ii=0; ii<cmap.getSurf(); ii++) {
		int xx = ii%cmap.getW();
		int yy = ii/cmap.getW();

		short cval = (bmap.point(xx,yy)<<3) + (bmap.point(xx+1,yy)<<2) + (bmap.point(xx+1, yy+1)<<1) + bmap.point(xx,yy+1);
		cmap.set(ii, cval);
	}
	
    	// close boundary
	for (size_t ii=0; ii<cmap.getW(); ii++) {
		cmap.set(ii, 0, cmap.point(ii, 0) &~ 12);
		cmap.set(ii, 0, cmap.point(ii, cmap.getH()-1) &~ 3);
	}
	for (size_t ii=0; ii<cmap.getH(); ii++) {
		cmap.set(0, ii, cmap.point(0, ii) &~ 9);
		cmap.set(cmap.getW()-1, ii, cmap.point(cmap.getW()-1, ii) &~ 6);
	}

	// 3. now find contours
	int stats[16];
	for (int i=0; i<16; i++)
		stats[i] = 0;
	for (size_t ii=0; ii<cmap.getSurf(); ii++)
		stats[cmap.point(ii)] ++;

	int b_points = 0;
	for (int ii=1; ii<15; ii++) b_points+=stats[ii]; // total number of boundary points

	DEBUG(5,"There are "<<stats[0]<<" points under threshold, "<<stats[15]<<" points over threshold and "<<b_points<<" boundary points"<<std::endl);

	
	// find only main contour
	if (stats[0] == 0 || stats[15] == 0) {
		DEBUG(5, "No contour possible");
		return;
	}

	int ls_x = orig.x();
	while (ls_x > -1 && cmap.point(ls_x, orig.y()) == 15)
		ls_x--;
	stats[cmap.point(ls_x, orig.y())]--;


	contour.resize(b_points);
	std::list<vec2>::iterator itr = contour.begin(), itr_last = contour.begin();
	*itr = vec2(ls_x, orig.y());

	int n_iter=0;
	while (itr != contour.end()) {
		short xx = (*itr).x();
		short yy = (*itr).y();
		short val = cmap.point(xx, yy);

		//std::cerr<<"[walker] I'm at "<<xx<<":"<<yy<<" with val "<<val<<std::endl;

		if (val==0 || val==15) {
			DEBUG(5, "Walker got sadly lost...");
			break;
		}
		
		stats[val]--;

		// saddles: check central value and last movement
		if (val==5 || val==10) {
			short central = ((.25*wimage.point(xx,yy) + wimage.point(xx+1,yy+1) + wimage.point(xx+1,yy) + wimage.point(xx,yy+1)) > level) ? 1 : -1;
			short saddle_type = (val == 5) ? 1 : -1;

			vec2 last = *itr- *itr_last; // let's hope we're not starting with a saddle...

			//std::cerr<<"[Walker] Saddle point! central: "<<central<<std::endl;

			short xadd, yadd;

			if (last.x() > 0) {
				xadd = 1;
				if (saddle_type < 0)
					xadd*= -1*central;
			} else if (last.x() < 0){
				xadd = -1;
				if (saddle_type < 0)
					xadd*= -1*central;

			} else {
				xadd = last.y();
				if (saddle_type < 0) {
					xadd *= central;
				}
			}

			if (last.y() > 0) {
				yadd = 1;
				if (saddle_type > 0) 
					yadd *= -1*central;
			} else if (last.y() < 0){
				yadd = -1;
				if (saddle_type > 0)
					yadd*= -1*central;

			} else {
				yadd = -last.x();
				if (saddle_type > 0) {
					yadd *= central;
				}
			}

			xx+=xadd;
			yy+=yadd;



		} else if ((val&4) && !(val&8)) {
			yy--;
		} else if ((val&2) && !(val&4)) {
			xx++;
		} else if ((val&1) && !(val&2)) {
			yy++;
		} else if ((val&8) && !(val&1)) {
			xx--;
		}


		itr_last = itr;
		itr++;
		*itr = vec2(xx,yy);
		n_iter++;

		if (*itr == *contour.begin()) {
			DEBUG(5,"Closed contour!! (distance: "<<std::distance(contour.begin(), itr_last));
			DEBUG(5, "Contour size: "<<contour.size()<<", n_iter: "<<n_iter);
			contour_ok = true;

			itr_last++;

			break;
		}

	}

	if (!contour_ok) {
		contour.clear();
		DEBUG(5, "Contour walk failed!");
	} else {
		DEBUG(5, "Contour walk finished");
		contour.resize(std::distance(contour.begin(), itr_last));
	}


}

std::list<double> contour_integrate(nPhysD &iimage, std::list<vec2> &contour, bool integrate_boundary)
{
	vec2 bbox_inf = contour.front(), bbox_sup = contour.front();

	nPhysD check_image(iimage);
	check_image.TscanBrightness();
	double check_val = check_image.get_min() - 1;
	int points_count = 0;
	//double c_integral = 0;

	for (std::list<vec2>::iterator itr = contour.begin(); itr != contour.end(); ++itr) {
		bbox_inf = vmath::min(bbox_inf, *itr);
		bbox_sup = vmath::max(bbox_sup, *itr);
		check_image.set((*itr).x(), (*itr).y(), check_val);
	}

	// coutour bbox subimage (to perform integral on)
	nPhysD intg_image = check_image.sub(bbox_inf.x(), bbox_inf.y(), bbox_sup.x()-bbox_inf.x()+1, bbox_sup.y()-bbox_inf.y()+1);
	intg_image.set_origin(iimage.get_origin()-bbox_inf);

	// integrate by scanline fill
	double intg=0;
	std::list<vec2> up_pl, scan_pl, tmplist;

	vec2 starting_point = intg_image.get_origin();

	for (int xx=starting_point.x(); intg_image.point(xx, starting_point.y()) != check_val; xx++) {
		up_pl.push_back(vec2(xx, starting_point.y()));
	}
	
	for (int xx=starting_point.x(); intg_image.point(xx, starting_point.y()) != check_val; xx--) {
		up_pl.push_front(vec2(xx, starting_point.y()));
	}
	

	//std::cerr<<"walk starting from "<<up_pl.front()<<" to "<<up_pl.back()<<std::endl;

	int line_check =0;


	while (!up_pl.empty()) {
		
		//scan_pl = up_pl;
		//up_pl.clear();

		tmplist.clear();
		scan_pl.clear();
		std::list<vec2>::iterator itr = up_pl.begin(), itrf = up_pl.begin();
		itrf++;

		while (itrf != up_pl.end()) {
			if (((*itrf).x()-(*itr).x()) > 1) {
				//std::cerr<<"separation at "<<*itr<<" -- "<<*itrf<<std::endl;
				scan_pl.push_back(*itr);
				tmplist.clear();
				tmplist.push_back(*itrf);


				//std::cerr<<"line "<<line_check<<": sep/walk starting from "<<*itr<<" to "<<*itrf<<std::endl;
				vec2 ref_sx = *itr, ref_dx = *itrf;
				while (intg_image.point(scan_pl.back(), check_val) != check_val) {
					ref_sx+=vec2(1,0);
					scan_pl.push_back(ref_sx);
				}
				//std::cerr<<"line "<<line_check<<": sep/walk starting from "<<scan_pl.back()<<" to "<<tmplist.front()<<std::endl;
				while (intg_image.point(tmplist.front(), check_val) != check_val) {
					ref_dx -= vec2(1,0);
					tmplist.push_front(ref_dx);
					//std::cerr<<"\t\tsep/walking to "<<tmplist.front()<<" - "<<intg_image.point(tmplist.front())<<std::endl;
				}
				//std::cerr<<"line "<<line_check<<": sep/walk starting from "<<scan_pl.back()<<" to "<<tmplist.front()<<std::endl;
				scan_pl.splice(scan_pl.end(), tmplist);

				itr++;
				itrf++;


			} else scan_pl.push_back(*itr); // ovvero se linea senza separazioni somma tutti i punti

			itr++;
			itrf++;
		}
		
		//std::cerr<<"line "<<line_check<<": walk starting from "<<scan_pl.front()<<" to "<<scan_pl.back()<<std::endl;

		
		while (intg_image.point(scan_pl.front(), check_val) != check_val) {
			scan_pl.push_front(scan_pl.front()+vec2(-1, 0));
			//std::cerr<<"--------------"<<intg_image.getPoint(scan_pl.front())<<std::endl;
		}

		while (intg_image.point(scan_pl.back(), check_val) != check_val) {
			scan_pl.push_back(scan_pl.back()+vec2(1, 0));
			//std::cerr<<"--------------"<<intg_image.point(scan_pl.back(), check_val)<<std::endl;
		}

		//std::cerr<<"line "<<line_check<<": walk starting from "<<scan_pl.front()<<" to "<<scan_pl.back()<<std::endl;
		//if (line_check == 38)
		//	break;

		up_pl.clear();

		while (!scan_pl.empty()) {
			vec2 pp = scan_pl.front();
			scan_pl.pop_front();
			if (intg_image.point(pp, check_val) != check_val) {
				intg+=intg_image.point(pp);
				points_count++;

				intg_image.set(pp.x(), pp.y(), check_val);

				up_pl.push_back(vec2(pp.x(), pp.y()+1));
				up_pl.push_back(vec2(pp.x(), pp.y()-1));
				//std::cerr<<"point read: "<<pp<<std::endl;
			} 
			//else std::cerr<<"--------------- cippacazzo ------------------"<<pp<<std::endl;
		}

		line_check++;

	}

	DEBUG(5, "contour integral: "<<intg);
	if (integrate_boundary) {
		// add boundary points
		double bps = 0;
		for (std::list<vec2>::iterator itr = contour.begin(); itr != contour.end(); ++itr) {
			bps+= iimage.point((*itr).x(), (*itr).y());
			points_count++;
		}
		DEBUG(5, "boundary points account for "<<bps);
		intg+=bps;	
	}

	std::list<double> ret;
	ret.push_front(points_count);
	ret.push_front(intg);
	return ret;

}


/*!
 * @}
 */
