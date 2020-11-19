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

/*! \addtogroup nPhysMaths
 * @{
 */

inline void physMath::planeFit(physD *pi, vec2f &coeffs)
{
#ifdef HAVE_LIBGSL
    int fit_n = pi->getSurf();	// grossino...
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

    coeffs.set_first(c->data[0]);
    coeffs.set_second(c->data[1]);

    // free all
    gsl_matrix_free(X);
    gsl_vector_free(y);
    gsl_vector_free(c);
    gsl_multifit_linear_free(work);
#endif
}

inline void physMath::phys_reverse_vector(double *buf, int size)
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


void 
physMath::phys_add(physD &iimage, double val) {
    if (val!=0.0) {
#pragma omp parallel for
        for (size_t ii=0; ii<iimage.getSurf(); ii++)
            iimage.set(ii, iimage.point(ii)+val);
        iimage.TscanBrightness();
    }
    std::ostringstream ostr;
    ostr << val;
    iimage.setName("("+iimage.getName()+")+"+ostr.str());
}

void 
physMath::phys_subtract(physD &iimage, double val) {
    if (val!=0.0) {
#pragma omp parallel for
        for (size_t ii=0; ii<iimage.getSurf(); ii++)
            iimage.set(ii, iimage.point(ii)-val);
        iimage.TscanBrightness();
    }
    std::ostringstream ostr;
    ostr << val;
    iimage.setName("("+iimage.getName()+")-"+ostr.str());
}

void 
physMath::phys_multiply(physD &iimage, double val) {
    if (val!=1.0) {
#pragma omp parallel for
        for (size_t ii=0; ii<iimage.getSurf(); ii++)
            iimage.set(ii, iimage.point(ii)*val);
        iimage.TscanBrightness();
    }
    std::ostringstream ostr;
    ostr << val;
    iimage.setName("("+iimage.getName()+")*"+ostr.str());
}

void 
physMath::phys_divide(physD &iimage, double val) {
    if (val!=1.0) {
#pragma omp parallel for
        for (size_t ii=0; ii<iimage.getSurf(); ii++)
            iimage.set(ii, iimage.point(ii)/val);
        iimage.TscanBrightness();
    }
    std::ostringstream ostr;
    ostr << val;
    iimage.setName("("+iimage.getName()+")/"+ostr.str());

}

void
physMath::phys_divide(physC &iimage, double val) {
    if (val!=1.0) {
#pragma omp parallel for
        for (size_t ii=0; ii<iimage.getSurf(); ii++)
            iimage.set(ii, iimage.point(ii)/val);
        iimage.TscanBrightness();
    }
    std::ostringstream ostr;
    ostr << val;
    iimage.setName("("+iimage.getName()+")/"+ostr.str());

}

void physMath::phys_remainder(physD &iimage, double val) {
#pragma omp parallel for
    for (size_t ii=0; ii<iimage.getSurf(); ii++) {
        double rem=std::remainder(iimage.point(ii), val);
        iimage.set(ii, (rem < 0) ? rem+1: rem );
    }
    iimage.TscanBrightness();
    std::ostringstream ostr;
    ostr << val;
    iimage.setName("remainder("+iimage.getName()+","+ostr.str()+")");
}

void 
physMath::phys_point_add(physD &iimage, physD &iimage2) {
    if (iimage.getSurf()==iimage2.getSurf()) {
#pragma omp parallel for
        for (size_t ii=0; ii<iimage.getSurf(); ii++)
            iimage.set(ii, iimage.point(ii)+iimage2.point(ii));
        iimage.TscanBrightness();
        iimage.setName("("+iimage.getName()+")+("+iimage2.getName()+")");
    }
}

void 
physMath::phys_point_subtract(physD &iimage, physD &iimage2) {
    if (iimage.getSurf()==iimage2.getSurf()) {
#pragma omp parallel for
        for (size_t ii=0; ii<iimage.getSurf(); ii++)
            iimage.set(ii, iimage.point(ii)-iimage2.point(ii));
        iimage.TscanBrightness();
        iimage.setName("("+iimage.getName()+")-("+iimage2.getName()+")");
    }
}

void 
physMath::phys_point_multiply(physD &iimage, physD &iimage2) {
    if (iimage.getSurf()==iimage2.getSurf()) {
#pragma omp parallel for
        for (size_t ii=0; ii<iimage.getSurf(); ii++)
            iimage.set(ii, iimage.point(ii)*iimage2.point(ii));
        iimage.TscanBrightness();
        iimage.setName("("+iimage.getName()+")*("+iimage2.getName()+")");
    }
}

void 
physMath::phys_point_divide(physD &iimage, physD &iimage2) {
    if (iimage.getSurf()==iimage2.getSurf()) {
#pragma omp parallel for
        for (size_t ii=0; ii<iimage.getSurf(); ii++)
            iimage.set(ii, iimage.point(ii)/iimage2.point(ii));
        iimage.TscanBrightness();
        iimage.setName("("+iimage.getName()+")/("+iimage2.getName()+")");
    }
}

double physMath::phys_sum_points(physD &iimage) {
    double retVal=0.0;
#pragma omp parallel for reduction(+:retVal)
    for (size_t ii=0; ii<iimage.getSurf(); ii++) {
        retVal+=iimage.point(ii);
    }
    return retVal;
}

double physMath::phys_sum_square_points(physD &iimage) {
    double retVal=0.0;
#pragma omp parallel for reduction(+:retVal)
    for (size_t ii=0; ii<iimage.getSurf(); ii++) {
        retVal+=iimage.point(ii)*iimage.point(ii);
    }
    return retVal;
}

void physMath::phys_opposite(physD &iimage) {
#pragma omp parallel for
    for (size_t ii=0; ii<iimage.getSurf(); ii++) {
        iimage.set(ii,-iimage.point(ii));
    }
    iimage.TscanBrightness();
    iimage.setName("-("+iimage.getName()+")");
}


void physMath::phys_inverse(physD &iimage) {
#pragma omp parallel for
    for (size_t ii=0; ii<iimage.getSurf(); ii++) {
        iimage.set(ii,1.0/iimage.point(ii));
    }
    iimage.TscanBrightness();
    iimage.setName("1/("+iimage.getName()+")");
}

void physMath::phys_replace(physD &iimage, double oldval, double newval) {
#pragma omp parallel for
    for (size_t ii=0; ii<iimage.getSurf(); ii++) {
        if (iimage.point(ii)==oldval) iimage.set(ii,newval);
    }
    iimage.TscanBrightness();
    std::ostringstream ostr;
    ostr << "replace(" << iimage.getName() << "," << oldval << "," << newval << ")";
    iimage.setName(ostr.str());
}

void physMath::phys_replace_NaN(physD &iimage, double newval) {
#pragma omp parallel for
    for (size_t ii=0; ii<iimage.getSurf(); ii++) {
        if (!std::isfinite(iimage.point(ii))) iimage.set(ii,newval);
    }
    iimage.TscanBrightness();

    std::ostringstream ostr;
    ostr << "replaceNaN(" << iimage.getName() << "," << newval << ")";
    iimage.setName(ostr.str());
}

void 
physMath::add_noise(physD &iimage, double vMax=1.0)
{
#pragma omp parallel for
    for (size_t ii=0; ii<iimage.getSurf(); ii++) {
        iimage.set(ii, iimage.point(ii) + vMax*((double)rand() / RAND_MAX));
    }
    iimage.TscanBrightness();
    std::ostringstream ostr;
    ostr << vMax;
    iimage.setName("("+iimage.getName()+")+rand("+ostr.str()+")");
}

void
physMath::phys_sin(physD &m1)
{
#pragma omp parallel for
    for (size_t i=0; i< m1.getSurf(); i++) {
        m1.set(i,sin(m1.point(i)));
    }
    m1.TscanBrightness();
    m1.setName("sin("+m1.getName()+")");
}

void
physMath::phys_cos(physD &m1)
{
#pragma omp parallel for
    for (size_t i=0; i< m1.getSurf(); i++) {
        m1.set(i,cos(m1.point(i)));
    }
    m1.TscanBrightness();
    m1.setName("cos("+m1.getName()+")");
}

void
physMath::phys_tan(physD &m1)
{
#pragma omp parallel for
    for (size_t i=0; i< m1.getSurf(); i++) {
        m1.set(i,tan(m1.point(i)));
    }
    m1.TscanBrightness();
    m1.setName("tan("+m1.getName()+")");
}

void
physMath::phys_pow(physD &m1, double exponent)
{
#pragma omp parallel for
    for (size_t i=0; i< m1.getSurf(); i++) {
        m1.set(i,pow(m1.point(i),exponent));
    }
    m1.TscanBrightness();
    std::ostringstream ostr;
    ostr << exponent;
    m1.setName("("+m1.getName()+")^"+ostr.str());
}

void
physMath::phys_square(physD &m1)
{
#pragma omp parallel for
    for (size_t i=0; i< m1.getSurf(); i++) {
        m1.set(i,pow(m1.point(i),2));
    }
    m1.TscanBrightness();
    m1.setName("("+m1.getName()+")^2");
}

void
physMath::phys_sqrt(physD &m1)
{
#pragma omp parallel for
    for (size_t i=0; i< m1.getSurf(); i++) {
        m1.set(i,sqrt(m1.point(i)));
    }
    m1.TscanBrightness();
    m1.setName("sqrt("+m1.getName()+")");
}

void
physMath::phys_abs(physD &m1)
{
#pragma omp parallel for
    for (size_t i=0; i< m1.getSurf(); i++) {
        m1.set(i,abs(m1.point(i)));
    }
    m1.TscanBrightness();
    m1.setName("abs("+m1.getName()+")");
}

void
physMath::phys_log(physD &m1)
{
#pragma omp parallel for
    for (size_t i=0; i< m1.getSurf(); i++) {
        m1.set(i,log(m1.point(i)));
    }
    m1.TscanBrightness();
    m1.setName("ln("+m1.getName()+")");
}

void
physMath::phys_log10(physD &m1)
{
#pragma omp parallel for
    for (size_t i=0; i< m1.getSurf(); i++) {
        m1.set(i,log10(m1.point(i)));
    }
    m1.TscanBrightness();
    m1.setName("log("+m1.getName()+")");
}

void
physMath::phys_transpose(physD &m1)
{
    physD m2(m1);
    m1.resize(m2.getH(),m2.getW());
#pragma omp parallel for collapse(2)
    for(size_t i = 0 ; i < m1.getW(); i++) {
        for(size_t j = 0 ; j < m1.getH(); j++) {
            m1.set(i,j,m2.point(j,i));
        }
    }

    m1.TscanBrightness();
    m1.setName("Traspose("+m1.getName()+")");
}

void
physMath::phys_fast_gaussian_blur(physD &m1, double radius)
{
    phys_fast_gaussian_blur(m1,radius,radius);
}

void physMath::phys_laplace(physD &image) {
    physD my_copy=image;
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

void physMath::phys_gauss_laplace(physD &image, double radius) {
    phys_fast_gaussian_blur(image, radius);
    physMath::phys_laplace(image);
}


// get sobel matrix
void physMath::phys_sobel(physD &image) {
    physD my_copy=image;
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

void physMath::phys_sobel_dir(physD &image) {
    physD my_copy=image;
    image.setShortName("SobelDir");
    image.setName("SobelDir "+image.getName());
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
            image.set(i,j,atan2(value_gy,value_gx));
        }
    }
    image.TscanBrightness();
}

void physMath::phys_scharr(physD &image) {
    physD my_copy=image;
    image.setShortName("Sobel");
    image.setName("Sobel "+image.getName());
    image.setFromName(image.getFromName());
    double Gx[9];
    Gx[0] = 3.0; Gx[1] = 0.0; Gx[2] = -3.0;
    Gx[3] =10.0; Gx[4] = 0.0; Gx[5] =-10.0;
    Gx[6] = 3.0; Gx[7] = 0.0; Gx[8] = -3.0;

    double Gy[9];
    Gy[0] = 3.0; Gy[1] =10.0; Gy[2] =  3.0;
    Gy[3] = 0.0; Gy[4] = 0.0; Gy[5] =  0.0;
    Gy[6] =-3.0; Gy[7] =-10.0; Gy[8] =-3.0;

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


void physMath::phys_gauss_sobel(physD &image, double radius) {
    phys_fast_gaussian_blur(image, radius);
    phys_sobel(image);
}

void physMath::phys_set_all(physD &image, double newval) {
#pragma omp parallel for
    for (size_t i=0; i< image.getSurf(); i++) {
        image.set(i,newval);
    }
    image.TscanBrightness();
}

void physMath::phys_median_filter(physD& image, unsigned int N){
    physD my_copy=image;
    int median_pos=(N*N)/2;
#pragma omp parallel for collapse(2)
    for(size_t i = 0 ; i < image.getW(); i++) {
        for(size_t j = 0 ; j < image.getH(); j++) {
            std::vector<double> mat(N*N);
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
physMath::phys_fast_gaussian_blur(physD &image, double radiusX, double radiusY)
{
    if (radiusX==0  && radiusY==0) return;
    std::vector<double> nan_free_phys(image.getSurf());
#pragma omp parallel for
    for (size_t i=0; i< image.getSurf(); i++) {
        nan_free_phys[i]=std::isfinite(image.point(i))? image.point(i):image.get_min();
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
        if (std::isfinite(image.point(i))) {
            image.set(i,nan_free_phys[i]/image.getSurf());
        }
    }
    image.TscanBrightness();
    fftw_destroy_plan(fb);
    fftw_destroy_plan(bb);
    fftw_free(b2);
}

void
physMath::phys_integratedNe(physD &image, double lambda_m)
{
    // image is fringeshifts (i.e. phase/2pi)
    double toNe = 8.0*M_PI*M_PI*_phys_emass*_phys_vacuum_eps*_phys_cspeed*_phys_cspeed/(_phys_echarge*_phys_echarge);

    WARNING("toNe" << toNe);

    phys_multiply(image, toNe/lambda_m);
    image.setShortName("integratedNePlasma");
    image.prop["intergratedNe_lambda_m"]=lambda_m;
    image.prop["unitsCB"]="m-2";

}

std::pair<double, bidimvec<int> > physMath::phys_cross_correlate(physD* img1, physD* img2) {
    size_t dx=img1->getW();
    size_t dy=img1->getH();
    double maxValue=-1.0;
    bidimvec<int> maxP(0,0);

    if (dx == img2->getW() && dy== img2->getH()) {
        physD rPhys(dx,dy,0.0,"Result");

        fftw_complex *myData1C=fftw_alloc_complex(dy*(dx/2+1));
        fftw_complex *myData2C=fftw_alloc_complex(dy*(dx/2+1));

        fftw_plan plan1R2C=fftw_plan_dft_r2c_2d(dy,dx, img1->Timg_buffer, myData1C, FFTW_ESTIMATE);
        fftw_plan plan2R2C=fftw_plan_dft_r2c_2d(dy,dx, img2->Timg_buffer, myData2C, FFTW_ESTIMATE);

        fftw_plan planC2R=fftw_plan_dft_c2r_2d(dy,dx, myData1C, rPhys.Timg_buffer, FFTW_ESTIMATE);

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
                if (rPhys.point(i,j) > rPhys.point(maxP.x(),maxP.y())) {
                    maxP=vec2f(i,j);
                }
            }
        }
        maxValue=rPhys.point(maxP.x(),maxP.y());
        bidimvec<int> shift(dx/2+1,dy/2+1);
        maxP+=shift;
        maxP=vec2f(maxP.x()%dx,maxP.y()%dy)-shift;

        DEBUG(5,"max corr " << maxP.x() << " " << maxP.y() << " " << maxValue);

        fftw_free(myData1C);
        fftw_free(myData2C);
        fftw_destroy_plan(plan1R2C);
        fftw_destroy_plan(plan2R2C);
        fftw_destroy_plan(planC2R);
    }

    return std::make_pair(maxValue, maxP);

}
void physMath::phys_get_vec_brightness(const double *ivec, size_t vlen, double &vmin, double &vmax)
{
    if (ivec==NULL || vlen<1) {
        vmin = 0;
        vmax = 0;
        return;
    }

    vmin = std::isfinite(ivec[0]) ? ivec[0]:0;
    vmax = std::isfinite(ivec[0]) ? ivec[0]:0;

    for (size_t ii=0; ii<vlen; ii++) {
        if (std::isfinite(ivec[ii])) {
            if (ivec[ii] < vmin)
                vmin = ivec[ii];
            else if (ivec[ii] > vmax)
                vmax = ivec[ii];
        }
    }
}	

bidimvec<size_t> physMath::phys_max_p(physD &image) {
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
std::map<std::string, physD > physMath::to_polar(physC &iphys) {
    physD rho, theta;
    std::map<std::string, physD > omap;

    rho.resize(iphys.getW(), iphys.getH());
    theta.resize(iphys.getW(), iphys.getH());

    for (size_t ii=0; ii<iphys.getSurf(); ii++) {
        mcomplex pt = iphys.point(ii);
        rho.set(ii, pt.mod() );
        theta.set(ii, pt.arg() );
    }

    omap["rho"] = rho;
    omap["theta"] = theta;
    return omap;
}

//! split mcomplex matrix on rectangular representation
std::map<std::string, physD > physMath::to_rect(const physC &iphys) {
    physD re, im;
    std::map<std::string, physD > omap;

    re.resize(iphys.getW(), iphys.getH());
    im.resize(iphys.getW(), iphys.getH());

    for (size_t ii=0; ii<iphys.getSurf(); ii++) {
        mcomplex pt = iphys.point(ii);
        re.set(ii, pt.real() );
        im.set(ii, pt.imag() );
    }

    omap["real"] = re;
    omap["imag"] = im;
    return omap;
}

//! split mcomplex matrix on power spectrum, representation
std::map<std::string, physD > physMath::to_powersp(physC &iphys, bool doLog) {
    physD psp;
    std::map<std::string, physD > omap;

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
    omap["power spectrum"] = psp;
    return omap;
}

// 2 real matrix to complex fftw
physC physMath::from_real_imaginary (physD& real, physD&imag) {
    physC ret;
    if (real.getSize() == imag.getSize()) {
        ret.resize(real.getW(),real.getH());
#pragma omp parallel for
        for (size_t ii=0; ii<real.getSurf(); ii++) {
            ret.set(ii, mcomplex(real.point(ii),imag.point(ii)));
        }
    }

    return ret;
}

physC physMath::from_real (physD&real, double val){
    physC ret;
    ret.resize(real.getW(),real.getH());
#pragma omp parallel for
    for (size_t ii=0; ii<real.getSurf(); ii++) {
        ret.set(ii, mcomplex(real.point(ii),val));
    }
    return ret;
}

// contour functions
//
void physMath::contour_trace(physD &iimage, std::list<vec2i> &contour, double level, bool blur, double blur_radius)
{
    // marching squares algorithm

    contour.clear();
    bool contour_ok = false;

    physD wimage(iimage); // work image
    if (blur) {
        DEBUG(5, "Blurring image, radius "<<blur_radius);
        phys_fast_gaussian_blur(wimage, blur_radius);
    }

    // 0. find centroid if not supplied
    vec2i centr;
    if (wimage.get_origin() == vec2i(0,0)) {
        wimage.set_origin(wimage.max_Tv);
        iimage.set_origin(wimage.max_Tv);
    } else {
        centr = wimage.get_origin();
    }


    // 1. generate boolean map
    vec2i orig = wimage.get_origin();
    //	double c_value = wimage.point(orig.x(),orig.y());

    nPhysImageF<short> bmap(wimage.getW(), wimage.getH(), 0);
    for (unsigned int ii=0; ii<wimage.getSurf(); ii++)
        if (wimage.point(ii) > level)
            bmap.set(ii, 1);

    // 2. cell map
    nPhysImageF<short> cmap(wimage.getW()-1, wimage.getH()-1, 0);
    for (unsigned int ii=0; ii<cmap.getSurf(); ii++) {
        unsigned int xx = ii%cmap.getW();
        unsigned int yy = ii/cmap.getW();

        short cval = (bmap.point(xx,yy)<<3) + (bmap.point(xx+1,yy)<<2) + (bmap.point(xx+1, yy+1)<<1) + bmap.point(xx,yy+1);
        cmap.set(ii, cval);
    }

    // close boundary
    for (unsigned int ii=0; ii<cmap.getW(); ii++) {
        cmap.set(ii, 0, cmap.point(ii, 0) &~ 12);
        cmap.set(ii, 0, cmap.point(ii, cmap.getH()-1) &~ 3);
    }
    for (unsigned int ii=0; ii<cmap.getH(); ii++) {
        cmap.set(0, ii, cmap.point(0, ii) &~ 9);
        cmap.set(cmap.getW()-1, ii, cmap.point(cmap.getW()-1, ii) &~ 6);
    }

    // 3. now find contours
    int stats[16];
    for (int i=0; i<16; i++)
        stats[i] = 0;
    for (unsigned int ii=0; ii<cmap.getSurf(); ii++)
        stats[cmap.point(ii)] ++;

    int b_points = 1;
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
    std::list<vec2i>::iterator itr = contour.begin(), itr_last = contour.begin();
    *itr = vec2i(ls_x, orig.y());

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

            vec2i last = *itr- *itr_last; // let's hope we're not starting with a saddle...

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
        *itr = vec2i(xx,yy);
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
        DEBUG(5, "Contour walk finished " << contour.size());
        contour.resize(std::distance(contour.begin(), itr_last));
    }


}

nPhysImageF<char> physMath::contour_surface_map(physD &iimage, std::list<vec2i> &contour)
{
    DEBUG("------------------------- init contour surface mapping ------------------------");
    DEBUG("got "<<contour.size()<<" points in the contour");

    vec2i bbox_inf = contour.front(), bbox_sup = contour.front();

    physD check_image(iimage);

    // image map relative to contour. Legend:
    // 'u': undef
    // 'i': inside
    // 'o': outside
    // 'c': contour
    nPhysImageF<char> ci_map(iimage.getW(), iimage.getH(), 'u');


    check_image.TscanBrightness();
    double check_val = check_image.get_min() - 1;
    //int points_count = 0;
    //double c_integral = 0;

    // set to check_val contour and image boundaries
    for (std::list<vec2i>::iterator itr = contour.begin(); itr != contour.end(); ++itr) {
        bbox_inf = vmath::min(bbox_inf, *itr);
        bbox_sup = vmath::max(bbox_sup, *itr);
        check_image.set((*itr).x(), (*itr).y(), check_val);
        ci_map.set((*itr).x(), (*itr).y(), 'c');
    }
    for (unsigned int xx=0; xx<check_image.getW(); xx++) {
        check_image.set(xx, 0, check_val);
        check_image.set(xx, check_image.getH()-1, check_val);
    }
    for (unsigned int yy=0; yy<check_image.getH(); yy++) {
        check_image.set(0, yy, check_val);
        check_image.set(check_image.getW()-1, yy, check_val);
    }
    DEBUG("bounding box corners are "<<bbox_inf<<", "<<bbox_sup);

    // integration on subimage is DISABLED
    // coutour bbox subimage (to perform integral on)
    //physD intg_image = check_image.sub(bbox_inf.x(), bbox_inf.y(), bbox_sup.x()-bbox_inf.x()+1, bbox_sup.y()-bbox_inf.y()+1);
    //intg_image.set_origin(iimage.get_origin()-bbox_inf);

    // integrate by scanline fill
    //double intg=0, intg_sq=0;
    std::list<vec2i> up_pl, scan_pl, tmplist;

    // starting point needs to be INSIDE the contour. If origin is not set (i.e. 0:0)
    // the center of the bbox is a good starting point.

    vec2i iimage_orig = iimage.get_origin();

    vec2i starting_point;
    if (iimage_orig == vec2i(0,0)) {
        starting_point = bbox_inf+vec2i(0.5*(bbox_sup-bbox_inf));
        DEBUG("origin is not set: recalculating");
    } else {
        DEBUG("origin is set: using it as starting point");
        starting_point = iimage_orig;
    }

    // check if starting point is inside or outside
    int sp_is_inside = true; bool state_change = false; // mind: image boundaries ARE boundaries
    for (unsigned int xx=starting_point.x(); xx<check_image.getW(); xx++) {
        if (check_image.point(xx, starting_point.y()) == check_val && !state_change) {
            state_change = true;
            sp_is_inside = !sp_is_inside;
        } else if (check_image.point(xx, starting_point.y()) != check_val && state_change) {
            state_change = false;
        }
    }
    if (sp_is_inside) {
        DEBUG("starting point is INSIDE: "<<starting_point);
    } else DEBUG("starting point is OUTSIDE: "<<starting_point);

    // ================ HIC SUNT LEONES ===============

    // populate starting vector
    for (int xx=starting_point.x(); check_image.point(xx, starting_point.y()) != check_val; xx++) {
        up_pl.push_back(vec2i(xx, starting_point.y()));
    }
    for (int xx=starting_point.x(); check_image.point(xx, starting_point.y()) != check_val; xx--) {
        up_pl.push_front(vec2i(xx, starting_point.y()));
    }

    DEBUG("walk starting from "<<up_pl.front()<<" to "<<up_pl.back());

    int line_check =0;

    // the maximum possible surface w/in a given contour is the circular case;
    // 20% for additional safety
    int safety_counter = 0;
    int safety_counter_max;
    if (sp_is_inside) {
        safety_counter_max = 1.2*((contour.size()*contour.size())/(4*3.14));
    } else {
        safety_counter_max = check_image.getSurf();
    }
    DEBUG("Safety counter max value is "<<safety_counter_max);


    while (!up_pl.empty() && (safety_counter<safety_counter_max)) {

        //scan_pl = up_pl;
        //up_pl.clear();

        tmplist.clear();
        scan_pl.clear();
        std::list<vec2i>::iterator itr = up_pl.begin(), itrf = up_pl.begin();
        itrf++;

        while (itrf != up_pl.end()) {
            if (((*itrf).x()-(*itr).x()) > 1) {
                //std::cerr<<"separation at "<<*itr<<" -- "<<*itrf<<std::endl;
                scan_pl.push_back(*itr);
                tmplist.clear();
                tmplist.push_back(*itrf);


                //std::cerr<<"line "<<line_check<<": sep/walk starting from "<<*itr<<" to "<<*itrf<<std::endl;
                vec2i ref_sx = *itr, ref_dx = *itrf;
                while (check_image.point(scan_pl.back(), check_val) != check_val) {
                    ref_sx+=vec2i(1,0);
                    scan_pl.push_back(ref_sx);
                }
                //std::cerr<<"line "<<line_check<<": sep/walk starting from "<<scan_pl.back()<<" to "<<tmplist.front()<<std::endl;
                while (check_image.point(tmplist.front(), check_val) != check_val) {
                    ref_dx -= vec2i(1,0);
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


        /*
         * Reason for the following lines:
         *
         * The two up_pl.push_back() methods in the last while loop do mix points from the upper and the lower lines. As a
         * consequence, the while loops meant to expand scan_pl on the front and on the back side, will
         * only expand the upper OR the lower on each side
         *
         */

        std::list<vec2i>::iterator h_front = scan_pl.begin(); ++h_front;
        while (check_image.point(scan_pl.front(), check_val) != check_val) {
            scan_pl.push_front(scan_pl.front()+vec2i(-1, 0));
        }

        scan_pl.push_front(*h_front);
        while (check_image.point(scan_pl.front(), check_val) != check_val) {
            scan_pl.push_front(scan_pl.front()+vec2i(-1, 0));
        }

        std::list<vec2i>::iterator h_back = scan_pl.end(); --h_back; --h_back;
        while (check_image.point(scan_pl.back(), check_val) != check_val) {
            scan_pl.push_back(scan_pl.back()+vec2i(1, 0));
        }
        scan_pl.push_back(*h_back);
        while (check_image.point(scan_pl.back(), check_val) != check_val) {
            scan_pl.push_back(scan_pl.back()+vec2i(1, 0));
        }

        //std::cerr<<"line "<<line_check<<": walk starting from "<<scan_pl.front()<<" to "<<scan_pl.back()<<std::endl;
        //if (line_check == 38)
        //	break;

        up_pl.clear();

        while (!scan_pl.empty()) {
            vec2i pp = scan_pl.front();
            scan_pl.pop_front();
            if (check_image.point(pp, check_val) != check_val) {
                //intg+=check_image.point(pp);
                //intg_sq+=pow(check_image.point(pp), 2);
                //points_count++;

                check_image.set(pp.x(), pp.y(), check_val);
                if (sp_is_inside) ci_map.set(pp.x(), pp.y(), 'i');
                else ci_map.set(pp.x(),pp.y(), 'o');
                safety_counter++;

                up_pl.push_back(vec2i(pp.x(), pp.y()+1));
                up_pl.push_back(vec2i(pp.x(), pp.y()-1));
                //std::cerr<<"point read: "<<pp<<std::endl;
            }
            //else std::cerr<<"--------------- cippacazzo ------------------"<<pp<<std::endl;
        }

        line_check++;


    }

    if (safety_counter >= safety_counter_max) {
        DEBUG("Maximum recursion reached, exit forced, integration failed");
    } else {
        // fill the remaining image
        if (sp_is_inside) {
            for (unsigned int xx=0; xx<ci_map.getSurf(); xx++)
                if (ci_map.point(xx) == 'u') ci_map.set(xx, 'o');
        } else {
            for (unsigned int xx=0; xx<ci_map.getSurf(); xx++)
                if (ci_map.point(xx) == 'u') ci_map.set(xx, 'i');

        }
    }

    return ci_map;
}


std::list<double> physMath::contour_integrate(physD &iimage, std::list<vec2i> &contour, bool integrate_boundary)
{

    DEBUG("------------------------- init contour integration ------------------------");
    DEBUG("got "<<contour.size()<<" points in the contour");

    vec2i bbox_inf = contour.front(), bbox_sup = contour.front();


    physD check_image(iimage);
    check_image.TscanBrightness();
    double check_val = check_image.get_min() - 1;
    int points_count = 0;
    //double c_integral = 0;

    for (std::list<vec2i>::iterator itr = contour.begin(); itr != contour.end(); ++itr) {
        bbox_inf = vmath::min(bbox_inf, *itr);
        bbox_sup = vmath::max(bbox_sup, *itr);
        check_image.set((*itr).x(), (*itr).y(), check_val);
    }
    DEBUG("bounding box corners are "<<bbox_inf<<", "<<bbox_sup);

    // coutour bbox subimage (to perform integral on)
    physD intg_image = check_image.sub(bbox_inf.x(), bbox_inf.y(), bbox_sup.x()-bbox_inf.x()+1, bbox_sup.y()-bbox_inf.y()+1);
    intg_image.set_origin(iimage.get_origin()-bbox_inf);

    // integrate by scanline fill
    double intg=0, intg_sq=0;
    std::list<vec2i> up_pl, scan_pl, tmplist;

    // starting point needs to be INSIDE the contour. If origin is not set (i.e. 0:0)
    // the center of the bbox is a good starting point.

    vec2i iimage_orig = iimage.get_origin();
    vec2i intg_orig = intg_image.get_origin();

    bool orig_is_inside = (
                intg_orig.x()<=bbox_sup.x() &&
                intg_orig.y()<=bbox_sup.y() &&
                intg_orig.x()>=bbox_inf.x() &&
                intg_orig.y()>=bbox_inf.y()) ? true : false;
    vec2i starting_point;

    if (iimage_orig == vec2i(0,0) || !orig_is_inside) {
        starting_point = vec2i(0.5*(bbox_sup-bbox_inf));
        DEBUG("origin is not set: recalculating");
    } else {
        DEBUG("origin is set: using it as starting point");
        starting_point = intg_orig;
    }
    DEBUG("starting point is "<<starting_point);


    for (int xx=starting_point.x(); intg_image.point(xx, starting_point.y()) != check_val; xx++) {
        up_pl.push_back(vec2i(xx, starting_point.y()));
    }

    for (int xx=starting_point.x(); intg_image.point(xx, starting_point.y()) != check_val; xx--) {
        up_pl.push_front(vec2i(xx, starting_point.y()));
    }



    //std::cerr<<"walk starting from "<<up_pl.front()<<" to "<<up_pl.back()<<std::endl;

    int line_check =0;

    // the maximum possible surface w/in a given contour is the circular case;
    // 20% for additional safety
    int safety_counter = 0;
    int safety_counter_max = 1.2*((contour.size()*contour.size())/(4*3.14));


    while (!up_pl.empty() && (safety_counter<safety_counter_max)) {

        //scan_pl = up_pl;
        //up_pl.clear();

        tmplist.clear();
        scan_pl.clear();
        std::list<vec2i>::iterator itr = up_pl.begin(), itrf = up_pl.begin();
        itrf++;

        while (itrf != up_pl.end()) {
            if (((*itrf).x()-(*itr).x()) > 1) {
                //std::cerr<<"separation at "<<*itr<<" -- "<<*itrf<<std::endl;
                scan_pl.push_back(*itr);
                tmplist.clear();
                tmplist.push_back(*itrf);


                //std::cerr<<"line "<<line_check<<": sep/walk starting from "<<*itr<<" to "<<*itrf<<std::endl;
                vec2i ref_sx = *itr, ref_dx = *itrf;
                while (intg_image.point(scan_pl.back(), check_val) != check_val) {
                    ref_sx+=vec2i(1,0);
                    scan_pl.push_back(ref_sx);
                }
                //std::cerr<<"line "<<line_check<<": sep/walk starting from "<<scan_pl.back()<<" to "<<tmplist.front()<<std::endl;
                while (intg_image.point(tmplist.front(), check_val) != check_val) {
                    ref_dx -= vec2i(1,0);
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
            scan_pl.push_front(scan_pl.front()+vec2i(-1, 0));
            //std::cerr<<"--------------"<<intg_image.getPoint(scan_pl.front())<<std::endl;
        }

        while (intg_image.point(scan_pl.back(), check_val) != check_val) {
            scan_pl.push_back(scan_pl.back()+vec2i(1, 0));
            //std::cerr<<"--------------"<<intg_image.point(scan_pl.back(), check_val)<<std::endl;
        }

        //std::cerr<<"line "<<line_check<<": walk starting from "<<scan_pl.front()<<" to "<<scan_pl.back()<<std::endl;
        //if (line_check == 38)
        //	break;

        up_pl.clear();

        while (!scan_pl.empty()) {
            vec2i pp = scan_pl.front();
            scan_pl.pop_front();
            if (intg_image.point(pp, check_val) != check_val) {
                intg+=intg_image.point(pp);
                intg_sq+=pow(intg_image.point(pp), 2);
                points_count++;

                intg_image.set(pp.x(), pp.y(), check_val);

                up_pl.push_back(vec2i(pp.x(), pp.y()+1));
                up_pl.push_back(vec2i(pp.x(), pp.y()-1));
                //std::cerr<<"point read: "<<pp<<std::endl;
            }
            //else std::cerr<<"--------------- cippacazzo ------------------"<<pp<<std::endl;
        }

        line_check++;

        safety_counter++;

    }

    if (safety_counter >= safety_counter_max) {
        DEBUG("Maximum recursion reached, exit forced, integration failed");
        return std::list<double>();
    }

    DEBUG(5, "contour integral: "<<intg);
    if (integrate_boundary) {
        // add boundary points
        double bps = 0, bps_sq = 0;
        for (std::list<vec2i>::iterator itr = contour.begin(); itr != contour.end(); ++itr) {
            bps+= iimage.point((*itr).x(), (*itr).y());
            bps_sq+= pow(iimage.point((*itr).x(), (*itr).y()), 2);
            points_count++;
        }
        DEBUG(5, "boundary points account for "<<bps);
        intg+=bps;
        intg_sq+=bps_sq;
    }

    std::list<double> ret;
    ret.push_front(points_count);
    ret.push_front(intg);
    ret.push_front(intg_sq);
    return ret;

}

template<> void
physC::TscanBrightness() {
    return;
}


/*!
 * @}
 */
