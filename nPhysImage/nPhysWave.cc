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

// sub-step position of the maximum of the filter response: parabola on log|z|^2 through the best filter
// and its two neighbours (in units of the scan step, within +-0.5). Values <=0 mean "no neighbour" -> 0
static double wavelet_parabolic_offset(double ym, double y0, double yp) {
    if (ym<=0 || y0<=0 || yp<=0) return 0;
    ym=log(ym);
    y0=log(y0);
    yp=log(yp);
    double den=ym-2*y0+yp;
    if (den>=0) return 0;
    return std::clamp(0.5*(ym-yp)/den,-0.5,0.5);
}

// the lambda values of the scan: linear, or geometric (wavelet_params::log_lambda)
static bool wavelet_log_lambda(const physWave::wavelet_params &params) {
    return params.log_lambda && params.n_lambdas>1 && params.init_lambda>0 && params.end_lambda>0;
}
static std::vector<double> wavelet_lambdas(const physWave::wavelet_params &params) {
    std::vector<double> lambdas(params.n_lambdas);
    if (params.n_lambdas==1) {
        lambdas[0]=0.5*(params.end_lambda+params.init_lambda);
    } else if (wavelet_log_lambda(params)) {
        for (size_t n=0; n<params.n_lambdas; n++)
            lambdas[n] = params.init_lambda*pow(params.end_lambda/params.init_lambda, n/(double)(params.n_lambdas-1));
    } else {
        for (size_t n=0; n<params.n_lambdas; n++)
            lambdas[n] = params.init_lambda + n*(params.end_lambda-params.init_lambda)/(params.n_lambdas-1);
    }
    return lambdas;
}
// lambda of the grid value lam moved by offset steps (the parabolic refinement): a step is a ratio for a geometric scan
static double wavelet_lambda_step(const physWave::wavelet_params &params, const std::vector<double> &lambdas, double lam, double offset) {
    if (lambdas.size()<2) return lam;
    if (wavelet_log_lambda(params)) return lam*pow(lambdas.back()/lambdas.front(), offset/(lambdas.size()-1));
    return lam + offset*(lambdas.back()-lambdas.front())/(lambdas.size()-1);
}

// the FFT spreads a single NaN over the whole image: fill the non finite pixels (Laplace) on a copy of the data.
// Returns false if there is no finite pixel at all
static bool wavelet_finite_input(physD &data, physD &fdata) {
    fdata=data;
    std::vector<vec2i> holes;
    for (size_t j=0; j<data.getH(); j++) {
        for (size_t i=0; i<data.getW(); i++) {
            if (!std::isfinite(data.point(i,j))) holes.push_back(vec2i(i,j));
        }
    }
    if (holes.size()==data.getSurf()) return false;
    if (holes.size()>0) physMath::laplace_fill(fdata,holes);
    return true;
}

// half width (1/e) of the analysis window in pixels, the largest of the scan: damp*lambda across the fringes,
// thick along them
static double wavelet_window(const physWave::wavelet_params &params) {
    double lambda_max=std::max(params.init_lambda,params.end_lambda);
    return std::max(params.damp*lambda_max, params.thick);
}

// smallest size >= n with only the factors 2, 3 and 5: measured with FFTW, the cost per point is the same as for a
// power of 2 (which would need up to 2x more points per side), while a factor 7 is 1.6x slower per point
// (360 = 2^3 3^2 5 is faster than 350 = 2 5^2 7) and a large prime factor 2-3x slower
static int wavelet_fft_size(int n) {
    int best=1;
    while (best<n) best*=2;
    for (long p5=1; p5<=best; p5*=5) {
        for (long p35=p5; p35<=best; p35*=3) {
            long v=p35;
            while (v<n) v*=2;
            if (v<best) best=v;
        }
    }
    return best;
}

// the FFT is periodic: without padding the opposite borders touched. Padded by 2 windows on each side (the window
// is down to exp(-4) at the image), to a size with factors 2,3,5 (fast for FFTW, see wavelet_fft_size), filled
// with the mean of the image
static physD wavelet_pad(physD &fdata, double window, vec2i &offset) {
    const int W=fdata.getW(), H=fdata.getH();
    const int pad=(int)std::ceil(2.0*window);
    vec2i newSize(wavelet_fft_size(W+2*pad), wavelet_fft_size(H+2*pad));
    offset=vec2i((newSize.x()-W)/2, (newSize.y()-H)/2);
    const double mean=fdata.sum()/fdata.getSurf();
    physD padded(newSize.x(), newSize.y(), mean);
    padded.set_origin(fdata.get_origin()+offset);
//     auto taper=[pad](int d) { return d<=0 ? 1.0 : (d>=pad ? 0.0 : 0.5*(1.0+cos(M_PI*d/pad))); };
// #pragma omp parallel for
//     for (int j=0; j<newSize.y(); j++) {
//         for (int i=0; i<newSize.x(); i++) {
//             int x=i-offset.x(), y=j-offset.y();
//             int cx=std::clamp(x,0,W-1), cy=std::clamp(y,0,H-1);
//             double t=taper(std::abs(x-cx))*taper(std::abs(y-cy));
//             padded.set(i,j,mean+(fdata.point(cx,cy)-mean)*t);
//         }
//     }
    // (the border pixels extended along the normal and faded to the mean: for oblique fringes they make false
    // fringes in the padding, which biased the phase near the borders and even inside, 10x worse than the plain
    // mean in the tests: the step to the mean has a broad spectrum, weak at the carrier)
    // the mean outside the image
#pragma omp parallel for
    for (int j=0; j<H; j++) {
        for (int i=0; i<W; i++) {
            padded.set(i+offset.x(),j+offset.y(),fdata.point(i,j));
        }
    }
    DEBUG("wavelet padding " << W << "x" << H << " -> " << newSize.x() << "x" << newSize.y() << " (window " << window << ")");
    return padded;
}

void physWave::phys_wavelet_field_2D_morlet(wavelet_params &params)
{
    DEBUG(5,"start");


    physD fdata;
    if (params.data && (params.n_angles > 0) && (params.n_lambdas > 0) && (params.data->getSurf() != 0) && wavelet_finite_input(*params.data, fdata)) {

        params.olist.clear();

        // unsigned int dx=params.data->getW();
        // unsigned int dy=params.data->getH();
        // computed on the padded image, the results cropped back to the image at the end
        const unsigned int dx0=params.data->getW();
        const unsigned int dy0=params.data->getH();
        vec2i offset;
        fdata=wavelet_pad(fdata, wavelet_window(params), offset);
        unsigned int dx=fdata.getW();
        unsigned int dy=fdata.getH();

        unsigned int surf=dx*dy;

        std::vector<int> xx(dx), yy(dy);

        physC zz_morlet("zz_morlet");

        zz_morlet.resize(dx,dy);

        auto qmap     = std::make_unique<physD>(dx, dy, 0.0, "quality");
        auto wphase   = std::make_unique<physD>(dx, dy, 0.0, "phase_2pi");
        auto lambda   = std::make_unique<physD>(dx, dy, 0.0, "lambda");
        auto angle    = std::make_unique<physD>(dx, dy, 0.0, "angle");
        auto intensity= std::make_unique<physD>(dx, dy, 0.0, "intensity");


        std::vector<double> angles(params.n_angles), lambdas(params.n_lambdas);
        if (params.n_angles==1) {
            angles[0]=0.5*(params.end_angle+params.init_angle);
        } else {
            for (size_t nangle=0; nangle<params.n_angles; nangle++)
                angles[nangle] = params.init_angle + nangle*(params.end_angle-params.init_angle)/(params.n_angles-1);
        }
        lambdas=wavelet_lambdas(params);

        params.iter=0;
        *params.iter_ptr=0;


        fftw_complex *t = fftw_alloc_complex(surf);
        fftw_complex *Ft = fftw_alloc_complex(surf);
        fftw_complex *Ftmorlet = fftw_alloc_complex(surf);

        fftw_plan plan_fwd = fftw_plan_dft_2d(dy, dx, t, Ft, FFTW_FORWARD, FFTW_ESTIMATE);
        // fftw_plan plan_bwd = fftw_plan_dft_2d(dy, dx, t, Ftmorlet, FFTW_BACKWARD, FFTW_ESTIMATE);
        // the backward FFT runs once per filter: a measured plan is 1.3-3x faster (much more with fftw threads).
        // Measuring costs about one second for a large image but FFTW keeps it (wisdom): the next plans of the same
        // size are immediate, also with FFTW_ESTIMATE. FFTW_MEASURE overwrites t and Ftmorlet: plan before filling t
        size_t n_filters = params.n_angles*params.n_lambdas;
        fftw_plan plan_bwd = fftw_plan_dft_2d(dy, dx, t, Ftmorlet, FFTW_BACKWARD, n_filters >= 16 ? FFTW_MEASURE : FFTW_ESTIMATE);

#pragma omp parallel for
        for (size_t i = 0; i < surf; i++) {
            // t[i][0] = params.data->Timg_buffer[i];
            t[i][0] = fdata.Timg_buffer[i];
            t[i][1] = 0;
        }

        fftw_execute(plan_fwd);

        double damp_norm=params.damp*M_PI;
        // Morlet admissibility: subtract kappa*exp(-(damp_norm*xr*lambda_norm)^2) so the filter is zero at zero frequency
        // (otherwise the background leaks into the fringes for small damp). Normalized to keep gain 1 at the carrier
        double kappa = damp_norm>0 ? exp(-damp_norm*damp_norm) : 0;
        double norm_adm = 1.0/(1.0-kappa*kappa);
        // the filter is a narrow gaussian around the carrier: where all its exponents are below this limit
        // |gauss| < exp(-50) = 2e-22 (well below the double precision of the peak, 1) and it is set to 0
        // without computing the three exp
        double log_negligible = -50.0 - log(norm_adm);
        double log_kappa = damp_norm>0 ? -damp_norm*damp_norm : -std::numeric_limits<double>::infinity();

        // refinement of lambda and angle maps: for each pixel keep the response of the filters next to the best one
        // (lambda-1, lambda+1, angle-1, angle+1). The previous responses are kept for one row
        // of the inner scan, which is the shorter one between angles and lambdas
        bool anglesInner = angles.size() <= lambdas.size();
        size_t nInner = anglesInner ? angles.size() : lambdas.size();
        size_t nOuter = anglesInner ? lambdas.size() : angles.size();
        auto rowIdx = [&](size_t il, size_t ia) { return anglesInner ? ia : il; };
        std::vector<float> prevRow(nInner*surf, -1);
        std::vector<float> nearLambda(2*surf, -1), nearAngle(2*surf, -1); // [0..surf) minus, [surf..2surf) plus
        std::vector<int> bestL(surf,-1), bestA(surf,-1);

        // regularization (wavelet_params::n_candidates): the K best filters of each pixel (response, phase, indices)
        const size_t K = params.n_candidates>=2 ? params.n_candidates : 0;
        std::vector<float> candQ(K*surf, -1), candPh(K*surf, 0);
        std::vector<short> candL(K*surf, -1), candA(K*surf, -1);


        for (size_t o=0; o<nOuter; o++) {
            for (size_t n=0; n<nInner; n++) {
                size_t i = anglesInner ? o : n; // lambda index
                size_t j = anglesInner ? n : o; // angle index

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

                // double thick_norm=thickness[l]*M_PI/sqrt(pow(sr*dx,2)+pow(cr*dy,2));
                // double lambda_norm=lambdas[i]/sqrt(pow(cr*dx,2)+pow(sr*dy,2));
                // the filter is built on true spatial frequencies (cycles/px) before rotating:
                // rotating the FFT indices directly misses the carrier on non-square images (unless angle is 0 or 90)
                double thick_norm=params.thick*M_PI;
                double lambda_norm=lambdas[i];

// #pragma omp parallel for
//                     for (unsigned int k=0;k<surf;k++) {
//                         unsigned int x=k%dx;
//                         unsigned int y=k/dx;
//
//                         int xx=(x+(dx+1)/2)%dx-(dx+1)/2;
//                         int yy=(y+(dy+1)/2)%dy-(dy+1)/2;
//
//                         double fx = xx/(double)dx;
//                         double fy = yy/(double)dy;
//                         double xr = fx*cr - fy*sr; //rotate
//                         double yr = fx*sr + fy*cr;
//
//                         double e_x = -pow(damp_norm*(xr*lambda_norm-1.0), 2.);
//                         double e_y = -pow(yr*thick_norm, 2.);
//
//                         // double gauss = exp(e_x)*exp(e_y);
//                         double gauss = norm_adm*(exp(e_x)-kappa*exp(-pow(damp_norm*xr*lambda_norm, 2.)))*exp(e_y);
//
//                         t[k][0]=Ft[k][0]*gauss;
//                         t[k][1]=Ft[k][1]*gauss;
//
//                     }
#pragma omp parallel for
                for (unsigned int y=0;y<dy;y++) {
                    int yy=(y+(dy+1)/2)%dy-(dy+1)/2;
                    double fy = yy/(double)dy;
                    for (unsigned int x=0;x<dx;x++) {
                        size_t k=(size_t)y*dx+x;
                        int xx=(x+(dx+1)/2)%dx-(dx+1)/2;
                        double fx = xx/(double)dx;
                        double xr = fx*cr - fy*sr; //rotate
                        double yr = fx*sr + fy*cr;

                        double e_x = -pow(damp_norm*(xr*lambda_norm-1.0), 2.);
                        double e_y = -pow(yr*thick_norm, 2.);
                        double e_k = -pow(damp_norm*xr*lambda_norm, 2.); // admissibility term

                        double gauss = 0;
                        // |gauss| <= norm_adm*max(exp(e_x),kappa*exp(e_k))*exp(e_y)
                        if (e_y > log_negligible && (e_x > log_negligible || e_k+log_kappa > log_negligible)) {
                            gauss = norm_adm*(exp(e_x)-kappa*exp(e_k))*exp(e_y);
                        }

                        t[k][0]=Ft[k][0]*gauss;
                        t[k][1]=Ft[k][1]*gauss;
                    }
                }

                fftw_execute(plan_bwd);

                // decision
#pragma omp parallel for
                for (size_t k=0; k<surf; k++) {
                    double qmap_local=pow(Ftmorlet[k][0],2)+pow(Ftmorlet[k][1],2);
                    if (bestA[k]==(int)j && bestL[k]+1==(int)i) nearLambda[surf+k]=qmap_local;
                    if (bestL[k]==(int)i && bestA[k]+1==(int)j) nearAngle[surf+k]=qmap_local;
                    if ( qmap_local > qmap->Timg_buffer[k]) {
                        qmap->Timg_buffer[k] = qmap_local;
                        wphase->Timg_buffer[k] = atan2(Ftmorlet[k][1], Ftmorlet[k][0]);
                        lambda->Timg_buffer[k] = lambdas[i];
                        angle->Timg_buffer[k] = angles[j];
                        bestL[k]=i;
                        bestA[k]=j;
                        nearLambda[k] = i>0 ? prevRow[rowIdx(i-1,j)*surf+k] : -1;
                        nearAngle[k]  = j>0 ? prevRow[rowIdx(i,j-1)*surf+k] : -1;
                        nearLambda[surf+k] = -1;
                        nearAngle[surf+k] = -1;
                    }
                    prevRow[n*surf+k]=qmap_local;
                    if (K) { // replaces the weakest of the K candidates
                        size_t kmin=k*K;
                        for (size_t c=k*K+1; c<(k+1)*K; c++) if (candQ[c]<candQ[kmin]) kmin=c;
                        if (qmap_local>candQ[kmin]) {
                            candQ[kmin]=qmap_local;
                            candPh[kmin]=atan2(Ftmorlet[k][1], Ftmorlet[k][0]);
                            candL[kmin]=i;
                            candA[kmin]=j;
                        }
                    }
                }

            }
            //! todo: this is awful: add exception?
            if ((*params.iter_ptr)==-1) {
                DEBUG("aborting");
                break;
            }
        }

        fftw_free(t);
        fftw_free(Ft);
        fftw_free(Ftmorlet);
        fftw_destroy_plan(plan_fwd);
        fftw_destroy_plan(plan_bwd);

        if ((*params.iter_ptr)!=-1) {

            double dLambda = lambdas.size()>1 ? (lambdas.back()-lambdas.front())/(lambdas.size()-1) : 0;
            double dAngle = angles.size()>1 ? (angles.back()-angles.front())/(angles.size()-1) : 0;

            // regularization: lambda and angle of the best filters smoothed over the neighbours (weighted by the
            // contrast; the angle as 2*angle: 0 and 180 are the same fringes), then each pixel takes the candidate
            // nearest to them (distance in steps of the scan). Without refinement for a candidate that is not the best
            if (K && (dLambda!=0 || dAngle!=0)) {
                physD wl(dx, dy, 0.0, "wl"), w(dx, dy, 0.0, "w"), wc(dx, dy, 0.0, "wc"), ws(dx, dy, 0.0, "ws");
                std::vector<double> lam;
                for (size_t k=0; k<surf; k++) {
                    if (!(qmap->Timg_buffer[k]>0) || bestL[k]<0) continue;
                    double q=sqrt(qmap->Timg_buffer[k]), a2=2*angles[bestA[k]]*_phys_deg;
                    wl.Timg_buffer[k]=q*lambdas[bestL[k]];
                    w.Timg_buffer[k]=q;
                    wc.Timg_buffer[k]=q*cos(a2);
                    ws.Timg_buffer[k]=q*sin(a2);
                    lam.push_back(lambdas[bestL[k]]);
                }
                if (lam.size()) {
                    auto med=lam.begin()+lam.size()/2;
                    std::nth_element(lam.begin(), med, lam.end());
                    double sigma=*med; // one interfringe
                    for (auto *img : {&wl, &w, &wc, &ws}) physMath::phys_fast_gaussian_blur(*img, sigma);
#pragma omp parallel for
                    for (size_t k=0; k<surf; k++) {
                        if (!(w.Timg_buffer[k]>0) || bestL[k]<0) continue;
                        double lamS=wl.Timg_buffer[k]/w.Timg_buffer[k];
                        double angS=0.5*atan2(ws.Timg_buffer[k], wc.Timg_buffer[k])/_phys_deg;
                        size_t cbest=k*K;
                        double dbest=std::numeric_limits<double>::infinity();
                        bool found=false;
                        for (size_t c=k*K; c<(k+1)*K; c++) {
                            if (candQ[c]<0) continue;
                            double dl = dLambda==0 ? 0 : wavelet_log_lambda(params) ?
                                        log(lambdas[candL[c]]/lamS)/log(pow(lambdas.back()/lambdas.front(), 1.0/(lambdas.size()-1))) :
                                        (lambdas[candL[c]]-lamS)/dLambda;
                            double da = dAngle!=0 ? remainder(angles[candA[c]]-angS, 180.0)/dAngle : 0;
                            double d=dl*dl+da*da;
                            if (d<dbest || (d==dbest && candQ[c]>candQ[cbest])) { dbest=d; cbest=c; found=true; }
                        }
                        if (found && (candL[cbest]!=bestL[k] || candA[cbest]!=bestA[k])) {
                            qmap->Timg_buffer[k]=candQ[cbest];
                            wphase->Timg_buffer[k]=candPh[cbest];
                            lambda->Timg_buffer[k]=lambdas[candL[cbest]];
                            angle->Timg_buffer[k]=angles[candA[cbest]];
                                nearLambda[k]=nearLambda[surf+k]=nearAngle[k]=nearAngle[surf+k]=-1;
                        }
                    }
                }
            }

#pragma omp parallel for
            // for (size_t k=0; k<params.data->getSurf(); k++) {
            for (size_t k=0; k<surf; k++) { // padded
                lambda->Timg_buffer[k] = wavelet_lambda_step(params, lambdas, lambda->Timg_buffer[k], wavelet_parabolic_offset(nearLambda[k], qmap->Timg_buffer[k], nearLambda[surf+k]));
                angle->Timg_buffer[k] += dAngle*wavelet_parabolic_offset(nearAngle[k], qmap->Timg_buffer[k], nearAngle[surf+k]);
                qmap->Timg_buffer[k]=sqrt(qmap->Timg_buffer[k])/surf;
                // intensity->Timg_buffer[k] = params.data->Timg_buffer[k] - 2.0 * qmap->Timg_buffer[k]*cos(wphase->Timg_buffer[k]);
                intensity->Timg_buffer[k] = fdata.Timg_buffer[k] - 2.0 * qmap->Timg_buffer[k]*cos(wphase->Timg_buffer[k]); // NaN would spread in the blur
                wphase->Timg_buffer[k]/=2*M_PI;
            }

            // the subtraction leaves ~1% of the fringes (phase and contrast errors): the blur cleans them and the
            // noise. sigma = smallest lambda/2 (the carrier of lambda_min damped to < 1%): a larger sigma (e.g.
            // lambda_max/2) is no better than blurring the data itself and smooths the real structure of the intensity
            physMath::phys_fast_gaussian_blur(*intensity,std::min(params.init_lambda,params.end_lambda)/2.0);

            // fringe visibility (Imax-Imin)/(Imax+Imin)
            auto visibility = std::make_unique<physD>(dx, dy, 0.0, "visibility");
#pragma omp parallel for
            for (size_t k=0; k<surf; k++) {
                visibility->Timg_buffer[k] = intensity->Timg_buffer[k]>0 ? 2.0*qmap->Timg_buffer[k]/intensity->Timg_buffer[k] : std::numeric_limits<double>::quiet_NaN();
            }

            params.olist["visibility"] = std::move(visibility);
            params.olist["phase_2pi"] = std::move(wphase);
            params.olist["contrast"]  = std::move(qmap);
            params.olist["lambda"]    = std::move(lambda);
            params.olist["angle"]     = std::move(angle);
            params.olist["intensity"] = std::move(intensity);

            for(auto itr = params.olist.begin(); itr != params.olist.end(); ++itr) {
                *itr->second=itr->second->sub(offset.x(), offset.y(), dx0, dy0); // back to the image (without padding)
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

// ---------------------- thread transport functions ------------------------

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
            case SIMPLE_HV : // rows, then the columns starting from the unwrapped first row
                unwrap::simple_h(wphase, uphase);
                unwrap::simple_v(wphase, uphase, true);
                break;

            case SIMPLE_VH :
                unwrap::simple_v(wphase, uphase);
                unwrap::simple_h(wphase, uphase, true);
                break;

            case GOLDSTEIN : // quality: order of the flood fill, <=0 are cuts (barrier)
                unwrap::goldstein(wphase, uphase, quality);
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

// the wavelet model is data = intensity + 2 contrast cos(phase): with the intensity it rebuilds the data,
// without it the fringes are drawn on a background 2 contrast
physD physWave::phys_synthetic_interferogram (physD *phase_over_2pi, physD *quality, physD *intensity){
    physD synthetic;
    if (phase_over_2pi && quality) {
        if (phase_over_2pi->getW()==quality->getW() && phase_over_2pi->getH()==quality->getH()) {
            if (intensity && (intensity->getW()!=phase_over_2pi->getW() || intensity->getH()!=phase_over_2pi->getH())) intensity=nullptr;
            synthetic.resize(phase_over_2pi->getW(),phase_over_2pi->getH());
#pragma omp parallel for
            for (size_t ii=0; ii<phase_over_2pi->getSurf(); ii++) {
                // synthetic.set(ii,M_PI*quality->point(ii)*(1.0+cos(phase_over_2pi->point(ii)*2*M_PI)));
                double fringes=2.0*quality->point(ii)*cos(phase_over_2pi->point(ii)*2*M_PI);
                synthetic.set(ii, (intensity ? intensity->point(ii) : 2.0*quality->point(ii)) + fringes);
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

        for (long unsigned int i=0; i<dx/2+1; i++) {
            for (long unsigned int j=0; j<dy; j++) {
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
                    for (long unsigned int ii=i-1;ii<=i+1;ii++) {
                        DEBUG("here"<< j-1 << " " << j+1);
                        for (long unsigned int jj=j-1;jj<=j+1;jj++) {
                            DEBUG("here2");
                            int iiclean=(ii+dx/2+1)%(dx/2+1);
                            int jjclean=(jj+dy)%(dy);
                            int j1clean=(dy/2+1)-(jjclean+dy/2+1)%dy;
                            int kk=k=iiclean+jjclean*(dx/2+1);
                            double rclean=sqrt(iiclean*iiclean+j1clean*j1clean);
                            vec2f my_l_a=lambda_angle(iiclean,j1clean,dx,dy);
                            double my_weight=pow(rclean,weight)*vec2f(myData[kk][0],myData[kk][1]).mod();
                            DEBUG(my_l_a << " " << my_weight);
                            my_vec.emplace_back(my_l_a, my_weight);
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
    invimage.prop["unitsCB"]="C m-3";
}


//! General function for Abel inversion
// inversion of the two profiles of one axis point (each starts on the axis and goes to the image border, padded
// with its last value); symmetric: the average of the two sides over the length both have, written on both sides
template <class F>
static void phys_invert_abel_point(physWave::abel_params &params, size_t ii, enum phys_direction sym_idx, enum phys_direction inv_idx,
                                   std::vector<double> &neg_buffer, std::vector<double> &pos_buffer, std::vector<double> &out_buffer, F invert_1D)
{
    size_t integral_size = neg_buffer.size();
    unsigned int axe_point[2] = { (unsigned int) params.iaxis[ii].x(), (unsigned int) params.iaxis[ii].y() };

    size_t neg_len = params.iimage->get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &neg_buffer[0], integral_size, PHYS_NEG);
    size_t pos_len = params.iimage->get_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &pos_buffer[0], integral_size, PHYS_POS);

    if (params.symmetric) {
        size_t len = std::min(neg_len, pos_len);
        for (size_t j=0; j<len; j++)
            neg_buffer[j] = 0.5*(neg_buffer[j]+pos_buffer[j]);
        for (size_t j=len; j<integral_size; j++)
            neg_buffer[j] = neg_buffer[len-1];	// boundary normalization
        invert_1D(neg_buffer, out_buffer);
        params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &out_buffer[0], integral_size, PHYS_NEG);
        params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &out_buffer[0], integral_size, PHYS_POS);
        return;
    }

    for (size_t j=neg_len; j<integral_size; j++)
        neg_buffer[j] = neg_buffer[neg_len-1];	// boundary normalization
    invert_1D(neg_buffer, out_buffer);
    double axis_val = 0.5*out_buffer[0];
    params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &out_buffer[0], integral_size, PHYS_NEG);

    for (size_t j=pos_len; j<integral_size; j++)
        pos_buffer[j] = pos_buffer[pos_len-1];	// boundary normalization
    invert_1D(pos_buffer, out_buffer);
    axis_val += 0.5*out_buffer[0];
    params.oimage->set_Tvector(inv_idx, axe_point[sym_idx], axe_point[inv_idx], &out_buffer[0], integral_size, PHYS_POS);

    // set on axis value
    params.oimage->set(params.iaxis[ii].x(), params.iaxis[ii].y(), axis_val);
}

void physWave::phys_invert_abel(abel_params &params)
{

    if (params.iimage->getSurf() == 0) {
        return;
    }

    params.oimage = std::make_unique<physD>(params.iimage->getW(), params.iimage->getH(), 0.0, "Inverted");
    params.oimage->set_origin(params.iimage->get_origin());
    params.oimage->set_scale(params.iimage->get_scale());

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

    size_t integral_size = params.iimage->getSizeByIndex(inv_idx);

    *params.iter_ptr = 0;
    // 2. switch on algo

    if (params.ialgo == ABEL) {
        DEBUG(1, "Plain ABEL inversion" << (params.symmetric ? " (symmetric)" : ""));
#pragma omp parallel for shared (params)
        for (size_t ii = 0; ii<params.iaxis.size(); ii++) {
            if ((*params.iter_ptr)!=-1) {

#pragma omp atomic
                (*params.iter_ptr)++;

                std::vector<double> neg_buffer(integral_size), pos_buffer(integral_size), out_buffer(integral_size);
                phys_invert_abel_point(params, ii, sym_idx, inv_idx, neg_buffer, pos_buffer, out_buffer,
                                       [](std::vector<double> &in, std::vector<double> &out) { phys_invert_abel_1D(in, out); });
            }
        }

        params.oimage->setName(std::string("ABEL ")+params.oimage->getName());
        params.oimage->setShortName("ABEL");

    } else if (params.ialgo == ABEL_HF) {

        DEBUG(1, "Hankel-Fourier implementation of ABEL inversion" << (params.symmetric ? " (symmetric)" : ""));
        // testing purposes only! Many problems in this one:
        // 1. code copy
        // 2. lut optimization not working properly
        // 3. should convert H_0 to FT
        std::vector<double> my_lut;

        //! by fixing integral_size as a single vector size for transformation, results in padding
        //! hence in a modification of the image resolution

        // the plan is made on the first input buffer: the same buffers for all the points
        std::vector<double> neg_buffer(integral_size), pos_buffer(integral_size), out_buffer(integral_size), copy_buffer(integral_size);

        std::vector<double> Fivec;
        fftw_plan r2rplan=nullptr;

        for (size_t ii = 0; ii<params.iaxis.size(); ii++) {
            if ((*params.iter_ptr)!=-1) {
                (*params.iter_ptr)++;

                phys_invert_abel_point(params, ii, sym_idx, inv_idx, neg_buffer, pos_buffer, out_buffer,
                                       [&](std::vector<double> &in, std::vector<double> &out) {
                                           copy_buffer = in;
                                           phys_invert_abelHF_1D(copy_buffer, out, my_lut, Fivec, r2rplan);
                                       });

                DEBUG(10,"step: "<<ii);
            }
        }

        if (r2rplan) fftw_destroy_plan(r2rplan);

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
