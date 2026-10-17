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
#include "Spectral_analysis.h"
#include "neutrino.h"

Spectral_analysis::Spectral_analysis(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);
    direction->setIcon(direction->style()->standardIcon(QStyle::SP_ArrowRight));
    show();
}

void Spectral_analysis::on_direction_toggled(bool val) {
    direction->setIcon(val? direction->style()->standardIcon(QStyle::SP_ArrowLeft) : direction->style()->standardIcon(QStyle::SP_ArrowRight));
}

// void Spectral_analysis::on_calculate_released() {
// (released: also without a click, and not with the keyboard)
void Spectral_analysis::on_calculate_clicked() {
    saveDefaults();
    int kind = spectral_transform->currentIndex();
    nPhysD *my_phys=getPhysFromCombo(image);
    phys_fft dir = direction->isChecked() ? PHYS_BACKWARD :PHYS_FORWARD;
    if (!my_phys) {
        statusBar()->showMessage(tr("Select an image"),5000);
        return;
    }
    if (my_phys) {
        physC ft;
        physC temp_complex;
        if (useImaginary->isChecked()) {
            nPhysD *my_phys_img=getPhysFromCombo(imaginary);
            // if (my_phys_img) {
            //     temp_complex= physMath::from_real_imaginary(*my_phys, *my_phys_img);
            // }
            // (a missing image or another size gave an empty complex matrix: empty images were added)
            if (!my_phys_img) {
                statusBar()->showMessage(tr("Select the imaginary part"),5000);
                return;
            }
            if (my_phys_img->getW()!=my_phys->getW() || my_phys_img->getH()!=my_phys->getH()) {
                statusBar()->showMessage(tr("Real and imaginary parts must have the same size"),5000);
                return;
            }
            temp_complex= physMath::from_real_imaginary(*my_phys, *my_phys_img);
        } else {
            temp_complex= physMath::from_real(*my_phys);
        }

        // ftshift before if backward
        // (the inverse shift: the zero frequency goes back from the centre to (0,0), also for odd sizes)
        if (doshift_cb->isChecked() && dir==PHYS_BACKWARD) {
            switch (kind) {
            // case 0: temp_complex = physMath::ftshift1(temp_complex, PHYS_X); break;   // 1D horizontal
            // case 1: temp_complex = physMath::ftshift1(temp_complex, PHYS_Y); break;   // 1D vertical
            // case 2: temp_complex = physMath::ftshift2(temp_complex); break;           // 2D
            case 0: temp_complex = physMath::iftshift1(temp_complex, PHYS_X); break;   // 1D horizontal
            case 1: temp_complex = physMath::iftshift1(temp_complex, PHYS_Y); break;   // 1D vertical
            case 2: temp_complex = physMath::iftshift2(temp_complex); break;           // 2D
            }
        }

        switch (kind) {
        case 0: ft = temp_complex.ft1(PHYS_X,dir); break;     // 1D horizontal
        case 1: ft = temp_complex.ft1(PHYS_Y,dir); break;     // 1D vertical
        case 2: ft = temp_complex.ft2(dir); break;           // 2D
        }

        // band-pass filter (the widgets were there, but not used): frequencies |f| between min and max
        // (max 0: no upper limit), in the units of the frequency axes 1/(N dx); on the unshifted spectrum
        physC filtered;
        if (useFilter->isChecked()) {
            if (dir==PHYS_BACKWARD) {
                statusBar()->showMessage(tr("The filter works on the forward transform"),5000);
                return;
            }
            const int W=ft.getW(), H=ft.getH();
            const double dx=my_phys->get_scale().x(), dy=my_phys->get_scale().y();
            const double minX=filterMinX->value(), maxX=filterMaxX->value(), minY=filterMinY->value(), maxY=filterMaxY->value();
            // frequency of the index k of N points (k > N/2 are the negative ones)
            auto freq=[](int k, int N, double d) { return std::fabs((k <= N/2 ? k : k-N)/(N*d)); };
            auto inside=[](double f, double fmin, double fmax) { return f >= fmin && (fmax <= 0 || f <= fmax); };
#pragma omp parallel for
            for (int y=0; y<H; y++) {
                for (int x=0; x<W; x++) {
                    bool keep = (kind==1 || inside(freq(x,W,dx),minX,maxX)) && (kind==0 || inside(freq(y,H,dy),minY,maxY));
                    if (!keep) ft.Timg_matrix[y][x]=mcomplex(0.,0.);
                }
            }
            // the filtered image: backward transform of the filtered spectrum, divided by the number of points
            switch (kind) {
            case 0: filtered = ft.ft1(PHYS_X,PHYS_BACKWARD); break;
            case 1: filtered = ft.ft1(PHYS_Y,PHYS_BACKWARD); break;
            case 2: filtered = ft.ft2(PHYS_BACKWARD); break;
            }
            physMath::phys_divide(filtered, (double)(kind==0 ? W : (kind==1 ? H : W*H)));
        }

        // ftshift after if forward
        if (doshift_cb->isChecked()&& dir==PHYS_FORWARD) {
            switch (kind) {
            case 0: ft = physMath::ftshift1(ft, PHYS_X); break;   // 1D horizontal
            case 1: ft = physMath::ftshift1(ft, PHYS_Y); break;   // 1D vertical
            case 2: ft = physMath::ftshift2(ft); break;           // 2D
            }
        }

        // if (normalize->isChecked()) {
        //     physMath::phys_divide(ft, sqrt(my_phys->getSurf()));
        // }
        // unitary: the number of points of the transform (W*H was used also for the 1D transforms)
        if (normalize->isChecked()) {
            double npoints = kind==0 ? my_phys->getW() : (kind==1 ? my_phys->getH() : my_phys->getSurf());
            physMath::phys_divide(ft, sqrt(npoints));
        }

        std::map<std::string, physD> omap;
        switch (output_format->currentIndex()) {
        case 0: omap = physMath::to_polar(ft); break; // polar
        case 1: omap = physMath::to_rect(ft); break; // rectangular
        case 2: omap = physMath::to_powersp(ft, false); break; // power spectrum linear
        case 3: omap = physMath::to_powersp(ft, true); break;// power spectrum log10
        }

        // the transform in the names (all the results were "rho", "theta"...)
        QString label = QString(dir==PHYS_BACKWARD ? "iFT" : "FT") + (kind==0 ? "1x" : (kind==1 ? "1y" : "2"));

        // forward with shift: frequency axes (origin at the zero frequency, scale 1/(N dx))
        bool freqAxes = doshift_cb->isChecked() && dir==PHYS_FORWARD;
        vec2f scale=my_phys->get_scale();
        vec2f fOrigin(kind==1 ? my_phys->get_origin().x() : (double)(my_phys->getW()/2),
                      kind==0 ? my_phys->get_origin().y() : (double)(my_phys->getH()/2));
        vec2f fScale(kind==1 ? scale.x() : 1.0/(my_phys->getW()*scale.x()),
                     kind==0 ? scale.y() : 1.0/(my_phys->getH()*scale.y()));

        for ( auto& itr : omap) {
            nPhysD *perm = new nPhysD(itr.second);
            perm->TscanBrightness();
            // perm->setShortName(itr.first);
            // perm->setName(itr.first);
            perm->setShortName((label+" "+QString::fromStdString(itr.first)).toStdString());
            perm->setName((label+"("+QString::fromStdString(my_phys->getName())+") "+QString::fromStdString(itr.first)).toStdString());
            perm->setFromName(my_phys->getFromName());
            if (freqAxes) {
                perm->set_origin(fOrigin);
                perm->set_scale(fScale);
            }
            nparent->addShowPhys( perm );
        }

        // the filtered image, with the origin and scale of the source (real part; also the imaginary one for a complex input)
        if (filtered.getSurf()>0) {
            std::map<std::string, physD> fmap=physMath::to_rect(filtered);
            for ( auto& itr : fmap) {
                if (itr.first=="imag" && !useImaginary->isChecked()) continue;
                nPhysD *perm = new nPhysD(itr.second);
                perm->TscanBrightness();
                QString part = useImaginary->isChecked() ? " "+QString::fromStdString(itr.first) : QString();
                perm->setShortName(("Filtered"+part).toStdString());
                perm->setName(("Filtered "+label+"("+QString::fromStdString(my_phys->getName())+")"+part).toStdString());
                perm->setFromName(my_phys->getFromName());
                perm->set_origin(my_phys->get_origin());
                perm->set_scale(my_phys->get_scale());
                nparent->addShowPhys( perm );
            }
        }
    }
}

