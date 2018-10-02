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
    my_w.setupUi(this);
    my_w.direction->setIcon(my_w.direction->style()->standardIcon(QStyle::SP_ArrowRight));
    show();
}

void Spectral_analysis::on_direction_toggled(bool val) {
    my_w.direction->setIcon(val? my_w.direction->style()->standardIcon(QStyle::SP_ArrowLeft) : my_w.direction->style()->standardIcon(QStyle::SP_ArrowRight));
}

void Spectral_analysis::on_calculate_released() {
    saveDefaults();
    int kind = my_w.spectral_transform->currentIndex();
    nPhysD *image=getPhysFromCombo(my_w.image);
    phys_fft dir = my_w.direction->isChecked() ? PHYS_BACKWARD :PHYS_FORWARD;
    if (image) {
        physC ft;
        physC temp_complex;
        if (my_w.useImaginary->isChecked()) {
            nPhysD *imaginary=getPhysFromCombo(my_w.imaginary);
            if (imaginary) {
                temp_complex= physMath::from_real_imaginary(*image, *imaginary);
            }
        } else {
            temp_complex= physMath::from_real(*image);
        }

        // ftshift before if backward
        if (my_w.doshift_cb->isChecked() && dir==PHYS_BACKWARD) {
            switch (kind) {
            case 0: temp_complex = physMath::ftshift1(temp_complex, PHYS_X); break;   // 1D horizontal
            case 1: temp_complex = physMath::ftshift1(temp_complex, PHYS_Y); break;   // 1D vertical
            case 2: temp_complex = physMath::ftshift2(temp_complex); break;           // 2D
            }
        }

        switch (kind) {
        case 0: ft = temp_complex.ft1(PHYS_X,dir); break;     // 1D horizontal
        case 1: ft = temp_complex.ft1(PHYS_Y,dir); break;     // 1D vertical
        case 2: ft = temp_complex.ft2(dir); break;           // 2D
        }

        // ftshift after if forward
        if (my_w.doshift_cb->isChecked()&& dir==PHYS_FORWARD) {
            switch (kind) {
            case 0: ft = physMath::ftshift1(ft, PHYS_X); break;   // 1D horizontal
            case 1: ft = physMath::ftshift1(ft, PHYS_Y); break;   // 1D vertical
            case 2: ft = physMath::ftshift2(ft); break;           // 2D
            }
        }

        if (my_w.normalize->isChecked()) {
            physMath::phys_divide(ft, sqrt(image->getSurf()));
        }

        std::map<std::string, physD> omap;
        switch (my_w.output_format->currentIndex()) {
        case 0: omap = physMath::to_polar(ft); break; // polar
        case 1: omap = physMath::to_rect(ft); break; // rectangular
        case 2: omap = physMath::to_powersp(ft, false); break; // power spectrum linear
        case 3: omap = physMath::to_powersp(ft, true); break;// power spectrum log10
        }

        for ( auto& itr : omap) {
            nPhysD *perm = new nPhysD(itr.second);
            perm->TscanBrightness();
            perm->setShortName(itr.first);
            perm->setName(itr.first);
            nparent->addShowPhys( perm );
        }
    }
}

