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
#include "nSpectralAnalysis.h"
#include "neutrino.h"

nSpectralAnalysis::nSpectralAnalysis(neutrino *nparent, QString winname)
	: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);
	decorate();
}

void nSpectralAnalysis::on_calculate_released() {
    int kind = my_w.spectral_transform->currentIndex();
    nPhysD *image=getPhysFromCombo(my_w.image);

    if (image) {
        nPhysC ft;

        switch (kind) {
        case 0: ft = image->ft1(PHYS_X); break;     // 1D horizontal
        case 1: ft = image->ft1(PHYS_Y); break;     // 1D vertical
        case 2: ft = image->ft2(); break;           // 2D
        }

        // ftshift
        if (my_w.doshift_cb->isChecked()) {
            switch (kind) {
            case 0: ft = ftshift1(ft, PHYS_X); break;   // 1D horizontal
            case 1: ft = ftshift1(ft, PHYS_Y); break;   // 1D vertical
            case 2: ft = ftshift2(ft); break;           // 2D
            }
        }


        std::map<string, nPhysD> omap;
        switch (my_w.output_format->currentIndex()) {
        case 0: omap = to_polar(ft); break; // polar
        case 1: omap = to_rect(ft); break; // rectangular
        case 2: omap = to_powersp(ft, false); break; // power spectrum linear
        case 3: omap = to_powersp(ft, true); break;// power spectrum log10
        }

        for (std::map<string, nPhysD>::iterator itr = omap.begin(); itr != omap.end(); itr++) {
            nPhysD *perm = new nPhysD;
            *perm = itr->second;
            perm->TscanBrightness();
            perm->setName(itr->first);
            nparent->showPhys( perm );
        }
    }
}

