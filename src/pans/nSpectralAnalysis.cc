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

	connect(my_w.ft2d_button, SIGNAL(clicked()), this, SLOT(calculate_ft2()));
	connect(my_w.ft1dv_button, SIGNAL(clicked()), this, SLOT(calculate_ft1()));
	connect(my_w.ft1dh_button, SIGNAL(clicked()), this, SLOT(calculate_ft1()));
}

void
nSpectralAnalysis::calculate_ft2()
{
	nPhysD *cur = nparent->getBuffer(-1);
	if (!cur) 
		return;
		
	nPhysC ft = cur->ft2();

	if (my_w.doshift_cb->isChecked()) {
		ft = ftshift2(ft);	// check for leaks here!!
	}

	std::map<string, nPhysD> omap;
	std::map<string, nPhysD>::iterator itr;

	if (my_w.ftoutput_polar->isChecked()) {
		omap = to_polar(ft);
	} else if (my_w.ftoutput_rectangular->isChecked()) {
		omap = to_rect(ft);
	} else if (my_w.ftoutput_power->isChecked()) {
		omap = to_powersp(ft);
	} else return;

	for (itr = omap.begin(); itr != omap.end(); itr++) {
		nPhysD *perm = new nPhysD;
		*perm = itr->second;
		perm->setName(itr->first);

		nparent->showPhys( perm );
	}


	
}

void
nSpectralAnalysis::calculate_ft1()
{
	nPhysD *cur = nparent->getBuffer(-1);
	if (!cur) 
		return;

	enum phys_direction fdir = PHYS_X;
	if (sender() == my_w.ft1dv_button) {
		fdir = PHYS_Y;
	}
		
	nPhysC ft = cur->ft1(fdir);

	if (my_w.doshift_cb->isChecked()) {
		ft = ftshift1(ft, fdir);	// check for leaks here!!
	}

	std::map<string, nPhysD> omap;
	std::map<string, nPhysD>::iterator itr;

	if (my_w.ftoutput_polar->isChecked()) {
		omap = to_polar(ft);
	} else if (my_w.ftoutput_rectangular->isChecked()) {
		omap = to_rect(ft);
	} else if (my_w.ftoutput_power->isChecked()) {
		omap = to_powersp(ft);
	} else return;

	for (itr = omap.begin(); itr != omap.end(); itr++) {
		nPhysD *perm = new nPhysD;
		*perm = itr->second;
		perm->setName(itr->first);

		nparent->showPhys( perm );
	}


}
