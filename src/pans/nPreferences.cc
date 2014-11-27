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

#include <unistd.h>

#include "nPreferences.h"
#include "neutrino.h"


#ifdef	__WIN32
#include <windows.h>
#endif

nPreferences::nPreferences(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname) {
	my_w.setupUi(this);

	int coreNum =1;
#ifdef	__WIN32
	SYSTEM_INFO sysinfo;
	GetSystemInfo( &sysinfo );

	coreNum = sysinfo.dwNumberOfProcessors;
#else
	coreNum=sysconf( _SC_NPROCESSORS_ONLN );
#endif

    int nthreads, procs, maxt, inpar, dynamic, nested;
#pragma omp parallel private(nthreads)
    {
        if (omp_get_thread_num() == 0) 
        {
            /* Get environment information */
            procs = omp_get_num_procs();
            nthreads = omp_get_num_threads();
            maxt = omp_get_max_threads();
            inpar = omp_in_parallel();
            dynamic = omp_get_dynamic();
            nested = omp_get_nested();
            
            /* Print environment information */
            my_w.infoCores->insertPlainText("Number of processors     = "+QString::number(procs));
            my_w.infoCores->insertPlainText("\nNumber of threads        = "+QString::number(nthreads));
            my_w.infoCores->insertPlainText("\nMax threads              = "+QString::number(maxt));
            my_w.infoCores->insertPlainText("\nIn parallel?             = "+QString(inpar==0?"No":"Yes"));
            my_w.infoCores->insertPlainText("\nDynamic threads enabled? = "+QString(dynamic==0?"No":"Yes"));
            my_w.infoCores->insertPlainText("\nNested supported?        = "+QString(nested==0?"No":"Yes"));
        }
    }
    
    
	my_w.threads->setMaximum(coreNum);

	if (coreNum==1) {
		my_w.threads->hide();
		my_w.labelThreads->hide();
	}
	if (!cudaEnabled()) {
		DEBUG("cuda not enabled");
		my_w.useCuda->setChecked(false);
		my_w.useCuda->setEnabled(false);
	} else {
		my_w.useCuda->setEnabled(true);
	}
	

	
	loadDefaults();
	decorate();
	
    DEBUG(nparent->my_w.toolBar->iconSize().width()/10-1);
    
    my_w.comboIconSize->setCurrentIndex(nparent->my_w.toolBar->iconSize().width()/10-1);

	connect(my_w.threads, SIGNAL(valueChanged(int)), this, SLOT(changeThreads(int)));
	connect(my_w.comboIconSize, SIGNAL(currentIndexChanged(int)), this, SLOT(changeIconSize(int)));
	connect(my_w.chooseFont, SIGNAL(pressed()), this, SLOT(changeFont()));
	connect(my_w.showDimPixel, SIGNAL(released()), this, SLOT(changeShowDimPixel()));

	my_w.labelFont->setFont(nparent->my_w.my_view->font());
	my_w.labelFont->setText(nparent->my_w.my_view->font().family()+" "+QString::number(nparent->my_w.my_view->font().pointSize()));
	
	connect(my_w.useDot, SIGNAL(released()), this, SLOT(useDot()));
	connect(my_w.askCloseUnsaved, SIGNAL(released()), this, SLOT(askCloseUnsaved()));
    
}

void nPreferences::changeThreads(int num) {
    if (num<=1) {
        fftw_cleanup_threads();
    } else {
        fftw_init_threads();
        fftw_plan_with_nthreads(num);
    }
    omp_set_num_threads(num);
    QSettings settings("neutrino","");
    settings.beginGroup("Preferences");
    settings.setValue("threads",num);
    settings.endGroup();
    DEBUG("THREADS THREADS THREADS THREADS THREADS THREADS " << num);
}

void nPreferences::useDot() {
    if (my_w.useDot->isChecked()) {
        QLocale::setDefault(QLocale(QLocale::English, QLocale::UnitedStates));
    } else {
        QLocale::setDefault(QLocale::system());
    }
    my_w.statusBar->showMessage(QLocale::countryToString(QLocale().country()), 5000);
}

void nPreferences::askCloseUnsaved() {
    nparent->setProperty("askCloseUnsaved",my_w.askCloseUnsaved->isChecked());
}

void nPreferences::changeShowDimPixel() {
	nparent->my_w.my_view->showDimPixel=my_w.showDimPixel->isChecked();
	nparent->my_w.my_view->update();
}

void nPreferences::changeFont() {
	bool ok;
	QFont font = QFontDialog::getFont(&ok, nparent->my_w.my_view->font(), this);
	if (ok) {
		nparent->my_w.my_view->setFont(font);
		my_w.labelFont->setFont(font);
		my_w.labelFont->setText(font.family()+" "+QString::number(font.pointSize()));
		QSettings settings("neutrino","");
		settings.beginGroup("Preferences");
		settings.setValue("defaultFont",font.toString());
		settings.endGroup();
		nparent->my_w.my_view->setSize();
	}
}

void nPreferences::changeIconSize(int val) {
	QSize mysize=QSize(10*(val+1),10*(val+1));
	
	foreach (QToolBar *obj, nparent->findChildren<QToolBar *>()) {
		if (obj->iconSize()!=mysize) {
			obj->hide();
			obj->setIconSize(mysize);
			obj->show();
		}
	}
	foreach (nGenericPan* pan, nparent->getPans()) {
		foreach (QToolBar *obj, pan->findChildren<QToolBar *>()) {
			if (obj->iconSize()!=mysize) {
				obj->hide();
				obj->setIconSize(mysize);
				obj->show();
			}
		}
	}
}

void nPreferences::hideEvent(QHideEvent*){
	disconnect(my_w.comboIconSize, SIGNAL(currentIndexChanged(int)), this, SLOT(changeIconSize(int)));
	saveDefaults();
}

void nPreferences::showEvent(QShowEvent*){
	loadDefaults();
	connect(my_w.comboIconSize, SIGNAL(currentIndexChanged(int)), this, SLOT(changeIconSize(int)));
};
