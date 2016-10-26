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

#ifdef HAVE_LIBCLFFT
#include "clFFT.h"
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
            my_w.infoCores->insertPlainText("Number of processors : "+QString::number(procs));
            my_w.infoCores->insertPlainText("\nNumber of threads : "+QString::number(nthreads));
            my_w.infoCores->insertPlainText("\nMax threads : "+QString::number(maxt));
            my_w.infoCores->insertPlainText("\nIn parallel? : "+QString(inpar==0?"No":"Yes"));
            my_w.infoCores->insertPlainText("\nDynamic threads enabled? = "+QString(dynamic==0?"No":"Yes"));
            my_w.infoCores->insertPlainText("\nNested supported? : "+QString(nested==0?"No":"Yes"));
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


    my_w.openclUnit->setMaximum(openclEnabled());

	loadDefaults();
    show();
	    
    my_w.comboIconSize->setCurrentIndex(nparent->my_w.toolBar->iconSize().width()/10-1);

    changeFont();

	connect(my_w.threads, SIGNAL(valueChanged(int)), this, SLOT(changeThreads(int)));
	connect(my_w.comboIconSize, SIGNAL(currentIndexChanged(int)), this, SLOT(changeIconSize(int)));
    connect(my_w.fontFace, SIGNAL(activated(int)), this, SLOT(changeFont()));
    connect(my_w.fontSize, SIGNAL(valueChanged(int)), this, SLOT(changeFont()));
    connect(my_w.showDimPixel, SIGNAL(released()), this, SLOT(changeShowDimPixel()));
    connect(my_w.actionReset_settings, SIGNAL(triggered()), this, SLOT(resetSettings()));


    connect(my_w.currentStepScaleFactor,SIGNAL(valueChanged(int)),nparent->my_w.my_view,SLOT(setZoomFactor(int)));


	connect(my_w.askCloseUnsaved, SIGNAL(released()), this, SLOT(askCloseUnsaved()));
    
    connect(my_w.physNameLength, SIGNAL(valueChanged(int)), this, SLOT(changephysNameLength(int)));

    QList<QLocale> allLocales = QLocale::matchingLocales(QLocale::AnyLanguage,QLocale::AnyScript,QLocale::AnyCountry);

    if(!allLocales.contains(QLocale())) { // custom locale defined
        my_w.localeCombo->addItem(tr("Current: ")+QLocale::languageToString(QLocale().language())+ " " +QLocale::scriptToString(QLocale().script())+ " " +QLocale::countryToString(QLocale().country())+ " " +QString(QLocale().decimalPoint()),QLocale());
    }

    if(!allLocales.contains(QLocale::system())) { // custom locale defined
        my_w.localeCombo->addItem(tr("System: ")+QLocale::languageToString(QLocale::system().language())+ " " +QLocale::scriptToString(QLocale::system().script())+ " " +QLocale::countryToString(QLocale::system().country())+ " " +QString(QLocale::system().decimalPoint()),QLocale::system());
    }

    for(auto &locale : allLocales) {
        QString my_str=QLocale::countryToString(locale.country())+ " " +QLocale::languageToString(locale.language())+ " " +QLocale::scriptToString(locale.script())+ " " +QString(locale.decimalPoint());
        qDebug() << my_str << locale.name();
        my_w.localeCombo->addItem(my_str,locale);
    }

    my_w.decimal->setText(QLocale().decimalPoint());
    my_w.localeCombo->setCurrentIndex(my_w.localeCombo->findData(QLocale()));
    connect(my_w.localeCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(changeLocale(int)));

}

void nPreferences::changeLocale(QLocale locale) {
    if (locale!=QLocale()) {

        qDebug() << QLocale::languageToString(locale.language()) <<
                    QLocale::scriptToString(locale.script()) <<
                    QLocale::countryToString(locale.country()) <<
                    locale.bcp47Name() <<
                    locale.country() <<
                    locale.name() <<
                    locale.decimalPoint();

        QLocale().setDefault(locale);
        QSettings settings("neutrino","");
        settings.beginGroup("Preferences");
        settings.setValue("locale",locale);
        settings.endGroup();


        QDirIterator it(":translations/", QDirIterator::Subdirectories);
        while (it.hasNext()) {
            QString fname=it.next();
            qDebug() << fname;
            if (fname.contains(locale.name())) {
                QTranslator *translator=new QTranslator(qApp);
                bool ok=translator->load(fname);
                if (ok) {
                    qApp->installTranslator(translator);
                    qDebug() << "installing translator" << fname;
                }
            }
        }
    }
}

void nPreferences::changeLocale(int num) {
    QLocale  locale=my_w.localeCombo->itemData(num).toLocale();
    changeLocale(locale);
    my_w.decimal->setText(QLocale().decimalPoint());
    my_w.statusBar->showMessage(tr("Decimal separator: ")+QString(QLocale().decimalPoint()), 5000);
}

void nPreferences::on_openclUnit_valueChanged(int num) {
my_w.openclDescription->clear();
saveDefaults();

#ifdef HAVE_LIBCLFFT
    if (num>0) {
        std::string desc;
        std::pair<cl_platform_id,cl_device_id> my_pair = get_platform_device_opencl(num);
        cl_device_id device=my_pair.second;

        size_t valueSize;
        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
        std::string value;

        value.resize(valueSize);
        clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, &value[0], NULL);
        desc+="Device : "+value;

        // print hardware device version
        clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &valueSize);
        value.resize(valueSize);
        clGetDeviceInfo(device, CL_DEVICE_VERSION, valueSize, &value[0], NULL);
        desc+="\nHardware version : "+value;

        // print software driver version
        clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &valueSize);
        value.resize(valueSize);
        clGetDeviceInfo(device, CL_DRIVER_VERSION, valueSize, &value[0], NULL);
        desc+="\nSoftware version : "+value;

        // print c version supported by compiler for device
        clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
        value.resize(valueSize);
        clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, valueSize, &value[0], NULL);
        desc+="\nOpenCL C version : "+value;

        // print parallel compute units
        cl_uint int_val;
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(int_val), &int_val, NULL);
        desc+="\nParallel compute units : "+std::to_string(int_val);

        clGetDeviceInfo(device,  CL_DEVICE_MAX_CLOCK_FREQUENCY ,sizeof(int_val), &int_val, NULL);
        desc+="\nClock frequency : "+std::to_string(int_val);

        cl_ulong ulong_val;
        clGetDeviceInfo(device,  CL_DEVICE_MAX_MEM_ALLOC_SIZE ,sizeof(ulong_val), &ulong_val, NULL);
        desc+="\nAllocatable Memory : "+std::to_string(ulong_val) +"bytes";
        DEBUG(desc);

        clGetDeviceInfo( device, CL_DEVICE_EXTENSIONS, 0, NULL, &valueSize );
        value.resize(valueSize);
        clGetDeviceInfo( device, CL_DEVICE_EXTENSIONS, valueSize, &value[0], NULL );
        desc+="\nExtensions : "+value;

        desc+="\nDouble support : ";
        desc+=((value.find("cl_khr_fp64") != std::string::npos) ? "Yes":"No");

        my_w.openclDescription->setPlainText(QString::fromStdString(desc));

    }

#endif
}

void nPreferences::resetSettings() {
    int res=QMessageBox::warning(this,tr("Attention"), tr("Are you sure you want to remove Settings?"),
                                 QMessageBox::Yes | QMessageBox::No);
    if (res==QMessageBox::Yes) {
        QSettings my_settings("neutrino","");
        my_settings.clear();
    }
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

void nPreferences::askCloseUnsaved() {
    nparent->setProperty("askCloseUnsaved",my_w.askCloseUnsaved->isChecked());
}

void nPreferences::changeShowDimPixel() {
	nparent->my_w.my_view->showDimPixel=my_w.showDimPixel->isChecked();
	nparent->my_w.my_view->update();
}

void nPreferences::changeFont() {
    QFont font=nparent->my_w.my_view->font();
    if (sender()) {
        font=my_w.fontFace->currentFont();
        font.setPointSize(my_w.fontSize->value());
    } else {
        my_w.fontFace->setCurrentFont(font);
        my_w.fontSize->setValue(font.pointSize());
    }
    nparent->my_w.my_view->setFont(font);
    QSettings settings("neutrino","");
    settings.beginGroup("Preferences");
    settings.setValue("defaultFont",font.toString());
    settings.endGroup();
    nparent->my_w.my_view->setSize();
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
	foreach (nGenericPan* pan, nparent->getPanList()) {
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
}

void nPreferences::changephysNameLength(int k) {
    nparent->setProperty("physNameLength",k);
}


