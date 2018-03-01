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

#include "nApp.h"

#include "nPreferences.h"
#include "neutrino.h"


#ifdef	__WIN32
#include <windows.h>
#endif

#ifdef HAVE_LIBCLFFT
#include "clFFT.h"
#endif


nPreferences::nPreferences(neutrino *nparent) : nGenericPan(nparent) {
	my_w.setupUi(this);

	int coreNum =1;

#ifdef HAVE_OPENMP

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
#endif

	if (coreNum==1) {
		my_w.threads->hide();
		my_w.labelThreads->hide();
	}

	my_w.defaultPluginDir->setText(nparent->property("defaultPluginDir").toString());


	my_w.openclUnit->setMaximum(openclEnabled());


	connect(my_w.openclUnit, SIGNAL(valueChanged(int)), this, SLOT(openclUnitValueChange(int)));
	connect(my_w.threads, SIGNAL(valueChanged(int)), this, SLOT(changeThreads(int)));

    show(true);

	my_w.comboIconSize->setCurrentIndex(nparent->my_w->toolBar->iconSize().width()/10-1);

	changeFont();

	connect(my_w.comboIconSize, SIGNAL(currentIndexChanged(int)), this, SLOT(changeIconSize(int)));
	connect(my_w.fontFace, SIGNAL(activated(int)), this, SLOT(changeFont()));
	connect(my_w.fontSize, SIGNAL(valueChanged(int)), this, SLOT(changeFont()));
	connect(my_w.showDimPixel, SIGNAL(released()), this, SLOT(changeShowDimPixel()));
	connect(my_w.actionReset_settings, SIGNAL(triggered()), this, SLOT(resetSettings()));

	connect(my_w.separateRGB, SIGNAL(toggled(bool)), this, SLOT(saveDefaults()));
	connect(my_w.openclUnit, SIGNAL(valueChanged(int)), this, SLOT(saveDefaults()));

	connect(my_w.currentStepScaleFactor,SIGNAL(valueChanged(int)),nparent->my_w->my_view,SLOT(setZoomFactor(int)));

    my_w.askCloseUnsaved->setChecked(nparent->property("NeuSave-askCloseUnsaved").toBool());
	connect(my_w.askCloseUnsaved, SIGNAL(released()), this, SLOT(askCloseUnsaved()));

    my_w.physNameLength->setValue(nparent->property("NeuSave-physNameLength").toInt());
	connect(my_w.physNameLength, SIGNAL(valueChanged(int)), this, SLOT(changephysNameLength(int)));

	QList<QLocale> allLocales = QLocale::matchingLocales(QLocale::AnyLanguage,QLocale::AnyScript,QLocale::AnyCountry);

	if(!allLocales.contains(QLocale::system())) { // custom locale defined
		my_w.localeCombo->addItem(tr("System: ")+nApp::localeToString(QLocale::system()),QLocale::system());
	}

	if(!allLocales.contains(QLocale())) { // custom locale defined
		my_w.localeCombo->addItem(tr("Current: ")+nApp::localeToString(QLocale()),QLocale());
	}

	qSort(allLocales.begin(),allLocales.end(), nApp::localeLessThan);

	for(auto &locale : allLocales) {
		QString my_str=nApp::localeToString(locale);
		my_w.localeCombo->addItem(my_str,locale);
	}

	my_w.decimal->setText(QLocale().decimalPoint());
	my_w.localeCombo->setCurrentIndex(my_w.localeCombo->findData(QLocale()));
	connect(my_w.localeCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(changeLocale(int)));

	for (auto& d : nparent->property("NeuSave-plugindirs").toStringList()) {
		my_w.pluginList->addItem(d);
	}

}

void nPreferences::changeThreads(int num) {
	nApp::changeThreads(num);
}

void nPreferences::changeLocale(int num) {
	QLocale  locale=my_w.localeCombo->itemData(num).toLocale();
	nApp::changeLocale(locale);
	my_w.decimal->setText(QLocale().decimalPoint());
	my_w.statusBar->showMessage(nApp::localeToString(QLocale()), 5000);
}

void nPreferences::openclUnitValueChange(int num) {
	my_w.openclDescription->clear();
#ifdef HAVE_LIBCLFFT
	if (num>0) {
		my_w.openclDescription->setPlainText(QString::fromStdString(get_platform_device_info_opencl(num)));
		setProperty("openclUnit",num);
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

void nPreferences::on_getOrigin_released() {
    if (currentBuffer) {
        vec2f my_vec=currentBuffer->get_origin();
        DEBUG(my_vec);
        my_w.originX->setText(QLocale().toString(my_vec.x()));
        my_w.originY->setText(QLocale().toString(my_vec.y()));
        my_w.originX->repaint();
        my_w.originY->repaint();
    }
}

void nPreferences::on_getScale_released() {
    if (currentBuffer) {
        vec2f my_vec=currentBuffer->get_scale();
        DEBUG(my_vec);
        my_w.scaleX->setText(locale().toString(my_vec.x()));
        my_w.scaleY->setText(locale().toString(my_vec.y()));
        my_w.scaleX->repaint();
        my_w.scaleY->repaint();
    }
}

void nPreferences::on_lockOrigin_released() {
    if (my_w.lockOrigin->isChecked()) {
        nparent->setProperty("NeuSave-lockOrigin",QPointF(locale().toDouble(my_w.originX->text()),locale().toDouble(my_w.originY->text())));
    } else {
        nparent->setProperty("NeuSave-lockOrigin",QVariant());
    }
}

void nPreferences::on_lockScale_released() {
    if (my_w.lockScale->isChecked()) {
        nparent->setProperty("NeuSave-lockScale",QPointF(locale().toDouble(my_w.scaleX->text()),locale().toDouble(my_w.scaleY->text())));
    } else {
        nparent->setProperty("NeuSave-lockScale",QVariant());
    }
}

void nPreferences::askCloseUnsaved() {
    nparent->setProperty("NeuSave-askCloseUnsaved",my_w.askCloseUnsaved->isChecked());
}

void nPreferences::changeShowDimPixel() {
	nparent->my_w->my_view->showDimPixel=my_w.showDimPixel->isChecked();
	nparent->my_w->my_view->update();
}

void nPreferences::changeFont() {
	QFont font=nparent->my_w->my_view->font();
	if (sender()) {
		font=my_w.fontFace->currentFont();
		font.setPointSize(my_w.fontSize->value());
	} else {
		my_w.fontFace->setCurrentFont(font);
		my_w.fontSize->setValue(font.pointSize());
	}
	nparent->my_w->my_view->setFont(font);
	QSettings settings("neutrino","");
	settings.beginGroup("nPreferences");
	settings.setValue("defaultFont",font.toString());
	settings.endGroup();
	nparent->my_w->my_view->setSize();
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

void nPreferences::hideEvent(QHideEvent*e){
	disconnect(my_w.comboIconSize, SIGNAL(currentIndexChanged(int)), this, SLOT(changeIconSize(int)));
	nGenericPan::hideEvent(e);
}

void nPreferences::showEvent(QShowEvent*e){
	connect(my_w.comboIconSize, SIGNAL(currentIndexChanged(int)), this, SLOT(changeIconSize(int)));
	nGenericPan::showEvent(e);
}

void nPreferences::changephysNameLength(int k) {
    nparent->setProperty("NeuSave-physNameLength",k);
}

void nPreferences::on_addPlugin_released() {
	QString dir = QFileDialog::getExistingDirectory(this, tr("Open Plugin Directory"),nparent->property("NeuSave-lastplugindir").toString());
	if (QFileInfo(dir).exists()) {
		nparent->scanPlugins(dir);
		my_w.pluginList->addItem(dir);
	}
}

void nPreferences::on_removePlugin_released() {
	qDeleteAll(my_w.pluginList->selectedItems());
	QStringList pluginList;
	for(int i = 0; i < my_w.pluginList->count(); ++i) {
		pluginList.append(my_w.pluginList->item(i)->text());
	}
	nparent->setProperty("NeuSave-plugindirs",pluginList);
}

void nPreferences::on_mouseThickness_valueChanged(double val){
    nparent->my_w->my_view->my_mouse.pen.setWidthF(val);
    nparent->my_w->my_view->update();
    qDebug() << "here";
}

void nPreferences::on_gridThickness_valueChanged(double val) {
    nparent->my_w->my_view->my_tics.setGridThickness(val);
}

void nPreferences::on_gridColor_released() {
    QColorDialog colordial(nparent->my_w->my_view->my_tics.gridColor,this);
    colordial.setOption(QColorDialog::ShowAlphaChannel);
    colordial.exec();
    if (colordial.result() && colordial.currentColor().isValid()) {
        nparent->my_w->my_view->my_tics.gridColor=colordial.currentColor();
        nparent->my_w->my_view->update();
    }
}

void nPreferences::on_mouseColor_released() {
    QColorDialog colordial(nparent->my_w->my_view->my_mouse.pen.color());
    colordial.setOption(QColorDialog::ShowAlphaChannel);
    colordial.exec();
    if (colordial.result() && colordial.currentColor().isValid()) {
        nparent->my_w->my_view->my_mouse.pen.setColor(colordial.currentColor());
        nparent->my_w->my_view->update();
    }
}

