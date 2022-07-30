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
#include <QMessageBox>
#include <QFileDialog>
#include <QColorDialog>

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
            my_w.infoCores->insertPlainText("Number of processors : "+QLocale().toString(procs));
			my_w.infoCores->insertPlainText("\nNumber of threads : "+QLocale().toString(nthreads));
			my_w.infoCores->insertPlainText("\nMax threads : "+QLocale().toString(maxt));
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


    connect(my_w.forceDecimalDot, SIGNAL(stateChanged(int)), napp, SLOT(forceDecimalDot(int)));

    my_w.openclUnit->setMaximum(physWave::openclEnabled());


	connect(my_w.openclUnit, SIGNAL(valueChanged(int)), this, SLOT(openclUnitValueChange(int)));
    connect(my_w.threads, SIGNAL(valueChanged(int)), this, SLOT(changeThreads(int)));

    show(true);

	my_w.comboIconSize->setCurrentIndex(nparent->my_w->toolBar->iconSize().width()/10-1);

	changeFont();

	connect(my_w.comboIconSize, SIGNAL(currentIndexChanged(int)), this, SLOT(changeIconSize(int)));
	connect(my_w.fontFace, SIGNAL(activated(int)), this, SLOT(changeFont()));
	connect(my_w.fontSize, SIGNAL(valueChanged(int)), this, SLOT(changeFont()));
    connect(my_w.showDimPixel, SIGNAL(released()), this, SLOT(changeDecorations()));
    connect(my_w.showXYaxes, SIGNAL(released()), this, SLOT(changeDecorations()));
    connect(my_w.showColorbar, SIGNAL(released()), this, SLOT(changeDecorations()));
    connect(my_w.showColorbarValues , SIGNAL(released()), this, SLOT(changeDecorations()));
    connect(my_w.mouseThickness, SIGNAL(valueChanged(double)), this, SLOT(changeDecorations()));
    connect(my_w.gridThickness, SIGNAL(valueChanged(double)), this, SLOT(changeDecorations()));
    connect(my_w.actionReset_settings, SIGNAL(triggered()), this, SLOT(resetSettings()));

    connect(my_w.currentStepScaleFactor,SIGNAL(valueChanged(int)),nparent->my_w->my_view,SLOT(setZoomFactor(int)));

    connect(my_w.pluginList, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(updatePlugindirs()));

    connect(my_w.physNameLength, SIGNAL(valueChanged(int)), this, SLOT(savedefaults()));

    QMap<QString, QVariant> pluginList(nparent->property("NeuSave-plugindirs").toMap());
    qDebug() << pluginList;
    for (auto& k : pluginList.keys()) {
        qDebug() << k << pluginList[k];
        QListWidgetItem *dd=new QListWidgetItem(my_w.pluginList);
        dd->setFlags(dd->flags() |  Qt::ItemIsUserCheckable);
        if (pluginList[k].toInt() == 0) {
            dd->setCheckState(Qt::Unchecked);
        } else {
            dd->setCheckState(Qt::Checked);
        }
        dd->setText(k);
        my_w.pluginList->addItem(dd);
    }
    qDebug() << "<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.";
    connect(my_w.openNewWindow, SIGNAL(toggled(bool)), this, SLOT(saveDefaults()));
    connect(my_w.openclUnit, SIGNAL(valueChanged(int)), this, SLOT(saveDefaults()));
    connect(my_w.separateRGB, SIGNAL(stateChanged(int)), this, SLOT(saveDefaults()));
    connect(my_w.askCloseUnsaved, SIGNAL(stateChanged(int)), this, SLOT(saveDefaults()));

    foreach (QCheckBox *wdg, my_w.groupBox->findChildren<QCheckBox *>()) {
        connect(wdg, SIGNAL(toggled(bool)), this, SLOT(saveDefaults()));
    }
    foreach (QToolButton *wdg, my_w.groupBox->findChildren<QToolButton *>()) {
        connect(wdg, SIGNAL(toggled(bool)), this, SLOT(saveDefaults()));
    }
    foreach (QLineEdit *wdg, my_w.groupBox->findChildren<QLineEdit *>()) {
        connect(wdg, SIGNAL(editingFinished()), this, SLOT(saveDefaults()));
    }
    foreach (QSpinBox *wdg, my_w.groupBox->findChildren<QSpinBox *>()) {
        connect(wdg, SIGNAL(valueChanged(int)), this, SLOT(saveDefaults()));
    }
    foreach (QDoubleSpinBox *wdg, my_w.groupBox->findChildren<QDoubleSpinBox *>()) {
        connect(wdg, SIGNAL(valueChanged(double)), this, SLOT(saveDefaults()));
    }

}

void nPreferences::changeThreads(int num) {
	nApp::changeThreads(num);
//    saveDefaults();
}


void nPreferences::openclUnitValueChange(int num) {
	my_w.openclDescription->clear();
#ifdef HAVE_LIBCLFFT
	if (num>0) {
        my_w.openclDescription->setPlainText(QString::fromStdString(physWave::get_platform_device_info_opencl(num)));
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

void nPreferences::on_getColors_released() {
    if (currentBuffer) {
        vec2f my_vec=currentBuffer->prop["display_range"];
        DEBUG(my_vec);
        my_w.colorMin->setText(QLocale().toString(my_vec.x()));
        my_w.colorMax->setText(QLocale().toString(my_vec.y()));
    }
}

void nPreferences::changeDecorations() {
    saveDefaults();
    nparent->my_w->my_view->my_mouse.pen.setWidthF(my_w.mouseThickness->value());
    nparent->my_w->my_view->my_tics.gridThickness=my_w.gridThickness->value();
    nparent->my_w->my_view->my_tics.showDimPixel=my_w.showDimPixel->isChecked();
    nparent->my_w->my_view->my_tics.showXYaxes=my_w.showXYaxes->isChecked();
    nparent->my_w->my_view->my_tics.showColorbar=my_w.showColorbar->isChecked();
    nparent->my_w->my_view->my_tics.showColorbarValues=my_w.showColorbarValues->isChecked();
    nparent->my_w->my_view->my_mouse.update();
    nparent->my_w->my_view->my_tics.update();
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
    saveDefaults();
}

void nPreferences::hideEvent(QHideEvent*e){
	disconnect(my_w.comboIconSize, SIGNAL(currentIndexChanged(int)), this, SLOT(changeIconSize(int)));
	nGenericPan::hideEvent(e);
}

void nPreferences::showEvent(QShowEvent*e){
	connect(my_w.comboIconSize, SIGNAL(currentIndexChanged(int)), this, SLOT(changeIconSize(int)));
	nGenericPan::showEvent(e);
}

void nPreferences::on_addPlugin_released() {
	QString dir = QFileDialog::getExistingDirectory(this, tr("Open Plugin Directory"),nparent->property("NeuSave-lastplugindir").toString());
	if (QFileInfo(dir).exists()) {
		nparent->scanPlugins(dir);
        QListWidgetItem *dd=new QListWidgetItem(my_w.pluginList);
        dd->setFlags(dd->flags() |  Qt::ItemIsUserCheckable);
        dd->setCheckState(Qt::Checked);
        dd->setText(dir);
        my_w.pluginList->addItem(dd);
	}
//    saveDefaults();
//    nparent->saveDefaults();
}

void nPreferences::updatePlugindirs() {
    QMap<QString, QVariant> pluginList;
    for(int i = 0; i < my_w.pluginList->count(); ++i) {
        pluginList[my_w.pluginList->item(i)->text()]=QVariant(my_w.pluginList->item(i)->checkState());
    }
    nparent->setProperty("NeuSave-plugindirs",pluginList);
    qDebug() << nparent->property("NeuSave-plugindirs");
//    nparent->saveDefaults();
}

void nPreferences::on_removePlugin_released() {
	qDeleteAll(my_w.pluginList->selectedItems());
    updatePlugindirs();
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

void nPreferences::on_checkUpdatesNow_released() {
    napp->checkUpdates();
}
