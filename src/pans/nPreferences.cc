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

// #include <unistd.h>
// _WIN32: MinGW, MSVC and clang-cl (__WIN32 is only MinGW)
#ifndef _WIN32
#include <unistd.h>
#endif

#include "nPreferences.h"
// #include "nApp.h"
#include "nCore.h"
#include "nPhysWave.h"

// #ifdef	__WIN32
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef HAVE_LIBCLFFT
#include "clFFT.h"
#endif


nPreferences::nPreferences(neutrino *nparent) : nGenericPan(nparent) {
    setupUi(this);

	int coreNum =1;


// #ifdef	__WIN32
#ifdef _WIN32
	SYSTEM_INFO sysinfo;
	GetSystemInfo( &sysinfo );

	coreNum = sysinfo.dwNumberOfProcessors;
#else
	coreNum=sysconf( _SC_NPROCESSORS_ONLN );
#endif


    int nthreads, procs, maxt, inpar, dynamic;
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

			/* Print environment information */
            infoCores->insertPlainText("Number of processors : "+QLocale().toString(procs));
            infoCores->insertPlainText("\nNumber of threads : "+QLocale().toString(nthreads));
            infoCores->insertPlainText("\nMax threads : "+QLocale().toString(maxt));
            infoCores->insertPlainText("\nIn parallel? : "+QString(inpar==0?"No":"Yes"));
            infoCores->insertPlainText("\nDynamic threads enabled? = "+QString(dynamic==0?"No":"Yes"));
		}
	}


    threads->setMaximum(coreNum);

	if (coreNum==1) {
        threads->hide();
        labelThreads->hide();
	}

//     defaultPluginDir->setText(nparent->property("defaultPluginDir").toString());


    connect(forceDecimalDot, &QCheckBox::toggled, napp, &nCore::forceDecimalDot);

    openclUnit->setMaximum(physWave::openclEnabled());


    connect(openclUnit, &QSpinBox::valueChanged, this, &nPreferences::openclUnitValueChange);
    connect(threads, &QSpinBox::valueChanged, this, &nPreferences::changeThreads);

    show(true);

    comboIconSize->setCurrentIndex(nparent->toolBar->iconSize().width()/10-1);

	changeFont();

    connect(comboIconSize, &QComboBox::currentIndexChanged, this, &nPreferences::changeIconSize);
    connect(fontFace, &QFontComboBox::activated, this, &nPreferences::changeFont);
    connect(fontSize, &QSpinBox::valueChanged, this, &nPreferences::changeFont);
    connect(showDimPixel, &QCheckBox::released, this, &nPreferences::changeDecorations);
    connect(showXYaxes, &QCheckBox::released, this, &nPreferences::changeDecorations);
    connect(showColorbar, &QCheckBox::released, this, &nPreferences::changeDecorations);
    connect(showColorbarValues, &QCheckBox::released, this, &nPreferences::changeDecorations);
    connect(mouseThickness, &QDoubleSpinBox::valueChanged, this, &nPreferences::changeDecorations);
    connect(gridThickness, &QDoubleSpinBox::valueChanged, this, &nPreferences::changeDecorations);
    connect(actionReset_settings, &QAction::triggered, this, &nPreferences::resetSettings);

    connect(currentStepScaleFactor, &QSpinBox::valueChanged, nparent->my_view, &nView::setZoomFactor);

//     connect(pluginList, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(updatePlugindirs(QListWidgetItem*)));

    connect(physNameLength, &QSpinBox::valueChanged, this, &nPreferences::saveDefaults);

    // read by nApp::selectIconTheme at the next start
    connect(iconTheme, &QComboBox::currentIndexChanged, this, [this](int) {
        saveDefaults();
        statusbar->showMessage(tr("Icons will change at the next start"));
    });

//     QMap<QString, QVariant> pluginListMap(nparent->property("NeuSave-plugindirs").toMap());
//     qDebug() << pluginListMap;
//     for (auto& k : pluginListMap.keys()) {
//         qDebug() << k << pluginListMap[k];
//         QListWidgetItem *dd=new QListWidgetItem(pluginList);
//         dd->setFlags(dd->flags() |  Qt::ItemIsUserCheckable);
//         if (pluginListMap[k].toInt() == 0) {
//             dd->setCheckState(Qt::Unchecked);
//         } else {
//             dd->setCheckState(Qt::Checked);
//         }
//         dd->setText(k);
//         pluginList->addItem(dd);
//     }
    qDebug() << "<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.";
    connect(openNewWindow, &QCheckBox::toggled, this, &nPreferences::saveDefaults);
    connect(openclUnit, &QSpinBox::valueChanged, this, &nPreferences::saveDefaults);
    connect(separateRGB, &QCheckBox::toggled, this, &nPreferences::saveDefaults);
    connect(askCloseUnsaved, &QCheckBox::toggled, this, &nPreferences::saveDefaults);

    foreach (QCheckBox *wdg, groupBox->findChildren<QCheckBox *>()) {
        connect(wdg, &QCheckBox::toggled, this, &nPreferences::saveDefaults);
    }
    foreach (QToolButton *wdg, groupBox->findChildren<QToolButton *>()) {
        connect(wdg, &QToolButton::toggled, this, &nPreferences::saveDefaults);
    }
    foreach (QLineEdit *wdg, groupBox->findChildren<QLineEdit *>()) {
        connect(wdg, &QLineEdit::editingFinished, this, &nPreferences::saveDefaults);
    }
    foreach (QSpinBox *wdg, groupBox->findChildren<QSpinBox *>()) {
        connect(wdg, &QSpinBox::valueChanged, this, &nPreferences::saveDefaults);
    }
    foreach (QDoubleSpinBox *wdg, groupBox->findChildren<QDoubleSpinBox *>()) {
        connect(wdg, &QDoubleSpinBox::valueChanged, this, &nPreferences::saveDefaults);
    }
    foreach (QSlider *wdg, groupBox->findChildren<QSlider *>()) {
        connect(wdg, &QSlider::valueChanged, this, &nPreferences::saveDefaults);
    }

    connect(gamma, &QSlider::valueChanged, this, &nPreferences::changeGamma);
}

void nPreferences::changeGamma(int gamma_val) {
    statusbar->showMessage(QString(QChar(0x03B3))+" "+QString(gamma_val<1? "1/"+ QLocale().toString(2-gamma_val) : QLocale().toString(gamma_val)));
}

void nPreferences::changeThreads(int num) {
	// nApp::changeThreads(num);
	nCore::changeThreads(num);
}

void nPreferences::openclUnitValueChange(int num) {
    openclDescription->clear();
#ifdef HAVE_LIBCLFFT
	if (num>0) {
        openclDescription->setPlainText(QString::fromStdString(physWave::get_platform_device_info_opencl(num)));
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
        originX->setText(QLocale().toString(my_vec.x()));
        originY->setText(QLocale().toString(my_vec.y()));
        originX->repaint();
        originY->repaint();
    }
}

void nPreferences::on_getScale_released() {
    if (currentBuffer) {
        vec2f my_vec=currentBuffer->get_scale();
        DEBUG(my_vec);
        scaleX->setText(locale().toString(my_vec.x()));
        scaleY->setText(locale().toString(my_vec.y()));
        scaleX->repaint();
        scaleY->repaint();
    }
}

void nPreferences::on_getColors_released() {
    if (currentBuffer) {
        vec2f my_vec=currentBuffer->displayRange();
        DEBUG(my_vec);
        colorMin->setText(QLocale().toString(my_vec.x()));
        colorMax->setText(QLocale().toString(my_vec.y()));
    }
}

void nPreferences::changeDecorations() {
    saveDefaults();
//    nparent->my_view->my_mouse.pen.setWidthF(mouseThickness->value());
    QPen mousePen=nparent->my_view->my_mouse.pen;
    mousePen.setWidthF(mouseThickness->value());
    nparent->my_view->my_mouse.setPen(mousePen);
    nparent->my_view->my_mouse.saveSettings();
    nparent->my_view->my_tics.gridThickness=gridThickness->value();
    nparent->my_view->my_tics.showDimPixel=showDimPixel->isChecked();
    nparent->my_view->my_tics.showXYaxes=showXYaxes->isChecked();
    nparent->my_view->my_tics.showColorbar=showColorbar->isChecked();
    nparent->my_view->my_tics.showColorbarValues=showColorbarValues->isChecked();
    nparent->my_view->my_mouse.update();
    nparent->my_view->my_tics.update();
    nparent->my_view->update();
}

void nPreferences::changeFont() {
    QFont font=nparent->my_view->font();
	if (sender()) {
        font=fontFace->currentFont();
        font.setPointSize(fontSize->value());
	} else {
        fontFace->setCurrentFont(font);
        fontSize->setValue(font.pointSize());
	}
    nparent->my_view->setFont(font);
	QSettings settings("neutrino","");
	settings.beginGroup("nPreferences");
	settings.setValue("defaultFont",font.toString());
	settings.endGroup();
    nparent->my_view->setSize();
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
    disconnect(comboIconSize, &QComboBox::currentIndexChanged, this, &nPreferences::changeIconSize);
	nGenericPan::hideEvent(e);
}

void nPreferences::showEvent(QShowEvent*e){
    connect(comboIconSize, &QComboBox::currentIndexChanged, this, &nPreferences::changeIconSize);
	nGenericPan::showEvent(e);
}

// (the plugin directories are gone: the plugins are compiled into libNeutrino)
// void nPreferences::on_addPlugin_released() {
// 	QString dir = QFileDialog::getExistingDirectory(this, tr("Open Plugin Directory"),nparent->property("NeuSave-lastplugindir").toString());
// 	if (QFileInfo(dir).exists()) {
// 		nparent->scanPlugins(dir);
//         QListWidgetItem *dd=new QListWidgetItem(pluginList);
//         dd->setFlags(dd->flags() | Qt::ItemIsUserCheckable);
//         dd->setCheckState(Qt::Checked);
//         dd->setText(dir);
//         pluginList->addItem(dd);
// 	}
// //    saveDefaults();
// //    nparent->saveDefaults();
// }

// void nPreferences::updatePlugindirs(QListWidgetItem* item=nullptr) {
//     if (item) {
//         if (item->checkState() == Qt::Checked) {
//             QString dir= item->text();
//             if (QFileInfo(dir).exists()) {
//                 nparent->scanPlugins(dir);
//             }
//         }
//     }
//     QMap<QString, QVariant> pluginListMap;
//     for(int i = 0; i < pluginList->count(); ++i) {
//         pluginListMap[pluginList->item(i)->text()]=QVariant(pluginList->item(i)->checkState());
//     }
//     nparent->setProperty("NeuSave-plugindirs",pluginListMap);
//     qDebug() << nparent->property("NeuSave-plugindirs");
// //    nparent->saveDefaults();
// }

// void nPreferences::on_removePlugin_released() {
//     qDeleteAll(pluginList->selectedItems());
//     updatePlugindirs();
// }

void nPreferences::on_gridColor_released() {
    QColorDialog colordial(nparent->my_view->my_tics.gridColor,this);
    colordial.setOption(QColorDialog::ShowAlphaChannel);
    colordial.exec();
    if (colordial.result() && colordial.currentColor().isValid()) {
        nparent->my_view->my_tics.gridColor=colordial.currentColor();
        nparent->my_view->update();
    }
}

void nPreferences::on_mouseColor_released() {
    QColorDialog colordial(nparent->my_view->my_mouse.pen.color());
    colordial.setOption(QColorDialog::ShowAlphaChannel);
    colordial.exec();
    if (colordial.result() && colordial.currentColor().isValid()) {
//        nparent->my_view->my_mouse.pen.setColor(colordial.currentColor());
        QPen mousePen=nparent->my_view->my_mouse.pen;
        mousePen.setColor(colordial.currentColor());
        nparent->my_view->my_mouse.setPen(mousePen);
        nparent->my_view->my_mouse.saveSettings();
        nparent->my_view->update();
    }
}

void nPreferences::on_checkUpdatesNow_released() {
    if (napp->checkUpdates() ==0) {
        QMessageBox::information(this, tr("Update"), tr("You're at the latest version"), QMessageBox::Ok);
    }
}
