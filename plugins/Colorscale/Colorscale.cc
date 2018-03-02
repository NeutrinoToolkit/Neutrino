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

#include "Colorscale.h"
#include "neutrino.h"
#include "nApp.h"

Colorscale::Colorscale (neutrino *parent) : nGenericPan(parent)
{
    my_w.setupUi(this);

    my_w.histogram->parentPan=this;

	connect(nparent->my_w->my_view, SIGNAL(updatecolorbar()), this, SLOT(updatecolorbar()));
	connect(nparent, SIGNAL(colorValue(double)), my_w.histogram, SLOT(colorValue(double)));

    connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    connect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));

    connect(my_w.actionLog,SIGNAL(triggered()),my_w.histogram,SLOT(repaint()));

    connect(my_w.actionCutoff,SIGNAL(triggered()),this,SLOT(cutOff()));

    connect(my_w.actionInvert,SIGNAL(triggered()),this,SLOT(invertColors()));

    QDoubleValidator *dVal = new QDoubleValidator(this);
    dVal->setNotation(QDoubleValidator::ScientificNotation);
    my_w.lineMin->setValidator(dVal);
    my_w.lineMax->setValidator(dVal);

    if (currentBuffer) {
        vec2f minmax=currentBuffer->property["display_range"];
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
    }
    connect(my_w.lineMin, SIGNAL(returnPressed()), this, SLOT(minChanged()));
    connect(my_w.lineMax, SIGNAL(returnPressed()), this, SLOT(maxChanged()));

    connect(my_w.setToMin,SIGNAL(pressed()),this,SLOT(setToMin()));
    connect(my_w.setToMax,SIGNAL(pressed()),this,SLOT(setToMax()));

    connect(my_w.addPaletteFile, SIGNAL(released()), this, SLOT(addPaletteFile()));
    connect(my_w.removePaletteFile, SIGNAL(released()), this, SLOT(removePaletteFile()));
    connect(my_w.resetPalettes, SIGNAL(released()), this, SLOT(resetPalettes()));

    connect(my_w.percent, SIGNAL(valueChanged(int)), nparent->my_w->my_view, SLOT(rescaleColor(int)));
	my_w.toolBar->addWidget(my_w.percent);

    loadPalettes();

    connect(my_w.palettes, SIGNAL(currentIndexChanged(QString)), nparent->my_w->my_view, SLOT(changeColorTable(QString)));
	//    connect(palettes, SIGNAL(highlighted(QString)), nparent, SLOT(changeColorTable(QString)));

    my_w.toolBar->insertWidget(my_w.actionInvert,my_w.palettes);

    show(true);

    if (currentBuffer) my_w.gamma->setValue(currentBuffer->property["gamma"]);

    updatecolorbar();
    cutOffPhys=NULL;
    QApplication::processEvents();
    my_w.histogram->repaint();

    if (nparent->my_w->my_view->property("percentPixels").isValid()) {
        my_w.percent->setValue(nparent->my_w->my_view->property("percentPixels").toInt());
    }

}

void Colorscale::on_fileList_itemSelectionChanged(){
    QString ctable = QFileInfo(my_w.fileList->selectedItems().at(0)->text()).baseName().replace("_"," ");
    qDebug() << ctable;
    nparent->my_w->my_view->changeColorTable(ctable);
}

void Colorscale::resetPalettes() {
    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    my_set.setValue("paletteFiles",QStringList());
    my_set.endGroup();

    napp->addDefaultPalettes();
    loadPalettes();
}

void Colorscale::on_gamma_valueChanged(int val) {
    if (currentBuffer) {
        currentBuffer->property["gamma"]=val;
        nparent->updatePhys();
    }
}

void Colorscale::setToMin () {
    if (currentBuffer) {
        my_w.lineMin->setText(QLocale().toString(currentBuffer->get_min()));
    }
}

void Colorscale::setToMax () {
    if (currentBuffer) {
        my_w.lineMax->setText(QLocale().toString(currentBuffer->get_max()));
    }
}

void Colorscale::minChanged () {
    QString value(my_w.lineMin->text());
    disconnect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    if (currentBuffer) {
        vec2f minmax=currentBuffer->property["display_range"];
        DEBUG(QLocale().toDouble(value) << " " << minmax);
        minmax.set_first(QLocale().toDouble(value));
        currentBuffer->property["display_range"]=minmax;
        my_w.sliderMin->setValue(sliderValues().first());
    }
    connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    nparent->updatePhys();
    my_w.histogram->repaint();
}

void Colorscale::maxChanged () {
    QString value(my_w.lineMax->text());
    disconnect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
    if (currentBuffer) {
        vec2f minmax=currentBuffer->property["display_range"];
        DEBUG(QLocale().toDouble(value) << " " << minmax);
        minmax.set_second(QLocale().toDouble(value));
        currentBuffer->property["display_range"]=minmax;
        my_w.sliderMax->setValue(sliderValues().second());
    }
    connect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
    nparent->updatePhys();
    my_w.histogram->repaint();
}

void Colorscale::invertColors () {
    if (currentBuffer) {
        vec2f oldrange = currentBuffer->property["display_range"];
        currentBuffer->property["display_range"]=oldrange.swap();
        nparent->updatePhys();
    } else {
        my_w.actionInvert->setChecked(false);
    }
}

void Colorscale::bufferChanged(nPhysD *phys) {
    nGenericPan::bufferChanged(phys);
    if (phys) {
        vec2f minmax=phys->property["display_range"];
        DEBUG(minmax);
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
        my_w.sliderMin->setValue(sliderValues().first());
    } else{
        my_w.lineMin->setText("");
        my_w.lineMax->setText("");
    }
    my_w.sliderMax->setValue(sliderValues().second());
    my_w.gamma->setValue(phys->property["gamma"]);
    my_w.histogram->repaint();

    disconnect(my_w.percent, SIGNAL(valueChanged(int)), nparent->my_w->my_view, SLOT(rescaleColor(int)));
    if (nparent->my_w->my_view->property("percentPixels").isValid()) {
        my_w.percent->setValue(nparent->my_w->my_view->property("percentPixels").toInt());
    }
    connect(my_w.percent, SIGNAL(valueChanged(int)), nparent->my_w->my_view, SLOT(rescaleColor(int)));

}

vec2f Colorscale::sliderValues() {
    if (currentBuffer) {
        vec2f minmax=currentBuffer->property["display_range"];
        double valmin=my_w.sliderMin->maximum()*(minmax.first()-currentBuffer->get_min())/(currentBuffer->get_max()-currentBuffer->get_min());
        double valmax=my_w.sliderMax->maximum()*(minmax.second()-currentBuffer->get_min())/(currentBuffer->get_max()-currentBuffer->get_min());
        return vec2f(valmin,valmax);
    }
    return vec2f(0,my_w.sliderMax->maximum());
}



void Colorscale::updatecolorbar() {
    qDebug() << "-------------------------------";
    disconnect(my_w.palettes, SIGNAL(currentIndexChanged(QString)), nparent->my_w->my_view, SLOT(changeColorTable(QString)));
	disconnect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    disconnect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
    my_w.palettes->clear();
    my_w.palettes->addItems(napp->nPalettes.keys());
    my_w.palettes->setCurrentIndex(my_w.palettes->findText(nparent->my_w->my_view->colorTable));

    if (currentBuffer) {
        vec2f minmax=currentBuffer->property["display_range"];
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
        my_w.sliderMin->setValue(sliderValues().first());
        my_w.sliderMax->setValue(sliderValues().second());
    }
    
    my_w.histogram->repaint();
    connect(my_w.palettes, SIGNAL(currentIndexChanged(QString)), nparent->my_w->my_view, SLOT(changeColorTable(QString)));
	connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    connect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
}

void Colorscale::slider_min_changed(int val) {
    double doubleVal=0.0;
    if (currentBuffer) doubleVal = (double)val/10000.*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min();
    my_w.lineMin->setText(QLocale().toString(doubleVal, 'g'));
    minChanged();
}

void Colorscale::slider_max_changed(int val) {
    double doubleVal=1.0;
    if (currentBuffer) doubleVal = (double)val/10000.*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min();
    my_w.lineMax->setText(QLocale().toString(doubleVal, 'g'));
    maxChanged();
}

void Colorscale::cutOff() {
    if (currentBuffer) {
        nPhysD *cut=new nPhysD(*currentBuffer);
        phys_cutoff(*cut,QLocale().toDouble(my_w.lineMin->text()),QLocale().toDouble(my_w.lineMax->text()));
        cutOffPhys=nparent->replacePhys(cut,cutOffPhys);
    }
}

void Colorscale::loadPalettes() {
    my_w.palettes->clear();
    my_w.palettes->addItems(napp->nPalettes.keys());
    my_w.fileList->clear();

    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    QStringList paletteFilesName=my_set.value("paletteFiles","").toStringList();
    for(auto &my_file : paletteFilesName) {
        new QListWidgetItem(my_file,my_w.fileList);
    }
    my_set.endGroup();
}

void Colorscale::addPaletteFile() {
    QStringList fnames = QFileDialog::getOpenFileNames(this,tr("Open Palette File"),NULL,tr("Any files")+QString(" (*)"));
    foreach (QString paletteFile, fnames) {
        napp->addPaletteFile(paletteFile);
        new QListWidgetItem(paletteFile, my_w.fileList);
    }
}

void Colorscale::removePaletteFile() {
    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    QStringList paletteFiles=my_set.value("paletteFiles","").toStringList();

    for (auto & my_item : my_w.fileList->selectedItems()) {
        QString paletteName=QFileInfo(my_item->text()).baseName().replace("_"," ");
        qDebug() << my_item->text() << paletteName;
        if (napp->nPalettes.contains(paletteName)) {
            if (nparent->my_w->my_view->colorTable!=paletteName) {
                napp->nPalettes.remove(paletteName);
                paletteFiles.removeAll(my_item->text());
                my_w.palettes->removeItem(my_w.palettes->findText(paletteName));
                delete my_item;
            }
        }
    }
    updatecolorbar();
    my_set.setValue("paletteFiles",paletteFiles);
    qDebug() << paletteFiles;
    my_set.endGroup();
}




