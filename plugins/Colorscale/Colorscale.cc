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

Colorscale::Colorscale (neutrino *parent) : nGenericPan(parent),
    dVal(this)
{
    my_w.setupUi(this);

    my_w.histogram->parentPan=this;

    connect(nparent->my_w->my_view, SIGNAL(updatecolorbar(QString)), this, SLOT(updatecolorbar(QString)));
    connect(nparent, SIGNAL(colorValue(double)), my_w.histogram, SLOT(colorValue(double)));

    connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    connect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));

    connect(my_w.sliderMin,SIGNAL(sliderPressed()),this,SLOT(sliderPressed()));
    connect(my_w.sliderMax,SIGNAL(sliderPressed()),this,SLOT(sliderPressed()));
    connect(my_w.sliderMin,SIGNAL(sliderReleased()),this,SLOT(sliderReleased()));
    connect(my_w.sliderMax,SIGNAL(sliderReleased()),this,SLOT(sliderReleased()));

    connect(my_w.actionLog,SIGNAL(triggered()),my_w.histogram,SLOT(repaint()));

    connect(my_w.actionCutoff,SIGNAL(triggered()),this,SLOT(cutOff()));

    connect(my_w.actionInvert,SIGNAL(triggered()),this,SLOT(invertColors()));

    dVal.setNotation(QDoubleValidator::ScientificNotation);
    my_w.lineMin->setValidator(&dVal);
    my_w.lineMax->setValidator(&dVal);

    if (currentBuffer) {
        vec2f minmax=currentBuffer->prop["display_range"];
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

    connect(my_w.palettes, SIGNAL(currentIndexChanged(int)), this, SLOT(paletteComboChange(int)));

    //    connect(palettes, SIGNAL(highlighted(QString)), nparent, SLOT(changeColorTable(QString)));

    my_w.toolBar->insertWidget(my_w.actionInvert,my_w.palettes);

    show(true);

    if (currentBuffer) my_w.gamma->setValue(currentBuffer->prop["gamma"]);

    updatecolorbar();
    cutOffPhys=NULL;
    QApplication::processEvents();
    my_w.histogram->repaint();

    if (nparent->my_w->my_view->property("percentPixels").isValid()) {
        my_w.percent->setValue(nparent->my_w->my_view->property("percentPixels").toInt());
    }

}

void Colorscale::sliderPressed() {
    qDebug() << "<<<<<<<<<<<<<<<<<<<<<<<<< PRESSED PRESSED";
    this->blockSignals(true);
}

void Colorscale::sliderReleased() {
    qDebug() << ">>>>>>>>>>>>>>>>>>>>>>>>> RELEASED RELEASED";
    this->blockSignals(false);
}

void Colorscale::paletteComboChange(int val) {
    QString pname=my_w.palettes->itemData(val).toString();
    qDebug() << pname;
    nparent->my_w->my_view->changeColorTable(pname);
}

void Colorscale::on_gamma_valueChanged(int val) {
    if (currentBuffer) {
        currentBuffer->prop["gamma"]=val;
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
        vec2f minmax=currentBuffer->prop["display_range"];
        DEBUG(QLocale().toDouble(value) << " " << minmax);
        minmax.set_first(QLocale().toDouble(value));
        currentBuffer->prop["display_range"]=minmax;
        my_w.sliderMin->setValue(sliderValues().first());
    }
    my_w.histogram->repaint();
    QApplication::processEvents();
    connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
}

void Colorscale::maxChanged () {
    QString value(my_w.lineMax->text());
    disconnect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
    if (currentBuffer) {
        vec2f minmax=currentBuffer->prop["display_range"];
        DEBUG(QLocale().toDouble(value) << " " << minmax);
        minmax.set_second(QLocale().toDouble(value));
        currentBuffer->prop["display_range"]=minmax;
        my_w.sliderMax->setValue(sliderValues().second());
    }
    my_w.histogram->repaint();
    QApplication::processEvents();
    connect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
}

void Colorscale::invertColors () {
    if (currentBuffer) {
        vec2f oldrange = currentBuffer->prop["display_range"];
        currentBuffer->prop["display_range"]=oldrange.swap();
        nparent->updatePhys();
    } else {
        my_w.actionInvert->setChecked(false);
    }
}

void Colorscale::bufferChanged(nPhysD *phys) {
    disconnect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    disconnect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));

    nGenericPan::bufferChanged(phys);
    if (phys) {
        vec2f minmax=phys->prop["display_range"];
        DEBUG(minmax);
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
        my_w.sliderMin->setValue(sliderValues().first());
    } else{
        my_w.lineMin->setText("");
        my_w.lineMax->setText("");
    }
    my_w.sliderMax->setValue(sliderValues().second());
    my_w.gamma->setValue(phys->prop["gamma"]);
    my_w.histogram->repaint();

    disconnect(my_w.percent, SIGNAL(valueChanged(int)), nparent->my_w->my_view, SLOT(rescaleColor(int)));
    if (nparent->my_w->my_view->property("percentPixels").isValid()) {
        my_w.percent->setValue(nparent->my_w->my_view->property("percentPixels").toInt());
    }
    connect(my_w.percent, SIGNAL(valueChanged(int)), nparent->my_w->my_view, SLOT(rescaleColor(int)));
    connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    connect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));

}

vec2f Colorscale::sliderValues() {
    if (currentBuffer) {
        vec2f minmax=currentBuffer->prop["display_range"];
        double valmin=my_w.sliderMin->maximum()*(minmax.first()-currentBuffer->get_min())/(currentBuffer->get_max()-currentBuffer->get_min());
        double valmax=my_w.sliderMax->maximum()*(minmax.second()-currentBuffer->get_min())/(currentBuffer->get_max()-currentBuffer->get_min());
        return vec2f(valmin,valmax);
    }
    return vec2f(0,my_w.sliderMax->maximum());
}



void Colorscale::updatecolorbar(QString) {
    qDebug() << "-------------------------------";
    disconnect(my_w.palettes, SIGNAL(currentIndexChanged(int)), this, SLOT(paletteComboChange(int)));
    disconnect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    disconnect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
    my_w.palettes->setCurrentIndex(my_w.palettes->findData(nparent->my_w->my_view->colorTable));
    connect(my_w.palettes, SIGNAL(currentIndexChanged(int)), this, SLOT(paletteComboChange(int)));

    if (currentBuffer) {
        vec2f minmax=currentBuffer->prop["display_range"];
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
        my_w.sliderMin->setValue(sliderValues().first());
        my_w.sliderMax->setValue(sliderValues().second());
    }
    
    my_w.histogram->repaint();
    connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    connect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
}

void Colorscale::slider_min_changed(int val) {
    double doubleVal=0.0;
    if (currentBuffer) doubleVal = (double)val/my_w.sliderMin->maximum()*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min();
    my_w.lineMin->setText(QLocale().toString(doubleVal, 'g'));
    minChanged();
    nparent->updatePhys();
}

void Colorscale::slider_max_changed(int val) {
    double doubleVal=1.0;
    if (currentBuffer) doubleVal = (double)val/my_w.sliderMax->maximum()*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min();
    my_w.lineMax->setText(QLocale().toString(doubleVal, 'g'));
    maxChanged();
    nparent->updatePhys();
}

void Colorscale::cutOff() {
    if (currentBuffer) {
        nPhysD *cut=new nPhysD(*currentBuffer);
        physMath::cutoff(*cut,QLocale().toDouble(my_w.lineMin->text()),QLocale().toDouble(my_w.lineMax->text()));
        cutOffPhys=nparent->replacePhys(cut,cutOffPhys);
    }
}

const QIcon Colorscale::getPaletteIconFile(QString paletteName) {
    QPixmap pix(256,256);
    pix.fill(QColor(Qt::transparent));
    QPainter paint(&pix);
    for (unsigned int i=0;i<napp->nPalettes[paletteName].size()/3;i++) {
        QColor col(napp->nPalettes[paletteName][3*i],napp->nPalettes[paletteName][3*i+1],napp->nPalettes[paletteName][3*i+2]);
        paint.setPen(QPen(col));
        paint.drawLine(i,50,i,240);
    }
    return QIcon(pix);
}

void Colorscale::loadPalettes() {
    disconnect(my_w.palettes, SIGNAL(currentIndexChanged(int)), this, SLOT(paletteComboChange(int)));
    my_w.palettes->clear();
    for (auto& name : napp->nPalettes.keys()) {
        my_w.palettes->addItem(getPaletteIconFile(name), QFileInfo(name).baseName().replace("_"," "), name);
    }
    connect(my_w.palettes, SIGNAL(currentIndexChanged(int)), this, SLOT(paletteComboChange(int)));
    my_w.fileList->clear();
    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    QStringList paletteFilesName=my_set.value("paletteFiles","").toStringList();
    for(auto &my_file : paletteFilesName) {
        new QListWidgetItem(getPaletteIconFile(my_file), my_file, my_w.fileList);
    }
    my_set.endGroup();
}

void Colorscale::addPaletteFile() {
    QStringList fnames = QFileDialog::getOpenFileNames(this,tr("Open Palette File"),NULL,tr("Any files")+QString(" (*)"));
    foreach (QString my_file, fnames) {
        napp->addPaletteFile(my_file);
        new QListWidgetItem(getPaletteIconFile(my_file), my_file, my_w.fileList);
    }
}

void Colorscale::removePaletteFile() {
    disconnect(my_w.palettes, SIGNAL(currentIndexChanged(int)), this, SLOT(paletteComboChange(int)));
    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    QStringList paletteFiles=my_set.value("paletteFiles","").toStringList();
    for (auto & my_item : my_w.fileList->selectedItems()) {
        napp->nPalettes.remove(my_item->text());
        paletteFiles.removeAll(my_item->text());
        my_w.palettes->removeItem(my_w.palettes->findData(my_item->text()));
        delete my_item;

    }
    if (napp->nPalettes.size()==0) {
        resetPalettes();
    }
    connect(my_w.palettes, SIGNAL(currentIndexChanged(int)), this, SLOT(paletteComboChange(int)));

    //    updatecolorbar();
    my_set.setValue("paletteFiles",paletteFiles);
    qDebug() << paletteFiles;
    my_set.endGroup();
}

void Colorscale::on_fileList_itemClicked(QListWidgetItem *item){
    nparent->my_w->my_view->changeColorTable(item->text());
}

void Colorscale::resetPalettes() {
    showMessage("Restoring colortables");
    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    my_set.setValue("paletteFiles",QStringList());
    my_set.endGroup();

    napp->addDefaultPalettes();
    loadPalettes();
}




