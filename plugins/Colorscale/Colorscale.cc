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

    connect(nparent->my_view, SIGNAL(updatecolorbar(QString)), this, SLOT(updatecolorbar(QString)));
    connect(nparent, SIGNAL(colorValue(double)), my_w.histogram, SLOT(colorValue(double)));

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
    connect(my_w.lineMin, SIGNAL(editingFinished()), this, SLOT(minChanged()));
    connect(my_w.lineMax, SIGNAL(editingFinished()), this, SLOT(maxChanged()));

    connect(my_w.setToMin,SIGNAL(pressed()),this,SLOT(setToMin()));
    connect(my_w.setToMax,SIGNAL(pressed()),this,SLOT(setToMax()));

    connect(my_w.addPaletteFile, SIGNAL(released()), this, SLOT(addPaletteFile()));
    connect(my_w.removePaletteFile, SIGNAL(released()), this, SLOT(removePaletteFile()));
    connect(my_w.resetPalettes, SIGNAL(released()), this, SLOT(resetPalettes()));

    connect(my_w.percent, SIGNAL(valueChanged(double)), nparent->my_view, SLOT(rescaleColor(double)));
    my_w.toolBar->addWidget(my_w.percent);

    loadPalettes();

    connect(my_w.palettes, SIGNAL(currentIndexChanged(int)), this, SLOT(paletteComboChange(int)));

    //    connect(palettes, SIGNAL(highlighted(QString)), nparent, SLOT(changeColorTable(QString)));

    my_w.toolBar->insertWidget(my_w.actionInvert,my_w.palettes);

    show(true);

    if (currentBuffer) my_w.gamma->setValue(currentBuffer->prop["gamma"]);

    updatecolorbar();
    cutOffPhys=nullptr;
    QApplication::processEvents();
    my_w.histogram->repaint();

    if (nparent->my_view->property("percentPixels").isValid()) {
        my_w.percent->setValue(nparent->my_view->property("percentPixels").toDouble());
    }

}


void Colorscale::paletteComboChange(int val) {
    QString pname=my_w.palettes->itemData(val).toString();
    qDebug() << pname;
    nparent->my_view->changeColorTable(pname);
}

void Colorscale::on_gamma_valueChanged(int val) {
    if (currentBuffer) {
        currentBuffer->prop["gamma"]=val;
        nparent->showPhys();
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
    my_w.sliderMin->blockSignals(true);
    if (currentBuffer) {
        vec2f minmax=currentBuffer->prop["display_range"];
        DEBUG(QLocale().toDouble(value) << " " << minmax);
        minmax.set_first(QLocale().toDouble(value));
        currentBuffer->prop["display_range"]=minmax;
        my_w.sliderMin->setValue(sliderValues().first());
    }
    nparent->showPhys();
    my_w.histogram->repaint();
    my_w.sliderMin->blockSignals(false);
}

void Colorscale::maxChanged () {
    my_w.sliderMax->blockSignals(true);
    QString value(my_w.lineMax->text());
    if (currentBuffer) {
        vec2f minmax=currentBuffer->prop["display_range"];
        DEBUG(QLocale().toDouble(value) << " " << minmax);
        minmax.set_second(QLocale().toDouble(value));
        currentBuffer->prop["display_range"]=minmax;
        my_w.sliderMax->setValue(sliderValues().second());
    }
    nparent->showPhys();
    my_w.histogram->repaint();
    my_w.sliderMax->blockSignals(false);
}

void Colorscale::invertColors () {
    if (currentBuffer) {
        vec2f oldrange = currentBuffer->prop["display_range"];
        currentBuffer->prop["display_range"]=oldrange.swap();
        nparent->showPhys();
    } else {
        my_w.actionInvert->setChecked(false);
    }
}

void Colorscale::bufferChanged(nPhysD *my_phys) {
    qDebug() << "------------------------------------------------------------ Colorscale::bufferChanged "<< my_phys;
    nGenericPan::bufferChanged(my_phys);
    if (nPhysExists(my_phys)) {
        vec2f minmax=my_phys->prop["display_range"];
        DEBUG(minmax);
        auto sval=sliderValues();
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
        my_w.gamma->setValue(my_phys->prop["gamma"]);
        my_w.sliderMin->blockSignals(true);
        my_w.sliderMax->blockSignals(true);
        my_w.sliderMin->setValue(sval.first());
        my_w.sliderMax->setValue(sval.second());
        my_w.sliderMin->blockSignals(false);
        my_w.sliderMax->blockSignals(false);
    } else{
        my_w.lineMin->setText("");
        my_w.lineMax->setText("");
    }
    my_w.histogram->repaint();

    disconnect(my_w.percent, SIGNAL(valueChanged(int)), nparent->my_view, SLOT(rescaleColor(double)));
    if (nparent->my_view->property("percentPixels").isValid()) {
        my_w.percent->setValue(nparent->my_view->property("percentPixels").toDouble());
    }
    connect(my_w.percent, SIGNAL(valueChanged(int)), nparent->my_view, SLOT(rescaleColor(double)));

}

vec2i Colorscale::sliderValues() {
    if (currentBuffer) {
        vec2f minmax=currentBuffer->prop["display_range"];
        int valmin=my_w.sliderMin->maximum()*(minmax.first()-currentBuffer->get_min())/(currentBuffer->get_max()-currentBuffer->get_min());
        int valmax=my_w.sliderMax->maximum()*(minmax.second()-currentBuffer->get_min())/(currentBuffer->get_max()-currentBuffer->get_min());
        return vec2i(valmin,valmax);
    }
    return vec2i(0,my_w.sliderMax->maximum());
}

void Colorscale::updatecolorbar(QString) {
    qDebug() << "-------------------------------";
    disconnect(my_w.palettes, SIGNAL(currentIndexChanged(int)), this, SLOT(paletteComboChange(int)));
    my_w.palettes->setCurrentIndex(my_w.palettes->findData(nparent->my_view->colorTable));
    connect(my_w.palettes, SIGNAL(currentIndexChanged(int)), this, SLOT(paletteComboChange(int)));

    if (currentBuffer) {
        vec2f minmax=currentBuffer->prop["display_range"];
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
        my_w.sliderMin->setValue(sliderValues().first());
        my_w.sliderMax->setValue(sliderValues().second());
    }
    
    my_w.histogram->repaint();
}


void Colorscale::on_sliderMin_sliderReleased() {
    qDebug() << "MINMINMINMINMINMINMINMINMINMINMINMINMINMINMIN";
    int val = my_w.sliderMin->value();
    on_sliderMin_valueChanged(val);
    minChanged();
}

void Colorscale::on_sliderMax_sliderReleased() {
    qDebug() << "MAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAX";
    int val = my_w.sliderMax->value();
    on_sliderMax_valueChanged(val);
    maxChanged();
}

void Colorscale::on_sliderMin_valueChanged(int val) {
    double doubleVal=0.0;
    if (currentBuffer) doubleVal = (double)val/my_w.sliderMin->maximum()*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min();
    my_w.lineMin->setText(QLocale().toString(doubleVal, 'g'));
    minChanged();
    my_w.histogram->repaint();
}

void Colorscale::on_sliderMax_valueChanged(int val) {
    double doubleVal=1.0;
    if (currentBuffer) doubleVal = (double)val/my_w.sliderMax->maximum()*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min();
    my_w.lineMax->setText(QLocale().toString(doubleVal, 'g'));
    maxChanged();
    my_w.histogram->repaint();
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
    foreach(QString my_file,napp->nPalettes.keys()) {
        addItem(my_file);
    }
}

void Colorscale::addItem(QString my_file) {
    QListWidgetItem * item = new QListWidgetItem(getPaletteIconFile(my_file),my_file,my_w.fileList);
    qDebug() << item;
}

void Colorscale::addPaletteFile() {
    QStringList fnames = QFileDialog::getOpenFileNames(this,tr("Open Palette File"),nullptr,tr("Any files")+QString(" (*)"));
    foreach (QString my_file, fnames) {
        napp->addPaletteFile(my_file);
        QSettings my_set("neutrino","");
        my_set.beginGroup("Palettes");
        QStringList userPalettes=my_set.value("userPalettes","").toStringList();
        userPalettes.append(my_file);
        userPalettes.removeDuplicates();
        my_set.setValue("userPalettes",userPalettes);
        my_set.endGroup();
        addItem(my_file);
    }
}

void Colorscale::removePaletteFile() {

    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    QStringList hiddenPalettes=my_set.value("hiddenPalettes","").toStringList();

    for (auto & my_item : my_w.fileList->selectedItems()) {
        qInfo() << "Remove palette" << my_item->text();
        napp->nPalettes.remove(my_item->text());
        my_w.palettes->removeItem(my_w.palettes->findData(my_item->text()));
        hiddenPalettes.append(my_item->text());
        delete my_item;
    }

    hiddenPalettes.removeDuplicates();
    hiddenPalettes.sort(Qt::CaseInsensitive);
    my_set.setValue("hiddenPalettes",hiddenPalettes);
    my_set.endGroup();

    if (napp->nPalettes.size()==0) {
        resetPalettes();
    }

    updatecolorbar();
}

void Colorscale::on_fileList_itemClicked(QListWidgetItem *item){
    nparent->my_view->changeColorTable(item->text());
}

void Colorscale::resetPalettes() {
    showMessage("Restoring colortables");
    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    my_set.setValue("hiddenPalettes",QStringList());
    my_set.endGroup();

    napp->addDefaultPalettes();
    loadPalettes();
}




