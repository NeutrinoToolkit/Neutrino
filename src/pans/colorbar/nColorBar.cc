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

#include "nColorBar.h"
#include "neutrino.h"

nColorBar::nColorBar (neutrino *parent) : nGenericPan(parent)
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

    colorBase=QColor("blue");
    connect(my_w.addPalette, SIGNAL(released()), this, SLOT(addPalette()));
    connect(my_w.removePalette, SIGNAL(released()), this, SLOT(removePalette()));
    connect(my_w.addColor, SIGNAL(released()), this, SLOT(addColor()));
    connect(my_w.paletteColorlist, SIGNAL(itemDoubleClicked(QTreeWidgetItem *,int)), this, SLOT(itemDoubleClicked(QTreeWidgetItem *,int)));


    connect(my_w.addPaletteFile, SIGNAL(released()), this, SLOT(addPaletteFile()));
    connect(my_w.removePaletteFile, SIGNAL(released()), this, SLOT(removePaletteFile()));

	connect(my_w.percent, SIGNAL(valueChanged(int)), this, SLOT(percentChange()));
	my_w.toolBar->addWidget(my_w.percent);

    palettes = new QComboBox(this);
    QFont f=palettes->font();
    f.setPointSize(10);
    palettes->setFont(f);
    palettes->addItems(nparent->my_w->my_view->nPalettes.keys());
    palettes->setCurrentIndex(nparent->my_w->my_view->nPalettes.keys().indexOf(parent->my_w->my_view->colorTable));
	connect(palettes, SIGNAL(currentIndexChanged(QString)), nparent->my_w->my_view, SLOT(changeColorTable(QString)));
	//    connect(palettes, SIGNAL(highlighted(QString)), nparent, SLOT(changeColorTable(QString)));

    my_w.toolBar->insertWidget(my_w.actionInvert,palettes);

    show();

    if (currentBuffer) my_w.gamma->setValue(currentBuffer->property["gamma"]);

    loadPalettes();

    updatecolorbar();
    cutOffPhys=NULL;
    QApplication::processEvents();
    my_w.histogram->repaint();

}

void nColorBar::percentChange() {
    if (currentBuffer) {
		currentBuffer->property["display_range"]=getColorPrecentPixels(*currentBuffer,my_w.percent->value());
        nparent->updatePhys();
        bufferChanged(currentBuffer);
    }
}

void nColorBar::on_gamma_valueChanged(int val) {
    if (currentBuffer) {
        currentBuffer->property["gamma"]=val;
        nparent->updatePhys();
    }
}

void nColorBar::setToMin () {
    if (currentBuffer) {
        my_w.lineMin->setText(QLocale().toString(currentBuffer->get_min()));
    }
}

void nColorBar::setToMax () {
    if (currentBuffer) {
        my_w.lineMax->setText(QLocale().toString(currentBuffer->get_max()));
    }
}

void nColorBar::minChanged () {
    QString value(my_w.lineMin->text());
    disconnect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    if (currentBuffer) {
        vec2f minmax=currentBuffer->property["display_range"];
        DEBUG(QLocale().toDouble(value));
        minmax.set_first(QLocale().toDouble(value));
        currentBuffer->property["display_range"]=minmax;
        my_w.sliderMin->setValue(sliderValues().first());
    }
    connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    nparent->updatePhys();
    my_w.histogram->repaint();
}

void nColorBar::maxChanged () {
    QString value(my_w.lineMax->text());
    disconnect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
    if (currentBuffer) {
        vec2f minmax=currentBuffer->property["display_range"];
        DEBUG(QLocale().toDouble(value));
        minmax.set_second(QLocale().toDouble(value));
        currentBuffer->property["display_range"]=minmax;
        my_w.sliderMax->setValue(sliderValues().second());
    }
    connect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
    nparent->updatePhys();
    my_w.histogram->repaint();
}

void nColorBar::invertColors () {
    QString mini=my_w.lineMin->text();
    QString maxi=my_w.lineMax->text();
    my_w.lineMin->setText(maxi);
    my_w.lineMax->setText(mini);
}

void nColorBar::bufferChanged(nPhysD *phys) {
    nGenericPan::bufferChanged(phys);
    if (phys) {
        vec2f minmax=phys->property["display_range"];
        DEBUG(minmax);
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
        my_w.gamma->setValue(phys->property["gamma"]);
    } else{
        my_w.lineMin->setText("");
        my_w.lineMax->setText("");
    }
    my_w.histogram->repaint();
}

vec2f nColorBar::sliderValues() {
    if (currentBuffer) {
        vec2f minmax=currentBuffer->property["display_range"];
        double valmin=my_w.sliderMin->maximum()*(minmax.first()-currentBuffer->get_min())/(currentBuffer->get_max()-currentBuffer->get_min());
        double valmax=my_w.sliderMax->maximum()*(minmax.second()-currentBuffer->get_min())/(currentBuffer->get_max()-currentBuffer->get_min());
        return vec2f(valmin,valmax);
    }
    return vec2f(0,my_w.sliderMax->maximum());
}



void nColorBar::updatecolorbar() {
	disconnect(palettes, SIGNAL(currentIndexChanged(QString)), nparent->my_w->my_view, SLOT(changeColorTable(QString)));
	disconnect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    disconnect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
    palettes->clear();
    palettes->addItems(nparent->my_w->my_view->nPalettes.keys());
    palettes->setCurrentIndex(palettes->findText(nparent->my_w->my_view->colorTable));

    if (currentBuffer) {
        vec2f minmax=currentBuffer->property["display_range"];
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
        my_w.sliderMin->setValue(sliderValues().first());
        my_w.sliderMax->setValue(sliderValues().second());
    }
    
    my_w.histogram->repaint();
	connect(palettes, SIGNAL(currentIndexChanged(QString)), nparent->my_w->my_view, SLOT(changeColorTable(QString)));
	connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    connect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
}

void nColorBar::slider_min_changed(int val)
{
    double doubleVal=0.0;
    if (currentBuffer) doubleVal = (double)val/10000.*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min();
    my_w.lineMin->setText(QLocale().toString(doubleVal, 'g'));
    minChanged();
}

void nColorBar::slider_max_changed(int val)
{
    double doubleVal=1.0;
    if (currentBuffer) doubleVal = (double)val/10000.*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min();
    my_w.lineMax->setText(QLocale().toString(doubleVal, 'g'));
    maxChanged();
}

void nColorBar::cutOff() {
    if (currentBuffer) {
        nPhysD *cut=new nPhysD(*currentBuffer);
        phys_cutoff(*cut,QLocale().toDouble(my_w.lineMin->text()),QLocale().toDouble(my_w.lineMax->text()));
        cutOffPhys=nparent->replacePhys(cut,cutOffPhys);
    }
}

void nColorBar::addColor() {
    QColorDialog colordial(colorBase,this);
    colordial.exec();
    if (colordial.result() && colordial.currentColor().isValid()) {
        colorBase=colordial.currentColor();
        my_w.colorlist->insert(colorBase.name());
    }
}

void nColorBar::removePalette() {
    QTreeWidgetItemIterator it(my_w.paletteColorlist);
    while (*it) {
        QString paletteName=(*it)->text(0);
        if (paletteName==my_w.paletteName->text()){
            if (nparent->my_w->my_view->nPalettes.contains(paletteName)) {
                if (nparent->my_w->my_view->colorTable==paletteName) nparent->my_w->my_view->nextColorTable();
                nparent->my_w->my_view->nPalettes.remove(paletteName);
                QSettings my_set("neutrino","");
                my_set.beginGroup("Palettes");
                QStringList paletteNames=my_set.value("paletteNames","").toStringList();
                QStringList paletteColors=my_set.value("paletteColors","").toStringList();
                if (paletteNames.size()==paletteColors.size()) {
                    int pos=paletteNames.indexOf(paletteName);
                    paletteNames.removeAt(pos);
                    paletteColors.removeAt(pos);
                }
                my_set.endGroup();
            }
            delete (*it);
        }
        ++it;
    }
    savePalettes();
}

void nColorBar::addPalette() {
    removePalette();
    if (nparent->addPaletteFromString(my_w.paletteName->text(), my_w.colorlist->text())) {
        QStringList liststring;
        liststring << my_w.paletteName->text() << my_w.colorlist->text();
        new QTreeWidgetItem(my_w.paletteColorlist,liststring);
        savePalettes();
    }
}

void nColorBar::savePalettes() {
    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    QStringList paletteNames;
    QStringList paletteColors;
    QTreeWidgetItemIterator it(my_w.paletteColorlist);
    while (*it) {
        paletteNames.append((*it)->text(0));
        paletteColors.append((*it)->text(1));
        ++it;
    }
    my_set.setValue("paletteNames",paletteNames);
    my_set.setValue("paletteColors",paletteColors);
    my_set.endGroup();
}

void nColorBar::loadPalettes() {
    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    QStringList paletteNames=my_set.value("paletteNames","").toStringList();
    QStringList paletteColors=my_set.value("paletteColors","").toStringList();
    if (paletteNames.size()==paletteColors.size()) {
        for (int i=0;i<paletteNames.size();i++) {
            QStringList liststring;
            liststring << paletteNames.at(i) << paletteColors.at(i);
            new QTreeWidgetItem(my_w.paletteColorlist,liststring);
        }
    }


    QStringList paletteFiles=my_set.value("paletteFiles","").toStringList();
    QStringList paletteFilesName=my_set.value("paletteFilesNames","").toStringList();
    if (paletteFiles.size()==paletteFilesName.size()) {
        for (int i=0;i<paletteFiles.size();i++) {
            QStringList liststring;
            liststring << paletteFilesName.at(i) << paletteFiles.at(i);
            new QTreeWidgetItem(my_w.fileList,liststring);
        }
    }
    my_set.endGroup();
}

void nColorBar::itemDoubleClicked(QTreeWidgetItem *item,int){
    if (item) {
        my_w.paletteName->setText(item->text(0));
        my_w.colorlist->setText(item->text(1));
    }
}

void nColorBar::addPaletteFile() {
    QStringList fnames = QFileDialog::getOpenFileNames(this,tr("Open Palette File"),NULL,tr("Any files")+QString(" (*)"));
    foreach (QString paletteFile, fnames) {
        QString name=nparent->addPaletteFromFile(paletteFile);
        if (!name.isNull()) {
            QStringList liststring;
            liststring << name << paletteFile;
            new QTreeWidgetItem(my_w.fileList,liststring);
        }
    }
}

void nColorBar::removePaletteFile() {
    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    QStringList paletteFiles=my_set.value("paletteFiles","").toStringList();


    QTreeWidgetItemIterator it(my_w.fileList);
    while (*it) {
        if ((*it)->isSelected()) {
            QString paletteName=(*it)->text(0);
            if (nparent->my_w->my_view->nPalettes.contains(paletteName)) {
                if (nparent->my_w->my_view->colorTable==paletteName) nparent->my_w->my_view->nextColorTable();
                nparent->my_w->my_view->nPalettes.remove(paletteName);
                paletteFiles.removeAll((*it)->text(1));
                delete (*it);
            }
        }
        ++it;
    }
    savePalettes();


    my_set.setValue("paletteFiles",paletteFiles);

    my_set.endGroup();
}




