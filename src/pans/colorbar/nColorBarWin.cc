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

#include "nColorBarWin.h"
#include "neutrino.h"

nColorBarWin::nColorBarWin (neutrino *parent, QString title) : nGenericPan(parent, title)
{
	my_w.setupUi(this);
	
    my_w.histogram->parentPan=this;

	connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(bufferChanged(nPhysD*)));

	connect(nparent, SIGNAL(updatecolorbar()), this, SLOT(updatecolorbar()));
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
		my_w.lineMin->setText(QString::number(minmax.first()));
		my_w.lineMax->setText(QString::number(minmax.second()));
	}
	connect(my_w.lineMin, SIGNAL(textChanged(QString)), this, SLOT(minChanged(QString)));
	connect(my_w.lineMax, SIGNAL(textChanged(QString)), this, SLOT(maxChanged(QString)));

	connect(my_w.setToMin,SIGNAL(pressed()),this,SLOT(setToMin()));
	connect(my_w.setToMax,SIGNAL(pressed()),this,SLOT(setToMax()));

	colorBase=QColor("blue");
	connect(my_w.addPalette, SIGNAL(released()), this, SLOT(addPalette()));
	connect(my_w.removePalette, SIGNAL(released()), this, SLOT(removePalette()));
	connect(my_w.addColor, SIGNAL(released()), this, SLOT(addColor()));
    connect(my_w.paletteColorlist, SIGNAL(itemDoubleClicked(QTreeWidgetItem *,int)), this, SLOT(itemDoubleClicked(QTreeWidgetItem *,int)));

	
	connect(my_w.addPaletteFile, SIGNAL(released()), this, SLOT(addPaletteFile()));
	connect(my_w.removePaletteFile, SIGNAL(released()), this, SLOT(removePaletteFile()));

    palettes = new QComboBox(this);
    QFont f=palettes->font();
    f.setPointSize(10);
    palettes->setFont(f);
    palettes->addItems(nparent->nPalettes.keys());
    palettes->setCurrentIndex(nparent->nPalettes.keys().indexOf(parent->colorTable));
    connect(palettes, SIGNAL(currentIndexChanged(QString)), nparent, SLOT(changeColorTable(QString)));
//    connect(palettes, SIGNAL(highlighted(QString)), nparent, SLOT(changeColorTable(QString)));

    my_w.toolBar->insertWidget(my_w.actionInvert,palettes);

    show();

    if (nparent->currentBuffer) my_w.gamma->setValue(nparent->currentBuffer->property["gamma"]);

    loadPalettes();

	

	updatecolorbar();
	cutOffPhys=NULL;
    QApplication::processEvents();
    my_w.histogram->repaint();
}

void nColorBarWin::on_gamma_valueChanged(int val) {
    if (currentBuffer) {
        currentBuffer->property["gamma"]=val;
        nparent->createQimage();
    }
}

void nColorBarWin::setToMin () {
	if (currentBuffer) {
		my_w.lineMin->setText(QString::number(currentBuffer->get_min()));
	}
}

void nColorBarWin::setToMax () {
	if (currentBuffer) {
		my_w.lineMax->setText(QString::number(currentBuffer->get_max()));
	}
}

void nColorBarWin::minChanged (QString value) {
	disconnect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
	if (currentBuffer) {
        my_w.sliderMin->setValue(sliderValues().first());
        vec2f minmax=currentBuffer->property["display_range"];
        minmax.set_first(QLocale().toDouble(value));
        currentBuffer->property["display_range"]=minmax;
	}
	connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
	nparent->createQimage();
	my_w.histogram->repaint();
}

void nColorBarWin::maxChanged (QString value) {
	disconnect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
	if (currentBuffer) {
        my_w.sliderMax->setValue(sliderValues().second());
        vec2f minmax=currentBuffer->property["display_range"];
        minmax.set_second(QLocale().toDouble(value));
        currentBuffer->property["display_range"]=minmax;
	}
	connect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
	nparent->createQimage();
	my_w.histogram->repaint();
}

void nColorBarWin::invertColors () {
	QString mini=my_w.lineMin->text();
	QString maxi=my_w.lineMax->text();
	my_w.lineMin->setText(maxi);
	my_w.lineMax->setText(mini);
}

void nColorBarWin::bufferChanged(nPhysD *phys) {
    nGenericPan::bufferChanged(phys);
    if (phys) {
        vec2f minmax=phys->property["display_range"];
        my_w.lineMin->setText(QString::number(minmax.first()));
        my_w.lineMax->setText(QString::number(minmax.second()));
        my_w.gamma->setValue(phys->property["gamma"]);
    } else{
        my_w.lineMin->setText("");
        my_w.lineMax->setText("");
    }
    my_w.histogram->repaint();
}

vec2f nColorBarWin::sliderValues() {
    if (currentBuffer) {
        vec2f minmax=currentBuffer->property["display_range"];
        double valmin=my_w.sliderMin->maximum()*(minmax.first()-currentBuffer->get_min())/(currentBuffer->get_max()-currentBuffer->get_min());
        double valmax=my_w.sliderMax->maximum()*(minmax.second()-currentBuffer->get_min())/(currentBuffer->get_max()-currentBuffer->get_min());
        return vec2f(valmin,valmax);
    }
    return vec2f(0,my_w.sliderMax->maximum());
}



void nColorBarWin::updatecolorbar() {
    disconnect(palettes, SIGNAL(currentIndexChanged(QString)), nparent, SLOT(changeColorTable(QString)));
    disconnect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    disconnect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
    palettes->clear();
    palettes->addItems(nparent->nPalettes.keys());
    palettes->setCurrentIndex(palettes->findText(nparent->colorTable));
	
    if (currentBuffer) {
        vec2f minmax=currentBuffer->property["display_range"];
        my_w.lineMin->setText(QString::number(minmax.first()));
        my_w.lineMax->setText(QString::number(minmax.second()));
        my_w.sliderMin->setValue(sliderValues().first());
        my_w.sliderMax->setValue(sliderValues().second());
    }
    
	my_w.histogram->repaint();
    connect(palettes, SIGNAL(currentIndexChanged(QString)), nparent, SLOT(changeColorTable(QString)));
    connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
    connect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
}

void nColorBarWin::slider_min_changed(int val)
{
	double doubleVal=0.0;
	if (currentBuffer) doubleVal = (double)val/10000.*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min();
 	my_w.lineMin->setText(QString::number(doubleVal, 'g'));
}

void nColorBarWin::slider_max_changed(int val)
{
	double doubleVal=1.0;
	if (currentBuffer) doubleVal = (double)val/10000.*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min();
 	my_w.lineMax->setText(QString::number(doubleVal, 'g'));
}

void nColorBarWin::cutOff() {
	if (currentBuffer) {
		nPhysD *cut=new nPhysD(*currentBuffer);


        phys_cutoff(*cut,QLocale().toDouble(my_w.lineMin->text()),QLocale().toDouble(my_w.lineMax->text()));
		cutOffPhys=nparent->replacePhys(cut,cutOffPhys);
	}
}

void nColorBarWin::addColor() {	
	QColorDialog colordial(colorBase,this);
	colordial.exec();
	if (colordial.result() && colordial.currentColor().isValid()) {
		colorBase=colordial.currentColor();
		my_w.colorlist->insert(colorBase.name());
	}
}

void nColorBarWin::removePalette() {
    QTreeWidgetItemIterator it(my_w.paletteColorlist);
	while (*it) {
		QString paletteName=(*it)->text(0);
		if (paletteName==my_w.paletteName->text()){
            if (nparent->nPalettes.contains(paletteName)) {
				if (nparent->colorTable==paletteName) nparent->nextColorTable();
				nparent->nPalettes.remove(paletteName);
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

void nColorBarWin::addPalette() {
	removePalette();
	if (nparent->addPaletteFromString(my_w.paletteName->text(), my_w.colorlist->text())) {
		QStringList liststring;
		liststring << my_w.paletteName->text() << my_w.colorlist->text();
        new QTreeWidgetItem(my_w.paletteColorlist,liststring);
		savePalettes();
	}
}

void nColorBarWin::savePalettes() {
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

void nColorBarWin::loadPalettes() {
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

void nColorBarWin::itemDoubleClicked(QTreeWidgetItem *item,int){
	if (item) {
		my_w.paletteName->setText(item->text(0));
		my_w.colorlist->setText(item->text(1));
	}
}

void nColorBarWin::addPaletteFile() {
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

void nColorBarWin::removePaletteFile() {
	QSettings my_set("neutrino","");
	my_set.beginGroup("Palettes");
	QStringList paletteFiles=my_set.value("paletteFiles","").toStringList();
	
	
	QTreeWidgetItemIterator it(my_w.fileList);
	while (*it) {
		if ((*it)->isSelected()) {
			QString paletteName=(*it)->text(0);
			if (nparent->nPalettes.contains(paletteName)) {
				if (nparent->colorTable==paletteName) nparent->nextColorTable();
				nparent->nPalettes.remove(paletteName);
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




