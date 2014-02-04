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

	connect(my_w.actionSavePDF,SIGNAL(triggered()),my_w.histogram,SLOT(export_PDF_slot()));
	connect(my_w.checkBox,SIGNAL(stateChanged(int)),my_w.histogram,SLOT(repaint()));

	connect(my_w.cutoff,SIGNAL(released()),this,SLOT(cutOff()));

	connect(my_w.invert,SIGNAL(pressed()),this,SLOT(invertColors()));

	connect(my_w.autoscale, SIGNAL(toggled(bool)), this, SLOT(toogleAutoscale(bool)));

	
	QDoubleValidator *dVal = new QDoubleValidator(this);
	dVal->setNotation(QDoubleValidator::ScientificNotation);
	my_w.lineMin->setValidator(dVal);
	my_w.lineMax->setValidator(dVal);

	getMinMax();
	connect(my_w.lineMin, SIGNAL(textChanged(QString)), this, SLOT(minChanged(QString)));
	connect(my_w.lineMax, SIGNAL(textChanged(QString)), this, SLOT(maxChanged(QString)));

	connect(my_w.setToMin,SIGNAL(pressed()),this,SLOT(setToMin()));
	connect(my_w.setToMax,SIGNAL(pressed()),this,SLOT(setToMax()));

	colorBase=QColor("blue");
	connect(my_w.addPalette, SIGNAL(released()), this, SLOT(addPalette()));
	connect(my_w.removePalette, SIGNAL(released()), this, SLOT(removePalette()));
	connect(my_w.addColor, SIGNAL(released()), this, SLOT(addColor()));
	connect(my_w.palettes, SIGNAL(itemDoubleClicked(QTreeWidgetItem *,int)), this, SLOT(itemDoubleClicked(QTreeWidgetItem *,int)));

	
	connect(my_w.addPaletteFile, SIGNAL(released()), this, SLOT(addPaletteFile()));
	connect(my_w.removePaletteFile, SIGNAL(released()), this, SLOT(removePaletteFile()));

	decorate();

	loadPalettes();

	my_w.comboBox->addItems(nparent->nPalettes.keys());
	my_w.comboBox->setCurrentIndex(nparent->nPalettes.keys().indexOf(parent->colorTable));
	connect(my_w.comboBox, SIGNAL(currentIndexChanged(QString)), nparent, SLOT(changeColorTable(QString)));
	

	updatecolorbar();
	bufferChanged(currentBuffer);
	cutOffPhys=NULL;
}

void nColorBarWin::getMinMax () {
	
	if (currentBuffer) {
		double mini=nparent->colorMin;
		double maxi=nparent->colorMax;
		if (nparent->colorRelative) {
			mini=currentBuffer->Tminimum_value+nparent->colorMin*(currentBuffer->Tmaximum_value - currentBuffer->Tminimum_value);
			maxi=currentBuffer->Tmaximum_value-(1.0-nparent->colorMax)*(currentBuffer->Tmaximum_value - currentBuffer->Tminimum_value);
		}
		my_w.lineMin->setText(QString::number(mini));
		my_w.lineMax->setText(QString::number(maxi));
	}
}	

void nColorBarWin::toogleAutoscale (bool val) {
	nparent->colorRelative=!val;
	nparent->colorMin=my_w.lineMin->text().toDouble();
	nparent->colorMax=my_w.lineMax->text().toDouble();
}

void nColorBarWin::setToMin () {
	if (currentBuffer) {
		my_w.lineMin->setText(QString::number(currentBuffer->Tminimum_value));
	}
}

void nColorBarWin::setToMax () {
	if (currentBuffer) {
		my_w.lineMax->setText(QString::number(currentBuffer->Tmaximum_value));
	}
}

void nColorBarWin::minChanged (QString value) {
	disconnect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
	if (currentBuffer) {
		double percentage=(value.toDouble()-currentBuffer->Tminimum_value)/(currentBuffer->Tmaximum_value-currentBuffer->Tminimum_value);
		my_w.sliderMin->setValue(percentage*100.0);
		if (nparent->colorRelative) {
			nparent->colorMin=percentage;
		} else {
			nparent->colorMin=value.toDouble();
		}
	}
	connect(my_w.sliderMin,SIGNAL(valueChanged(int)),this,SLOT(slider_min_changed(int)));
	nparent->createQimage();
	my_w.histogram->repaint();
}

void nColorBarWin::maxChanged (QString value) {
	disconnect(my_w.sliderMax,SIGNAL(valueChanged(int)),this,SLOT(slider_max_changed(int)));
	if (currentBuffer) {
		double percentage=(value.toDouble()-currentBuffer->Tminimum_value)/(currentBuffer->Tmaximum_value-currentBuffer->Tminimum_value);
		my_w.sliderMax->setValue(percentage*100.0);
		if (nparent->colorRelative) {
			nparent->colorMax=percentage;
		} else {
			nparent->colorMax=value.toDouble();
		}
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

void nColorBarWin::bufferChanged(nPhysD *phys){
	if (phys) {
		if (!my_w.actionAutoscale->isChecked()) {
			double valmin=phys->Tminimum_value+my_w.sliderMin->value()/100.0*(phys->Tmaximum_value-phys->Tminimum_value);
			double valmax=phys->Tminimum_value+my_w.sliderMax->value()/100.0*(phys->Tmaximum_value-phys->Tminimum_value);
			disconnect(my_w.lineMin, SIGNAL(textChanged(QString)), this, SLOT(minChanged(QString)));
			disconnect(my_w.lineMax, SIGNAL(textChanged(QString)), this, SLOT(maxChanged(QString)));
			my_w.lineMin->setText(QString::number(valmin));
			my_w.lineMax->setText(QString::number(valmax));
			connect(my_w.lineMin, SIGNAL(textChanged(QString)), this, SLOT(minChanged(QString)));
			connect(my_w.lineMax, SIGNAL(textChanged(QString)), this, SLOT(maxChanged(QString)));
		}
	}
	my_w.histogram->repaint();
}

void nColorBarWin::updatecolorbar() {
	disconnect(my_w.lineMin, SIGNAL(textChanged(QString)), this, SLOT(minChanged(QString)));
	disconnect(my_w.lineMax, SIGNAL(textChanged(QString)), this, SLOT(maxChanged(QString)));
	disconnect(my_w.comboBox, SIGNAL(currentIndexChanged(QString)), nparent, SLOT(changeColorTable(QString)));
	my_w.comboBox->clear();
	my_w.comboBox->addItems(nparent->nPalettes.keys());
	my_w.comboBox->setCurrentIndex(my_w.comboBox->findText(nparent->colorTable));
	

	my_w.actionAutoscale->setChecked(!nparent->colorRelative);
	if (nparent->colorRelative) {
		my_w.sliderMin->setValue(nparent->colorMin*100.0);
		my_w.sliderMax->setValue(nparent->colorMax*100.0);
	} else {
		my_w.lineMin->setText(QString::number(nparent->colorMin));
		my_w.lineMax->setText(QString::number(nparent->colorMax));		
	}
	my_w.histogram->repaint();
	connect(my_w.lineMin, SIGNAL(textChanged(QString)), this, SLOT(minChanged(QString)));
	connect(my_w.lineMax, SIGNAL(textChanged(QString)), this, SLOT(maxChanged(QString)));
	connect(my_w.comboBox, SIGNAL(currentIndexChanged(QString)), nparent, SLOT(changeColorTable(QString)));
}

void nColorBarWin::slider_min_changed(int val)
{
	double doubleVal=0.0;
	if (currentBuffer) doubleVal = (double)val/100.*(currentBuffer->Tmaximum_value-currentBuffer->Tminimum_value)+currentBuffer->Tminimum_value;
	my_w.lineMin->setText(QString::number(doubleVal, 'g'));
}

void nColorBarWin::slider_max_changed(int val)
{
	double doubleVal=1.0;
	if (currentBuffer) doubleVal = (double)val/100.*(currentBuffer->Tmaximum_value-currentBuffer->Tminimum_value)+currentBuffer->Tminimum_value;
	my_w.lineMax->setText(QString::number(doubleVal, 'g'));
}

void nColorBarWin::refreshNeutrino(){
	nparent->createQimage();
	repaint();
}


void nColorBarWin::cutOff() {
//	QMessageBox::warning(this,"FIXME", QString(__FILE__)+QString(" ")+QString(__FUNCTION__)+QString(" ")+QString::number(__LINE__),QMessageBox::Cancel);
	if (currentBuffer) {
		double mini=my_w.lineMin->text().toDouble();
		double maxi=my_w.lineMax->text().toDouble();

		nPhysD *cut=new nPhysD(currentBuffer->getW(),currentBuffer->getH(),0.0, "IntensityCutoff("+QString::number(mini).toStdString()+","+QString::number(maxi).toStdString()+")");
		cut->setShortName("IntensityCutoff");
		cut->setFromName(currentBuffer->getFromName());
		for (size_t k=0;k<currentBuffer->getSurf();k++) {
			cut->Timg_buffer[k]=min(max(currentBuffer->Timg_buffer[k],mini),maxi);
		}
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
	QTreeWidgetItemIterator it(my_w.palettes);
	while (*it) {
		QString paletteName=(*it)->text(0);
		if (paletteName==my_w.paletteName->text()){
			if (nparent->nPalettes.contains(paletteName)) {
				if (nparent->colorTable==paletteName) nparent->nextColorTable();
				delete nparent->nPalettes[paletteName];
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
		new QTreeWidgetItem(my_w.palettes,liststring);
		savePalettes();
	}
}

void nColorBarWin::savePalettes() {
	QSettings my_set("neutrino","");
	my_set.beginGroup("Palettes");
	QStringList paletteNames;
	QStringList paletteColors;
	QTreeWidgetItemIterator it(my_w.palettes);
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
			new QTreeWidgetItem(my_w.palettes,liststring);				
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
	} else {
		qDebug() << "CLICK outside";
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
				delete nparent->nPalettes[paletteName];
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




