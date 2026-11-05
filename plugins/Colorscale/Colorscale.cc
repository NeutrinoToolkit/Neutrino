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
// #include "nApp.h"
#include "nCore.h"

Colorscale::Colorscale (neutrino *parent) : nGenericPan(parent),
    dVal(this)
{
    my_w.setupUi(this);

    my_w.histogram->parentPan=this;

    connect(nparent->my_view, &nView::updatecolorbar, this, &Colorscale::updatecolorbar);
    connect(nparent, &neutrino::colorValue, my_w.histogram, &nHistogram::colorValue);

    connect(my_w.actionLog, &QAction::triggered, my_w.histogram, [h=my_w.histogram]() { h->update(); });

    connect(my_w.actionCutoff, &QAction::triggered, this, &Colorscale::cutOff);

    connect(my_w.actionInvert, &QAction::triggered, this, &Colorscale::invertColors);

    dVal.setNotation(QDoubleValidator::ScientificNotation);
    my_w.lineMin->setValidator(&dVal);
    my_w.lineMax->setValidator(&dVal);

    if (currentBuffer) {
        vec2f minmax=currentBuffer->displayRange();
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
    }
    connect(my_w.lineMin, &QLineEdit::editingFinished, this, &Colorscale::minChanged);
    connect(my_w.lineMax, &QLineEdit::editingFinished, this, &Colorscale::maxChanged);

    connect(my_w.setToMin, &QPushButton::clicked, this, &Colorscale::setToMin);
    connect(my_w.setToMax, &QPushButton::clicked, this, &Colorscale::setToMax);

    connect(my_w.addPaletteFile, &QPushButton::released, this, &Colorscale::addPaletteFile);
    connect(my_w.removePaletteFile, &QPushButton::released, this, &Colorscale::removePaletteFile);
    connect(my_w.resetPalettes, &QPushButton::released, this, &Colorscale::resetPalettes);

    // the value when typing is finished (it rescaled at every keystroke)
    my_w.percent->setKeyboardTracking(false);
    connect(my_w.percent, &QDoubleSpinBox::valueChanged, nparent->my_view, &nView::rescaleColor);
    my_w.toolBar->addWidget(my_w.percent);

    loadPalettes();

    connect(my_w.palettes, &QComboBox::currentIndexChanged, this, &Colorscale::paletteComboChange);

    //    connect(palettes, SIGNAL(highlighted(QString)), nparent, SLOT(changeColorTable(QString)));

    my_w.toolBar->insertWidget(my_w.actionInvert,my_w.palettes);

    show(true);

    if (currentBuffer) my_w.gamma->setValue(currentBuffer->prop["gamma"]);

    cutOffPhys=nullptr;
    // the cut-off image was closed by the user
    connect(nparent, &neutrino::physDel, this, [this](nPhysD *buf) { if (buf==cutOffPhys) cutOffPhys=nullptr; });
    updatecolorbar(); // (it updates the histogram; there was a processEvents and a repaint here)

    if (nparent->my_view->property("percentPixels").isValid()) {
        my_w.percent->setValue(nparent->my_view->property("percentPixels").toDouble());
    }

}


void Colorscale::paletteComboChange(int val) {
    QString pname=my_w.palettes->itemData(val).toString();
    // qDebug() << pname;
    nparent->my_view->changeColorTable(pname);
}

void Colorscale::on_gamma_valueChanged(int val) {
    if (currentBuffer) {
        currentBuffer->prop["gamma"]=val;
        nparent->showPhys();
    }
}

// The line edits show the range rounded (QLocale().toString: 6 digits); the exact values are set in the image by
// setRangeMin/Max (sliders, set to min/max, histogram) and the text is only read back when the user edits it.
// (Everything went through the text before: each slider move or click rounded the range, e.g. 1234567.8 -> 1.23457e+06)
void Colorscale::setRangeMin(double val) {
    if (!currentBuffer) return;
    vec2f minmax=currentBuffer->displayRange();
    minmax.set_first(val);
    currentBuffer->prop["display_range"]=minmax;
    nparent->showPhys(); // bufferChanged: the text, the sliders and the histogram of this pan
}

void Colorscale::setRangeMax(double val) {
    if (!currentBuffer) return;
    vec2f minmax=currentBuffer->displayRange();
    minmax.set_second(val);
    currentBuffer->prop["display_range"]=minmax;
    nparent->showPhys();
}

void Colorscale::setToMin () {
    if (currentBuffer) setRangeMin(currentBuffer->get_min());
}

void Colorscale::setToMax () {
    if (currentBuffer) setRangeMax(currentBuffer->get_max());
}

// typed by the user (editingFinished also comes when the focus leaves an unedited field: its rounded text must
// not replace the exact value); not a number: back to the current value (it set 0)
void Colorscale::minChanged () {
    if (!my_w.lineMin->isModified()) return;
    bool ok=false;
    double val=QLocale().toDouble(my_w.lineMin->text(), &ok);
    my_w.lineMin->setModified(false);
    if (ok) {
        setRangeMin(val);
    } else {
        showMessage(tr("Not a number: ")+my_w.lineMin->text());
        updatecolorbar();
    }
}

void Colorscale::maxChanged () {
    if (!my_w.lineMax->isModified()) return;
    bool ok=false;
    double val=QLocale().toDouble(my_w.lineMax->text(), &ok);
    my_w.lineMax->setModified(false);
    if (ok) {
        setRangeMax(val);
    } else {
        showMessage(tr("Not a number: ")+my_w.lineMax->text());
        updatecolorbar();
    }
}

void Colorscale::invertColors () {
    if (currentBuffer) {
        vec2f oldrange=currentBuffer->displayRange();
        currentBuffer->prop["display_range"]=oldrange.swap();
        nparent->showPhys();
    } else {
        my_w.actionInvert->setChecked(false);
    }
}

void Colorscale::bufferChanged(nPhysD *my_phys) {
    // qDebug() << "------------------------------------------------------------ Colorscale::bufferChanged "<< my_phys;
    nGenericPan::bufferChanged(my_phys);
    // only show the values of the new image: nothing here must write back into it or re-render
    const QSignalBlocker blockMin(my_w.sliderMin), blockMax(my_w.sliderMax), blockGamma(my_w.gamma), blockPercent(my_w.percent), blockInvert(my_w.actionInvert);
    if (nPhysExists(my_phys)) {
        vec2f minmax=my_phys->displayRange();
        // DEBUG(minmax);
        auto sval=sliderValues();
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
        my_w.gamma->setValue(my_phys->prop["gamma"]);
        my_w.sliderMin->setValue(sval.first());
        my_w.sliderMax->setValue(sval.second());
        my_w.actionInvert->setChecked(minmax.first()>minmax.second());
    } else{
        my_w.lineMin->setText("");
        my_w.lineMax->setText("");
    }
    my_w.histogram->update();

    if (nparent->my_view->property("percentPixels").isValid()) {
        my_w.percent->setValue(nparent->my_view->property("percentPixels").toDouble());
    }
}

vec2i Colorscale::sliderValues() {
    if (currentBuffer) {
        vec2f minmax=currentBuffer->displayRange();
        double range=currentBuffer->get_max()-currentBuffer->get_min();
        if (std::isfinite(range) && range>0) { // constant image: no meaningful position
            auto pos=[&](QSlider *slider, double v) {
                double f=std::min(1.0,std::max(0.0,(v-currentBuffer->get_min())/range));
                return std::isfinite(f) ? int(slider->maximum()*f) : 0;
            };
            return vec2i(pos(my_w.sliderMin,minmax.first()),pos(my_w.sliderMax,minmax.second()));
        }
    }
    return vec2i(0,my_w.sliderMax->maximum());
}

void Colorscale::updatecolorbar(QString) {
    // qDebug() << "-------------------------------";
    disconnect(my_w.palettes, &QComboBox::currentIndexChanged, this, &Colorscale::paletteComboChange);
    my_w.palettes->setCurrentIndex(my_w.palettes->findData(nparent->my_view->colorTable));
    connect(my_w.palettes, &QComboBox::currentIndexChanged, this, &Colorscale::paletteComboChange);

    // the sliders only show the range: their (quantized) values must not be written back to the image
    const QSignalBlocker blockMin(my_w.sliderMin), blockMax(my_w.sliderMax);
    if (currentBuffer) {
        vec2f minmax=currentBuffer->displayRange();
        my_w.lineMin->setText(QLocale().toString(minmax.first()));
        my_w.lineMax->setText(QLocale().toString(minmax.second()));
        my_w.sliderMin->setValue(sliderValues().first());
        my_w.sliderMax->setValue(sliderValues().second());
    }
    
    my_w.histogram->update();
}


// with tracking the value is already applied while dragging
void Colorscale::on_sliderMin_sliderReleased() {
    // qDebug() << "MINMINMINMINMINMINMINMINMINMINMINMINMINMINMIN";
    if (!my_w.sliderMin->hasTracking()) on_sliderMin_valueChanged(my_w.sliderMin->value());
}

void Colorscale::on_sliderMax_sliderReleased() {
    // qDebug() << "MAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAX";
    if (!my_w.sliderMax->hasTracking()) on_sliderMax_valueChanged(my_w.sliderMax->value());
}

// the exact value of the slider position (not through the rounded text)
void Colorscale::on_sliderMin_valueChanged(int val) {
    if (currentBuffer) setRangeMin((double)val/my_w.sliderMin->maximum()*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min());
}

void Colorscale::on_sliderMax_valueChanged(int val) {
    if (currentBuffer) setRangeMax((double)val/my_w.sliderMax->maximum()*(currentBuffer->get_max()-currentBuffer->get_min())+currentBuffer->get_min());
}

void Colorscale::cutOff() {
    if (currentBuffer) {
        nPhysD *cut=new nPhysD(*currentBuffer);
        // the exact range (the line edits show it rounded)
        vec2f range=currentBuffer->displayRange();
        physMath::cutoff(*cut,range.first(),range.second());
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
    disconnect(my_w.palettes, &QComboBox::currentIndexChanged, this, &Colorscale::paletteComboChange);
    my_w.palettes->clear();
    for (auto& name : napp->nPalettes.keys()) {
        my_w.palettes->addItem(getPaletteIconFile(name), QFileInfo(name).baseName().replace("_"," "), name);
    }
    connect(my_w.palettes, &QComboBox::currentIndexChanged, this, &Colorscale::paletteComboChange);
    my_w.fileList->clear();
    foreach(QString my_file,napp->nPalettes.keys()) {
        addItem(my_file);
    }
}

void Colorscale::addItem(QString my_file) {
    // QListWidgetItem * item = new QListWidgetItem(getPaletteIconFile(my_file),my_file,my_w.fileList);
    // qDebug() << item;
    new QListWidgetItem(getPaletteIconFile(my_file),my_file,my_w.fileList); // owned by the list
}

void Colorscale::addPaletteFile() {
    QStringList fnames = QFileDialog::getOpenFileNames(this,tr("Open Palette File"),nullptr,tr("Any file")+QString(" (*)"));
    foreach (QString my_file, fnames) {
        QSettings my_set("neutrino","");
        my_set.beginGroup("Palettes");
        // explicitly added again: no longer hidden
        QStringList hiddenPalettes=my_set.value("hiddenPalettes","").toStringList();
        hiddenPalettes.removeAll(my_file);
        my_set.setValue("hiddenPalettes",hiddenPalettes);
        my_set.endGroup();
        my_set.sync();
        if (!napp->addPaletteFile(my_file)) {
            showMessage(tr("Not a palette file (256 lines of R G B, 0-255): ")+my_file);
            continue;
        }
        my_set.beginGroup("Palettes");
        QStringList userPalettes=my_set.value("userPalettes","").toStringList();
        userPalettes.append(my_file);
        userPalettes.removeDuplicates();
        my_set.setValue("userPalettes",userPalettes);
        my_set.endGroup();
        if (my_w.fileList->findItems(my_file,Qt::MatchExactly).isEmpty()) addItem(my_file);
        if (my_w.palettes->findData(my_file)<0) my_w.palettes->addItem(getPaletteIconFile(my_file), QFileInfo(my_file).baseName().replace("_"," "), my_file);
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

    // the palette in use may be gone: changeColorTable falls back to the first one
    if (!napp->nPalettes.contains(nparent->my_view->colorTable)) {
        nparent->my_view->changeColorTable(nparent->my_view->colorTable);
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




