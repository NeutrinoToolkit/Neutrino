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
#include "XRD.h"
#include "neutrino.h"

#include "ui_XRD2.h"


XRD::XRD(neutrino *parent) : nGenericPan(parent) {
    setupUi(this);

    show();

    unsigned int kMax=1;
    if (property("NeuSave-numXRDs").isValid()) {
        kMax=property("NeuSave-numXRDs").toUInt();
    }
    for (unsigned int k=0; k<kMax; k++) {
        addXRD();
    }

    connect(actionAddXRD, SIGNAL(triggered()), this, SLOT(addXRD()));
    connect(actionDelXRD, SIGNAL(triggered()), this, SLOT(delXRD()));
    connect(actionSaveXRDs, SIGNAL(triggered()), this, SLOT(saveAllIPs()));
    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(setObjectVisibility(nPhysD*)));

    loadDefaults();
}

XRD::~XRD() {

}

void XRD::setObjectVisibility(nPhysD*phys) {
    for (int k=0;k<tabIPs->count();k++){
        IPrect[k]->setVisible(phys == getPhysFromCombo(settingsUi[k]->image));
    }
}

void XRD::addXRD() {
    QWidget *tab1 = new QWidget();
    QGridLayout *gridLayout1 = new QGridLayout(tab1);
    gridLayout1->setContentsMargins(0, 0, 0, 0);
    QWidget*wXRD1 = new QWidget(tab1);
    wXRD1->setObjectName(QStringLiteral("wXRD1"));
    gridLayout1->addWidget(wXRD1, 0, 0, 1, 1);
    int numXRDs=tabIPs->count();
    tabIPs->addTab(tab1, "IP"+QLocale().toString(numXRDs+1));

    Ui::XRD2* ui_Settings=new Ui::XRD2();
    ui_Settings->setupUi(wXRD1);
    ui_Settings->name->setText("IP"+QLocale().toString(numXRDs+1));
    settingsUi.push_back(ui_Settings);

    //hack to save diffrent uis!!!
    foreach (QWidget *obj, wXRD1->findChildren<QWidget*>()) {
        obj->setObjectName(obj->objectName()+"-XRD"+QLocale().toString(numXRDs+1));
        obj->setProperty("id", numXRDs);
    }

    decorate(tab1);
    IPs.push_back(nullptr);


    nRect *my_rect=new nRect(this,1);
    my_rect->setRect(QRectF(0,0,100,100));
    my_rect->setProperty("id", numXRDs);
    my_rect->changeToolTip("XRD region "+QLocale().toString(numXRDs+1));
    connect(ui_Settings->rectROI, SIGNAL(released()),my_rect, SLOT(togglePadella()));
    connect(ui_Settings->show, SIGNAL(released()),this, SLOT(cropImage()));
    connect(ui_Settings->original, SIGNAL(released()),this, SLOT(showOriginal()));
    IPrect.push_back(my_rect);

    connect(ui_Settings->rectROI, SIGNAL(released()),my_rect, SLOT(togglePadella()));

    numXRDs++;

    setProperty("NeuSave-numXRDs",numXRDs);
}

void XRD::delXRD() {
    int numXRDs=tabIPs->count();
    if (numXRDs>1) {
        QWidget* my_widget=tabIPs->widget(numXRDs-1);
        tabIPs->removeTab(numXRDs-1);
        my_widget->deleteLater();

        QApplication::processEvents();
        delete IPrect.back();
        IPrect.pop_back();

        IPs.pop_back();

        settingsUi.pop_back();

        numXRDs--;
        setProperty("NeuSave-numXRDs",numXRDs);
    } else {
        statusbar->showMessage("Cannot remove last XRD");
    }
    qDebug() << numXRDs;
}

void XRD::loadSettings(QString my_settings) {
    QSettings settings(my_settings,QSettings::IniFormat);
    settings.beginGroup("Properties");
    int kMax=settings.value("NeuSave-numXRDs",1).toUInt();
    while (tabIPs->count()>1) {
        delXRD();
    }
    while (tabIPs->count()<kMax) {
        addXRD();
    }
    settings.endGroup();

    nGenericPan::loadSettings(settings);

}


void XRD::showOriginal() {
    if (sender() && sender()->property("id").isValid()) {
        unsigned int k=sender()->property("id").toUInt();
        qDebug() << k;
        nPhysD *img=getPhysFromCombo(settingsUi[k]->image);
        if (img) {
            nparent->showPhys(getPhysFromCombo(settingsUi[k]->image));
        } else {
            statusbar->showMessage("BIG PROBLEM");
        }
    }
}

void XRD::cropImage() {
    if (sender() && sender()->property("id").isValid()) {
        int k=sender()->property("id").toInt();
        cropImage(k);
    }
}

void XRD::cropImage(int k, bool show) {
    qDebug() << k;

    nPhysD* img=getPhysFromCombo(settingsUi[k]->image);

    QRect geom2=IPrect[k]->getRect(img);
    nPhysD cropped(img->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height()));
    nPhysD *my_phys=new nPhysD(cropped.rotated(settingsUi[k]->angle->value()));
    qDebug() << my_phys->getSize().x() << " " << my_phys->getSize().y();

    if (settingsUi[k]->flipUD->isChecked()) {
        physMath::phys_flip_ud(*dynamic_cast<physD*>(my_phys));
    }
    if (settingsUi[k]->flipLR->isChecked()) {
        physMath::phys_flip_lr(*dynamic_cast<physD*>(my_phys));
    }
    if (settingsUi[k]->transpose->isChecked()) {
        physMath::phys_transpose(*dynamic_cast<physD*>(my_phys));
    }

    IPs[k]=nparent->replacePhys(my_phys,IPs[k],show);

    qDebug() << IPs[k]->getSize().x() << " " << IPs[k]->getSize().y();

}

void XRD::saveAllIPs() {
    QFileInfo my_file(nparent->getFileSave());
    if (!my_file.filePath().isEmpty()) {
        QDir my_dir(my_file.absolutePath());
        QString my_prefix= my_file.baseName();
        QString my_ext=my_file.suffix();

        for (int k=0; k<tabIPs->count(); k++) {
            cropImage(k,false);
            QString my_name=my_dir.filePath(my_prefix+settingsUi[k]->name->text()+"."+my_ext);
            qDebug() << my_name;
            nparent->fileSave(IPs[k],my_name);
        }
    }
}

