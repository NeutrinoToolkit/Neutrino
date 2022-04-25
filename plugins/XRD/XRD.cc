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

#include "ui_IP.h"


XRD::XRD(neutrino *parent) : nGenericPan(parent) {
    setupUi(this);

    show();

    unsigned int kMax=1;
    if (property("NeuSave-numIPs").isValid()) {
        kMax=property("NeuSave-numIPs").toUInt();
    }
    for (unsigned int k=0; k<kMax; k++) {
        actionAddIP->trigger();
    }

    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(setObjectVisibility(nPhysD*)));
    connect(source, SIGNAL(released()),this, SLOT(showOriginal()));

    loadDefaults();

}

void XRD::setObjectVisibility(nPhysD*phys) {
    for (int k=0;k<tabIPs->count();k++){
        IPrect[k]->setVisible(phys == getPhysFromCombo(image));
    }
}

void XRD::on_actionAddIP_triggered() {
    QWidget *tab1 = new QWidget();
    QGridLayout *gridLayout1 = new QGridLayout(tab1);
    gridLayout1->setContentsMargins(0, 0, 0, 0);
    QWidget*wIP1 = new QWidget(tab1);
    wIP1->setObjectName(QStringLiteral("wIP1"));
    gridLayout1->addWidget(wIP1, 0, 0, 1, 1);
    int numIPs=tabIPs->count();
    tabIPs->addTab(tab1, "IP"+QLocale().toString(numIPs+1));

    Ui::IP* ui_IP=new Ui::IP();
    ui_IP->setupUi(wIP1);
    settingsUi.push_back(ui_IP);

    //hack to save diffrent uis!!!
    foreach (QWidget *obj, wIP1->findChildren<QWidget*>()) {
        obj->setObjectName(obj->objectName()+"-IP"+QLocale().toString(numIPs+1));
        obj->setProperty("id", numIPs);
    }

    decorate(tab1);
    IPs.push_back(nullptr);


    nRect *my_rect=new nRect(this,1);
    my_rect->setRect(QRectF(0,0,100,100));
    my_rect->setProperty("id", numIPs);
    my_rect->changeToolTip("IP region "+QLocale().toString(numIPs+1));
    IPrect.push_back(my_rect);

    connect(ui_IP->rectROI, SIGNAL(released()),my_rect, SLOT(togglePadella()));
    connect(ui_IP->saveImage, SIGNAL(released()),this, SLOT(saveImage()));

    connect(ui_IP->angle, SIGNAL(valueChanged(double)),this, SLOT(cropImage()));
    connect(ui_IP->flipLR, SIGNAL(released()),this, SLOT(cropImage()));
    connect(ui_IP->flipUD, SIGNAL(released()),this, SLOT(cropImage()));
    connect(ui_IP->transpose, SIGNAL(released()),this, SLOT(cropImage()));

    numIPs++;

    setProperty("NeuSave-numIPs",numIPs);
}

void XRD::on_actionDelIP_triggered() {
    int numIPs=tabIPs->count();
    if (numIPs>0) {
        QWidget* my_widget=tabIPs->widget(numIPs-1);
        tabIPs->removeTab(numIPs-1);
        my_widget->deleteLater();

        QApplication::processEvents();
        delete IPrect.back();
        IPrect.pop_back();

        IPs.pop_back();

        settingsUi.pop_back();

        numIPs--;
        setProperty("NeuSave-numIPs",numIPs);
    } else {
        statusbar->showMessage("Cannot remove last IP");
    }
    qDebug() << numIPs;
}

void XRD::loadSettings(QString my_settings) {
    QSettings settings(my_settings,QSettings::IniFormat);
    settings.beginGroup("Properties");
    int kMax=settings.value("NeuSave-numIPs",1).toUInt();
    while (tabIPs->count()>kMax) {
        actionDelIP->trigger();
    }
    while (tabIPs->count()<kMax) {
        actionAddIP->trigger();
    }
    settings.endGroup();

    nGenericPan::loadSettings(settings);

    on_cropAll_triggered();

}


void XRD::showOriginal() {
    nPhysD *img=getPhysFromCombo(image);
    if (img) {
        nparent->showPhys(img);
    }
}

void XRD::cropImage() {
    if (sender() && sender()->property("id").isValid()) {
        int k=sender()->property("id").toInt();
        cropImage(k);
    }
}

void XRD::saveImage() {
    if (sender() && sender()->property("id").isValid()) {
        int k=sender()->property("id").toInt();
        nparent->fileSave(IPs[k]);
    }
}

void XRD::cropImage(int k, bool show) {
    qDebug() << k;

    nPhysD* img=getPhysFromCombo(image);
    if (img) {
        QRect geom2=IPrect[k]->getRect(img);
        nPhysD cropped(img->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height()));
        nPhysD *my_phys=new nPhysD(cropped.rotated(settingsUi[k]->angle->value()));
        qDebug() << my_phys->getSize().x() << " " << my_phys->getSize().y();

        if (settingsUi[k]->flipLR->isChecked()) {
            physMath::phys_flip_lr(*dynamic_cast<physD*>(my_phys));
        }
        if (settingsUi[k]->flipUD->isChecked()) {
            physMath::phys_flip_ud(*dynamic_cast<physD*>(my_phys));
        }
        if (settingsUi[k]->transpose->isChecked()) {
            physMath::phys_transpose(*dynamic_cast<physD*>(my_phys));
        }
        my_phys->set_scale(1,1);
        my_phys->set_origin(0,0);
        my_phys->prop["display_range"]=img->prop["display_range"];
        my_phys->setShortName(tabIPs->tabText(k).toStdString());
        IPs[k]=nparent->replacePhys(my_phys,IPs[k],false);
        if(show) {
            IPs[k]->prop["display_range"]=img->prop["display_range"];
            nparent->showPhys(IPs[k]);
        }
        qDebug() << IPs[k]->getSize().x() << " " << IPs[k]->getSize().y();

    }
}

void XRD::on_actionSaveIPs_triggered() {
    QFileInfo my_file(nparent->getFileSave());
    if (!my_file.filePath().isEmpty()) {
        QDir my_dir(my_file.absolutePath());
        QString my_prefix= my_file.baseName();
        QString my_ext=my_file.suffix();

        for (int k=0; k<tabIPs->count(); k++) {
            if (IPs[k]) {
                cropImage(k,false);
                QString my_name=my_dir.filePath(my_prefix+"_"+tabIPs->tabText(k)+"."+my_ext);
                qDebug() << my_name;
                nparent->fileSave(IPs[k],my_name);
            }
        }
    }
}

void XRD::on_cropAll_triggered() {
    for (int k=0; k<tabIPs->count(); k++) {
        cropImage(k,false);
    }
}

void XRD::on_removeTransformed_triggered() {
    showOriginal();
    for (int k=0; k<tabIPs->count(); k++) {
        qDebug() <<"here" << k;
        nparent->removePhys(IPs[k]);
        IPs[k]=nullptr;
    }
    qDebug() <<"here";
}

void XRD::on_tabIPs_tabBarClicked(int k) {
    qDebug() << k;
    if(k<static_cast<int>(IPs.size())) {
        cropImage(k,true);
    }
}

void XRD::on_tabIPs_tabBarDoubleClicked(int k) {
    bool ok;
    QString text = QInputDialog::getText(this, tr("Change IP Name"),tr("IP name:"), QLineEdit::Normal,tabIPs->tabText(k) , &ok);
    if (ok && !text.isEmpty()) {
        tabIPs->setTabText(k,text);
        qDebug() << k;
    }
}

