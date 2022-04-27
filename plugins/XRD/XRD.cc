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

    loadDefaults();

    int ksave=std::max(1,property("NeuSave-numIPs").toInt());
    for (int k=0; k<ksave; k++) {
        actionAddIP->trigger();
    }

    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(setObjectVisibility(nPhysD*)));

    show();

    QApplication::processEvents();

    showSource();
    qDebug() << std::max(1,property("NeuSave-numIPs").toInt()) << property("NeuSave-numIPs");

}

void XRD::setObjectVisibility(nPhysD*phys) {
    for (unsigned int k=0;k < static_cast<unsigned int>(tabIPs->count());k++){
        IPrect[k]->setVisible(phys == getPhysFromCombo(settingsUi[k]->image));
    }
}

void XRD::on_actionAddIP_triggered() {
    QWidget *newtab = new QWidget();
    QGridLayout *gridLayout = new QGridLayout(newtab);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    QWidget*newtab_widget = new QWidget(newtab);
    newtab_widget->setObjectName(QStringLiteral("wIP1"));
    gridLayout->addWidget(newtab_widget, 0, 0, 1, 1);

    Ui::IP* ui_IP=new Ui::IP();
    ui_IP->setupUi(newtab_widget);
    settingsUi.push_back(ui_IP);

    //hack to save diffrent uis!!!
    foreach (QWidget *obj, newtab_widget->findChildren<QWidget*>()) {
        obj->setObjectName(obj->objectName()+"-IP"+QLocale().toString(tabIPs->count()+1));
        obj->setProperty("id", tabIPs->count());
    }

    IPs.push_back(nullptr);

    nRect *my_rect=new nRect(this,1);
    my_rect->setRect(QRectF(0,0,100,100));
    my_rect->setProperty("id", tabIPs->count());
    my_rect->changeToolTip("IP "+QLocale().toString(tabIPs->count()+1));
    IPrect.push_back(my_rect);

    connect(my_rect, SIGNAL(sceneChanged()), this, SLOT(cropImageNoShow()));

    connect(ui_IP->rectROI, SIGNAL(released()),my_rect, SLOT(togglePadella()));
    connect(ui_IP->saveImage, SIGNAL(released()),this, SLOT(saveImage()));

    connect(ui_IP->angle, SIGNAL(valueChanged(double)),this, SLOT(cropImage()));
    connect(ui_IP->flipLR, SIGNAL(released()),this, SLOT(cropImage()));
    connect(ui_IP->flipUD, SIGNAL(released()),this, SLOT(cropImage()));
    connect(ui_IP->transpose, SIGNAL(released()),this, SLOT(cropImage()));
    connect(ui_IP->crop, SIGNAL(released()),this, SLOT(cropImage()));
    connect(ui_IP->source, SIGNAL(released()),this, SLOT(showSource()));

    tabIPs->addTab(newtab, "IP"+QLocale().toString(tabIPs->count()+1));
    setProperty("NeuSave-numIPs",tabIPs->count());

}

void XRD::on_actionDelIP_triggered() {
    if (tabIPs->count()>1) {
        delete IPrect.back();
        IPrect.pop_back();
        IPs.pop_back();
        settingsUi.pop_back();

        QWidget* my_widget=tabIPs->widget(tabIPs->count()-1);
        tabIPs->removeTab(tabIPs->count()-1);
        my_widget->deleteLater();

        QApplication::processEvents();

        setProperty("NeuSave-numIPs",tabIPs->count());
    } else {
        statusbar->showMessage("Cannot remove last IP",5000);
    }
    qDebug() << tabIPs->count();
}

void XRD::loadSettings(QString my_settings) {
    qDebug() << "here";
    if (my_settings.isEmpty()) {
        QString fname = QFileDialog::getOpenFileName(this, tr("Open INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf);; Any files (*.*)"));
        if (!fname.isNull()) {
            setProperty("NeuSave-fileIni",fname);
            loadSettings(fname);
        }
    } else {
        QSettings settings(my_settings,QSettings::IniFormat);
        settings.beginGroup("Properties");
        int kMax=settings.value("NeuSave-numIPs",1).toInt();
        while (tabIPs->count()>kMax) {
            actionDelIP->trigger();
        }
        while (tabIPs->count()<kMax) {
            actionAddIP->trigger();
        }
        settings.endGroup();
        nGenericPan::loadSettings(my_settings);
    }

    on_cropAll_triggered();

}


void XRD::showSource() {
    if (sender() && sender()->property("id").isValid()) {
        unsigned int k=sender()->property("id").toUInt();
        nPhysD *img=getPhysFromCombo(settingsUi[k]->image);
        if (img) {
            nparent->showPhys(img);
        }
    }
}


void XRD::cropImageNoShow() {
    if (sender() && sender()->property("id").isValid()) {
        unsigned int k=sender()->property("id").toUInt();
        cropImage(k,false);
    }
}

void XRD::cropImage(bool show) {
    if (sender() && sender()->property("id").isValid()) {
        unsigned int k=sender()->property("id").toUInt();
        cropImage(k,show);
    }
}

void XRD::saveImage() {
    if (sender() && sender()->property("id").isValid()) {
        unsigned int k=sender()->property("id").toUInt();
        nparent->fileSave(IPs[k]);
    }
}

void XRD::cropImage(unsigned int k, bool show) {
    qDebug() << k;
    if (k < IPs.size()) {
        nPhysD* img=getPhysFromCombo(settingsUi[k]->image);
        if (img) {
            QRect geom2=IPrect[k]->getRect(img);
            nPhysD cropped(img->sub(geom2.x(),geom2.y(),static_cast<unsigned int>(geom2.width()),static_cast<unsigned int>(geom2.height())));
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
            my_phys->setShortName(tabIPs->tabText(static_cast<int>(k)).toStdString());
            IPs[k]=nparent->replacePhys(my_phys,IPs[k],false);
            if(show) {
                IPs[k]->prop["display_range"]=img->prop["display_range"];
                nparent->showPhys(IPs[k]);
            }
            qDebug() << IPs[k]->getSize().x() << " " << IPs[k]->getSize().y();
            statusbar->showMessage("IP " + QString::number(k) + " : " + tabIPs->tabText(static_cast<int>(k)) + " cropped",2000);
        }
    }
}

void XRD::on_actionSaveIPs_triggered() {
    QFileInfo my_file(nparent->getFileSave());
    if (!my_file.filePath().isEmpty()) {
        QDir my_dir(my_file.absolutePath());
        QString my_prefix= my_file.baseName();
        QString my_ext=my_file.suffix();

        for (unsigned int k=0; k<static_cast<unsigned int>(tabIPs->count()); k++) {
            if (IPs[k]) {
                cropImage(k,false);
                QString my_name=my_dir.filePath(my_prefix+"_"+tabIPs->tabText(static_cast<int>(k))+"."+my_ext);
                qDebug() << my_name;
                nparent->fileSave(IPs[k],my_name);
            }
        }
    }
}

void XRD::on_cropAll_triggered() {
    for (int k=0; k<tabIPs->count(); k++) {
        cropImage(static_cast<unsigned int>(k),false);
    }
}

void XRD::on_removeTransformed_triggered() {
    for (unsigned int k=0; k<static_cast<unsigned int>(tabIPs->count()); k++) {
        qDebug() <<"here" << k;
        nparent->removePhys(IPs[k]);
        IPs[k]=nullptr;
    }
    showSource();
    qDebug() <<"here";
}

void XRD::on_tabIPs_currentChanged(int k) {
    qDebug() << k;
    unsigned int uk=static_cast<unsigned int>(k);
    if(uk<IPs.size()) {
        cropImage(uk);
    }
}

void XRD::on_tabIPs_tabBarDoubleClicked(int k) {
    bool ok;
    QString text = QInputDialog::getText(this, tr("Change IP Name"),tr("IP name:"), QLineEdit::Normal,tabIPs->tabText(k) , &ok);
    if (ok) {
        if (text.isEmpty()) {
            tabIPs->setTabText(k,"IP "+QString::number(k+1));
        } else {
            tabIPs->setTabText(k,text);
        }
        qDebug() << k;
    }
}

