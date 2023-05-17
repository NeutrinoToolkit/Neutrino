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

    toolBar->insertWidget(cropAll,createDir);

    loadDefaults();

    int ksave=std::max(1,property("NeuSave-numIPs").toInt());
    for (int k=0; k<ksave; k++) {
        actionAddIP->trigger();
    }

    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(setObjectVisibility(nPhysD*)));

    show();

    QApplication::processEvents();

    connect(source, SIGNAL(released()),this, SLOT(showSource()));
    qDebug() << std::max(1,property("NeuSave-numIPs").toInt()) << property("NeuSave-numIPs");

    for (int k=0; k< tabIPs->count(); k++) {
        qDebug() << "TOTO" << k <<  tabIPs->tabText(k);
        IPrect[k]->changeToolTip(tabIPs->tabText(k));
        for (auto& my_phys: nparent->getBufferList()) {
            if (tabIPs->tabText(k) == QString::fromStdString(my_phys->getShortName())) {
                qDebug() << "\t found !" << my_phys << IPs[k];
                if (!IPs[k]) {
                    IPs[k]=my_phys;
                }
            }
        }
    }

}

void XRD::setObjectVisibility(nPhysD*phys) {
    for (unsigned int k=0;k < static_cast<unsigned int>(tabIPs->count());k++){
        IPrect[k]->setVisible(phys == getPhysFromCombo(image));
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
    QApplication::processEvents();
    showSource();
}


void XRD::showSource() {
    nPhysD *img=getPhysFromCombo(image);
    if (nPhysExists(img)) {
        nparent->showPhys(img);
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
        nPhysD* img=getPhysFromCombo(image);
        if (img) {
            QRect geom2=IPrect[k]->getRect(img);
            nPhysD cropped(img->sub(geom2));
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

            double tfad=fadeMin->value();
            if (tfad > 0 ) {
                QString kind=IPmodel->currentText();
                double tfad=fadeMin->value();
                std::map<QString, std::array<double,4>> fad = {
                    {"MS", {0.334, 0.666, 107.320, 33974}},
                    {"TR", {0.535, 0.465, 28.812,  3837.2}},
                    {"SR", {0.579, 0.421, 15.052,  3829.5}},
                    {"ND", {0.559, 0.441, 18.179,  3907.0}},
                    {"MP", {0.565, 0.435, 18.461,  6117.5}}};
                double myfad = 1/(fad[kind][0]*exp(-tfad/fad[kind][2])+fad[kind][1]*exp(-tfad/fad[kind][3]));
                physMath::phys_multiply(*my_phys,myfad);
            }


            my_phys->set_scale(1,1);
            my_phys->set_origin(0,0);
            my_phys->setType(PHYS_DYN);

            my_phys->prop["display_range"]=img->prop["display_range"];
            QString tabName=tabIPs->tabText(static_cast<int>(k));

            my_phys->setShortName(tabName.toStdString());

            IPs[k]=nparent->replacePhys(my_phys,IPs[k],false);
            if(show) {
                IPs[k]->prop["display_range"]=img->prop["display_range"];
                nparent->showPhys(IPs[k]);
            }
            qDebug() << IPs[k]->getSize().x() << " " << IPs[k]->getSize().y();
            statusbar->showMessage("IP" + QString::number(k) + " : " + tabName + " cropped",2000);
        }
    }
    saveDefaults();
}

void XRD::on_actionSaveIPs_triggered() {
    if (nPhysExists(currentBuffer)) {
        QDir my_dir ("");
        QString my_prefix("");
        QString my_ext("tif");
        if (createDir->isChecked()) {
            QString currentdir = QFileInfo(QString::fromStdString(currentBuffer->getFromName())).dir().absolutePath();
            QString my_dir_str = QFileDialog::getExistingDirectory(this,tr("Change monitor directory"),currentdir);
            if (!my_dir_str.isEmpty()) {
                my_dir=QDir(my_dir_str);
                my_prefix=QFileInfo(my_dir_str).baseName()+"_";
            }
        } else {
            QFileInfo my_file(nparent->getFileSave());
            if (!my_file.filePath().isEmpty()) {
                my_dir= my_file.absolutePath();
                my_prefix= my_file.baseName()+"_";
                my_ext=my_file.suffix();
            }
        }
        if (my_dir.exists()) {
            for (unsigned int k=0; k<static_cast<unsigned int>(tabIPs->count()); k++) {
                if (IPs[k]) {
                    cropImage(k,false);
                    QString my_name=my_dir.filePath(my_prefix+tabIPs->tabText(static_cast<int>(k))+"."+my_ext);
                    qDebug() << my_name;
                    nparent->fileSave(IPs[k],my_name);
                }
            }
            if(my_prefix.isEmpty()) {
                saveSettings(my_dir.filePath("crop.ini"));
            } else {
                saveSettings(my_dir.filePath(my_prefix+".ini"));
            }
        }
    }
    saveDefaults();
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
            text="IP "+QString::number(k+1);
        } else {

        }
        tabIPs->setTabText(k,text);
        IPrect[k]->changeToolTip(text);
        saveDefaults();
        qDebug() << k;
    }
}

