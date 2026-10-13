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

#include "ui_IPframe.h"


XRDChannel::XRDChannel(unsigned int idx) : QObject(nullptr), k(idx), ui(new Ui::IPframe()) {
}

XRDChannel::~XRDChannel() {
    // delete now (not deleteLater): no widget of a removed IP must stay reachable via findChildren
    delete tab;
    delete rect;
}

XRD::XRD(neutrino *parent) : nGenericPan(parent) {
    setupUi(this);

    toolBar->insertWidget(cropAll,createDir);

    actionAddIP->trigger();

    connect(nparent, &neutrino::bufferChanged, this, &XRD::setObjectVisibility);
    connect(nparent, &neutrino::physDel, this, &XRD::forgetPhys);

    show(true);

    QApplication::processEvents();

    connect(source, &QAbstractButton::released, this, &XRD::showSource);

    // re-attach IP images already open (same name as the tab)
    for (auto &ch : ips) {
        for (auto& my_phys: nparent->getBufferList()) {
            if (!ch->phys && tabIPs->tabText(ch->k) == QString::fromStdString(my_phys->getShortName())) {
                ch->phys=my_phys;
            }
        }
    }
}

XRD::~XRD() {
    // deleting a tab makes QTabWidget emit currentChanged (-> cropImage): block it and keep ips consistent
    const QSignalBlocker blockTabs(tabIPs);
    while (!ips.empty()) {
        std::unique_ptr<XRDChannel> ch=std::move(ips.back());
        ips.pop_back();
    }
}

void XRD::setObjectVisibility(nPhysD*phys) {
    for (auto &ch : ips) {
        if (ch->rect) ch->rect->setVisible(phys == getPhysFromCombo(image));
    }
}

void XRD::on_actionAddIP_triggered() {
    const unsigned int k=ips.size();
    auto ch=std::make_unique<XRDChannel>(k);

    ch->tab = new QWidget();
    QGridLayout *gridLayout = new QGridLayout(ch->tab);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    QWidget*newtab_widget = new QWidget(ch->tab);
    newtab_widget->setObjectName(QStringLiteral("wIP1"));
    gridLayout->addWidget(newtab_widget, 0, 0, 1, 1);

    ch->ui->setupUi(newtab_widget);

    //hack to save diffrent uis!!!
    foreach (QWidget *obj, newtab_widget->findChildren<QWidget*>()) {
        obj->setObjectName(obj->objectName()+"-IP"+QLocale().toString(k+1));
    }

    ch->rect=new nRect(this,1);
    ch->rect->setRect(QRectF(0,0,100,100));
    ch->rect->changeToolTip("IP"+QLocale().toString(k+1));

    // complete: publish, add the tab, then connect
    XRDChannel *c=ch.get();
    Ui::IPframe *ui_IP=c->ui.get();
    ips.push_back(std::move(ch));
    tabIPs->addTab(c->tab, "IP"+QLocale().toString(k+1));

    auto cropK=[this,c]() { cropImage(c->k); };
    connect(c->rect, &nRect::sceneChanged, c, [this,c]() { cropImage(c->k,false); });
    connect(ui_IP->rectROI, SIGNAL(released()), c->rect, SLOT(togglePadella()));
    connect(ui_IP->saveImage, &QAbstractButton::released, c, [this,c]() { saveImage(c->k); });

    connect(ui_IP->angle, &QDoubleSpinBox::valueChanged, c, cropK);
    connect(ui_IP->flipLR, &QAbstractButton::released, c, cropK);
    connect(ui_IP->flipUD, &QAbstractButton::released, c, cropK);
    connect(ui_IP->transpose, &QAbstractButton::released, c, cropK);
    connect(ui_IP->crop, &QAbstractButton::released, c, cropK);
}

void XRD::on_actionDelIP_triggered() {
    if (ips.size()>1) {
        // out of ips first, so that the signals emitted while removing the tab can't reach it
        std::unique_ptr<XRDChannel> ch=std::move(ips.back());
        ips.pop_back();
        tabIPs->removeTab(ch->k);
        // ch (tab, widgets, region, connections) destroyed here; its cropped image stays open
    } else {
        statusbar->showMessage("Cannot remove last IP",5000);
    }
}

void XRD::loadSettings(QSettings &my_settings) {
    QStringList labels= my_settings.value(tabIPs->objectName()+"neutrinoLabels").toStringList();
    int kMax=labels.size();
    while (kMax>0 && (int)ips.size()>kMax) {
        on_actionDelIP_triggered();
    }
    while ((int)ips.size()<kMax) {
        on_actionAddIP_triggered();
    }
    // the regions are saved under their tooltip (= IP name): set it before loading
    for (int k=0; k< kMax; k++) {
        tabIPs->setTabText(k,labels[k]);
        ips[k]->rect->changeToolTip(labels[k]);
    }

    nGenericPan::loadSettings(my_settings); // widgets (angle, flips, image...), regions, properties

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

void XRD::saveImage(unsigned int k) {
    if (k<ips.size()) nparent->fileSave(ips[k]->phys); // refused if not open
}

// an IP image was closed by the user
void XRD::forgetPhys(nPhysD *buf) {
    for (auto &ch : ips) {
        if (ch->phys==buf) ch->phys=nullptr;
    }
}

void XRD::cropImage(unsigned int k, bool show) {
    if (k < ips.size()) {
        XRDChannel &ch=*ips[k];
        nPhysD* img=getPhysFromCombo(image);
        if (nPhysExists(img) && ch.rect) {
            QRect geom2=ch.rect->getRect(img).intersected(QRect(0,0,img->getW(),img->getH()));
            if (geom2.isEmpty()) {
                statusbar->showMessage(tr("IP region outside the image"),2000);
                return;
            }
            nPhysD cropped(img->sub(geom2));
            nPhysD *my_phys=new nPhysD(cropped.rotated(ch.ui->angle->value()));

            if (ch.ui->flipLR->isChecked()) {
                physMath::phys_flip_lr(*dynamic_cast<physD*>(my_phys));
            }
            if (ch.ui->flipUD->isChecked()) {
                physMath::phys_flip_ud(*dynamic_cast<physD*>(my_phys));
            }
            if (ch.ui->transpose->isChecked()) {
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

            if (my_phys->getSurf()==0) { // replacePhys would not take it
                delete my_phys;
                return;
            }
            ch.phys=nparent->replacePhys(my_phys,ch.phys,false);
            if(show) {
                ch.phys->prop["display_range"]=img->prop["display_range"];
                nparent->showPhys(ch.phys);
            }
            statusbar->showMessage("IP" + QString::number(k) + " : " + tabName + " cropped",2000);
        }
    }
}

void XRD::on_actionSaveIPs_triggered() {
    if (nPhysExists(currentBuffer)) {
        QDir my_dir ("");
        QString my_prefix("");
        QString my_ext("tif");
        bool my_dir_chosen=false;
        if (createDir->isChecked()) {
            QString currentdir = QFileInfo(QString::fromStdString(currentBuffer->getFromName())).dir().absolutePath();
            QString my_dir_str = QFileDialog::getExistingDirectory(this,tr("Change monitor directory"),currentdir);
            if (!my_dir_str.isEmpty()) {
                my_dir_chosen=true;
                my_dir=QDir(my_dir_str);
                my_prefix=QFileInfo(my_dir_str).baseName()+"_";
            }
        } else {
            QFileInfo my_file(nparent->getFileSave());
            if (!my_file.filePath().isEmpty()) {
                my_dir_chosen=true;
                my_dir= my_file.absolutePath();
                my_prefix= my_file.baseName()+"_";
                my_ext=my_file.suffix();
            }
        }
        if (!my_dir_chosen) return; // dialog cancelled: QDir("") would be the working directory
        if (my_dir.exists()) {
            for (auto &ch : ips) {
                cropImage(ch->k,false);
                if (nPhysExists(ch->phys)) {
                    QString my_name=my_dir.filePath(my_prefix+tabIPs->tabText(static_cast<int>(ch->k))+"."+my_ext);
                    nparent->fileSave(ch->phys,my_name);
                }
            }
            if(my_prefix.isEmpty()) {
                saveSettings(my_dir.filePath("crop.ini"));
            } else {
                saveSettings(my_dir.filePath(my_prefix+".ini"));
            }
        }
    }
}

void XRD::on_cropAll_triggered() {
    for (unsigned int k=0; k<ips.size(); k++) {
        cropImage(k,false);
    }
}

void XRD::on_removeTransformed_triggered() {
    for (auto &ch : ips) {
        nPhysD *my_phys=ch->phys;
        ch->phys=nullptr;
        nparent->removePhys(my_phys); // ignores images already closed
    }
    showSource();
}

void XRD::on_tabIPs_currentChanged(int k) {
    if(k>=0 && k<(int)ips.size()) {
        cropImage(k);
    }
}

void XRD::on_tabIPs_tabBarDoubleClicked(int k) {
    if (k<0 || k>=(int)ips.size()) return;
    bool ok;
    QString text = QInputDialog::getText(this, tr("Change IP Name"),tr("IP name:"), QLineEdit::Normal,tabIPs->tabText(k) , &ok).trimmed();
    if (ok && !text.isEmpty()) { // the name is also the image name, the saved file name and the region's settings key
        tabIPs->setTabText(k,text);
        ips[k]->rect->changeToolTip(text);
    }
}

