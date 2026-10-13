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
#ifndef XRD_H
#define XRD_H

#include <array>
#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"

#include "nPhysWave.h"
#include "neutrino.h"

#include "ui_XRD.h"

#include <memory>
#include <vector>

class nRect;

namespace Ui {
class IPframe;
}

// everything belonging to one image plate: tab, settings, region and cropped image.
// It is the context object of its connections, so they die with it.
class XRDChannel : public QObject {
    Q_OBJECT
public:
    XRDChannel(unsigned int);
    ~XRDChannel();

    const unsigned int k; // position in XRD::ips and in the tabs (IPs are removed only from the end)
    std::unique_ptr<Ui::IPframe> ui;
    QPointer<QWidget> tab;
    QPointer<nRect> rect;
    nPhysD *phys=nullptr; // cropped image, owned by neutrino
};

class XRD : public nGenericPan, private Ui::XRD {
    Q_OBJECT
    
    using nGenericPan::nGenericPan;
    using nGenericPan::loadSettings;
    using Ui::XRD::retranslateUi;
public:
    
    Q_INVOKABLE XRD(neutrino *);
    ~XRD() override;

private:

    std::vector<std::unique_ptr<XRDChannel>> ips;

public slots:
    
    void setObjectVisibility(nPhysD*);
    void loadSettings(QSettings &settings) override;

    void cropImage(unsigned int, bool=true);
    void saveImage(unsigned int);
    void forgetPhys(nPhysD*);
    void showSource();

    void on_actionAddIP_triggered();
    void on_actionDelIP_triggered();
    void on_actionSaveIPs_triggered();
    void on_cropAll_triggered();
    void on_removeTransformed_triggered();
    void on_tabIPs_currentChanged(int);
    void on_tabIPs_tabBarDoubleClicked(int);

};

NEUTRINO_PLUGIN(XRD,Analysis);

#endif
