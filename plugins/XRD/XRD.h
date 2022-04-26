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

class nRect;

namespace Ui {
class IP;
}

class XRD : public nGenericPan, private Ui::XRD {
    Q_OBJECT
    
    using nGenericPan::nGenericPan;
    using nGenericPan::loadSettings;
    using Ui::XRD::retranslateUi;
public:
    
    Q_INVOKABLE XRD(neutrino *);

private:

    std::vector<Ui::IP*> settingsUi;
    
    std::vector<nRect*> IPrect;

    std::vector<nPhysD*> IPs;

public slots:
    
    void setObjectVisibility(nPhysD*);
    void loadSettings(QString=QString());

    void cropImageNoShow();
    void cropImage(bool=true);
    void cropImage(int, bool=true);
    void saveImage();

    void on_source_released();
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
