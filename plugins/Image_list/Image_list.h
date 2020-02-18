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
#ifndef __Image_list_h
#define __Image_list_h

#include <QtGui>
#include <QWidget>

#include "ui_Image_list.h"

#include "nTreeWidget.h"
#include "nPhysImageF.h"
#include "nGenericPan.h"

class neutrino;

class Image_list : public nGenericPan {
    Q_OBJECT

public:
    Q_INVOKABLE Image_list(neutrino*);

    Ui::Image_list my_w;

    // stuff for static scale/origin
    // (should pass to nPhysProperties once this has been merged to Image_list)
    bool freezedFrame;
    vec2f frScale, frOrigin;

    std::map<nPhysD*,QTreeWidgetItem*> itemsMap;

public slots:
    void selectionChanged();
    void updatePad(nPhysD* = nullptr);
    void physAdd(nPhysD*);
    void physDel(nPhysD*);
    void buttonRemovePhys();
    void buttonCopyPhys();

    nPhysD*	getPhys(QTreeWidgetItem*);
    void changeProperties();

    void setFreezed(bool);
    void originChanged();

    void keyPressEvent(QKeyEvent *);
};

NEUTRINO_PLUGIN(Image_list,Image,":icons/info.png", Qt::Key_I, -100);

#endif
