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
#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"
#include "ui_Compare_lines.h"

#ifndef __Compare_lines
#define __Compare_lines

#include "neutrino.h"
class nLine;

class Compare_lines : public nGenericPan {
    Q_OBJECT

public:

    Q_INVOKABLE Compare_lines(neutrino *);
    Ui::Compare_lines my_w;

public slots:
    
    void addImage();
    void removeImage();

    void addImages();
    void removeImages();

    void updatePlot();

    void sceneChanged();

    void physDel(nPhysD*);
    void physReplace(std::pair<nPhysD*,nPhysD*>);
    
private:
    QList <nPhysD*> images;

    nLine line;
};

NEUTRINO_PLUGIN(Compare_lines,Analysis;Lineout);

#endif
