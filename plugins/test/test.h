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

#include "nGenericPan.h"

#ifndef __test_plugin_plugin
#define __test_plugin_plugin

// this includes the tests  .ui designer file
#include "ui_test.h"

// this is a nGenericPan (note the Q_INVOKABLE beforre ctor)
class test : public nGenericPan {
Q_OBJECT
public:

    Q_INVOKABLE test(neutrino *nparent)
        : nGenericPan(nparent) {
        my_w.setupUi(this);
        show();
    }

public slots:

    // here the reiplemneted nGenericPan slots example:
    void mouseAtMatrix(QPointF p) {
        my_w.label->setText(QString::number(p.x())+ " : " +QString::number(p.y()));
    }

private:
    Ui::test_plugin my_w;

};

// this is going to declare a plugin interface registered to neutrino
NEUTRINO_PLUGIN(test)

// same as above but will add th entry in the menu Analysis
// NEUTRINO_PLUGIN(test,Analysis)

#endif
