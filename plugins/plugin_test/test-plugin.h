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

// you should include here the relevant (if any) ui_??.h
#include "ui_test-plugin.h"

// This object does the real work, here you write a nGenericPan as if it were in the main tree
class mySkelGUI : public nGenericPan {
Q_OBJECT
public:
    Q_INVOKABLE mySkelGUI(neutrino *, QString);

    Ui::test_plugin my_w;

public slots:

    // here the GUI slots example:
    void mouseAtMatrix(QPointF);

private:

    // here your private stuff

};

NEUTRINO_PLUGIN(mySkelGUI)

#endif
