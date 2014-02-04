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
#ifndef __nwinlist_h
#define __nwinlist_h

#include <QtGui>
#include <QWidget>

#include "ui_nWinList.h"

#include "nTreeWidget.h"
#include "nPhysImageF.h"
#include "nGenericPan.h"

using namespace std;

class neutrino;

class nWinList : public nGenericPan {
Q_OBJECT
	
public:
	nWinList(neutrino*, QString);	
	Ui::nWinList my_w;

public slots:
	void updatePad(nPhysD* = NULL);
	void physAdd(nPhysD*);
	void physDel(nPhysD*);
	void buttonRemovePhys();
	void buttonCopyPhys();
	
	nPhysD*	getPhys(QTreeWidgetItem*);
	void changeProperties();
	void panAdd(nGenericPan*);
	void panDel(nGenericPan*);
	void panClicked(QListWidgetItem*);
	
};

#endif
