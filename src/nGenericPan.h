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
#include <iostream>
#include "nPhysImageF.h"
#include "panThread.h"
#ifndef __generic_pan
#define __generic_pan

class neutrino;

typedef void (*ifunc)(void *, int &); 

class nGenericPan : public QMainWindow {
	Q_OBJECT

public:
	nGenericPan(){};
	nGenericPan(neutrino *, QString);
	~nGenericPan();
	QGraphicsScene *my_s;

	neutrino *nparent;
	QString panName;
	nPhysD *currentBuffer;
	
	QStringList neutrinoProperties;

	// thread stuff
	panThread nThread;

signals:
	void register_paintPath(QMap<QString, QPainterPath> &, nGenericPan *);
	void unregister_paintPath(nGenericPan *);
	void refresh_paintPath();
	void changeCombo(QComboBox*);


public slots:
	void showMessage(QString);
	void showMessage(QString,int);
	virtual void mouseAtMatrix(QPointF) { }
	virtual void mouseAtWorld(QPointF) { }

	virtual void nZoom(double) { }

	virtual void imageMousePress(QPointF) { }
	virtual void imageMouseRelease(QPointF) { }

	void physAdd(nPhysD *);
	void physDel(nPhysD *);

	virtual void bufferChanged(nPhysD *);

	void decorate();

	// threads
    void runThread(void *iparams, ifunc, QString=QString("Calculating"), int=0);
    
	// to sync image list on combos on the widget
	void comboChanged(int);
	nPhysD* getPhysFromCombo(QComboBox*);

	QString getNameForCombo(QComboBox*,nPhysD *);
	void addPhysToCombos(nPhysD*);

	void loadUi(QSettings*);
	void saveUi(QSettings*);

	void closeEvent(QCloseEvent*);

	//settings

	void loadDefaults();
	void saveDefaults();

	void loadSettings();
	void loadSettings(QString);
	virtual void loadSettings(QSettings *);
	void saveSettings();
	virtual void saveSettings(QSettings *);
	
    bool nPhysExists(nPhysD*);
    
	// python stuff
	void set(QString, QVariant, int=1);
	QVariant get(QString, int=1);
	QList<QList<qreal> >  getData(QString, int=1);
	void button(QString, int=1);	

};



#endif
