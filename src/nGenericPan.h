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

#ifndef __generic_pan
#define __generic_pan

class neutrino;

class panThread : public QThread {
public:
	panThread() : idata(NULL), params(NULL), calculation_function(NULL), n_iter(-1), winTitle("Calculating...") { }
	~panThread(){};
	
	void setThread(nPhysD *iimage, void *iparams, std::list<nPhysD *> (*ifunc)(nPhysD *, void *, int &))
	{
		idata = iimage;
		params = iparams;
		calculation_function = ifunc;
	}
	
	void run() {
		if (idata==NULL || calculation_function==NULL)
			return;
		
		std::cerr<<"[nGenericPan] pan thread running..."<<std::flush;
		odata = (*calculation_function)(idata, params, n_iter);
		std::cerr<<"finished!"<<std::endl;
	}
	
	void stop() {
		std::cerr<<"killed!"<<std::endl;
		n_iter = -1;
	}
	
	nPhysD *idata;
	void *params;
	std::list<nPhysD *> (*calculation_function)(nPhysD *, void *, int &);
	int n_iter;
	std::list<nPhysD *> odata;
	
	
	// style
	QString winTitle;
	
	void setTitle(QString my_title)
	{ winTitle = my_title; }
	
};

class sleeper_thread : public QThread
{
public:
	static void msleep(unsigned long msecs)
	{
		QThread::msleep(msecs);
	}
};

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
	void bufferChanged(nPhysD *);

	void decorate();

	// threads
	void progressRun(int);

	// to sync image list on combos on the widget
	void comboChanged(int);
	nPhysD* getPhysFromCombo(QComboBox*);

	QString getNameForCombo(nPhysD *);
	void addPhysToCombos(nPhysD*);

	void loadUi(QSettings*);
	void saveUi(QSettings*);

	void closeEvent(QCloseEvent*);

	//settings

	void loadDefaults();
	void saveDefaults();

	void loadSettings();
	virtual void loadSettings(QString);
	void loadSettings(QSettings *);
	void saveSettings();
	void saveSettings(QSettings *);
	
	// python stuff
	void set(QString, QVariant, int=1);
	QVariant get(QString, int=1);
	nPhysD* getPhys(QString, int=1);
	void button(QString, int=1);	

};



#endif
