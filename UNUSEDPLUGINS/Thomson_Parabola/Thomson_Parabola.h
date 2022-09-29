/*
 * This file is NOT open source. 
 *
 * Any use is FORBIDDEN unless by written permission
 *
 * (C) Alessandro Flacco 2014
 *
 */


#include <QtGui>
#include <QWidget>
#include <QBoxLayout>
#include <QPoint>
#include <QPolygon>

#include "nPlug.h"
#include "nGenericPan.h"

// you should include here the relevant (if any) ui_??.h
#include "ui_Thomson_Parabola.h"

#include "tpGlDraw.h"

// data manipulation
#include "tpSystem.h"

#ifndef __tp_plugin_plugin
#define __tp_plugin_plugin

class neutrino;
class nSpectrum; // derivation of nLine	

struct simulParams {
	simulParams()
		: A(1), Z(1), iE_mev(1), eE_mev(10), n_points(10)
	{ nsp = NULL; }


	int A;
	int Z;
	float iE_mev;
	float eE_mev;
	int n_points;

	nSpectrum *nsp;
};


// This object does the real work, here you write a nGenericPan as if it were in the main tree
class Thomson_Parabola : public nGenericPan {
Q_OBJECT
public:
    Q_INVOKABLE Thomson_Parabola(neutrino *);

	Ui::tpPlugin my_w;
	
	tpSystem *my_tp;

	QVector<nSpectrum *> my_tracks;
	//nSpectrum *my_line;

public slots:

	// here the GUI slots
	void vecInput(f3point);
	
//	void load_config(void);
//	void save_config(void);
	void run_simulation(void);

	void addTrack(void);
	void removeTrack(void);
	void updateTracks(void);
	void updateSingleTrack(int, int);

protected:
	tpGlDraw *tpDraw;

protected slots:
	//void tpRedraw(void);

private:
	// here your private stuff
	simulParams sp;

};

NEUTRINO_PLUGIN(Thomson_Parabola)

#endif
