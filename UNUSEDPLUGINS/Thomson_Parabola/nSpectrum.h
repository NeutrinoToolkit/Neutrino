// nLine overloading for spectrum manipulation
//
// (forse la strada piu' semplice..)

#include <QtGui>

#include <iostream>

#include "nLine.h"
#include "neutrino.h"

#ifndef __nSpectrum
#define __nSpectrum

class neutrino;
class nGenericPan;

class nSpectrum : public nLine {
	Q_OBJECT

public:

    nSpectrum (nGenericPan * = NULL);

	~nSpectrum()
	{ }

	// reimplement getStringData method from nLine
	QString getStringData(QPolygonF);

public slots:
	void updatePlot();
	void export_txt();


};

#endif
