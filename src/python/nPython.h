#ifndef __npython
#define __npython

#include <cmath>
#include <PythonQt.h>
//#include <PythonQt_QtAll.h>

#include <QtGui>
#include <QWidget>
#include <QShortcut>

#include "nGenericPan.h"
#include "ui_nPython.h"


class neutrino;
/**
 @short Python Subsystem

 This class provides a simple implementation of python bindings via PythonQt library
 */
class nPython : public nGenericPan {
	Q_OBJECT
	
public:
	
    nPython(neutrino *);
	Ui::nPython my_w;

public slots:
	void loadScript(void);
	void runScript(void);
    void runScript(QString);
    void changeScriptsFolder();
    void changeSiteFolder();
};

#endif

