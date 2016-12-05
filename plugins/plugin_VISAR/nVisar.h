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
#ifndef __nVisar
#define __nVisar

#include <array>
#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"

#include "nCustomPlots.h"

#include "nPhysWave.h"

class nVisarPhasePlot : public nCustomPlotMouseX3Y {
    Q_OBJECT

public:
    nVisarPhasePlot(QWidget*);
};

class nVisarPlot : public nCustomPlotMouseX3Y {
    Q_OBJECT

public:
    nVisarPlot(QWidget*);
};

class nSOPPlot : public nCustomPlotMouseX2Y {
    Q_OBJECT
public:
    nSOPPlot(QWidget*);
};

#include "ui_nVISAR1.h"
#include "ui_nVISAR2.h"
#include "ui_nVISAR3.h"

class neutrino;
class nLine;
class nRect;


template<class T>
inline T SIGN(T x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }




class nVisar : public nGenericPan {
    Q_OBJECT

    using nGenericPan::loadSettings;

public:

    nVisar(neutrino *, QString);
    ~nVisar();
    Ui::nVISAR1 my_w;

    std::array<Ui::nVISAR2,2> visar;
    std::array<Ui::nVISAR3,2> setvisar;

    double getTime(int k,double p);

    std::array<std::vector<double>, 3> sweepCoeff;

public slots:

    void doWave();
    void doWave(int);

    void getCarrier();
    void getCarrier(int);

    void getPhase();
    void getPhase(int);

    void updatePlot();

    int direction(int);

    void export_txt();
    void export_txt_multiple();

    QString export_one(int);
    QString export_sop();

    void export_clipboard();

    void connections();
    void disconnections();

    void updatePlotSOP();

    void tabChanged(int=0);

    void mouseAtMatrix(QPointF);

    void mouseAtPlot(QMouseEvent* e);

    void loadSettings(QString);

    void bufferChanged(nPhysD*);

    void sweepChanged(QLineEdit*line=nullptr);

private:

    std::array<std::array<QVector<double>,2>,2> cPhase, cIntensity, cContrast;
    std::array<QVector<double>,2> time_phase;

    std::array<QVector<double>,2> velocity, reflectivity, quality, time_vel;

    std::array<QVector<double>,4> sopCurve;
    QVector<double> time_sop;

    std::array<std::array<nPhysD,2>,2> phase;
    std::array<std::array<nPhysD,2>,2> contrast;
    std::array<std::array<nPhysD,2>,2> intensity;

    std::array<QPointer<nLine>,2> fringeLine;
    std::array<QPointer<nRect>,2> fringeRect;
    QPointer<nRect> sopRect;
};


#include "nPlug.h"
// the nVisarPlug object is in charge of reconstructing connections (runtime) with neutrino. It is in charge
// of the real object instantiation.
class nVisarPlug : public QObject, nPlug {
Q_OBJECT
Q_INTERFACES(nPlug)
Q_PLUGIN_METADATA(IID "org.neutrino.plug")

public:
    // The following methods are virtual and can be redifined:

//    nVisarPlug() { }
//    ~nVisarPlug() { std::cerr<<"~nVisarPlug"<<  std::endl; }
//    bool unload(){ return true;} // where we dismantle everything when politely asked to
    QString name() { return QString("Analysis;Visar"); } // this plugin is added to the Analysis menu

    // The following methods are pure virtual and must be redifined:

    // where the construction is performed
    bool instantiate(neutrino *neu) {
        my_GP = new nVisar(neu, "Visar");
        return true;
    }

    nGenericPan *my_GP;

};


#endif
