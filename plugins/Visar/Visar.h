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
#ifndef __Visar
#define __Visar

#include <array>
#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"

#include "nCustomPlots.h"

#include "nPhysWave.h"
#include "neutrino.h"

class VisarPhasePlot : public nCustomPlotMouseX3Y {
    Q_OBJECT
    
public:
    VisarPhasePlot(QWidget*);
};

class VisarPlot : public nCustomPlotMouseX3Y {
    Q_OBJECT
    
public:
    VisarPlot(QWidget*);
};

class nSOPPlot : public nCustomPlotMouseX2Y {
    Q_OBJECT
public:
    nSOPPlot(QWidget*);
};

#include "ui_Visar1.h"

class nLine;
class nRect;
namespace Ui {
class Visar2;
class Visar3;
}

template<class T>
inline T SIGN(T x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }




class Visar : public nGenericPan, private Ui::Visar1 {
    Q_OBJECT
    
    using nGenericPan::loadSettings;
    using Ui::Visar1::retranslateUi;
public:
    
    Q_INVOKABLE Visar(neutrino *);
    ~Visar(){}

private:

    std::vector<Ui::Visar2*> velocityUi;
    std::vector<Ui::Visar3*> phaseUi;
    
    double getTime(std::vector<double> &vecsweep,double p);
    
    std::vector<std::vector<double>> sweepCoeff;
    std::vector<double> sweepCoeffSOP;

    unsigned int numVisars;

    std::array<std::vector<QVector<double>>,2> cPhase, cIntensity, cContrast; // indices = 1st: ref/shot. 2nd: numVisar. 3rd: time
    std::vector<QVector<double>> time_phase;
    std::vector<QVector<double>> cPhaseErr;

    std::vector<QVector<double>> velocity, velError, reflectivity, reflError, quality, time_vel;

    std::array<QVector<double>,4> sopCurve;
    QVector<double> time_sop;

    std::vector<std::array<nPhysD,2>> phase;

    std::vector<std::array<nPhysD,2>> contrast;
    std::vector<std::array<nPhysD,2>> intensity;

    std::vector<nLine*> fringeLine;
    std::vector<nRect*> fringeRect;
    QPointer<nRect> sopRect;

    std::vector<std::map<std::string, nPhysD *>> localPhys;

public slots:
    
    void addVisar();
    void delVisar();

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
    
    QString export_one(unsigned int);
    QString export_sop();
    
    void export_clipboard();
    
    void connections();
    void disconnections();
    
    void updatePlotSOP();
    
    void tabChanged(int);
    
    void mouseAtMatrix(QPointF);
    
    void mouseAtPlot(QMouseEvent* e);
    
    void loadSettings(QString);
    
    void bufferChanged(nPhysD*);
    
    void sweepChanged(QLineEdit*line=nullptr);
    
    void changeEvent(QEvent *e)
    {
        qDebug() << panName() << e;
        
        QWidget::changeEvent(e);
        switch (e->type()) {
            case QEvent::LanguageChange: {
                    retranslateUi(this);
                    break;
                }
            default:
                break;
        }
    }

    void calculate_etalon();
    void on_showFiltered_triggered(bool);
    

};

NEUTRINO_PLUGIN(Visar,Analysis);

#endif
