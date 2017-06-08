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
class Ui::Visar2;
class Ui::Visar3;

template<class T>
inline T SIGN(T x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }




class Visar : public nGenericPan, private Ui::Visar1 {
    Q_OBJECT
    
    using nGenericPan::loadSettings;
    using Ui::Visar1::retranslateUi;
public:
    
    Q_INVOKABLE Visar(neutrino *);
    ~Visar();
    
    std::array<Ui::Visar2*,2> velocityUi;
    std::array<Ui::Visar3*,2> phaseUi;
    
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

NEUTRINO_PLUGIN(Visar,Analysis);

#endif
