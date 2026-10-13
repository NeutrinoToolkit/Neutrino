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
#ifndef Visar_H
#define Visar_H

#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"

#include "nCustomPlots.h"

#include "nPhysWave.h"
#include "neutrino.h"
#include <array>
#include <memory>
#include <vector>


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

#include "ui_Visar.h"

class nLine;
class nRect;

namespace Ui {
    class VisarSettings;
    class VisarVelocity;
}

template<class T>
inline T SIGN(T x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }




// everything belonging to a single VISAR: widgets, image objects, intermediate images and curves.
// It is the context object of all its connections, so they die with it.
class VisarChannel : public QObject {
    Q_OBJECT

public:
    VisarChannel(unsigned int, QObject*);
    ~VisarChannel();

    unsigned int k; // position in Visar::visars (and in the tabs): change it only via Visar::renumberVisar

    std::unique_ptr<Ui::VisarSettings> settingsUi;
    std::unique_ptr<Ui::VisarVelocity> velocityUi;
    QPointer<QWidget> tabSettings, tabVelocity;
    QPointer<QAction> actionRect;

    QPointer<nLine> fringeLine;
    QPointer<nRect> fringeRect;
    QPointer<nLine> maskRegion;
    nPhysD* ghostPhys=nullptr; // owned by neutrino

    QPointer<QCPGraph> graphVelocity, graphReflectivity, graphQuality;
    QPointer<QCPErrorBars> errVelocity, errReflectivity;

    std::vector<double> sweepCoeff;

    std::array<nPhysD,2> phaseUnwrap, contrast, intensity; // [ref/shot]

    std::array<QVector<double>,2> cPhase, cIntensity, cContrast; // [ref/shot][time]
    QVector<double> time_phase, cPhaseErr, cReflErr;
    QVector<double> velocity, velError, reflectivity, reflError, quality, time_vel;
};


class Visar : public nGenericPan, private Ui::Visar {
    Q_OBJECT
    
    using nGenericPan::nGenericPan;
    using nGenericPan::loadSettings;

public:
    
    Q_INVOKABLE Visar(neutrino *);
    ~Visar() override;

private:

    std::vector<std::unique_ptr<VisarChannel>> visars;

    double getTime(std::vector<double> &vecsweep,double p);
    bool parseSweep(QLineEdit*, bool, std::vector<double>&);
    void refreshWaveIcon();
    void nameVisar(VisarChannel&);
    void renumberVisar(VisarChannel&, unsigned int);

    std::vector<double> sweepCoeffSOP;

    std::array<QVector<double>,3> sopCurve;
    QVector<double> time_sop;

    QPointer<nRect> sopRect;

public slots:

    void fillComboShot();
    void changeShot(QString);

    void addVisar();
    void delVisar();
    void delVisarAsk();
    void removeVisar(unsigned int);

    void refreshGlob(int);
    void getPhysFromNameSetCombo(QFileInfo, QComboBox*);

    void needWave(unsigned int);

    void doWave();
    void doWave(unsigned int);
    
    void getCarrier(unsigned int);
    
    void getPhase();
    void getPhase(unsigned int);
    
    void updatePlot();
    
    int direction(int);
    
    void export_txt();
    void export_txt_multiple();
    
    QString export_one(unsigned int);
    QString export_sop();
    
    void export_clipboard();
    
    void updatePlotSOP();
    
    void tabChanged(int);
    
    void mouseAtMatrix(QPointF) override;
    void imageMousePress(QPointF) override;
    void imageMouseRelease(QPointF) override;

    QPointF getTimeSpaceFromPixel(QPointF);

    void mouseAtPlot(QMouseEvent* e);
    
    void loadSettings(QString=QString()) override;
    
    void setObjectVisibility(nPhysD*);
    
    void sweepChanged();
    void sweepChanged(unsigned int);
    void sweepChangedSOP();
    
    void calculate_etalon();
    
    void copy_image();

    void on_actionNext_triggered();
    void on_actionPrevious_triggered();

};

NEUTRINO_PLUGIN(Visar,Analysis);

#endif
