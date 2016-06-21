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

#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"

#include "nCustomPlots.h"
class nVisarPlot : public nCustomPlotMouseX {
    Q_OBJECT
public:
    nVisarPlot(QWidget* parent):
        nCustomPlotMouseX(parent),
        yAxis3(axisRect(0)->addAxis(QCPAxis::atRight,0))
    {
        yAxis2->setVisible(true);

        yAxis->setLabelColor(Qt::red);
        yAxis->setTickLabelColor(Qt::red);

        yAxis2->setLabelColor(Qt::blue);
        yAxis2->setTickLabelColor(Qt::blue);

        yAxis3->setLabelColor(Qt::darkCyan);
        yAxis3->setTickLabelColor(Qt::darkCyan);

        yAxis3->setLabelPadding(-1);

        QSettings settings("neutrino","");
        settings.beginGroup("Preferences");
        QVariant fontString=settings.value("defaultFont");
        if (fontString.isValid()) {
            QFont fontTmp;
            if (fontTmp.fromString(fontString.toString())) {
                    yAxis3->setTickLabelFont(fontTmp);
                    yAxis3->setLabelFont(fontTmp);
            }
        }
    };
    QCPAxis *yAxis3;
};

#include "ui_nVISAR1.h"
#include "ui_nVISAR2.h"
#include "ui_nVISAR3.h"

#include "nPhysWave.h"


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
    Ui::nVISAR1 my_w;

    Ui::nVISAR2 visar[2];
    Ui::nVISAR3 setvisar[2];

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

    void export_pdf();
    void export_clipboard();

    void connections();
    void disconnections();

    void updatePlotSOP();

    void tabChanged(int=0);

    void mouseAtMatrix(QPointF);

    void loadSettings(QString);

    void bufferChanged(nPhysD*);

    
private:

    QVector<double> cPhase[2][2], cIntensity[2][2], cContrast[2][2];
    QVector<double> time_phase[2];

    QVector<double> velocity[2], reflectivity[2], quality[2];
    QVector<double> time_vel[2];

    QVector<double> sopCurve[4];
    QVector<double> time_sop;

    nPhysD phase[2][2];
    nPhysD contrast[2][2];
    nPhysD intensity[2][2];

    QPointer<nLine> fringeLine[2];
    QPointer<nRect> fringeRect[2];
    QPointer<nRect> sopRect;
};


#endif
