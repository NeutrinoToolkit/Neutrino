/*
 *
 *    Copyright (C) 2014 Alessandro Flacco, Tommaso Vinci All Rights Reserved
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

#include "nGenericPan.h"
#include "nPhysMaths.h"
#include "ui_FocalSpot.h"

#ifndef FocalSpot_H_
#define FocalSpot_H_

class neutrino;
class nLine;

class FocalSpot : public nGenericPan {
    Q_OBJECT

    using nGenericPan::loadSettings;

public:
    Q_INVOKABLE FocalSpot(neutrino *);

    Ui::FocalSpot my_w;

    QPointer<nLine> nContour;

    // python: Pan.results(): the quantities of the table and the radial profile
    QVariantMap results(QVariant what=QVariant()) override;

public slots:
    void calculate_stats();

    void bufferChanged(nPhysD*);

    void setPosZero(QPointF pos);
    void setOriginToCentre();

    void mouseAtPlot(QMouseEvent*);
    void mouseAtWorld(QPointF);

    void loadSettings(QString=QString());

    void on_actionClipboard_triggered();
    void on_actionTxt_triggered();

private:
    // blurred image: maximum, contour and FWHM
    nPhysD decimated;
    vec2f centre;          // spot centre used for the statistics (pixels)
    bool computing=false;  // calculate_stats may change the origin, which triggers it again

    // the results: group (one line of the view), sub-name, value, unit
    struct Row {
        QString group, sub; double value; QString unit;
        QString name() const { return sub.isEmpty() ? group : group+" "+sub; }
    };
    QList<Row> rows;
    QVector<double> radius, profile, encircled; // radial profile (world units) and encircled energy fraction

    void addRow(QString group, QString sub, double value, QString unit=QString());
    void showRows();
    QString resultsText();

    // the previous computation of the energies (Manual (old method)): {threshold %, contour %}
    QList<double> legacyEnergies(double th, double zl, const physMath::contour_integral &ci, bool contourOk);
};

NEUTRINO_PLUGIN(FocalSpot,Analysis);


#endif
