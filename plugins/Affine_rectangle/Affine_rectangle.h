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

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>   


#include "nGenericPan.h"
#include "ui_Affine_rectangle.h"
#include "nLine.h"
#include "nRect.h"


#ifndef Affine_rectangle_
#define Affine_rectangle_

class neutrino;

class Affine_rectangle : public nGenericPan {
    Q_OBJECT

public:	
    Q_INVOKABLE Affine_rectangle(neutrino *);

    Ui::Affine_rectangle my_w;

    nPhysD *affined;

    nLine l1;

    vec2f affine(vec2f, std::array<double,6>&);

public slots:
    std::array<double,6> getAffine(QPolygonF, QPolygonF);
    void affine();
    void bufferChanged (nPhysD*) override;
    void resetPoints();
};

NEUTRINO_PLUGIN(Affine_rectangle, Analysis);


#endif
