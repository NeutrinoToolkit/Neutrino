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
#include "ui_Affine_transformation.h"
#include "nLine.h"


#ifndef __Affine_transformation
#define __Affine_transformation

class neutrino;

class Affine_transformation : public nGenericPan {
    Q_OBJECT

public:	
    Q_INVOKABLE Affine_transformation(neutrino *);

    Ui::Affine_transformation my_w;

    nPhysD *affined;

    nLine l1, l2;

    vec2f affine(vec2f, std::vector<double>);

    std::vector<double> forward, backward;

public slots:
    std::vector<double> getAffine(QPolygonF, QPolygonF);
    void apply();
    void affine();
    void bufferChanged(nPhysD*);

};

NEUTRINO_PLUGIN(Affine_transformation, Analysis);


#endif
