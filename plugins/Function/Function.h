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

#ifndef Function_H
#define Function_H

#include "nGenericPan.h"
#include "ui_Function.h"
#include "neutrino.h"
#include "exprtk.hpp"




class Function : public nGenericPan, private Ui::Function {
    Q_OBJECT

public:

    Q_INVOKABLE Function(neutrino *);

public slots:
    void on_doIt_released();
    void on_actionCopy_Size_triggered();

private:

    std::vector<nPhysD*> physFunction;


};

NEUTRINO_PLUGIN(Function,Analysis);

struct fPhys : public exprtk::ifunction<double>
{
    fPhys(nPhysD *);
    virtual ~fPhys() override;

public:
    virtual double operator()(const double&, const double&) override;

private:
    nPhysD* my_phys;
};

struct physFunc3 : public exprtk::ifunction<double>
{
    physFunc3(Function*);
    virtual ~physFunc3() override;

public:
    virtual double operator()(const double &, const double&, const double&) override;

private:
    QList<nPhysD *> mylist;
};


#endif
