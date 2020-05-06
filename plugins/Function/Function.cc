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

#include "Function.h"
#include "nPhysImageF.h"
#include "exprtk.hpp"

struct physFunc : public exprtk::ifunction<double>
{
    physFunc(Function *fparent) : exprtk::ifunction<double>(3), parent(fparent) {
        exprtk::disable_has_side_effects(*this);
    }

    inline double operator()(const double &imgnum, const double& x, const double& y) {
        int imgnumint=(int)imgnum;
        nPhysD *my_phys(nullptr);
        QList<nPhysD *> mylist=parent->nparent->getBufferList();
        if (imgnumint >= 0  && imgnumint < mylist.size()) {
            my_phys = mylist[imgnumint];
        } else {
            my_phys=parent->currentBuffer;
        }
        if (my_phys) {
            return my_phys->getPoint(x,y);
        } else {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

private:
    Function* parent;
};


Function::Function(neutrino *nparent) : nGenericPan(nparent),
    physFunction(nullptr)
{
    setupUi(this);
    toolBar->addWidget(label_2);
    toolBar->addWidget(sb_width);
    toolBar->addWidget(label);
    toolBar->addWidget(sb_height);
    show();
}

void Function::on_doIt_released() {
    saveDefaults();
    nPhysD *my_phys = new nPhysD(sb_width->value(), sb_height->value(), 0.0, "");

    double x, y;

    exprtk::symbol_table<double> symbol_table;
    symbol_table.add_variable("x",x);
    symbol_table.add_variable("y",y);

    physFunc mf(this);
    symbol_table.add_function("phys",mf);

    symbol_table.add_constant("width",my_phys->getW());
    symbol_table.add_constant("height",my_phys->getH());

    symbol_table.add_constants();


    exprtk::expression<double> my_exprtk;
    my_exprtk.register_symbol_table(symbol_table);

    exprtk::parser<double> parser;
    parser.compile(function->toPlainText().toStdString(),my_exprtk);


    QProgressDialog progress("", "Cancel", 0, my_phys->getW(), this);
    progress.setCancelButton(0);
    progress.setWindowModality(Qt::WindowModal);
    progress.setValue(0);
    progress.show();

    for (x=0; x < my_phys->getW(); x++) {
        progress.setValue(x);
        for (y=0; y < my_phys->getH(); y++) {
            my_phys->set(x,y, my_exprtk.value());
        }
    }
    QString name="Function("+function->toPlainText()+", "+QString::number(my_phys->getW())+", "+QString::number(my_phys->getH())+")";

    my_phys->TscanBrightness();
    my_phys->setName(name.toStdString());
    my_phys->setShortName("Function");

    if (erasePrevious->isChecked()) {
        physFunction=nparent->replacePhys(my_phys,physFunction,true);
    } else {
        nparent->addShowPhys(my_phys);
        physFunction=my_phys;
    }

    erasePrevious->setEnabled(true);
}
