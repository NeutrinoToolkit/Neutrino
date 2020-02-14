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

Function::Function(neutrino *nparent) : nGenericPan(nparent),
    physFunction(nullptr)
{
    setupUi(this);
    connect(doIt,SIGNAL(released()), this, SLOT(on_function_returnPressed()));
    show();
}

void Function::on_function_returnPressed() {
    saveDefaults();
    nPhysD *my_phys=nullptr;
    if (radioImage->isChecked()) {
        nPhysD *orig=getPhysFromCombo(image);
        if (orig) {
            my_phys = new nPhysD(*orig);
        }
    } else {
        my_phys = new nPhysD(sb_width->value(), sb_height->value(), 0.0, "");
    }
    if (my_phys) {

        engine.globalObject().setProperty("width", (int)my_phys->getW());
        engine.globalObject().setProperty("height", (int)my_phys->getH());

        engine.evaluate("function neutrinoFunction(x,y,z) { return "+function->toPlainText()+"; }");
        if (engine.uncaughtException().isValid()) {
            qDebug() << engine.uncaughtExceptionBacktrace().join(" ");
            qDebug() << engine.uncaughtException().toString();
            QMessageBox::critical(this,"Error",engine.uncaughtException().toString()+"\n"+engine.uncaughtExceptionBacktrace().join("\n"),  QMessageBox::Ok);
        } else {
            QProgressDialog progress("", "Cancel", 0, my_phys->getSurf(), this);
            progress.setCancelButton(0);
            progress.setWindowModality(Qt::WindowModal);
            progress.setValue(0);
            progress.show();
            QScriptValue neutrinoFunction = engine.globalObject().property("neutrinoFunction");

            for (int x=0; x<(int)my_phys->getW(); x++) {
                for (int y=0; y<(int)my_phys->getH(); y++) {
                    double z=my_phys->point(x,y);
                    my_phys->set(x,y, neutrinoFunction.call(QScriptValue(), QScriptValueList() << x << y << z).toNumber());
                    progress.setValue(y+x*my_phys->getH());
                }
            }
            QString name="function("+function->toPlainText()+", ";
            if (radioImage->isChecked()) {
                name.append(QString::fromStdString(my_phys->getName()));
            } else {
                name.append(QString::number(my_phys->getW())+", "+QString::number(my_phys->getH()));
            }
            name.append(")");
            my_phys->TscanBrightness();
            my_phys->setName(name.toStdString());
            my_phys->setShortName("function");

            if (erasePrevious->isChecked()) {
                physFunction=nparent->replacePhys(my_phys,physFunction,true);
            } else {
                nparent->addShowPhys(my_phys);
                physFunction=my_phys;
            }

            erasePrevious->setEnabled(true);
        }
    }
}
