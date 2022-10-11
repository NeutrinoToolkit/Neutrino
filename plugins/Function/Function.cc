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
#include "nPhysD.h"
#include "nPhysImageF.h"


Function::Function(neutrino *mynparent) : nGenericPan(mynparent),
    physFunction(nullptr)
{
    setupUi(this);
    toolBar->addWidget(sb_width);
    toolBar->addWidget(sb_height);
    show();
}

void Function::on_actionCopy_Size_triggered() {
    qDebug() << "here";
    if (currentBuffer) {
        auto size=currentBuffer->getSize();
        sb_width->setValue(size.x());
        sb_height->setValue(size.y());
    }

}

void Function::on_doIt_released() {
    saveDefaults();
    nPhysD *my_phys = new nPhysD(static_cast<unsigned int>(sb_width->value()), static_cast<unsigned int>(sb_height->value()), 0.0, "");

    double x, y;

    exprtk::symbol_table<double> symbol_table;
    symbol_table.add_variable("x",x);
    symbol_table.add_variable("y",y);
    symbol_table.add_constant("num_phys",nparent->getBufferList().size());

    physFunc3 mf(this);
    symbol_table.add_function("n_phys",mf);

    fPhys my_func(currentBuffer);
    if (currentBuffer) {
        symbol_table.add_function("phys",my_func);
    }

    std::list<fPhys*> my_list;
    for (int i=0; i<nparent->getBufferList().size(); i++) {
        std::ostringstream oss;
        oss << "phys" << std::setfill('0') << std::setw(int(log10(nparent->getBufferList().size()))) << i;
        qDebug() << QString::fromStdString(oss.str());
        fPhys *my_fun= new fPhys(nparent->getBuffer(i));
        my_list.push_back(my_fun);
        symbol_table.add_function(oss.str().c_str(),*my_fun);
    }

    symbol_table.add_constant("width",my_phys->getW());
    symbol_table.add_constant("height",my_phys->getH());

    symbol_table.add_constants();

    exprtk::expression<double> my_exprtk;
    my_exprtk.register_symbol_table(symbol_table);

    exprtk::parser<double> parser;
    if (parser.compile(function->toPlainText().toStdString(),my_exprtk)) {
        QProgressDialog progress("", "Cancel", 0, static_cast<int>(my_phys->getW()), this);
        progress.setCancelButton(nullptr);
        progress.setWindowModality(Qt::WindowModal);
        progress.setValue(0);
        progress.show();

        for (x=0; x < my_phys->getW(); x++) {
            progress.setValue(static_cast<int>(x));
            for (y=0; y < static_cast<double>(my_phys->getH()); y++) {
                my_phys->set(static_cast<unsigned int>(x),static_cast<unsigned int>(y), my_exprtk.value());
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
    } else {
        qWarning() << "Error: " << QString::fromStdString(parser.error());
        for (std::size_t i = 0; i < parser.error_count(); ++i) {
            exprtk::parser_error::type error = parser.get_error(i);
            QTextCursor my_cursor=function->textCursor();
            my_cursor.setPosition(error.token.position);
            function->setTextCursor(my_cursor);
            qWarning() << i << error.token.position << QString::fromStdString(exprtk::parser_error::to_str(error.mode)) << QString::fromStdString(error.diagnostic.c_str());
        }
    }
    for (auto &my_fun: my_list) {
        delete my_fun;
    }
}
