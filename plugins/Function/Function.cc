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



inline double my_rand(double my_max) {
   return my_max*static_cast<double>(rand())/RAND_MAX;
}

fPhys::fPhys(nPhysD *physparent) : exprtk::ifunction<double>(2), my_phys(physparent) {
    exprtk::disable_has_side_effects(*this);
}

fPhys::~fPhys() {}
double fPhys::operator()(const double& x, const double& y) {
    return my_phys ? my_phys->getPoint(x,y) : std::numeric_limits<double>::quiet_NaN();
}


physFunc3::physFunc3(Function *fparent) : exprtk::ifunction<double>(3), mylist(fparent->nparent->getBufferList()) {
    exprtk::disable_has_side_effects(*this);
}
physFunc3::~physFunc3() {}

double physFunc3::operator()(const double &imgnum, const double& x, const double& y) {
    int imgnumint=static_cast<int>(imgnum);
    if (imgnumint >= 0  && imgnumint < mylist.size() && mylist[imgnumint]) {
        return mylist[imgnumint]->getPoint(x,y);
    } else {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

Function::Function(neutrino *mynparent) : nGenericPan(mynparent) {
    setupUi(this);
    toolBar->addWidget(sb_width);
    toolBar->addWidget(sb_height);

    QKeySequence key_seq=QKeySequence(Qt::CTRL | Qt::Key_Return);
    function->setToolTip("Press "+key_seq.toString(QKeySequence::NativeText)+" to execute"+function->toolTip());
    QShortcut* my_shortcut = new QShortcut(key_seq, function);
    connect(my_shortcut, SIGNAL(activated()), this, SLOT(on_doIt_released()));

    show();
}

void Function::on_actionCopy_Size_triggered() {
    // qDebug() << "here";
    if (currentBuffer) {
        auto size=currentBuffer->getSize();
        sb_width->setValue(size.x());
        sb_height->setValue(size.y());
    }

}

void Function::on_doIt_released() {
    saveDefaults();

    if (sb_width->value()<1 || sb_height->value()<1) {
        statusbar->showMessage(tr("Width and height must be at least 1"),5000);
        return;
    }

    // previous results are removed only once the new ones exist (a typo must not lose them)
    std::vector<nPhysD*> previous;
    if (erasePrevious->isChecked()) {
        previous.swap(physFunction);
    }

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

    symbol_table.add_function("rand",my_rand);

    // physN: N padded to the digits of the last index (11 images: phys00..phys10);
    // the unpadded name (phys3) is registered too, so both spellings work
    std::list<fPhys*> my_list;
    const int nimages=nparent->getBufferList().size();
    const int digits=QString::number(std::max(0,nimages-1)).size();
    for (int i=0; i<nimages; i++) {
        fPhys *my_fun= new fPhys(nparent->getBuffer(i));
        my_list.push_back(my_fun);
        const std::string padded="phys"+QString::number(i).rightJustified(digits,'0').toStdString();
        const std::string plain="phys"+std::to_string(i);
        symbol_table.add_function(padded,*my_fun);
        if (plain!=padded) symbol_table.add_function(plain,*my_fun);
    }

    // the progress dialog runs the event loop: an image closed meanwhile reads as NaN, not freed memory
    QMetaObject::Connection forgetClosed=connect(nparent, &neutrino::physDel, this, [&](nPhysD *closed) {
        my_func.forget(closed);
        mf.forget(closed);
        for (auto &f : my_list) f->forget(closed);
    });

    symbol_table.add_constant("width",sb_width->value());
    symbol_table.add_constant("height",sb_height->value());

    symbol_table.add_constants();

    exprtk::expression<double> my_exprtk;
    my_exprtk.register_symbol_table(symbol_table);

    // several images: expressions separated by a line containing only <br>
    QString text=function->toPlainText();
    text.replace("\r\n","\n");
    QStringList functions_str=text.split(QRegularExpression("^\\s*<br>\\s*$", QRegularExpression::MultilineOption), Qt::SkipEmptyParts);

    std::vector<nPhysD*> created;
    for (auto &function_str :functions_str) {
        if (function_str.trimmed().isEmpty()) continue;

        exprtk::parser<double> parser;
        if (parser.compile(function_str.toStdString(),my_exprtk)) {
            nPhysD *my_phys = new nPhysD(static_cast<unsigned int>(sb_width->value()), static_cast<unsigned int>(sb_height->value()), 0.0, "");
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
            QString name="Function("+function_str.trimmed()+", "+QString::number(my_phys->getW())+", "+QString::number(my_phys->getH())+")";

            my_phys->TscanBrightness();
            my_phys->setName(name.toStdString());
            my_phys->setShortName("Function");

            created.push_back(my_phys);
            physFunction.push_back(my_phys);
            nparent->addShowPhys(my_phys);
            erasePrevious->setEnabled(true);

        } else {
            statusbar->showMessage(QString::fromStdString(parser.error())+" Check log",5000);
            for (std::size_t i = 0; i < parser.error_count(); ++i) {
                exprtk::parser_error::type error = parser.get_error(i);
                QTextCursor my_cursor=function->textCursor();
                my_cursor.setPosition(error.token.position);
                function->setTextCursor(my_cursor);
                qWarning() << "Error #" << i << error.token.position << QString::fromStdString(exprtk::parser_error::to_str(error.mode)) << QString::fromStdString(error.diagnostic.c_str());
            }
        }
    }
    disconnect(forgetClosed);

    if (created.empty()) {
        // nothing new: keep the previous results
        physFunction.insert(physFunction.begin(), previous.begin(), previous.end());
    } else {
        for (auto &my_phys : previous) {
            nparent->removePhys(my_phys); // ignores images already closed
        }
    }

    for (auto &my_fun: my_list) {
        delete my_fun;
    }
}
