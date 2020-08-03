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

#include "JavaScript.h"
#include <QScriptEngine>
#include "nPhysImageF.h"

Q_DECLARE_METATYPE(QList<nGenericPan*>);
Q_DECLARE_METATYPE(QList<neutrino*>);

JavaScript::JavaScript(neutrino *nparent) : nGenericPan(nparent) {
    setupUi(this);
    engine.globalObject().setProperty("neu", engine.newQObject(nparent));
    engine.globalObject().setProperty("nApp", engine.newQObject(qApp));

    qScriptRegisterSequenceMetaType<QList<nGenericPan*> >(&engine);
    qScriptRegisterSequenceMetaType<QList<neutrino*> >(&engine);
    qScriptRegisterSequenceMetaType<QList<nPhysD*> >(&engine);

    qRegisterMetaType<nGenericPan*>("nGenericPan*");
    qRegisterMetaType<nPhysD*>("nPhysD*");

    show();
}

void JavaScript::on_command_returnPressed() {
    saveDefaults();
    QScriptValue retval;
    QString fname=command->text();
    if(QFileInfo::exists(fname)) {
        QFile t(fname);
        t.open(QIODevice::ReadOnly| QIODevice::Text);
        QTextStream out(&t);
        QString toRun=out.readAll();
        t.close();
        retval = engine.evaluate(toRun, fname);
    } else {
        qDebug() << command->text();
        retval = engine.evaluate(command->text());
    }
    output->setPlainText(retval.toString());
}
