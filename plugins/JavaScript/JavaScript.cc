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

JavaScript::JavaScript(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);
    engine.globalObject().setProperty("neu", engine.newQObject(nparent));
    show();

    const int typeId = qRegisterMetaType<nGenericPan*>("nGenericPan*");
    QScriptValue prototype = engine.newQObject(new nGenericPan(nparent));
    engine.setDefaultPrototype(typeId, prototype);
    engine.setDefaultPrototype(qMetaTypeId<nGenericPan*>(), prototype);

    QScriptValue namespaceObj = engine.newObject();
    engine.globalObject().setProperty("nGenericPan", namespaceObj);


}

void JavaScript::on_command_returnPressed() {
    saveDefaults();
    qDebug() << command->text();
    QScriptValue retval = engine.evaluate(command->text());
    output->setPlainText(retval.toString());
}
