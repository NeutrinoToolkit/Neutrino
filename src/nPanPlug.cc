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
#include <QApplication>
#include "nPanPlug.h"
#include "nGenericPan.h"

bool nPanPlug::instantiate(neutrino *neu) {
    QPointer<QTranslator> translator(new QTranslator(qApp));
    if (translator->load(QLocale(), QLatin1String(name()), QLatin1String("_"), QLatin1String(":/translations"))) {
        qDebug() << "installed translation" << QLocale() << name();
        qApp->installTranslator(translator);
    } else {
        delete translator;
    }

    const QByteArray className(name()+"*");
    const int type = QMetaType::type( className );
    if(type != QMetaType::UnknownType) {
        const QMetaObject *mo = QMetaType::metaObjectForType(type);
        if(mo) {
            QObject *objectPtr = mo->newInstance(Q_ARG(neutrino*,neu));
            if(objectPtr) {
                my_pan=qobject_cast<nGenericPan*>(objectPtr);
                return true;
            }
        }
    }
    return false;
}

bool nPanPlug::unload() {
    if (my_pan) {
        my_pan->deleteLater();
        QApplication::processEvents();
        return true;
    } else {
        return false;
    }
}

