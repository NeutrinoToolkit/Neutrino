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
#include <iostream>

#include "nPlug.h"
#ifndef __npanplug
#define __npanplug

class neutrino;
class nGenericPan;

class nPanPlug : public nPlug {

public slots:

    nGenericPan* pan() { return my_pan;}

    bool unload() override;

    bool instantiate(neutrino *neu) override;

    virtual QString menuEntryPoint() { return QString(); }

    virtual QIcon icon() { return QIcon(); }

    virtual QKeySequence shortcut() { return QKeySequence(); }

    virtual int order() { return 0; }

protected:
    QPointer<nGenericPan> my_pan;

};

Q_DECLARE_INTERFACE(nPanPlug, "org.neutrino.plug")

#define NEUTRINO_PLUGIN5(__class_name,__menu_entry, __menu_icon, __key_shortcut, __order) Q_DECLARE_METATYPE(__class_name *); /*
*/ class __class_name ## Plug : public QObject, nPanPlug {  /*
*/Q_OBJECT  Q_INTERFACES(nPanPlug) Q_PLUGIN_METADATA(IID "org.neutrino.panPlug")  /*
*/public: /*
*/__class_name## Plug() {qRegisterMetaType<__class_name *>(name()+"*");} /*
*/QByteArray name() {return #__class_name;} /*
*/QString menuEntryPoint() { return QString(tr(#__menu_entry)); } /*
*/QIcon icon() {return QIcon(__menu_icon);} /*
*/QKeySequence shortcut() {return QKeySequence(__key_shortcut);} /*
*/int order() {return int(__order);}/*
*/};

#define NEUTRINO_PLUGIN4(__class_name,__menu_entry, __menu_icon, __key_shortcut) NEUTRINO_PLUGIN5(__class_name,__menu_entry, __menu_icon, __key_shortcut, 1 )
#define NEUTRINO_PLUGIN3(__class_name,__menu_entry, __menu_icon) NEUTRINO_PLUGIN4(__class_name,__menu_entry, __menu_icon, )
#define NEUTRINO_PLUGIN2(__class_name,__menu_entry) NEUTRINO_PLUGIN3(__class_name,__menu_entry, )
#define NEUTRINO_PLUGIN1(__class_name) NEUTRINO_PLUGIN2(__class_name, )
#define NEUTRINO_PLUGIN_B(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,...) arg9
#define NEUTRINO_PLUGIN_A(...) NEUTRINO_PLUGIN_B(__VA_ARGS__,NEUTRINO_PLUGIN8,NEUTRINO_PLUGIN7,NEUTRINO_PLUGIN6,NEUTRINO_PLUGIN5,NEUTRINO_PLUGIN4,NEUTRINO_PLUGIN3,NEUTRINO_PLUGIN2,NEUTRINO_PLUGIN1,)
#define NEUTRINO_PLUGIN(...) NEUTRINO_PLUGIN_A(__VA_ARGS__)(__VA_ARGS__)



#endif

