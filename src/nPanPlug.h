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
#ifndef npanplug_H
#define npanplug_H

#include <QString>
#include <QIcon>
#include <QKeySequence>
#include <functional>
#include <vector>

class neutrino;
class nGenericPan;

// The plugins are compiled into libNeutrino (OBJECT libraries, see resources/cmake/NeutrinoPlugin.cmake)
// and register themselves with NEUTRINO_PLUGIN (at the end of their header):
// neutrino builds its menus and toolbar from nPanPlug::all(), without knowing the plugin headers.
// The registration runs during the static initialisation, before the QApplication exists:
// the icon and the shortcut are made only when the menus are built
struct nPanPlug {
    QString name;                               // class name (openPan, sessions); menu text with spaces for _
    QString menuEntry;                          // menus and submenus, e.g. "Analysis;Lineout" (empty: Analysis)
    std::function<QIcon()> icon;                // a plugin with an icon also gets a toolbar button
    std::function<QKeySequence()> shortcut;
    int order;                                  // position of the toolbar button
    std::function<nGenericPan*(neutrino*)> create;

    // all the plugins, sorted by name
    static const std::vector<nPanPlug>& all();
    // returns nullptr if there is no plugin with that name
    static const nPanPlug* find(const QString &name);
    // the plugin registered with a name already present is ignored
    static bool add(nPanPlug plug);

private:
    static std::vector<nPanPlug>& registry();
};

// NEUTRINO_PLUGIN(class, menu entry, icon, shortcut, order): only the class is needed.
// The inline variable is initialised once, even if the header is included by several files (moc, other plugins)
#define NEUTRINO_PLUGIN5(__class_name,__menu_entry, __menu_icon, __key_shortcut, __order) \
    inline const bool __class_name##_registered = nPanPlug::add({ \
        #__class_name, #__menu_entry, \
        [] { return QIcon(__menu_icon); }, \
        [] { return QKeySequence(__key_shortcut); }, \
        int(__order), \
        [](neutrino *neu) -> nGenericPan* { return new __class_name(neu); } });

#define NEUTRINO_PLUGIN4(__class_name,__menu_entry, __menu_icon, __key_shortcut) NEUTRINO_PLUGIN5(__class_name,__menu_entry, __menu_icon, __key_shortcut, 1 )
#define NEUTRINO_PLUGIN3(__class_name,__menu_entry, __menu_icon) NEUTRINO_PLUGIN4(__class_name,__menu_entry, __menu_icon, )
#define NEUTRINO_PLUGIN2(__class_name,__menu_entry) NEUTRINO_PLUGIN3(__class_name,__menu_entry, )
#define NEUTRINO_PLUGIN1(__class_name) NEUTRINO_PLUGIN2(__class_name, )

#define NEUTRINO_PLUGIN_B(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,...) arg9
#define NEUTRINO_PLUGIN_A(...) NEUTRINO_PLUGIN_B(__VA_ARGS__,NEUTRINO_PLUGIN8,NEUTRINO_PLUGIN7,NEUTRINO_PLUGIN6,NEUTRINO_PLUGIN5,NEUTRINO_PLUGIN4,NEUTRINO_PLUGIN3,NEUTRINO_PLUGIN2,NEUTRINO_PLUGIN1,)
#define NEUTRINO_PLUGIN(...) NEUTRINO_PLUGIN_A(__VA_ARGS__)(__VA_ARGS__)

#endif
