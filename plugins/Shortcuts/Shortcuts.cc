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
#include "neutrino.h"
#include "Shortcuts.h"
#include <QMenu>
Shortcuts::Shortcuts(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);
    setWindowFlags(Qt::Tool);
    
    QFile lic(":/html/Shortcuts.html");
    if (lic.open(QFile::ReadOnly | QFile::Text)) {
        // QString htmltext=QTextStream(&lic).readAll().replace("%CTRL%",QKeySequence(Qt::CTRL).toString(QKeySequence::NativeText)).replace("%OPTION%",QKeySequence(Qt::META).toString(QKeySequence::NativeText)).replace("%COMMAND%",QKeySequence(Qt::ALT).toString(QKeySequence::NativeText));
        // the menu shortcuts from the actions themselves (the fixed list was out of date)
        QString htmltext=QTextStream(&lic).readAll().replace("%MENUS%",menuShortcuts()).replace("%CTRL%",QKeySequence(Qt::CTRL).toString(QKeySequence::NativeText)).replace("%OPTION%",QKeySequence(Qt::META).toString(QKeySequence::NativeText)).replace("%COMMAND%",QKeySequence(Qt::ALT).toString(QKeySequence::NativeText));
        textBrowser->insertHtml(htmltext);
    }
    show(true);
}

// the shortcuts of all the actions of neutrino and of the plugins, by menu;
// a shortcut used by more than one action is shown in red (Qt runs none of them)
QString Shortcuts::menuShortcuts() {
    QMap<QString, QList<QPair<QString,QString>>> groups; // menu -> (action, keys)
    QMap<QString, QStringList> byKey;                     // key -> actions
    foreach (QAction *action, nparent->findChildren<QAction*>()) {
        if (action->shortcut().isEmpty()) continue;
        // only the actions in a menu/toolbar: the others have no active shortcut
        if (action->associatedObjects().isEmpty()) continue;
        QString menuName=tr("Other");
        for (QObject *obj : action->associatedObjects()) {
            QMenu *menu=qobject_cast<QMenu*>(obj);
            if (menu) {
                QStringList path;
                for (QMenu *m=menu; m; m=qobject_cast<QMenu*>(m->parentWidget())) {
                    path.prepend(m->title().remove('&'));
                }
                menuName=path.join(" / ");
                break;
            }
        }
        QStringList keys;
        for (auto &seq : action->shortcuts()) {
            keys << seq.toString(QKeySequence::NativeText);
            byKey[seq.toString(QKeySequence::PortableText)] << action->text().remove('&');
        }
        groups[menuName] << qMakePair(action->text().remove('&'), keys.join(", "));
    }
    QString html;
    for (auto it=groups.begin(); it!=groups.end(); ++it) {
        auto list=it.value();
        std::sort(list.begin(), list.end());
        html+="<B>"+it.key().toHtmlEscaped()+"</B><BR>\n";
        for (auto &entry : list) {
            html+=entry.second.toHtmlEscaped()+" : "+entry.first.toHtmlEscaped()+"<BR>\n";
        }
        html+="<HR>\n";
    }
    QStringList conflicts;
    for (auto it=byKey.begin(); it!=byKey.end(); ++it) {
        if (it.value().size()>1) {
            conflicts << QKeySequence(it.key()).toString(QKeySequence::NativeText).toHtmlEscaped()+" : "+it.value().join(", ").toHtmlEscaped();
        }
    }
    if (!conflicts.isEmpty()) {
        html.prepend("<FONT COLOR=\"red\"><B>"+tr("Shortcuts used more than once (they do not work):")+"</B><BR>\n"+conflicts.join("<BR>\n")+"</FONT><HR>\n");
    }
    return html;
}


