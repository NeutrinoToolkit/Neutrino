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
        // QString htmltext=QTextStream(&lic).readAll().replace("%MENUS%",menuShortcuts()).replace("%CTRL%",QKeySequence(Qt::CTRL).toString(QKeySequence::NativeText)).replace("%OPTION%",QKeySequence(Qt::META).toString(QKeySequence::NativeText)).replace("%COMMAND%",QKeySequence(Qt::ALT).toString(QKeySequence::NativeText));
        // the keys of the gear menu of the pans (nGenericPan) as Qt writes them (the modifiers in the order of the
        // platform), and Alt for the reset (on macOS %OPTION% was Qt::META, i.e. Control)
        auto panKey=[](Qt::Key key) {
            return QKeySequence(Qt::CTRL | Qt::ALT | Qt::META | key).toString(QKeySequence::NativeText).toHtmlEscaped();
        };
        QString alt=QKeySequence(Qt::ALT | Qt::Key_A).toString(QKeySequence::NativeText);
        alt.chop(1); // (the modifier alone)
        if (alt.endsWith('+')) alt.chop(1);
        QString htmltext=QTextStream(&lic).readAll().replace("%MENUS%",menuShortcuts()).replace("%PAN_S%",panKey(Qt::Key_S)).replace("%PAN_J%",panKey(Qt::Key_J)).replace("%PAN_N%",panKey(Qt::Key_N)).replace("%ALT%",alt.toHtmlEscaped());
        textBrowser->insertHtml(htmltext);
    }
    show(true);
}

// the shortcuts of all the actions of neutrino and of the plugins, by menu;
// a shortcut used by more than one action, or also by the keys of the objects, is shown in red
// (Qt runs none of two actions; a menu shortcut is run instead of the key of the selected object)
QString Shortcuts::menuShortcuts() {
    QMap<QString, QList<QPair<QString,QString>>> groups; // menu -> (action, keys)
    QMap<QString, QStringList> byKey;                     // key -> actions
    foreach (QAction *action, nparent->findChildren<QAction*>()) {
        if (action->shortcut().isEmpty()) continue;
        // not the actions of the analysis windows (children of nparent): e.g. their gear menu, one per open window,
        // active only in its window (listed in the html)
        bool inPan=false;
        for (QObject *obj=action->parent(); obj && obj!=nparent; obj=obj->parent()) {
            if (qobject_cast<nGenericPan*>(obj)) { inPan=true; break; }
        }
        if (inPan) continue;
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
    // the keys of the selected objects (nLine, nObject, nPoint, keyPressEvent) and of the image (nView):
    // the menu shortcuts are run before them, an object never gets a key that is also a menu shortcut
    const QList<QPair<QString,QString>> objectKeys={
        {"Up", tr("move (selected object, mouse cursor)")}, {"Down", tr("move (selected object, mouse cursor)")},
        {"Left", tr("move (selected object, mouse cursor)")}, {"Right", tr("move (selected object, mouse cursor)")},
        {"Shift+Up", tr("move faster")}, {"Shift+Down", tr("move faster")},
        {"Shift+Left", tr("move faster")}, {"Shift+Right", tr("move faster")},
        {"Return", tr("stop adding points (line), click (image)")}, {"Esc", tr("stop adding points (line, rectangle)")},
        {"Backspace", tr("delete the selected object")},
        {"W", tr("properties window (line, rectangle, point)")}, {"B", tr("Bezier (line)")},
        {"C", tr("closed line, center the rectangle")}, {"Shift+C", tr("center the line, copy the rectangle points")},
        {"S", tr("antialias (line), submatrix (rectangle)")}, {"A", tr("append point (line)")},
        {"P", tr("prepend point (line)")}, {"D", tr("delete point (line)")},
        {"X", tr("horizontal (line), expand width (rectangle)")}, {"Y", tr("vertical (line), expand height (rectangle)")},
        {"R", tr("rectangle (line)")}, {"Shift+R", tr("expand to the image (line)")},
        {"E", tr("expand (rectangle)")}, {"I", tr("intersection with the image (rectangle)")},
        {"Shift+V", tr("paste the points (rectangle)")}};
    for (auto &key : objectKeys) {
        if (byKey.contains(key.first)) byKey[key.first] << tr("key of the objects: ")+key.second;
    }
    QStringList conflicts;
    for (auto it=byKey.begin(); it!=byKey.end(); ++it) {
        if (it.value().size()>1) {
            conflicts << QKeySequence(it.key()).toString(QKeySequence::NativeText).toHtmlEscaped()+" : "+it.value().join(", ").toHtmlEscaped();
        }
    }
    if (!conflicts.isEmpty()) {
        html.prepend("<FONT COLOR=\"red\"><B>"+tr("Shortcuts used more than once (only the menu one works, two menu ones: none):")+"</B><BR>\n"+conflicts.join("<BR>\n")+"</FONT><HR>\n");
    }
    return html;
}


