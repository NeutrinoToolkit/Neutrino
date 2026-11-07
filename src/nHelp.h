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
#ifndef nHelp_H_
#define nHelp_H_

#include <QMainWindow>
#include <QPointer>
#include <QMap>

class QTreeWidget;
class QTreeWidgetItem;
class QLineEdit;
class nHelpTextBrowser;
class neutrino;

// The help of Neutrino: one window for the application, with the general page (resources/help/README.md) and a
// page for each plugin with a help (its README.md, as html in the resource :/<plugin>README.html), listed on the
// left by menu. The Help button of a pan opens it on the page of the pan
class nHelp : public QMainWindow {
    Q_OBJECT

public:
    // shows the help (created the first time) on a page: the name of a plugin, its resource (":/...README.html"),
    // "Shortcuts" (the keyboard shortcuts of the window), or empty for the general page. window: the Neutrino
    // window whose shortcuts are listed (otherwise the last one given, or any)
    static nHelp* showPage(const QString &page=QString(), neutrino *window=nullptr);
    // the resource of the help of a plugin (empty if it has none)
    static QString pageFile(const QString &page);

protected:
    void closeEvent(QCloseEvent *e) override;

private:
    nHelp();
    void fillPages();
    void filter(const QString &text);
    void sourceChanged(const QUrl &url);
    void itemChosen(QTreeWidgetItem *item);

    static QPointer<nHelp> instance;
    static QPointer<neutrino> window_shortcuts;
    static neutrino* shortcutsWindow();

    nHelpTextBrowser *browser;
    QTreeWidget *tree;
    QLineEdit *search;
    QMap<QString, QTreeWidgetItem*> items; // by resource (":/...README.html")
    QMap<QString, QString> texts;          // the plain text of the pages, for the search (by resource)
};

#endif
