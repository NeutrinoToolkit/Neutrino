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
#include <cmath>
#include <signal.h>

#include <QtGui>
#include <QApplication>
#include <QMessageBox>

#include "neutrino.h"
// #include "nApp.h"
#include "nCore.h"
#include "nVersion.h"

// only a safety net for the executable: an exception that reaches the event loop goes to the log window
// instead of aborting. Neutrino catches its exceptions itself (nCore::showException), as there is no notify
// when it is a library (e.g. in a PySide application)
class nNotifyApp : public QApplication {
public:
    using QApplication::QApplication;
    bool notify(QObject *rec, QEvent *ev) override {
        try {
            return QApplication::notify(rec, ev);
        } catch (std::exception &e) {
            nCore::self()->showException(e.what());
        }
        return false;
    }
};

int main(int argc, char **argv) {
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);

    qSetMessagePattern("%{file}(%{line}): %{message}");

    // nApp my_app(argc,argv);
    //
    // return my_app.exec();

    // was nApp::exec
    nNotifyApp my_app(argc,argv);

    QCoreApplication::setOrganizationName("polytechnique");
    QCoreApplication::setOrganizationDomain("edu");
    QCoreApplication::setApplicationName("Neutrino");
    QCoreApplication::setApplicationVersion(SHAVERSION__);

    // log window, message handler, icon theme, palettes, threads and decimal dot
    nCore *core=nCore::self();
    // files opened from the Finder or the Dock (macOS)
    core->handleFileOpen(true);

    // connect(this, SIGNAL(lastWindowClosed()), this, SLOT(quit()));
    // (QApplication quits when the last window is closed: quitOnLastWindowClosed)

    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    if (!my_set.contains("checkUpdates")) {
        my_set.setValue("checkUpdates",QMessageBox::Yes == QMessageBox::question(nullptr,QObject::tr("Attention"),QObject::tr("Do you want to check for updates?"), QMessageBox::Yes | QMessageBox::No));
    }
    if (my_set.value("checkUpdates",true).toBool()) {
        core->checkUpdates();
    }
    my_set.endGroup();

    QStringList args=my_app.arguments();
    args.removeFirst();

    neutrino *my_neu = new neutrino();
    for (auto &arg : args)
        my_neu->fileOpen(arg);

    my_neu->activateWindow();
    return my_app.exec();
}
