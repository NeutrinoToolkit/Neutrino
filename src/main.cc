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
#include <string>
#include <iostream>
#include <cmath>
#include <signal.h>

#include <QtGui>
//#include <QtSql>

#include "nPreferences.h"

#include "neutrino.h"
#include "nApp.h"

#ifdef HAVE_PYTHONQT
#include "PythonQt_QtBindings.h"
#include "nPhysPyWrapper.h"
#include "nPython.h"
#endif

#include <QTranslator>

void my_handler(int s){
    printf("Caught signal %d\n",s);
    QCoreApplication::quit();
}

int main(int argc, char **argv)
{
#ifndef __WIN32
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
#endif

    qSetMessagePattern("%{function}:%{line} : %{message}");

    NApplication qapp(argc,argv);

    QSettings my_set("neutrino","");
    my_set.beginGroup("Preferences");
    nPreferences::changeLocale(my_set.value("locale",QLocale()).toLocale());
    my_set.endGroup();

    nPreferences::changeThreads(my_set.value("threads",1).toInt());

    QCoreApplication::addLibraryPath(QCoreApplication::applicationDirPath()+QString("/plugins"));

    bool somethingDone=false;

    QStringList args=QCoreApplication::arguments();
    args.removeFirst();

#ifdef HAVE_PYTHONQT

    PythonQt::init(PythonQt::IgnoreSiteModule|PythonQt::RedirectStdOut);

    //PythonQt_QtAll::init();
    PythonQt_init_QtBindings();

    PythonQt::self()->addDecorators(new nPhysPyWrapper());
    PythonQt::self()->registerCPPClass("nPhysD",NULL,"neutrino");

    PythonQt::self()->addDecorators(new nPanPyWrapper());
    PythonQt::self()->registerClass(& nGenericPan::staticMetaObject, "nPan", PythonQtCreateObject<nPanPyWrapper>);

    PythonQt::self()->registerClass(& nCustomPlot::staticMetaObject, "nPlot");

    PythonQt::self()->addDecorators(new nPyWrapper());
    PythonQt::self()->registerClass(& neutrino::staticMetaObject, "neutrino", PythonQtCreateObject<nPyWrapper>);

    QSettings settings("neutrino","");
    settings.beginGroup("nPython");
    foreach (QString spath, settings.value("siteFolder").toString().split(QRegExp("\\s*:\\s*"))) {
        qDebug() << "Python site folder " << spath;
        if (QFileInfo(spath).isDir()) PythonQt::self()->addSysPath(spath);
    }
    settings.endGroup();

    PythonQt::self()->getMainModule().addObject("nApp", &qapp);
    foreach (QString filename, args) {
        QFileInfo my_file(filename);
        if (my_file.exists() && my_file.suffix()=="py") {
            somethingDone=true;
            QFile t(filename);
            t.open(QIODevice::ReadOnly| QIODevice::Text);
            PythonQt::self()->getMainModule().evalScript(QTextStream(&t).readAll());
            t.close();
        }
    }
#endif


    if (!somethingDone) {
        neutrino *neu = new neutrino();
        neu->fileOpen(args);
    }

    return qapp.exec();
}
