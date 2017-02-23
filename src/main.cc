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
    my_set.beginGroup("nPreferences");
    nPreferences::changeLocale(my_set.value("locale",QLocale()).toLocale());

    qDebug() << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<< my_set.value("threads",1);
    nPreferences::changeThreads(my_set.value("threads",1).toInt());
    my_set.endGroup();

    QCoreApplication::addLibraryPath(QCoreApplication::applicationDirPath()+QString("/plugins"));

    QStringList args=QCoreApplication::arguments();
    args.removeFirst();

    neutrino *neu = new neutrino();
    neu->fileOpen(args);

    return qapp.exec();
}
