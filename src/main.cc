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

#include "neutrino.h"
#include "nApp.h"

#include <QTranslator>

void my_handler(int s){
    qCritical() << "Caught signal" << s;
    QCoreApplication::quit();
}

int main(int argc, char **argv)
{
#if defined(Q_OS_MAC)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

#if QT_VERSION >= QT_VERSION_CHECK(5, 4, 0)
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
#endif

#ifdef __WIN32
#ifdef __phys_debug
    AllocConsole();
    freopen("conin$", "r", stdin);
    freopen("conout$", "w", stdout);
    freopen("conout$", "w", stderr);
    qSetMessagePattern("%{message}");
#else
    qSetMessagePattern("%{file}(%{line}): %{message}");
#endif
#else
    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, nullptr);
#endif

    nApp my_app(argc,argv);

    return my_app.exec();
}
