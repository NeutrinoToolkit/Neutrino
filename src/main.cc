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

#include <QtGui>
//#include <QtSql>

#ifdef Q_OS_MAC
#include "osxApp.h"
#endif 

#include "neutrino.h"
#include "nApp.h"

#ifdef __neutrino_key
#include "nHash.h"
#endif

int main(int argc, char **argv)
{

#ifdef Q_OS_MAC
	osxApp *qapp = new osxApp(argc,argv);	
#else
	NApplication *qapp = new NApplication(argc,argv);
#endif
	
	qapp->setOrganizationName("ParisTech");
	qapp->setOrganizationDomain("edu");
	qapp->setApplicationName("Neutrino");
	qapp->setApplicationVersion(__VER);

#ifdef __neutrino_key
	std::string hh = getNHash();
	std::cerr<<"got nHash: "<<hh<<std::endl;
	qapp->setProperty("nHash", hh.c_str());
#endif
	
	QCoreApplication::addLibraryPath(QCoreApplication::applicationDirPath()+QString("/plugins"));

	neutrino* neu = new neutrino();
	QStringList args=QCoreApplication::arguments();
    args.removeFirst();
	foreach (QString filename, args) {
		neu->fileOpen(filename);
	}

	qapp->exec();
}
