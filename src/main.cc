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
int main(int argc, char **argv)
{

#ifdef Q_OS_MAC
	DEBUG("Initializing osx app");
	osxApp *qapp = new osxApp(argc,argv);	
#else
	QApplication *qapp = new QApplication(argc,argv);
#endif
	
	qapp->setOrganizationName("ParisTech");
	qapp->setOrganizationDomain("edu");
	qapp->setApplicationName("Neutrino");
	qapp->setApplicationVersion(__VER);
	
	QCoreApplication::addLibraryPath(QCoreApplication::applicationDirPath()+QString("/plugins"));

	new neutrino();

	qapp->exec();
}
