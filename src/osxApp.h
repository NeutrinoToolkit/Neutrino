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
#ifndef osxApp_H
#define osxApp_H

#include <QApplication>
#include <QtGui>
#include "neutrino.h"

class osxApp : public QApplication {
    Q_OBJECT
public:
	osxApp( int &argc, char **argv ) : QApplication(argc, argv) {}
protected:
	bool event(QEvent *ev) {
//		DEBUG(5,"MAC APPLICATION EVENT " << ev->type());
		if (ev->type() == QEvent::FileOpen) {
			QWidget *widget = QApplication::activeWindow();
			neutrino *neu=qobject_cast<neutrino *>(widget);
			if (neu == NULL) {
				nGenericPan *pan=qobject_cast<nGenericPan *>(widget);
				if (pan) neu = pan->nparent;
			}
			if (neu == NULL) neu = new neutrino(); 
			neu->fileOpen(static_cast<QFileOpenEvent *>(ev)->file());
		} else {
			return QApplication::event(ev);
		}
		return true;
	}
};
#endif

