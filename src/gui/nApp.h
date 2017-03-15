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
#ifndef nApp_H
#define nApp_H

#include <QApplication>
#include <QSettings>
#include <QDebug>

class neutrino;

class nApp : public QApplication {
    Q_OBJECT
public:
    nApp( int &argc, char **argv );


protected:
    virtual bool notify(QObject *rec, QEvent *ev) override;

    bool event(QEvent *ev);


public slots:
//    QList<neutrino*> neus();

    static void changeThreads(int);
    static void changeLocale(QLocale locale);
    static bool localeLessThan(const QLocale&, const QLocale&);
    static QString localeToString(const QLocale &);
};
#endif

