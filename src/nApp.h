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
#include <QPlainTextEdit>
#include <QSettings>
#include <QDebug>
#ifndef __clang__
#include "grabStream.h"
#endif

class neutrino;
class nGenericPan;
namespace Ui {
class nLogWin;
}

class nApp : public QApplication {
    Q_OBJECT
public:
    nApp( int &argc, char **argv );

    QMap<QString, std::vector<unsigned char>> nPalettes;

    QWidget log_win;
    Ui::nLogWin *log_win_ui;

protected:
    virtual bool notify(QObject *rec, QEvent *ev) override;

    bool event(QEvent *ev) override;

    static void myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg);

private:
#ifndef __clang__
#ifdef __phys_debug
    grabStream qerr;
#endif
    grabStream qout;
#endif

public slots:


    void closeAllWindows();

    QList<neutrino*> neus();

    static void changeThreads(int);
    static void changeLocale(QLocale locale);
    static bool localeLessThan(const QLocale&, const QLocale&);
    static QString localeToString(const QLocale &);

    void checkUpdates();

    void addPaletteFile(QString);
    void addDefaultPalettes();

    void copyLog();
    void saveLog();

signals:
    void logWinVisibility(bool);
};

#endif

