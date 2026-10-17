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
#include <QtGui>
#include <QWidget>
#include <QFileSystemModel>
#include <QTimer>
#include <QMap>

#include "nGenericPan.h"
#include "ui_Monitor.h"

#ifndef Monitor_H_
#define Monitor_H_

class neutrino;

class Monitor : public nGenericPan, private Ui::Monitor {
    Q_OBJECT

public:
    Q_INVOKABLE Monitor(neutrino *);
    QFileSystemModel *fileModel;
    QCompleter *completer;

public slots:
    void listViewClicked(QModelIndex);
    void listViewDoubleClicked(QModelIndex);
    void listViewActivated(QModelIndex);
    void textChanged();
    // void on_openAll_released();
    void on_openAll_clicked();

    void changeDir();

    void openFiles(QStringList);
    void sortFiles();
    void directoryLoaded(QString);
    void filesInserted(QModelIndex, int, int);
    void openPending();

private:
    void openIndex(QModelIndex);
    QList<nPhysD*> openedPhys;         // images opened by the monitor (the ones "erase previous" removes)
    bool dirLoaded=false;              // rows inserted before the directory is loaded are not new files
    QMap<QString,qint64> pendingFiles; // new files (auto-open) with their size, opened when the size is stable
    QTimer pendingTimer;

};

NEUTRINO_PLUGIN(Monitor,File);

#endif
