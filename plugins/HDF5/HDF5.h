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

#include "nGenericPan.h"
#include "ui_HDF5.h"

#ifndef __HDF5
#define __HDF5

#include "hdf5.h"
#include "hdf5_hl.h"

class neutrino;

class HDF5 : public nGenericPan {
    Q_OBJECT

public:
    Q_INVOKABLE HDF5(neutrino *);
    Ui::HDF5 my_w;

public slots:
    QString getFilename(QTreeWidgetItem*);

    void itemEntered(QTreeWidgetItem*, int=-1);
    void openData(QTreeWidgetItem*, int=-1);
    void removeFile();
    void showFile();
    void showFile(QString);
    void scanGroup(hid_t, QTreeWidgetItem *);
    void scanAttribute(hid_t, QTreeWidgetItem *, nPhysD* = NULL);
    void scanDataset(hid_t, QTreeWidgetItem *);

    int phys_write_HDF5(nPhysD *phys, std::string fname);
    nPhysD* phys_open_HDF5(std::string fileName, std::string dataName);
    void scan_hdf5_attributes(hid_t aid, nPhysD *my_data);

    void copyPath();
};

NEUTRINO_PLUGIN(HDF5,Analysis)

#endif
