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
#include "Monitor.h"
#include "neutrino.h"

Monitor::Monitor(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);
    show();

    completer = new QCompleter(dirName);
    completer->setModel(new QFileSystemModel());
    dirName->setCompleter(completer);
	
	fileModel=new QFileSystemModel(this);
	fileModel->setFilter(QDir::Files);
    fileModel->setNameFilterDisables(false);
//    proxyModel = new QSortFilterProxyModel(this);
//    proxyModel->setSourceModel(fileModel);

    listView->setModel(fileModel);
    connect(dirName, SIGNAL(textChanged(QString)), this, SLOT(textChanged()));
    connect(pattern, SIGNAL(textChanged(QString)), this, SLOT(textChanged()));

    connect(listView, SIGNAL(clicked(QModelIndex)), this, SLOT(listViewClicked(QModelIndex)));
    connect(listView, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(listViewDoubleClicked(QModelIndex)));
    connect(listView, SIGNAL(entered(QModelIndex)), this, SLOT(listViewActivated(QModelIndex)));
	
    textChanged();

    connect(changeDirB, SIGNAL(released()), this, SLOT(changeDir()));
}


void
Monitor::listViewClicked(QModelIndex index) {
    if (oneClick->isChecked()) {
        listViewDoubleClicked(index);
    }
}

void
Monitor::listViewDoubleClicked(QModelIndex index) {
    QFileInfo fInfo=fileModel->fileInfo(index);
    if (fInfo.isFile() && fInfo.isReadable()) {
        if (erase->isChecked()) {
            nparent->removePhys(currentBuffer);
        }
        nparent->fileOpen(fInfo.absoluteFilePath());
    }
}

void
Monitor::listViewActivated(QModelIndex index) {
    QFileInfo fInfo=fileModel->fileInfo(index);
    if (fInfo.isFile() && fInfo.isReadable()){
		QStringList list;
		list << tr("Kb") << tr("Mb") << tr("Gb") << tr("Tb");
		
		QStringListIterator i(list);
		QString unit(tr("bytes"));
		double num=fInfo.size();
		while(num >= 1024.0 && i.hasNext()) {
			unit = i.next();
			num /= 1024.0;
		}
		QString sizeF=" ("+QLocale().toString(num,'f',1)+unit+")";
        statusBar()->showMessage(fInfo.lastModified().toString() +sizeF,5000);
	}
}

void
Monitor::textChanged() {
    fileModel->setNameFilters(pattern->text().split(" "));
    listView->setRootIndex(fileModel->setRootPath(dirName->text()));
}

void Monitor::on_openAll_released() {
    QList<QFileInfo> path_list;
    QModelIndex parentIndex = fileModel->index(fileModel->rootPath());
    int numRows = fileModel->rowCount(parentIndex);
    QStringList filestoopen;

    for (int row = 0; row < numRows; ++row) {
        QModelIndex childIndex = fileModel->index(row, 0, parentIndex);
        QFileInfo path = QFileInfo(fileModel->rootPath(), fileModel->data(childIndex).toString());
        if (path.isFile())  {
            filestoopen.append(path.absoluteFilePath());
        }
    }
    nparent->fileOpen(filestoopen);
}

void Monitor::changeDir() {
    QString newDir;
    newDir = QFileDialog::getExistingDirectory(this,tr("Change monitor directory"),dirName->text());
    if (!newDir.isEmpty()) dirName->setText(newDir);
}

