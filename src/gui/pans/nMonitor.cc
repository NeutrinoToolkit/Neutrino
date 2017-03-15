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
#include "nMonitor.h"
#include "neutrino.h"

nMonitor::nMonitor(neutrino *nparent) : nGenericPan(nparent)
{
	my_w.setupUi(this);
    show();

	completer = new QCompleter(my_w.lineEdit);
	completer->setModel(new QDirModel());
	my_w.lineEdit->setCompleter(completer);
	
	fileModel=new QFileSystemModel(this);
	fileModel->setFilter(QDir::Files);
	my_w.listView->setModel(fileModel);

	connect(my_w.lineEdit, SIGNAL(textChanged(QString)), this, SLOT(textChanged(QString)));	
	connect(my_w.listView, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(listViewDoubleClicked(QModelIndex)));
	connect(my_w.listView, SIGNAL(entered(QModelIndex)), this, SLOT(listViewActivated(QModelIndex)));
	
	textChanged(my_w.lineEdit->text());

	connect(my_w.changeDir, SIGNAL(released()), this, SLOT(changeDir()));
	connect(fileModel, SIGNAL(rowsInserted(QModelIndex, int, int)), this, SLOT(rowsInserted(QModelIndex, int, int)));
	connect(fileModel, SIGNAL(rowsRemoved(QModelIndex, int, int)), this, SLOT(rowsRemoved(QModelIndex, int, int)));
}

void
nMonitor::rowsInserted(QModelIndex index, int inizio, int fine) {
	qDebug() << ">>>>>>>>>>>>>>>> " << __PRETTY_FUNCTION__ << fileModel->fileInfo(index).absoluteFilePath() << inizio << fine;
}

void
nMonitor::rowsRemoved(QModelIndex index, int inizio, int fine) {
	qDebug() << ">>>>>>>>>>>>>>>> " << __PRETTY_FUNCTION__ << fileModel->fileInfo(index).absoluteFilePath() << inizio << fine;
}

void
nMonitor::listViewDoubleClicked(QModelIndex index) {
	if (fileModel->fileInfo(index).isFile() && fileModel->fileInfo(index).isReadable())
		nparent->fileOpen(fileModel->fileInfo(index).absoluteFilePath());
}

void
nMonitor::listViewActivated(QModelIndex index) {
	if (fileModel->fileInfo(index).isFile() && fileModel->fileInfo(index).isReadable()){
		QFileInfo fInfo=fileModel->fileInfo(index);

		QStringList list;
		list << tr("Kb") << tr("Mb") << tr("Gb") << tr("Tb");
		
		QStringListIterator i(list);
		QString unit(tr("bytes"));
		double num=fInfo.size();
		while(num >= 1024.0 && i.hasNext()) {
			unit = i.next();
			num /= 1024.0;
		}
		QString sizeF=" ("+QString::number(num,'f',1)+unit+")";
		my_w.statusBar->showMessage(fInfo.lastModified().toString(Qt::DefaultLocaleShortDate) +sizeF,5000);
	}
}

void
nMonitor::textChanged(QString dirName) {
	my_w.listView->setRootIndex(fileModel->setRootPath(dirName));
	// FIXME: this an absolute bug in qt check here:
	// http://qt-project.org/forums/viewthread/7265
	fileModel->setNameFilters(QStringList());
}

void
nMonitor::changeDir() {
	QString dirName;
	dirName = QFileDialog::getExistingDirectory(this,tr("Change monitor directory"),my_w.lineEdit->text());
	if (!dirName.isEmpty()) my_w.lineEdit->setText(dirName);
}

