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
    // completer->setModel(new QFileSystemModel());
    // (no parent: leaked, and no root path: the model is never populated)
    QFileSystemModel *dirModel = new QFileSystemModel(completer);
    dirModel->setFilter(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::Drives);
    dirModel->setRootPath("");
    completer->setModel(dirModel);
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
    // connect(listView, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(listViewDoubleClicked(QModelIndex)));
    // activated: double click or Enter (or single click, if that is the platform setting)
    connect(listView, SIGNAL(activated(QModelIndex)), this, SLOT(listViewDoubleClicked(QModelIndex)));
    connect(listView, SIGNAL(entered(QModelIndex)), this, SLOT(listViewActivated(QModelIndex)));
    // the file info also when moving with the keyboard
    connect(listView->selectionModel(), SIGNAL(currentChanged(QModelIndex,QModelIndex)), this, SLOT(listViewActivated(QModelIndex)));

    // auto-open: new files are opened when their size has not changed for a second (still being written otherwise)
    pendingTimer.setSingleShot(true);
    pendingTimer.setInterval(1000);
    connect(&pendingTimer, SIGNAL(timeout()), this, SLOT(openPending()));
    connect(fileModel, SIGNAL(directoryLoaded(QString)), this, SLOT(directoryLoaded(QString)));
    connect(fileModel, SIGNAL(rowsInserted(QModelIndex,int,int)), this, SLOT(filesInserted(QModelIndex,int,int)));
    connect(newestFirst, SIGNAL(toggled(bool)), this, SLOT(sortFiles()));

    textChanged();

    // connect(changeDirB, SIGNAL(released()), this, SLOT(changeDir()));
    connect(changeDirB, SIGNAL(clicked()), this, SLOT(changeDir()));
}


void
Monitor::listViewClicked(QModelIndex index) {
    if (oneClick->isChecked()) {
        // listViewDoubleClicked(index);
        openIndex(index);
    }
}

void
Monitor::listViewDoubleClicked(QModelIndex index) {
    // with 1-click open the click already opened it (a double click opened the file twice)
    if (!oneClick->isChecked()) {
        openIndex(index);
    }
}

void Monitor::openIndex(QModelIndex index) {
    QFileInfo fInfo=fileModel->fileInfo(index);
    if (fInfo.isFile() && fInfo.isReadable()) {
        // if (erase->isChecked()) {
        //     nparent->removePhys(currentBuffer);
        // }
        // nparent->fileOpen(fInfo.absoluteFilePath());
        openFiles(QStringList(fInfo.absoluteFilePath()));
    }
}

// open the files; "erase previous" removes the images opened before by the monitor, only if the new ones opened
// (it removed currentBuffer, which may be any image, and before knowing if the new file could be opened)
void Monitor::openFiles(QStringList files) {
    QList<nPhysD*> opened=nparent->fileOpen(files);
    if (opened.isEmpty()) {
        statusBar()->showMessage(tr("Cannot open ")+files.join(", "),5000);
        return;
    }
    if (erase->isChecked()) {
        for (nPhysD *phys : openedPhys) {
            if (nparent->nPhysExists(phys) && !opened.contains(phys)) {
                nparent->removePhys(phys);
            }
        }
    }
    openedPhys=opened;
}

void
Monitor::listViewActivated(QModelIndex index) {
    QFileInfo fInfo=fileModel->fileInfo(index);
    if (fInfo.isFile() && fInfo.isReadable()){
		// QStringList list;
		// list << tr("Kb") << tr("Mb") << tr("Gb") << tr("Tb");
		//
		// QStringListIterator i(list);
		// QString unit(tr("bytes"));
		// double num=fInfo.size();
		// while(num >= 1024.0 && i.hasNext()) {
		// 	unit = i.next();
		// 	num /= 1024.0;
		// }
		// QString sizeF=" ("+QLocale().toString(num,'f',1)+unit+")";
		// (Kb, Mb... are bits)
		QString sizeF=" ("+QLocale().formattedDataSize(fInfo.size())+")";
        statusBar()->showMessage(fInfo.lastModified().toString() +sizeF,5000);
	}
}

void
Monitor::textChanged() {
    // fileModel->setNameFilters(pattern->text().split(" "));
    // (an empty pattern gave the filter "" which matches no file): no filter shows all the files
    fileModel->setNameFilters(pattern->text().split(QRegularExpression("\\s+"), Qt::SkipEmptyParts));
    dirLoaded=false;
    pendingFiles.clear();
    listView->setRootIndex(fileModel->setRootPath(dirName->text()));
    sortFiles();
}

void Monitor::sortFiles() {
    if (newestFirst->isChecked()) {
        fileModel->sort(3, Qt::DescendingOrder); // column 3: date modified
    } else {
        fileModel->sort(0, Qt::AscendingOrder);
    }
}

void Monitor::directoryLoaded(QString path) {
    if (QDir(path)==QDir(fileModel->rootPath())) {
        dirLoaded=true;
        sortFiles();
    }
}

void Monitor::filesInserted(QModelIndex parent, int first, int last) {
    // only files appearing after the directory was loaded, in the monitored directory
    if (!autoOpen->isChecked() || !dirLoaded || parent!=listView->rootIndex()) return;
    for (int row=first; row<=last; row++) {
        QFileInfo fInfo=fileModel->fileInfo(fileModel->index(row, 0, parent));
        if (fInfo.isFile()) {
            pendingFiles[fInfo.absoluteFilePath()]=-1;
        }
    }
    if (!pendingFiles.isEmpty()) pendingTimer.start();
}

void Monitor::openPending() {
    QStringList ready;
    for (auto it=pendingFiles.begin(); it!=pendingFiles.end(); ) {
        QFileInfo fInfo(it.key());
        if (!fInfo.exists()) {
            it=pendingFiles.erase(it);
        } else if (fInfo.size()>0 && fInfo.size()==it.value() && fInfo.isReadable()) {
            ready << it.key();
            it=pendingFiles.erase(it);
        } else {
            it.value()=fInfo.size(); // still growing (or empty): check again
            ++it;
        }
    }
    if (!pendingFiles.isEmpty()) pendingTimer.start();
    if (!ready.isEmpty() && autoOpen->isChecked()) {
        ready.sort();
        openFiles(ready);
    }
}

// void Monitor::on_openAll_released() {
void Monitor::on_openAll_clicked() {
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
    // nparent->fileOpen(filestoopen);
    // (through openFiles: "erase previous" replaces the previous batch)
    if (!filestoopen.isEmpty()) openFiles(filestoopen);
}

void Monitor::changeDir() {
    QString newDir;
    newDir = QFileDialog::getExistingDirectory(this,tr("Change monitor directory"),dirName->text());
    if (!newDir.isEmpty()) dirName->setText(newDir);
}

