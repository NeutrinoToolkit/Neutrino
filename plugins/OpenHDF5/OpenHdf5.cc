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
#include "OpenHdf5.h"
#include <QtGui>
#include <QMenu>
#include <QMessageBox>
#include <QInputDialog>

// item columns: 0 name, 1 type, 2 data, 3 (hidden) path in the file (file name for the top-level items)
// Qt::UserRole of column 0: nHdf5::Kind of a dataset

OpenHdf5::OpenHdf5(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);
    treeWidget->setColumnHidden((treeWidget->columnCount()-1),true);

    connect(actionOpen, SIGNAL(triggered()), this, SLOT(showFile()));
    connect(actionClose, SIGNAL(triggered()), this, SLOT(removeFile()));
    connect(actionCopy, SIGNAL(triggered()), this, SLOT(copyPath()));
    connect(actionSave, SIGNAL(triggered()), this, SLOT(saveFile()));
    connect(treeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(openData(QTreeWidgetItem*, int)));
    connect(treeWidget, SIGNAL(itemPressed(QTreeWidgetItem*, int)), this, SLOT(itemEntered(QTreeWidgetItem*, int)));
    treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(treeWidget, SIGNAL(customContextMenuRequested(QPoint)), this, SLOT(contextMenu(QPoint)));

    setProperty("NeuSave-fileHDF","nHDF.h5");

    show();

}

void OpenHdf5::copyPath(){
    // the old text, fileOpen("file","dataset"), is not a command of neutrino nor of PyShell:
    // file::dataset (the usual notation, e.g. silx)
    QStringList clipText;
    foreach(QTreeWidgetItem *item, treeWidget->selectedItems()) {
        if (item->parent()) {
            clipText << getFilename(item)+"::"+item->data(3,0).toString();
        } else {
            clipText << getFilename(item);
        }
    }
    QApplication::clipboard()->setText(clipText.join("\n"));
}

void OpenHdf5::removeFile(){
    // the top-level items once each (two items of the same file deleted it twice)
    QSet<QTreeWidgetItem *> files;
    foreach(QTreeWidgetItem *item, treeWidget->selectedItems()) {
        while (item->parent()) {
            item=item->parent();
        }
        files.insert(item);
    }
    qDeleteAll(files);
}

void OpenHdf5::itemEntered(QTreeWidgetItem *item, int) {
    statusbar->showMessage(item->data(3,0).toString(),5000);
}

QString OpenHdf5::getFilename(QTreeWidgetItem *item) {
    while (item->parent()) {
        item=item->parent();
    }
    return item->data(3,0).toString();
}

void OpenHdf5::addAttributes(QTreeWidgetItem *parentItem, const std::vector<nHdf5::Attribute> &attributes) {
    for (auto &attr : attributes) {
        QTreeWidgetItem *item=new QTreeWidgetItem(parentItem,QStringList(QString::fromStdString(attr.name)));
        item->setFlags(item->flags()|Qt::ItemIsEditable);
        item->setData(1,0,"Attr "+QString::fromStdString(attr.type));
        item->setData(2,0,QString::fromStdString(attr.text()));
    }
}

void OpenHdf5::addNode(QTreeWidgetItem *item, const nHdf5::Node &node) {
    addAttributes(item, node.attributes);
    for (auto &child : node.children) {
        QTreeWidgetItem *item2=new QTreeWidgetItem(item,QStringList(QString::fromStdString(child.name)));
        QString type=QString::fromStdString(child.typeText);
        if (child.type==nHdf5::Node::dataset) type="DS "+type;
        item2->setData(1,0,type);
        item2->setData(2,0,QString::fromStdString(child.dataText));
        if (child.type==nHdf5::Node::dataset) {
            item2->setData(3,0,QString::fromStdString(child.path));
            item2->setData(0,Qt::UserRole,(int)child.kind);
            if (child.kind==nHdf5::Kind::frames) item2->setData(0,Qt::UserRole+1,(qulonglong)child.dims[0]);
            if (child.kind==nHdf5::Kind::image || child.kind==nHdf5::Kind::frames) {
                // images in blue, with their groups expanded
                QTreeWidgetItem *dataParent=item2->parent();
                while (dataParent) {
                    dataParent->setExpanded(true);
                    dataParent->setForeground(0,QBrush(QColor(0,0,200)));
                    dataParent=dataParent->parent();
                }
                for (int col=0; col<3; col++) item2->setForeground(col,QBrush(QColor(0,0,200)));
            } else if (child.kind==nHdf5::Kind::none) {
                item2->setBackground(2,QBrush(QColor(200,200,255)));
            }
        } else if (child.type==nHdf5::Node::other) {
            // compound members
            item2->setFlags(item2->flags()|Qt::ItemIsEditable);
        }
        addNode(item2, child);
    }
}

void OpenHdf5::showFile(QString fname) {
    statusbar->showMessage("Parsing "+fname);
    QApplication::processEvents();
    // for (int i=0;i<treeWidget->topLevelItemCount();i++) {
    // (deleting while going forward skipped the next item)
    for (int i=treeWidget->topLevelItemCount()-1;i>=0;i--) {
        QTreeWidgetItem *item=treeWidget->topLevelItem(i);
        if (item->data(3,0).toString()==fname) {
            delete item;
        }
    }
    nHdf5::Node root;
    std::string error;
    if (nHdf5::scanFile(QFile::encodeName(fname).toStdString(), root, error)) {
        nparent->updateRecentFileActions(fname);
        QTreeWidgetItem *item=new QTreeWidgetItem(treeWidget,QStringList(QFileInfo(fname).fileName()));
        item->setData(1,0,"File");
        item->setData(3,0,fname);
        addNode(item, root);
        treeWidget->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
        statusbar->showMessage("Done",2000);
    } else {
        statusbar->showMessage(QString::fromStdString(error),5000);
    }
}

void OpenHdf5::showFile() {
    QStringList fnames=QFileDialog::getOpenFileNames(this,tr("Open HDF file source"),property("NeuSave-fileHDF").toString(),tr("HDF5")+QString(" (*.h5 *.hdf5 *.hdf *.nxs);;")+tr("Any file")+QString(" (*)"));
    foreach (QString fname, fnames) {
        showFile(fname);
    }
    // if (!fnames.isEmpty()) setProperty("NeuSave-fileHDF", fnames);
    // (a QStringList of several files is not a QString: the dialog lost the directory)
    if (!fnames.isEmpty()) setProperty("NeuSave-fileHDF", fnames.last());
}

void OpenHdf5::saveFile() {
    if (!currentBuffer) {
        statusbar->showMessage(tr("No image to save"),5000);
        return;
    }
    QString fname=QFileDialog::getSaveFileName(this,tr("Save image in HDF5"),property("NeuSave-fileHDF").toString(),tr("HDF5")+QString(" (*.h5 *.hdf5);;")+tr("Any file")+QString(" (*)"));
    if (fname.isEmpty()) return;
    setProperty("NeuSave-fileHDF", fname);
    std::string error;
    if (nHdf5::writeImage(*currentBuffer, QFile::encodeName(fname).toStdString(), qApp->applicationVersion().toStdString(), error)) {
        showFile(fname);
        statusbar->showMessage(tr("Saved ")+fname,5000);
    } else {
        QMessageBox::warning(this,tr("Save HDF5"),QString::fromStdString(error),QMessageBox::Ok);
    }
}

// double click: images are opened, 1D data copied, for frames the choice
void OpenHdf5::openData(QTreeWidgetItem *item, int) {
    // QString dataName=item->data(3,0).toString();
    // nparent->showPhys(phys_open_Hdf5(getFilename(item),dataName.toStdString()));
    // (showPhys shows only images already in neutrino: nothing was ever shown)
    switch ((nHdf5::Kind)item->data(0,Qt::UserRole).toInt()) {
    case nHdf5::Kind::image:
        openImage(item);
        break;
    case nHdf5::Kind::data1D:
        copyValues(item);
        break;
    case nHdf5::Kind::frames: {
        QMenu *menu=dataMenu(item);
        menu->exec(QCursor::pos());
        delete menu;
        break;
    }
    default:
        if (!item->data(3,0).toString().isEmpty() && item->parent()) {
            statusbar->showMessage(tr("Cannot open ")+item->data(1,0).toString()+" "+item->data(2,0).toString(),5000);
        }
        break;
    }
}

QMenu *OpenHdf5::dataMenu(QTreeWidgetItem *item) {
    QMenu *menu=new QMenu(this);
    switch ((nHdf5::Kind)item->data(0,Qt::UserRole).toInt()) {
    case nHdf5::Kind::image:
        menu->addAction(tr("Open"), this, [=](){ openImage(item); });
        break;
    case nHdf5::Kind::frames:
        menu->addAction(tr("Open mean of the frames"), this, [=](){ openFramesMean(item); });
        menu->addAction(tr("Open frame..."), this, [=](){ openFrame(item); });
        menu->addAction(tr("Open all frames"), this, [=](){ openAllFrames(item); });
        break;
    case nHdf5::Kind::data1D:
        menu->addAction(tr("Copy values"), this, [=](){ copyValues(item); });
        break;
    default:
        break;
    }
    menu->addAction(tr("Copy path"), this, SLOT(copyPath()));
    return menu;
}

void OpenHdf5::contextMenu(QPoint pos) {
    QTreeWidgetItem *item=treeWidget->itemAt(pos);
    if (!item) return;
    treeWidget->setCurrentItem(item);
    QMenu *menu=dataMenu(item);
    menu->exec(treeWidget->viewport()->mapToGlobal(pos));
    delete menu;
}

void OpenHdf5::openImage(QTreeWidgetItem *item) {
    physD image;
    std::string error;
    if (nHdf5::readImage(QFile::encodeName(getFilename(item)).toStdString(), item->data(3,0).toString().toStdString(), image, error)) {
        image.TscanBrightness();
        nparent->addShowPhys(new nPhysD(std::move(image)));
    } else {
        statusbar->showMessage(QString::fromStdString(error),5000);
    }
}

void OpenHdf5::openFrame(QTreeWidgetItem *item) {
    int nframes=item->data(0,Qt::UserRole+1).toInt();
    bool ok;
    int frame=QInputDialog::getInt(this,tr("Open frame"),tr("Frame (0 - %1)").arg(nframes-1),0,0,std::max(0,nframes-1),1,&ok);
    if (!ok) return;
    physD image;
    std::string error;
    if (nHdf5::readFrame(QFile::encodeName(getFilename(item)).toStdString(), item->data(3,0).toString().toStdString(), frame, image, error)) {
        image.TscanBrightness();
        nparent->addShowPhys(new nPhysD(std::move(image)));
    } else {
        statusbar->showMessage(QString::fromStdString(error),5000);
    }
}

void OpenHdf5::openAllFrames(QTreeWidgetItem *item) {
    int nframes=item->data(0,Qt::UserRole+1).toInt();
    if (nframes>20 && QMessageBox::question(this,tr("Open all frames"),tr("Open %1 images?").arg(nframes),QMessageBox::Yes|QMessageBox::No)!=QMessageBox::Yes) {
        return;
    }
    std::vector<physD> images;
    std::string error;
    if (nHdf5::readFrames(QFile::encodeName(getFilename(item)).toStdString(), item->data(3,0).toString().toStdString(), images, error)) {
        for (auto &image : images) {
            image.TscanBrightness();
            nparent->addShowPhys(new nPhysD(std::move(image)));
        }
    } else {
        statusbar->showMessage(QString::fromStdString(error),5000);
    }
}

void OpenHdf5::openFramesMean(QTreeWidgetItem *item) {
    physD image;
    std::string error;
    if (nHdf5::readFramesMean(QFile::encodeName(getFilename(item)).toStdString(), item->data(3,0).toString().toStdString(), image, error)) {
        image.TscanBrightness();
        nparent->addShowPhys(new nPhysD(std::move(image)));
    } else {
        statusbar->showMessage(QString::fromStdString(error),5000);
    }
}

void OpenHdf5::copyValues(QTreeWidgetItem *item) {
    std::vector<double> data;
    std::string error;
    if (nHdf5::readData1D(QFile::encodeName(getFilename(item)).toStdString(), item->data(3,0).toString().toStdString(), data, error)) {
        // one value per line, C locale (the system locale gave decimal commas)
        QString str_data;
        QLocale cloc(QLocale::C);
        for (auto &val : data) {
            str_data += cloc.toString(val,'g',10) + "\n";
        }
        QApplication::clipboard()->setText(str_data);
        statusbar->showMessage(QLocale().toString((int)data.size()) + tr(" values copied into the clipboard"),2000);
    } else {
        statusbar->showMessage(QString::fromStdString(error),5000);
    }
}

// old implementation, before nHdf5io (not compiled):
// - showPhys of an image never added to neutrino: nothing was shown
// - integer datasets not read, arrays crashed (shadowed dims), file byte order, names with a trailing '\0'
//   (physOrigin, physScale... never applied), int8/int16 attributes overflowing, compound data read at scan,
//   files left open when the dataset could not be opened, HDF5 ids not closed
#if 0


#define HDF5_MAX_NAME 2048

OpenHdf5::OpenHdf5(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);
    treeWidget->setColumnHidden((treeWidget->columnCount()-1),true);

    connect(actionOpen, SIGNAL(triggered()), this, SLOT(showFile()));
    connect(actionClose, SIGNAL(triggered()), this, SLOT(removeFile()));
    connect(actionCopy, SIGNAL(triggered()), this, SLOT(copyPath()));
    connect(treeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(openData(QTreeWidgetItem*, int)));
    connect(treeWidget, SIGNAL(itemPressed(QTreeWidgetItem*, int)), this, SLOT(itemEntered(QTreeWidgetItem*, int)));

    setProperty("NeuSave-fileHDF","nHDF.h5");

    show();

}

void OpenHdf5::copyPath(){
    QString clipText;
    foreach(QTreeWidgetItem *item, treeWidget->selectedItems()) {
        clipText+="fileOpen(\""+getFilename(item)+"\",\""+item->data(3,0).toString()+"\") ";
    }
    QApplication::clipboard()->setText(clipText);
}

void OpenHdf5::removeFile(){
    foreach(QTreeWidgetItem *item, treeWidget->selectedItems()) {
        while (item->parent()) {
            item=item->parent();
        }
        delete item;
    }
}

void OpenHdf5::itemEntered(QTreeWidgetItem *item, int) {
    statusbar->showMessage(item->data(3,0).toString(),5000);
}

QString OpenHdf5::getFilename(QTreeWidgetItem *item) {
    while (item->parent()) {
        item=item->parent();
    }
    return item->data(3,0).toString();
}

void OpenHdf5::openData(QTreeWidgetItem *item, int) {
    QString dataName=item->data(3,0).toString();
    nparent->showPhys(phys_open_Hdf5(getFilename(item),dataName.toStdString()));
}

void OpenHdf5::showFile(QString fname) {
    statusbar->showMessage("Parsing "+fname);
    QApplication::processEvents();
    for (int i=0;i<treeWidget->topLevelItemCount();i++) {
        QTreeWidgetItem *item=treeWidget->topLevelItem(i);
        if (item->data(3,0).toString()==fname) {
            delete item;
        }
    }
    hid_t faplist_id=H5Pcreate(H5P_FILE_ACCESS);
    hid_t file_id = H5Fopen (QFile::encodeName(fname).constData(), H5F_ACC_RDONLY, faplist_id);
    if (file_id>0) {
        nparent->updateRecentFileActions(fname);
        hid_t grp = H5Gopen(file_id,"/", H5P_DEFAULT);
        QTreeWidgetItem *item=new QTreeWidgetItem(treeWidget,QStringList(QFileInfo(fname).fileName()));
        item->setData(1,0,"File");
        item->setData(3,0,fname);
        scanGroup(grp,item);
        treeWidget->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
        H5Gclose(grp);
        H5Fclose(file_id);
        statusbar->showMessage("Done",2000);
    } else {
        statusbar->showMessage("Problem opening file",5000);
    }
}

void OpenHdf5::showFile() {
    QStringList fnames=QFileDialog::getOpenFileNames(this,tr("Open HDF file source"),property("NeuSave-fileHDF").toString(),tr("HDF5")+QString(" (*.h5);;")+tr("Any file")+QString(" (*)"));
    foreach (QString fname, fnames) {
        showFile(fname);
    }
    if (!fnames.isEmpty()) setProperty("NeuSave-fileHDF", fnames);
}

void OpenHdf5::scanDataset(hid_t did, QTreeWidgetItem *item2) {
    for (int i = 0; i < H5Aget_num_attrs(did); i++) {
        hid_t aid =	H5Aopen_idx(did, (unsigned int)i );
        scanAttribute(aid, item2);
        H5Aclose(aid);
    }
    ssize_t sizeName=1+H5Iget_name(did, nullptr,0);
    std::vector<char> ds_name(sizeName);
    H5Iget_name(did, &ds_name[0],sizeName);
    hid_t sid = H5Dget_space(did);
    hid_t tid = H5Dget_type(did);

    H5T_class_t t_class = H5Tget_class(tid);

    int ndims=0;
    std::vector<hsize_t> dims;

    ndims=H5Sget_simple_extent_ndims(sid);
    DEBUG("t_class : "<< t_class << " ndims " << ndims << " H5T_FLOAT " << H5T_FLOAT);
    if (t_class == H5T_FLOAT) {
        item2->setData(1,0,"DS Float");
        dims.resize(ndims);
        H5Sget_simple_extent_dims(sid,&dims[0],nullptr);
        DEBUG("here");
    } if (t_class == H5T_INTEGER) {
        QString title="DS Integer "+QLocale().toString(ndims)+"D ";
        dims.resize(ndims);
        H5Sget_simple_extent_dims(sid,&dims[0],nullptr);
        for (int d=ndims-1;d>=0;d--) title += QLocale().toString(dims[d])+"x";
        title.chop(1);
        item2->setData(1,0,title);
    } else if(t_class == H5T_ARRAY) {
        //! TODO: check this
        QString title="DS Array "+QLocale().toString(ndims)+"D ";
        dims.resize(ndims);
        H5Sget_simple_extent_dims(sid,&dims[0],nullptr);
        for (int d=ndims-1;d>=0;d--) title += QLocale().toString(dims[d])+"x";
        title.chop(1);
        item2->setData(1,0,title);

        item2->setData(1,0,title);
        ndims = H5Tget_array_ndims(tid);
        dims.resize(ndims);
        H5Tget_array_dims2(tid, &dims[0]);
    } else if(t_class == H5T_COMPOUND) {

        int size=H5Tget_size(tid);

        ndims=H5Sget_simple_extent_ndims(sid);
        dims.resize(ndims);
        H5Sget_simple_extent_dims(sid,&dims[0],nullptr);
        for (auto &val : dims) {
            size*=val;
        }


        item2->setData(1,0,"DS Compound");
        int nCompund=H5Tget_nmembers(tid);
        item2->setData(2,0,QLocale().toString(nCompund)+" objs");
        std::vector<char> buffer(size*nCompund);
        hid_t dataread=H5Dread(did, tid, sid, H5S_ALL, H5P_DEFAULT, &buffer[0]);
        if (dataread>=0) {
//            int position=0;
            for (int i=0; i< nCompund ; i++) {
                std::string compoundName(H5Tget_member_name (tid,i));
                QTreeWidgetItem *item3=new QTreeWidgetItem(item2,QStringList(QString(&compoundName[0])));
                item3->setFlags(item3->flags()|Qt::ItemIsEditable);
                hid_t compoundType=H5Tget_member_type(tid,i);
                size_t compundOffset=H5Tget_member_offset(tid,i);
                hid_t classCompoundType=H5Tget_class(compoundType);
                hid_t nativeType = H5Tget_native_type(compoundType,H5T_DIR_DEFAULT);

                if (H5Tequal(nativeType,H5T_NATIVE_DOUBLE)) {
                    double *val=(double*)(&(buffer[compundOffset]));
                    item3->setData(1,0,"double");
                    item3->setData(2,0,QLocale().toString(*val));
                } else if (H5Tequal(nativeType,H5T_NATIVE_FLOAT)) {
                    float *val=(float*)(&(buffer[compundOffset]));
                    item3->setData(1,0,"float");
                    item3->setData(2,0,QLocale().toString(*val));
                } else if (H5Tequal(nativeType,H5T_NATIVE_INT)) {
                    int *val=(int*)(&(buffer[compundOffset]));
                    item3->setData(1,0,"int");
                    item3->setData(2,0,QLocale().toString(*val));
                } else if (H5Tequal(nativeType,H5T_NATIVE_USHORT)) {
                    unsigned short *val=(unsigned short*)(&(buffer[compundOffset]));
                    item3->setData(1,0,"ushort");
                    item3->setData(2,0,QLocale().toString(*val));
                } else if (H5Tequal(nativeType,H5T_NATIVE_SHORT)) {
                    short *val=(short*)(&(buffer[compundOffset]));
                    item3->setData(1,0,"short");
                    item3->setData(2,0,QLocale().toString(*val));
                } else if (H5Tequal(nativeType,H5T_NATIVE_UINT)) {
                    unsigned int *val=(unsigned int*)(&(buffer[compundOffset]));
                    item3->setData(1,0,"uint");
                    item3->setData(2,0,QLocale().toString(*val));
                } else if (H5Tequal(nativeType,H5T_NATIVE_LONG)) {
                    long int *val=(long int*)(&(buffer[compundOffset]));
                    item3->setData(1,0,"long int");
                    item3->setData(2,0,QLocale().toString((long long int)(*val)));
                } else if (H5Tequal(nativeType,H5T_NATIVE_UCHAR)) {
                    unsigned char *val=(unsigned char*)(&(buffer[compundOffset]));
                    item3->setData(1,0,"uchar");
                    item3->setData(2,0,QLocale().toString((int)*val));
                } else if (H5Tequal(nativeType,H5T_NATIVE_CHAR)) {
                    char *val=(char*)(&(buffer[compundOffset]));
                    item3->setData(1,0,"char");
                    item3->setData(2,0,QLocale().toString((int)*val));
                } else if (classCompoundType==H5T_STRING) {
                    item3->setData(1,0,"String");
                    item3->setData(2,0,QLocale().toString((qintptr)H5Tget_size(nativeType)));
                } else if (classCompoundType==H5T_COMPOUND) {
                    item3->setData(1,0,"Compound");
                    item3->setData(2,0,QLocale().toString((qintptr)H5Tget_size(nativeType)));
                } else if (classCompoundType==H5T_ENUM) {
                    item3->setData(1,0,"Enum");
                    item3->setData(2,0,QLocale().toString((qintptr)H5Tget_size(nativeType)));
                } else if (classCompoundType==H5T_VLEN) {
                    item3->setData(1,0,"Vlen");
                    item3->setData(2,0,QLocale().toString((qintptr)H5Tget_size(nativeType)));
                } else {
                    item3->setData(1,0,"unknown "+QLocale().toString((unsigned int)classCompoundType));
                    item3->setData(2,0,QLocale().toString((qintptr)H5Tget_size(nativeType)));
                }

//                position+=H5Tget_size(nativeType);
            }
        }
    }
    QString strKind=QLocale().toString(ndims)+"D ";
    if (dims.size()==1) {
        item2->setData(1,0,"DS 1D");
        item2->setData(2,0,"Data 1D");
        item2->setData(3,0,QString::fromStdString(std::string(ds_name.begin(),ds_name.end())));
    } else if (dims.size()==2) {
        strKind+=QLocale().toString(dims[1])+"x"+QLocale().toString(dims[0]);
        QTreeWidgetItem *dataParent=item2->parent();
        while (dataParent) {
            dataParent->setExpanded(true);
            dataParent->setForeground(0,QBrush(QColor(0,0,200)));
            dataParent=dataParent->parent();
        }
        item2->setExpanded(true);
        item2->setData(2,0,strKind);
        qDebug() << QString::fromStdString(std::string(ds_name.begin(),ds_name.end()));

        item2->setData(3,0,QString::fromStdString(std::string(ds_name.begin(),ds_name.end())));
        item2->setForeground(0,QBrush(QColor(0,0,200)));
        item2->setForeground(1,QBrush(QColor(0,0,200)));
        item2->setForeground(2,QBrush(QColor(0,0,200)));
    } else {
        item2->setData(1,0,"DS "+QLocale().toString(t_class));
        item2->setData(2,0,"Data not supported");
        item2->setBackground(2,QBrush(QColor(200,200,255)));
    }

    H5Tclose(tid);
    H5Sclose(sid);
}

void OpenHdf5::scanAttribute(hid_t aid, QTreeWidgetItem *parentItem, nPhysD *my_data) {
    ssize_t len = 1+H5Aget_name(aid, 0, nullptr );
    std::vector<char> attrName(len);
    H5Aget_name(aid, len, &attrName[0] );

    QTreeWidgetItem *item3=new QTreeWidgetItem(parentItem,QStringList(QString(&attrName[0])));
    item3->setFlags(item3->flags()|Qt::ItemIsEditable);
    hid_t aspace = H5Aget_space(aid); /* the dimensions of the attribute data */
    hid_t atype  = H5Aget_type(aid);
    H5A_info_t aInfo;
    H5Aget_info(aid, &aInfo);
    hid_t nativeType = H5Tget_native_type(atype,H5T_DIR_DEFAULT);
    hid_t classAType=H5Tget_class(atype);
    if (classAType ==  H5T_FLOAT) {
        if (H5Tequal(nativeType,H5T_NATIVE_DOUBLE)) {
            int nelem=aInfo.data_size/sizeof(double);
            item3->setData(1,0,"Attr double");
            std::vector<double> val(nelem);
            if (H5Aread(aid, nativeType, (void*)(&val[0])) >= 0) {
                if (my_data && nelem==2) {
                    if (strcmp(&attrName[0],"physOrigin")==0) {
                        my_data->set_origin(val[0],val[1]);
                    } else if (strcmp(&attrName[0],"physScale")==0) {
                        my_data->set_scale(val[0],val[1]);
                    }
                }
                QString strData=QLocale().toString(val[0]);
                for (int i=1;i<nelem;i++) strData+=" "+QLocale().toString(val[i]);
                item3->setData(2,0,strData);
            }
        }
    } else if (classAType ==  H5T_INTEGER) {
        int nelem=aInfo.data_size/sizeof(int);
        item3->setData(1,0,"Attr int");
        std::vector<int> val(nelem);
        if (H5Aread(aid, nativeType, (void*)(&val[0])) >= 0) {
            QString strData=QLocale().toString(val[0]);
            for (int i=1;i<nelem;i++) strData+=" "+QLocale().toString(val[i]);
            item3->setData(2,0,strData);
        }
    } else if (classAType == H5T_STRING) {
        std::vector<char> val;
        item3->setData(1,0,"Attr string");
        if (my_data) {
            val.resize(1+aInfo.data_size);
            if (H5Aread(aid, nativeType, &val[0]) >= 0) {
                if (my_data) {
                    if (strcmp(&attrName[0],"physShortName")==0) my_data->setShortName(std::string(&val[0]));
                    if (strcmp(&attrName[0],"physName")==0) my_data->setName(std::string(&val[0]));
                }
                item3->setData(2,0,QString(&val[0]));
            }
        } else {
            hssize_t ssiz=H5Sget_simple_extent_npoints(aspace);
            size_t tsiz=H5Tget_size(atype);
            int sizeStr=1+ssiz*tsiz;
            //			hsize_t dims=0;
            if( sizeStr >= 0) {
                val.resize(sizeStr);
                if (H5Aread(aid, nativeType, &val[0]) >= 0) {
                    if (H5Tis_variable_str(atype)) {
                        item3->setData(2,0,QString(*((char**) (&val[0]))));
                    } else {
                        item3->setData(2,0,QString((&val[0])));
                    }
                }
            }

        }
    }
    H5Tclose(atype);
    H5Sclose(aspace);
}

void OpenHdf5::scanGroup(hid_t gid, QTreeWidgetItem *parentItem) {
    ssize_t sizeName=1+H5Iget_name(gid, nullptr,0);
    std::vector<char> group_name(sizeName);
    H5Iget_name(gid, &group_name[0],sizeName);

    for (int i = 0; i < H5Aget_num_attrs(gid); i++) {
        hid_t aid =	H5Aopen_idx(gid, (unsigned int)i );
        scanAttribute(aid,parentItem);
        H5Aclose(aid);
    }

    H5G_info_t infoGroup;
    H5Gget_info(gid,&infoGroup);
    for (hsize_t i = 0; i < infoGroup.nlinks; i++) {
        int size = 1+H5Lget_name_by_idx (gid, ".", H5_INDEX_NAME, H5_ITER_INC,i, nullptr, 0, H5P_DEFAULT);
        std::vector<char> memb_name(size);
        H5Lget_name_by_idx (gid, ".", H5_INDEX_NAME, H5_ITER_INC,i, &memb_name[0], size, H5P_DEFAULT);

        QString str_memb_name(&memb_name[0]);

        qDebug() << str_memb_name;
        QTreeWidgetItem *item2=new QTreeWidgetItem(parentItem,QStringList(str_memb_name));
        switch(H5Gget_objtype_by_idx(gid, i)) {
        case H5G_LINK: {
            item2->setData(1,0,"Link");
            std::vector<char> target(HDF5_MAX_NAME);
            H5Gget_linkval(gid, &memb_name[0], HDF5_MAX_NAME, &target[0]  ) ;
            item2->setData(2,0,QString(&target[0]));
            break;
        }
        case H5G_GROUP: {
            hid_t grpId = H5Gopen(gid,&memb_name[0], H5P_DEFAULT);
            scanGroup(grpId,item2);
            H5Gclose(grpId);
            break;
        }
        case H5G_TYPE: {
            hid_t typeId = H5Topen(gid,&memb_name[0], H5P_DEFAULT);
            item2->setData(1,0,H5Tget_class(typeId) < 0?"Type invalid":"Type valid");
            item2->setData(2,0,"Don't know what to do with this!");
            H5Tclose(typeId);
            break;
        }
        case H5G_DATASET: {
            hid_t did = H5Dopen(gid, &memb_name[0], H5P_DEFAULT);
            scanDataset(did, item2);
            H5Dclose(did);
            break;
        }
        default:
            item2->setData(1,0,"Unknown");
            break;
        }
    }
    QApplication::processEvents();
}


nPhysD* OpenHdf5::phys_open_Hdf5(QString fname, std::string dataName) {
    nPhysD *my_data=nullptr;
    hid_t fid = H5Fopen (QFile::encodeName(fname).constData(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid >= 0) {
        hid_t did = H5Dopen(fid,dataName.c_str(), H5P_DEFAULT);
        if (did>=0) {
            ssize_t sizeName=1+H5Iget_name(did, nullptr,0);
            std::vector<char> ds_name(sizeName);
            H5Iget_name(did, &ds_name[0],sizeName);
            hid_t sid = H5Dget_space(did);
            hid_t tid = H5Dget_type(did);

            size_t tsiz = H5Tget_size(tid);
            H5T_class_t t_class = H5Tget_class(tid);

            hid_t nativeType;
            std::vector<char> buffer;
            int ndims=0;
            hsize_t *dims=nullptr;

            hid_t file_space_id=H5S_ALL;
            int narray=1;
            if (t_class == H5T_FLOAT) {
                ndims=H5Sget_simple_extent_ndims(sid);
                dims=new hsize_t[ndims];
                H5Sget_simple_extent_dims(sid,dims,nullptr);
                int npoints=H5Sget_simple_extent_npoints(sid);
                buffer.resize(tsiz*npoints);
                nativeType=H5Tget_native_type(tid,H5T_DIR_DEFAULT);
            } else if (t_class == H5T_COMPOUND) {
                DEBUG(10, "compound not implemented");
            } else if(t_class == H5T_ARRAY) {
                ndims=H5Sget_simple_extent_ndims(sid);
                std::vector<hsize_t> dims(ndims);
                H5Sget_simple_extent_dims(sid,&dims[0],nullptr);
                for (auto &val :dims) {
                    narray*=val;
                }

                ndims = H5Tget_array_ndims(tid);
                dims.resize(ndims);
                H5Tget_array_dims2(tid, &dims[0]);
                buffer.resize(tsiz*narray);
                nativeType=H5Tget_native_type(H5Tget_super(tid),H5T_DIR_DEFAULT);
            }
            if (buffer.size()) {
                if(ndims==1) {
                    DEBUG("here");
                    std::vector<double> data(dims[0]);

                    H5Dread(did, tid, sid, file_space_id, H5P_DEFAULT, &buffer[0]);
                    for (int na=0;na<narray;na++) {
                        if (H5Tequal(nativeType,H5T_NATIVE_USHORT)) {
                            for (size_t k=0;k<data.size();k++) {
                                data[k]+=((unsigned short*) &buffer[0])[k];
                            }
                        } else if (H5Tequal(nativeType,H5T_NATIVE_INT)) {
                            for (size_t k=0;k<data.size();k++) {
                                data[k]+=((int*) &buffer[0])[k];
                            }
                        } else if (H5Tequal(nativeType,H5T_NATIVE_UINT)) {
                            for (size_t k=0;k<data.size();k++) {
                                data[k]+=((unsigned int*) &buffer[0])[k];
                            }
                        } else if (H5Tequal(nativeType,H5T_NATIVE_FLOAT)) {
                            for (size_t k=0;k<data.size();k++) {
                                data[k]+=((float*) &buffer[0])[k];
                            }
                        } else if (H5Tequal(nativeType,H5T_NATIVE_DOUBLE)) {
                            for (size_t k=0;k<data.size();k++) {
                                data[k]+=((double*) &buffer[0])[k];
                            }
                        }
                    }
                    QString str_data;
                    for (auto&val : data) {
                        str_data += QLocale().toString(val) + "\n";
                    }
                    QClipboard *clipboard = QGuiApplication::clipboard();
                    clipboard->setText(str_data);
                    statusbar->showMessage(QLocale().toString((int)data.size()) + " values copied into the clipboard",2000);


                } else if(ndims==2) {
                    std::string strName(ds_name.begin(),ds_name.end());
                    my_data=new nPhysD(strName);
                    my_data->setType(PHYS_FILE);
                    strName.erase(0,strName.find_last_of('/'));
                    my_data->setShortName(strName);

                    for (int i = 0; i < H5Aget_num_attrs(did); i++) {
                        hid_t aid =	H5Aopen_idx(did, (unsigned int)i );
                        scan_attributes(aid, my_data);
                        H5Aclose(aid);
                    }

                    H5Dread(did, tid, sid, file_space_id, H5P_DEFAULT, &buffer[0]);
                    my_data->resize(dims[1],dims[0]);
                    for (int na=0;na<narray;na++) {
                        if (H5Tequal(nativeType,H5T_NATIVE_USHORT)) {
                            for (size_t k=0;k<my_data->getSurf();k++) {
                                my_data->set(k,my_data->point(k)+((unsigned short*) &buffer[0])[k]);
                            }
                        } else if (H5Tequal(nativeType,H5T_NATIVE_INT)) {
                            for (size_t k=0;k<my_data->getSurf();k++) {
                                my_data->set(k,my_data->point(k)+((int*) &buffer[0])[k+na*my_data->getSurf()]);
                            }
                        } else if (H5Tequal(nativeType,H5T_NATIVE_UINT)) {
                            for (size_t k=0;k<my_data->getSurf();k++) {
                                my_data->set(k,my_data->point(k)+((unsigned int*) &buffer[0])[k+na*my_data->getSurf()]);
                            }
                        } else if (H5Tequal(nativeType,H5T_NATIVE_FLOAT)) {
                            for (size_t k=0;k<my_data->getSurf();k++) {
                                my_data->set(k,my_data->point(k)+((float*) &buffer[0])[k+na*my_data->getSurf()]);
                            }
                        } else if (H5Tequal(nativeType,H5T_NATIVE_DOUBLE)) {
                            for (size_t k=0;k<my_data->getSurf();k++) {
                                my_data->set(k,my_data->point(k)+((double*) &buffer[0])[k+na*my_data->getSurf()]);
                            }
                        }
                    }
                    for (size_t k=0;k<my_data->getSurf();k++) {
                        my_data->set(k,my_data->point(k)/narray);
                    }
                }

            }

            H5Tclose(tid);
            H5Sclose(sid);
            H5Dclose(did);
            H5Fclose(fid);
        }
    }
    if (my_data) my_data->TscanBrightness();
    return my_data;
}


int OpenHdf5::phys_write_Hdf5(nPhysD *phys, std::string fname) {
    if (phys) {
        if (H5Zfilter_avail(H5Z_FILTER_DEFLATE)) {
            unsigned int	filter_info;
            H5Zget_filter_info (H5Z_FILTER_DEFLATE, &filter_info);
            if ( (filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED) && (filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED) ) {
                hid_t file_id = H5Fcreate (fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

                hsize_t dims[2]={phys->getH(),phys->getW()};
                hid_t space = H5Screate_simple (2, dims, nullptr);

                hid_t dcpl = H5Pcreate (H5P_DATASET_CREATE);
                H5Pset_deflate (dcpl, 9);
                hsize_t chunk[2] = {4, 8};
                H5Pset_chunk (dcpl, 2, chunk);

                hid_t dset = H5Dcreate (file_id, "/neutrino", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
                H5Dwrite (dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, phys->Timg_buffer);

                H5LTset_attribute_string(file_id,"/","version", qApp->applicationVersion().toStdString().c_str());
                double data[2];
                data[0]=phys->get_origin().x();
                data[1]=phys->get_origin().y();
                H5LTset_attribute_double(file_id,"neutrino","physOrigin", data, 2);
                data[0]=phys->get_scale().x();
                data[1]=phys->get_scale().y();
                H5LTset_attribute_double(file_id,"neutrino","physScale", data, 2);
                H5LTset_attribute_string(file_id,"neutrino","physName", phys->getName().c_str());
                H5LTset_attribute_string(file_id,"neutrino","physShortName", phys->getShortName().c_str());


                H5Pclose (dcpl);
                H5Dclose (dset);
                H5Sclose (space);
                H5Fclose (file_id);
            }
        }
        return 0;
    }
    return -1;
}

void OpenHdf5::scan_attributes(hid_t aid, nPhysD *my_data){
    ssize_t len = 1+H5Aget_name(aid, 0, nullptr );
    std::vector<char> attrName(len);
    H5Aget_name(aid, len, &attrName[0] );
    std::string attrNameStr(attrName.begin(),attrName.end());
    DEBUG(attrNameStr);
    hid_t aspace = H5Aget_space(aid); /* the dimensions of the attribute data */
    hid_t atype  = H5Aget_type(aid);
    H5A_info_t aInfo;
    H5Aget_info(aid, &aInfo);
    hid_t nativeType = H5Tget_native_type(atype,H5T_DIR_DEFAULT);
    hid_t classAType=H5Tget_class(atype);
    if (classAType ==  H5T_FLOAT) {
        if (H5Tequal(nativeType,H5T_NATIVE_DOUBLE)) {
            int nelem=aInfo.data_size/sizeof(double);
            std::vector<double> val(nelem);
            if (H5Aread(aid, nativeType, (void*)(&val[0])) >= 0) {
                if (my_data && nelem==2) {
                    if (attrNameStr=="physOrigin") {
                        my_data->set_origin(val[0],val[1]);
                    } else if (attrNameStr=="physScale") {
                        my_data->set_scale(val[0],val[1]);
                    }
                }
            }
        }
    } else if (classAType ==  H5T_INTEGER) {
        int nelem=aInfo.data_size/sizeof(int);
        std::vector<int> val(nelem);
        if (H5Aread(aid, nativeType, (void*)(&val[0])) >= 0) {
            my_data->prop[attrNameStr]=val[0];
        }
    } else if (classAType == H5T_STRING) {
        std::vector<char> val;
        if (my_data) {
            int sizeStr=1+aInfo.data_size;
            val.resize(sizeStr);
            if (H5Aread(aid, nativeType, &val[0]) >= 0) {
                if (my_data) {
                    if (attrNameStr=="physShortName") my_data->setShortName(std::string(val.begin(),val.end()));
                    if (attrNameStr=="physName") my_data->setName(std::string(val.begin(),val.end()));
                }
            }
        } else {
            hssize_t ssiz=H5Sget_simple_extent_npoints(aspace);
            size_t tsiz=H5Tget_size(atype);
            int sizeStr=ssiz*tsiz;
            //			hsize_t dims=0;
            if( sizeStr >= 0) {
                val.resize(sizeStr);
                if (H5Aread(aid, nativeType, &val[0]) >= 0) {
                    if (H5Tis_variable_str(atype)) {
                        // use this *((char**) val);
                    } else {
                        // use this val);
                    }
                }
            }

        }
    }
    H5Tclose(atype);
    H5Sclose(aspace);
}
#endif
