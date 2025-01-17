/*
 *
 *    Copyright (C) 2013 Alessand Flacco, Tommaso Vinci All Rights Reserved
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
#include "neutrino.h"
#include "Image_list.h"

Image_list::~Image_list() {
    qDebug() << "close Image_list";
    disconnect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePad(nPhysD*)));
    disconnect(nparent, SIGNAL(physAdd(nPhysD*)), this, SLOT(physAdd(nPhysD*)));
    disconnect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
    disconnect(images, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()));
    qDebug() << "close Image_list ...... really";
}

Image_list::Image_list(neutrino *nparent) : nGenericPan(nparent) {
    setupUi(this);

    images->header()->setSectionResizeMode(0,QHeaderView::ResizeToContents);

    images->header()->setStretchLastSection (true);
    images->sortByColumn(0,Qt::AscendingOrder);

    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePad(nPhysD*)));
    connect(nparent, SIGNAL(physAdd(nPhysD*)), this, SLOT(physAdd(nPhysD*)));
    connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));

    connect(images, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()));

	foreach (nPhysD *phys, nparent->getBufferList()) {
		physAdd(phys);
	}

    connect(actionShort, SIGNAL(triggered()), this, SLOT(changeProperties()));
    connect(actionName, SIGNAL(triggered()), this, SLOT(changeProperties()));
    connect(actionOrigin, SIGNAL(triggered()), this, SLOT(changeProperties()));
    connect(actionScale, SIGNAL(triggered()), this, SLOT(changeProperties()));
    connect(actionRemove, SIGNAL(triggered()), this, SLOT(buttonRemovePhys()));
    connect(actionCopy, SIGNAL(triggered()), this, SLOT(buttonCopyPhys()));
    connect(actionRescale, SIGNAL(triggered()), this, SLOT(changeProperties()));

    show(true);
    updatePad(currentBuffer);
}

void Image_list::on_horizontalSlider_valueChanged(int val) {
    int numphys=nparent->getBufferList().size();
    if (val<numphys) {
        nPhysD *to_display=nparent->getBufferList().at(val);
        if (nPhysExists(to_display)) {
            nparent->showPhys(to_display);
        }
    }
}

void Image_list::selectionChanged() {
    QList<QTreeWidgetItem *> sel=images->selectedItems();
    if (sel.size()) {
        QTreeWidgetItem *item=sel.last();
        if (item) {
            nPhysD *my_phys=getPhys(item);
            if (nPhysExists(my_phys)) {
                disconnect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePad(nPhysD*)));
                lineEdit->setText(QString::fromUtf8(my_phys->getFromName().c_str()));
                lineEdit->setCursorPosition(0);
                nparent->showPhys(my_phys);
                connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePad(nPhysD*)));
            } else {
                delete item;
            }
        }
    }
}

nPhysD*
Image_list::getPhys(QTreeWidgetItem* item) {
    for (auto & my_key : itemsMap) {
        if(my_key.second == item) return my_key.first;
    }
    return nullptr;
}

void
Image_list::buttonCopyPhys() {
    foreach (QTreeWidgetItem* item, images->selectedItems()) {
        nPhysD *copyPhys=new nPhysD(*getPhys(item));
        nparent->addShowPhys(copyPhys);
    }
}


void
Image_list::buttonRemovePhys() {
    QList<nPhysD*> my_list;
    for (auto& item: images->selectedItems()) {
        my_list << getPhys(item);
    }
    for (auto& my_phys: my_list) {
        nparent->removePhys(my_phys);
    }
}

void
Image_list::changeProperties() {
    QList<nPhysD*> physSelected;
    QList<QTreeWidgetItem*> itemsSelected;
    foreach (QTreeWidgetItem* item, images->selectedItems()) {
        nPhysD *phys=getPhys(item);
        if (phys) {
            physSelected << phys;
            itemsSelected << item;
        }
    }
    if (physSelected.isEmpty() && currentBuffer) {
        for( int i = 0; i < images->topLevelItemCount(); ++i ) {
            QTreeWidgetItem *item = images->topLevelItem( i );
            nPhysD *phys=getPhys(item);
            if (phys==currentBuffer) {
                physSelected << phys;
                itemsSelected << item;
            }
        }
    }
    bool ok;
    QString text;

    if (physSelected.size()>0) {
        if (sender()==actionShort) {
            text = QInputDialog::getText(this, tr("Change Short Name"),tr("Short name:"), QLineEdit::Normal, itemsSelected.last()->data(1,Qt::DisplayRole).toString(), &ok);
            if (ok && !text.isEmpty()) {
                foreach (QTreeWidgetItem* item, itemsSelected) {
                    item->setData(1,Qt::DisplayRole,text);
                }
                foreach (nPhysD* phys, physSelected) {
                    phys->setShortName(text.toStdString());
                    nparent->emitBufferChanged(phys);
                }
            }
        } else if (sender()==actionScale) {
            text = QInputDialog::getText(this, tr("Change Scale"),tr("Scale:"), QLineEdit::Normal, itemsSelected.last()->data(2,Qt::DisplayRole).toString(), &ok);
            if (ok && !text.isEmpty()) {
                QStringList lista=text.split(' ', Qt::SkipEmptyParts);
                bool ok1,ok2;
                switch (lista.size()) {
                case 1: {
                    bool ok1;
                    double val=QLocale().toDouble(lista.at(0),&ok1);
                    if (ok1) {
                        foreach (nPhysD* phys, physSelected) {
                            phys->set_scale(val,val);
                            nparent->emitBufferChanged(phys);
                        }
                        foreach (QTreeWidgetItem* item, itemsSelected) {
                            item->setData(2,Qt::DisplayRole,lista.at(0));
                        }
                        nparent->my_view->update();
                    }
                    break;
                }
                case 2: {
                    double xVal=QLocale().toDouble(lista.at(0),&ok1);
                    double yVal=QLocale().toDouble(lista.at(1),&ok2);
                    if (ok1 && ok2) {
                        foreach (nPhysD* phys, physSelected) {
                            phys->set_scale(xVal,yVal);
                            nparent->emitBufferChanged(phys);
                        }
                        foreach (QTreeWidgetItem* item, itemsSelected) {
                            item->setData(2,Qt::DisplayRole,lista.at(0)+" "+lista.at(1));
                        }
                        nparent->my_view->update();
                    }
                    break;
                }
                default:
                    break;
                }
            }
        } else if (sender()==actionOrigin) {
            text = QInputDialog::getText(this, tr("Change Origin"),tr("Origin:"), QLineEdit::Normal, itemsSelected.last()->data(3,Qt::DisplayRole).toString(), &ok);
            if (ok && !text.isEmpty()) {
                QStringList lista=text.split(' ', Qt::SkipEmptyParts);
                if (lista.size()==2) {
                    bool ok1,ok2;
                    int xOrigin=QLocale().toDouble(lista.at(0),&ok1);
                    int yOrigin=QLocale().toDouble(lista.at(1),&ok2);
                    if (ok1 && ok2) {
                        foreach (nPhysD* phys, physSelected) {
                            phys->set_origin(xOrigin,yOrigin);
                            nparent->emitBufferChanged(phys);
                        }
                        nparent->my_view->update();
                        foreach (QTreeWidgetItem* item, itemsSelected) {
                            item->setData(3,Qt::DisplayRole,lista.at(0)+" "+lista.at(1));
                        }
                    }
                }
            }

        } else if (sender()==actionName) {
            text = QInputDialog::getText(this, tr("Change Name"),tr("Name:"), QLineEdit::Normal, itemsSelected.last()->data(4,0).toString(), &ok);
            if (ok && !text.isEmpty()) {
                foreach (QTreeWidgetItem* item, itemsSelected) {
                    item->setData(4,0,text);
                }
                foreach (nPhysD* phys, physSelected) {
                    phys->setName(text.toStdString());
                    nparent->emitBufferChanged(phys);
                }
            }
        } else if (sender()==actionRescale) {
            vec2i lastsize=physSelected.last()->getSize();
            std::stringstream ss;
            ss << lastsize.x() << " " << lastsize.y();
            text = QInputDialog::getText(this, tr("Change Size"),tr("Size:"), QLineEdit::Normal, QString::fromStdString(ss.str()), &ok);
            if (ok && !text.isEmpty()) {
                QStringList lista=text.split(' ', Qt::SkipEmptyParts);
                if (lista.size()==1) {
                    bool ok1;
                    double scale=QLocale().toDouble(lista.at(0),&ok1);
                    if (ok1) {
                        foreach (nPhysD* phys, physSelected) {
                            vec2i newsize= phys->getSize()*scale;
                            nPhysD *resized=new nPhysD(physMath::phys_resample(*phys, newsize));
                            nparent->addShowPhys(resized);
                        }
                    }
                } else if (lista.size()==2) {
                    bool ok1,ok2;
                    int sizex=QLocale().toInt(lista.at(0),&ok1);
                    int sizey=QLocale().toInt(lista.at(1),&ok2);
                    if (ok1  && ok2) {
                        foreach (nPhysD* phys, physSelected) {
                            nPhysD *resized=new nPhysD(physMath::phys_resample(*phys, vec2i(sizex, sizey)));
                            nparent->addShowPhys(resized);
                        }
                    }
                }
            }
        }
    }
}

void Image_list::keyPressEvent(QKeyEvent *e){
    switch (e->key()) {
    case Qt::Key_Return:
        nparent->showPhys(getPhys(images->selectedItems().first()));
        break;
    case Qt::Key_Backspace:
    case Qt::Key_Delete:
        buttonRemovePhys();
        break;
    }
    e->accept();
}


void
Image_list::updatePad(nPhysD *my_phys) {
    disconnect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePad(nPhysD*)));
    disconnect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
    disconnect(images, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()));
    if (nPhysExists(my_phys)) {
        qDebug() << "<=><=><=><=><=><=> 1" << itemsMap.size();
        QTreeWidgetItem* it=itemsMap[my_phys];
        if (!it) {
            it=new QTreeWidgetItem(images, QTreeWidgetItem::UserType);
            itemsMap[my_phys]=it;
        }
        qDebug() << "<=><=><=><=><=><=> 2" << itemsMap.size();
        if (it) {
            it->setData(1,0,QString(my_phys->getShortName().c_str()));
            if (my_phys->get_scale().x()==my_phys->get_scale().y()) {
                it->setData(2,0,QLocale().toString(my_phys->get_scale().x()));
            } else {
                it->setData(2,0,QLocale().toString(my_phys->get_scale().x())+" "+QLocale().toString(my_phys->get_scale().y()));
            }
            it->setData(3,0,QLocale().toString(my_phys->get_origin().x())+" "+QLocale().toString(my_phys->get_origin().y()));
            it->setData(4,0,QString::fromUtf8(my_phys->getName().c_str()));
            if (nPhysExists(my_phys)) {
                lineEdit->setText(QString::fromUtf8(my_phys->getFromName().c_str()));
                lineEdit->setCursorPosition(0);
            }
            qDebug() << "<=><=><=><=><=><=> 3 " << itemsMap.size();
            for (auto const & my_key : itemsMap) {
                nPhysD* key_phys = my_key.first;
                bool sel (key_phys == my_phys);
                if (my_key.first && my_key.second) {
                    my_key.second->setSelected(sel);
                }
            }
            if (nparent->getBufferList().size())
                horizontalSlider->setMaximum(nparent->getBufferList().size()-1);
        }
        // renumber everything in case some images are removed
        for (int k=0;k<nparent->getBufferList().size();k++){
            nPhysD* phys=nparent->getBuffer(k);
            if (phys) {
                    QTreeWidgetItem* it2=itemsMap[phys];
                    if (it2) {
                        QString mynum=QString::number(k).rightJustified(1+log10(nparent->getBufferList().size()), ' ');;
                        if (mynum!=it2->data(0,Qt::DisplayRole).toString()) {
                            it2->setData(0,0,mynum);
                        }
                    }
            }
        }
        horizontalSlider->setVisible(nparent->getBufferList().size()>1);

    }
    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePad(nPhysD*)));
    connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
    connect(images, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()));
}


void
Image_list::physDel(nPhysD *my_phys) {
    DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> enter");
    qDebug() << itemsMap[my_phys];
    qDebug() << "<=><=><=><=><=><=> del before " << itemsMap.size();
    qDebug() << "<=><=><=><=><=><=> del before " << my_phys;
    qDebug() << "<=><=><=><=><=><=> del before " << itemsMap[my_phys];
    delete itemsMap[my_phys];
    itemsMap.erase(my_phys);
    if (itemsMap.size() == 0) {
        lineEdit->setText(tr("No image"));
    }
    qDebug() << "<=><=><=><=><=><=> del after " << itemsMap.size();
}

/// new image entry point
void Image_list::physAdd(nPhysD *my_phys) {
    updatePad(my_phys);
}

