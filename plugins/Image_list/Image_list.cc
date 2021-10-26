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
Image_list::Image_list(neutrino *nparent) : nGenericPan(nparent),
    freezedFrame(false), frScale(1,1), frOrigin(0,0)
{
    my_w.setupUi(this);

    my_w.images->header()->setSectionResizeMode(0,QHeaderView::ResizeToContents);

    my_w.images->header()->setStretchLastSection (true);
//    my_w.images->header()->setSectionsMovable(false);

    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePad(nPhysD*)));
    connect(nparent, SIGNAL(physAdd(nPhysD*)), this, SLOT(physAdd(nPhysD*)));
    connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
    connect(nparent->my_w->my_view, SIGNAL(bufferOriginChanged()), this, SLOT(originChanged()));

    connect(my_w.images, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()));


	foreach (nPhysD *phys, nparent->getBufferList()) {
		physAdd(phys);
	}

    connect(my_w.actionShort, SIGNAL(triggered()), this, SLOT(changeProperties()));
    connect(my_w.actionName, SIGNAL(triggered()), this, SLOT(changeProperties()));
    connect(my_w.actionOrigin, SIGNAL(triggered()), this, SLOT(changeProperties()));
    connect(my_w.actionScale, SIGNAL(triggered()), this, SLOT(changeProperties()));
    connect(my_w.actionRemove, SIGNAL(triggered()), this, SLOT(buttonRemovePhys()));
    connect(my_w.actionCopy, SIGNAL(triggered()), this, SLOT(buttonCopyPhys()));
    connect(my_w.actionFreeze, SIGNAL(toggled(bool)), this, SLOT(setFreezed(bool)));
    connect(my_w.actionRescale, SIGNAL(triggered()), this, SLOT(changeProperties()));

    show(true);
    updatePad(currentBuffer);
}

void Image_list::on_horizontalSlider_valueChanged(int val) {
    int numphys=nparent->getBufferList().size();
    if (val<numphys) {
        nparent->showPhys(nparent->getBufferList().at(val));
    }
}

void Image_list::selectionChanged() {
    QList<QTreeWidgetItem *> sel=my_w.images->selectedItems();
    if (sel.size()) {
        QTreeWidgetItem *item=sel.last();
        if (item) {
            nPhysD *my_phys=getPhys(item);
            if (nPhysExists(my_phys)) {
                disconnect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePad(nPhysD*)));
                my_w.lineEdit->setText(QString::fromUtf8(my_phys->getFromName().c_str()));
                my_w.lineEdit->setCursorPosition(0);
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
    foreach (QTreeWidgetItem* item, my_w.images->selectedItems()) {
        nPhysD *copyPhys=new nPhysD(*getPhys(item));
        nparent->addShowPhys(copyPhys);
    }
}


void
Image_list::buttonRemovePhys() {
    QList<nPhysD*> my_list;
    for (auto& item: my_w.images->selectedItems()) {
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
    foreach (QTreeWidgetItem* item, my_w.images->selectedItems()) {
        nPhysD *phys=getPhys(item);
        if (phys) {
            physSelected << phys;
            itemsSelected << item;
        }
    }
    if (physSelected.isEmpty() && currentBuffer) {
        for( int i = 0; i < my_w.images	->topLevelItemCount(); ++i ) {
            QTreeWidgetItem *item = my_w.images->topLevelItem( i );
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
        if (sender()==my_w.actionShort) {
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
        } else if (sender()==my_w.actionScale) {
            text = QInputDialog::getText(this, tr("Change Scale"),tr("Scale:"), QLineEdit::Normal, itemsSelected.last()->data(2,Qt::DisplayRole).toString(), &ok);
            if (ok && !text.isEmpty()) {
                QStringList lista=text.split(' ', QString::SkipEmptyParts);
                bool ok1,ok2;
                switch (lista.size()) {
                case 1: {
                    bool ok1;
                    double val=QLocale().toDouble(lista.at(0),&ok1);


                    if (ok1) {
                        // also update ref. scale (in case button was toggled)
                        frScale = vec2f(val, val);

                        foreach (nPhysD* phys, physSelected) {
                            phys->set_scale(val,val);
                            nparent->emitBufferChanged(phys);
                        }
                        foreach (QTreeWidgetItem* item, itemsSelected) {
                            item->setData(2,Qt::DisplayRole,lista.at(0));
                        }
                        nparent->my_w->my_view->update();
                    }
                    break;
                }
                case 2: {
                    double xVal=QLocale().toDouble(lista.at(0),&ok1);
                    double yVal=QLocale().toDouble(lista.at(1),&ok2);
                    if (ok1 && ok2) {
                        // also update ref. scale (in case button was toggled)
                        frScale = vec2f(xVal, yVal);

                        foreach (nPhysD* phys, physSelected) {
                            phys->set_scale(xVal,yVal);
                            nparent->emitBufferChanged(phys);
                        }
                        foreach (QTreeWidgetItem* item, itemsSelected) {
                            item->setData(2,Qt::DisplayRole,lista.at(0)+" "+lista.at(1));
                        }
                        nparent->my_w->my_view->update();
                    }
                    break;
                }
                default:
                    break;
                }
            }
        } else if (sender()==my_w.actionOrigin) {
            text = QInputDialog::getText(this, tr("Change Origin"),tr("Origin:"), QLineEdit::Normal, itemsSelected.last()->data(3,Qt::DisplayRole).toString(), &ok);
            if (ok && !text.isEmpty()) {
                QStringList lista=text.split(' ', QString::SkipEmptyParts);
                if (lista.size()==2) {
                    bool ok1,ok2;
                    int xOrigin=QLocale().toDouble(lista.at(0),&ok1);
                    int yOrigin=QLocale().toDouble(lista.at(1),&ok2);

                    // also update ref. origin (in case button was toggled)
                    frOrigin = vec2f(xOrigin, yOrigin);

                    if (ok1 && ok2) {
                        foreach (nPhysD* phys, physSelected) {
                            phys->set_origin(xOrigin,yOrigin);
                            nparent->emitBufferChanged(phys);
                        }
                        nparent->my_w->my_view->update();
                        foreach (QTreeWidgetItem* item, itemsSelected) {
                            item->setData(3,Qt::DisplayRole,lista.at(0)+" "+lista.at(1));
                        }
                    }
                }
            }

        } else if (sender()==my_w.actionName) {
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
        } else if (sender()==my_w.actionRescale) {
            vec2i lastsize=physSelected.last()->getSize();
            std::stringstream ss;
            ss << lastsize.x() << " " << lastsize.y();
            text = QInputDialog::getText(this, tr("Change Size"),tr("Size:"), QLineEdit::Normal, QString::fromStdString(ss.str()), &ok);
            if (ok && !text.isEmpty()) {
                QStringList lista=text.split(' ', QString::SkipEmptyParts);
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
        nparent->showPhys(getPhys(my_w.images->selectedItems().first()));
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
    disconnect(my_w.images, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()));
    if (nPhysExists(my_phys)) {
        qDebug() << "<=><=><=><=><=><=> 1" << itemsMap.size();
        QTreeWidgetItem* it=itemsMap[my_phys];
        if (!it) {
            it=new QTreeWidgetItem(my_w.images, QTreeWidgetItem::UserType);
            itemsMap[my_phys]=it;
        }
        qDebug() << "<=><=><=><=><=><=> 2" << itemsMap.size();
        if (it) {
            std::ostringstream oss;
            oss << std::setw(5) << std::setfill(' ') << int(my_phys->prop["uuid"]);
            it->setData(0,0,QString::fromStdString(oss.str()));
            it->setData(1,0,QString(my_phys->getShortName().c_str()));
            if (my_phys->get_scale().x()==my_phys->get_scale().y()) {
                it->setData(2,0,QLocale().toString(my_phys->get_scale().x()));
            } else {
                it->setData(2,0,QLocale().toString(my_phys->get_scale().x())+" "+QLocale().toString(my_phys->get_scale().y()));
            }
            it->setData(3,0,QLocale().toString(my_phys->get_origin().x())+" "+QLocale().toString(my_phys->get_origin().y()));
            it->setData(4,0,QString::fromUtf8(my_phys->getName().c_str()));
            if (nPhysExists(my_phys)) {
                my_w.lineEdit->setText(QString::fromUtf8(my_phys->getFromName().c_str()));
                my_w.lineEdit->setCursorPosition(0);
            }
            qDebug() << "<=><=><=><=><=><=> 3 " << itemsMap.size();
            for (auto const & my_key : itemsMap) {
                nPhysD* pippo=my_key.first;
                bool sel (pippo ==my_phys);
                my_key.second->setSelected(sel);
            }
            if (nparent->getBufferList().size())
                my_w.horizontalSlider->setMaximum(nparent->getBufferList().size()-1);
        }
    }
    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePad(nPhysD*)));
    connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
    connect(my_w.images, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()));
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
        my_w.lineEdit->setText(tr("No image"));
    }
    qDebug() << "<=><=><=><=><=><=> del after " << itemsMap.size();
}

/// new image entry point
void Image_list::physAdd(nPhysD *my_phys) {
    if (freezedFrame) {
        my_phys->set_scale(frScale);
        my_phys->set_origin(frOrigin);
    }
    updatePad(my_phys);
}


/// set static scale/origin
void
Image_list::setFreezed(bool st) {
    if (st) {
        DEBUG("setFreezed");
        if (!freezedFrame) {
            freezedFrame = true;

            // blabla
            if (currentBuffer) {
                frScale = currentBuffer->get_scale();
                frOrigin = currentBuffer->get_origin();

                DEBUG("freeze scale: "<<frScale);
                DEBUG("freeze origin: "<<frOrigin);
            }

            return;
        }
    } else {
        DEBUG("unsetFreezed");
        freezedFrame = false;
    }
}


/// called for external origin change
void
Image_list::originChanged() {
    if (currentBuffer) {
        frScale = currentBuffer->get_scale();
        frOrigin = currentBuffer->get_origin();

        DEBUG("freeze scale: "<<frScale);
        DEBUG("freeze origin: "<<frOrigin);
    }
}
