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
#include "neutrino.h"
#include "nWinList.h"
nWinList::nWinList(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname) {
	my_w.setupUi(this);

	// CHECK: this should be ok...

	my_w.images->nparent=nparent;
	
	my_w.images->header()->setResizeMode(0,QHeaderView::ResizeToContents);
	my_w.images->header()->setResizeMode(1,QHeaderView::ResizeToContents);
	my_w.images->header()->setResizeMode(2,QHeaderView::ResizeToContents);
	my_w.images->header()->setStretchLastSection (true);

	my_w.images->setColumnHidden((my_w.images->columnCount()-1),true);

	connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePad(nPhysD*)));
	connect(nparent, SIGNAL(physAdd(nPhysD*)), this, SLOT(physAdd(nPhysD*)));
	connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
	
	foreach (nPhysD *phys, nparent->physList) physAdd(phys);
	updatePad(nparent->currentBuffer);

	QWidget* empty = new QWidget(this);
	empty->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Preferred);
	my_w.toolBar->insertWidget(my_w.actionPans,empty);
	
	my_w.pans->setVisible(my_w.actionPans->isChecked());
	my_w.images->setHidden(my_w.actionPans->isChecked());
	
//	my_w.statusBar->addPermanentWidget(my_w.line,1);
	
	connect(my_w.pans, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(panClicked(QListWidgetItem*)));
	connect(my_w.actionShort, SIGNAL(triggered()), this, SLOT(changeProperties()));
	connect(my_w.actionName, SIGNAL(triggered()), this, SLOT(changeProperties()));
	connect(my_w.actionOrigin, SIGNAL(triggered()), this, SLOT(changeProperties()));
	connect(my_w.actionScale, SIGNAL(triggered()), this, SLOT(changeProperties()));
	connect(my_w.actionRemove, SIGNAL(triggered()), this, SLOT(buttonRemovePhys()));
	connect(my_w.actionCopy, SIGNAL(triggered()), this, SLOT(buttonCopyPhys()));

	connect(nparent, SIGNAL(panAdd(nGenericPan*)), this, SLOT(panAdd(nGenericPan*)));
	connect(nparent, SIGNAL(panDel(nGenericPan*)), this, SLOT(panDel(nGenericPan*)));
	foreach (nGenericPan* pan, nparent->getPans()) {
		panAdd(pan);
	}
	decorate();
}

nPhysD*
nWinList::getPhys(QTreeWidgetItem* item) {
	nPhysD *retphys=(nPhysD*) (item->data((my_w.images->columnCount()-1),0).value<void*>());
	retphys->property.dumper(std::cerr);
	return retphys;
}

void
nWinList::buttonCopyPhys() {
	foreach (QTreeWidgetItem* item, my_w.images->selectedItems()) {
		nPhysD *copyPhys=new nPhysD(*getPhys(item));
		copyPhys->TscanBrightness();
		nparent->addShowPhys(copyPhys);
	}
}

void
nWinList::buttonRemovePhys() {
	foreach (QTreeWidgetItem* item, my_w.images->selectedItems()) {
		nparent->removePhys(getPhys(item));
	}
}

void
nWinList::changeProperties() {
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
	qDebug() << physSelected;
	qDebug() << itemsSelected;
	if (physSelected.size()>0) {
		if (sender()==my_w.actionShort) {
			text = QInputDialog::getText(this, tr("Change Short Name"),tr("Short name:"), QLineEdit::Normal, itemsSelected.last()->data(1,0).toString(), &ok);
			if (ok && !text.isEmpty()) {
				foreach (QTreeWidgetItem* item, itemsSelected) {
					item->setData(1,0,text);
				}
				foreach (nPhysD* phys, physSelected) {
					phys->setShortName(text.toStdString());
					nparent->emitBufferChanged(phys);
				}
			}
		} else if (sender()==my_w.actionName) {
			text = QInputDialog::getText(this, tr("Change Name"),tr("Name:"), QLineEdit::Normal, itemsSelected.last()->data(2,0).toString(), &ok);
			if (ok && !text.isEmpty()) {
				foreach (QTreeWidgetItem* item, itemsSelected) {
					item->setData(2,0,text);
				}
				foreach (nPhysD* phys, physSelected) {
					phys->setName(text.toStdString());
					nparent->emitBufferChanged(phys);
				}
			}
		} else if (sender()==my_w.actionOrigin) {
			text = QInputDialog::getText(this, tr("Change Origin"),tr("Origin:"), QLineEdit::Normal, itemsSelected.last()->data(3,0).toString(), &ok);
			if (ok && !text.isEmpty()) {
				QStringList lista=text.split(' ', QString::SkipEmptyParts);
				if (lista.size()==2) {
					bool ok1,ok2;
					int xOrigin=lista.at(0).toDouble(&ok1);
					int yOrigin=lista.at(1).toDouble(&ok2);
					if (ok1 && ok2) {
						foreach (nPhysD* phys, physSelected) {
							phys->set_origin(xOrigin,yOrigin);
							nparent->emitBufferChanged(phys);
						}
						nparent->my_tics.update();
						foreach (QTreeWidgetItem* item, itemsSelected) {
							item->setData(3,0,lista.at(0)+" "+lista.at(1));
						}
					}
				}	
			}
		} else if (sender()==my_w.actionScale) {
			text = QInputDialog::getText(this, tr("Change Scale"),tr("Scale:"), QLineEdit::Normal, itemsSelected.last()->data(4,0).toString(), &ok);
			if (ok && !text.isEmpty()) {
				QStringList lista=text.split(' ', QString::SkipEmptyParts);
				bool ok1,ok2;
				switch (lista.size()) {
					case 1: {
						bool ok1;
						double val=lista.at(0).toDouble(&ok1);
						if (ok1) {
							foreach (nPhysD* phys, physSelected) {
								phys->set_scale(val,val);
								nparent->emitBufferChanged(phys);
							}
							foreach (QTreeWidgetItem* item, itemsSelected) {
								item->setData(4,0,lista.at(0));
							}
							nparent->my_tics.update();
						}
						break;
					}
					case 2: {
						double xVal=lista.at(0).toDouble(&ok1);
						double yVal=lista.at(1).toDouble(&ok2);
						if (ok1 && ok2) {
							foreach (nPhysD* phys, physSelected) {
								phys->set_scale(xVal,yVal);
								nparent->emitBufferChanged(phys);
							}
							foreach (QTreeWidgetItem* item, itemsSelected) {
								item->setData(4,0,lista.at(0)+" "+lista.at(1));
							}
							nparent->my_tics.update();
						}
						break;
					}
					default:
						break;
				}
			}
		}
	} 
}

void
nWinList::panAdd(nGenericPan *pan) {
	if (pan->panName!=panName) {
		QListWidgetItem *item=new QListWidgetItem(pan->panName,my_w.pans);
		item->setData(Qt::UserRole,qVariantFromValue((void*)pan));
		my_w.pans->addItem(item);
	}
}

void
nWinList::panDel(nGenericPan *pan) {
	foreach (QListWidgetItem * item,my_w.pans->findItems(pan->panName,Qt::MatchExactly)) {
		delete item;
	}
}

void
nWinList::panClicked(QListWidgetItem* item) {
	nGenericPan* pan=(nGenericPan*)(item->data(Qt::UserRole).value<void*>());
	pan->setWindowState(windowState() & ~Qt::WindowMinimized | Qt::WindowActive);
}

void
nWinList::updatePad(nPhysD *my_phys) {
	QTreeWidgetItemIterator it(my_w.images);
	while (*it) {
		nPhysD *thisPhys=getPhys(*it);
		if (thisPhys) {
			(*it)->setData(0,0,nparent->physList.indexOf(thisPhys));
			(*it)->setData(1,0,QString::fromStdString(thisPhys->getShortName()));
			(*it)->setData(2,0,QString::fromStdString(thisPhys->getName()));
			(*it)->setData(3,0,QString::number(thisPhys->get_origin().x())+" "+QString::number(thisPhys->get_origin().y()));
			if (thisPhys->get_scale().x()==thisPhys->get_scale().y()) {
				(*it)->setData(4,0,QString::number(thisPhys->get_scale().x()));
			} else {
				(*it)->setData(4,0,QString::number(thisPhys->get_scale().x())+" "+QString::number(thisPhys->get_scale().y()));
			}
			if (thisPhys==my_phys) {
				(*it)->setSelected(true);
			} else {
				(*it)->setSelected(false);
			}
		} else {
			WARNING("This should not happend");
		}
		++it;
	}
	if (my_phys) {
		my_w.lineEdit->setText(QString::fromStdString(my_phys->getFromName()));
		my_w.lineEdit->setCursorPosition(0);
	} else {
		my_w.lineEdit->setText(tr("No image"));
	}
}


void
nWinList::physDel(nPhysD *my_phys) {
	QTreeWidgetItemIterator it(my_w.images);
	while (*it) {
		if (getPhys(*it)==my_phys) delete (*it);
		++it;
	}
	updatePad();
}

void nWinList::physAdd(nPhysD *my_phys) {
	QTreeWidgetItem *my_item = new QTreeWidgetItem(my_w.images);
	QString name=QString::fromStdString(my_phys->getName());
	my_item->setData(0,0,nparent->physList.indexOf(my_phys));
	my_item->setData(1,0,QString::fromStdString(my_phys->getShortName()));
	my_item->setData(2,0,QString::fromStdString(my_phys->getName()));
	my_item->setData(3,0,QString::number(my_phys->get_origin().x())+" "+QString::number(my_phys->get_origin().y()));
	if (my_phys->get_scale().x()==my_phys->get_scale().y()) {
		my_item->setData(4,0,QString::number(my_phys->get_scale().x()));
	} else {
		my_item->setData(4,0,QString::number(my_phys->get_scale().x())+" "+QString::number(my_phys->get_scale().y()));
	}
	my_item->setData((my_w.images->columnCount()-1),0,qVariantFromValue((void*) my_phys));
	my_w.images->sortItems(0,Qt::AscendingOrder);
	updatePad();
}









