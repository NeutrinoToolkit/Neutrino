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
#include "nTreeWidget.h"
#include "Image_list.h"
#include "neutrino.h"

void nTreeWidget::mousePressEvent(QMouseEvent *e) {
    dragitems.clear();
    foreach (QTreeWidgetItem * item, selectedItems()) {
        dragitems << item;
    }
    dragposition=e->pos();
    dragtime.start();

    setSortingEnabled(false);
    header()->setSortIndicatorShown(true);
    header()->setSectionsClickable(true);
    connect(header(), SIGNAL(sectionClicked(int)), this, SLOT(customSort(int)));
    //    customSortByColumn(header()->sortIndicatorSection());

    QTreeWidget::mousePressEvent(e);
}

void nTreeWidget::mouseMoveEvent(QMouseEvent *e) {
    if (dragitems.size() && (dragposition - e->pos()).manhattanLength()>=QApplication::startDragDistance() && dragtime.elapsed() > qApp->startDragTime() ) {
        Image_list* imParent=qobject_cast<Image_list *> (parent()->parent());
        if(imParent) {
            QByteArray dragPhysPointers;
            QList<QUrl> lista;
            foreach (QTreeWidgetItem * item, dragitems) {
                nPhysD *my_phys=imParent->getPhys(item);
                if (my_phys) {
                    dragPhysPointers+=QByteArray::number((qlonglong) my_phys)+ " ";
                    lista << QUrl(QString::fromUtf8(my_phys->getName().c_str()));
                }
            }
            if (lista.size()) {
                QMimeData *mymimeData=new QMimeData;
                mymimeData->setUrls(lista);
                mymimeData->setData(QString("data/neutrino"), dragPhysPointers);
                QDrag *drag = new QDrag(this);
                drag->setMimeData(mymimeData);
                drag->exec();
            }
            DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>> FOUND PARENT");
        }
    }
}

void nTreeWidget::mouseReleaseEvent(QMouseEvent *e) {
    dragitems.clear();
    QTreeWidget::mouseReleaseEvent(e);
}

// Drag and Drop
void nTreeWidget::dragEnterEvent(QDragEnterEvent *e)
{
    e->acceptProposedAction();
}

void nTreeWidget::dragMoveEvent(QDragMoveEvent *e)
{
    e->acceptProposedAction();
}

void nTreeWidget::dropEvent(QDropEvent *e) {
    neutrino* nparent=qobject_cast<neutrino *> (parent()->parent()->parent());
    if (nparent) {
        nparent->dropEvent(e);
    }
    e->acceptProposedAction();
    dragitems.clear();
}

void nTreeWidget::customSort(int column)
{
    DEBUG("CUSTOM SORT");
    Qt::SortOrder order = header()->sortIndicatorOrder();
    QTreeWidget::sortItems(column, order);
    QApplication::processEvents();
    Image_list* imParent=qobject_cast<Image_list *> (parent()->parent());
    if (imParent) {
        QList<nPhysD*> myPhysList;
        for (int i=0; i < topLevelItemCount(); ++i) {
            QTreeWidgetItem *thisItem=topLevelItem(i);
            for (auto &item: imParent->itemsMap) {
                if (item.second == thisItem) {
                    myPhysList << item.first;
                }
            }
        }
        if (myPhysList.size() == imParent->nparent->physList.size()) {
            DEBUG("--------------------------------------------------------------------------");
            for (int i=0; i < myPhysList.size(); ++i) {
                DEBUG(i << " " << myPhysList.at(i) << " " << imParent->nparent->physList.at(i));
            }
            DEBUG("--------------------------------------------------------------------------");
        }
        imParent->nparent->physList=myPhysList;

    }
}
