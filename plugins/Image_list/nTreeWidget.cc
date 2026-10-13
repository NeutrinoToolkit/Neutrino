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

// the Image_list pan this tree belongs to
Image_list* nTreeWidget::imageList() {
    QObject *p=parent();
    while (p && !qobject_cast<Image_list *>(p)) p=p->parent();
    return qobject_cast<Image_list *>(p);
}

// header clicks sort through customSort (which also reorders neutrino's list): Qt's own sorting
// would only reorder the view. Call once, after setupUi.
void nTreeWidget::setupSorting() {
    setSortingEnabled(false);
    header()->setSortIndicatorShown(true);
    header()->setSectionsClickable(true);
    connect(header(), &QHeaderView::sectionClicked, this, &nTreeWidget::customSort, Qt::UniqueConnection);
}

void nTreeWidget::mousePressEvent(QMouseEvent *e) {
    dragitems.clear();
    foreach (QTreeWidgetItem * item, selectedItems()) {
        dragitems << item;
    }
    dragposition=e->pos();
    dragtime.start();

    QTreeWidget::mousePressEvent(e);
}
void nTreeWidget::mouseMoveEvent(QMouseEvent *e) {
    if (dragitems.size() && (dragposition - e->pos()).manhattanLength()>=QApplication::startDragDistance() && dragtime.elapsed() > qApp->startDragTime() ) {
        Image_list* imParent=imageList();
        if(imParent) {
            QByteArray dragPhysPointers;
            QList<QUrl> lista;
            foreach (QTreeWidgetItem * item, dragitems) {
                nPhysD *my_phys=imParent->getPhys(item);
                if (imParent->nPhysExists(my_phys)) {
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
    Image_list* imParent=imageList();
    if (imParent && imParent->nparent) {
        imParent->nparent->dropEvent(e);
    }
    e->acceptProposedAction();
    dragitems.clear();
}

void nTreeWidget::customSort(int column)
{
    QTreeWidget::sortItems(column, header()->sortIndicatorOrder());
    Image_list* imParent=imageList();
    if (imParent && imParent->nparent) {
        QList<nPhysD*> myPhysList;
        for (int i=0; i < topLevelItemCount(); ++i) {
            myPhysList << imParent->getPhys(topLevelItem(i));
        }
        // refused if the list is not a permutation of neutrino's one: then restore the numbering order
        if (!imParent->nparent->reorderPhys(myPhysList)) {
            qWarning() << "Image list out of sync with neutrino: not reordering";
            imParent->renumber();
            QTreeWidget::sortItems(0, Qt::AscendingOrder);
        }
    }
}
