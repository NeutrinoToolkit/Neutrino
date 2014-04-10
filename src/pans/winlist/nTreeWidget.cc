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
#include "nWinList.h"
#include "neutrino.h"

using namespace std;

nTreeWidget::nTreeWidget(QWidget * ){
	dragitem=NULL;
};

nTreeWidget::~nTreeWidget(){
	if (dragitem) delete dragitem;
};

void nTreeWidget::mousePressEvent(QMouseEvent *e) {
	dragposition=e->pos();
	dragitem=itemAt(e->pos());
	dragtime.start();
	QTreeWidget::mousePressEvent(e);
}

void nTreeWidget::mouseMoveEvent(QMouseEvent *e) {
	if (dragitem && (dragposition - e->pos()).manhattanLength()>=QApplication::startDragDistance() && dragtime.elapsed() > qApp->startDragTime() ) {
		QMimeData *mymimeData=new QMimeData;
		nPhysD *my_phys=(nPhysD*) (dragitem->data(columnCount()-1,Qt::DisplayRole).value<void*>());
		DEBUG((void*)my_phys);
		QList<QUrl> lista;
		lista << QUrl(QString::fromUtf8(my_phys->getName().c_str()));
		if (lista.size()) mymimeData->setUrls(lista);
		QByteArray physPointer;
		physPointer.append(QString::number((long) my_phys));
		mymimeData->setData(QString("data/neutrino"), physPointer);
		QDrag *drag = new QDrag(this);
		drag->setMimeData(mymimeData);
		drag->exec();
	}
}

void nTreeWidget::mouseReleaseEvent(QMouseEvent *e) {
	dragitem=NULL;
	if (e->modifiers() == Qt::NoModifier) {
		QTreeWidgetItem *item=itemAt(e->pos());
		if (item) {
			nPhysD *phys=(nPhysD*) (item->data(columnCount()-1,Qt::DisplayRole).value<void*>());
			nparent->showPhys(phys);
		}
	}
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
	nparent->dropEvent(e);
	e->acceptProposedAction();
	dragitem=NULL;
}

void nTreeWidget::keyPressEvent(QKeyEvent *e){
	foreach (QTreeWidgetItem * item, selectedItems()) {
		nPhysD *phys=(nPhysD*) (item->data(columnCount()-1,0).value<void*>());
		switch (e->key()) {
			case Qt::Key_Return:
				nparent->showPhys(phys);
				break;
			case Qt::Key_Backspace:
			case Qt::Key_Delete:
				nparent->removePhys(phys);
				break;
			case Qt::Key_Up:
				nparent->actionPrevBuffer();
				break;
			case Qt::Key_Down:
				nparent->actionNextBuffer();
				break;
			default:
				QTreeWidget::keyPressEvent(e);
				break;
		}
	}
	e->accept();
}

