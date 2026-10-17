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
#include "Mouse_info.h"


Mouse_info::Mouse_info (neutrino *parent) : nGenericPan(parent)
{

    setupUi(this);

    // connect(removeRowB, SIGNAL(released()),this, SLOT(remove_point()));
    // connect(copyPointsB, SIGNAL(released()),this, SLOT(copyPoints()));
    // connect(exportTxtB, SIGNAL(released()),this, SLOT(export_txt()));
    connect(removeRowB, SIGNAL(clicked()),this, SLOT(remove_point()));
    connect(copyPointsB, SIGNAL(clicked()),this, SLOT(copyPoints()));
    connect(exportTxtB, SIGNAL(clicked()),this, SLOT(export_txt()));

    connect(nparent, SIGNAL(mouseAtMatrix(QPointF)), this, SLOT(setMouse(QPointF)));
    connect(rx, SIGNAL(editingFinished()), this, SLOT(updateOrigin()));
    connect(ry, SIGNAL(editingFinished()), this, SLOT(updateOrigin()));
    connect(sc_x, SIGNAL(editingFinished()), this, SLOT(updateScale()));
    connect(sc_y, SIGNAL(editingFinished()), this, SLOT(updateScale()));
    connect(unit_x, SIGNAL(editingFinished()), this, SLOT(updateUnits()));
    connect(unit_y, SIGNAL(editingFinished()), this, SLOT(updateUnits()));

    // origin, scale and units are applied when the editing is finished (also textChanged was connected,
    // in updateLabels: every keystroke changed the image); they are connected only here, once
    connect(nparent->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(addPoint(QPointF)));

    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updateLabels()));

    mouse=nparent->my_view->my_mouse.pos();
	updateLabels();
    show(true);
}

// table item showing the number in the user locale, the value itself in Qt::UserRole
static QTableWidgetItem *numberItem(double val) {
    QTableWidgetItem *item= new QTableWidgetItem(QLocale().toString(val));
    item->setData(Qt::UserRole, val);
    item->setTextAlignment(Qt::AlignCenter);
    return item;
}

void Mouse_info::addPoint(QPointF position) {
    if (getpoints->isChecked()) {
        int pos=points->rowCount();
        points->insertRow(pos);
        // the numbers are kept in Qt::UserRole: the export writes them with the C locale
        // QTableWidgetItem *xitem= new QTableWidgetItem(QLocale().toString(position.x()));
        // QTableWidgetItem *yitem= new QTableWidgetItem(QLocale().toString(position.y()));
        // QTableWidgetItem *x2item= new QTableWidgetItem(px->text());
        // QTableWidgetItem *y2item= new QTableWidgetItem(py->text());
        QPointF real=position;
        if (currentBuffer) {
            real=QPointF((position.x()-currentBuffer->get_origin().x())*currentBuffer->get_scale().x(),
                         (position.y()-currentBuffer->get_origin().y())*currentBuffer->get_scale().y());
        }
        QTableWidgetItem *xitem= numberItem(position.x());
        QTableWidgetItem *yitem= numberItem(position.y());
        QTableWidgetItem *x2item= numberItem(real.x());
        QTableWidgetItem *y2item= numberItem(real.y());
        // xitem->setTextAlignment(Qt::AlignHCenter & Qt::AlignVCenter);
        // yitem->setTextAlignment(Qt::AlignHCenter & Qt::AlignVCenter);
        // x2item->setTextAlignment(Qt::AlignHCenter & Qt::AlignVCenter);
        // y2item->setTextAlignment(Qt::AlignHCenter & Qt::AlignVCenter);
        // (& gave 0: not centred; numberItem centres)
		xitem->setFlags(Qt::ItemIsSelectable|Qt::ItemIsEnabled);
		yitem->setFlags(Qt::ItemIsSelectable|Qt::ItemIsEnabled);
		x2item->setFlags(Qt::ItemIsSelectable|Qt::ItemIsEnabled);
		y2item->setFlags(Qt::ItemIsSelectable|Qt::ItemIsEnabled);
        points->setItem(pos, 0, xitem);
        points->setItem(pos, 1, yitem);
        points->setItem(pos, 2, x2item);
        points->setItem(pos, 3, y2item);
		// QTableWidgetItem *zitem= new QTableWidgetItem();
		// if (currentBuffer) {
		// 	zitem->setText(QLocale().toString(currentBuffer->point(position.x(),position.y())));
		// } else {
		// 	zitem->setText(QString(""));
		// }
		// point() takes unsigned: the pixel as int (outside the image -> NaN)
		QTableWidgetItem *zitem= currentBuffer ?
		    numberItem(currentBuffer->point(vec2i((int)floor(position.x()),(int)floor(position.y())))) :
		    new QTableWidgetItem(QString(""));

        // zitem->setTextAlignment(Qt::AlignHCenter & Qt::AlignVCenter);
        zitem->setTextAlignment(Qt::AlignCenter);
		zitem->setFlags(Qt::ItemIsSelectable|Qt::ItemIsEnabled);
        points->setItem(pos, 4, zitem);
        points->resizeRowToContents(pos);
		statusBar()->showMessage(tr("Added point ")+QLocale().toString(pos+1),2000);
	}
}

void Mouse_info::updateOrigin() {
	if (currentBuffer) {
		bool ok1, ok2;
        double valx=QLocale().toDouble(rx->text(),&ok1);
        double valy=QLocale().toDouble(ry->text(),&ok2);
		// if (ok1&&ok2) {
		// 	currentBuffer->set_origin(vec2f(valx,valy));
		// }
		// tell the other pans (Image_list, lineouts...), not only the view
		if (ok1&&ok2 && currentBuffer->get_origin()!=vec2f(valx,valy)) {
			currentBuffer->set_origin(vec2f(valx,valy));
			nparent->emitPhysChanged(currentBuffer);
		}
	}
    nparent->my_view->update();
}

void Mouse_info::updateScale() {
	if (currentBuffer) {
		bool ok1, ok2;
        double valx=QLocale().toDouble(sc_x->text(),&ok1);
        double valy=QLocale().toDouble(sc_y->text(),&ok2);
		// if (ok1&&ok2) {
		// 	currentBuffer->set_scale(vec2f(valx,valy));
		// }
		if (ok1&&ok2 && currentBuffer->get_scale()!=vec2f(valx,valy)) {
			currentBuffer->set_scale(vec2f(valx,valy));
			nparent->emitPhysChanged(currentBuffer);
		}
	}
    nparent->my_view->update();
}

void Mouse_info::updateUnits() {
    if (currentBuffer) {
        // currentBuffer->prop["unitsX"] = unit_x->text().toStdString();
        // currentBuffer->prop["unitsY"] = unit_y->text().toStdString();
        if (currentBuffer->prop["unitsX"].get_str()!=unit_x->text().toStdString() ||
            currentBuffer->prop["unitsY"].get_str()!=unit_y->text().toStdString()) {
            currentBuffer->prop["unitsX"] = unit_x->text().toStdString();
            currentBuffer->prop["unitsY"] = unit_y->text().toStdString();
            nparent->emitPhysChanged(currentBuffer);
        }
    }
    nparent->my_view->update();
}


void Mouse_info::setMouse(QPointF pos) {
	mouse=pos;
	updateLabels();
}

void Mouse_info::updateLabels() {
    ix->setText(QLocale().toString(mouse.x()));
    iy->setText(QLocale().toString(mouse.y()));
		
	QPointF dist=QPointF(mouse);
	QPointF resolution=QPointF(1,1);
	if (currentBuffer) {
		dist=mouse-QPointF(currentBuffer->get_origin().x(),currentBuffer->get_origin().y());
		resolution=QPointF(currentBuffer->get_scale().x(),currentBuffer->get_scale().y());
        // disconnect(rx, SIGNAL(textChanged(QString)), this, SLOT(updateOrigin()));
        // disconnect(ry, SIGNAL(textChanged(QString)), this, SLOT(updateOrigin()));
        // disconnect(sc_x, SIGNAL(textChanged(QString)), this, SLOT(updateScale()));
        // disconnect(sc_y, SIGNAL(textChanged(QString)), this, SLOT(updateScale()));
        // connect(unit_x, SIGNAL(editingFinished()), this, SLOT(updateUnits()));
        // connect(unit_y, SIGNAL(editingFinished()), this, SLOT(updateUnits()));
        // rx->setText(QLocale().toString(currentBuffer->get_origin().x()));
        // ry->setText(QLocale().toString(currentBuffer->get_origin().y()));
        // sc_x->setText(QLocale().toString(currentBuffer->get_scale().x()));
        // sc_y->setText(QLocale().toString(currentBuffer->get_scale().y()));
        // unit_x->setText(QString::fromStdString(currentBuffer->prop["unitsX"].get_str()));
        // unit_y->setText(QString::fromStdString(currentBuffer->prop["unitsY"].get_str()));
        // connect(rx, SIGNAL(textChanged(QString)), this, SLOT(updateOrigin()));
        // connect(ry, SIGNAL(textChanged(QString)), this, SLOT(updateOrigin()));
        // connect(sc_x, SIGNAL(textChanged(QString)), this, SLOT(updateScale()));
        // connect(sc_y, SIGNAL(textChanged(QString)), this, SLOT(updateScale()));
        // connect(unit_x, SIGNAL(editingFinished()), this, SLOT(updateUnits()));
        // connect(unit_y, SIGNAL(editingFinished()), this, SLOT(updateUnits()));
        // (the "disconnect" of the units was a connect: 2 more connections at every mouse move)
        // only editingFinished is connected (constructor) and setText does not emit it: no disconnect needed;
        // a field being edited is not overwritten
        if (!rx->hasFocus()) rx->setText(QLocale().toString(currentBuffer->get_origin().x()));
        if (!ry->hasFocus()) ry->setText(QLocale().toString(currentBuffer->get_origin().y()));
        if (!sc_x->hasFocus()) sc_x->setText(QLocale().toString(currentBuffer->get_scale().x()));
        if (!sc_y->hasFocus()) sc_y->setText(QLocale().toString(currentBuffer->get_scale().y()));
        if (!unit_x->hasFocus()) unit_x->setText(QString::fromStdString(currentBuffer->prop["unitsX"].get_str()));
        if (!unit_y->hasFocus()) unit_y->setText(QString::fromStdString(currentBuffer->prop["unitsY"].get_str()));
    }

	QPointF p=QPointF(resolution.x()*dist.x(),resolution.y()*dist.y());

    px->setText(QLocale().toString(p.x()));
    py->setText(QLocale().toString(p.y()));
    dx_dy->setText(QLocale().toString(p.x()/p.y()));
    dy_dx->setText(QLocale().toString(p.y()/p.x()));
    atan_dx_dy->setText(QLocale().toString(180.0/M_PI*atan2(p.x(),p.y())));
    atan_dy_dx->setText(QLocale().toString(180.0/M_PI*atan2(p.y(),p.x())));
    dist_px->setText(QLocale().toString(qSqrt(dist.x()*dist.x()+dist.y()*dist.y())));
    dist_real->setText(QLocale().toString(qSqrt(p.x()*p.x()+p.y()*p.y())));
}

QString 
Mouse_info::getPointText() {
	return getPointText(false);
}

// cLocale (file export): header and numbers with the C locale; otherwise (clipboard) as shown, for spreadsheets
QString
Mouse_info::getPointText(bool cLocale) {
	QString retText;
	// for (int i=0; i<points->rowCount(); i++) {
	//     for (int j=0; j<points->columnCount();j++) {
	//         retText += points->item(i, j)->text() + "\t";
	// 	}
	// 	retText += "\n";
	// }
	if (cLocale) {
		QStringList header;
		for (int j=0; j<points->columnCount();j++) {
			QTableWidgetItem *hitem=points->horizontalHeaderItem(j);
			header << (hitem ? hitem->text() : QString::number(j));
		}
		retText += "# " + header.join("\t") + "\n";
	}
	QLocale cloc(QLocale::C);
	for (int i=0; i<points->rowCount(); i++) {
		QStringList row;
		for (int j=0; j<points->columnCount();j++) {
			QTableWidgetItem *item=points->item(i, j);
			if (!item) {
				row << "";
			} else if (cLocale && item->data(Qt::UserRole).isValid()) {
				row << cloc.toString(item->data(Qt::UserRole).toDouble(),'g',10);
			} else {
				row << item->text();
			}
		}
		retText += row.join("\t") + "\n"; // no trailing tab
	}
	return retText;
}

void Mouse_info::copyPoints() {
	QClipboard *clipboard = QApplication::clipboard();
	// clipboard->setText(getPointText());
	clipboard->setText(getPointText(false));
	statusBar()->showMessage(tr("Point values copied"),2000);
}

void
Mouse_info::export_txt() {
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save data"),property("NeuSave-fileTxt").toString(),tr("Text files (*.txt *.csv);;Any file (*)"));
	if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileTxt",fnametmp);
		QFile t(fnametmp);
		// t.open(QIODevice::WriteOnly| QIODevice::Text);
		if (!t.open(QIODevice::WriteOnly| QIODevice::Text)) {
			statusBar()->showMessage(tr("Cannot write file ")+fnametmp+": "+t.errorString(),5000);
			return;
		}
		QTextStream out(&t);
		// out << getPointText();
		out << getPointText(true);
		t.close();
		statusBar()->showMessage(tr("Export in file:")+fnametmp,2000);
	} else {
		statusBar()->showMessage(tr("Export canceled"),2000);
	}
}

void
Mouse_info::remove_point() {
	QString removedrows;
    // foreach (QTableWidgetSelectionRange r, points->selectedRanges()) {
	// 	for (int i=r.topRow(); i <=r.bottomRow(); i++) {
    //         points->removeRow(r.topRow());
	// 		removedrows+=QString(" ")+QLocale().toString(i+1);
	// 	}
	// }
	// (removing a range shifted the rows of the next ones): all the rows, removed from the bottom
	QList<int> rows;
	foreach (QTableWidgetSelectionRange r, points->selectedRanges()) {
		for (int i=r.topRow(); i <=r.bottomRow(); i++) {
			if (!rows.contains(i)) rows << i;
		}
	}
	std::sort(rows.begin(), rows.end());
	for (int i : rows) {
		removedrows+=QString(" ")+QLocale().toString(i+1);
	}
	std::sort(rows.begin(), rows.end(), std::greater<int>());
	for (int i : rows) {
		points->removeRow(i);
	}
	if (removedrows.isEmpty()) {
		statusBar()->showMessage(tr("No row was selected"));
	} else {
		statusBar()->showMessage(tr("Removed Rows:")+removedrows,2000);
	}
}

