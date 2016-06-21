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
#include "nBoxLineout.h"
    
nBoxLineout::nBoxLineout(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);
	
	// signals
	box =  new nRect(nparent);
	box->setParentPan(panName,1);
	box->setRect(QRectF(0,0,100,100));
	connect(my_w.actionRect, SIGNAL(triggered()), box, SLOT(togglePadella()));

	connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
	connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
	connect(my_w.actionSaveClipboard, SIGNAL(triggered()), this, SLOT(copy_clip()));
	connect(my_w.actionSaveTxt, SIGNAL(triggered()), this, SLOT(export_txt()));
	connect(my_w.actionSavePDF, SIGNAL(triggered()), this, SLOT(export_pdf()));


	decorate();
	loadDefaults();
	connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(updatePlot()));
	connect(box, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
	updatePlot();
}

void nBoxLineout::sceneChanged() {
	if (sender()==box) updatePlot();
}

void nBoxLineout::mouseAtWorld(QPointF p) {
	if (currentBuffer) {
        my_w.plot->setMousePosition(p.x(),p.y());
	}
}

void nBoxLineout::updatePlot() {
	if (currentBuffer) {
        QRect geomBox=box->getRect(currentBuffer);
        if (geomBox.isEmpty()) {
			my_w.statusBar->showMessage(tr("Attention: the region is outside the image!"),2000);
			return;
		}

		int dx=geomBox.width();
		int dy=geomBox.height();

        QVector<double> xd(dx);
        QVector<double> yd(dy);
        for (int j=0;j<dy;j++){
			for (int i=0;i<dx; i++) {
				double val=currentBuffer->point(i+geomBox.x(),j+geomBox.y(),0.0);
				xd[i]+=val;
				yd[j]+=val;
			}
		}

		transform(xd.begin(), xd.end(), xd.begin(),bind2nd(std::divides<double>(), dy));
		transform(yd.begin(), yd.end(), yd.begin(),bind2nd(std::divides<double>(), dx));
		
        QVector <double> xdata(dx);
        QVector <double> ydata(dy);

		vec2f orig=currentBuffer->get_origin();
		vec2f scal=currentBuffer->get_scale();
		
        for (int i=0;i<dx;i++) xdata[i]=(geomBox.x()+i-orig.x())*scal.x();
        for (int j=0;j<dy;j++) ydata[j]=(geomBox.y()+j-orig.y())*scal.y();

        my_w.plot->graph(0)->setData(xdata,xd);
        my_w.plot->graph(1)->setData(ydata,yd);

        my_w.plot->rescaleAxes();
        my_w.plot->replot();
    }

}

void nBoxLineout::export_data(QTextStream &out) {
    out << "# " << panName << " " << QString::fromStdString(currentBuffer->getName()) <<endl;
    for (int g=0; g<my_w.plot->graphCount(); g++) {
        out << "# " << (g==0?"Horizontal":"Vertical") << endl;
        const QCPDataMap *dataMap = my_w.plot->graph(g)->data();
        QMap<double, QCPData>::const_iterator i = dataMap->constBegin();
        while (i != dataMap->constEnd()) {
            out << i.value().key << " " << i.value().value << endl;
            ++i;
        }
        out << endl << endl;
    }
}

void nBoxLineout::copy_clipboard() {
	if (currentBuffer) {
        QString point_table;
        QTextStream out(&point_table);
        export_data(out);
        QApplication::clipboard()->setText(point_table);
        my_w.statusBar->showMessage(tr("Points copied to clipboard"),2000);
	}
}

void nBoxLineout::export_txt() {
	if (currentBuffer) {
		QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save data in text"),property("fileTxt").toString(),tr("Text files (*.txt *.csv);;Any files (*)"));
		if (!fnametmp.isEmpty()) {
			setProperty("fileTxt", fnametmp);
			QFile t(fnametmp);
			t.open(QIODevice::WriteOnly| QIODevice::Text);
			QTextStream out(&t);
            export_data(out);
            t.close();
            my_w.statusBar->showMessage(tr("Export :")+fnametmp,2000);
        }
	}
}

void
nBoxLineout::export_pdf() {
	QString fout;
    QString fnametmp = QFileDialog::getSaveFileName(this,tr("Save Drawing"),property("fileExport").toString(),"Vector files (*.pdf)");
	if (!fnametmp.isEmpty()) {
		setProperty("fileExport", fnametmp);
        my_w.plot->savePdf(fnametmp,true,0,0,"Neutrino", panName);
	}

}

