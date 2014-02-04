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

#include <qwt_plot_zoomer.h>
#include <qwt_plot_panner.h>
#include <qwt_plot_renderer.h>


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

	my_w.plot->setAxisTitle(QwtPlot::xBottom, tr("X (red)"));
	my_w.plot->setAxisTitle(QwtPlot::yRight, tr("X value (red)"));
	my_w.plot->setAxisTitle(QwtPlot::yLeft, tr("Y (blue)"));
	my_w.plot->setAxisTitle(QwtPlot::xTop, tr("Y value (blue)"));
	my_w.plot->enableAxis(QwtPlot::xTop);
	my_w.plot->enableAxis(QwtPlot::yRight);
	my_w.plot->enableAxis(QwtPlot::xBottom);
	my_w.plot->enableAxis(QwtPlot::yLeft);
	(qobject_cast<QFrame*> (my_w.plot->canvas()))->setLineWidth(0);

	xCut.setPen(QPen(Qt::red,1));
	yCut.setPen(QPen(Qt::blue,1));

	xCut.setXAxis(QwtPlot::xBottom);
	xCut.setYAxis(QwtPlot::yRight);
	yCut.setXAxis(QwtPlot::xTop);
	yCut.setYAxis(QwtPlot::yLeft);

	xCut.attach(my_w.plot);
	yCut.attach(my_w.plot);


	QPen marker_pen;
	marker_pen.setColor(QColor(255,0,0));
	xMarker.setLinePen(marker_pen);
	yMarker.setLinePen(marker_pen);


	xMarker.setLineStyle(QwtPlotMarker::VLine);
	xMarker.attach(my_w.plot);

	yMarker.setLineStyle(QwtPlotMarker::HLine);
	yMarker.attach(my_w.plot);

	marker_pen.setColor(QColor(0,0,255));
	rxMarker.setLinePen(marker_pen);
	ryMarker.setLinePen(marker_pen);

	rxMarker.setLineStyle(QwtPlotMarker::VLine);
	rxMarker.attach(my_w.plot);

	ryMarker.setLineStyle(QwtPlotMarker::HLine);
	ryMarker.attach(my_w.plot);

	rxMarker.setXValue(0);
	ryMarker.setYValue(0);

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
		QPen marker_pen;
		marker_pen.setColor(nparent->my_mouse.color);
		xMarker.setLinePen(marker_pen);
		yMarker.setLinePen(marker_pen);
		marker_pen.setColor(nparent->my_tics.rulerColor);
		rxMarker.setLinePen(marker_pen);
		ryMarker.setLinePen(marker_pen);
		
		xMarker.setXValue(p.x());
		yMarker.setYValue(p.y());
				
		rxMarker.setVisible(nparent->my_tics.rulerVisible);
		ryMarker.setVisible(nparent->my_tics.rulerVisible);
		
		my_w.plot->replot();
	}
}

void nBoxLineout::updatePlot() {
	if (currentBuffer) {
		QRect geom2=box->getRect().intersected(QRect(0,0,currentBuffer->getW(),currentBuffer->getH()));
		if (geom2.isEmpty()) {
			my_w.statusBar->showMessage(tr("Attention: the region is outside the image!"),2000);
			return;
		}

		int dx=geom2.width();
		int dy=geom2.height();

		double xd[dx];
		double yd[dy];
		for (int j=0;j<dy;j++) yd[j]=0.0;
		for (int i=0;i<dx;i++) xd[i]=0.0;

		for (int j=0;j<dy;j++){
			for (int i=0;i<dx; i++) {
				double val=currentBuffer->point(i+geom2.x(),j+geom2.y(),0.0);
				xd[i]+=val;
				yd[j]+=val;
			}
		}

		QVector <QPointF> xdata(dx);
		QVector <QPointF> ydata(dy);
		
		for (int i=0;i<dx;i++) xdata[i]=QPointF((geom2.x()+i-currentBuffer->get_origin().x())*currentBuffer->get_scale().x(),xd[i]/dy);
		for (int j=0;j<dy;j++) ydata[j]=QPointF(yd[j]/dx,(geom2.y()+j-currentBuffer->get_origin().y())*currentBuffer->get_scale().y());

		xCut.setSamples(xdata);
		yCut.setSamples(ydata);

		my_w.plot->setAxisScale(xCut.xAxis(),xCut.minXValue(),xCut.maxXValue(),0);
		my_w.plot->setAxisScale(xCut.yAxis(),xCut.minYValue(),xCut.maxYValue(),0);
		my_w.plot->setAxisScale(yCut.xAxis(),yCut.minXValue(),yCut.maxXValue(),0);
		my_w.plot->setAxisScale(yCut.yAxis(),yCut.maxYValue(),yCut.minYValue(),0);

		my_w.plot->replot();
	}

}

void nBoxLineout::copy_clip() {
	if (currentBuffer) {
		QClipboard *clipboard = QApplication::clipboard();
		QString point_table=QString::fromStdString(currentBuffer->getName())+"\n";
		point_table.append("# Horizontal : "+QString::number(xCut.data()->size()) +"\n");
		for (unsigned int i=0;i<xCut.data()->size();i++) {
			point_table.append(QString("%1\t%2\n").arg(xCut.data()->sample(i).x()).arg(xCut.data()->sample(i).y()));
		}
		point_table.append("# Vertical : "+QString::number(yCut.data()->size()) +"\n");
		for (unsigned int i=0;i<yCut.data()->size();i++) {
			point_table.append(QString("%1\t%2\n").arg(yCut.data()->sample(i).x()).arg(yCut.data()->sample(i).y()));
		}
		clipboard->setText(point_table);
		showMessage(tr("Points copied to clipboard"));
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
			out << "# Box Lineout " << QString::fromStdString(currentBuffer->getName()) <<endl;
			out << "# Horizontal : " << xCut.data()->size() <<endl;
			for (unsigned int i=0;i<xCut.data()->size();i++) {
				out << xCut.data()->sample(i).x() << " " << xCut.data()->sample(i).y() << endl;
			}
			out << endl  << endl << "# Vertical : " << yCut.data()->size() <<endl;
			for (unsigned int j=0;j<yCut.data()->size();j++) {
				out << yCut.data()->sample(j).y() << " " << yCut.data()->sample(j).x() << endl;
			}
			t.close();
			showMessage(tr("Export in file:")+fnametmp,2000);
		}
	}
}

void
nBoxLineout::export_pdf() {
	QString fout;
	QString fnametmp = QFileDialog::getSaveFileName(this,tr("Save Drawing"),property("fileExport").toString(),"Vector files (*.pdf,*.svg)");
	if (!fnametmp.isEmpty()) {
		setProperty("fileExport", fnametmp);
		QwtPlotRenderer renderer;

		// flags to make the document look like the widge
		renderer.setDiscardFlag(QwtPlotRenderer::DiscardBackground, false);
//		renderer.setLayoutFlag(QwtPlotRenderer::KeepFrames, true);

		renderer.renderDocument(my_w.plot, fnametmp, QFileInfo(fnametmp).suffix(), QSizeF(150, 100), 85);

	}

}

