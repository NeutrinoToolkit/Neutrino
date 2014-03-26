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
#include "nFindPeaks.h"

#include <qwt_plot_zoomer.h>
#include <qwt_plot_panner.h>
#include <qwt_plot_renderer.h>
#include <qwt_symbol.h>

#include <gsl/gsl_fit.h>

nFindPeaks::nFindPeaks(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);

	// signals
	box =  new nRect(nparent);
	box->setParentPan(panName,1);
	box->setRect(QRectF(0,0,100,100));


	connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
	connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
	connect(my_w.actionSaveClipboard, SIGNAL(triggered()), this, SLOT(copy_clip()));
	connect(my_w.actionSaveTxt, SIGNAL(triggered()), this, SLOT(export_txt()));
	connect(my_w.actionSavePDF, SIGNAL(triggered()), this, SLOT(export_pdf()));

	connect(my_w.setOrigin, SIGNAL(pressed()), this, SLOT(setOrigin()));
	connect(my_w.setScale, SIGNAL(pressed()), this, SLOT(setScale()));

	connect(my_w.actionRect, SIGNAL(triggered()), box, SLOT(togglePadella()));

	my_w.toolBar->addWidget(my_w.direction);
	my_w.toolBar->addWidget(my_w.param);

	my_w.plot->setAxisTitle(QwtPlot::xBottom, tr("Position [px]"));
	my_w.plot->setAxisTitle(QwtPlot::yLeft, tr("Value"));
	my_w.plot->enableAxis(QwtPlot::xBottom);
	my_w.plot->enableAxis(QwtPlot::yLeft);
	(qobject_cast<QFrame*> (my_w.plot->canvas()))->setLineWidth(0);
	
	lineout.setPen(QPen(Qt::red,1));

	lineout.setXAxis(QwtPlot::xBottom);
	lineout.setYAxis(QwtPlot::yLeft);

	lineout.attach(my_w.plot);

	QPen marker_pen;
	marker_pen.setColor(QColor(255,0,0));
	xMarker.setLinePen(marker_pen);


	xMarker.setLineStyle(QwtPlotMarker::VLine);
	xMarker.attach(my_w.plot);

	marker_pen.setColor(QColor(0,0,255));
	rxMarker.setLinePen(marker_pen);

	rxMarker.setLineStyle(QwtPlotMarker::VLine);
	rxMarker.attach(my_w.plot);

	rxMarker.setXValue(0);
	
	decorate();
	loadDefaults();
	connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(updatePlot()));
	connect(box, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
 	connect(my_w.direction, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlot()));
 	connect(my_w.param, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
	updatePlot();
}


void nFindPeaks::setOrigin() {
	if (currentBuffer) {
		bool ok=true;
		double originOffset=0.0;
		if (!my_w.originOffset->text().isEmpty()) originOffset=my_w.originOffset->text().toDouble(&ok);
		if (ok) {
			double origin=my_w.origin->text().toDouble(&ok)-originOffset;
			if (ok) {
				if (my_w.direction->currentIndex()==0) {
					currentBuffer->set_origin(origin,currentBuffer->get_origin().y());
				} else {
					currentBuffer->set_origin(currentBuffer->get_origin().x(),origin);
				}
				nparent->my_tics.update();
			}
		}
	}
}

void nFindPeaks::setScale() {
	if (currentBuffer) {
		bool ok=true;
		double scaleMult=1.0;
		if (!my_w.scaleOffset->text().isEmpty()) scaleMult=my_w.scaleOffset->text().toDouble(&ok);
		if (ok) {
			double scale=scaleMult/my_w.scale->text().toDouble(&ok);
			if (ok) {
				if (my_w.direction->currentIndex()==0) {
					currentBuffer->set_scale(scale,currentBuffer->get_scale().y());
				} else {
					currentBuffer->set_scale(currentBuffer->get_scale().x(),scale);
				}
				nparent->my_tics.update();
			}
		}
	}
}

void nFindPeaks::sceneChanged() {
	if (sender()==box) updatePlot();
}

void nFindPeaks::mouseAtMatrix(QPointF p) {
	if (currentBuffer) {
		QPen marker_pen;
		marker_pen.setColor(nparent->my_mouse.color);
		xMarker.setLinePen(marker_pen);
		marker_pen.setColor(nparent->my_tics.rulerColor);
		rxMarker.setLinePen(marker_pen);
		
		xMarker.setXValue(my_w.direction->currentIndex()==0?p.x():p.y());
				
		rxMarker.setVisible(nparent->my_tics.rulerVisible);
		
		my_w.plot->replot();
	}
}

void nFindPeaks::updatePlot() {
	if (currentBuffer && isVisible()) {
		QRect geom2=box->getRect().intersected(QRect(0,0,currentBuffer->getW(),currentBuffer->getH()));
		if (geom2.isEmpty()) {
			my_w.statusBar->showMessage(tr("Attention: the region is outside the image!"),2000);
			return;
		}
		
		int dx=geom2.width();
		int dy=geom2.height();

		double *xd=new double[dx];
		double *yd=new double[dy];
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
		
		for (int i=0;i<dx;i++) xdata[i]=QPointF(i+geom2.x(),xd[i]/dy);
		for (int j=0;j<dy;j++) ydata[j]=QPointF(j+geom2.y(),yd[j]/dx);

		
		lineout.setSamples((my_w.direction->currentIndex()==0)?xdata:ydata);
	
		int sizeCut=lineout.dataSize();
		double *myData=new double[sizeCut];
		
		for (int i=0;i<sizeCut;i++) {
			myData[i]=lineout.sample(i).y()/sizeCut;
		}
		
		fftw_complex *myDataC=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(sizeCut/2+1));
		fftw_complex *myDataC2=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(sizeCut/2+1));
		fftw_plan planR2C=fftw_plan_dft_r2c_1d(sizeCut, myData, myDataC, FFTW_ESTIMATE);
		fftw_plan planC2R=fftw_plan_dft_c2r_1d(sizeCut, myDataC, myData, FFTW_ESTIMATE);
		fftw_plan planC2R2=fftw_plan_dft_c2r_1d(sizeCut, myDataC2, myData, FFTW_ESTIMATE);
		
		fftw_execute(planR2C);
		
		for (int i=0;i<sizeCut/2+1;i++) {
			myDataC2[i][0]=myDataC[i][0];
			myDataC2[i][1]=myDataC[i][1];
			double aR=myDataC[i][0];
			double aI=myDataC[i][1];
			myDataC[i][0]=aR*aR+aI*aI;
			myDataC[i][1]=0.0;
		}
		
		fftw_execute(planC2R);
		
		double cutoff=1.0;
		for (int i=1;i<sizeCut/2;i++) {
			if (myData[i+1]>myData[i] && myData[i-1]>myData[i]){
				cutoff=M_PI*i/2.0;
				break;
			}
		}

		double sx=my_w.param->value()*pow(sizeCut/cutoff,2);
		
		for (int i=0;i<sizeCut/2+1;i++) {
			double blur=exp(-i*i/sx);
			myDataC2[i][0]*=blur;
			myDataC2[i][1]*=blur;
		}
		fftw_execute(planC2R2);
		
		foreach(QwtPlotMarker *mark, markers) {
			mark->detach();
		}
		markers.clear();
		
		QFont labFont;
		labFont.setPointSize(10);
		
		delete[] xd;
		delete[] yd;

		xd=new double[sizeCut];
		yd=new double[sizeCut];

		int k=0;
		for (int i=1;i<sizeCut-1;i++) {
			if (myData[i]>myData[i-1] && myData[i]>myData[i+1]){
				QwtPlotMarker *mark=new QwtPlotMarker();
				mark->setValue(lineout.sample(i));
				mark->setSymbol(new QwtSymbol(QwtSymbol::Ellipse,QBrush(QColor(255, 0, 0, 127)),QPen(Qt::black),QSize(5,5)));
				mark->attach(my_w.plot);
				markers << mark;
				xd[k]=k;
				yd[k]=lineout.sample(i).x();
				k++;
			}
		}
		k--;
		if (k>1) {
			double c0, c1, cov00, cov01, cov11, sumsq;
			gsl_fit_linear (xd, 1, yd, 1, k, &c0, &c1, &cov00, &cov01, &cov11, &sumsq);
			my_w.statusBar->showMessage(QString::number(cutoff)+" c00:"+QString::number(cov00)+" c01:"+QString::number(cov01)+" c11:"+QString::number(cov11)+" sq:"+QString::number(sqrt(sumsq)/k));
			my_w.origin->setText(QString::number(c0));
			my_w.scale->setText(QString::number(2*c1));
		}
		
		my_w.plot->setAxisScale(lineout.xAxis(),lineout.minXValue(),lineout.maxXValue(),0);
		my_w.plot->setAxisScale(lineout.yAxis(),lineout.minYValue(),lineout.maxYValue(),0);
		
		my_w.plot->replot();
		fftw_destroy_plan(planR2C);
		fftw_destroy_plan(planC2R);
		fftw_destroy_plan(planC2R2);
		fftw_free(myDataC);	
		fftw_free(myDataC2);	
		delete myData;
		delete[] xd;
		delete[] yd;
	}

}

void nFindPeaks::copy_clip() {
	if (currentBuffer) {
		QClipboard *clipboard = QApplication::clipboard();
		QString point_table="# FindPeaks "+QString::fromUtf8(currentBuffer->getName().c_str())+"\n";
		int k=0;
		foreach(QwtPlotMarker* mark,markers) {
			point_table.append(QString("%1\t%2\n").arg(k++).arg(mark->value().x()));
		}
		clipboard->setText(point_table);
		showMessage(tr("Points copied to clipboard"));
	}
}

void nFindPeaks::export_txt() {
	if (currentBuffer) {
		QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save data in text"),property("fileTxt").toString(),tr("Text files (*.txt *.csv);;Any files (*)"));
		if (!fnametmp.isEmpty()) {
			setProperty("fileTxt", fnametmp);
			QFile t(fnametmp);
			t.open(QIODevice::WriteOnly| QIODevice::Text);
			QTextStream out(&t);
			out << "# FindPeaks " << QString::fromUtf8(currentBuffer->getName().c_str()) <<endl;
			int k=0;
			foreach(QwtPlotMarker* mark,markers) {
				out << k++ << "\t" << mark->value().x() << endl;
			}
			t.close();
			showMessage(tr("Export in file:")+fnametmp,2000);
		}
	}
}

void
nFindPeaks::export_pdf() {
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

