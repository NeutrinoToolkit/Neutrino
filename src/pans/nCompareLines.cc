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
#include "nCompareLines.h"

#include <qwt_plot_zoomer.h>
#include <qwt_plot_panner.h>
#include <qwt_plot_renderer.h>
//#include <qwt_legend.h>
//#if QWT_VERSION < 0x060100
//#include <qwt_legend_item.h>
//#else
//#include <qwt_legend_label.h>
//#endif


nCompareLines::nCompareLines(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);

	// signals
	line =  new nLine(nparent);
	line->setParentPan(panName,1);
	QPolygonF poly;
	poly << QPointF(0,0) << QPointF(100,100);
	line->setPoints(poly);

	connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
	connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
	connect(my_w.actionSaveClipboard, SIGNAL(triggered()), this, SLOT(copy_clip()));
	connect(my_w.actionSaveTxt, SIGNAL(triggered()), this, SLOT(export_txt()));
	connect(my_w.actionSavePDF, SIGNAL(triggered()), this, SLOT(export_pdf()));


	connect(my_w.actionLine, SIGNAL(triggered()), line, SLOT(togglePadella()));
	connect(my_w.addImage, SIGNAL(released()), this, SLOT(addImage()));
	connect(my_w.removeImage, SIGNAL(released()), this, SLOT(removeImage()));

	connect(my_w.current, SIGNAL(released()), this, SLOT(updatePlot()));

	connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
	connect(nparent, SIGNAL(physMod(std::pair<nPhysD*,nPhysD*>)), this, SLOT(physMod(std::pair<nPhysD*,nPhysD*>)));
    
    
	my_w.plot->setAxisTitle(QwtPlot::xBottom, tr("Distance"));
	my_w.plot->setAxisTitle(QwtPlot::yLeft, tr("Value"));
	my_w.plot->enableAxis(QwtPlot::xBottom);
	my_w.plot->enableAxis(QwtPlot::yLeft);
	(qobject_cast<QFrame*> (my_w.plot->canvas()))->setLineWidth(0);

//	QwtLegend *legend = new QwtLegend;
//!TODO: check this: it is not compatible with qwt6.1.0
//    legend->setItemMode(QwtLegend::CheckableItem);
//    my_w.plot->insertLegend(legend, QwtPlot::ExternalLegend);
//    my_w.plot->insertLegend(legend);
	
	decorate();
	loadDefaults();
	connect(line, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
	connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePlot()));
	updatePlot();
}


void nCompareLines::physDel(nPhysD* my_phys) {
    images.removeAll(my_phys);    
}

void nCompareLines::physMod(std::pair<nPhysD*,nPhysD*> my_mod) {
    images.removeAll(my_mod.first);    
    images.append(my_mod.second);        
}

void nCompareLines::addImage() {
    images.append(nGenericPan::getPhysFromCombo(my_w.image));    
    updatePlot();
}

void nCompareLines::removeImage() {
    images.removeAll(nGenericPan::getPhysFromCombo(my_w.image));        
    updatePlot();
}

void nCompareLines::sceneChanged() {
	if (sender()==line) updatePlot();
}

void nCompareLines::updatePlot() {
	if (currentBuffer && isVisible()) {
		foreach (QwtPlotCurve *profile, profiles) {
			profile->detach();
			profile->setData(NULL);
			profile->~QwtPlotCurve();
		}
		profiles.clear();
		
		QPolygonF my_points;
		foreach(QGraphicsEllipseItem *item, line->ref){
			my_points<<item->pos();
		}
        
		QPolygonF my_poly=line->poly(line->numPoints);
		for (int i=0; i<nparent->getBufferList().size(); i++) {
			nPhysD *phys=nparent->getBufferList().at(i);
            
            if (images.contains(phys) || (my_w.current->isChecked() && phys==currentBuffer)) {
                QVector< QPointF > toPlot;
                
                double dist=0.0;
                double my_val=0.0;
                for(int ii=0;ii<my_poly.size()-1;ii++) {
                    QPointF p=my_poly.at(ii);
                    my_val=phys->getPoint(p.x()-phys->get_origin().x(),p.y()-phys->get_origin().y());
                    if (std::isfinite(my_val)) toPlot << QPointF(dist, my_val);
                    dist+=sqrt(pow((my_poly.at(ii+1)-my_poly.at(ii)).x(),2)+pow((my_poly.at(ii+1)-my_poly.at(ii)).y(),2));
                }
                QPointF p=my_poly.last();
                my_val=phys->getPoint(p.x(),p.y());
                if (std::isfinite(my_val)) toPlot << QPointF(dist, my_val);
                
                QwtPlotCurve *profile=new QwtPlotCurve(QString::number(i)+": "+QString::fromUtf8(phys->getShortName().c_str()));
                if (phys==currentBuffer) {
                    profile->setPen(QPen(Qt::blue,1.0));
                } else {
                    profile->setPen(QPen(Qt::black,1.0));
                }	
                profile->setXAxis(QwtPlot::xBottom);
                profile->setYAxis(QwtPlot::yLeft);
                profile->setSamples(toPlot);
                profile->attach(my_w.plot);
                profiles << profile;
            }
            
		}
		my_w.plot->replot();
	}
}

void nCompareLines::copy_clip() {
	QClipboard *clipboard = QApplication::clipboard();
	clipboard->setText(getText());
	showMessage(tr("Points copied to clipboard"));
}

QString nCompareLines::getText() {
	QString point_table;
	if (profiles.size()>0) {
		point_table.append("# Distance");
		foreach (QwtPlotCurve *profile, profiles) {
			point_table.append("\t"+profile->title().text());
		}
		for (unsigned int i=0;i<profiles.first()->data()->size();i++) {
			point_table.append("\n"+QString::number(profiles.first()->data()->sample(i).x()));
			foreach (QwtPlotCurve *profile, profiles) {
				point_table.append("\t"+QString::number(profile->data()->sample(i).y()));
			}
		}
		point_table.append("\n");
	}
	return point_table;
}

void nCompareLines::export_txt() {
	QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save data in text"),property("fileTxt").toString(),tr("Text files (*.txt *.csv);;Any files (*)"));
	if (!fnametmp.isEmpty()) {
		setProperty("fileTxt", fnametmp);
		QFile t(fnametmp);
		t.open(QIODevice::WriteOnly| QIODevice::Text);
		QTextStream out(&t);
		out << getText();
		t.close();
		showMessage(tr("Export in file:")+fnametmp,2000);
	}
}

void
nCompareLines::export_pdf() {
	QString fnametmp = QFileDialog::getSaveFileName(this,tr("Save Drawing"),property("fileExport").toString(),"Vector files (*.pdf,*.svg)");
	if (!fnametmp.isEmpty()) {
		setProperty("fileExport", fnametmp);
		QwtPlotRenderer renderer;
		renderer.setDiscardFlag(QwtPlotRenderer::DiscardBackground, false);
//		renderer.setLayoutFlag(QwtPlotRenderer::KeepFrames, true);
		renderer.renderDocument(my_w.plot, fnametmp, QFileInfo(fnametmp).suffix(), QSizeF(150, 100), 85);
	}
}

