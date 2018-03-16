#include "nSpectrum.h"

#include "neutrino.h"

nSpectrum::nSpectrum(nGenericPan *sparent)
    : nLine(sparent,3)
{
	disconnect(my_w.saveGraphPoints, SIGNAL(released()), my_w.plot, SLOT(save_data()));	
	connect(my_w.saveGraphPoints, SIGNAL(released()), this, SLOT(export_txt()));	

	disconnect(my_w.tabWidget, SIGNAL(currentChanged(int)), this, SLOT(updatePlot()));
	connect(my_w.tabWidget, SIGNAL(currentChanged(int)), this, SLOT(updatePlot()));

	// config
	my_w.points->setColumnCount(5);
	QStringList lbls; lbls<<"X"<<"Y"<<"K [MeV/nucl.]"<<"Q [e+]"<<"m [amu]";
	my_w.points->setHorizontalHeaderLabels(lbls);
}

QString
nSpectrum::getStringData(QPolygonF vals)
{
	// 1. interpolate energy
	double dist=0.;
	std::vector<float> dist_v, ener_v;
	dist_v.reserve(ref.size());
	ener_v.reserve(ref.size());

	QPolygonF vs;

	foreach (QGraphicsEllipseItem *it, ref) {
		vs << it->pos();
	}

	for (int ii=0; ii<vs.size(); ii++) {
		if (ii>0)
			dist+=sqrt(pow(vs.at(ii).x()-vs.at(ii-1).x(),2)+pow(vs.at(ii).y()-vs.at(ii-1).y(),2));
		
		dist_v.push_back(dist);
		ener_v.push_back(my_w.points->item(ii, 2)->text().toDouble());
	}
	


	dist = 0.;
	int lp=0;
	QString point_table;
	for (int i=0; i<vals.size(); i++) {
		if (i>0) 
			dist+=sqrt(pow(vals.at(i).x()-vals.at(i-1).x(),2)+pow(vals.at(i).y()-vals.at(i-1).y(),2));

		for (int ii=lp; ii<vs.size()-1; ii++) {
			if (dist_v[ii] < dist && dist_v[ii+1] >= dist) {
				lp = ii;
				break;
			}
		}

		// versione scacionissima
		double cur_en = ener_v[lp] + ((ener_v[lp+1]-ener_v[lp])/(dist_v[lp+1]-dist_v[lp]))*(dist-dist_v[lp]);

		point_table.append(QString("%1\t%2\t%3").arg(dist).arg(vals.at(i).x()).arg(vals.at(i).y()));

        if (nparent->getCurrentBuffer()) {
            vec2f orig = nparent->getCurrentBuffer()->get_origin();
            point_table.append(QString("\t%1").arg(nparent->getCurrentBuffer()->point(vals.at(i).x()+orig.x(),vals.at(i).y()+orig.y())));
		}

		point_table.append(QString("\t%1").arg(cur_en));

		point_table.append(QString("\n"));
	}
	return point_table;
}

void
nSpectrum::updatePlot()
{
    nPhysD *my_phys=nparent->getCurrentBuffer();
	if (my_w.plot->isVisible() && my_phys) {

		// 1. plot cleanup
		if (my_w.plot->graphCount()==0) {
			my_w.plot->addGraph(my_w.plot->xAxis, my_w.plot->yAxis);
			my_w.plot->graph(0)->setPen(QPen(Qt::blue));
			my_w.plot->xAxis->setLabel(tr("distance"));
			my_w.plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
			my_w.plot->xAxis->setLabelPadding(-1);
			my_w.plot->yAxis->setLabelPadding(-1);
            my_w.plot->xAxis->setTickLabelFont(nparent->my_w->my_view->font());
            my_w.plot->yAxis->setTickLabelFont(nparent->my_w->my_view->font());
		}


		// 2. interpolate energies on curvilinear coordinate
		double dist=0.;
		std::vector<float> dist_v, ener_v;
		dist_v.reserve(ref.size());
		ener_v.reserve(ref.size());

		QPolygonF vs;

		foreach (QGraphicsEllipseItem *it, ref) {
			vs << it->pos();
		}

		for (int ii=0; ii<vs.size(); ii++) {
			if (ii>0)
				dist+=sqrt(pow(vs.at(ii).x()-vs.at(ii-1).x(),2)+pow(vs.at(ii).y()-vs.at(ii-1).y(),2));

			dist_v.push_back(dist);
			ener_v.push_back(my_w.points->item(ii, 2)->text().toDouble());
		}



		dist = 0.;
		int lp=0;

		QVector<double> toPlotx;
		QVector<double> toPloty;
		QPolygonF vals = poly(numPoints);

		for (int i=0; i<vals.size(); i++) {
			if (i>0) 
				dist+=sqrt(pow(vals.at(i).x()-vals.at(i-1).x(),2)+pow(vals.at(i).y()-vals.at(i-1).y(),2));

			for (int ii=lp; ii<vs.size()-1; ii++) {
				if (dist_v[ii] < dist && dist_v[ii+1] >= dist) {
					lp = ii;
					break;
				}
			}

			// versione scacionissima
			double cur_en = ener_v[lp] + ((ener_v[lp+1]-ener_v[lp])/(dist_v[lp+1]-dist_v[lp]))*(dist-dist_v[lp]);

			toPlotx << cur_en;
            vec2f orig = nparent->getCurrentBuffer()->get_origin();
			toPloty << my_phys->point(vals.at(i).x()+orig.x(),vals.at(i).y()+orig.y());
		}

		// 3. do the plot
		my_w.plot->graph(0)->setData(toPlotx,toPloty);
		my_w.plot->rescaleAxes();
		my_w.plot->replot();
	}
}

void 
nSpectrum::export_txt(){
	// exports poly to file
	QVariant varia=property("txtFile");
	QString fname;
	if (varia.isValid()) {
		fname=varia.toString();
	} else {
		fname=QString("lineData.txt");
	}

	QString fnametmp=QFileDialog::getSaveFileName(&my_pad,tr("Save data in text"),fname,tr("Text files (*.txt *.csv);;Any files (*)"));
	
	if (!fnametmp.isEmpty()) {
		setProperty("txtFile",fnametmp);
		QFile t(fnametmp);
		t.open(QIODevice::WriteOnly| QIODevice::Text);
		QTextStream out(&t);
		out<<"# neutrino tp-tracker plugin\n";
		out<<"# s\tx\ty\tval\tK[MeV]\n#\n";
		out << getStringData(poly(numPoints));
		t.close();
		showMessage(tr("Data saved in file ")+fnametmp);
	}
}

