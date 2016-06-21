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

    my_w.plot->addGraph(my_w.plot->xAxis, my_w.plot->yAxis);
    my_w.plot->graph(0)->setPen(QPen(Qt::blue));

	decorate();
	loadDefaults();
	connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(updatePlot()));
    connect(box, SIGNAL(sceneChanged()), this, SLOT(updatePlot()));
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

void nFindPeaks::mouseAtMatrix(QPointF p) {
	if (currentBuffer) {
        if (my_w.direction->currentIndex()==0) {
            my_w.plot->setMousePosition(p.x());
        } else {
            my_w.plot->setMousePosition(p.y());
        }
    }
}

void nFindPeaks::updatePlot() {
	if (currentBuffer && isVisible()) {
        saveDefaults();
        QRect geom2=box->getRect(currentBuffer);
		if (geom2.isEmpty()) {
			my_w.statusBar->showMessage(tr("Attention: the region is outside the image!"),2000);
			return;
		}
		
		int dx=geom2.width();
		int dy=geom2.height();

        QVector<double> xd(dx,0.0);
        QVector<double> yd(dy,0.0);

		for (int j=0;j<dy;j++){
			for (int i=0;i<dx; i++) {
				double val=currentBuffer->point(i+geom2.x(),j+geom2.y(),0.0);
				xd[i]+=val;
				yd[j]+=val;
			}
		}
        transform(xd.begin(), xd.end(), xd.begin(),bind2nd(std::divides<double>(), dy));
        transform(yd.begin(), yd.end(), yd.begin(),bind2nd(std::divides<double>(), dx));

        QVector<double> xdata(dx);
        QVector<double> ydata(dy);
		
        for (int i=0;i<dx;i++) xdata[i]=i+geom2.x();
        for (int j=0;j<dy;j++) ydata[j]=j+geom2.y();


        std::vector<double> myData;

        if (my_w.direction->currentIndex()==0) {
            my_w.plot->graph(0)->setData(xdata,xd);
            myData.resize(xd.size());
            std::copy ( xd.begin(), xd.end(), myData.begin() );
        } else {
            my_w.plot->graph(0)->setData(ydata,yd);
            myData.resize(yd.size());
            std::copy ( yd.begin(), yd.end(), myData.begin() );
        }

        int sizeCut=myData.size();
        transform(myData.begin(), myData.end(), myData.begin(),bind2nd(std::divides<double>(), sizeCut));
		
        fftw_complex *myDataC=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(sizeCut/2+1));
        fftw_complex *myDataC2=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(sizeCut/2+1));
        fftw_plan planR2C=fftw_plan_dft_r2c_1d(sizeCut, &myData[0], myDataC, FFTW_ESTIMATE);
        fftw_plan planC2R=fftw_plan_dft_c2r_1d(sizeCut, myDataC, &myData[0], FFTW_ESTIMATE);
        fftw_plan planC2R2=fftw_plan_dft_c2r_1d(sizeCut, myDataC2, &myData[0], FFTW_ESTIMATE);
		
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
		
        my_w.plot->clearItems();

        std::vector<double> fitx;
        std::vector<double> fity;

        int k=0;
        for (int i=1;i<sizeCut-1;i++) {
            if (myData[i]>myData[i-1] && myData[i]>myData[i+1]){
                double posx=i+(my_w.direction->currentIndex()==0?geom2.x():geom2.y());
                double posy=my_w.direction->currentIndex()==0?xd[i]:yd[i];

                QCPItemEllipse *marker=new QCPItemEllipse(my_w.plot);
                marker->topLeft->setCoords(posx-1, posy-1);
                marker->bottomRight->setCoords(posx+1, posy+1);

                marker->setPen(QPen(Qt::red));
                my_w.plot->addItem(marker);

                fitx.push_back(k);
                fity.push_back(posx);
                k++;
            }
        }

        if (fitx.size()>2) {
            double c0, c1, cov00, cov01, cov11, sumsq;
            gsl_fit_linear (&fitx[0], 1, &fity[0], 1, fitx.size(), &c0, &c1, &cov00, &cov01, &cov11, &sumsq);
            my_w.statusBar->showMessage(QString::number(cutoff)+" c00:"+QString::number(cov00)+" c01:"+QString::number(cov01)+" c11:"+QString::number(cov11)+" sq:"+QString::number(sqrt(sumsq)/fitx.size()));
            my_w.origin->setText(QString::number(c0));
            my_w.scale->setText(QString::number(c1));
        }
		
        fftw_destroy_plan(planR2C);
        fftw_destroy_plan(planC2R);
        fftw_destroy_plan(planC2R2);
        fftw_free(myDataC);
        fftw_free(myDataC2);

        my_w.plot->rescaleAxes();
        my_w.plot->replot();

    }

}

void nFindPeaks::copy_clip() {
	if (currentBuffer) {
		QClipboard *clipboard = QApplication::clipboard();
        QString point_table;
        QTextStream out(&point_table);
        export_data(out);
        clipboard->setText(point_table);
		showMessage(tr("Points copied to clipboard"));
	}
}

void nFindPeaks::export_data(QTextStream &out) {
    out << "# FindPeaks " << QString::fromUtf8(currentBuffer->getName().c_str()) <<endl;
    for(int i=0;i<my_w.plot->itemCount();i++) {
        QCPItemEllipse *elli=qobject_cast<QCPItemEllipse*>(my_w.plot->item(i));
        if (elli) {
            out << i << " " << 0.5*(elli->bottomRight->coords().x() + elli->topLeft->coords().x()) <<endl;
        }
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
            export_data(out);
			t.close();
			showMessage(tr("Export in file:")+fnametmp,2000);
		}
	}
}

void
nFindPeaks::export_pdf() {
	QString fout;
    QString fnametmp = QFileDialog::getSaveFileName(this,tr("Save Drawing"),property("fileExport").toString(),"Vector files (*.pdf)");
	if (!fnametmp.isEmpty()) {
		setProperty("fileExport", fnametmp);
        my_w.plot->savePdf(fnametmp,true,0,0,"Neutrino", panName);
	}
}

