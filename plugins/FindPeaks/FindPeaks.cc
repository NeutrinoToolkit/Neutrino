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
#include "FindPeaks.h"

#include <gsl/gsl_fit.h>

FindPeaks::FindPeaks(neutrino *nparent) : nGenericPan(nparent)
{
	my_w.setupUi(this);

	// signals
    box =  new nRect(this,1);
	box->setRect(QRectF(0,0,100,100));


	connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
	connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));

    connect(my_w.actionSaveClipboard, SIGNAL(triggered()), my_w.plot, SLOT(copy_data()));
    connect(my_w.actionSaveTxt      , SIGNAL(triggered()), my_w.plot, SLOT(save_data()));
    connect(my_w.actionSavePDF      , SIGNAL(triggered()), my_w.plot, SLOT(export_image()));

	connect(my_w.setOrigin, SIGNAL(pressed()), this, SLOT(setOrigin()));
	connect(my_w.setScale, SIGNAL(pressed()), this, SLOT(setScale()));

	connect(my_w.actionRect, SIGNAL(triggered()), box, SLOT(togglePadella()));

	my_w.toolBar->addWidget(my_w.direction);
	my_w.toolBar->addWidget(my_w.param);

    show();

    my_w.plot->addGraph(my_w.plot->xAxis, my_w.plot->yAxis);
    my_w.plot->graph(0)->setPen(QPen(Qt::blue));
    my_w.plot->graph(0)->setName("FindPeaks");

    my_w.plot->addGraph(my_w.plot->xAxis, my_w.plot->yAxis);
    QPen pen;
    pen.setColor(Qt::green);
    my_w.plot->graph(1)->setPen(pen);
    my_w.plot->graph(1)->setName("Blurred");

	connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(updatePlot()));
    connect(box, SIGNAL(sceneChanged()), this, SLOT(updatePlot()));
 	connect(my_w.direction, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlot()));
 	connect(my_w.param, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
	updatePlot();
}


void FindPeaks::setOrigin() {
	if (currentBuffer) {
		bool ok=true;
		double originOffset=0.0;
        if (!my_w.originOffset->text().isEmpty()) originOffset=QLocale().toDouble(my_w.originOffset->text(),&ok);
		if (ok) {
            double origin=QLocale().toDouble(my_w.origin->text(),&ok)-originOffset;
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

void FindPeaks::setScale() {
	if (currentBuffer) {
		bool ok=true;
		double scaleMult=1.0;
        if (!my_w.scaleOffset->text().isEmpty()) scaleMult=QLocale().toDouble(my_w.scaleOffset->text(),&ok);
		if (ok) {
            double scale=scaleMult/QLocale().toDouble(my_w.scale->text(),&ok);
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

void FindPeaks::mouseAtMatrix(QPointF p) {
	if (currentBuffer) {
        if (my_w.direction->currentIndex()==0) {
            my_w.plot->setMousePosition(p.x());
        } else {
            my_w.plot->setMousePosition(p.y());
        }
    }
}

void FindPeaks::updatePlot() {
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


        QVector<double> myData;

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
        fftw_plan planC2R2=fftw_plan_dft_c2r_1d(sizeCut, myDataC2, &myData[0], FFTW_ESTIMATE);
		
        fftw_execute(planR2C);

        for (int i=0;i<sizeCut/2+1;i++) {
            myDataC2[i][0]=myDataC[i][0];
            myDataC2[i][1]=myDataC[i][1];
        }

        double sx=my_w.param->value()*sqrt(sizeCut);
		
        for (int i=0;i<sizeCut/2+1;i++) {
            double blur=exp(-i*i/sx);
            myDataC2[i][0]*=blur;
            myDataC2[i][1]*=blur;
        }
        fftw_execute(planC2R2);
		
        for (int i=0; i< my_w.plot->itemCount(); i++) {
            QCPItemStraightLine *marker = qobject_cast<QCPItemStraightLine *>(my_w.plot->item(i));
            if (marker && marker->property(panName().toLatin1()).isValid()) my_w.plot->removeItem(marker);
        }

        std::vector<double> fitx;
        std::vector<double> fity;

        if (my_w.direction->currentIndex()==0) {
            my_w.plot->graph(1)->setData(xdata,myData);
        } else {
            my_w.plot->graph(1)->setData(ydata,myData);
        }

        int k=0;
        for (int i=1;i<sizeCut-1;i++) {
            if (myData[i]>myData[i-1] && myData[i]>myData[i+1]){
                double posx=i+(my_w.direction->currentIndex()==0?geom2.x():geom2.y());

                QCPItemStraightLine *marker=new QCPItemStraightLine(my_w.plot);
                marker->point1->setCoords(posx,0);
                marker->point2->setCoords(posx,1);
                marker->setProperty(panName().toLatin1(),true);

                marker->setPen(QPen(Qt::red));

                fitx.push_back(k);
                fity.push_back(posx);
                k++;
            }
        }

        if (fitx.size()>2) {
            double c0, c1, cov00, cov01, cov11, sumsq;
            gsl_fit_linear (&fitx[0], 1, &fity[0], 1, fitx.size(), &c0, &c1, &cov00, &cov01, &cov11, &sumsq);
            my_w.statusBar->showMessage(" c00:"+QString::number(cov00)+" c01:"+QString::number(cov01)+" c11:"+QString::number(cov11)+" sq:"+QString::number(sqrt(sumsq)/fitx.size()));
            my_w.origin->setText(QString::number(c0));
            my_w.scale->setText(QString::number(c1));
        }
		
        fftw_destroy_plan(planR2C);
        fftw_destroy_plan(planC2R2);
        fftw_free(myDataC);
        fftw_free(myDataC2);

        my_w.plot->rescaleAxes();
        my_w.plot->replot();

    }

}

