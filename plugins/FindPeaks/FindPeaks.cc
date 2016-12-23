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
    setupUi(this);

    // signals
    box =  new nRect(this,1);
    box->setRect(QRectF(0,0,100,100));


    connect(actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
    connect(actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));

    connect(setOrigin, SIGNAL(pressed()), this, SLOT(setOrigin()));
    connect(setScale, SIGNAL(pressed()), this, SLOT(setScale()));

    connect(actionRect, SIGNAL(triggered()), box, SLOT(togglePadella()));

    toolBar->addWidget(direction);
    toolBar->addWidget(param);

    show();

    plot->addGraph(plot->xAxis, plot->yAxis);
    plot->graph(0)->setName("FindPeaks");

    plot->addGraph(plot->xAxis, plot->yAxis);
    plot->graph(1)->setName("Blurred");
    QPen p=plot->graph(1)->pen();
    p.setStyle(Qt::DotLine);
    plot->graph(1)->setPen(p);

    connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(updatePlot()));
    connect(box, SIGNAL(sceneChanged()), this, SLOT(updatePlot()));
    connect(direction, SIGNAL(toggled(bool)), this, SLOT(updatePlot()));
    connect(param, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
    updatePlot();
}


void FindPeaks::set_origin() {
    if (currentBuffer) {
        bool ok=true;
        double my_origin=0.0;
        if (!originOffset->text().isEmpty()) my_origin=QLocale().toDouble(originOffset->text(),&ok);
        if (ok) {
            double origin=QLocale().toDouble(originOffset->text(),&ok)-my_origin;
            if (ok) {
                if (direction->isChecked()) {
                    currentBuffer->set_origin(currentBuffer->get_origin().x(),origin);
                } else {
                    currentBuffer->set_origin(origin,currentBuffer->get_origin().y());
                }
                nparent->my_tics.update();
            }
        }
    }
}

void FindPeaks::set_scale() {
    if (currentBuffer) {
        bool ok=true;
        double scaleMult=1.0;
        if (!scaleOffset->text().isEmpty()) scaleMult=QLocale().toDouble(scaleOffset->text(),&ok);
        if (ok) {
            double my_scale=scaleMult/QLocale().toDouble(scale->text(),&ok);
            if (ok) {
                if (direction->isChecked()) {
                    currentBuffer->set_scale(currentBuffer->get_scale().x(),my_scale);
                } else {
                    currentBuffer->set_scale(my_scale,currentBuffer->get_scale().y());
                }
                nparent->my_tics.update();
            }
        }
    }
}

void FindPeaks::mouseAtMatrix(QPointF p) {
    if (currentBuffer) {
        if (direction->isChecked()) {
            plot->setMousePosition(p.y());
        } else {
            plot->setMousePosition(p.x());
        }
    }
}

void FindPeaks::updatePlot() {
    if (currentBuffer && isVisible()) {
        saveDefaults();
        QRect geom2=box->getRect(currentBuffer);
        if (geom2.isEmpty()) {
            statusbar->showMessage(tr("Attention: the region is outside the image!"),2000);
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

        if (direction->isChecked()) {
            plot->graph(0)->setData(ydata,yd);
            myData.resize(yd.size());
            std::copy ( yd.begin(), yd.end(), myData.begin() );
        } else {
            plot->graph(0)->setData(xdata,xd);
            myData.resize(xd.size());
            std::copy ( xd.begin(), xd.end(), myData.begin() );
        }

        int sizeCut=myData.size();
        transform(myData.begin(), myData.end(), myData.begin(),bind2nd(std::divides<double>(), sizeCut));

        fftw_complex *myDataC=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(sizeCut/2+1));
        fftw_plan planR2C=fftw_plan_dft_r2c_1d(sizeCut, &myData[0], myDataC, FFTW_ESTIMATE);
        fftw_plan planC2R2=fftw_plan_dft_c2r_1d(sizeCut, myDataC, &myData[0], FFTW_ESTIMATE);

        fftw_execute(planR2C);

        for (int i=0;i<sizeCut/2+1;i++) {
            myDataC[i][0]=myDataC[i][0];
            myDataC[i][1]=myDataC[i][1];
        }

        double sx=pow(sizeCut/param->value(),2)/2.0;

        for (int i=0;i<sizeCut/2+1;i++) {
            double blur=exp(-i*i/sx);
            myDataC[i][0]*=blur;
            myDataC[i][1]*=blur;
        }
        fftw_execute(planC2R2);

        std::vector<double> fitx;
        std::vector<double> fity;
        std::vector<double> fitz;

        if (direction->isChecked()) {
            plot->graph(1)->setData(ydata,myData);
        } else {
            plot->graph(1)->setData(xdata,myData);
        }


        int k=0;
        for (int i=1;i<sizeCut-1;i++) {
            if (myData[i]>myData[i-1] && myData[i]>myData[i+1]){
                double posx=i+(direction->isChecked()?geom2.y():geom2.x());

                fitx.push_back(k);
                fity.push_back(posx);
                fitz.push_back(myData[i]);
                k++;

            }
        }



        if (fitx.size()>2) {
            double c0, c1, cov00, cov01, cov11, sum_square;
            gsl_fit_linear (&fitx[0], 1, &fity[0], 1, fitx.size(), &c0, &c1, &cov00, &cov01, &cov11, &sum_square);
            QString msg=QString::number(fitx.size())+
                    " pts [c00:"+QString::number(cov00)+
                    " c01:"+QString::number(cov01)+
                    " c11:"+QString::number(cov11)+
                    " sq:"+QString::number(sqrt(sum_square)/fitx.size())+
                    "]";
            statusbar->showMessage(msg);
            origin->setText(QLocale().toString(c0));
            scale->setText(QLocale().toString(c1));
        }

        for (int i=0; i< plot->itemCount(); i++) {
            if (plot->item(i)->property(panName().toLatin1()).isValid()) plot->removeItem(plot->item(i));
        }
        points->setRowCount(0);
        for (unsigned int i=0;i<fitx.size();i++) {
            QCPItemStraightLine *marker=new QCPItemStraightLine(plot);
            marker->point1->setCoords(fity[i],0);
            marker->point2->setCoords(fity[i],1);
            marker->setProperty(panName().toLatin1(),true);
            marker->setPen(QPen(Qt::red));

            int pos=points->rowCount();
            points->insertRow(pos);
            QTableWidgetItem *xitem= new QTableWidgetItem(QLocale().toString(fity[i]));
            QTableWidgetItem *yitem= new QTableWidgetItem(QLocale().toString(fitz[i]));
            xitem->setTextAlignment(Qt::AlignHCenter + Qt::AlignVCenter);
            yitem->setTextAlignment(Qt::AlignHCenter + Qt::AlignVCenter);
            points->setItem(pos, 0, xitem);
            points->setItem(pos, 1, yitem);
            points->resizeRowToContents(pos);
        }

        fftw_destroy_plan(planR2C);
        fftw_destroy_plan(planC2R2);
        fftw_free(myDataC);

        plot->rescaleAxes();
        plot->replot();

    }

}

void FindPeaks::on_actionClipboard_triggered() {
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(getPoints());
    statusbar->showMessage(tr("Point values copied"),2000);
}

void FindPeaks::on_actionTxt_triggered() {
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save data"),property("NeuSave-fileTxt").toString(),tr("Text files (*.txt *.csv);;Any files (*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileTxt",fnametmp);
        QFile t(fnametmp);
        t.open(QIODevice::WriteOnly| QIODevice::Text);
        QTextStream out(&t);
        out << getPoints();
        t.close();
        statusbar->showMessage(tr("Export in file:")+fnametmp,2000);
    } else {
        statusbar->showMessage(tr("Export canceled"),2000);
    }
}

QString FindPeaks::getPoints() {
    QString retText;
    for (int i=0; i<points->rowCount(); i++) {
        retText += QLocale().toString(i) + "\t";
        for (int j=0; j<points->columnCount();j++) {
            retText += points->item(i, j)->text() + "\t";
        }
        retText += "\n";
    }
    return retText;
}

