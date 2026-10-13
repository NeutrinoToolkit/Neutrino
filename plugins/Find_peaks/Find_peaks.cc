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
#include "Find_peaks.h"
#include <functional>
#include <algorithm>
#include <numeric>

#include <gsl/gsl_fit.h>

Find_peaks::Find_peaks(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);

    // signals
    box =  new nRect(this,1);
    box->setRect(QRectF(0,0,100,100));

    connect(setOrigin, SIGNAL(clicked()), this, SLOT(set_origin()));
    connect(setScale, SIGNAL(clicked()), this, SLOT(set_scale()));

    connect(actionRect, SIGNAL(triggered()), box, SLOT(togglePadella()));

    toolBar->addWidget(direction);
    toolBar->addWidget(param);
    toolBar->addWidget(order);
    toolBar->addWidget(prominence);


    show();

    plot->addGraph(plot->xAxis, plot->yAxis);
    plot->graph(0)->setName("Find_peaks");

    plot->addGraph(plot->xAxis, plot->yAxis);
    plot->graph(1)->setName("Blurred");
    QPen p=plot->graph(1)->pen();
    p.setStyle(Qt::DotLine);
    plot->graph(1)->setPen(p);

    connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(updatePlot()));
    connect(box, SIGNAL(sceneChanged()), this, SLOT(updatePlot()));
    connect(direction, SIGNAL(toggled(bool)), this, SLOT(updatePlot()));
    connect(param, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
    connect(order, SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));
    connect(prominence, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
    connect(transpose, SIGNAL(toggled(bool)), this, SLOT(updatePlot()));
    updatePlot();
}


void Find_peaks::set_origin() {
    if (currentBuffer) {
        bool ok=true;
        double my_origin=0.0;
        if (!originOffset->text().isEmpty()) my_origin=locale().toDouble(originOffset->text(),&ok);
        if (ok) {
            // fitted position of the first peak, minus the requested offset
            double new_origin=locale().toDouble(origin->text(),&ok)-my_origin;
            if (ok && std::isfinite(new_origin)) {
                if (direction->isChecked()) {
                    currentBuffer->set_origin(currentBuffer->get_origin().x(),new_origin);
                } else {
                    currentBuffer->set_origin(new_origin,currentBuffer->get_origin().y());
                }
                nparent->emitPhysChanged(currentBuffer);
            }
        }
    }
}

void Find_peaks::set_scale() {
    if (currentBuffer) {
        bool ok=true;
        double scaleMult=1.0;
        if (!scaleOffset->text().isEmpty()) scaleMult=locale().toDouble(scaleOffset->text(),&ok);
        if (ok) {
            double my_scale=scaleMult/locale().toDouble(scale->text(),&ok);
            if (ok && std::isfinite(my_scale) && my_scale!=0) {
                if (direction->isChecked()) {
                    currentBuffer->set_scale(currentBuffer->get_scale().x(),my_scale);
                } else {
                    currentBuffer->set_scale(my_scale,currentBuffer->get_scale().y());
                }
                nparent->emitPhysChanged(currentBuffer);
            } else {
                statusbar->showMessage(tr("Invalid scale: the fitted period is zero or missing"),5000);
            }
        }
    }
}

void Find_peaks::mouseAtMatrix(QPointF p) {
    if (currentBuffer) {
        if (direction->isChecked()) {
            plot->setMousePosition(p.y());
        } else {
            plot->setMousePosition(p.x());
        }
    }
}

bool polynomialfit(std::vector<double> dx, std::vector<double> dy, std::vector<double> &store) /* n, p */
{
    gsl_multifit_linear_workspace *ws;
    gsl_matrix *cov, *X;
    gsl_vector *y, *c;
    double chisq;

    int i, j;

    if (dx.size() != dy.size()) return false;

    int degree=store.size();
    int obs = dx.size();
    if (degree>=obs+1) return false;


    X = gsl_matrix_alloc(obs, degree);
    y = gsl_vector_alloc(obs);
    c = gsl_vector_alloc(degree);
    cov = gsl_matrix_alloc(degree, degree);

    for(i=0; i < obs; i++) {
        for(j=0; j < degree; j++) {
            gsl_matrix_set(X, i, j, pow(dx[i], j));
        }
        gsl_vector_set(y, i, dy[i]);
    }

    ws = gsl_multifit_linear_alloc(obs, degree);
    gsl_multifit_linear(X, y, c, cov, &chisq, ws);

    /* store result ... */
    for(i=0; i < degree; i++)
    {
        store[i] = gsl_vector_get(c, i);
    }

    gsl_multifit_linear_free(ws);
    gsl_matrix_free(X);
    gsl_matrix_free(cov);
    gsl_vector_free(y);
    gsl_vector_free(c);

    return true;
}

void Find_peaks::updatePlot() {
    if (!isVisible()) return;
    QRect geom2;
    if (currentBuffer) {
        geom2=box->getRect(currentBuffer);
        if (geom2.isEmpty()) statusbar->showMessage(tr("Attention: the region is outside the image!"),2000);
    }
    if (!currentBuffer || geom2.isEmpty()) { // no stale results
        for (int i=plot->itemCount()-1; i>=0; i--) {
            if (plot->item(i)->property(panName().toLatin1()).isValid()) plot->removeItem(plot->item(i));
        }
        plot->graph(0)->data()->clear();
        plot->graph(1)->data()->clear();
        plot->replot();
        points->setRowCount(0);
        coeff->clear();
        fourier->clear();
        return;
    }
    {
        int dx=geom2.width();
        int dy=geom2.height();

        // averages over the finite pixels only (NaN would spread over everything through the FFT)
        QVector<double> xd(dx,0.0), yd(dy,0.0);
        QVector<int> xn(dx,0), yn(dy,0);

        for (int j=0;j<dy;j++){
            for (int i=0;i<dx; i++) {
                double val=currentBuffer->point(i+geom2.x(),j+geom2.y(),0.0);
                if (std::isfinite(val)) {
                    xd[i]+=val; xn[i]++;
                    yd[j]+=val; yn[j]++;
                }
            }
        }
        for (int i=0;i<dx;i++) xd[i] = xn[i] ? xd[i]/xn[i] : std::numeric_limits<double>::quiet_NaN();
        for (int j=0;j<dy;j++) yd[j] = yn[j] ? yd[j]/yn[j] : std::numeric_limits<double>::quiet_NaN();

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

        const int N=myData.size();
        const double offset=(direction->isChecked()?geom2.y():geom2.x());

        // empty rows/columns (all NaN): use the mean of the profile so the FFT stays finite
        double meanFinite=0; int nFinite=0;
        for (auto &v : myData) if (std::isfinite(v)) { meanFinite+=v; nFinite++; }
        meanFinite = nFinite ? meanFinite/nFinite : 0.0;
        std::vector<double> raw(N);
        for (int i=0;i<N;i++) raw[i] = std::isfinite(myData[i]) ? myData[i] : meanFinite;

        // 1. smoothing: Gaussian low-pass on the mirrored profile (2N points), so that the two
        //    ends don't wrap into each other as they would in a plain periodic FFT
        const int M=2*N;
        std::vector<double> ext(M);
        for (int i=0;i<N;i++) { ext[i]=raw[i]/M; ext[M-1-i]=raw[i]/M; }
        fftw_complex *spec=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(M/2+1));
        fftw_plan planR2C=fftw_plan_dft_r2c_1d(M, ext.data(), spec, FFTW_ESTIMATE);
        fftw_plan planC2R=fftw_plan_dft_c2r_1d(M, spec, ext.data(), FFTW_ESTIMATE);
        fftw_execute(planR2C);
        double sx=pow(M/param->value(),2)/2.0;
        for (int i=0;i<M/2+1;i++) {
            double blur=exp(-i*i/sx);
            spec[i][0]*=blur;
            spec[i][1]*=blur;
        }
        fftw_execute(planC2R);
        fftw_destroy_plan(planR2C);
        fftw_destroy_plan(planC2R);
        fftw_free(spec);
        QVector<double> smooth(N);
        for (int i=0;i<N;i++) smooth[i]=ext[i];

        if (direction->isChecked()) {
            plot->graph(1)->setData(ydata,smooth);
        } else {
            plot->graph(1)->setData(xdata,smooth);
        }

        // 2. local maxima, kept if prominent enough, at sub-pixel position (parabola through 3 points)
        struct Peak { double pos, val; };
        std::vector<Peak> peaks;
        if (N>2) {
            const auto [lo,hi]=std::minmax_element(smooth.begin(),smooth.end());
            const double minProm=prominence->value()/100.0*(*hi-*lo);
            for (int i=1;i<N-1;i++) {
                if (smooth[i]>smooth[i-1] && smooth[i]>=smooth[i+1]) { // a 2-point flat top counts once
                    // prominence: height above the higher of the two valleys reached before a higher point
                    double lmin=smooth[i], rmin=smooth[i];
                    for (int j=i-1; j>=0 && smooth[j]<=smooth[i]; j--) lmin=std::min(lmin,smooth[j]);
                    for (int j=i+1; j<N && smooth[j]<=smooth[i]; j++) rmin=std::min(rmin,smooth[j]);
                    if (smooth[i]-std::max(lmin,rmin) < minProm) continue;
                    const double a1=smooth[i-1], b1=smooth[i], c1=smooth[i+1];
                    const double den=a1-2*b1+c1;
                    double d = den<0 ? 0.5*(a1-c1)/den : 0.0;
                    d=std::min(0.5,std::max(-0.5,d));
                    peaks.push_back({i+d+offset, b1-0.25*(a1-c1)*d});
                }
            }
        }

        // 3. peak number from the distance to the first one in units of the typical spacing:
        //    a missing peak leaves a gap instead of shifting all the following ones
        std::vector<double> fitx;
        std::vector<double> fity;
        std::vector<double> fitz;
        double spacing=0;
        if (peaks.size()>=2) {
            std::vector<double> gaps;
            for (size_t i=1;i<peaks.size();i++) gaps.push_back(peaks[i].pos-peaks[i-1].pos);
            std::nth_element(gaps.begin(),gaps.begin()+gaps.size()/2,gaps.end());
            spacing=gaps[gaps.size()/2];
        }
        for (size_t i=0;i<peaks.size();i++) {
            fitx.push_back(spacing>0 ? std::round((peaks[i].pos-peaks[0].pos)/spacing) : double(i));
            fity.push_back(peaks[i].pos);
            fitz.push_back(peaks[i].val);
        }

        // 4. independent estimate: period and origin from the dominant frequency of the (raw) profile
        fourier->setText(tr("Fourier: no dominant frequency"));
        if (N>=8) {
            double mean=std::accumulate(raw.begin(),raw.end(),0.0)/N;
            std::vector<double> win(N);
            for (int i=0;i<N;i++) win[i]=(raw[i]-mean)*(0.5-0.5*cos(2*M_PI*i/(N-1))); // Hann
            fftw_complex *fs=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(N/2+1));
            fftw_plan pf=fftw_plan_dft_r2c_1d(N, win.data(), fs, FFTW_ESTIMATE);
            fftw_execute(pf);
            int kmax=0; double mmax=0;
            std::vector<double> mag(N/2+1);
            for (int k=0;k<N/2+1;k++) {
                mag[k]=sqrt(fs[k][0]*fs[k][0]+fs[k][1]*fs[k][1]);
                if (k>=1 && mag[k]>mmax) { mmax=mag[k]; kmax=k; }
            }
            fftw_destroy_plan(pf);
            fftw_free(fs);
            if (kmax>=1 && mmax>0) {
                double d=0;
                if (kmax<N/2 && mag[kmax-1]>0 && mag[kmax+1]>0) { // gaussian interpolation of the peak bin
                    double l0=log(mag[kmax-1]), l1=log(mag[kmax]), l2=log(mag[kmax+1]);
                    double den=l0-2*l1+l2;
                    if (den<0) d=std::min(0.5,std::max(-0.5,0.5*(l0-l2)/den));
                }
                const double f=(kmax+d)/N; // cycles per pixel
                double re=0, im=0;
                for (int i=0;i<N;i++) { // phase of the profile at that frequency
                    double w=(raw[i]-mean)*(0.5-0.5*cos(2*M_PI*i/(N-1)));
                    re+=w*cos(2*M_PI*f*i);
                    im-=w*sin(2*M_PI*f*i);
                }
                const double period=1.0/f;
                double first=std::fmod(-atan2(im,re)/(2*M_PI)*period, period); // first maximum of cos(2 pi f n + phase)
                if (first<0) first+=period;
                fourier->setText(tr("Fourier: period %1 px, first maximum at %2 px").arg(QLocale().toString(period,'f',4)).arg(QLocale().toString(first+offset,'f',3)));
            }
        }

        std::vector<double> my_coeff(order->value()+1);

        bool res= transpose->isChecked()? polynomialfit(fity,fitx,my_coeff) : polynomialfit(fitx,fity,my_coeff);
        QString outcoeff;
        if (res) {
            for (auto &i:my_coeff) {
                outcoeff+=QString::number(i,'e',3)+"\t";
            }
        } else {
            outcoeff=tr("Not enough points for order ")+QString::number(order->value());
        }
        coeff->setText(outcoeff);

        if (fitx.size()>2) {
            double c0, c1, cov00, cov01, cov11, sum_square;
            gsl_fit_linear (&fitx[0], 1, &fity[0], 1, fitx.size(), &c0, &c1, &cov00, &cov01, &cov11, &sum_square);
            QString msg=QLocale().toString((unsigned int)(fitx.size()))+
                    " pts [c00:"+QLocale().toString(cov00)+
                    " c01:"+QLocale().toString(cov01)+
                    " c11:"+QLocale().toString(cov11)+
                    " sq:"+QLocale().toString(sqrt(sum_square)/fitx.size())+
                    "]";
            statusbar->showMessage(msg);
            origin->setText(locale().toString(c0));
            scale->setText(locale().toString(c1));
        }

        for (int i=plot->itemCount()-1; i>=0; i--) { // backwards: removing shifts the following items
            if (plot->item(i)->property(panName().toLatin1()).isValid()) plot->removeItem(plot->item(i));
        }
        points->setRowCount(0);
        for (unsigned int i=0;i<fitx.size();i++) {
            QCPItemStraightLine *marker=new QCPItemStraightLine(plot);
			marker->point1->setTypeY(QCPItemPosition::ptAbsolute);
			marker->point2->setTypeY(QCPItemPosition::ptAbsolute);
			marker->point1->setCoords(fity[i],0);
            marker->point2->setCoords(fity[i],1);
            marker->setProperty(panName().toLatin1(),true);
            marker->setPen(QPen(Qt::red));

            int pos=points->rowCount();
            points->insertRow(pos);
            QTableWidgetItem *xitem= new QTableWidgetItem(locale().toString(fity[i]));
            QTableWidgetItem *yitem= new QTableWidgetItem(locale().toString(fitz[i]));
            xitem->setTextAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
            yitem->setTextAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
            points->setItem(pos, 0, xitem);
            points->setItem(pos, 1, yitem);
            points->resizeRowToContents(pos);
        }

        plot->rescaleAxes();
        plot->replot();

    }

}

void Find_peaks::on_actionClipboard_triggered() {
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(getPoints());
    statusbar->showMessage(tr("Point values copied"),2000);
}

void Find_peaks::on_actionTxt_triggered() {
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save data"),property("NeuSave-fileTxt").toString(),tr("Text files (*.txt *.csv);;Any file (*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileTxt",fnametmp);
        QFile t(fnametmp);
        if (!t.open(QIODevice::WriteOnly| QIODevice::Text)) {
            statusbar->showMessage(tr("Cannot write ")+fnametmp+": "+t.errorString(),5000);
            return;
        }
        QTextStream out(&t);
        out << getPoints();
        t.close();
        statusbar->showMessage(tr("Export in file:")+fnametmp,2000);
    } else {
        statusbar->showMessage(tr("Export canceled"),2000);
    }
}

QString Find_peaks::getPoints() {
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

