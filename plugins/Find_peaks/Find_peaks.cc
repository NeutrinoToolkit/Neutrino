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

    connect(setOrigin, &QPushButton::clicked, this, &Find_peaks::set_origin);
    connect(setScale, &QPushButton::clicked, this, &Find_peaks::set_scale);
    connect(calibrateBoth, &QPushButton::clicked, this, &Find_peaks::calibrate);
    connect(setScalePoly, &QPushButton::clicked, this, &Find_peaks::set_scale_poly);
    connect(spacingWorld, &QLineEdit::textChanged, this, &Find_peaks::updateCalibration);
    connect(firstPeakWorld, &QLineEdit::textChanged, this, &Find_peaks::updateCalibration);

    connect(actionRect, &QAction::triggered, box, &nRect::togglePadella);

    toolBar->addWidget(direction);
    toolBar->addWidget(param);
    toolBar->addWidget(prominence);


    show();

    plot->addGraph(plot->xAxis, plot->yAxis);
    plot->graph(0)->setName("Find_peaks");

    plot->addGraph(plot->xAxis, plot->yAxis);
    plot->graph(1)->setName("Blurred");
    QPen p=plot->graph(1)->pen();
    p.setStyle(Qt::DotLine);
    plot->graph(1)->setPen(p);

    connect(nparent, &neutrino::bufferChanged, this, &Find_peaks::updatePlot);
    connect(box, &nRect::sceneChanged, this, &Find_peaks::updatePlot);
    connect(direction, &QToolButton::toggled, this, &Find_peaks::updatePlot);
    connect(param, &QDoubleSpinBox::valueChanged, this, &Find_peaks::updatePlot);
    connect(order, &QSpinBox::valueChanged, this, &Find_peaks::updatePlot);
    connect(prominence, &QDoubleSpinBox::valueChanged, this, &Find_peaks::updatePlot);
    connect(transpose, &QCheckBox::toggled, this, &Find_peaks::updatePlot);
    updatePlot();
}


// scale [world units/px] from the peak spacing in world units and the fitted spacing [px]; origin [px] such that
// the first peak is at its world value with that scale
bool Find_peaks::calibration(double &my_scale, double &my_origin) {
    bool ok1, ok2;
    const double spacing=locale().toDouble(spacingWorld->text(),&ok1);
    const double first=locale().toDouble(firstPeakWorld->text(),&ok2);
    if (!ok1 || !ok2 || spacing==0 || !std::isfinite(fitSpacing) || !std::isfinite(fitFirst) || fitSpacing==0) return false;
    my_scale=spacing/fitSpacing;
    my_origin=fitFirst-first/my_scale;
    return std::isfinite(my_scale) && std::isfinite(my_origin);
}

void Find_peaks::updateCalibration() {
    if (!std::isfinite(fitSpacing) || !std::isfinite(fitFirst)) {
        scaleInfo->setText(tr("(not enough peaks)"));
        originInfo->clear();
        return;
    }
    double my_scale, my_origin;
    if (calibration(my_scale, my_origin)) {
        scaleInfo->setText(tr("fit %1 px → scale %2/px").arg(locale().toString(fitSpacing,'g',6), locale().toString(my_scale,'g',6)));
        originInfo->setText(tr("fit %1 px → origin %2 px").arg(locale().toString(fitFirst,'g',6), locale().toString(my_origin,'g',6)));
    } else {
        scaleInfo->setText(tr("fit %1 px").arg(locale().toString(fitSpacing,'g',6)));
        originInfo->setText(tr("fit %1 px").arg(locale().toString(fitFirst,'g',6)));
    }
}

// (previous version: origin = fitted position of the first peak minus an offset in pixels, and scale = the
// multiplier divided by the fitted spacing)
void Find_peaks::set_origin() {
    double my_scale, my_origin;
    if (currentBuffer && calibration(my_scale, my_origin)) {
        // the objects follow the origin of the image: the region goes back on the same pixels
        const QRectF region=box->getRectF();
        const phys_direction d = direction->isChecked() ? PHYS_Y : PHYS_X;
        if (currentBuffer->has_scale_poly(d) && !peakPositions.empty()) {
            // non-linear axis: the values are shifted through c0 (the value at the origin), the origin stays.
            // (moving the origin shifts them by P(o) - P(o'), which cannot reach every value on a curved axis)
            std::vector<double> c=currentBuffer->get_scale_poly(d);
            // the measured first peak: the linear fit is biased on a curved axis
            c[0]+=locale().toDouble(firstPeakWorld->text())-currentBuffer->to_world(d,peakPositions.front());
            currentBuffer->set_scale_poly(d,c);
            nparent->emitPhysChanged(currentBuffer);
            return;
        }
        if (direction->isChecked()) {
            currentBuffer->set_origin(currentBuffer->get_origin().x(),my_origin);
        } else {
            currentBuffer->set_origin(my_origin,currentBuffer->get_origin().y());
        }
        nparent->emitPhysChanged(currentBuffer);
        box->setRect(region);
    } else {
        statusbar->showMessage(tr("No calibration: not enough peaks or invalid world values"));
    }
}

void Find_peaks::set_scale() {
    double my_scale, my_origin;
    if (currentBuffer && calibration(my_scale, my_origin)) {
        currentBuffer->set_scale_poly(direction->isChecked() ? PHYS_Y : PHYS_X, {}); // a linear scale replaces the non-linear one
        if (direction->isChecked()) {
            currentBuffer->set_scale(currentBuffer->get_scale().x(),my_scale);
        } else {
            currentBuffer->set_scale(my_scale,currentBuffer->get_scale().y());
        }
        nparent->emitPhysChanged(currentBuffer);
    } else {
        statusbar->showMessage(tr("No calibration: not enough peaks or invalid world values"));
    }
}

void Find_peaks::calibrate() {
    set_scale();
    set_origin();
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

// non-linear scale: the peak number n is fitted as a polynomial of the position p [px] (order of the fit line),
// and the world coordinate is P(p) = first + spacing n(p), an absolute calibration: the origin is kept and c0 is
// chosen so that the world value is P(p) whatever the origin (to_world is P(p) - P(o) + c0)
void Find_peaks::set_scale_poly() {
    bool ok1, ok2;
    const double spacing=locale().toDouble(spacingWorld->text(),&ok1);
    const double first=locale().toDouble(firstPeakWorld->text(),&ok2);
    if (!currentBuffer || !ok1 || !ok2 || spacing==0) {
        statusbar->showMessage(tr("No calibration: invalid world values"));
        return;
    }
    if (order->value()<1) {
        statusbar->showMessage(tr("The order of the fit must be at least 1"));
        return;
    }
    // positions scaled to ~1 for the conditioning of the fit, then the coefficients scaled back
    double L=0;
    for (auto &p : peakPositions) L=std::max(L,std::abs(p));
    std::vector<double> u(peakPositions.size()), c(order->value()+1);
    for (size_t i=0;i<u.size();i++) u[i]=peakPositions[i]/L;
    if (L==0 || !polynomialfit(u,peakNumbers,c)) {
        statusbar->showMessage(tr("Not enough peaks for order ")+QString::number(order->value()));
        return;
    }
    for (size_t k=0;k<c.size();k++) c[k]*=spacing/pow(L,k);
    c[0]+=first;

    const phys_direction d = direction->isChecked() ? PHYS_Y : PHYS_X;
    const double o = d==PHYS_X ? currentBuffer->get_origin().x() : currentBuffer->get_origin().y();
    c[0]=physPoly::eval(c,o);
    const QRectF region=box->getRectF();
    if (!currentBuffer->set_scale_poly(d,c)) {
        statusbar->showMessage(tr("The fit is not monotonic over the image: lower the order"));
        return;
    }
    nparent->emitPhysChanged(currentBuffer);
    box->setRect(region); // the region stays on the same pixels (order 1 moves the origin)
    if (currentBuffer->has_scale_poly(d)) {
        statusbar->showMessage(tr("Non-linear scale %1: %2").arg(d==PHYS_X ? "x" : "y", QString::fromStdString(physPoly::str(currentBuffer->get_scale_poly(d)))));
    } else { // order 1: linear
        statusbar->showMessage(tr("Linear scale %1: %2/px, origin %3 px").arg(d==PHYS_X ? "x" : "y",
                               locale().toString(currentBuffer->get_scale(d),'g',6),
                               locale().toString(d==PHYS_X ? currentBuffer->get_origin().x() : currentBuffer->get_origin().y(),'g',6)));
    }
}

void Find_peaks::updatePlot() {
    if (!isVisible()) return;
    QRect geom2;
    if (currentBuffer) {
        geom2=box->getRect(currentBuffer);
        if (geom2.isEmpty()) statusbar->showMessage(tr("Attention: the region is outside the image!"));
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
        fitFirst=fitSpacing=std::numeric_limits<double>::quiet_NaN();
        peakNumbers.clear();
        peakPositions.clear();
        updateCalibration();
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
        // gaussian of sigma = param [px]: its transform is exp(-2 pi^2 sigma^2 f^2), f=i/M (sigma 0: no blur)
        // double sx=pow(M/param->value(),2)/2.0; // (this was a sigma of param/pi)
        double sx=pow(M/(M_PI*param->value()),2)/2.0;
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

        // 3. peak numbers, incrementally: each gap in units of the local spacing (the previous gap per step,
        //    the median gap for the first one). A missing peak leaves a gap instead of shifting all the following
        //    ones, and a spacing that changes along the profile (e.g. the dispersion of a spectrometer) is followed
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
        // (previous version: the distance to the first peak in units of the median gap, which was wrong as soon as
        // the accumulated change of the spacing was more than half a spacing)
        // for (size_t i=0;i<peaks.size();i++) {
        //     fitx.push_back(spacing>0 ? std::round((peaks[i].pos-peaks[0].pos)/spacing) : double(i));
        double number=0, step=spacing;
        for (size_t i=0;i<peaks.size();i++) {
            if (i>0) {
                const double gap=peaks[i].pos-peaks[i-1].pos;
                const double n = step>0 ? std::max(1.0,std::round(gap/step)) : 1.0;
                number+=n;
                step=gap/n;
            }
            fitx.push_back(number);
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

        peakNumbers=fitx;
        peakPositions=fity;
        fitFirst=fitSpacing=std::numeric_limits<double>::quiet_NaN();
        if (fitx.size()>=2) {
            double c0, c1, cov00, cov01, cov11, sum_square;
            gsl_fit_linear (&fitx[0], 1, &fity[0], 1, fitx.size(), &c0, &c1, &cov00, &cov01, &cov11, &sum_square);
            QString msg=QLocale().toString((unsigned int)(fitx.size()))+
                    " pts [c00:"+QLocale().toString(cov00)+
                    " c01:"+QLocale().toString(cov01)+
                    " c11:"+QLocale().toString(cov11)+
                    " sq:"+QLocale().toString(sqrt(sum_square)/fitx.size())+
                    "]";
            statusbar->showMessage(msg);
            fitFirst=c0;
            fitSpacing=c1;
        }
        updateCalibration();

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
    statusbar->showMessage(tr("Point values copied"));
}

void Find_peaks::on_actionTxt_triggered() {
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save data"),property("NeuSave-fileTxt").toString(),tr("Text files (*.txt *.csv);;Any file (*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileTxt",fnametmp);
        QFile t(fnametmp);
        if (!t.open(QIODevice::WriteOnly| QIODevice::Text)) {
            statusbar->showMessage(tr("Cannot write ")+fnametmp+": "+t.errorString());
            return;
        }
        QTextStream out(&t);
        out << getPoints();
        t.close();
        statusbar->showMessage(tr("Export in file:")+fnametmp);
    } else {
        statusbar->showMessage(tr("Export canceled"));
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

