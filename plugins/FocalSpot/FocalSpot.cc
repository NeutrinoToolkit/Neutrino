/*
 *
 *    Copyright (C) 2014 Alessandro Flacco, Tommaso Vinci All Rights Reserved
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


#include "FocalSpot.h"
#include "nPhysImageF.h"
#include "neutrino.h"
#include <algorithm>
#include <numeric>
#include <QScrollBar>

// Statistics of a focal spot: background, centre, contour at a level of the peak, energy inside it, FWHM,
// second moments (D4sigma), radial profile and encircled energy.
// The previous version (energies with the zero level clamped below the threshold, centre = image origin, text
// output) is in the git history; its energy computation is kept as "Manual (old method)".

FocalSpot::FocalSpot(neutrino *nparent) : nGenericPan(nparent)
{
    my_w.setupUi(this);
    nContour = new nLine(this,3);
    nContour->toggleClosedLine(true);

    // live update (it was only editingFinished: the arrows of the spin boxes did nothing)
    connect(my_w.zero_dsb, &QDoubleSpinBox::valueChanged, this, &FocalSpot::calculate_stats);
    connect(my_w.check_dsb, &QDoubleSpinBox::valueChanged, this, &FocalSpot::calculate_stats);
    connect(my_w.blur_radius_sb, &QDoubleSpinBox::valueChanged, this, &FocalSpot::calculate_stats);
    connect(my_w.centreMode, &QComboBox::currentIndexChanged, this, &FocalSpot::calculate_stats);
    connect(my_w.backgroundMode, &QComboBox::currentIndexChanged, this, [this]() {
        my_w.zero_dsb->setEnabled(my_w.backgroundMode->currentIndex()!=0);
        calculate_stats();
    });
    connect(nparent->my_view, &nView::mouseDoubleClickEvent_sig, this, &FocalSpot::setPosZero);
    connect(my_w.setOrigin, &QPushButton::clicked, this, &FocalSpot::setOriginToCentre);

    // radial profile (left axis) and encircled energy (right axis)
    // (a new object name: the saved style of the previous plot had one axis only)
    my_w.radialPlot->addGraph(my_w.radialPlot->xAxis, my_w.radialPlot->yAxis);
    my_w.radialPlot->graph(0)->setPen(QPen(my_w.radialPlot->yAxis->labelColor()));
    my_w.radialPlot->graph(0)->setName(tr("Radial profile"));
    my_w.radialPlot->addGraph(my_w.radialPlot->xAxis, my_w.radialPlot->yAxis2);
    my_w.radialPlot->graph(1)->setPen(QPen(my_w.radialPlot->yAxis2->labelColor()));
    my_w.radialPlot->graph(1)->setName(tr("Encircled energy"));
    my_w.radialPlot->xAxis->setLabel(tr("Radius"));
    my_w.radialPlot->yAxis->setLabel(tr("Mean value - background"));
    my_w.radialPlot->yAxis2->setLabel(tr("Encircled energy"));
    my_w.radialPlot->yAxis2->setRange(0,1.05);

    connect(my_w.radialPlot, &nCustomPlotMouseX2Y::mouseMove, this, &FocalSpot::mouseAtPlot);
    show();
    my_w.zero_dsb->setEnabled(my_w.backgroundMode->currentIndex()!=0);
    calculate_stats();
}

void FocalSpot::loadSettings(QString my_settings) {
    nGenericPan::loadSettings(my_settings);
    calculate_stats();
}

// double click on the image: the centre in Manual mode (scene coordinates: pixel i spans [i,i+1))
void FocalSpot::setPosZero(QPointF pos) {
    if (my_w.centreMode->currentIndex()==2) {
        setProperty("NeuSave-manualCentre", QPointF(pos.x()-0.5, pos.y()-0.5));
        calculate_stats();
    }
}

void FocalSpot::setOriginToCentre() {
    if (currentBuffer && std::isfinite(centre.x()) && std::isfinite(centre.y())) {
        currentBuffer->set_origin(centre);
        nparent->emitPhysChanged(currentBuffer); // Image_list, pans, view; recomputes the stats via bufferChanged
    }
}

void FocalSpot::addRow(QString group, QString sub, double value, QString unit) {
    rows << Row{group, sub, value, unit};
}

// compact view of the results: one line per group (e.g. FWHM: x y major minor), the unit once if shared
void FocalSpot::showRows() {
    const QString grey=palette().color(QPalette::PlaceholderText).name();
    auto dim=[&](QString t) { return "<span style='color:"+grey+"'>"+t.toHtmlEscaped()+"</span>"; };
    QLocale loc=locale();
    loc.setNumberOptions(QLocale::OmitGroupSeparator);
    auto num=[&](double v) { return std::isfinite(v) ? loc.toString(v,'g',5) : QString("-"); };
    QString html="<table cellspacing=0 cellpadding=1>";
    for (int k=0; k<rows.size();) {
        int e=k;
        bool sameUnit=true;
        while (e<rows.size() && rows[e].group==rows[k].group) sameUnit &= rows[e++].unit==rows[k].unit;
        QStringList items;
        for (int i=k; i<e; i++) {
            QString it;
            if (!rows[i].sub.isEmpty()) it+=dim(rows[i].sub)+"&nbsp;";
            it+=num(rows[i].value);
            if (!sameUnit && !rows[i].unit.isEmpty()) it+="&nbsp;"+dim(rows[i].unit);
            items << it;
        }
        if (sameUnit && !rows[k].unit.isEmpty()) items.last()+="&nbsp;"+dim(rows[k].unit);
        html+="<tr><td style='white-space:nowrap'><b>"+rows[k].group.toHtmlEscaped()+"</b></td><td style='padding-left:8px'>"
              +items.join(" &nbsp;&nbsp; ")+"</td></tr>";
        k=e;
    }
    html+="</table>";
    const int scroll=my_w.stats->verticalScrollBar()->value();
    my_w.stats->setHtml(html);
    my_w.stats->verticalScrollBar()->setValue(scroll);
}

QList<double> FocalSpot::legacyEnergies(double th, double zl, const physMath::contour_integral &ci, bool contourOk) {
    // previous computation: pixels between the zero level and the threshold count as the zero level, pixels below
    // the zero level as themselves (the noise around the background was rectified)
    double above_th_energy = 0, below_zero_energy = 0;
    int point_count = 0, zero_point_count = 0;
    for (size_t ii=0; ii<currentBuffer->getSurf(); ii++) {
        double zz=currentBuffer->point(ii);
        if (!std::isnan(zz)) {
            if (zz > th) {
                above_th_energy += zz;
                point_count++;
            } else if (zz < zl) {
                below_zero_energy += zz;
                zero_point_count++;
            } else {
                below_zero_energy += zl;
                zero_point_count++;
            }
        }
    }
    double background = zero_point_count ? below_zero_energy/zero_point_count : zl;
    double zero_energy_in_peak = background*point_count;
    double total_energy = currentBuffer->sum()-below_zero_energy-zero_energy_in_peak;
    above_th_energy -= zero_energy_in_peak;
    double threshold_ratio = total_energy!=0 ? 100*above_th_energy/total_energy : 0;
    double contour_ratio = (contourOk && total_energy!=0) ? 100*(ci.integral-zero_energy_in_peak)/total_energy
                                                          : std::numeric_limits<double>::quiet_NaN();
    return {threshold_ratio, contour_ratio};
}

void
FocalSpot::calculate_stats()
{
    if (!currentBuffer || computing)
        return;
    computing=true;

    const double NaN=std::numeric_limits<double>::quiet_NaN();
    rows.clear();
    radius.clear(); profile.clear(); encircled.clear();
    centre=vec2f(NaN,NaN);

    nPhysD &img=*currentBuffer;
    const int W=img.getW(), H=img.getH();
    const vec2f sc=img.get_scale();
    const double sx=std::fabs(sc.x()), sy=std::fabs(sc.y());
    const QString uL = (sx==1 && sy==1) ? QString("px") : tr("units");
    const QString uA = (sx==1 && sy==1) ? QString("px²") : tr("units²");

    // 0. blurred image (maximum, contour, FWHM)
    decimated = nPhysD(img);
    if (my_w.blur_radius_sb->value()>0) physMath::phys_fast_gaussian_blur(decimated, my_w.blur_radius_sb->value());
    decimated.TscanBrightness();

    // 1. background: the median of a frame at the border (5% of the size), or the manual value
    double bg;
    const int bgMode=my_w.backgroundMode->currentIndex();
    if (bgMode==0) {
        const int fw=std::max(1,(int)std::lround(0.05*std::min(W,H)));
        std::vector<double> border;
        for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
            if (x<fw || x>=W-fw || y<fw || y>=H-fw) {
                double v=img.point(x,y);
                if (std::isfinite(v)) border.push_back(v);
            }
        }
        if (border.empty()) {
            bg=0;
        } else {
            std::nth_element(border.begin(), border.begin()+border.size()/2, border.end());
            bg=border[border.size()/2];
        }
    } else {
        bg=my_w.zero_dsb->value();
    }
    addRow(tr("Background"), "", bg, tr("counts"));

    // 2. seed of the spot: the maximum of the blurred image, or the manual point
    vec2f seed(decimated.max_Tv.x(), decimated.max_Tv.y());
    if (my_w.centreMode->currentIndex()==2) {
        QPointF m=property("NeuSave-manualCentre").toPointF();
        if (property("NeuSave-manualCentre").isValid() && m.x()>=0 && m.y()>=0 && m.x()<=W-1 && m.y()<=H-1) {
            seed=vec2f(m.x(),m.y());
        } else {
            my_w.statusBar->showMessage(tr("Manual centre: double click on the spot"));
        }
    }
    const double peakSeed=decimated.getPoint(seed.x(),seed.y());
    if (!std::isfinite(peakSeed) || peakSeed<=bg) {
        my_w.statusBar->showMessage(tr("The spot is not above the background"));
        nContour->hide();
        showRows();
        my_w.radialPlot->graph(0)->data()->clear();
        my_w.radialPlot->graph(1)->data()->clear();
        my_w.radialPlot->replot();
        my_w.centreInfo->setText("--");
        computing=false;
        return;
    }

    // 3. contour at the level (% of the peak above the background), traced around the seed
    const double level=my_w.check_dsb->value()/100.;
    const double th=bg+level*(peakSeed-bg);
    std::list<vec2f> contour;
    decimated.set_origin(vec2i(std::lround(seed.x()),std::lround(seed.y()))); // contour_trace starts there
    physMath::contour_trace(decimated, contour, th);
    const bool contourOk=contour.size()>2;

    QPolygonF myp;
    double area=NaN, rmin=NaN, rmax=NaN;
    vec2f contourCentre(NaN,NaN);
    physMath::contour_integral ci;
    if (contourOk) {
        for (auto& itr : contour) myp<<QPointF(itr.x(), itr.y());
        // area and centroid of the enclosed polygon (the vertices of a marching-squares line are unevenly spaced)
        double a2=0, gx=0, gy=0;
        for (int k=0;k<myp.size();k++) {
            const QPointF &p=myp[k], &q=myp[(k+1)%myp.size()];
            double cr=p.x()*q.y()-q.x()*p.y();
            a2+=cr; gx+=(p.x()+q.x())*cr; gy+=(p.y()+q.y())*cr;
        }
        if (a2!=0) {
            contourCentre=vec2f(gx/(3*a2), gy/(3*a2));
        } else {
            contourCentre=vec2f(0,0);
            for (auto& itr : contour) contourCentre+=itr;
            contourCentre/=contour.size();
        }
        area=std::fabs(a2)/2*sx*sy;
        rmin=std::numeric_limits<double>::max(); rmax=0;
        for (auto &itr : contour) {
            double dd=vec2f((itr.x()-contourCentre.x())*sx,(itr.y()-contourCentre.y())*sy).mod();
            rmin=std::min(rmin,dd); rmax=std::max(rmax,dd);
        }
        // pixels whose centre is inside the contour
        ci = physMath::contour_integrate(img, contour, false);
    }

    // 4. centre
    centre = (my_w.centreMode->currentIndex()==1 && contourOk) ? contourCentre : seed;
    my_w.centreInfo->setText(QString("%1 : %2 px").arg(locale().toString(centre.x(),'f',2), locale().toString(centre.y(),'f',2)));
    addRow(tr("Centre"), "x", centre.x(), "px");
    addRow(tr("Centre"), "y", centre.y(), "px");
    const double peakC=decimated.getPoint(centre.x(),centre.y());

    // 5. energies above the background (all the pixels, negative ones included: the noise averages out)
    double Etot=0, maxPix=-std::numeric_limits<double>::max();
    for (size_t ii=0; ii<img.getSurf(); ii++) {
        double v=img.point(ii);
        if (std::isfinite(v)) { Etot+=v-bg; maxPix=std::max(maxPix,v); }
    }
    addRow(tr("Peak"), tr("blurred"), peakC-bg, tr("counts"));
    addRow(tr("Peak"), tr("pixel/total"), Etot!=0 ? 100*(maxPix-bg)/Etot : NaN, "%");
    addRow(tr("Energy"), tr("total"), Etot, tr("counts"));
    addRow(tr("Energy"), tr("in contour"), (contourOk && Etot!=0) ? 100*(ci.integral-bg*ci.count)/Etot : NaN, "%");
    if (bgMode==2) {
        QList<double> old=legacyEnergies(th, bg, ci, contourOk);
        addRow(tr("Old method"), tr("threshold"), old[0], "%");
        addRow(tr("Old method"), tr("contour"), old[1], "%");
    }
    addRow(tr("Contour"), tr("level"), th, tr("counts"));
    addRow(tr("Contour"), tr("area"), area, uA);
    addRow(tr("Contour radius"), tr("√(area/π)"), contourOk ? std::sqrt(area/M_PI) : NaN, uL);
    addRow(tr("Contour radius"), tr("min"), rmin, uL);
    addRow(tr("Contour radius"), tr("max"), rmax, uL);

    // 6. second moments (ISO 11146): iterated in a rectangle of 3 times the D4sigma widths around the centroid
    double mcx=centre.x(), mcy=centre.y();
    double hx=std::max(W,H)/2., hy=hx;
    if (contourOk) hx=hy=3*std::max(rmax/std::min(sx,sy),2.0);
    double sxx=NaN, syy=NaN, sxy=NaN;
    for (int iter=0; iter<30; iter++) {
        const int i0=std::max(0,(int)std::floor(mcx-hx)), i1=std::min(W-1,(int)std::ceil(mcx+hx));
        const int j0=std::max(0,(int)std::floor(mcy-hy)), j1=std::min(H-1,(int)std::ceil(mcy+hy));
        double m0=0, mx=0, my=0;
        for (int j=j0;j<=j1;j++) for (int i=i0;i<=i1;i++) {
            double v=img.point(i,j);
            if (std::isfinite(v)) { v-=bg; m0+=v; mx+=v*(i-mcx); my+=v*(j-mcy); }
        }
        if (m0<=0) { sxx=syy=sxy=NaN; break; }
        const double ncx=mcx+mx/m0, ncy=mcy+my/m0;
        double vxx=0, vyy=0, vxy=0;
        for (int j=j0;j<=j1;j++) for (int i=i0;i<=i1;i++) {
            double v=img.point(i,j);
            if (std::isfinite(v)) { v-=bg; vxx+=v*(i-ncx)*(i-ncx); vyy+=v*(j-ncy)*(j-ncy); vxy+=v*(i-ncx)*(j-ncy); }
        }
        sxx=vxx/m0; syy=vyy/m0; sxy=vxy/m0;
        mcx=ncx; mcy=ncy;
        if (sxx<=0 || syy<=0) { sxx=syy=sxy=NaN; break; }
        const double nhx=1.5*4*std::sqrt(sxx), nhy=1.5*4*std::sqrt(syy);
        const bool converged=std::fabs(nhx-hx)<0.01*hx && std::fabs(nhy-hy)<0.01*hy;
        hx=nhx; hy=nhy;
        if (converged) break;
    }
    addRow(tr("Centroid"), "x", std::isfinite(sxx) ? mcx : NaN, "px");
    addRow(tr("Centroid"), "y", std::isfinite(sxx) ? mcy : NaN, "px");
    // in world units
    const double wxx=sxx*sx*sx, wyy=syy*sy*sy, wxy=sxy*sx*sy;
    const double dd=std::sqrt(std::pow((wxx-wyy)/2,2)+wxy*wxy);
    const double wMaj=(wxx+wyy)/2+dd, wMin=(wxx+wyy)/2-dd;
    const double angle=0.5*std::atan2(2*wxy, wxx-wyy); // of the major axis, from x toward y (down on the image)
    addRow(tr("D4σ"), "x", 4*std::sqrt(wxx), uL);
    addRow(tr("D4σ"), "y", 4*std::sqrt(wyy), uL);
    addRow(tr("D4σ"), tr("major"), 4*std::sqrt(wMaj), uL);
    addRow(tr("D4σ"), tr("minor"), wMin>0 ? 4*std::sqrt(wMin) : NaN, uL);

    // 7. FWHM of the blurred image through the centre: along x, y and the axes of the moments
    auto fwhm=[&](double ux, double uy) -> double { // direction in world units
        if (!std::isfinite(peakC) || peakC<=bg || !std::isfinite(ux) || !std::isfinite(uy)) return NaN;
        const double half=bg+0.5*(peakC-bg);
        const double px=ux/sx, py=uy/sy;      // pixels per world unit along the direction
        const double ds=0.25*std::min(sx,sy); // step in world units (a quarter of pixel)
        const double dmax=std::hypot(W*sx,H*sy);
        double width=0;
        for (int sgn : {-1,1}) {
            double prevv=peakC, prevd=0;
            bool found=false;
            for (double d=ds; d<dmax; d+=ds) {
                const double x=centre.x()+sgn*d*px, y=centre.y()+sgn*d*py;
                if (x<0 || y<0 || x>W-1 || y>H-1) return NaN;
                const double v=decimated.getPoint(x,y);
                if (!std::isfinite(v)) return NaN;
                if (v<half) {
                    width+=prevd+(prevv-half)/(prevv-v)*(d-prevd);
                    found=true;
                    break;
                }
                prevv=v; prevd=d;
            }
            if (!found) return NaN;
        }
        return width;
    };
    const double fMaj=fwhm(std::cos(angle),std::sin(angle)), fMin=fwhm(-std::sin(angle),std::cos(angle));
    const double fx=fwhm(1,0), fy=fwhm(0,1);
    addRow(tr("FWHM"), "x", fx, uL);
    addRow(tr("FWHM"), "y", fy, uL);
    addRow(tr("FWHM"), tr("major"), fMaj, uL);
    addRow(tr("FWHM"), tr("minor"), fMin, uL);
    addRow(tr("Major axis"), tr("angle"), angle*180/M_PI, "˚");
    addRow(tr("Major axis"), tr("ellipticity"), fMin/fMaj, ""); // FWHM minor/major

    // 8. radial profile and encircled energy around the centre, up to 4 FWHM (or the farthest corner)
    double rCorner=0;
    for (double cx : {0.0, W-1.0}) for (double cy : {0.0, H-1.0}) {
        rCorner=std::max(rCorner, vec2f((cx-centre.x())*sx,(cy-centre.y())*sy).mod());
    }
    const double rlim = std::isfinite(fMaj) ? std::min(rCorner, 4*fMaj) : rCorner;
    double r50=NaN, r86=NaN;
    if (rlim>0) {
        const int nb=std::clamp((int)std::ceil(rlim/(0.5*std::min(sx,sy))),10,2000);
        const double dr=rlim/nb;
        QVector<double> sum(nb,0.0), num(nb,0.0);
        for (int j=0;j<H;j++) for (int i=0;i<W;i++) {
            double v=img.point(i,j);
            if (std::isfinite(v)) {
                double rr=vec2f((i-centre.x())*sx,(j-centre.y())*sy).mod();
                int b=(int)(rr/dr);
                if (b<nb) { sum[b]+=v-bg; num[b]++; }
            }
        }
        double cumul=0;
        for (int b=0;b<nb;b++) {
            radius << (b+0.5)*dr;
            profile << (num[b] ? sum[b]/num[b] : NaN);
            cumul+=sum[b];
            encircled << (Etot!=0 ? cumul/Etot : NaN);
        }
        auto crossing=[&](double f) -> double { // radius of the first crossing of f (outer edge of the bins)
            double prevR=0, prevE=0;
            for (int b=0;b<nb;b++) {
                double r=(b+1)*dr, e=encircled[b];
                if (e>=f) return prevR+(f-prevE)/(e-prevE)*(r-prevR);
                prevR=r; prevE=e;
            }
            return NaN;
        };
        r50=crossing(0.5); r86=crossing(0.865);
    }
    addRow(tr("Encircled radius"), "50%", r50, uL);
    addRow(tr("Encircled radius"), "86.5%", r86, uL); // 1/e² radius of a gaussian

    showRows();

    my_w.radialPlot->graph(0)->setData(radius,profile,true);
    my_w.radialPlot->graph(1)->setData(radius,encircled,true);
    my_w.radialPlot->xAxis->setLabel(tr("Radius [%1]").arg(uL));
    my_w.radialPlot->graph(0)->rescaleAxes();
    my_w.radialPlot->yAxis2->setRange(0,1.05);
    my_w.radialPlot->replot();

    if (contourOk) {
        // setPoints takes scene coordinates and maps them to the line, which sits at the image origin: put it there
        // first (its own bufferChanged comes after this one, the line was still at the origin of the previous image).
        // The contour is in pixel coordinates, pixel (i,j) has its centre at (i+0.5,j+0.5) in the scene
        nContour->setPos(currentBuffer->get_origin().x(), currentBuffer->get_origin().y());
        nContour->setPoints(myp.translated(0.5,0.5));
        nContour->show();
        my_w.statusBar->showMessage(tr("Contour ok"));
    } else {
        my_w.statusBar->showMessage(tr("Contour not ok"));
        nContour->hide(); // don't show the contour of another image/threshold
    }

    computing=false;
}

QString FocalSpot::resultsText() {
    QString out;
    out += QString("# FocalSpot: %1\n").arg(currentBuffer ? QString::fromStdString(currentBuffer->getName()) : QString());
    for (auto &r : rows) {
        out += QString("# %1\t%2\t%3\n").arg(r.name(), QLocale().toString(r.value,'g',8), r.unit);
    }
    out += "# radius\tprofile\tencircled\n";
    for (int i=0;i<radius.size();i++) {
        out += QLocale().toString(radius[i],'g',8)+"\t"+QLocale().toString(profile[i],'g',8)+"\t"+QLocale().toString(encircled[i],'g',8)+"\n";
    }
    return out;
}

void FocalSpot::on_actionClipboard_triggered() {
    QApplication::clipboard()->setText(resultsText());
    my_w.statusBar->showMessage(tr("Results copied to the clipboard"));
}

void FocalSpot::on_actionTxt_triggered() {
    QString fname=QFileDialog::getSaveFileName(this,tr("Save data"),property("NeuSave-fileTxt").toString(),tr("Text files (*.txt *.csv);;Any file (*)"));
    if (fname.isEmpty()) {
        my_w.statusBar->showMessage(tr("Export canceled"));
        return;
    }
    setProperty("NeuSave-fileTxt",fname);
    QFile t(fname);
    if (!t.open(QIODevice::WriteOnly | QIODevice::Text)) {
        my_w.statusBar->showMessage(tr("Cannot write ")+fname+": "+t.errorString());
        return;
    }
    QTextStream(&t) << resultsText();
    my_w.statusBar->showMessage(tr("Exported to ")+fname);
}

QVariantMap FocalSpot::results(QVariant) {
    QVariantMap map;
    for (auto &r : rows) map[r.name()]=r.value;
    map["radius"]=QVariant::fromValue(QList<double>(radius));
    map["profile"]=QVariant::fromValue(QList<double>(profile));
    map["encircled"]=QVariant::fromValue(QList<double>(encircled));
    return map;
}

void FocalSpot::mouseAtWorld(QPointF p){
    double trueLength = std::sqrt(std::pow(p.x(), 2) + std::pow(p.y(), 2));
    if (currentBuffer && std::isfinite(centre.x())) { // world -> pixel -> distance from the spot centre
        vec2f sc=currentBuffer->get_scale(), og=currentBuffer->get_origin();
        trueLength = vec2f((p.x()/sc.x()+og.x()-centre.x())*sc.x(), (p.y()/sc.y()+og.y()-centre.y())*sc.y()).mod();
    }
    my_w.radialPlot->setMousePosition(trueLength);
    my_w.statusBar->showMessage(tr("Distance to the centre: ")+QLocale().toString(trueLength));
}

void
FocalSpot::bufferChanged(nPhysD *buf)
{
    nGenericPan::bufferChanged(buf);
    calculate_stats();
}

void FocalSpot::mouseAtPlot(QMouseEvent* e) {
    QString msg;
    QTextStream(&msg) << my_w.radialPlot->xAxis->pixelToCoord(e->pos().x()) << ","
                      << my_w.radialPlot->yAxis->pixelToCoord(e->pos().y());

    my_w.statusBar->showMessage(msg);
}
