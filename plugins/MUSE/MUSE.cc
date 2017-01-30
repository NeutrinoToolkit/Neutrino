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
 *    You should have received a co py of the GNU Lesser General Public License
 *    along with neutrino.  If not, see <http://www.gnu.org/licenses/>.
 *
 *    Contact Information:
 *	Alessandro Flacco <alessandro.flacco@polytechnique.edu>
 *	Tommaso Vinci <tommaso.vinci@polytechnique.edu>
 *
 */
#include "MUSE.h"
#include <QtGui>

#include "nPhysFormats.h"
#include "fitsio.h"

#define HDF5_MAX_NAME 2048

MUSE::MUSE(neutrino *nparent) : nGenericPan(nparent),
    cubeSlice(nullptr)
{
    setupUi(this);

    connect(actionOpen, SIGNAL(triggered()), this, SLOT(loadCube()));
    connect(slicesSlider,SIGNAL(sliderMoved(int)),slices,SLOT(setValue(int)));
    connect(radius,SIGNAL(valueChanged(int)),this,SLOT(updateLastPoint()));

    connect(plot,SIGNAL(mouseDoubleClick(QMouseEvent*)), this, SLOT(plotClick(QMouseEvent*)));

    setProperty("NeuSave-fileMUSE","myfile.fits");
    plot->addGraph(plot->xAxis, plot->yAxis);
    plot->addGraph(plot->xAxis, plot->yAxis);

    toolBar->addWidget(radius);

    show();
    on_actionMode_toggled();

}

void MUSE::plotClick(QMouseEvent* e) {
    int nslice=xvals.size()*(plot->xAxis->pixelToCoord(e->pos().x())-wavelen.first())/(wavelen.second()-wavelen.first());
    slices->setValue(nslice);
}


void MUSE::updateLastPoint() {
    doSpectrum(nparent->my_mouse.pos());
}

void MUSE::doSpectrum(QPointF point) {
    if (cubesize.size()==3 && point.x()>0 && point.y()>0 &&  point.x()*point.y() < cubesize[0]*cubesize[1]) {
        vec2 p(point.x(),point.y());
        for (int zz=0; zz< yvals.size(); zz++) {
            yvals[zz]=0;
        }

        int surf=cubesize[0]*cubesize[1];

#pragma omp parallel for collapse(3)
        for (int xx=std::max((int)0,p.x()-radius->value());xx<=std::min((int)(cubesize[0]),p.x()+radius->value()); xx++) {
            for (int yy=std::max((int)0,p.y()-radius->value());yy<=std::min((int)(cubesize[1]),p.y()+radius->value()); yy++) {
                for (unsigned int zz=0; zz< cubesize[2]; zz++) {
                    yvals[zz]+=cubevect[zz*surf + xx+yy*cubesize[0]];
                }
            }
        }
        for (unsigned int zz=0; zz< cubesize[2]; zz++) {
            yvals[zz]=yvals[zz]/(1+2*radius->value()*radius->value());
        }
        plot->setTitle(QString::number(p.x())+" " +QString::number(p.y()));
        plot->graph(1)->setData(xvals,yvals,true);

        plot->replot();

    }
}

void MUSE::on_slices_valueChanged(int i) {
    slicesSlider->setValue(i);
    if (cubesize.size()==3 && i < (int)cubesize[2]) {
        nPhysD *my_phys=new nPhysD(cubesize[0],cubesize[1],0.0,QString::number(i).toStdString());
        my_phys->property=cube_prop;
#pragma omp parallel for
        for (unsigned int k=0; k < my_phys->getSurf(); k++) {
            my_phys->set(k,cubevect[i*my_phys->getSurf()+k]);
        }
        my_phys->TscanBrightness();
        cubeSlice=nparent->replacePhys(my_phys,cubeSlice);
        plot->setMousePosition(xvals[i]);
    }
}

void MUSE::on_actionMode_toggled() {
    if (actionMode->isChecked()) {
        disconnect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(doSpectrum(QPointF)));
        connect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(doSpectrum(QPointF)));
    } else {
        disconnect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(doSpectrum(QPointF)));
        connect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(doSpectrum(QPointF)));
    }
}

void MUSE::loadCube() {
    QString ifilename=QFileDialog::getOpenFileName(this,tr("Open MUSE file"),property("NeuSave-fileMUSE").toString(),tr("MUSE Cube")+QString(" (*.fits);;")+tr("Any files")+QString(" (*)"));

    if (!ifilename.isEmpty()) {
        setProperty("NeuSave-fileMUSE", ifilename);


        fitsfile *fptr;
        char card[FLEN_CARD];
        int status = 0, ii;

        fits_open_file(&fptr, ifilename.toLatin1().data(), READONLY, &status);
        int bitpix;
        int anaxis;

        fits_is_compressed_image(fptr, &status);
        if (fits_check_error(status)) return;

        int hdupos=0;
        fits_get_hdu_num(fptr, &hdupos);
        if (fits_check_error(status)) return;


        bool wavelfound=false;
        wavelen=vec2f(0,1);

        for (; !status; hdupos++)  {


            int hdutype;
            fits_get_hdu_type(fptr, &hdutype, &status);
            if (fits_check_error(status)) return;



            // 		if (hdutype == IMAGE_HDU) {
            // 			long naxes[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
            // 			for (ii = 0; ii < 9; ii++)
            // 				naxes[ii] = 1;
            // 			  int naxis = 0;
            // 			fits_get_img_param(fptr, 9, &bitpix, &naxis, naxes, &status);
            //
            // 			long totpix = naxes[0] * naxes[1] * naxes[2] * naxes[3] * naxes[4] * naxes[5] * naxes[6] * naxes[7] * naxes[8];
            // // 			DEBUG("totpix " << totpix);
            // 		}

            fits_get_img_type(fptr,&bitpix,&status);

            fits_get_img_dim(fptr,&anaxis,&status);

            int nkeys;
            fits_get_hdrspace(fptr, &nkeys, NULL, &status);

            for (ii = 1; ii <= nkeys; ii++)  {
                fits_read_record(fptr, ii, card, &status);
                if (fits_check_error(status)) return;

                std::string cardStr(card);
                QStringList pippo(QString(card).split("=",QString::SkipEmptyParts));
                if (pippo.size()>1) {
                    if(wavelfound==false && (pippo.first()=="WAVELMIN" || pippo.first()=="WAVELMAX")) {
                        QStringList pippo2(pippo.at(1).split("/",QString::SkipEmptyParts));
                        qDebug() << pippo2;
                        if (pippo2.size()>1) {
                            bool ok;
                            double val=pippo2.first().toDouble(&ok);
                            if (ok) {
                                if(pippo.first()=="WAVELMIN") {
                                    wavelen.set_first(val);
                                } else {
                                    wavelen.set_second(val);
                                }
                            }
                        }
                    } else {
                        cube_prop["fits-"+pippo.first().toStdString()]=pippo.at(1).toStdString();
                    }
                }
                std::stringstream ss; ss << std::setw(log10(nkeys)+1) << std::setfill('0') << ii;
            }

            if (wavelen!=vec2f(0,1)) {
                wavelfound=true;
            }

            DEBUG("here " << wavelen);

            std::vector<long> axissize(anaxis,0),fpixel(anaxis,1);

            fits_get_img_size(fptr,anaxis,&axissize[0],&status);
            if (fits_check_error(status)) return;

            long totalsize=1;
            for(int i=0; i<anaxis; i++) {
                totalsize*=axissize[i];
            }
            DEBUG("totalsize " << totalsize);

            if (anaxis==3) {
                cubevect.resize(totalsize);
                cubesize.resize(anaxis);
                fits_read_pix(fptr, TDOUBLE, &fpixel[0], totalsize, NULL, (void *)&cubevect[0], NULL, &status);
                DEBUG("got a cube : " << totalsize << " = " << axissize[0] << " x " << axissize[1] << " x " << axissize[2]);
                for(int i=0; i<anaxis; i++) {
                    cubesize[i]=axissize[i];
                }

                xvals.resize(cubesize[2]);
                yvals.resize(cubesize[2]);
                ymean.resize(cubesize[2]);
                for (int zz=0; zz< xvals.size(); zz++) {
                    xvals[zz]=wavelen.first()+zz*(wavelen.second()-wavelen.first())/xvals.size();
                    qDebug() << zz << " " << xvals[zz];
                    ymean[zz]=0;
                }
                int surf=cubesize[0]*cubesize[1];
#pragma omp parallel for
                for (unsigned int kk=0;kk<totalsize; kk++) {
                    if (std::isfinite(cubevect[kk]))
                        ymean[kk/surf]+=cubevect[kk];
                }

                for (int xx=0; xx< xvals.size(); xx++) {
                    ymean[xx]/=surf;
                }
                plot->graph(0)->setData(xvals,ymean,true);
                QPen my_pen=plot->graph(0)->pen();
                my_pen.setColor(Qt::gray);
                plot->graph(0)->setPen(my_pen);
                plot->rescaleAxes();
                plot->replot();

                slices->setMaximum(axissize[2]);
                slicesSlider->setMaximum(axissize[2]);

                return;

            }

            fits_movrel_hdu(fptr, 1, NULL, &status);  /* try to move to next HDU */

            if (status == END_OF_FILE) {
                status=0;
                break;
            }

            if (fits_check_error(status)) {
                return;
            }
        }
        fits_close_file(fptr, &status);
        fits_check_error(status);

        on_slices_valueChanged(slices->value());
        DEBUG("out of here");
    }
}
