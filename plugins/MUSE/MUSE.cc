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
    my_offset(0,0),
    my_offset_val(0,0),
    my_scale(1,1),
    cubeSlice(nullptr),
    meanSlice(nullptr)
{
    setupUi(this);

    connect(horizontalScrollBar, SIGNAL(valueChanged(int)), this, SLOT(horzScrollBarChanged(int)));
    connect(plot->xAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(xAxisChanged(QCPRange)));

    connect(radius,SIGNAL(valueChanged(int)),this,SLOT(updateLastPoint()));

    connect(slices,SIGNAL(valueChanged(int)),this,SLOT(showImagePlane(int)));
    connect(slicesSlider,SIGNAL(valueChanged(int)),this,SLOT(showImagePlane(int)));

    connect(plot,SIGNAL(mouseDoubleClick(QMouseEvent*)), this, SLOT(plotClick(QMouseEvent*)));

    connect(nparent,SIGNAL(mouseAtWorld(QPointF)), this, SLOT(setSpectrumTitle(QPointF)));

    setProperty("NeuSave-fileMUSE","myfile.fits");
    plot->addGraph(plot->xAxis, plot->yAxis2);
    plot->addGraph(plot->xAxis, plot->yAxis);

    toolBar->addWidget(radiusLambda);
    toolBar->addWidget(radius);

    show();
    on_actionMode_toggled();

    loadCube();
}

void MUSE::horzScrollBarChanged(int value)
{
    if (qAbs(plot->xAxis->range().center()) > 0.01)
    {
        plot->xAxis->setRange(value, plot->xAxis->range().size(), Qt::AlignCenter);
        plot->replot();
    }
}

void MUSE::xAxisChanged(QCPRange range)
{
    horizontalScrollBar->setValue(qRound(range.center()));
    horizontalScrollBar->setPageStep(qRound(range.size()));
}

void MUSE::keyPressEvent (QKeyEvent *e) {
    switch (e->key()) {
    case Qt::Key_Left:
        slices->setValue(slices->value()-1);
        break;
    case Qt::Key_Right:
        slices->setValue(slices->value()+1);
        break;
    case Qt::Key_M:
        if (currentBuffer==cubeSlice) {
            nparent->showPhys(meanSlice);
        } else if (currentBuffer==meanSlice) {
            nparent->showPhys(cubeSlice);
        }
    default:
        break;
    }
}


void MUSE::on_actionFFT_triggered() {

    QProgressDialog progress("Copy data", "Cancel", 0, 5, this);
    progress.setCancelButton(0);
    progress.setWindowModality(Qt::WindowModal);
    progress.show();
    progress.setValue(progress.value()+1);
    QApplication::processEvents();


    int nx=cubesize[2];
    int ny=cubesize[1];
    int nz=cubesize[0];

    if (cubesize.size()==3) {
        std::vector<double> cube(cubevect.size(),0);
#pragma omp parallel for
        for (size_t i=0; i< cubevect.size(); i++) {
            if (isfinite(cubevect[i]))
                cube[i]=cubevect[i];
        }
        int surf=nx*ny;
        int fftSize=surf*(nz/2+1);

        fftw_complex *cubeFFT = fftw_alloc_complex(fftSize);

        fftw_plan forw_blur = fftw_plan_dft_r2c_3d(nx, ny, nz, &cube[0], cubeFFT, FFTW_ESTIMATE);
        fftw_plan back_blur = fftw_plan_dft_c2r_3d(nx, ny, nz, cubeFFT, &cube[0], FFTW_ESTIMATE);

        progress.setLabelText("FFT Forward");
        progress.setValue(progress.value()+1);
        QApplication::processEvents();
        fftw_execute(forw_blur);

        progress.setLabelText("Blur");
        progress.setValue(progress.value()+1);
        QApplication::processEvents();
        double gx=1.0/pow(nx/(radiusLambda->value()),2);
        double gy=1.0/pow(ny/(radius->value()+1),2);
        double gz=1.0/pow(nz/(radius->value()+1),2);

#pragma omp parallel for collapse(3)
        for (int iz = 0 ; iz < nz/2+1; iz++) {
            for (int iy = 0 ; iy < ny; iy++) {
                for (int ix = 0 ; ix < nx ; ix++) {
                    int kx = (ix<nx/2+1) ? ix : ix-nx;
                    int ky = (iy<ny/2+1) ? iy : iy-ny;
                    int kz = iz;

                    double blur=exp(-(pow(kx,2)*gx+pow(ky,2)*gy+pow(kz,2)*gz));
                    int kk = iz+(nz/2+1)*(iy+ny*ix);
                    cubeFFT[kk][0]*=blur;
                    cubeFFT[kk][1]*=blur;
                }
            }
        }
        progress.setLabelText("Backward");
        progress.setValue(progress.value()+1);
        QApplication::processEvents();
        fftw_execute(back_blur);

        progress.setLabelText("Copy back");
        qDebug() << progress.value();
        progress.setValue(progress.value()+1);
        QApplication::processEvents();
        qDebug() << progress.value();

#pragma omp parallel for
        for (size_t i=0; i< cubevect.size(); i++) {
            if (isfinite(cubevect[i])) {
                cubevect[i]=cube[i]/cubevect.size();
            }
        }

        fftw_destroy_plan(forw_blur);
        fftw_destroy_plan(back_blur);
        fftw_free(cubeFFT);

        showImagePlane(slices->value());

        statusbar->showMessage(QString::number(progress.value()));

    }
}

void MUSE::plotClick(QMouseEvent* e) {
    QPointF my_pos(plot->xAxis->pixelToCoord(e->pos().x()),plot->yAxis->pixelToCoord(e->pos().y()));
    if (my_pos.x()>plot->xAxis->range().lower && my_pos.x()<plot->xAxis->range().upper  && my_pos.y()>plot->yAxis->range().lower && my_pos.y()<plot->yAxis->range().upper ) {
        int nslice=xvals.size()*(plot->xAxis->pixelToCoord(e->pos().x())-wavelen.first())/(wavelen.second()-wavelen.first());
        slices->setValue(nslice);
    }
}


void MUSE::updateLastPoint() {
    doSpectrum(lastpoint);
}

void MUSE::doSpectrum(QPointF point) {

    double prealx=(point.x()+1.0-my_offset.x())*my_scale.x()+my_offset_val.x();
    double prealy=(point.y()+1.0-my_offset.y())*my_scale.y()+my_offset_val.y();

    QPointF preal=QPointF(prealx,prealy);
    if (cubesize.size()==3 && point.x()>0 && point.y()>0 &&  point.x()*point.y() < cubesize[0]*cubesize[1]) {
        lastpoint=point.toPoint();
        vec2 p(point.x(),point.y());
        for (int zz=0; zz< yvals.size(); zz++) {
            yvals[zz]=0;
        }

        int surf=cubesize[0]*cubesize[1];

#pragma omp parallel for collapse(3)
        for (int xx=std::max((int)0,p.x()-radius->value());xx<=std::min((int)(cubesize[0]),p.x()+radius->value()); xx++) {
            for (int yy=std::max((int)0,p.y()-radius->value());yy<=std::min((int)(cubesize[1]),p.y()+radius->value()); yy++) {
                for (unsigned int zz=0; zz< cubesize[2]; zz++) {
                    yvals[zz]+=cubevect[xx+yy*cubesize[0]+zz*surf];
                }
            }
        }
        for (unsigned int zz=0; zz< cubesize[2]; zz++) {
            yvals[zz]=yvals[zz]/(1+2*radius->value()*radius->value());
        }
        plot->graph(1)->setData(xvals,yvals,true);
        if (currentBuffer) {
            QString spec_name("Ra:" + QString::number(preal.x(),'g',8)+" Dec:" +QString::number(preal.y(),'g',8) + " " + trUtf8("\xce\xbb") + ":" + QLocale().toString(xvals[slices->value()]));
            qDebug() << point << spec_name;
            plot->setTitle(spec_name);
        }
        plot->replot();

    }
}

void MUSE::showImagePlane(int z) {
    qDebug() << z;
    disconnect(slices,SIGNAL(valueChanged(int)),this,SLOT(showImagePlane(int)));
    disconnect(slicesSlider,SIGNAL(valueChanged(int)),this,SLOT(showImagePlane(int)));
    slices->setValue(z);
    slicesSlider->setValue(z);
    if (cubesize.size()==3 && z < (int)cubesize[2]) {
        nPhysD *my_phys=new nPhysD(cubesize[0],cubesize[1],0.0,QString::number(z).toStdString());
        my_phys->property=cube_prop;
        //        std::copy(cubevect.begin()+z*my_phys->getSurf(), cubevect.begin()+(z+1)*my_phys->getSurf(), my_phys->Timg_buffer);
        int rl=radiusLambda->value();
#pragma omp parallel for
        for (int l=std::max(0,z-rl/2); l < std::min((int)cubesize[2],z+rl-rl/2); l++) {
            int offset=l*my_phys->getSurf();
            for (unsigned int k=0; k < my_phys->getSurf(); k++) {
                my_phys->Timg_buffer[k]+=cubevect[offset+k];
            }
        }
        phys_divide(*my_phys,rl);
        my_phys->TscanBrightness();
        cubeSlice=nparent->replacePhys(my_phys,cubeSlice);
        plot->setMousePosition(xvals[z]);
    }
    QApplication::processEvents();
    connect(slices,SIGNAL(valueChanged(int)),this,SLOT(showImagePlane(int)));
    connect(slicesSlider,SIGNAL(valueChanged(int)),this,SLOT(showImagePlane(int)));
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

void MUSE::on_actionExport_triggered () {
    QString ftypes="SVG (*.svg);; PDF (*.PDF);; PNG (*.png);; Any files (*)";
    QString fout = QFileDialog::getSaveFileName(this,tr("Save All Drawings"),property("NeuSave-fileExport").toString(),ftypes);
    if (!fout.isEmpty()) {
        for (int i=0;i<slices->maximum() ; i++) {
            showImagePlane(i);
            QFileInfo fi(fout);
            nparent->exportGraphics(fi.path()+"/"+fi.baseName()+QString("_")+QString("%1").arg(i, 3, 10, QChar('0'))+QString("_")+QString::fromStdString(currentBuffer->getShortName())+"."+fi.completeSuffix());
        }
        setProperty("NeuSave-fileExport",fout);
    }

}

QVariant MUSE::extractData(QString key, QStringList values) {
    //    qDebug() << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " << key;
    key=key.leftJustified(8,' ',true);
    QVariant retval;
    for(auto &my_str: values) {
        //        qDebug() << "\t " << my_str;
        QStringList wavelist1(QString(my_str).split("=",QString::SkipEmptyParts));
        if (wavelist1.size()>1) {
            //            qDebug() << "here" << wavelist1.first();
            if(wavelist1.first()==key) {
                QStringList wavelist2(wavelist1.at(1).split(' ',QString::SkipEmptyParts));
                //                qDebug() << wavelist2;
                if (wavelist2.size()>1) {
                    bool ok;
                    QVariant val=wavelist2.first().toDouble(&ok);
                    if (ok) {
                        retval=QVariant::fromValue(val);
                    } else {
                        retval=wavelist2.first();
                    }
                } else {
                    retval=wavelist1.at(1);
                }
            }
        }
    }
    return retval;
}

void MUSE::loadCube() {
    QFileDialog fd;
    QString ifilename=fd.getOpenFileName(this,tr("Open MUSE file"),property("NeuSave-fileMUSE").toString(),tr("MUSE Cube")+QString(" (*.fits);;")+tr("Any files")+QString(" (*)"));

    if (!ifilename.isEmpty()) {
        fd.close();
        QApplication::processEvents();
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

            QStringList desc;
            for (ii = 1; ii <= nkeys; ii++)  {
                fits_read_record(fptr, ii, card, &status);
                if (fits_check_error(status)) return;
                desc << QString(card);

            }
            cube_prop["fits-header"]=desc.join('\n').toStdString();

            bool ok1,ok2;
            double val_dbl1,val_dbl2;
            val_dbl1=extractData("WAVELMIN",desc).toDouble(&ok1);
            val_dbl2=extractData("WAVELMAX",desc).toDouble(&ok2);
            if(ok1 && ok2) {
                wavelen.set_first(val_dbl1*10);
                wavelen.set_second(val_dbl2*10);
                DEBUG("wavelen " << wavelen);
            }

            QString val_str1=extractData("CTYPE1",desc).toString();
            QString val_str2=extractData("CTYPE2",desc).toString();
            qDebug() << val_str1 << val_str2;
            if (val_str1=="'RA---TAN'" && val_str2=="'DEC--TAN'") {
                DEBUG("here ");
                val_dbl1=extractData("CRPIX1",desc).toDouble(&ok1);
                val_dbl2=extractData("CRPIX2",desc).toDouble(&ok2);
                if(ok1 && ok2) {
                    qDebug() << "CRPIX1" << val_dbl1 << val_dbl2;
                    my_offset=QPointF(val_dbl1,val_dbl2);
                }
                val_dbl1=extractData("CD1_2",desc).toDouble(&ok1);
                val_dbl2=extractData("CD2_1",desc).toDouble(&ok2);
                if(ok1 && ok2 && val_dbl1==0 && val_dbl2==0) {
                    val_dbl1=extractData("CD1_1",desc).toDouble(&ok1);
                    val_dbl2=extractData("CD2_2",desc).toDouble(&ok2);
                    if(ok1 && ok2) {
                        qDebug() << "CD1_1" << val_dbl1 << val_dbl2;
                        my_scale=QPointF(val_dbl1,val_dbl2);
                    }
                }
                val_dbl1=extractData("CRVAL1",desc).toDouble(&ok1);
                val_dbl2=extractData("CRVAL2",desc).toDouble(&ok2);
                if(ok1 && ok2) {
                    qDebug() << "CRVAL1" << val_dbl1 << val_dbl2;
                    my_offset_val=QPointF(val_dbl1,val_dbl2);
                }
            }


            horizontalScrollBar->setRange(wavelen.x(),wavelen.y());

            std::vector<long> axissize(anaxis,0),fpixel(anaxis,1);

            fits_get_img_size(fptr,anaxis,&axissize[0],&status);
            if (fits_check_error(status)) return;

            long totalsize=1;
            for(int i=0; i<anaxis; i++) {
                totalsize*=axissize[i];
            }
            DEBUG("totalsize " << totalsize);

            if (anaxis==3) {
                int ret = QMessageBox::question(
                            this, tr("MUSE"),
                            tr("Found data cube") + QString::number(hdupos) +" : "+QString::number(axissize[0])+"x"+QString::number(axissize[1])+"x"+QString::number(axissize[2])+"\n"+tr("Open it?"),
                        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
                if (ret==QMessageBox::Yes) {
                    QProgressDialog progress("Reading Cube", "Cancel", 0, 3, this);
                    progress.setCancelButton(0);
                    progress.setWindowModality(Qt::WindowModal);
                    progress.setValue(progress.value()+1);
                    progress.show();
                    QApplication::processEvents();

                    cubevect.resize(totalsize);
                    cubesize.resize(anaxis);
                    fits_read_pix(fptr, TDOUBLE, &fpixel[0], totalsize, NULL, (void *)&cubevect[0], NULL, &status);
                    DEBUG("got a cube : " << totalsize << " = " << axissize[0] << " x " << axissize[1] << " x " << axissize[2]);
                    for(int i=0; i<anaxis; i++) {
                        cubesize[i]=axissize[i];
                    }

                    progress.setLabelText("Mean spectrum");
                    progress.setValue(progress.value()+1);
                    QApplication::processEvents();

                    xvals.resize(cubesize[2]);
                    yvals.resize(cubesize[2]);
                    ymean.resize(cubesize[2]);
                    for (int zz=0; zz< xvals.size(); zz++) {
                        xvals[zz]=wavelen.first()+zz*(wavelen.second()-wavelen.first())/xvals.size();
                        ymean[zz]=0;
                    }
                    int surf=cubesize[0]*cubesize[1];
                    std::vector<unsigned int> not_nan(ymean.size(),0);
#pragma omp parallel for
                    for (unsigned int kk=0;kk<totalsize; kk++) {
                        if (std::isfinite(cubevect[kk])) {
                            ymean[kk/surf]+=cubevect[kk];
                            not_nan[kk/surf]++;
                        }
                    }

                    for (int xx=0; xx< xvals.size(); xx++) {
                        ymean[xx]/=not_nan[xx];
                    }

                    progress.setLabelText("Mean image");
                    progress.setValue(progress.value()+1);
                    QApplication::processEvents();
                    if (!meanSlice) nparent->removePhys(meanSlice);
                    meanSlice=new nPhysD(cubesize[0],cubesize[1],0.0,"mean slice");
                    nPhysImageF<int> my_num(cubesize[0],cubesize[1],0,"number");

                    meanSlice->property=cube_prop;

#pragma omp parallel for collapse(2)
                    for (unsigned int l=0; l < cubesize[2]; l++) {
                        for (int k=0; k < surf; k++) {
                            double val=cubevect[l*surf+k];
                            if (std::isfinite(val)) {
                                meanSlice->Timg_buffer[k]+=val;
                                my_num.Timg_buffer[k]++;
                            }
                        }
                    }
#pragma omp parallel for
                    for (int k=0; k < surf; k++) {
                        meanSlice->Timg_buffer[k]/=my_num.Timg_buffer[k];
                    }

                    meanSlice->TscanBrightness();
                    nparent->addShowPhys(meanSlice);

                    plot->graph(0)->setName("Mean spectrum");
                    plot->graph(0)->setData(xvals,ymean,true);

                    plot->rescaleAxes();
                    plot->replot();

                    slices->setMaximum(axissize[2]);
                    slicesSlider->setMaximum(axissize[2]);

                    showImagePlane(slices->value());
                    break;
                }
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

        fits_check_error(status);

        DEBUG("out of here");
    }
}
