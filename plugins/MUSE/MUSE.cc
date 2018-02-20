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

#include "neutrino.h"
#include "nPhysFormats.h"
#include "fitsio.h"

#define HDF5_MAX_NAME 2048

MUSE::MUSE(neutrino *nparent) : nGenericPan(nparent),
	my_offset(0,0),
	my_offset_val(0,0),
	my_scale(1,1),
	cubeSlice(nullptr),
	meanSlice(nullptr),
	wavelen(0,1)
{
	setupUi(this);

    my_point =  new nPoint(nparent);

	connect(horizontalScrollBar, SIGNAL(valueChanged(int)), this, SLOT(horzScrollBarChanged(int)));
	connect(plot->xAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(xAxisChanged(QCPRange)));

	connect(radius,SIGNAL(valueChanged(int)),this,SLOT(updateLastPoint()));

	connect(slices,SIGNAL(valueChanged(int)),this,SLOT(showImagePlane(int)));
	connect(slicesSlider,SIGNAL(valueChanged(int)),this,SLOT(showImagePlane(int)));

	connect(plot,SIGNAL(mouseDoubleClick(QMouseEvent*)), this, SLOT(plotClick(QMouseEvent*)));

	new QShortcut(QKeySequence( Qt::Key_S),this, SLOT(on_actionExportTxt_triggered()));

	setProperty("NeuSave-fileMUSE","myfile.fits");
	plot->addGraph(plot->xAxis, plot->yAxis2);
	plot->addGraph(plot->xAxis, plot->yAxis);

	toolBar->addWidget(percent);
	toolBar->addWidget(radius);
	toolBar->addWidget(restLambda);
	toolBar->addWidget(lambdaz);

	setProperty("NeuSave-MUSEprefix","spec_");
	setProperty("NeuSave-MUSEdir",QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
	qDebug() << property("NeuSave-MUSEdir");
	my_timer.setInterval(property("NeuSave-interval").toInt());
	connect(&my_timer,SIGNAL(timeout()), this, SLOT(nextPlane()));

	connect(nparent->my_w->my_view,SIGNAL(keypressed(QKeyEvent*)),this,SLOT(keyPressEvent(QKeyEvent*)));

	show();
	on_actionMode_toggled();

    QShortcut *openFile = new QShortcut(QKeySequence(Qt::Key_O),this);
    connect(openFile, SIGNAL(activated()), this, SLOT(loadCube()) );

	loadCube();
}

void MUSE::on_actionMovie_triggered() {
	DEBUG("here");
	if (actionMovie->isChecked()) {
		my_timer.start();
	} else {
		my_timer.stop();
	}
}

void MUSE::nextPlane(){
	DEBUG("here" << slices->value());

	slices->setValue((slices->value()+1)%slices->maximum());
}

void MUSE::on_percent_valueChanged(double val) {
	if (meanSlice) meanSlice->property["display_range"] = getColorPrecentPixels(*meanSlice,val);
	if (cubeSlice) cubeSlice->property["display_range"] = getColorPrecentPixels(*cubeSlice,val);
    nparent->updatePhys();
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


void MUSE::on_actionMean_triggered() {
	if (actionMean->isChecked()) {
		nparent->showPhys(meanSlice);
	} else {
		nparent->showPhys(cubeSlice);
	}
}

void MUSE::on_actionExportTxt_triggered() {
	qDebug() << sender();
	QString prefix=property("NeuSave-MUSEprefix").toString();
	QDir dirName(property("NeuSave-MUSEdir").toString());
	qDebug() << dirName;
	if (sender() && sender()->objectName()=="actionExportTxt") {
		bool ok;
		QString dirNamestr = QFileDialog::getExistingDirectory(this,tr("Spectra dir"),dirName.path());
		if (QFileInfo(dirNamestr).isDir()) {
			setProperty("NeuSave-MUSEdir",dirNamestr);
			dirName.setCurrent(dirNamestr);
		} else {
			statusbar->showMessage("Cannot change to dir "+ dirNamestr+ ". Using"+ dirName.path());
		}

		QString text = QInputDialog::getText(this, tr("Prefix"),tr("Spectrum File prefix:"), QLineEdit::Normal, prefix, &ok);
		if (ok) {
			setProperty("NeuSave-MUSEprefix",text);
		} else {
			return;
		}
	}

	int max_len=cubeSlice?log10(std::max(cubeSlice->getW(),cubeSlice->getH()))+1:5;
	QString fname(prefix+QString("%1_%2.txt").arg(lastpoint.x(), max_len, 10, QLatin1Char('0')).arg(lastpoint.y(), max_len, 10, QLatin1Char('0')));
	QFile t(dirName.filePath(fname));
	qDebug() << fname<<    t.fileName();
	t.open(QIODevice::WriteOnly| QIODevice::Text);
	if (t.isOpen()) {
		QTextStream out(&t);
		QLocale loc("C");
		out << "# ( " << loc.toString(lastpoint.x()) << " , " << loc.toString(lastpoint.y()) << " ) " << endl;
		out << "# r = " << radius->value() << " px. Tot pixels: (2r+1)^2 = " << pow(radius->value()*2+1,2) << endl;
		out << "# " << plot->graph(1)->name() << endl;
		for (int xx=0; xx< xvals.size(); xx++) {
			out << loc.toString(xvals[xx],'g',6) << " "<< loc.toString(yvals[xx],'g',6) << " "<< loc.toString(ymean[xx],'g',6) << endl;
		}
		t.close();
	}
}

void MUSE::keyPressEvent (QKeyEvent *e) {
	int delta = (e->modifiers() & Qt::ShiftModifier) ? 10 : 1;
	switch (e->key()) {
		case Qt::Key_Left:
			slices->setValue((slices->maximum()+slices->value()-delta)%slices->maximum());
			break;
		case Qt::Key_Right:
			slices->setValue((slices->maximum()+slices->value()+delta)%slices->maximum());
			break;
		case Qt::Key_S:
			on_actionExportTxt_triggered();
			break;
		case Qt::Key_Space:
		case Qt::Key_P:
			actionMovie->trigger();
			break;
		case Qt::Key_Plus:
			if (my_timer.interval()>=50)
				my_timer.setInterval(my_timer.interval()-50);
			setProperty("NeuSave-interval",my_timer.interval());
			break;
		case Qt::Key_Minus:
			my_timer.setInterval(my_timer.interval()+50);
			setProperty("NeuSave-interval",my_timer.interval());
			break;
		default:
			break;
	}
}

void MUSE::plotClick(QMouseEvent* e) {
	QPointF my_pos(plot->xAxis->pixelToCoord(e->pos().x()),plot->yAxis->pixelToCoord(e->pos().y()));
	if (my_pos.x()>plot->xAxis->range().lower && my_pos.x()<plot->xAxis->range().upper  && my_pos.y()>plot->yAxis->range().lower && my_pos.y()<plot->yAxis->range().upper ) {
        int nslice=abs(xvals.size()*(my_pos.x()-wavelen.first())/(wavelen.second()-wavelen.first()));
        qDebug() << my_pos << nslice;
        showImagePlane(nslice);
	}
}


void MUSE::updateLastPoint() {
	doSpectrum(lastpoint);
}

void MUSE::doSpectrum(QPointF point) {

	QPoint pFloor(floor(point.x())+1.0,floor(point.y())+1.0);
    qDebug() << pFloor;
    my_point->setPoint(point);

	double prealx=(pFloor.x()-my_offset.x())*my_scale.x()+my_offset_val.x();
	double prealy=(pFloor.y()-my_offset.y())*my_scale.y()+my_offset_val.y();

	QPointF preal=QPointF(prealx,prealy);

	//    qDebug() << toNum(pFloor) << toNum(my_offset) << toNum(my_scale) << toNum(my_offset_val) << toNum(preal);

	if (cubesize.size()==3 && point.x()>0 && point.y()>0 &&  point.x()*point.y() < cubesize[0]*cubesize[1]) {
		lastpoint=pFloor;
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
		plot->graph(1)->setData(xvals,yvals,true);
        QString spec_name("("+locale().toString(pFloor.x())+","+locale().toString(pFloor.y())+ ") Ra:" + locale().toString(preal.x(),'g',8)+" Dec:" +locale().toString(preal.y(),'g',8));
		plot->graph(1)->setName(spec_name);
		plot->setTitle(spec_name);
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
        nPhysD *my_phys=new nPhysD(cubesize[0],cubesize[1],0.0,locale().toString(z).toStdString());
		my_phys->property=cube_prop;

		int offset=z*my_phys->getSurf();
#pragma omp parallel for
		for (unsigned int k=0; k < my_phys->getSurf(); k++) {
			my_phys->Timg_buffer[k]+=cubevect[offset+k];
		}
		my_phys->TscanBrightness();

		my_phys->property["display_range"]=getColorPrecentPixels(*my_phys,percent->value());

		if (cubeSlice) {
			cubeSlice->property["display_range"]=my_phys->property["display_range"];
		} else {
			cube_prop["display_range"]=my_phys->property["display_range"];
		}
		cubeSlice=nparent->replacePhys(my_phys,cubeSlice);
		plot->setMousePosition(xvals[z]);
		setstatusbar();
	}
	QApplication::processEvents();
	connect(slices,SIGNAL(valueChanged(int)),this,SLOT(showImagePlane(int)));
	connect(slicesSlider,SIGNAL(valueChanged(int)),this,SLOT(showImagePlane(int)));
}

void MUSE::on_restLambda_valueChanged(double) {
	setstatusbar();
}

void MUSE::setstatusbar() {
	if (xvals.size()>slices->value()) {
		double lambda=xvals[slices->value()];
		double redshift=lambda/restLambda->value()-1.0;
        lambdaz->setText(trUtf8("\xce\xbb") + ":" + locale().toString(lambda) + " z=" + locale().toString(redshift));
	}
}

void MUSE::on_actionMode_toggled() {
	if (actionMode->isChecked()) {
		disconnect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(doSpectrum(QPointF)));
		connect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(doSpectrum(QPointF)));
        my_point->show();
    } else {
		disconnect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(doSpectrum(QPointF)));
		connect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(doSpectrum(QPointF)));
        my_point->hide();
	}
}

//void MUSE::on_actionExport_triggered () {
//    QString ftypes="SVG (*.svg);; PDF (*.PDF);; PNG (*.png);; Any files (*)";
//    QString fout = QFileDialog::getSaveFileName(this,tr("Save All Drawings"),property("NeuSave-fileExport").toString(),ftypes);
//    if (!fout.isEmpty()) {
//        for (int i=0;i<slices->maximum() ; i++) {
//            showImagePlane(i);
//            QFileInfo fi(fout);
//            nparent->exportGraphics(fi.path()+"/"+fi.baseName()+QString("_")+QString("%1").arg(i, 3, 10, QChar('0'))+QString("_")+"."+fi.completeSuffix());
//        }
//        setProperty("NeuSave-fileExport",fout);
//    }

//}

QVariant MUSE::extractData(QString key, QStringList values) {
	qDebug() << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " << key;
	key=key.leftJustified(8,' ',true);
	QVariant retval;
	for(auto &my_str: values) {
		qDebug() << "\t " << my_str;
		QStringList wavelist1(QString(my_str).split("=",QString::SkipEmptyParts));
		if (wavelist1.size()>1) {
			qDebug() << "here" << wavelist1.first();
			if(wavelist1.first()==key) {
				QStringList wavelist2(wavelist1.at(1).split(' ',QString::SkipEmptyParts));
				qDebug() << wavelist2;
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
		DEBUG(hdupos);

		for (; !status; hdupos++)  {


			int hdutype;
			fits_get_hdu_type(fptr, &hdutype, &status);
			if (fits_check_error(status)) return;

			if (hdutype == IMAGE_HDU) {
                std::array<long,9> naxes={{1, 1, 1, 1, 1, 1, 1, 1, 1}};
				int naxis = 0;
				fits_get_img_param(fptr, 9, &bitpix, &naxis, &naxes[0], &status);
				DEBUG("IMAGE_HDU " << naxis);
				for (ii = 0; ii < 9; ii++) {
					DEBUG(ii << " " << naxes[ii]);
				}
			}

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
			std::stringstream ss;
			ss << "fits-header" << std::setfill('0') << std::setw(2) << hdupos;
			cube_prop[ss.str()]=desc.join('\n').toStdString();
			DEBUG("Fits header:\n" << cube_prop[ss.str()]);

			bool ok1,ok2;
			double val_dbl1,val_dbl2;
			val_dbl1=extractData("CRVAL3",desc).toDouble(&ok1);
			val_dbl2=extractData("CD3_3",desc).toDouble(&ok2);
			if(ok1 && ok2) {
				wavelen.set_first(val_dbl1);
				wavelen.set_second(val_dbl2);
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



			std::vector<long> axissize(anaxis,0),fpixel(anaxis,1);

			fits_get_img_size(fptr,anaxis,&axissize[0],&status);
			if (fits_check_error(status)) return;

			unsigned long totalsize=1;
			for(int i=0; i<anaxis; i++) {
				totalsize*=axissize[i];
			}
			DEBUG("totalsize " << totalsize);

			if (anaxis==3) {
                int ret = QMessageBox::information(
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
						xvals[zz]=wavelen.first()+zz*wavelen.second();
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
					meanSlice->property["display_range"]=getColorPrecentPixels(*meanSlice,percent->value());
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

//void MUSE::on_actionFFT_triggered() {

//    QProgressDialog progress("Copy data", "Cancel", 0, 5, this);
//    progress.setCancelButton(0);
//    progress.setWindowModality(Qt::WindowModal);
//    progress.show();
//    progress.setValue(progress.value()+1);
//    QApplication::processEvents();


//    int nx=cubesize[0];
//    int ny=cubesize[1];
//    int nz=cubesize[2];

//    int surf=nx*ny;

//    if (cubesize.size()==3) {
//        std::vector<double> cube(cubevect.size(),0);
//#pragma omp parallel for
//        for (size_t i=0; i< cubevect.size(); i++) {
//            if (std::isfinite(cubevect[i])) {
//                cube[i]=cubevect[i];
//            } else {
//                int kk=i/surf;
//                if (std::isfinite(ymean[kk]))
//                    cube[i]=ymean[kk];
//            }

//        }
//        int surf=nz*ny;
//        int fftSize=surf*(nx/2+1);

//        fftw_complex *cubeFFT = fftw_alloc_complex(fftSize);

//        fftw_plan forw_blur = fftw_plan_dft_r2c_3d(nz, ny, nx, &cube[0], cubeFFT, FFTW_ESTIMATE);
//        fftw_plan back_blur = fftw_plan_dft_c2r_3d(nz, ny, nx, cubeFFT, &cube[0], FFTW_ESTIMATE);

//        progress.setLabelText("FFT Forward");
//        progress.setValue(progress.value()+1);
//        QApplication::processEvents();
//        fftw_execute(forw_blur);

//        progress.setLabelText("Blur");
//        progress.setValue(progress.value()+1);
//        QApplication::processEvents();
//        double gx=1.0/pow(nx/(radius->value()+1),2);
//        double gy=1.0/pow(ny/(radius->value()+1),2);
//        double radiusLambda=4
//        double gz=1.0/pow(nz/(radiusLambda),2);

//#pragma omp parallel for collapse(3)
//        for (int ix = 0 ; ix < nx/2+1; ix++) {
//            for (int iy = 0 ; iy < ny; iy++) {
//                for (int iz = 0 ; iz < nz ; iz++) {
//                    int kx = ix;
//                    int ky = (iy<ny/2+1) ? iy : iy-ny;
//                    int kz = (iz<nz/2+1) ? iz : iz-nz;

//                    double blur=exp(-(pow(kz,2)*gz+pow(ky,2)*gy+pow(kx,2)*gx));
//                    int kk = ix+(nx/2+1)*(iy+ny*iz);
//                    cubeFFT[kk][0]*=blur;
//                    cubeFFT[kk][1]*=blur;
//                }
//            }
//        }
//        progress.setLabelText("Backward");
//        progress.setValue(progress.value()+1);
//        QApplication::processEvents();
//        fftw_execute(back_blur);

//        progress.setLabelText("Copy back");
//        qDebug() << progress.value();
//        progress.setValue(progress.value()+1);
//        QApplication::processEvents();
//        qDebug() << progress.value();

//#pragma omp parallel for
//        for (size_t i=0; i< cubevect.size(); i++) {
//            if (std::isfinite(cubevect[i])) {
//                cubevect[i]=cube[i]/cubevect.size();
//            }
//        }

//        fftw_destroy_plan(forw_blur);
//        fftw_destroy_plan(back_blur);
//        fftw_free(cubeFFT);

//        showImagePlane(slices->value());

//        statusbar->showMessage(locale().toString(progress.value()));

//    }
//}
