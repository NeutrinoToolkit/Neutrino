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
#include "nCamera.h"
#include "neutrino.h"
#include <QCameraInfo>
#include <QCameraImageCapture>

nCamera::nCamera(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname),
  camera(NULL),
  imageCapture(NULL),
  imgGray(NULL),
  imgColor(3,nullptr)
{
	my_w.setupUi(this);
	decorate();

    cameraMenu=new QMenu(this);
    QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
    foreach (const QCameraInfo &cameraInfo, cameras) {
        QAction *my_act = new QAction(cameraInfo.description(),cameraMenu);
        my_act->setData(cameraInfo.deviceName());
        connect(my_act, SIGNAL(triggered()), this, SLOT(changeCameraAction()));
        cameraMenu->addAction(my_act);
    }
    if (cameraMenu->actions().size()) {
        cameraMenu->actions().first()->trigger();
    } else {
        QLabel *not_found = new QLabel("Camera not present", my_w.centralwidget);
        my_w.centralwidget->layout()->addWidget(not_found);
    }
}

nCamera::~nCamera()
{
    delete imageCapture;
    delete camera;
}

void nCamera::on_grab_clicked() {
    if (imageCapture)
        imageCapture->capture();
}

void nCamera::changeCameraAction() {
    QAction *action = qobject_cast<QAction *>(sender());
    if (action) {
        QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
        foreach (const QCameraInfo &cameraInfo, cameras) {
            if (cameraInfo.deviceName() == action->data().toString())
                setupCam(cameraInfo);
        }
    }

}

void nCamera::setupCam (const QCameraInfo &cameraInfo) {
    delete imageCapture;
    delete camera;

    camera = new QCamera(cameraInfo);
    imageCapture = new QCameraImageCapture(camera);
    imageCapture->setCaptureDestination(QCameraImageCapture::CaptureToBuffer);
    if (imageCapture->isCaptureDestinationSupported(QCameraImageCapture::CaptureToBuffer)) {
        connect(imageCapture, SIGNAL(imageCaptured(int,QImage)), this, SLOT(processCapturedImage(int,QImage)));
    } else {
        connect(imageCapture, SIGNAL(imageSaved(int,QString)), this, SLOT(processCapturedImage(int,QString)));
    }

    camera->setViewfinder(my_w.viewfinder);
    camera->start();
}

void nCamera::giveNeutrino(const QImage& image) {
    if(!image.isNull()) {
        if (my_w.gray->isChecked()) {
            nPhysD *datamatrix = new nPhysD(image.width(), image.height(),0,"Camera gray");
            datamatrix->setShortName("gray");
            for (int i=0;i<image.height();i++) {
                for (int j=0;j<image.width();j++) {
                    QRgb px = image.pixel(j,i);
                    datamatrix->Timg_matrix[i][j]= (qRed(px)+qGreen(px)+qBlue(px))/3.0;
                }
            }
            datamatrix->TscanBrightness();
            if (!my_w.keep_copy->isChecked()) {
                imgGray=NULL;
            }
            imgGray=nparent->replacePhys(datamatrix,imgGray);
        } else {
            nPhysD *loc_red = new nPhysD(image.width(), image.height(),0,"Camera red");
            nPhysD *loc_gre = new nPhysD(image.width(), image.height(),0,"Camera green");
            nPhysD *loc_blu = new nPhysD(image.width(), image.height(),0,"Camera blue");
            loc_red->setShortName("red");
            loc_gre->setShortName("green");
            loc_blu->setShortName("blue");
            for (int i=0;i<image.height();i++) {
                for (int j=0;j<image.width();j++) {
                    QRgb px = image.pixel(j,i);
                    loc_red->Timg_matrix[i][j]= qRed(px);
                    loc_gre->Timg_matrix[i][j]= qGreen(px);
                    loc_blu->Timg_matrix[i][j]= qBlue(px);
                }
            }
            loc_red->TscanBrightness();
            loc_gre->TscanBrightness();
            loc_blu->TscanBrightness();
            if (!my_w.keep_copy->isChecked()) {
                for (int i=0;i<3;i++) {
                    imgColor[0]=nullptr;
                }
            }
            imgColor[0]=nparent->replacePhys(loc_red,imgColor[0]);
            imgColor[1]=nparent->replacePhys(loc_gre,imgColor[1]);
            imgColor[2]=nparent->replacePhys(loc_blu,imgColor[2]);
        }
    }
}

void nCamera::processCapturedImage(int requestId, const QImage& image)
{
    Q_UNUSED(requestId);
    giveNeutrino(image);
}

void nCamera::processCapturedImage(int requestId, const QString& imageFile)
{
    Q_UNUSED(requestId);
    const QImage image(imageFile);
    giveNeutrino(image);
    QFile::remove(imageFile);

}

void nCamera::contextMenuEvent (QContextMenuEvent *ev) {
    cameraMenu->exec(ev->globalPos());
}


