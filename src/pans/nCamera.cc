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
  imgGray(NULL)
{
	my_w.setupUi(this);
	decorate();

    QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
    foreach (const QCameraInfo &cameraInfo, cameras) {
        my_w.cameras->addItem(cameraInfo.description(), cameraInfo.deviceName());
    }

    connect(my_w.doIt,SIGNAL(pressed()),this,SLOT(doIt()));
    connect(my_w.cameras,SIGNAL(activated(int)),this, SLOT(changeCamera()));
    changeCamera();
}

nCamera::~nCamera()
{
    delete imageCapture;
    delete camera;
}

void nCamera::doIt() {
    if (imageCapture)
        imageCapture->capture();
}

void nCamera::changeCamera() {
    QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
    foreach (const QCameraInfo &cameraInfo, cameras) {
        if (cameraInfo.deviceName() == my_w.cameras->currentData())
            setupCam(cameraInfo);
    }
}

void nCamera::setupCam (const QCameraInfo &cameraInfo) {
    delete imageCapture;
    delete camera;

    camera = new QCamera(cameraInfo);
    imageCapture = new QCameraImageCapture(camera);
    connect(imageCapture, SIGNAL(imageCaptured(int,QImage)), this, SLOT(processCapturedImage(int,QImage)));
    camera->setViewfinder(my_w.viewfinder);
    camera->start();
}

void nCamera::processCapturedImage(int requestId, const QImage& image)
{
    Q_UNUSED(requestId);
    if (!image.isNull()) {
        if (my_w.keep_copy->isChecked())
            imgGray=NULL;
        nPhysD *datamatrix = new nPhysD("Camera gray");
        datamatrix->resize(image.width(), image.height());
        for (int i=0;i<image.height();i++) {
            for (int j=0;j<image.width();j++) {
                QRgb px = image.pixel(j,i);
                datamatrix->Timg_matrix[i][j]= (qRed(px)+qGreen(px)+qBlue(px))/3.0;
            }
        }
        datamatrix->TscanBrightness();
        imgGray=nparent->replacePhys(datamatrix,imgGray);
    }
}
