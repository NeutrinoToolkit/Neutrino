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
// physWavelets

nCamera::nCamera(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname),
  camera(NULL),
  imageCapture(NULL)
{
	my_w.setupUi(this);
	
	decorate();
	
    QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
    foreach (const QCameraInfo &cameraInfo, cameras) {
        my_w.cameras->addItem(cameraInfo.description(), cameraInfo.deviceName());
    }

    connect(my_w.doIt,SIGNAL(pressed()),this,SLOT(doIt()));

    changeCamera();
}

nCamera::~nCamera()
{
    delete imageCapture;
    delete camera;
}

void nCamera::doIt() {
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

void nCamera::keyPressEvent(QKeyEvent * event)
{
    if (event->isAutoRepeat())
        return;

    switch (event->key()) {
    case Qt::Key_CameraFocus:
//        displayViewfinder();
        WARNING("here");
        camera->searchAndLock();
        event->accept();
        break;
    case Qt::Key_Camera:
        WARNING("here");
        if (camera->captureMode() == QCamera::CaptureStillImage) {
            WARNING("here");
            doIt();
        }
        event->accept();
        break;
    default:
        QMainWindow::keyPressEvent(event);
    }
}

void nCamera::keyReleaseEvent(QKeyEvent *event)
{
    if (event->isAutoRepeat())
        return;

    switch (event->key()) {
    case Qt::Key_CameraFocus:
        camera->unlock();
        break;
    default:
        QMainWindow::keyReleaseEvent(event);
    }
}

void nCamera::processCapturedImage(int requestId, const QImage& image)
{
    Q_UNUSED(requestId);

    vector <nPhysD *> imagelist;
    if (!image.isNull()) {
        if (image.isGrayscale()) {
            nPhysD *datamatrix = new nPhysD("Camera gray");
            datamatrix->resize(image.width(), image.height());
            for (int i=0;i<image.height();i++) {
                for (int j=0;j<image.width();j++) {
                    datamatrix->Timg_matrix[i][j]= qRed(image.pixel(j,i));
                }
            }
            imagelist.push_back(datamatrix);
        } else {
            nPhysD *datamatrix[3];
            string name[3];
            name[0]="Red";
            name[1]="Green";
            name[2]="Blue";
            for (int k=0;k<3;k++) {
                datamatrix[k] = new nPhysD("Camera color");
                datamatrix[k]->setShortName(name[k]);
                datamatrix[k]->setName("Camera "+name[k]);
                datamatrix[k]->resize(image.width(), image.height());
                imagelist.push_back(datamatrix[k]);
            }
            for (int i=0;i<image.height();i++) {
                for (int j=0;j<image.width();j++) {
                    QRgb px = image.pixel(j,i);
                    datamatrix[0]->Timg_matrix[i][j]= (double) (qRed(px));
                    datamatrix[1]->Timg_matrix[i][j]= (double) (qGreen(px));
                    datamatrix[2]->Timg_matrix[i][j]= (double) (qBlue(px));
                }
            }

        }
    }

    for (vector<nPhysD *>::iterator it=imagelist.begin(); it!=imagelist.end();) {
        (*it)->TscanBrightness();
        nparent->addShowPhys(*it);
        it++;
    }


}
