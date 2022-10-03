// Copyright (C) 2017 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

#include "Camera.h"

#include <QAction>
#include <QActionGroup>
#include <QCameraDevice>
#include <QImage>
#include <QLineEdit>
#include <QMediaDevices>
#include <QMediaFormat>
#include <QVideoWidget>

Camera::Camera(neutrino *nparent) :
    nGenericPan(nparent),
    imgColor{{nullptr,nullptr,nullptr}},
    cameraMenu(this)
{
    setupUi(this);
    updateCameras();
    connect(&m_devices, &QMediaDevices::videoInputsChanged, this, &Camera::updateCameras);
    setCamera(QMediaDevices::defaultVideoInput());
    show();
}

void Camera::setCamera(const QCameraDevice &cameraDevice) {
    m_camera.reset(new QCamera(cameraDevice));
    m_captureSession.setCamera(m_camera.data());

    connect(m_camera.data(), &QCamera::errorOccurred, this, &Camera::displayCameraError);

    m_imageCapture = new QImageCapture;
    m_captureSession.setImageCapture(m_imageCapture);

    m_captureSession.setVideoOutput(viewfinder);

    connect(m_imageCapture, &QImageCapture::errorOccurred, this, &Camera::displayCaptureError);
    connect(m_imageCapture, &QImageCapture::imageCaptured, this, &Camera::processCapturedImage);

    if (m_camera->cameraFormat().isNull()) {
        auto formats = cameraDevice.videoFormats();
        if (!formats.isEmpty()) {
            // Choose a decent camera format: Maximum resolution at at least 30 FPS
            // we use 29 FPS to compare against as some cameras report 29.97 FPS...
            QCameraFormat bestFormat;
            for (const auto &fmt : formats) {
                if (bestFormat.maxFrameRate() < 29
                    && fmt.maxFrameRate() > bestFormat.maxFrameRate())
                    bestFormat = fmt;
                else if (bestFormat.maxFrameRate() == fmt.maxFrameRate()
                         && bestFormat.resolution().width() * bestFormat.resolution().height()
                                 < fmt.resolution().width() * fmt.resolution().height())
                    bestFormat = fmt;
            }

            m_camera->setCameraFormat(bestFormat);
        }
    }

    m_camera->start();
}

void Camera::on_grab_clicked() {
    saveDefaults();
    if (m_imageCapture)
        m_imageCapture->capture();
}

void Camera::takeImage()
{
    m_imageCapture->captureToFile();
}

void Camera::displayCaptureError(int id, const QImageCapture::Error error, const QString &errorString) {
    Q_UNUSED(id);
    Q_UNUSED(error);
    statusbar->showMessage(errorString,5000);
}

void Camera::displayCameraError()
{
    if (m_camera->error() != QCamera::NoError)
        statusbar->showMessage(m_camera->errorString(),5000);
}

void Camera::updateCameraDevice(QAction *action)
{
    setCamera(qvariant_cast<QCameraDevice>(action->data()));
}

void Camera::updateCameras() {
    cameraMenu.clear();
    const QList<QCameraDevice> availableCameras = QMediaDevices::videoInputs();
    for (const QCameraDevice &cameraDevice : availableCameras) {
        QAction *videoDeviceAction = new QAction(cameraDevice.description(), &cameraMenu);
        videoDeviceAction->setCheckable(true);
        videoDeviceAction->setData(QVariant::fromValue(cameraDevice));
        if (cameraDevice == QMediaDevices::defaultVideoInput())
            videoDeviceAction->setChecked(true);

        cameraMenu.addAction(videoDeviceAction);
    }
    if (availableCameras.size()==0) {
        statusbar->showMessage("No camera found!",5000);
    }
}

void Camera::contextMenuEvent (QContextMenuEvent *ev) {
    cameraMenu.exec(ev->globalPos());
}

void Camera::processCapturedImage(int requestId, const QImage& image) {
    Q_UNUSED(requestId)
    giveNeutrino(image);
}

void Camera::giveNeutrino(const QImage& image) {
    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    bool separate_rgb= my_set.value("separateRGB",false).toBool();
    my_set.endGroup();

    if(!image.isNull()) {
        if (!separate_rgb) {
            nPhysD *datamatrix = new nPhysD(image.width(), image.height(),0,"Camera gray");
            datamatrix->setShortName("gray");
            for (int i=0;i<image.height();i++) {
                for (int j=0;j<image.width();j++) {
                    QRgb px = image.pixel(j,i);
                    datamatrix->Timg_matrix[i][j]= (qRed(px)+qGreen(px)+qBlue(px))/3.0;
                }
            }
            datamatrix->TscanBrightness();
            if (!erasePrevious->isChecked()) {
                imgGray=nullptr;
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
            if (!erasePrevious->isChecked()) {
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
