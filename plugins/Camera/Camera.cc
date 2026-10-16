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
#if defined(QT_FEATURE_permissions) && QT_CONFIG(permissions) // QPermissions exists only since Qt 6.5
#include <QPermissions>
#endif

Camera::Camera(neutrino *nparent) :
    nGenericPan(nparent),
    cameraMenu(this)
{
    setupUi(this);
    updateCameras();
    connect(&m_devices, &QMediaDevices::videoInputsChanged, this, &Camera::updateCameras);
    connect(&cameraMenu, &QMenu::triggered, this, &Camera::updateCameraDevice);
    // images we created and the user closed
    connect(nparent, &neutrino::physDel, this, [this](nPhysD *buf) {
        if (buf==imgGray) imgGray=nullptr;
        for (auto &img : imgColor) if (buf==img) img=nullptr;
    });
    show();
    startCamera();
}

// since Qt 6.5 the camera must be explicitly allowed (macOS asks the user once)
void Camera::startCamera() {
#if defined(QT_FEATURE_permissions) && QT_CONFIG(permissions) // QPermissions exists only since Qt 6.5
    QCameraPermission cameraPermission;
    switch (qApp->checkPermission(cameraPermission)) {
        case Qt::PermissionStatus::Undetermined:
            qApp->requestPermission(cameraPermission, this, &Camera::startCamera);
            return;
        case Qt::PermissionStatus::Denied:
            statusbar->showMessage(tr("Camera access denied: allow Neutrino in System Settings > Privacy & Security > Camera"));
            return;
        case Qt::PermissionStatus::Granted:
            break;
    }
#endif
    setCamera(QMediaDevices::defaultVideoInput());
}

void Camera::setCamera(const QCameraDevice &cameraDevice) {
    if (cameraDevice.isNull()) {
        statusbar->showMessage(tr("No camera found!"),5000);
        return;
    }
    m_camera.reset(new QCamera(cameraDevice));
    m_captureSession.setCamera(m_camera.data());

    connect(m_camera.data(), &QCamera::errorOccurred, this, &Camera::displayCameraError);

    // replace (not leak) the previous capture object and its connections
    QImageCapture *oldCapture=m_imageCapture;
    m_imageCapture = new QImageCapture(this);
    m_captureSession.setImageCapture(m_imageCapture);
    delete oldCapture;

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
    updateCameras(); // check mark on the camera in use
}

void Camera::on_grab_clicked() {
    saveDefaults();
    if (m_imageCapture)
        m_imageCapture->capture();
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
    delete cameraGroup;
    cameraGroup=new QActionGroup(&cameraMenu);
    cameraGroup->setExclusive(true);
    const QCameraDevice current = m_camera ? m_camera->cameraDevice() : QMediaDevices::defaultVideoInput();
    const QList<QCameraDevice> availableCameras = QMediaDevices::videoInputs();
    for (const QCameraDevice &cameraDevice : availableCameras) {
        QAction *videoDeviceAction = new QAction(cameraDevice.description(), cameraGroup);
        videoDeviceAction->setCheckable(true);
        videoDeviceAction->setData(QVariant::fromValue(cameraDevice));
        videoDeviceAction->setChecked(cameraDevice == current);
        cameraMenu.addAction(videoDeviceAction);
    }
    if (availableCameras.size()==0) {
        statusbar->showMessage(tr("No camera found!"),5000);
    }
}

void Camera::contextMenuEvent (QContextMenuEvent *ev) {
    cameraMenu.exec(ev->globalPos());
}

void Camera::processCapturedImage(int requestId, const QImage& image) {
    Q_UNUSED(requestId)
    giveNeutrino(image);
}

void Camera::giveNeutrino(const QImage& capture) {
    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    bool separate_rgb= my_set.value("separateRGB",false).toBool();
    my_set.endGroup();

    if(!capture.isNull()) {
        // one conversion, then direct access to the lines (QImage::pixel is slow)
        const QImage image=capture.convertToFormat(QImage::Format_RGB32);
        if (!separate_rgb) {
            nPhysD *datamatrix = new nPhysD(image.width(), image.height(),0,"Camera gray");
            datamatrix->setShortName("gray");
            for (int i=0;i<image.height();i++) {
                const QRgb *line=reinterpret_cast<const QRgb*>(image.constScanLine(i));
                for (int j=0;j<image.width();j++) {
                    datamatrix->Timg_matrix[i][j]= (qRed(line[j])+qGreen(line[j])+qBlue(line[j]))/3.0;
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
                const QRgb *line=reinterpret_cast<const QRgb*>(image.constScanLine(i));
                for (int j=0;j<image.width();j++) {
                    loc_red->Timg_matrix[i][j]= qRed(line[j]);
                    loc_gre->Timg_matrix[i][j]= qGreen(line[j]);
                    loc_blu->Timg_matrix[i][j]= qBlue(line[j]);
                }
            }
            loc_red->TscanBrightness();
            loc_gre->TscanBrightness();
            loc_blu->TscanBrightness();
            if (!erasePrevious->isChecked()) {
                for (int i=0;i<3;i++) {
                    imgColor[i]=nullptr;
                }
            }
            imgColor[0]=nparent->replacePhys(loc_red,imgColor[0]);
            imgColor[1]=nparent->replacePhys(loc_gre,imgColor[1]);
            imgColor[2]=nparent->replacePhys(loc_blu,imgColor[2]);
        }
    }
}
