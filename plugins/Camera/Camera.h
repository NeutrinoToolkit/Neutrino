// Copyright (C) 2017 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

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

#ifndef CAMERA_H_
#define CAMERA_H_

#include <QAudioInput>
#include <QCamera>
#include <QImageCapture>
#include <QMediaCaptureSession>
#include <QMediaDevices>
#include <QMediaMetaData>
#include <QMediaRecorder>
#include <QScopedPointer>

#include <QMainWindow>
#include "nGenericPan.h"
#include "neutrino.h"
#include "ui_Camera.h"

class Camera : public nGenericPan, private Ui::Camera {
    Q_OBJECT

public:
    Q_INVOKABLE Camera(neutrino *);

private slots:
    void setCamera(const QCameraDevice &cameraDevice);

    void takeImage();
    void displayCaptureError(int, QImageCapture::Error, const QString &errorString);

    void displayCameraError();

    void updateCameraDevice(QAction *action);

    void updateCameras();

    void contextMenuEvent (QContextMenuEvent *) override;

    void processCapturedImage(int requestId, const QImage &img);

    void giveNeutrino(const QImage &img);

public slots:
    void on_grab_clicked();

private:
    nPhysD *imgGray;
    std::array<nPhysD *,3> imgColor;

    QMediaDevices m_devices;
    QMediaCaptureSession m_captureSession;
    QScopedPointer<QCamera> m_camera;
    QImageCapture *m_imageCapture;
    QMenu cameraMenu;
};

NEUTRINO_PLUGIN(Camera,File);

#endif
