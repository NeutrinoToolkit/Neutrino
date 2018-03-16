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
#include "nGenericPan.h"
#include "ui_nPanHelp.h"

#include "nApp.h"
#include "neutrino.h"
#include <QtSvg>

nGenericPan::nGenericPan(neutrino *myparent)
    : QMainWindow(myparent),
      nparent(myparent),
      napp(qobject_cast<nApp*> (qApp)),
      currentBuffer(myparent->getCurrentBuffer())
{
    setAttribute(Qt::WA_DeleteOnClose);
    connect(qApp,SIGNAL(aboutToQuit()),this,SLOT(saveDefaults()));

    if (nparent==nullptr || napp==nullptr) return;


    setProperty("numpan",nparent->property("numpan").toInt()+1);
    nparent->setProperty("numpan",property("numpan"));

    connect(nparent, SIGNAL(destroyed()), this, SLOT(close()));

    connect(nparent, SIGNAL(mouseAtMatrix(QPointF)), this, SLOT(mouseAtMatrix(QPointF)));
    connect(nparent, SIGNAL(mouseAtWorld(QPointF)), this, SLOT(mouseAtWorld(QPointF)));

    connect(nparent, SIGNAL(nZoom(double)), this, SLOT(nZoom(double)));

    connect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(imageMousePress(QPointF)));
    connect(nparent->my_w->my_view, SIGNAL(mouseReleaseEvent_sig(QPointF)), this, SLOT(imageMouseRelease(QPointF)));

    connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(bufferChanged(nPhysD *)));

    connect(nparent, SIGNAL(physAdd(nPhysD*)), this, SLOT(physAdd(nPhysD*)));
    connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));

    bufferChanged(nparent->getCurrentBuffer());

}

void nGenericPan::raiseIt() {
    setWindowState(windowState() & (~Qt::WindowMinimized | Qt::WindowActive));
    raise();  // for MacOS
    activateWindow(); // for Windows
}

QString nGenericPan::panName() {
    return property("panName").isValid()? property("panName").toString() : QString(metaObject()->className()).replace("_"," ");
}


QString nGenericPan::getNameForCombo(QComboBox* combo, nPhysD *buffer) {
    QString name="";
    if (nparent) {
        int position = nparent->getBufferList().indexOf(buffer);
        name=QString::fromUtf8(buffer->getName().c_str());
        if (!combo->property("NeuSave-physNameLength").isValid()) {
            combo->setProperty("NeuSave-physNameLength",nparent->property("NeuSave-physNameLength"));
        }
        int len=combo->property("NeuSave-physNameLength").toInt();
        qDebug() << combo << len << name << name.length();
        if (name.length()>len) {
            name=name.left((len-5)/2)+"[...]"+name.right((len-5)/2);
        }
        name.prepend(QString::number(position)+" : ");
    }
    return name;
}

void nGenericPan::keyPressEvent(QKeyEvent *event) {
    qDebug() << event;
    event->accept();
}

void nGenericPan::physAdd(nPhysD *buffer) {
    DEBUG("here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>here<>");
    foreach (QComboBox *combo, findChildren<QComboBox *>()) {
        if (combo->property("neutrinoImage").isValid()) {
            int alreadyThere = combo->findData(qVariantFromValue((void*) buffer));
            if (alreadyThere == -1) {
                combo->addItem(getNameForCombo(combo,buffer),qVariantFromValue((void*) buffer));
            }
        }
    }
    QApplication::processEvents();
}


void nGenericPan::raiseNeutrino() {
    if (nparent) {
        nparent->setWindowState(windowState() & (~Qt::WindowMinimized | Qt::WindowActive));
        nparent->raise();  // for MacOS
        nparent->activateWindow(); // for Windows
        QApplication::processEvents();
    }
}

void nGenericPan::help() {
    qDebug() << "\n\n\n" << staticMetaObject.className() << "\n\n\n\n";
    qDebug() << "\n\n\n" << metaObject()->className() << "\n\n\n\n";

    if (helpwin.isNull()) {
        helpwin= new QMainWindow();
        QString helpFile(property("helpFile").toString());
        if (QFileInfo(helpFile).exists()) {
            Ui::PanHelp my_help;
            my_help.setupUi(helpwin);
            helpwin->setWindowTitle(panName()+" help");
            my_help.help->setSource(QUrl("qrc"+helpFile));
            connect(my_help.actionHome, SIGNAL(triggered()), my_help.help, SLOT(home()));
            connect(my_help.actionBack, SIGNAL(triggered()), my_help.help, SLOT(backward()));
            connect(my_help.actionForward, SIGNAL(triggered()), my_help.help, SLOT(forward()));
            connect(my_help.actionPrint, SIGNAL(triggered()), my_help.help, SLOT(print()));
            helpwin->show();
        }
    } else {
        helpwin->setWindowState(windowState() & (~Qt::WindowMinimized | Qt::WindowActive));
        helpwin->raise();  // for MacOS
        helpwin->activateWindow(); // for Windows
    }
}

void nGenericPan::changeEvent(QEvent *e)
{
    qDebug() << panName() << e;

    QWidget::changeEvent(e);
    switch (e->type()) {
        case QEvent::LanguageChange: {
                QMainWindow *my_mainWindow=qobject_cast<QMainWindow *>(nparent);
                if(my_mainWindow) {
                    qDebug() << "found!";
                    for(auto& pan: nparent->getPanList())
                        for(int i =  0; i < pan->metaObject()->methodCount(); ++i) {
                            if (pan->metaObject()->method(i).methodSignature() == "retranslateUi(QMainWindow*)") {
                                qDebug() << "found retranslateUi";
                                QMetaObject::invokeMethod(pan,"retranslateUi",Q_ARG(QMainWindow *,my_mainWindow));
                            }
                        }
                }
                break;
            }
        default:
            break;
    }
}

void nGenericPan::grabSave() {
    int progNum=0;
    while (progNum<10000) {
        QString fname=QDir::homePath()+"/Grab_"+panName()+"_"+QString("%1").arg(progNum++, 5, 10, QChar('0'))+".png";
        if (!QFileInfo(fname).exists()) {
            showMessage(fname);
            setUnifiedTitleAndToolBarOnMac(false);
            grab().save(fname);
            setUnifiedTitleAndToolBarOnMac(true);
            break;
        }
    }
}

void nGenericPan::decorate(QWidget *main_widget) {

    foreach (nPhysD *buffer, nparent->getBufferList()) physAdd(buffer);

    foreach (QComboBox *combo, main_widget->findChildren<QComboBox *>()) {
        if (combo->property("neutrinoImage").isValid()) {
            if (combo->property("neutrinoImage").toBool()) {
                //connect(combo, SIGNAL(currentIndexChanged(int)), this, SLOT(comboChanged(int)));
                connect(combo,SIGNAL(highlighted(int)),this, SLOT(comboChanged(int)));
                connect(combo,SIGNAL(activated(int)),this, SLOT(comboChanged(int)));
            }
        }
    }
    int occurrency=0;
    foreach (QWidget *wdgt, main_widget->findChildren<QWidget *>()) {

        if (wdgt->property("neutrinoSave").isValid() ||
                wdgt->property("neutrinoImage").isValid() ||
                qobject_cast<QPushButton*>(wdgt) ||
                qobject_cast<QToolButton*>(wdgt)
                ) {
            if (wdgt->objectName().isEmpty()) {
                wdgt->setObjectName(wdgt->metaObject()->className()+QString::number(occurrency++));
            }
            wdgt->setToolTip(wdgt->toolTip()+" ["+wdgt->objectName()+"]");
        }
    }

    foreach (QAction *wdgt, main_widget->findChildren<QAction *>()) {
        wdgt->setToolTip(wdgt->toolTip()+" ["+wdgt->objectName()+"]");
    }

}

void nGenericPan::show(bool onlyOneAllowed) {
    if (onlyOneAllowed) {
        for (auto & pan: nparent->getPanList()) {
            qDebug() << this << pan;
            if (pan!=this && pan->metaObject()->className() == metaObject()->className()) {
                pan->close();
                QApplication::processEvents();
            }
        }
    }
    qDebug() << metaObject()->className();

    connect(new QShortcut(QKeySequence(Qt::CTRL + Qt::ALT + Qt::META + Qt::Key_G),this), SIGNAL(activated()), this, SLOT(grabSave()) );

    // these properties will be automatically saved
    setProperty("NeuSave-fileIni",panName()+".ini");
    setProperty("NeuSave-fileTxt",panName()+".txt");

    setWindowTitle(nparent->property("winId").toString()+": "+panName()+" "+property("numpan").toString());

    foreach (QToolBar *my_tool, findChildren<QToolBar *>()) {
        if (my_tool->objectName() == "toolBar") { // this has been created by the designer

            QWidget* spacer = new QWidget();
            spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
            spacer->setProperty("helpSpacer",true);
            my_tool->addWidget(spacer);

            my_tool->addAction(QIcon(":icons/icon.png"),tr("Raise viewer"),this,SLOT(raiseNeutrino()));

            bool needPrefToolButtons=false;
            foreach (QWidget *wdgt, findChildren<QWidget *>()) {
                if (wdgt->property("neutrinoSave").isValid()) {
                    needPrefToolButtons=true;
                    break;
                }
            }
            if(needPrefToolButtons) {
                my_tool->addAction(QIcon(":icons/loadPref.png"),tr("Load preferences"),this,SLOT(loadSettings()));
                my_tool->addAction(QIcon(":icons/savePref.png"),tr("Save preferences"),this,SLOT(saveSettings()));
            }
            QFile helpFile(":/"+panName()+"README.html");
            if (helpFile.exists()) {
                setProperty("helpFile",helpFile.fileName());
                my_tool->addAction(QIcon(":icons/help.png"),tr("Help"),this,SLOT(help()));
            }
            break;
        }
    }

    decorate(this);

    QSize iconSize;
    foreach (QToolBar *widget, nparent->findChildren<QToolBar *>()) {
        iconSize=widget->iconSize();
        widget->show();
        break;
    }
    foreach (QToolBar *widget, findChildren<QToolBar *>()) {
        widget->setIconSize(iconSize);
    }

    QSettings settings("neutrino","");
    settings.beginGroup("nPreferences");
    QVariant fontString=settings.value("defaultFont");
    settings.endGroup();

    if (fontString.isValid()) {
        QFont fontTmp;
        if (fontTmp.fromString(fontString.toString())) {
            foreach (nCustomPlot *my_plot, findChildren<nCustomPlot *>()) {
                foreach (QCPAxisRect *re, my_plot->axisRects()) {
                    foreach (QCPAxis *axis, re->axes()) {
                        axis->setLabelFont(fontTmp);
                        axis->setTickLabelFont(fontTmp);
                    }
                }
                my_plot->setTitleFont(fontTmp);
                my_plot->legend->setFont(fontTmp);
                my_plot->replot();
            }
        }
    }

    if (!(QGuiApplication::keyboardModifiers() & Qt::AltModifier)) {
        QApplication::processEvents();
        loadDefaults();
        QApplication::processEvents();
    }
    nparent->emitPanAdd(this);

    QMainWindow::show();
}

void nGenericPan::physDel(nPhysD * buffer) {
    foreach (QComboBox *combo, findChildren<QComboBox *>()) {
        if (combo->property("neutrinoImage").isValid()) {
            if (combo->property("neutrinoImage").toBool()) {
                //				disconnect(combo, SIGNAL(currentIndexChanged(int)), this, SLOT(comboChanged(int)));
                disconnect(combo,SIGNAL(highlighted(int)),this, SLOT(comboChanged(int)));
                disconnect(combo,SIGNAL(activated(int)),this, SLOT(comboChanged(int)));
            }
            int position=combo->findData(qVariantFromValue((void*) buffer));

            combo->removeItem(position);
            if (combo->property("neutrinoImage").toBool()) {
                //				connect(combo, SIGNAL(currentIndexChanged(int)), this, SLOT(comboChanged(int)));
                connect(combo,SIGNAL(highlighted(int)),this, SLOT(comboChanged(int)));
                connect(combo,SIGNAL(activated(int)),this, SLOT(comboChanged(int)));
            }
        }
    }
}

void nGenericPan::bufferChanged(nPhysD * buffer)
{
    currentBuffer = buffer;
}

void nGenericPan::showMessage(QString message) {
    nparent->statusBar()->showMessage(message);
}

void nGenericPan::showMessage(QString message,int msec) {
    nparent->statusBar()->showMessage(message,msec);
}

void nGenericPan::comboChanged(int k) {
    QComboBox *combo = qobject_cast<QComboBox *>(sender());
    if (combo) {
        nPhysD *image=(nPhysD*) (combo->itemData(k).value<void*>());
        if (image) {
            nparent->showPhys(image);
        }
        emit changeCombo(combo);
    }
}

nPhysD* nGenericPan::getPhysFromCombo(QComboBox* combo) {
    nPhysD* retVal=nullptr;
    QApplication::processEvents();
    if (combo->count())
        retVal = (nPhysD*) (combo->itemData(combo->currentIndex()).value<void*>());
    return retVal;
}

void
nGenericPan::loadUi(QSettings *settings) {
    repaint();
    QApplication::processEvents();
    foreach (QLineEdit *widget, findChildren<QLineEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            widget->setText(settings->value(widget->objectName(),widget->text()).toString());
        }
    }
    foreach (QSlider *widget, findChildren<QSlider *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            widget->setValue(settings->value(widget->objectName(),widget->value()).toInt());
        }
    }
    foreach (QPlainTextEdit *widget, findChildren<QPlainTextEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            widget->clear();
            widget->setPlainText(settings->value(widget->objectName(),widget->toPlainText()).toString());
        }
    }
    foreach (QTextEdit *widget, findChildren<QTextEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            widget->clear();
            widget->insertHtml(settings->value(widget->objectName(),widget->toHtml()).toString());
        }
    }
    foreach (QDoubleSpinBox *widget, findChildren<QDoubleSpinBox *>()) {
        // do not use Locale for storing values
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) widget->setValue(settings->value(widget->objectName(),widget->value()).toDouble());
    }
    foreach (QSpinBox *widget, findChildren<QSpinBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            qDebug() << widget << settings->value(widget->objectName());
            widget->setValue(settings->value(widget->objectName(),widget->value()).toInt());
        }
    }
    foreach (QTabWidget *widget, findChildren<QTabWidget *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) widget->setCurrentIndex(settings->value(widget->objectName(),widget->currentIndex()).toInt());
    }
    foreach (QCheckBox *widget, findChildren<QCheckBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) widget->setChecked(settings->value(widget->objectName(),widget->isChecked()).toBool());
    }
    foreach (QAction *widget, findChildren<QAction *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) widget->setChecked(settings->value(widget->objectName(),widget->isChecked()).toBool());
    }
    foreach (QToolButton *widget, findChildren<QToolButton *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) widget->setChecked(settings->value(widget->objectName(),widget->isChecked()).toBool());
    }
    foreach (QRadioButton *widget, findChildren<QRadioButton *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) widget->setChecked(settings->value(widget->objectName(),widget->isChecked()).toBool());
    }
    foreach (QGroupBox *widget, findChildren<QGroupBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) widget->setChecked(settings->value(widget->objectName(),widget->isChecked()).toBool());
    }
    
    foreach (QComboBox *widget, findChildren<QComboBox *>()) {
        if (widget->property("neutrinoSave").isValid()) {
            QString currText=widget->currentText();
            if (widget->property("neutrinoSave").toBool()) {
                QStringList lista;
                for (int i=0; i< widget->count(); i++) {
                    lista << widget->itemText(i);
                }
                lista << settings->value(widget->objectName(),lista).toStringList();
                lista.removeDuplicates();
                widget->clear();
                widget->addItems(lista);
            }
            widget->setCurrentIndex(settings->value(widget->objectName()+"Default",0).toInt());
        }
        DEBUG("here " << widget->objectName().toStdString());
        if (widget->property("neutrinoImage").isValid() && widget->property("neutrinoImage").toBool()) {
            std::string imageName=settings->value(widget->objectName()).toString().toStdString();
            foreach (nPhysD *physAperto,nparent->getBufferList()) {
                if (physAperto->getName()==imageName) {
                    for (int i=0; i<widget->count();i++) {
                        if (physAperto==(nPhysD*) (widget->itemData(i).value<void*>())) {
                            widget->setCurrentIndex(i);
                            break;
                        }
                    }
                }
            }
        }
    }

    foreach (nCustomPlot *widget, findChildren<nCustomPlot *>()) {
        widget->loadSettings(settings);
    }

    foreach (QObject* widget, nparent->children()) {
        if (widget->parent()==this) {
            nLine *linea=qobject_cast<nLine *>(widget);
            if (linea) {
                linea->loadSettings(settings);
            }
            nRect *rect=qobject_cast<nRect *>(widget);
            if (rect) {
                rect->loadSettings(settings);
            }
            nPoint *point=qobject_cast<nPoint *>(widget);
            if (point) {
                point->loadSettings(settings);
            }
            nEllipse *elli=qobject_cast<nEllipse *>(widget);
            if (elli) {
                elli->loadSettings(settings);
            }
        }
    }

}

void
nGenericPan::saveUi(QSettings *settings) {
    foreach (QLineEdit *widget, findChildren<QLineEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            settings->setValue(widget->objectName(),widget->text());
        }
    }
    foreach (QSlider *widget, findChildren<QSlider *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            settings->setValue(widget->objectName(),widget->value());
        }
    }
    foreach (QPlainTextEdit *widget, findChildren<QPlainTextEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            settings->setValue(widget->objectName(),widget->toPlainText());
        }
    }
    foreach (QTextEdit *widget, findChildren<QTextEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            settings->setValue(widget->objectName(),widget->toHtml());
        }
    }
    foreach (QDoubleSpinBox *widget, findChildren<QDoubleSpinBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) settings->setValue(widget->objectName(),widget->value());
    }
    foreach (QSpinBox *widget, findChildren<QSpinBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            qDebug() << widget->objectName() << widget->value() ;
            settings->setValue(widget->objectName(),widget->value());
        }
    }
    foreach (QTabWidget *widget, findChildren<QTabWidget *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) settings->setValue(widget->objectName(),widget->currentIndex());
    }
    foreach (QCheckBox *widget, findChildren<QCheckBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) settings->setValue(widget->objectName(),widget->isChecked());
    }
    foreach (QAction *widget, findChildren<QAction *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) settings->setValue(widget->objectName(),widget->isChecked());
    }
    foreach (QToolButton *widget, findChildren<QToolButton *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) settings->setValue(widget->objectName(),widget->isChecked());
    }
    foreach (QRadioButton *widget, findChildren<QRadioButton *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) settings->setValue(widget->objectName(),widget->isChecked());
    }
    foreach (QGroupBox *widget, findChildren<QGroupBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) settings->setValue(widget->objectName(),widget->isChecked());
    }
    foreach (QComboBox *widget, findChildren<QComboBox *>()) {
        if (widget->property("neutrinoSave").isValid()) {
            if (widget->property("neutrinoSave").toBool()) {
                QStringList lista;
                for (int i=0; i< widget->count(); i++) {
                    lista << widget->itemText(i);
                }
                settings->setValue(widget->objectName(),lista);
            }
            settings->setValue(widget->objectName()+"Default",widget->currentIndex());
        }

        if (widget->property("neutrinoImage").isValid() && widget->property("neutrinoImage").toBool()) {
            for (int i=0; i< widget->count(); i++) {
                nPhysD *phys=(nPhysD*) (widget->itemData(widget->currentIndex()).value<void*>());
                if (phys) {
                    settings->setValue(widget->objectName(),QString::fromUtf8(phys->getName().c_str()));
                    settings->setValue(widget->objectName()+"-From",QString::fromUtf8(phys->getFromName().c_str()));
                }
            }
        }
    }

    foreach (nCustomPlot *widget, findChildren<nCustomPlot *>()) {
        widget->saveSettings(settings);
    }

    foreach (QObject* widget, nparent->children()) {
        if (widget->parent()==this) {
            nLine *line=qobject_cast<nLine *>(widget);
            if (line) {
                line->saveSettings(settings);
            }
            nRect *rect=qobject_cast<nRect *>(widget);
            if (rect) {
                rect->saveSettings(settings);
            }
            nPoint *point=qobject_cast<nPoint *>(widget);
            if (point) {
                point->saveSettings(settings);
            }
            nEllipse *elli=qobject_cast<nEllipse *>(widget);
            if (elli) {
                elli->saveSettings(settings);
            }
        }
    }
}

void nGenericPan::closeEvent(QCloseEvent*){
    if (!helpwin.isNull()) {
        helpwin->close();
    }
    qDebug() << "Going to close" << this;
    foreach (QComboBox *combo, findChildren<QComboBox *>()) {
        if (combo->property("neutrinoImage").isValid()) {
            if (combo->property("neutrinoImage").toBool()) {
                disconnect(combo, SIGNAL(currentIndexChanged(int)), this, SLOT(comboChanged(int)));
            }
        }
    }
    saveDefaults();
    foreach (QObject* widget, nparent->children()) {
        qDebug() << widget->parent();
        if (widget->parent()==this) {
            QApplication::processEvents();
            nLine *line=qobject_cast<nLine *>(widget);
            if (line) {
                line->my_pad.hide();
                line->deleteLater();
            }
            nRect *rect=qobject_cast<nRect *>(widget);
            if (rect) {
                rect->my_pad.hide();
                rect->deleteLater();
            }
            nPoint *point=qobject_cast<nPoint *>(widget);
            if (point) {
                point->my_pad.hide();
                point->deleteLater();
            }
            nEllipse *elli=qobject_cast<nEllipse *>(widget);
            if (elli) {
                elli->my_pad.hide();
                elli->deleteLater();
            }
            QApplication::processEvents();
        }
    }
//    foreach (QWidget *widget, QApplication::allWidgets()) {
//        neutrino *neu=qobject_cast<neutrino *>(widget);
//        if (neu==nparent) {
//            disconnect(nparent, SIGNAL(mouseAtMatrix(QPointF)), this, SLOT(mouseAtMatrix(QPointF)));
//            disconnect(nparent, SIGNAL(mouseAtWorld(QPointF)), this, SLOT(mouseAtWorld(QPointF)));

//            disconnect(nparent, SIGNAL(nZoom(double)), this, SLOT(nZoom(double)));

//            disconnect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(imageMousePress(QPointF)));
//            disconnect(nparent->my_w->my_view, SIGNAL(mouseReleaseEvent_sig(QPointF)), this, SLOT(imageMouseRelease(QPointF)));

//            disconnect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(bufferChanged(nPhysD *)));

//            disconnect(nparent, SIGNAL(physAdd(nPhysD*)), this, SLOT(physAdd(nPhysD*)));
//            disconnect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
//        }
//    }
    nparent->emitPanDel(this);
    QApplication::processEvents();
}

void nGenericPan::focusOutEvent(QFocusEvent *event) {
    saveDefaults();
    QMainWindow::focusOutEvent(event);
}

//////////////////// SETTINGS
void nGenericPan::loadSettings() {
    QString fnametmp = QFileDialog::getOpenFileName(this, tr("Open INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf);; Any files (*.*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileIni",fnametmp);
        loadSettings(fnametmp);
    }
}

void nGenericPan::loadSettings(QString settingsFile) {
    QSettings settings(settingsFile,QSettings::IniFormat);
    loadSettings(&settings);
}

void nGenericPan::saveSettings() {
    QString fnametmp = QFileDialog::getSaveFileName(this, tr("Save INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileIni",fnametmp);
        QSettings settings(fnametmp,QSettings::IniFormat);
        settings.clear();
        saveSettings(&settings);
    }
}

void nGenericPan::loadDefaults() {
    QSettings settings("neutrino","");
    settings.beginGroup(panName());
    loadSettings(&settings);
    settings.endGroup();
}

void nGenericPan::saveDefaults() {
    QSettings settings("neutrino","");
    settings.beginGroup(panName());
    saveSettings(&settings);
    settings.endGroup();
}

/// THESE are specialized
void nGenericPan::loadSettings(QSettings *settings) {
    if (settings->childGroups().contains("Properties")) {
        settings->beginGroup("Properties");
        foreach(QString my_key, settings->allKeys()) {
            setProperty(my_key.toStdString().c_str(), settings->value(my_key));
        }
        settings->endGroup();
    }
    if (property("NeuSave-locale").isValid()) {
        setLocale(property("NeuSave-locale").toLocale());
    }
    loadUi(settings);
}

void nGenericPan::saveSettings(QSettings *settings) {
    saveUi(settings);
    settings->beginGroup("Properties");
    setProperty("NeuSave-locale",locale());
    foreach(QByteArray ba, dynamicPropertyNames()) {
        if(ba.startsWith("NeuSave")) {
            settings->setValue(ba, property(ba));
        }
    }
    settings->endGroup();
}

// thread run
//
void
nGenericPan::runThread(void *iparams, ifunc my_func, QString title, int max_calc) {
    QProgressDialog progress(title, "Stop", 0, max_calc, this);
    if (max_calc > 0) {
        progress.setWindowModality(Qt::WindowModal);
        progress.show();
    }
    nThread.setThread(iparams,my_func);

    nThread.start();
    while (nThread.isRunning()) {
        if (max_calc > 0) {
            progress.setValue(nThread.n_iter);
            if (progress.wasCanceled()) {
                nThread.stop();
                break;
            }
        }
        QApplication::processEvents();
        QThread::msleep(100);
    }
    progress.setValue(0);
    progress.hide();
    if (nThread.n_iter==0) {
        QMessageBox::critical(this, tr("Thread problems"),nThread.err_message,QMessageBox::Ok);
    }
    
}

bool nGenericPan::nPhysExists(nPhysD* phys){
    return nparent->getBufferList().contains(phys);
}

void nGenericPan::set(QString name, QVariant my_val, int occurrence) {
    bool ok;
    int my_occurrence=1;
    //	foreach (QComboBox *obj, findChildren<QComboBox *>()) {
    //		if (obj->property("neutrinoImage").isValid()&&obj->objectName()==name) {
    //			if (my_occurrence==occurrence) {
    //				bool found=false;
    //				for (int i=0;i<obj->count();i++) {
    //					nPhysD *objPhys=(nPhysD*) (obj->itemData(i).value<void *>());
    //					if (*objPhys == *(nPhysD*) (my_val.value<void *>())){
    //						obj->setCurrentIndex(i);
    //						found=true;
    //					}
    //				}
    //				if (!found) {
    //					nparent->addPhys((nPhysD*) (my_val.value<void *>()));
    //					QApplication::processEvents();
    //					if (obj->findData(my_val)>-1) {
    //						obj->setCurrentIndex(obj->findData(my_val));
    //						return;
    //					} else {
    //						if (obj->findText(my_val.toString())>-1) {
    //							obj->setCurrentIndex(obj->findText(my_val.toString()));
    //							QApplication::processEvents();
    //							return;
    //						}
    //					}
    //				}
    //			}
    //			my_occurrence++;
    //		}
    //	}
    //	my_occurrence=1;
    foreach (QComboBox *obj, findChildren<QComboBox *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                int val=my_val.toInt(&ok);
                if (ok && val>=0 && val < obj->maxVisibleItems()) {
                    obj->setCurrentIndex(val);
                } else {
                    int pos=obj->findData(my_val);
                    if (pos>-1) {
                        obj->setCurrentIndex(pos);
                    }
                }
                return;
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QDoubleSpinBox *obj, findChildren<QDoubleSpinBox *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                // do not use Locale for reading values
                double val=my_val.toDouble(&ok);
                if (ok) {
                    obj->setValue(val);
                    return;
                }
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QSpinBox *obj, findChildren<QSpinBox *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                int val=my_val.toInt(&ok);
                if (ok) {
                    obj->setValue(val);
                    return;
                }
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QPlainTextEdit *obj, findChildren<QPlainTextEdit *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                qDebug() << my_val;
                obj->setPlainText(my_val.toString());
                return;
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QTextEdit *obj, findChildren<QTextEdit *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                qDebug() << my_val;
                obj->setPlainText(my_val.toString());
                return;
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QLineEdit *obj, findChildren<QLineEdit *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                obj->setText(my_val.toString());
                return;
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QCheckBox *obj, findChildren<QCheckBox *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                obj->setChecked(my_val.toBool());
                return;
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QGroupBox *group, findChildren<QGroupBox *>()) {
        if (group->objectName()==name) {
            if (my_occurrence==occurrence) {
                foreach (QRadioButton *obj, group->findChildren<QRadioButton *>()) {
                    obj->setChecked(obj->objectName()==my_val.toString());
                }
                return;
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QTabWidget *obj, findChildren<QTabWidget *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                int val=my_val.toInt(&ok);
                if (ok) {
                    if (val>=0 && val < obj->count()) {
                        obj->setCurrentIndex(val);
                        return;
                    }
                } else {
                    for (int i=0; i< obj->count();i++) {
                        if (obj->tabText(i)==my_val.toString()) {
                            obj->setCurrentIndex(i);
                            return;
                        }
                    }
                }

            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nLine *widget, nparent->findChildren<nLine *>()) {
        if (widget->parent()==this) {
            if (my_occurrence==occurrence) {
                QPolygonF poly;
                foreach (QVariant p, my_val.toList()) {
                    poly << p.toPoint();
                }
                if (poly.size()>1) widget->setPoints(poly);
                return;
            }
            my_occurrence++;
        }

    }
    my_occurrence=1;
    foreach (nRect *widget, nparent->findChildren<nRect *>()) {
        if (widget->parent()==this) {
            if (my_occurrence==occurrence) {
                if (my_val.canConvert(QVariant::RectF)) {
                    widget->setRect(my_val.toRectF());
                    return;
                }
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nPoint *widget, nparent->findChildren<nPoint *>()) {
        if (widget->parent()==this) {
            if (my_occurrence==occurrence) {
                if (my_val.canConvert(QVariant::PointF)) {
                    widget->setPoint(my_val.toPointF());
                    return;
                }
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nEllipse *widget, nparent->findChildren<nEllipse *>()) {
        if (widget->parent()==this) {
            if (my_occurrence==occurrence) {
                if (my_val.canConvert(QVariant::RectF)) {
                    widget->setRect(my_val.toRectF());
                    return;
                }
            }
            my_occurrence++;
        }
    }
}

QStringList nGenericPan::widgets() {
    QStringList retList;
    foreach (QWidget *obj, findChildren<QWidget *>()) {
        retList.append(obj->objectName());
    }
    return retList;
}

QVariant nGenericPan::get(QString name, int occurrence) {
    int my_occurrence=1;
    foreach (QComboBox *obj, findChildren<QComboBox *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                return QVariant(obj->currentIndex());
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QDoubleSpinBox *obj, findChildren<QDoubleSpinBox *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                return QVariant(obj->value());
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QSpinBox *obj, findChildren<QSpinBox *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                return QVariant(obj->value());
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QPlainTextEdit *obj, findChildren<QPlainTextEdit *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                return QVariant(obj->toPlainText());
            }
            my_occurrence++;
        }
    }
    foreach (QTextEdit *obj, findChildren<QTextEdit *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                return QVariant(obj->toPlainText());
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QLineEdit *obj, findChildren<QLineEdit *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                return QVariant(obj->text());
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QCheckBox *obj, findChildren<QCheckBox *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                return QVariant(obj->isChecked());
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QGroupBox *group, findChildren<QGroupBox *>()) {
        if (group->objectName()==name) {
            if (my_occurrence==occurrence) {
                foreach (QRadioButton *obj, group->findChildren<QRadioButton *>()) {
                    if (obj->isChecked()) return QVariant(obj->objectName());
                }
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QTabWidget *obj, findChildren<QTabWidget *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                return QVariant(obj->currentIndex());
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nLine *widget, nparent->findChildren<nLine *>()) {
        if (widget->parent()==this) {
            if (my_occurrence==occurrence) {
                QVariantList variantList;
                foreach (QPointF p, widget->getPoints()) {
                    variantList << p;
                }
                return QVariant(variantList);
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nCustomPlot *widget, findChildren<nCustomPlot *>()) {
        qDebug() << "here" << widget;
        if (widget->objectName()==name) {
            qDebug() << widget << my_occurrence;
            if (my_occurrence==occurrence) {
                return QVariant::fromValue(widget);
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nRect *widget, nparent->findChildren<nRect *>()) {
        if (widget->parent()==this) {
            if (my_occurrence==occurrence) {
                return QVariant(widget->getRectF());
            }
            my_occurrence++;

        }
    }
    my_occurrence=1;
    foreach (nPoint *widget, nparent->findChildren<nPoint *>()) {
        if (widget->parent()==this) {
            if (my_occurrence==occurrence) {
                return QVariant(widget->getPointF());
            }
            my_occurrence++;
        }

    }
    my_occurrence=1;
    foreach (nEllipse *widget, nparent->findChildren<nEllipse *>()) {
        if (widget->parent()==this) {
            if (my_occurrence==occurrence) {
                return QVariant(widget->getRectF());
            }
            my_occurrence++;
        }
    }
    return QVariant();
}

QList<QList<qreal> >  nGenericPan::getData(QString name, int occurrence) {
    QList<QList<qreal> > myListList;
    int my_occurrence=1;
    nPhysD *my_phys=NULL;
    foreach (QComboBox *obj, findChildren<QComboBox *>()) {
        if (obj->property("neutrinoImage").isValid()&&obj->objectName()==name) {
            my_phys=getPhysFromCombo(obj);
            if (my_occurrence==occurrence) {
                if (my_phys) {
                    for (size_t i=0; i<my_phys->getH(); i++) {
                        QList<qreal> myList;
                        for (size_t j=0; j<my_phys->getW(); j++) {
                            myList.append(my_phys->point(j,i));
                        }
                        myListList.append(myList);
                    }
                }
            }
            my_occurrence++;
        }
    }
    return myListList;
}

void nGenericPan::button(QString name , int occurrence) {
    QApplication::processEvents();
    int my_occurrence;
    my_occurrence=1;
    foreach (QPushButton *obj, findChildren<QPushButton *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                obj->click();
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QToolButton *obj, findChildren<QToolButton *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                obj->click();
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (QAction *obj, findChildren<QAction *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {
                obj->trigger();
            }
            my_occurrence++;
        }
    }
}
