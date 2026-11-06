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

// #include "nApp.h"
#include "nCore.h"
#include "neutrino.h"
#include "nLine.h"
#include "nRect.h"
#include "nEllipse.h"
#include "nPoint.h"
#include "nCustomPlots.h"
#include <QtSvg>

nGenericPan::nGenericPan(neutrino *myparent)
    : QMainWindow(myparent),
      nparent(myparent),
      // napp(qobject_cast<nApp*> (qApp)),
      napp(nCore::self()),
      currentBuffer(myparent->getCurrentBuffer())
{
    setAttribute(Qt::WA_DeleteOnClose);
    connect(qApp, &QCoreApplication::aboutToQuit, this, &nGenericPan::saveDefaults);

    if (nparent==nullptr || napp==nullptr) return;

    connect(nparent, &neutrino::destroyed, this, &nGenericPan::close);

    connect(nparent, &neutrino::mouseAtMatrix, this, &nGenericPan::mouseAtMatrix);
    connect(nparent, &neutrino::mouseAtWorld, this, &nGenericPan::mouseAtWorld);

    connect(nparent, &neutrino::nZoom, this, &nGenericPan::nZoom);

    connect(nparent->my_view, &nView::mousePressEvent_sig, this, &nGenericPan::imageMousePress);
    connect(nparent->my_view, &nView::mouseReleaseEvent_sig, this, &nGenericPan::imageMouseRelease);

    connect(nparent, &neutrino::bufferChanged, this, &nGenericPan::bufferChanged);

    connect(nparent, &neutrino::physAdd, this, &nGenericPan::physAdd);
    connect(nparent, &neutrino::physDel, this, &nGenericPan::physDel);
    connect(nparent, &neutrino::physChanged, this, &nGenericPan::refreshComboNames);
    connect(nparent, &neutrino::physReordered, this, &nGenericPan::refreshComboNames);

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


QString nGenericPan::getNameForCombo(QComboBox*, nPhysD *buffer) {
    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    if (!my_set.contains("physNameLength")) {
        my_set.setValue("physNameLength",45);
    }
    int len=my_set.value("physNameLength").toInt();
    my_set.endGroup();
    QString name="";
    if (nparent) {
        name=QString::fromUtf8(buffer->getName().c_str());
        if (name.length()>len) {
            name=name.left((len-5)/2)+"[...]"+name.right((len-5)/2);
        }
        int position = nparent->getBufferList().indexOf(buffer);
        name.prepend(QLocale().toString(position)+" : ");
    }
    return name;
}

void nGenericPan::keyPressEvent(QKeyEvent *event) {
    qDebug() << event;
    event->accept();
}

void nGenericPan::physAdd(nPhysD *buffer) {
    foreach (QComboBox *combo, findChildren<QComboBox *>()) {
        if (combo->property("neutrinoImage").isValid()) {
            QVariant varBuffer=QVariant::fromValue(buffer);
            if (combo->findData(varBuffer) == -1) {
                combo->addItem(getNameForCombo(combo,buffer),varBuffer);
            }
        }
    }
}


void nGenericPan::raiseNeutrino() {
    if (nparent) {
        nparent->setWindowState(windowState() & (~Qt::WindowMinimized | Qt::WindowActive));
        nparent->raise();  // for MacOS
        nparent->activateWindow(); // for Windows
    }
}

void nGenericPan::help() {
    if (helpwin.isNull()) {
        QString helpFile(property("helpFile").toString());
        if (QFileInfo(helpFile).exists()) {
            helpwin= new QMainWindow(this);
            Ui::PanHelp my_help;
            my_help.setupUi(helpwin);
            helpwin->setWindowTitle(panName()+" help");
            my_help.help->setSource(QUrl("qrc"+helpFile));
            connect(my_help.actionHome, &QAction::triggered, my_help.help, &nHelpTextBrowser::home);
            connect(my_help.actionBack, &QAction::triggered, my_help.help, &nHelpTextBrowser::backward);
            connect(my_help.actionForward, &QAction::triggered, my_help.help, &nHelpTextBrowser::forward);
            connect(my_help.actionPrint, &QAction::triggered, my_help.help, &nHelpTextBrowser::print);
            connect(my_help.actionIncreaseFont, &QAction::triggered, my_help.help, &nHelpTextBrowser::zoomIn);
            connect(my_help.actionDecreaseFont, &QAction::triggered, my_help.help, &nHelpTextBrowser::zoomOut);
            helpwin->show();
        }
    } else {
        helpwin->setWindowState(windowState() & (~Qt::WindowMinimized | Qt::WindowActive));
        helpwin->raise();  // for MacOS
        helpwin->activateWindow(); // for Windows
        helpwin->show();
    }
}

void nGenericPan::reopenWithoutSavedValues() {
    const nPanPlug *plug=nPanPlug::find(metaObject()->className());
    QPointer<neutrino> neu=nparent;
    if (!plug || !neu) return;
    close(); // (it saves the current values: the next time the pan opens with them, as usual)
    neu->setProperty("panWithoutSavedValues",true);
    plug->create(neu);
    neu->setProperty("panWithoutSavedValues",QVariant());
}

// draws the object names of the widgets usable from python on top of them
class nWidgetNames : public QWidget {
public:
    nWidgetNames(nGenericPan *pan) : QWidget(pan), my_pan(pan) {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setAttribute(Qt::WA_NoSystemBackground);
        setGeometry(pan->rect());
        pan->installEventFilter(this);
        QTimer *timer=new QTimer(this); // follows the tabs, the scrolls and the layout
        connect(timer, &QTimer::timeout, this, qOverload<>(&QWidget::update));
        timer->start(300);
        raise();
        show();
    }
protected:
    bool eventFilter(QObject *obj, QEvent *ev) override {
        if (obj==my_pan && ev->type()==QEvent::Resize) {
            setGeometry(my_pan->rect());
            raise();
        }
        return false;
    }
    void paintEvent(QPaintEvent *) override {
        QPainter painter(this);
        QFont font=painter.font();
        font.setPointSizeF(std::max(7.0,font.pointSizeF()*0.85));
        painter.setFont(font);
        QFontMetrics fm(font);
        QList<QRect> drawn;
        for (auto &wdgt : my_pan->findChildren<QWidget *>()) {
            QString name=wdgt->objectName();
            if (auto *tool=qobject_cast<QToolButton*>(wdgt)) {
                if (tool->defaultAction()) name=tool->defaultAction()->objectName(); // the actions of the toolbars
            }
            if (name.isEmpty() || name.startsWith("qt_") || !wdgt->isVisible() || wdgt->visibleRegion().isEmpty()) continue;
            // the widgets of decorate (their name is in the tooltip) and the actions of the toolbars
            if (!(wdgt->property("neutrinoSave").isValid() || wdgt->property("neutrinoImage").isValid() ||
                  qobject_cast<QPushButton*>(wdgt) || qobject_cast<QToolButton*>(wdgt))) continue;
            QRect box=fm.boundingRect(name).adjusted(-2,-1,2,1);
            box.moveTopLeft(wdgt->mapTo(my_pan, QPoint(0,0)));
            if (box.right()>width()) box.moveRight(width()-1); // inside the window
            for (int i=0; i<6; i++) { // below the labels already there
                bool overlaps=false;
                for (auto &other : drawn) overlaps |= box.intersects(other);
                if (!overlaps) break;
                box.translate(0,box.height());
            }
            drawn.append(box);
            painter.setPen(QColor(160,140,0));
            painter.setBrush(QColor(255,255,160,230));
            painter.drawRect(box);
            painter.setPen(Qt::black);
            painter.drawText(box, Qt::AlignCenter, name);
        }
    }
private:
    nGenericPan *my_pan;
};

void nGenericPan::showWidgetNames(bool show) {
    if (show) {
        if (!widgetNames) widgetNames=new nWidgetNames(this);
    } else if (widgetNames) {
        widgetNames->deleteLater();
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
        nparent->statusBar()->showMessage("Image saved to file " + fname);
    }
}

void nGenericPan::decorate(QWidget *main_widget) {

    foreach (nPhysD *buffer, nparent->getBufferList()) physAdd(buffer);

    foreach (QComboBox *combo, main_widget->findChildren<QComboBox *>()) {
        if (combo->property("neutrinoImage").isValid()) {
            if (combo->property("neutrinoImage").toBool()) {
                //connect(combo, SIGNAL(currentIndexChanged(int)), this, SLOT(comboChanged(int)));
                connect(combo, &QComboBox::highlighted, this, &nGenericPan::comboChanged);
                connect(combo, &QComboBox::activated, this, &nGenericPan::comboChanged);
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
                wdgt->setObjectName(wdgt->metaObject()->className()+QLocale().toString(occurrency++));
            }
            wdgt->setToolTip(wdgt->toolTip()+" ["+wdgt->objectName()+"]");
        }
    }

    foreach (QAction *wdgt, main_widget->findChildren<QAction *>()) {
        if (!wdgt->objectName().isEmpty()) {
            wdgt->setToolTip(wdgt->toolTip()+" ["+wdgt->objectName()+"]");
        } else {
            wdgt->setToolTip(wdgt->toolTip());
        }
    }

}

void nGenericPan::show(bool onlyOneAllowed) {
    if (onlyOneAllowed) {
        for (auto & pan: nparent->getPanList()) {
            qDebug() << this << pan;
            if (pan!=this && pan->metaObject()->className() == metaObject()->className()) {
                pan->close();
            }
        }
    }
    qDebug() << metaObject()->className();


    // these properties will be automatically saved
    setProperty("NeuSave-fileIni",panName()+".ini");
    setProperty("NeuSave-fileTxt",panName()+".txt");

    setWindowTitle(nparent->property("winId").toString()+": "+panName());

    foreach (QToolBar *my_tool, findChildren<QToolBar *>()) {
        if (my_tool->objectName() == "toolBar") { // this has been created by the designer

            QWidget* spacer = new QWidget();
            spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
            spacer->setProperty("helpSpacer",true);
            my_tool->addWidget(spacer);

            bool needPrefToolButtons=false;
            foreach (QWidget *wdgt, findChildren<QWidget *>()) {
                if (wdgt->property("neutrinoSave").isValid()) {
                    needPrefToolButtons=true;
                    break;
                }
            }
            if(needPrefToolButtons) {
                my_tool->addAction(QIcon(":icons/loadPref.png"),tr("Load preferences"),this,[this]() { loadSettings(); });
                my_tool->addAction(QIcon(":icons/savePref.png"),tr("Save preferences"),this,[this]() { saveSettings(); });
            }
            // the help is named after the plugin directory, as its class (e.g. Ghost_fringes): not panName, which
            // has spaces for the underscores
            QFile helpFile(":/"+QString(metaObject()->className())+"README.html");
            if (!helpFile.exists()) helpFile.setFileName(":/"+panName()+"README.html");
            if (helpFile.exists()) {
                setProperty("helpFile",helpFile.fileName());
                my_tool->addAction(QIcon(":icons/help.png"),tr("Help"),this,&nGenericPan::help);
            }

            // the hidden gestures of the pans, in a menu
            QMenu *menu=new QMenu(this);
            menu->addAction(QIcon(":icons/iconBW.png"),tr("Raise viewer"),this,&nGenericPan::raiseNeutrino);
            menu->addSeparator();
            QAction *act=menu->addAction(QIcon(":icons/savePy.png"),tr("Copy as Python"),this,&nGenericPan::print);
            act->setShortcut(QKeySequence(Qt::CTRL | Qt::ALT | Qt::META | Qt::Key_J));
            act->setToolTip(tr("Copy to the clipboard the python commands that set the values of this window"));
            act=menu->addAction(QIcon(":icons/eye.png"),tr("Show widget names"),this,&nGenericPan::showWidgetNames);
            act->setCheckable(true);
            act->setShortcut(QKeySequence(Qt::CTRL | Qt::ALT | Qt::META | Qt::Key_N));
            act->setToolTip(tr("The names of the widgets, for python (set, get, button)"));
            act=menu->addAction(QIcon(":icons/copyImage.png"),tr("Screenshot to home folder"),this,&nGenericPan::grabSave);
            act->setShortcut(QKeySequence(Qt::CTRL | Qt::ALT | Qt::META | Qt::Key_S));
            if (nPanPlug::find(metaObject()->className())) {
                menu->addSeparator();
                act=menu->addAction(QIcon(":icons/back.png"),tr("Reset (reopen without saved values)"),this,&nGenericPan::reopenWithoutSavedValues);
                act->setToolTip(tr("Close the window and open it again with the default values (as opening it holding Alt)"));
            }
            for (auto &a : menu->actions()) a->setShortcutContext(Qt::WindowShortcut);
            addActions(menu->actions()); // the shortcuts work with the menu closed
            QToolButton *menuButton=new QToolButton(my_tool);
            menuButton->setObjectName("panMenu");
            menuButton->setIcon(QIcon(":icons/gearPy.png"));
            menuButton->setToolTip(tr("Raise viewer, python, screenshot, reset"));
            menuButton->setMenu(menu);
            menuButton->setPopupMode(QToolButton::InstantPopup);
            my_tool->addWidget(menuButton);
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

    // if (!(QGuiApplication::keyboardModifiers() & Qt::AltModifier)) {
    // Alt alone (e.g. Alt+click on the menu) opens the pan without its saved values;
    // not Ctrl+Alt: it is held by the shortcuts of some plugins (ZoomWin Ctrl+Alt+Z, Colorscale Ctrl+Alt+C...)
    // (or reopenWithoutSavedValues)
    if (QGuiApplication::keyboardModifiers() != Qt::AltModifier && !nparent->property("panWithoutSavedValues").toBool()) {
        loadDefaults();
    }
    nparent->emitPanAdd(this);

    QMainWindow::show();
}

void nGenericPan::physDel(nPhysD * buffer) {
    DEBUG(panName().toStdString() <<  " >>>>> enter");
    if (currentBuffer==buffer) currentBuffer=nullptr;
    foreach (QComboBox *combo, findChildren<QComboBox *>()) {
        if (combo->property("neutrinoImage").isValid()) {
            if (combo->property("neutrinoImage").toBool()) {
                disconnect(combo, &QComboBox::highlighted, this, &nGenericPan::comboChanged);
                disconnect(combo, &QComboBox::activated, this, &nGenericPan::comboChanged);
            }
            int position=combo->findData(QVariant::fromValue(buffer));
            combo->removeItem(position);
            if (combo->property("neutrinoImage").toBool()) {
                connect(combo, &QComboBox::highlighted, this, &nGenericPan::comboChanged);
                connect(combo, &QComboBox::activated, this, &nGenericPan::comboChanged);
            }
        }
    }
    refreshComboNames(); // the names show the position in the list
    DEBUG(panName().toStdString() << " >>>>> exit");
}

void nGenericPan::bufferChanged(nPhysD * my_phys)
{
    qDebug() << panName() << "here" << my_phys;
    if (nPhysExists(my_phys)) {
        DEBUG(my_phys->getFromName());
        currentBuffer = my_phys;
    } else {
        currentBuffer = nullptr;
    }

    qDebug() << panName() << "here" ;
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
        nPhysD *image=(nPhysD*) (combo->itemData(k).value<nPhysD*>());
        if (image) {
            nparent->showPhys(image);
        }
    }
}

nPhysD* nGenericPan::getPhysFromCombo(QComboBox* combo) {
    nPhysD* retVal=nullptr;
    if (combo && combo->count()) {
        retVal = (nPhysD*) (combo->itemData(combo->currentIndex()).value<nPhysD*>());
        if (nparent->nPhysExists(retVal)) {
            return retVal;
        }
    }
    return nullptr;
}

void
nGenericPan::loadUi(QSettings &settings) {
    foreach (QLineEdit *widget, findChildren<QLineEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            widget->setText(settings.value(widget->objectName(),widget->text()).toString());
        }
    }
    foreach (QSlider *widget, findChildren<QSlider *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            widget->setValue(settings.value(widget->objectName(),widget->value()).toInt());
        }
    }
    foreach (QPlainTextEdit *widget, findChildren<QPlainTextEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            widget->clear();
            widget->setPlainText(settings.value(widget->objectName(),widget->toPlainText()).toString());
        }
    }
    foreach (QTextEdit *widget, findChildren<QTextEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            widget->clear();
            widget->insertHtml(settings.value(widget->objectName(),widget->toHtml()).toString());
        }
    }
    foreach (QDoubleSpinBox *widget, findChildren<QDoubleSpinBox *>()) {
        // do not use Locale for storing values
        qDebug() << widget;
        qDebug() << settings.value(widget->objectName(),widget->value());
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            double val=settings.value(widget->objectName(),widget->value()).toDouble();
            qDebug() << val;
            widget->setValue(val);
            qDebug() << "done";
        }
        qDebug() << "widget";
    }
    foreach (QSpinBox *widget, findChildren<QSpinBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            qDebug() << widget << settings.value(widget->objectName());
            widget->setValue(settings.value(widget->objectName(),widget->value()).toInt());
        }
    }
    foreach (QTabWidget *widget, findChildren<QTabWidget *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            widget->setCurrentIndex(settings.value(widget->objectName(),widget->currentIndex()).toInt());
            QStringList labels=settings.value(widget->objectName()+"neutrinoLabels").toStringList();
            for (int k=0; k< std::min(static_cast<int>(labels.size()),widget->count());k++) {
                widget->setTabText(k,labels[k]);
            }
        }
    }
    foreach (QCheckBox *widget, findChildren<QCheckBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            if (widget->isTristate()) {
                widget->setCheckState(static_cast<Qt::CheckState>(settings.value(widget->objectName(),widget->checkState()).toInt()));
            } else {
                widget->setChecked(settings.value(widget->objectName(),widget->isChecked()).toBool());
            }
        }
    }
    foreach (QAction *widget, findChildren<QAction *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) widget->setChecked(settings.value(widget->objectName(),widget->isChecked()).toBool());
    }
    foreach (QToolButton *widget, findChildren<QToolButton *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            qDebug() << widget << settings.value(widget->objectName()) << widget->isChecked();
            widget->setChecked(settings.value(widget->objectName(),widget->isChecked()).toBool());
        }
    }
    foreach (QRadioButton *widget, findChildren<QRadioButton *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) widget->setChecked(settings.value(widget->objectName(),widget->isChecked()).toBool());
    }
    foreach (QGroupBox *widget, findChildren<QGroupBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) widget->setChecked(settings.value(widget->objectName(),widget->isChecked()).toBool());
    }

    foreach (QComboBox *widget, findChildren<QComboBox *>()) {
        if (widget->property("neutrinoSave").isValid()) {
            QString currText=widget->currentText();
            if (widget->property("neutrinoSave").toBool()) {
                QStringList lista;
                for (int i=0; i< widget->count(); i++) {
                    lista << widget->itemText(i);
                }
                lista << settings.value(widget->objectName(),lista).toStringList();
                lista.removeDuplicates();
                widget->clear();
                widget->addItems(lista);
            }
            widget->setCurrentIndex(settings.value(widget->objectName()+"Default",0).toInt());
        }
        DEBUG("here " << widget->objectName().toStdString());
        if (widget->property("neutrinoImage").isValid() && widget->property("neutrinoImage").toBool()) {
            std::string imageName=settings.value(widget->objectName()).toString().toStdString();
            foreach (nPhysD *physAperto,nparent->getBufferList()) {
                if (physAperto->getName()==imageName) {
                    for (int i=0; i<widget->count();i++) {
                        if (physAperto==(nPhysD*) (widget->itemData(i).value<nPhysD*>())) {
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
    foreach (nLine *widget, findChildren<nLine *>()) {
        widget->loadSettings(settings);
    }
    foreach (nRect *widget, findChildren<nRect *>()) {
        widget->loadSettings(settings);
    }
    foreach (nEllipse *widget, findChildren<nEllipse *>()) {
        widget->loadSettings(settings);
    }
    foreach (nPoint *widget, findChildren<nPoint *>()) {
        widget->loadSettings(settings);
    }
}

void
nGenericPan::saveUi(QSettings &settings) {
    foreach (QLineEdit *widget, findChildren<QLineEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            settings.setValue(widget->objectName(),widget->text());
        }
    }
    foreach (QSlider *widget, findChildren<QSlider *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            settings.setValue(widget->objectName(),widget->value());
        }
    }
    foreach (QPlainTextEdit *widget, findChildren<QPlainTextEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            settings.setValue(widget->objectName(),widget->toPlainText());
        }
    }
    foreach (QTextEdit *widget, findChildren<QTextEdit *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            settings.setValue(widget->objectName(),widget->toHtml());
        }
    }
    foreach (QDoubleSpinBox *widget, findChildren<QDoubleSpinBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) settings.setValue(widget->objectName(),widget->value());
    }
    foreach (QSpinBox *widget, findChildren<QSpinBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            qDebug() << widget->objectName() << widget->value() ;
            settings.setValue(widget->objectName(),widget->value());
        }
    }
    foreach (QTabWidget *widget, findChildren<QTabWidget *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            settings.setValue(widget->objectName(),widget->currentIndex());
            QStringList labels;
            for (int k=0; k<widget->count();k++) {
                labels.append(widget->tabText(k));
            }
            settings.setValue(widget->objectName()+"neutrinoLabels",labels);
        }
    }
    foreach (QCheckBox *widget, findChildren<QCheckBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            if (widget->isTristate()) {
                settings.setValue(widget->objectName(),widget->checkState());
            } else {
                settings.setValue(widget->objectName(),widget->isChecked());
            }

        }
    }
    foreach (QAction *widget, findChildren<QAction *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) settings.setValue(widget->objectName(),widget->isChecked());
    }
    foreach (QToolButton *widget, findChildren<QToolButton *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) {
            qDebug() << widget << widget->isChecked();
            settings.setValue(widget->objectName(),widget->isChecked());
        }
    }
    foreach (QRadioButton *widget, findChildren<QRadioButton *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) settings.setValue(widget->objectName(),widget->isChecked());
    }
    foreach (QGroupBox *widget, findChildren<QGroupBox *>()) {
        if (widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) settings.setValue(widget->objectName(),widget->isChecked());
    }
    foreach (QComboBox *widget, findChildren<QComboBox *>()) {
        if (widget->property("neutrinoSave").isValid()) {
            if (widget->property("neutrinoSave").toBool()) {
                QStringList lista;
                for (int i=0; i< widget->count(); i++) {
                    lista << widget->itemText(i);
                }
                settings.setValue(widget->objectName(),lista);
            }
            settings.setValue(widget->objectName()+"Default",widget->currentIndex());
        }

        if (widget->property("neutrinoImage").isValid() && widget->property("neutrinoImage").toBool()) {
            for (int i=0; i< widget->count(); i++) {
                nPhysD *phys=(nPhysD*) (widget->itemData(widget->currentIndex()).value<nPhysD*>());
                if (nparent && nparent->nPhysExists(phys)) {
                    settings.setValue(widget->objectName(),QString::fromUtf8(phys->getName().c_str()));
                    settings.setValue(widget->objectName()+"-From",QString::fromUtf8(phys->getFromName().c_str()));
                }
            }
        }
    }

    foreach (nCustomPlot *widget, findChildren<nCustomPlot *>()) {
        widget->saveSettings(settings);
    }

    foreach (nLine *widget, findChildren<nLine *>()) {
        widget->saveSettings(settings);
    }

    foreach (nRect *widget, findChildren<nRect *>()) {
        widget->saveSettings(settings);
    }

    foreach (nEllipse *widget, findChildren<nEllipse *>()) {
        widget->saveSettings(settings);
    }

    foreach (nPoint *widget, findChildren<nPoint *>()) {
        widget->saveSettings(settings);
    }

}

bool nGenericPan::event(QEvent *event) {
    if (event->type()==QEvent::DeferredDelete) nCore::disconnectChildren(this);
    return QMainWindow::event(event);
}

void nGenericPan::closeEvent(QCloseEvent *event){
    // during runThread (the pan is not modal any more): stop the calculation, the pan stays open (its caller
    // is still running and uses it)
    if (nThread.isRunning()) {
        nThread.stop();
        event->ignore();
        return;
    }
    qDebug() << "Going to close" << this;
    foreach (QComboBox *combo, findChildren<QComboBox *>()) {
        if (combo->property("neutrinoImage").isValid()) {
            if (combo->property("neutrinoImage").toBool()) {
                disconnect(combo, &QComboBox::currentIndexChanged, this, &nGenericPan::comboChanged); // (never connected)
            }
        }
    }
    saveDefaults();
    foreach (QObject* widget, nparent->children()) {
        if (widget->parent()==this) {
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
        }
    }
//    foreach (QWidget *widget, QApplication::allWidgets()) {
//        neutrino *neu=qobject_cast<neutrino *>(widget);
//        if (neu==nparent) {
//            disconnect(nparent, SIGNAL(mouseAtMatrix(QPointF)), this, SLOT(mouseAtMatrix(QPointF)));
//            disconnect(nparent, SIGNAL(mouseAtWorld(QPointF)), this, SLOT(mouseAtWorld(QPointF)));

//            disconnect(nparent, SIGNAL(nZoom(double)), this, SLOT(nZoom(double)));

//            disconnect(nparent->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(imageMousePress(QPointF)));
//            disconnect(nparent->my_view, SIGNAL(mouseReleaseEvent_sig(QPointF)), this, SLOT(imageMouseRelease(QPointF)));

//            disconnect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(bufferChanged(nPhysD *)));

//            disconnect(nparent, SIGNAL(physAdd(nPhysD*)), this, SLOT(physAdd(nPhysD*)));
//            disconnect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
//        }
//    }
    nparent->emitPanDel(this);
    deleteLater();
}

void nGenericPan::focusOutEvent(QFocusEvent *event) {
    saveDefaults();
    QMainWindow::focusOutEvent(event);
}

//////////////////// SETTINGS
void nGenericPan::loadSettings(QString fname) {
    qDebug() << fname;
    if (fname.isNull()) {
        QString fname = QFileDialog::getOpenFileName(this, tr("Open INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf);; Any file (*.*)"));
        if (!fname.isNull()) {
            setProperty("NeuSave-fileIni",fname);
            loadSettings(fname);
        }
    } else {
        QSettings settings(fname,QSettings::IniFormat);
        loadSettings(settings);
    }
}

void nGenericPan::saveSettings(QString fname) {
    if (fname.isNull()) {
        fname = QFileDialog::getSaveFileName(this, tr("Save INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf)"));
        if (!fname.isNull()){
            setProperty("NeuSave-fileIni",fname);
            saveSettings(fname);
        }
    } else {
        QSettings settings(fname,QSettings::IniFormat);
        settings.clear();
        saveSettings(settings);
    }
}

void nGenericPan::loadDefaults() {
    QSettings settings("neutrino","");
    settings.beginGroup(panName());
    qDebug() << panName() << " : " << settings.fileName();
    loadSettings(settings);
    qDebug() << "end loadDefaults";
    settings.endGroup();
}

void nGenericPan::saveDefaults() {
    QSettings settings("neutrino","");
    settings.beginGroup(panName());
    saveSettings(settings);
    settings.endGroup();
}

/// THESE are specialized
void nGenericPan::loadSettings(QSettings &settings) {
    if (settings.childGroups().contains("Properties")) {
        settings.beginGroup("Properties");
        qDebug() << "----------------- Properties ----------------- " << panName();
        foreach(QString my_key, settings.allKeys()) {
            qDebug() << my_key << settings.value(my_key);
            setProperty(my_key.toStdString().c_str(), settings.value(my_key));
        }
        qDebug() << "----------------- END Properties ----------------- " << panName();
        settings.endGroup();
    }
    loadUi(settings);
}

void nGenericPan::saveSettings(QSettings &settings) {
    saveUi(settings);
    settings.beginGroup("Properties");
    qDebug() << "----------------- Properties ----------------- " << panName();
    foreach(QByteArray ba, dynamicPropertyNames()) {
        if(ba.startsWith("NeuSave")) {
            qDebug() << ba << property(ba);
            settings.setValue(ba, property(ba));
        }
    }
    qDebug() << "----------------- END Properties ----------------- " << panName();
    settings.endGroup();
}

// thread run
//
void
nGenericPan::runThread(void *iparams, ifunc my_func, QString title, int max_calc) {
    // the progress is in the status bar of the pan, with a Stop button: a modal dialog took the focus (e.g. from the
    // main window while moving a line with auto update). The pan is disabled meanwhile, as the dialog did, except
    // its status bar; the main window stays usable (the calculations work on copies of the images)
    bool canceled=false;
    QProgressBar *bar=nullptr;
    QToolButton *stop=nullptr;
    if (max_calc > 0) {
        bar=new QProgressBar();
        bar->setRange(0,max_calc);
        bar->setFormat(title+" %p%");
        bar->setMaximumHeight(statusBar()->fontMetrics().height()+4);
        stop=new QToolButton();
        stop->setText(tr("Stop"));
        connect(stop, &QToolButton::clicked, this, [&canceled]() { canceled=true; });
        statusBar()->addPermanentWidget(bar,1);
        statusBar()->addPermanentWidget(stop);
    }
    QList<QPointer<QWidget>> disabled;
    for (auto &w : findChildren<QWidget *>(Qt::FindDirectChildrenOnly)) {
        if (w!=statusBar() && w->isEnabled()) {
            w->setEnabled(false);
            disabled << w;
        }
    }

    nThread.setThread(iparams,my_func);

    nThread.start();
    // wait() with a timeout instead of spinning on processEvents: a busy GUI thread steals a core from the OpenMP loops
    while (!nThread.wait(20)) {
        if (bar) bar->setValue(nThread.n_iter);
        if (canceled) {
            nThread.stop();
            break;
        }
        QApplication::processEvents();
    }
    // the calculation stops at its next check of n_iter: wait for it, the caller owns its data
    nThread.wait();
    if (canceled) {
        nThread.n_iter=-1; // a running iteration may have incremented it again
    }
    delete bar;
    delete stop;
    for (auto &w : disabled) {
        if (w) w->setEnabled(true);
    }
    if (!canceled && nThread.n_iter==0) {
        QMessageBox::critical(this, tr("Thread problems"),nThread.err_message,QMessageBox::Ok);
    }

}

bool nGenericPan::nPhysExists(nPhysD* phys){
    return nparent->getBufferList().contains(phys);
}

void nGenericPan::set(QString name, QVariant my_val, int occurrence) {
    bool ok;
    int my_occurrence=1;
    foreach (QComboBox *obj, findChildren<QComboBox *>()) {
        if (obj->objectName()==name) {
            if (my_occurrence==occurrence) {

                nPhysD* phys = static_cast<nPhysD*>(my_val.value<void*>());
                if (phys && obj->property("neutrinoImage").isValid()) {
                    qDebug()<<"trovato" << QString::fromStdString(phys->getShortName());
                    for( int pos = 0; pos < obj->count(); pos++ ) {
                        nPhysD* combophys = static_cast<nPhysD*>(obj->itemData(pos).value<nPhysD*>());
                        if (combophys==phys) {
                            qDebug()<<"trovato davvero" << pos;
                            obj->setCurrentIndex(pos);
                            return;
                        }
                    }
                }


                int val=my_val.toInt(&ok);
                if (ok && val>=0 && val < obj->maxVisibleItems()) {
                    obj->setCurrentIndex(val);
                } else {
                    int pos=obj->findData(my_val);
                    if (pos>-1) {
                        obj->setCurrentIndex(pos);
                    }
                }
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
    // objects
    my_occurrence=1;
    foreach (nLine *widget, nparent->findChildren<nLine *>()) {
        if (widget->parent()==this) {
            if (my_occurrence==occurrence && widget->toolTip() == name) {
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
        if (widget->parent() == this && widget->toolTip() == name) {
            if (my_occurrence==occurrence) {
                qDebug() << "pippo nrect" << QMetaType::fromName("RectF");
                if (my_val.canConvert(QMetaType::fromName("RectF"))) {
                    qDebug() << "here" << my_val.toRectF();
                    // if (my_val.canConvert(QVariant::RectF)) {
                    widget->setRect(my_val.toRectF());
                    return;
                }
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nPoint *widget, nparent->findChildren<nPoint *>()) {
        if (widget->parent()==this && widget->toolTip() == name) {
            if (my_occurrence==occurrence) {
                qDebug() << "pippo npoint" << QMetaType::fromName("PointF");
                if (my_val.canConvert(QMetaType::fromName("PointF"))) {
                // if (my_val.canConvert(QVariant::PointF)) {
                    widget->setPoint(my_val.toPointF());
                    return;
                }
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nEllipse *widget, nparent->findChildren<nEllipse *>()) {
        if (widget->parent()==this && widget->toolTip() == name) {
            if (my_occurrence==occurrence) {
                qDebug() << "pippo nellipse" << QMetaType::fromName("RectF");
                if (my_val.canConvert(QMetaType::fromName("RectF"))) {
                    // if (my_val.canConvert(QVariant::RectF)) {
                    widget->setRect(my_val.toRectF());
                    return;
                }
            }
            my_occurrence++;
        }
    }
}

// the status of the pan as commands of the neutrino python module
void nGenericPan::print() {
    auto quoted = [](QString str) {
        return "\""+str.replace("\\","\\\\").replace("\"","\\\"").replace("\n","\\n")+"\"";
    };
    int occurrence=0;
    for (auto &pan : nparent->getPanList()) {
        if (pan->panName()==panName()) occurrence++;
        if (pan==this) break;
    }
    QString out="import neutrino\npan = neutrino.windows()["+QString::number(nCore::self()->neus().indexOf(nparent))+
                  "].getPan("+quoted(panName())+(occurrence>1 ? ", "+QString::number(occurrence) : "")+")\n";
    foreach (QWidget *widget, findChildren<QWidget *>()) {
        if ((widget->property("neutrinoSave").isValid() && widget->property("neutrinoSave").toBool()) ||
            (widget->property("neutrinoImage").isValid()) ) {
            QString name=widget->objectName();
            QString val=get(name).toString();
            out+="pan.set("+quoted(name)+", "+quoted(val)+")\n";
        }
    }
    QApplication::clipboard()->setText(out);
    nparent->statusBar()->showMessage("Copied to clipboard");
    qWarning().noquote() << out;
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
    // objects
    my_occurrence=1;
    foreach (nLine *widget, nparent->findChildren<nLine *>()) {
        if (widget->parent()==this) {
            if (my_occurrence==occurrence && widget->toolTip() == name) {
                return QVariant::fromValue(widget);
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nCustomPlot *widget, findChildren<nCustomPlot *>()) {
        qDebug() << "here" << widget;
        if (widget->objectName()==name && widget->toolTip() == name) {
            qDebug() << widget << my_occurrence;
            if (my_occurrence==occurrence) {
                return QVariant::fromValue(widget);
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nRect *widget, nparent->findChildren<nRect *>()) {
        if (widget->parent()==this && widget->toolTip() == name) {
            if (my_occurrence==occurrence) {
                return QVariant::fromValue(widget);
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nPoint *widget, nparent->findChildren<nPoint *>()) {
        if (widget->parent()==this && widget->toolTip() == name) {
            if (my_occurrence==occurrence) {
                return QVariant::fromValue(widget);
            }
            my_occurrence++;
        }
    }
    my_occurrence=1;
    foreach (nEllipse *widget, nparent->findChildren<nEllipse *>()) {
        if (widget->parent()==this && widget->toolTip() == name) {
            if (my_occurrence==occurrence) {
                return QVariant::fromValue(widget);
            }
            my_occurrence++;
        }
    }
    return QVariant();
}

QList<QList<qreal> >  nGenericPan::getData(QString name, int occurrence) {
    QList<QList<qreal> > myListList;
    int my_occurrence=1;
    nPhysD *my_phys=nullptr;
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

QVariantMap nGenericPan::results(QVariant what) {
    QVariantMap ret;
    for (auto &plot : findChildren<nCustomPlot *>()) {
        if (what.isValid() && plot->objectName()!=what.toString()) continue;
        QVariantMap graphs;
        for (int i=0; i<plot->graphCount(); i++) {
            QCPGraph *graph=plot->graph(i);
            QList<double> x, y;
            for (auto it=graph->data()->constBegin(); it!=graph->data()->constEnd(); it++) {
                x.append(it->key);
                y.append(it->value);
            }
            QString name=graph->name().isEmpty() ? "graph"+QString::number(i) : graph->name();
            while (graphs.contains(name)) name+="_"+QString::number(i);
            graphs[name]=QVariantMap({{"x",QVariant::fromValue(x)},{"y",QVariant::fromValue(y)}});
        }
        if (!graphs.isEmpty()) {
            QString name=plot->objectName();
            while (ret.contains(name)) name+="_";
            ret[name]=graphs;
        }
    }
    if (what.isValid()) return ret.isEmpty() ? ret : ret.first().toMap(); // only that plot: its graphs
    return ret;
}

void nGenericPan::button(QString name , int occurrence) {
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


// image combos show "position : name": refresh them when an image is renamed, removed or the list is reordered
void nGenericPan::refreshComboNames() {
    foreach (QComboBox *combo, findChildren<QComboBox *>()) {
        if (combo->property("neutrinoImage").isValid()) {
            for (int i=0; i<combo->count(); i++) {
                nPhysD *phys=combo->itemData(i).value<nPhysD*>();
                if (nPhysExists(phys)) {
                    QString name=getNameForCombo(combo,phys);
                    if (combo->itemText(i)!=name) combo->setItemText(i,name);
                }
            }
        }
    }
}