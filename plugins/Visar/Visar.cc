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
#include "Visar.h"
#include "nCustomPlots.h"
#include "nPhysD.h"
#include "neutrino.h"

#include "ui_VisarSettings.h"
#include "ui_VisarVelocity.h"
#include <cmath>

VisarPhasePlot::VisarPhasePlot(QWidget* parent):
    nCustomPlotMouseX3Y(parent) {

    xAxis->setLabel(tr("Pixel"));
    yAxis->setLabel(tr("Fringeshift"));
    yAxis2->setLabel(tr("Intensity"));
    yAxis3->setLabel(tr("Contrast"));

    QPen pen;
    for (unsigned int k=0; k<2; k++) {
        QString my_type(k==0?"reference":"shot");
        QCPGraph* graph;
        graph = addGraph(xAxis, yAxis3);
        graph->setName("Contrast "+my_type);
        pen.setColor(yAxis3->labelColor());
        graph->setPen(pen);

        graph = addGraph(xAxis, yAxis2);
        graph->setName("Intensity "+my_type);
        pen.setColor(yAxis2->labelColor());
        graph->setPen(pen);

        graph = addGraph(xAxis, yAxis);
        graph->setName("Fringeshift "+my_type);
        pen.setColor(yAxis->labelColor());
        graph->setPen(pen);
    }
}

VisarPlot::VisarPlot(QWidget* parent):
    nCustomPlotMouseX3Y(parent) {
    xAxis->setLabel(tr("Time"));
    yAxis->setLabel(tr("Velocity"));
    yAxis2->setLabel(tr("Reflectivity"));
    yAxis3->setLabel(tr("Quality"));
}


nSOPPlot::nSOPPlot(QWidget* parent):
    nCustomPlotMouseX2Y(parent) {
    QCPGraph* graph;
    graph = addGraph(xAxis, yAxis);
    graph->setName("SOP");
    graph->setPen(QPen(yAxis->labelColor()));

    graph = addGraph(xAxis, yAxis2);
    graph->setName("SOP 1");
    graph->setPen(QPen(yAxis2->labelColor()));

    graph = addGraph(xAxis, yAxis2);
    graph->setName("SOP 2");
    graph->setPen(QPen(yAxis2->labelColor(),0.3));

}

Visar::Visar(neutrino *parent) : nGenericPan(parent)
{
    setupUi(this);

    //!START SOP stuff
    sopRect =  new nRect(this,1);
    sopRect->changeToolTip("SOP region");
    sopRect->setRect(QRectF(0,0,100,100));
    connect(actionRect3, &QAction::triggered, sopRect, &nRect::togglePadella);
    sopPlot->xAxis->setLabel(tr("Time"));
    sopPlot->yAxis->setLabel(tr("Counts"));
    sopPlot->yAxis2->setLabel(tr("Temperature"));

    toolBar->insertWidget(actionNext,comboShot);

    show();


    setProperty("NeuSave-alphagraph",50);


    for (int l=2+visars.size(); l<whichRefl->count();l++) {
        whichRefl->removeItem(l);
    }

    unsigned int kMax=1;
    if (property("NeuSave-numVisars").isValid()) {
        kMax=property("NeuSave-numVisars").toUInt();
    }
    for (unsigned int k=0; k<kMax; k++) {
        addVisar();
    }

    connect(actionDoWavelets, SIGNAL(triggered()), this, SLOT(doWave()));
    connect(actionAddVisar, SIGNAL(triggered()), this, SLOT(addVisar()));
    connect(actionDelVisar, SIGNAL(triggered()), this, SLOT(delVisar()));

    connect(actionSaveTxt, SIGNAL(triggered()), this, SLOT(export_txt()));
    connect(actionSaveTxtMultiple, SIGNAL(triggered()), this, SLOT(export_txt_multiple()));
    connect(actionCopy, SIGNAL(triggered()), this, SLOT(export_clipboard()));
    connect(actionCopyImage, SIGNAL(triggered()), this, SLOT(copy_image()));

    connect(etalon_thickness, &QDoubleSpinBox::valueChanged, this, &Visar::calculate_etalon);
    connect(etalon_dn_over_dlambda, &QDoubleSpinBox::valueChanged, this, &Visar::calculate_etalon);
    connect(etalon_n0, &QDoubleSpinBox::valueChanged, this, &Visar::calculate_etalon);
    connect(etalon_lambda, &QDoubleSpinBox::valueChanged, this, &Visar::calculate_etalon);

    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(setObjectVisibility(nPhysD*)));

    connect(tabs, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));
    connect(tabPhase, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));
    connect(tabVelocity, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));

    connect(sopRect, SIGNAL(sceneChanged()), this, SLOT(updatePlotSOP()));
    connect(sopRef, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    connect(sopShot, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    connect(sopTimeOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    connect(sopOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    connect(sopOrigin, SIGNAL(valueChanged(int)), this, SLOT(updatePlotSOP()));
    connect(sopDirection, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    connect(sopCalibT0, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    connect(sopCalibA, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    connect(whichRefl, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    connect(enableSOP, SIGNAL(toggled(bool)), this, SLOT(updatePlotSOP()));

    connect(actionRefreshComboShot, SIGNAL(triggered()), this, SLOT(fillComboShot()));

    connect(plotVelocity,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    connect(sopPlot,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    connect(sopScale, &QLineEdit::editingFinished, this, &Visar::sweepChangedSOP);
    connect(sopScaleType, &QComboBox::currentIndexChanged, this, &Visar::sweepChangedSOP);

    connect(comboShot, SIGNAL(currentTextChanged(QString)), this, SLOT(changeShot(QString)));

    connect(globDirRef, SIGNAL(textChanged(QString)), this, SLOT(fillComboShot()));
    connect(globDirShot, SIGNAL(textChanged(QString)), this, SLOT(fillComboShot()));
    connect(globRef, SIGNAL(textChanged(QString)), this, SLOT(fillComboShot()));
    connect(globShot, SIGNAL(textChanged(QString)), this, SLOT(fillComboShot()));
    connect(globRefresh, &QAbstractButton::released, this, [this]() { refreshGlob(-1); });

    loadDefaults();
    sweepChanged();
    calculate_etalon();
}

void Visar::on_actionNext_triggered() {
    if(comboShot->isVisible()) {
        QStringList itemsInComboBox;
        for (int index = 0; index < comboShot->count(); index++)
            itemsInComboBox << comboShot->itemText(index);
        int nextpos=(comboShot->currentIndex()+1)%comboShot->count();
        comboShot->setCurrentIndex(nextpos);
    }
}

void Visar::on_actionPrevious_triggered() {
    if(comboShot->isVisible()) {
        QStringList itemsInComboBox;
        for (int index = 0; index < comboShot->count(); index++)
            itemsInComboBox << comboShot->itemText(index);
        int nextpos=(comboShot->currentIndex()+comboShot->count()-1)%comboShot->count();
        comboShot->setCurrentIndex(nextpos);
    }
}

void Visar::changeShot(QString num) {
    qDebug() << num;
    QSet<nPhysD*> old_phys;
    for (unsigned int k=0; k< visars.size(); k++) {
        old_phys << getPhysFromCombo(visars[k]->settingsUi->refImage) << getPhysFromCombo(visars[k]->settingsUi->shotImage) ;
        old_phys << getPhysFromCombo(sopRef) << getPhysFromCombo(sopShot) ;
    }
    for(auto & phys : old_phys) {
        nparent->removePhys(phys);
    }

    for (unsigned int k=0; k< visars.size(); k++) {
        if (visars[k]->velocityUi->enableVisar->isChecked()) {
            QSet<QString> matchRef;
            QSet<QString> matchShot;
            QFileInfoList listref = QDir(visars[k]->settingsUi->globDirRef->text()).entryInfoList(QDir::Files);
            foreach(QFileInfo finfo, listref) {
                QString fname=finfo.fileName();
                QRegularExpressionMatch my_match=QRegularExpression(visars[k]->settingsUi->globRef->text()).match(fname);
                if (my_match.lastCapturedIndex()==1) {
                    if (my_match.captured(1) == num) {
                        getPhysFromNameSetCombo(finfo,visars[k]->settingsUi->refImage);
                    }
                }
            }
            QFileInfoList listshot = QDir(visars[k]->settingsUi->globDirShot->text()).entryInfoList(QDir::Files);
            foreach(QFileInfo finfo, listshot) {
                QString fname=finfo.fileName();
                QRegularExpressionMatch my_match=QRegularExpression(visars[k]->settingsUi->globShot->text()).match(fname);
                if (my_match.lastCapturedIndex()==1) {
                    if (my_match.captured(1) == num) {
                        getPhysFromNameSetCombo(finfo,visars[k]->settingsUi->shotImage);
                    }
                }
            }
        }
    }
    doWave();
    // SOP
    if (enableSOP->isChecked()) {
        QSet<QString> matchRef;
        QSet<QString> matchShot;
        QFileInfoList listref = QDir(globDirRef->text()).entryInfoList(QDir::Files);
        foreach(QFileInfo finfo, listref) {
            QString fname=finfo.fileName();
            QRegularExpressionMatch my_match;
            my_match=QRegularExpression(globRef->text()).match(fname);
            if (my_match.lastCapturedIndex()==1) {
                if (my_match.captured(1) == num) {
                    getPhysFromNameSetCombo(finfo,sopRef);
                }
            }
        }
        if (globDirShot->text() != globDirRef->text() || globShot->text() != globRef->text()) {
            QFileInfoList listshot = QDir(globDirShot->text()).entryInfoList(QDir::Files);
            foreach(QFileInfo finfo, listshot) {
                QString fname=finfo.fileName();
                QRegularExpressionMatch my_match;
                my_match=QRegularExpression(globShot->text()).match(fname);
                if (my_match.lastCapturedIndex()==1) {
                    if (my_match.captured(1) == num) {
                        getPhysFromNameSetCombo(finfo,sopShot);
                    }
                }
            }
        } else {
            sopShot->setCurrentIndex(sopRef->currentIndex());
        }
        updatePlotSOP();
    }

    tabChanged(0);

    QApplication::processEvents();
}

void Visar::fillComboShot() {
    disconnect(comboShot, SIGNAL(currentTextChanged(QString)), this, SLOT(changeShot(QString)));
    QString oldvalue=comboShot->currentText();

    QSet<QString> match;
    for (unsigned int k=0; k< visars.size(); k++) {
        if (visars[k]->velocityUi->enableVisar->isChecked() && (!visars[k]->settingsUi->globDirRef->text().isEmpty()) && (!visars[k]->settingsUi->globDirShot->text().isEmpty())) {
            QSet<QString> matchRef;
            QSet<QString> matchShot;
            QFileInfoList listref = QDir(visars[k]->settingsUi->globDirRef->text()).entryInfoList(QDir::Files);
            foreach(QFileInfo finfo, listref) {
                QString fname=finfo.fileName();
                QRegularExpressionMatch my_match;
                my_match=QRegularExpression(visars[k]->settingsUi->globRef->text()).match(fname);
                if (my_match.lastCapturedIndex()==1) {
                    matchRef << my_match.captured(1);
                }
            }
            QFileInfoList listshot = QDir(visars[k]->settingsUi->globDirShot->text()).entryInfoList(QDir::Files);
            foreach(QFileInfo finfo, listshot) {
                QString fname=finfo.fileName();
                QRegularExpressionMatch my_match;
                my_match=QRegularExpression(visars[k]->settingsUi->globShot->text()).match(fname);
                if (my_match.lastCapturedIndex()==1) {
                    matchShot << my_match.captured(1);
                }
            }
            if (matchRef.size()==0 || matchShot.size()==0) {
                statusbar->showMessage("Error: Visar "+QString::number(k+1)+" "+ QString::number(matchRef.size()) + " ref and " + QString::number(matchShot.size()) + " shot images", 1000);
            }
            matchRef.intersect(matchShot);
            match+=matchRef;
        }
    }
    if (enableSOP->isChecked() && (!globDirRef->text().isEmpty()) && (!globDirShot->text().isEmpty())) {
        QSet<QString> matchRef;
        QSet<QString> matchShot;
        QFileInfoList listref = QDir(globDirRef->text()).entryInfoList(QDir::Files);
        foreach(QFileInfo finfo, listref) {
            QString fname=finfo.fileName();
            QRegularExpressionMatch my_match;
            my_match=QRegularExpression(globRef->text()).match(fname);
            if (my_match.lastCapturedIndex()==1) {
                matchRef << my_match.captured(1);
            }
        }
        QFileInfoList listshot = QDir(globDirShot->text()).entryInfoList(QDir::Files);
        foreach(QFileInfo finfo, listshot) {
            QString fname=finfo.fileName();
            QRegularExpressionMatch my_match;
            my_match=QRegularExpression(globShot->text()).match(fname);
            if (my_match.lastCapturedIndex()==1) {
                matchShot << my_match.captured(1);
            }
        }
        if (matchRef.size()==0 || matchShot.size()==0) {
            statusbar->showMessage("Error: SOP"+ QString::number(matchRef.size()) + " ref and " + QString::number(matchShot.size()) + " shot images", 1000);
        }
        matchRef.intersect(matchShot);
        match+=matchRef;
    }
    qDebug() << comboShot->count();
    comboShot->clear();
    qDebug() << comboShot->count();
    qDebug() << match;

    QStringList my_list;
    for (auto &e : match) {
        my_list << e;
    }
    my_list.sort();
    qDebug() << my_list;
    for (auto &e: my_list) {
        comboShot->addItem(e);
        if (e==oldvalue) {
            comboShot->setCurrentText(oldvalue);
        }
    }
    if(match.size()==0) {
        comboShot->hide();
    } else {
        comboShot->show();
    }
    comboShot->setEnabled(match.size()!=0);
    statusbar->showMessage("Found "+QString::number(my_list.size())+" images", 1000);
    connect(comboShot, SIGNAL(currentTextChanged(QString)), this, SLOT(changeShot(QString)));
}

void Visar::getPhysFromNameSetCombo(QFileInfo finfo, QComboBox* combo) {
    QApplication::processEvents();
    QList <nPhysD *> my_phys=nparent->fileOpen(finfo.absoluteFilePath());
    QApplication::processEvents();
    if (my_phys.size()) {
        for (int i=0; i<combo->count(); i++) {
            nPhysD* phys= combo->itemData(i).value<nPhysD*>();
            if (phys == my_phys[0]) {
                combo->setCurrentIndex(i);
                nparent->showPhys(phys);
            }
        }
    }
    QApplication::processEvents();
}

QFileInfo getFileFromGlob(QFileInfoList list, QString my_glob, QString num) {
    QFileInfo my_file;
    foreach(QFileInfo finfo, list) {
        QString fname=finfo.fileName();
        QRegularExpressionMatch my_match=QRegularExpression(my_glob).match(fname);
        if (my_match.lastCapturedIndex()==1) {
            if (my_match.captured(1) == num) {
                my_file=finfo;
            }
        }
    }
    return my_file;
}


// k<0 : SOP
void Visar::refreshGlob(int k) {
    const QString mymatch("(\\d\\d+)"); // we look for at least 2 digits for the number
    const QString myreplace("(\\d+)"); // but we replace for multiple digits
    std::array<QComboBox*,2> combos={{sopRef,sopShot}};
    std::array<QLineEdit*,2> dirs={{globDirRef,globDirShot}};
    std::array<QLineEdit*,2> globs={{globRef,globShot}};
    if (k>=0) {
        if (k>=(int)visars.size()) return;
        Ui::VisarSettings *ui=visars[k]->settingsUi.get();
        combos={{ui->refImage,ui->shotImage}};
        dirs={{ui->globDirRef,ui->globDirShot}};
        globs={{ui->globRef,ui->globShot}};
    }
    for (unsigned int m=0; m<2; m++) {
        nPhysD *img=getPhysFromCombo(combos[m]);
        if (img) {
            QFileInfo finfo(QString::fromStdString(img->getFromName()));
            dirs[m]->setText(finfo.absoluteDir().path());
            globs[m]->setText(finfo.fileName().replace(QRegularExpression(mymatch),myreplace));
        }
    }
    fillComboShot();
}

void Visar::calculate_etalon() {
    double lam=etalon_lambda->value();
    double e=etalon_thickness->value();
    double n0=etalon_n0->value();
    double dn_dl = etalon_dn_over_dlambda->value();
    double c=_phys_cspeed;

    double delta= -1e-3*lam*n0/(n0*n0-1)*dn_dl;
    double deplace= e*(1-1/n0);
    double tau=2*e/c*(n0-1.0/n0);

    double sens=lam*1e-9/(2*tau*(1+delta)); // km/s

    qDebug() << "delta tau sens deplace"  << delta << tau << sens << deplace;
    visar_sensitivity->setText(QString::number(sens));
    mirror_displacement->setText(QString::number(deplace));

}

VisarChannel::VisarChannel(unsigned int idx, QObject *parent) : QObject(parent),
    k(idx),
    settingsUi(new Ui::VisarSettings()),
    velocityUi(new Ui::VisarVelocity())
{
}

VisarChannel::~VisarChannel() {
    // the Ui structs point into the tab widgets: delete those now (not deleteLater) so that
    // nothing can reach this channel's widgets once the channel is gone
    delete tabSettings;
    delete tabVelocity;
    delete actionRect;
    delete fringeLine;
    delete fringeRect;
    delete maskRegion;
}

void Visar::addVisar() {
    const unsigned int k=visars.size();
    qDebug() << k << tabPhase->count();

    auto ch=std::make_unique<VisarChannel>(k, nullptr);
    Ui::VisarSettings* ui_settings=ch->settingsUi.get();
    Ui::VisarVelocity* ui_velocity=ch->velocityUi.get();

    ch->tabSettings = new QWidget();
    QGridLayout *gridLayout1 = new QGridLayout(ch->tabSettings);
    gridLayout1->setContentsMargins(0, 0, 0, 0);
    QWidget*wVisarSettings = new QWidget(ch->tabSettings);
    wVisarSettings->setObjectName(QStringLiteral("wVisarVelocity"));
    gridLayout1->addWidget(wVisarSettings, 0, 0, 1, 1);

    tabPhase->addTab(ch->tabSettings, "Visar"+QLocale().toString(tabPhase->count()+1));

    ui_settings->setupUi(wVisarSettings);
    ui_settings->doWaveButton->setProperty("needWave",true);

    decorate(wVisarSettings);

    ch->tabVelocity = new QWidget();
    QGridLayout *gridLayout2 = new QGridLayout(ch->tabVelocity);
    gridLayout2->setContentsMargins(0, 0, 0, 0);
    QWidget*wVisarVelocity = new QWidget(ch->tabVelocity);
    wVisarVelocity->setObjectName(QStringLiteral("wvisarVelocity"));
    gridLayout2->addWidget(wVisarVelocity, 0, 0, 1, 1);

    tabVelocity->addTab(ch->tabVelocity, "Visar"+QLocale().toString(tabVelocity->count()+1));

    ui_velocity->setupUi(wVisarVelocity);

    //hack to save diffrent uis!!! (must stay between the two decorate calls: it fixes the saved names)
    foreach (QWidget *obj, wVisarSettings->findChildren<QWidget*>()+wVisarVelocity->findChildren<QWidget*>()) {
        obj->setObjectName(obj->objectName()+"-VISAR"+QLocale().toString(k+1));
    }

    decorate(wVisarVelocity);

    QString name="Visar "+QLocale().toString(k+1);
    for (uint m=0;m<2;m++){
        QString name2=name+" "+QLocale().toString(m);
        ch->phaseUnwrap[m].setName(name2.toUtf8().constData());
        ch->phaseUnwrap[m].setShortName("phase");
        ch->contrast[m].setName(name2.toUtf8().constData());
        ch->contrast[m].setShortName("contrast");
        ch->intensity[m].setName(name2.toUtf8().constData());
        ch->intensity[m].setShortName("intensity");
    }

    ch->actionRect = new QAction(QIcon(":icons/rect.png"), "Region Visar"+QLocale().toString(k+1),this);
    toolBar->insertAction(actionDelVisar,ch->actionRect);

    whichRefl->addItem("Visar"+QLocale().toString(k+1));

    ch->fringeLine=new nLine(this,3);
    ch->fringeLine->changeToolTip("Fringeshift Visar "+QLocale().toString(k+1));

    ch->fringeRect=new nRect(this,1);
    ch->fringeRect->setRect(QRectF(0,0,100,100));
    ch->fringeRect->changeToolTip("Visar region "+QLocale().toString(k+1));

    ch->maskRegion=new nLine(this,1);
    ch->maskRegion->changeToolTip("Ghost Mask Visar"+QLocale().toString(k+1));
    QPolygonF poly;
    poly << QPointF(50,50) << QPointF(50,150) << QPointF(150,150) << QPointF(150,50);
    ch->maskRegion->setPoints(poly);
    ch->maskRegion->toggleClosedLine(true);

    QPen pen;
    QColor my_color;

    ch->graphQuality = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis3);
    ch->graphQuality->setName("Quality Visar "+QLocale().toString(k+1));
    pen.setColor(plotVelocity->yAxis3->labelColor());
    ch->graphQuality->setPen(pen);

    ch->graphReflectivity = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis2);
    ch->graphReflectivity->setName("Reflectivity Visar "+QLocale().toString(k+1));
    pen.setColor(plotVelocity->yAxis2->labelColor());
    ch->graphReflectivity->setPen(pen);

    ch->errReflectivity = new QCPErrorBars(plotVelocity->xAxis, plotVelocity->yAxis2);
    ch->errReflectivity->setName("Reflectivity error Visar "+QLocale().toString(k+1));
    ch->errReflectivity->setDataPlottable(ch->graphReflectivity);
    my_color=plotVelocity->yAxis2->labelColor();
    my_color.setAlpha(property("NeuSave-alphagraph").toInt());
    pen.setColor(my_color);
    ch->errReflectivity->setPen(pen);
    ch->errReflectivity->setWhiskerWidth(0);
    ch->errReflectivity->setSymbolGap(1);

    ch->graphVelocity = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis);
    ch->graphVelocity->setName("Velocity Visar "+QLocale().toString(k+1));
    pen.setColor(plotVelocity->yAxis->labelColor());
    ch->graphVelocity->setPen(pen);

    ch->errVelocity = new QCPErrorBars(plotVelocity->xAxis, plotVelocity->yAxis);
    ch->errVelocity->setName("Velocity error Visar "+QLocale().toString(k+1));
    ch->errVelocity->setDataPlottable(ch->graphVelocity);
    my_color=plotVelocity->yAxis->labelColor();
    my_color.setAlpha(property("NeuSave-alphagraph").toInt());
    pen.setColor(my_color);
    ch->errVelocity->setPen(pen);
    ch->errVelocity->setWhiskerWidth(0);
    ch->errVelocity->setSymbolGap(1);

    // the channel is complete: publish it, then connect (no slot can see a half-built channel)
    VisarChannel *c=ch.get();
    visars.push_back(std::move(ch));
    setProperty("NeuSave-numVisars",(unsigned int)visars.size());

    auto phaseK = [this,c]() { getPhase(c->k); updatePlot(); };
    auto sweepK = [this,c]() { sweepChanged(c->k); };
    auto needWaveK = [this,c]() { needWave(c->k); };

    connect(ui_settings->sensitivity, &QDoubleSpinBox::valueChanged, this, &Visar::updatePlot);

    connect(ui_settings->guess, &QAbstractButton::released, c, [this,c]() { getCarrier(c->k); });
    connect(ui_settings->doWaveButton, &QAbstractButton::released, c, [this,c]() { doWave(c->k); refreshWaveIcon(); });

    connect(ui_settings->multRef, &QDoubleSpinBox::valueChanged, c, phaseK);
    connect(ui_settings->offRef, &QDoubleSpinBox::editingFinished, c, phaseK);
    connect(ui_settings->intensityShift, &QSpinBox::valueChanged, c, phaseK);
    connect(c->fringeRect, &nRect::sceneChanged, c, phaseK);

    connect(ui_settings->physScale, &QLineEdit::editingFinished, c, sweepK);
    connect(ui_settings->physScaleType, &QComboBox::currentIndexChanged, c, sweepK);

    connect(ui_settings->plotPhaseIntensity, SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));

    connect(ui_settings->interfringe, &QDoubleSpinBox::valueChanged, c, needWaveK);
    connect(ui_settings->angle, &QDoubleSpinBox::valueChanged, c, needWaveK);
    connect(ui_settings->resolution, &QDoubleSpinBox::valueChanged, c, needWaveK);
    connect(ui_settings->refImage, &QComboBox::currentIndexChanged, c, needWaveK);
    connect(ui_settings->shotImage, &QComboBox::currentIndexChanged, c, needWaveK);
    // connect(ui_settings->method, &QComboBox::currentIndexChanged, c, needWaveK);
    connect(ui_settings->border, &QSpinBox::valueChanged, c, needWaveK);
    connect(ui_settings->DeghostCheck, &QCheckBox::checkStateChanged, c, needWaveK);
    connect(c->maskRegion, &nLine::sceneChanged, c, needWaveK);

    connect(ui_settings->DeghostCheck, &QCheckBox::toggled, c, [c](bool checked) { c->maskRegion->setVisible(checked); });
    connect(ui_settings->Deghost, SIGNAL(released()), c->maskRegion, SLOT(togglePadella()));
    connect(c->actionRect, SIGNAL(triggered()), c->fringeRect, SLOT(togglePadella()));

    connect(ui_settings->globDirRef, &QLineEdit::textChanged, this, &Visar::fillComboShot);
    connect(ui_settings->globDirShot, &QLineEdit::textChanged, this, &Visar::fillComboShot);
    connect(ui_settings->globRef, &QLineEdit::textChanged, this, &Visar::fillComboShot);
    connect(ui_settings->globShot, &QLineEdit::textChanged, this, &Visar::fillComboShot);
    connect(ui_settings->globRefresh, &QAbstractButton::released, c, [this,c]() { refreshGlob(c->k); });

    connect(ui_velocity->offsetShift, &QDoubleSpinBox::valueChanged, this, &Visar::updatePlot);
    connect(ui_velocity->jumpst, &QLineEdit::editingFinished, this, &Visar::updatePlot);
    connect(ui_velocity->reflRef, &QDoubleSpinBox::valueChanged, this, &Visar::updatePlot);
    connect(ui_velocity->reflOffset, &QDoubleSpinBox::valueChanged, this, &Visar::updatePlot);
    connect(ui_velocity->jump, &QSpinBox::valueChanged, this, &Visar::updatePlot);
    connect(ui_velocity->physOrigin, &QDoubleSpinBox::valueChanged, this, &Visar::updatePlot);
    connect(ui_velocity->offsetTime, &QDoubleSpinBox::valueChanged, this, &Visar::updatePlot);
    connect(ui_velocity->enableVisar, &QAbstractButton::released, this, &Visar::updatePlot);
    connect(ui_velocity->enableVisar, &QAbstractButton::released, this, &Visar::fillComboShot);
    connect(ui_velocity->plotR, &QAbstractButton::released, this, &Visar::updatePlot);
    connect(ui_velocity->plotQ, &QAbstractButton::released, this, &Visar::updatePlot);
}

void Visar::delVisar() {
    if (visars.size()>0) {
        // take it out first: from here on no slot can find it through visars
        std::unique_ptr<VisarChannel> ch=std::move(visars.back());
        visars.pop_back();
        setProperty("NeuSave-numVisars",(unsigned int)visars.size());

        tabPhase->removeTab(ch->k);
        tabVelocity->removeTab(ch->k);
        whichRefl->removeItem(ch->k+2);

        for (QCPAbstractPlottable *plot : {(QCPAbstractPlottable*)ch->errVelocity.data(), (QCPAbstractPlottable*)ch->errReflectivity.data(),
                                           (QCPAbstractPlottable*)ch->graphVelocity.data(), (QCPAbstractPlottable*)ch->graphReflectivity.data(),
                                           (QCPAbstractPlottable*)ch->graphQuality.data()}) {
            if (plot) plotVelocity->removePlottable(plot);
        }

        if (nPhysExists(ch->ghostPhys)) {
            nparent->removePhys(ch->ghostPhys);
        }
        // ch destroyed here: widgets, action, lines and its connections go with it
    } else {
        statusbar->showMessage("Cannot remove last Visar");
    }
    updatePlot();
}

void Visar::loadSettings(QString my_settings) {
    if (my_settings.isEmpty()) {
        QString fname = QFileDialog::getOpenFileName(this, tr("Open INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf);; Any file (*.*)"));
        if (!fname.isNull()) {
            setProperty("NeuSave-fileIni",fname);
            loadSettings(fname);
        }
    } else {
        QSettings settings(my_settings,QSettings::IniFormat);
        settings.beginGroup("Properties");
        int kMax=settings.value("NeuSave-numVisars",2).toUInt();
        unsigned int numVisars_save=visars.size();
        for (unsigned int k=0;k<numVisars_save;k++) {
            delVisar();
        }
        for (int k=0; k<kMax; k++) {
            addVisar();
        }
        int whichReflSaved=settings.value("NeuSave-whichRefl",0).toUInt();
        settings.endGroup();

        nGenericPan::loadSettings(my_settings);

        whichRefl->setCurrentIndex(whichReflSaved);

        QApplication::processEvents();
        sweepChanged();
        doWave();
        calculate_etalon();
        QApplication::processEvents();
        setObjectVisibility (currentBuffer);
        QApplication::processEvents();
    }
}

void Visar::mouseAtPlot(QMouseEvent* e) {
    if (sender()) {
        nCustomPlotMouseX3Y *plot=qobject_cast<nCustomPlotMouseX3Y *>(sender());
        if(plot) {
            QString msg;
            QTextStream(&msg) << plot->xAxis->pixelToCoord(e->pos().x()) << ","
                              << plot->yAxis->pixelToCoord(e->pos().y()) << " R="
                              << plot->yAxis2->pixelToCoord(e->pos().y()) << " Q="
                              << plot->yAxis3->pixelToCoord(e->pos().y());

            statusbar->showMessage(msg);
        } else {
            nSOPPlot *plot=qobject_cast<nSOPPlot *>(sender());
            if(plot) {
                QString msg;
                QTextStream(&msg) << plot->xAxis->pixelToCoord(e->pos().x()) << ","
                                  << plot->yAxis->pixelToCoord(e->pos().y()) << " "
                                  << plot->yAxis2->pixelToCoord(e->pos().y());

                statusbar->showMessage(msg);
            }
        }
    }
}

void Visar::sweepChanged() {
    for (unsigned int k=0; k< visars.size(); k++) {
        sweepChanged(k);
    }
    sweepChangedSOP();
}

void Visar::sweepChanged(unsigned int k) {
    if (k<visars.size()) {
        parseSweep(visars[k]->settingsUi->physScale, visars[k]->settingsUi->physScaleType->currentText() == "Normal", visars[k]->sweepCoeff);
        getPhase(k);
        updatePlot();
    }
}

void Visar::sweepChangedSOP() {
    parseSweep(sopScale, sopScaleType->currentText() == "Normal", sweepCoeffSOP);
    updatePlotSOP();
}

bool Visar::parseSweep(QLineEdit *line, bool normalPoly, std::vector<double> &vecsweep) {
    vecsweep.clear();
    line->setPalette(QApplication::palette());
    QRegularExpression separator("[(;| |\t)]");
    QStringList strSweep=line->text().split(separator, Qt::SkipEmptyParts);
    int index=0;
    foreach(QString str, strSweep) {
        bool ok;
        double coeff=locale().toDouble(str,&ok);
        if(ok) {
            index++;
            if (normalPoly) {
                vecsweep.push_back(coeff);
            } else {
                vecsweep.push_back(coeff/index);
            }
        } else {
            QPalette my_palette=line->palette();
            my_palette.setColor(QPalette::Base,Qt::red);
            line->setPalette(my_palette);
            statusbar->showMessage("Cannot understand sweep coefficient "+str);
            return false;
        }
    }
    return true;
}

double Visar::getTime(std::vector<double> &vecsweep, double p) {
    double time=0;
    for (unsigned int i=0;i<vecsweep.size();i++) {
        time+=vecsweep.at(i)*pow(p,i+1);
    }
    return time;
}

QPointF Visar::getTimeSpaceFromPixel(QPointF p) {
    int k=0;
    double posTime=0.0;
    double posSpace=0.0;
    if (tabs->currentIndex()==0) {
        k=tabPhase->currentIndex();
        if (k >= 0 && k<(int)visars.size()) {
            posTime=(direction(k)==0 ? p.y() : p.x());
            posSpace=(direction(k)==0 ? p.x() : p.y())-visars[k]->settingsUi->physOriginSpace->value();
            visars[k]->settingsUi->plotPhaseIntensity->setMousePosition(posTime);
        }
    } else if (tabs->currentIndex()==1) {
        k=tabVelocity->currentIndex();
        if (k >= 0 && k<(int)visars.size()) {
            double pos=direction(k)==0 ? p.y() : p.x();
            posTime=getTime(visars[k]->sweepCoeff,pos) - getTime(visars[k]->sweepCoeff,visars[k]->velocityUi->physOrigin->value()) + visars[k]->velocityUi->offsetTime->value();
            posSpace=((direction(k)==0 ? p.x() : p.y())-visars[k]->settingsUi->physOriginSpace->value())*visars[k]->settingsUi->magnification->value();
            plotVelocity->setMousePosition(posTime);
        }
    } else {
        double pos=sopDirection->currentIndex()==0 ? p.y() : p.x();
        posTime=getTime(sweepCoeffSOP,pos) - getTime(sweepCoeffSOP,sopOrigin->value()) + sopTimeOffset->value();
        posSpace=((direction(k)==0 ? p.x() : p.y())-physOriginSpace->value())*magnification->value();
        sopPlot->setMousePosition(posTime);
    }
    return QPointF(posTime,posSpace);
}

void Visar::imageMousePress(QPointF p) {
    setProperty("timeClick",getTimeSpaceFromPixel(p));
}

void Visar::imageMouseRelease(QPointF p) {
    QPointF timeSpace=getTimeSpaceFromPixel(p);
    QPointF refTimeSpace=property("timeClick").toPointF();
    double deltatime=timeSpace.x()-refTimeSpace.x();
    double deltaspace=timeSpace.y()-refTimeSpace.y();
    statusbar->showMessage("Delta : "+QString::number(deltatime)+" "+QString::number(deltaspace));
    setProperty("timeClick",QVariant());
}

void Visar::mouseAtMatrix(QPointF p) {
    QPointF timeSpace=getTimeSpaceFromPixel(p);
    QPointF refTimeSpace=property("timeClick").toPointF();

    double deltatime=timeSpace.x();
    double deltaspace=timeSpace.y();
    QString prefix=tr("Time: ")+QString::number(deltatime);
    if (property("timeClick").isValid()) {
        prefix += tr(" Delay: ")+QString::number(deltatime-refTimeSpace.x());
    }
    prefix += " Position: "+QString::number(deltaspace);
    if (property("timeClick").isValid()) {
        prefix += tr(" Distance: ")+QString::number(deltaspace-refTimeSpace.y());
    }
    statusbar->showMessage(prefix);
}

int Visar::direction(int k) {
    int dir=((int) ((visars[k]->settingsUi->angle->value()+360+45)/90.0) )%2;
    return dir;
}

void Visar::setObjectVisibility(nPhysD*phys) {
    if (nPhysExists(phys)) {
        for (unsigned int k=0;k<visars.size();k++){
            bool ismyimg = (phys == getPhysFromCombo(visars[k]->settingsUi->shotImage) || phys == getPhysFromCombo(visars[k]->settingsUi->refImage));
            visars[k]->fringeLine->setVisible(ismyimg);
            visars[k]->fringeRect->setVisible(ismyimg);
            visars[k]->maskRegion->setVisible(visars[k]->settingsUi->DeghostCheck->isChecked() && (ismyimg || phys == visars[k]->ghostPhys));
        }
        sopRect->setVisible(enableSOP->isChecked() && (phys == getPhysFromCombo(sopRef) || phys == getPhysFromCombo(sopShot)));
    }
}

void Visar::tabChanged(int k) {
    QTabWidget *tabWidget=nullptr;

    if (sender()) tabWidget=qobject_cast<QTabWidget *>(sender());
    if (!tabWidget) {
        if (tabs->currentIndex()==0) {
            tabWidget=tabPhase;
            k=tabPhase->currentIndex();
        } else if (tabs->currentIndex()==1) {
            tabWidget=tabVelocity;
            k=tabVelocity->currentIndex();
        } else {
            tabWidget=tabs;
        }
    }

    if (tabWidget) {

        if (tabWidget==tabs) {
            if (k==0) {
                tabWidget=tabPhase;
            } else if (k==1) {
                tabWidget=tabVelocity;
            }
        } else if (tabWidget == tabVelocity) {
            updatePlot();
        } else if (tabWidget == tabPhase) {
            getPhase(k);
        }
        if (tabWidget==tabs && tabs->currentIndex()==2 && enableSOP->isChecked()) {
            nparent->showPhys(getPhysFromCombo(sopShot));
        } else {
            k=tabWidget->currentIndex();
            if (k>=0 && k<(int)visars.size()) {
                if (visars[k]->velocityUi->enableVisar->isChecked()) {
                    nparent->showPhys(getPhysFromCombo(visars[k]->settingsUi->shotImage));
                }
            }
        }
    }
}

void Visar::updatePlotSOP() {
    if (!enableSOP->isChecked()) return;
    nPhysD *shot=getPhysFromCombo(sopShot);
    nPhysD *ref=getPhysFromCombo(sopRef);


    int dir=sopDirection->currentIndex();
    if (shot) {
        QRect geom2=sopRect->getRect(shot);
        if (ref) {
            geom2=geom2.intersected(QRect(0,0,ref->getW(),ref->getH()));
        }
        if (geom2.isEmpty()) {
            statusbar->showMessage(tr("Attention: the region is outside the image!"),2000);
            return;
        }
        int dx=geom2.width();
        int dy=geom2.height();
        QVector<double> sopData(dir==0?dy:dx);
        for (int j=0;j<dy;j++){
            for (int i=0;i<dx; i++) {
                double val=shot->point(i+geom2.x(),j+geom2.y(),0.0);
                if (ref && ref!=shot) val-=ref->point(i+geom2.x(),j+geom2.y(),0.0);
                sopData[dir==0?j:i]+=val;
            }
        }
        time_sop.resize(sopData.size());
        sopCurve[0].resize(sopData.size());

        switch (dir) {
            case 0:
                for (int j=0;j<dy;j++) {
                    time_sop[j]=getTime(sweepCoeffSOP,geom2.y()+j)-getTime(sweepCoeffSOP,sopOrigin->value()) + sopTimeOffset->value();
                    sopCurve[0][j]=sopData[j]/dx-sopOffset->value();
                }
                break;
            case 1:
                for (int i=0;i<dx;i++) {
                    time_sop[i]=getTime(sweepCoeffSOP,geom2.x()+i)-getTime(sweepCoeffSOP,sopOrigin->value()) + sopTimeOffset->value();
                    sopCurve[0][i]=sopData[i]/dy-sopOffset->value();
                }
                break;
            default:
                break;
        }
        sopPlot->graph(0)->setData(time_sop,sopCurve[0]);

        // TEMPERATURE FROM REFLECTIVITY
        QVector<int> reflList;
        setProperty("NeuSave-whichRefl",whichRefl->currentIndex());
        switch (whichRefl->currentIndex()) {
            case 0: // Zero
                break;
            case 1: // mean
                for (unsigned int i=0; i<visars.size(); i++) {
                    reflList << i;
                }
                break;
            default: // single refl
                reflList << whichRefl->currentIndex()-2;
                break;
        }

        sopCurve[1].resize(time_sop.size());
        sopCurve[2].resize(time_sop.size());

        double my_T0=sopCalibT0->value();
        double my_A=sopCalibA->value();

        for (int i=0; i<time_sop.size(); i++) {
            double my_reflectivity=0;

            int numrefl=0;
            for (int numk=0;numk<reflList.size();numk++) {
                int k=reflList[numk];

                if ( k>=0 && k<(int)visars.size() ) {
                    for (int j=1;j<visars[k]->time_vel.size();j++) {
                        double t_j1=visars[k]->time_vel[j-1];
                        double t_j=visars[k]->time_vel[j];

                        if (time_sop[i]>t_j1 && time_sop[i]<=t_j ) {
                            double valRj_1=visars[k]->reflectivity[j-1];
                            double valRj=visars[k]->reflectivity[j];

                            my_reflectivity+=valRj_1+(time_sop[i]-t_j1)*(valRj-valRj_1)/(t_j-t_j1);

                            numrefl++;
                        }
                    }
                }
            }
            if (numrefl) {
                my_reflectivity/=numrefl;
            }

            my_reflectivity=std::min(std::max(my_reflectivity,0.0),1.0);
            double my_temp=my_T0/log(1.0+(1.0-my_reflectivity)*my_A/sopCurve[0][i]);

            if (numrefl) {
                sopCurve[1][i]=my_temp;
                sopCurve[2][i]=0.0;
            } else {
                sopCurve[1][i]=0.0;
                sopCurve[2][i]=my_temp;
            }
        }

        sopPlot->graph(1)->setData(time_sop,sopCurve[1]);

        sopPlot->graph(2)->setData(time_sop,sopCurve[2]);

        sopPlot->rescaleAxes();
        sopPlot->replot();
    }
}

void Visar::updatePlot() {
    bool needReflectivityAxe=false;
    bool needQualityAxe=false;
    for (unsigned int k=0;k<visars.size();k++){
        if (visars[k]->velocityUi->enableVisar->isChecked()) {
            if (visars[k]->velocityUi->plotR->isChecked()) needReflectivityAxe=true;
            if (visars[k]->velocityUi->plotQ->isChecked()) needQualityAxe=true;
        }
    }
    plotVelocity->yAxis2->setVisible(needReflectivityAxe);
    plotVelocity->yAxis3->setVisible(needQualityAxe);

    plotVelocity->clearItems();

    for (int g=plotVelocity->plottableCount()-1; g>=0; g--) {
        if (plotVelocity->plottable(g)->property("JumpGraph").isValid()) {
            plotVelocity->removePlottable(plotVelocity->plottable(g));
        }
    }

    std::vector<double> minmax;
    for (unsigned int k=0;k<visars.size();k++){
        if (visars[k]->cPhase[1].size()==visars[k]->time_phase.size()){

            Qt::PenStyle pstyle=((int)k==tabVelocity->currentIndex()?Qt::SolidLine : Qt::DashLine);

            double sensitivity=visars[k]->settingsUi->sensitivity->value();
            double deltat=visars[k]->velocityUi->offsetTime->value()-getTime(visars[k]->sweepCoeff,visars[k]->velocityUi->physOrigin->value());

            QVector<double> tjump,njump,rjump;
            QStringList jumpt=visars[k]->velocityUi->jumpst->text().split(";", Qt::SkipEmptyParts);
            visars[k]->velocityUi->jumpst->setPalette(QApplication::palette());
            QPalette my_palette=visars[k]->velocityUi->jumpst->palette();
            my_palette.setColor(QPalette::Base,Qt::red);

            foreach (QString piece, jumpt) {
                QString err_msg=" "+piece+QString("' VISAR ")+QLocale().toString(k+1)+tr(" Decimal separator is: ")+locale().decimalPoint();
                QStringList my_jumps=piece.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
                if (my_jumps.size()>1 && my_jumps.size()<=3) {
                    if (my_jumps.size()>1 && my_jumps.size()<=3) {
                        bool ok1, ok2, ok3=true;
                        double valdt=locale().toDouble(my_jumps.at(0),&ok1);
                        double valdn=locale().toDouble(my_jumps.at(1),&ok2);
                        double valdrefr_index=1.0;
                        if (my_jumps.size()==3) {
                            valdrefr_index=locale().toDouble(my_jumps.at(2),&ok3);
                            if (!ok3) {
                                visars[k]->velocityUi->jumpst->setPalette(my_palette);
                                statusbar->showMessage(tr("Skipped unreadable refraction index '")+err_msg,5000);
                            }
                        }
                        if (sensitivity<0) valdn*=-1.0;
                        if (ok1 && ok2) {
                            tjump << valdt;
                            njump << valdn;
                        } else {
                            visars[k]->velocityUi->jumpst->setPalette(my_palette);
                            statusbar->showMessage(tr("Skipped unreadable jump '")+err_msg,5000);
                        }
                        if (ok3) {
                            rjump << valdrefr_index;
                        } else {
                            rjump << 1.0;
                        }
                    }
                } else {
                    visars[k]->velocityUi->jumpst->setPalette(my_palette);
                    statusbar->showMessage(tr("Skipped unreadable jump '")+err_msg,5000);
                }
            }
            foreach (double a, tjump) {
                QCPItemStraightLine* my_jumpLine=new QCPItemStraightLine(plotVelocity);
                QPen pen(Qt::gray);
                pen.setStyle(pstyle);
                my_jumpLine->setPen(pen);
                my_jumpLine->point1->setTypeY(QCPItemPosition::ptAbsolute);
                my_jumpLine->point2->setTypeY(QCPItemPosition::ptAbsolute);
                my_jumpLine->point1->setCoords(a,0);
                my_jumpLine->point2->setCoords(a,1);
            }

            double offset=visars[k]->velocityUi->offsetShift->value();

            QVector<QVector< double > > velJump_array(abs(visars[k]->velocityUi->jump->value()));

            for (int i=0;i<abs(visars[k]->velocityUi->jump->value());i++) {
                velJump_array[i].resize(visars[k]->time_phase.size());
            }

            visars[k]->time_vel.resize(visars[k]->time_phase.size());
            visars[k]->velocity.resize(visars[k]->time_phase.size());
            visars[k]->velError.resize(visars[k]->time_phase.size());
            visars[k]->reflectivity.resize(visars[k]->time_phase.size());
            visars[k]->reflError.resize(visars[k]->time_phase.size());
            visars[k]->quality.resize(visars[k]->time_phase.size());

            for (int j=0;j<visars[k]->time_phase.size();j++) {
                visars[k]->time_vel[j] = getTime(visars[k]->sweepCoeff,visars[k]->time_phase[j])+deltat;

                double fShot=visars[k]->cPhase[1][j];
                double iRef=visars[k]->cIntensity[0][j];
                double iShot=visars[k]->cIntensity[1][j];
                if (getPhysFromCombo(visars[k]->settingsUi->shotImage)==getPhysFromCombo(visars[k]->settingsUi->refImage)) {
                    iRef=1.0;
                }

                double njumps=0;
                double refr_index=1.0;
                for (int i=0;i<tjump.size();i++) {
                    if (visars[k]->time_vel[j]>tjump.at(i)) {
                        njumps+=njump.at(i);
                        refr_index=rjump.at(i);
                    }
                }

                double speed=(offset+fShot+njumps)*sensitivity/refr_index;
                double Rg=visars[k]->velocityUi->reflOffset->value();
                double Rmat=visars[k]->velocityUi->reflRef->value();
                double beta=-Rg/pow(1.0-Rg,2);
                double refle=iShot/iRef * (Rmat-beta) + beta;

                visars[k]->velocity[j] = speed;
                visars[k]->reflectivity[j] = refle;
                visars[k]->quality[j] = visars[k]->cContrast[1][j]/visars[k]->cContrast[0][j];
                visars[k]->velError[j] = abs(visars[k]->cPhaseErr[j]*sensitivity/refr_index);
                visars[k]->reflError[j] = visars[k]->cReflErr[j]* (Rmat-beta) + beta;

                for (int i=0;i<abs(visars[k]->velocityUi->jump->value());i++) {
                    int jloc=i+1;
                    if (sensitivity<0) jloc*=-1;
                    if (visars[k]->velocityUi->jump->value()<0) jloc*=-1;
                    velJump_array[i][j] = (offset+fShot+jloc)*sensitivity/refr_index;
                }
            }

            QPen pen;
            pen.setStyle(pstyle);

            VisarChannel &ch=*visars[k];
            const bool enabled=ch.velocityUi->enableVisar->isChecked();
            const bool hasFringes=ch.settingsUi->interfringe->value() != 0.0;
            auto setGraph=[&](QCPGraph *my_graph, const QVector<double> &values, bool show) {
                if (!my_graph) return;
                pen=my_graph->pen();
                pen.setStyle(pstyle);
                my_graph->setPen(pen);
                if (show) {
                    my_graph->setData(ch.time_vel,values);
                } else {
                    my_graph->data()->clear();
                }
                my_graph->setVisible(enabled);
            };
            setGraph(ch.graphVelocity, ch.velocity, hasFringes);
            setGraph(ch.graphReflectivity, ch.reflectivity, ch.velocityUi->plotR->isChecked());
            setGraph(ch.graphQuality, ch.quality, hasFringes && ch.velocityUi->plotQ->isChecked());

            auto setErr=[&](QCPErrorBars *my_err, const QVector<double> &values) {
                if (!my_err) return;
                pen=my_err->pen();
                pen.setStyle(pstyle);
                my_err->setPen(pen);
                my_err->setData(values);
                my_err->setVisible(enabled);
            };
            setErr(ch.errVelocity, ch.velError);
            setErr(ch.errReflectivity, ch.reflError);

            if (visars[k]->velocityUi->jump->value()!=0) {
                for (int i=0;i<abs(visars[k]->velocityUi->jump->value());i++) {
                    QCPGraph* graph = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis);
                    graph->setProperty("JumpGraph",k);
                    graph->setName("VelJump Visar"+QLocale().toString(k+1) + " #" +QLocale().toString(i));
                    QColor color(plotVelocity->yAxis->labelColor());
                    color.setAlpha(property("NeuSave-alphagraph").toInt()*3);
                    pen.setColor(color);
                    graph->setPen(pen);
                    graph->setData(visars[k]->time_vel,velJump_array[i]);
                    graph->setVisible(visars[k]->velocityUi->enableVisar->isChecked());
                }

            }
            if (visars[k]->velocityUi->enableVisar->isChecked() && visars[k]->time_vel.size()) {
                const auto [mmin, mmax] = std::minmax_element(std::begin(visars[k]->time_vel), std::end(visars[k]->time_vel));
                minmax.push_back(*mmin);
                minmax.push_back(*mmax);
                qDebug() << k << *mmin << *mmax;
            }
        }
    }

    plotVelocity->rescaleAxes(true);
    if (minmax.size()) {
        const auto [mmin, mmax] = std::minmax_element(std::begin(minmax), std::end(minmax));
        qDebug() << *mmin << *mmax;
        plotVelocity->xAxis->setRange(*mmin,*mmax);
    }

    plotVelocity->replot();
    updatePlotSOP();
}

void Visar::getCarrier(unsigned int k) {
    if (k>=visars.size()) return;
    QComboBox *combo=visars[k]->settingsUi->refImage;

    nPhysD *phys=getPhysFromCombo(combo);
    if (phys && visars[k]->fringeRect) {
        QRect geom2=visars[k]->fringeRect->getRect(phys);
        nPhysD datamatrix = phys->sub(geom2);

        std::vector<vec2f> vecCarr=physWave::phys_guess_carrier(datamatrix, visars[k]->settingsUi->guessWeight->value());

        if (vecCarr.size()==0) {
            statusbar->showMessage("ERROR: Problem finding the carrier try to change the weight", 5000);
        } else {
            visars[k]->settingsUi->interfringe->setValue(vecCarr[0].first());
            visars[k]->settingsUi->angle->setValue(vecCarr[0].second());
            std::stringstream ss;
            for (auto &v: vecCarr) {
                ss << v << " ";
            }
            qInfo() << QString::fromStdString(ss.str());
            if (tabPhase->currentIndex() == static_cast<int>(k)) {
                statusbar->showMessage(tr("Carrier :")+QLocale().toString(vecCarr[0].first())+tr("px, ")+QLocale().toString(vecCarr[0].second())+tr("deg"));
            }
        }
    }
}

void Visar::doWave() {
    for (unsigned int k=0;k<visars.size();k++){
        if (visars[k]->settingsUi->doWaveButton->property("needWave").toBool()) {
            doWave(k);
        }
    }
    refreshWaveIcon();
}

void Visar::refreshWaveIcon() {
    for (auto &ch : visars) {
        if (ch->settingsUi->doWaveButton->property("needWave").toBool()) {
            return;
        }
    }
    actionDoWavelets->setIcon(QIcon(":icons/refresh.png"));
}

void Visar::needWave(unsigned int k) {
    if (k< visars.size()) {
        QIcon my_icon=QApplication::style()->standardIcon(QStyle::SP_MessageBoxWarning);
        tabVelocity->setTabIcon(k,my_icon);
        tabPhase->setTabIcon(k,my_icon);
        visars[k]->settingsUi->doWaveButton->setIcon(my_icon);
        visars[k]->settingsUi->doWaveButton->setProperty("needWave",true);
        actionDoWavelets->setIcon(my_icon);
    }
}

void Visar::doWave(unsigned int k) {
    if (k< visars.size()) {

        std::array<nPhysD*,2> imgs={{getPhysFromCombo(visars[k]->settingsUi->refImage),getPhysFromCombo(visars[k]->settingsUi->shotImage)}};

        if (imgs[0] && imgs[1]  && imgs[0]->getSize() == imgs[1]->getSize()) {

            QProgressDialog progress("Filter visar "+QLocale().toString(k+1), "Cancel", 0, 23, this);
            progress.setCancelButton(nullptr);
            progress.setWindowModality(Qt::WindowModal);
            progress.setValue(0);
            progress.show();
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            size_t dx=imgs[1]->getW();
            size_t dy=imgs[1]->getH();

            std::vector<int> xx(dx), yy(dy);

            for (size_t i=0;i<dx;i++)
                xx[i]=(i+(dx+1)/2)%dx-(dx+1)/2; // swap and center
            for (size_t i=0;i<dy;i++)
                yy[i]=(i+(dy+1)/2)%dy-(dy+1)/2;

            double cr = cos((visars[k]->settingsUi->angle->value()) * _phys_deg);
            double sr = sin((visars[k]->settingsUi->angle->value()) * _phys_deg);

            nPhysD* physDeghost=nullptr;

            if (visars[k]->settingsUi->DeghostCheck->checkState()>0) {

                physC imageFFT = imgs[1]->ft2(PHYS_FORWARD);

                progress.setValue(progress.value()+1);
                // double lambda=sqrt(pow(cr*dx,2)+pow(sr*dy,2))/(M_PI*visars[k]->settingsUi->interfringe->value());
                double thick_norm= M_PI* visars[k]->settingsUi->resolution->value()/sqrt(pow(sr*dx,2)+pow(cr*dy,2));
                double lambda_norm=M_PI*visars[k]->settingsUi->interfringe->value()/sqrt(pow(cr*dx,2)+pow(sr*dy,2));

                for (size_t x=0;x<dx;x++) {
                    for (size_t y=0;y<dy;y++) {
                        double xr = xx[x]*cr - yy[y]*sr;
                        double yr = xx[x]*sr + yy[y]*cr;
                        // double e_tot = 1.0-exp(-pow(yr/M_PI,2))/(1.0+exp(lambda-std::abs(xr)));
                        double e_tot = 1.0-exp(-pow(yr/thick_norm,2))*exp(-pow(std::abs(xr)*lambda_norm-M_PI, 2));
                        imageFFT.set(x,y,imageFFT.point(x,y) * e_tot);
                    }
                }

                progress.setValue(progress.value()+1);

                imageFFT = imageFFT.ft2(PHYS_BACKWARD);

                progress.setValue(progress.value()+1);

                physDeghost=new nPhysD(*imgs[1],"deghost VISAR "+std::to_string(tabPhase->currentIndex()+1));

                QRect geom=visars[k]->maskRegion->path().boundingRect().toRect();

                QPolygonF regionPoly=visars[k]->maskRegion->poly(1);
                regionPoly=regionPoly.translated(imgs[1]->get_origin().x(),imgs[1]->get_origin().y());
                std::vector<vec2f> vecPoints(regionPoly.size());
                for(int k=0;k<regionPoly.size();k++) {
                    vecPoints[k]=vec2f(regionPoly[k].x(),regionPoly[k].y());
                }

                for(int i=geom.left();i<geom.right(); i++) {
                    for(int j=geom.top();j<geom.bottom(); j++) {
                        vec2f pp(i,j);
                        if (point_inside_poly(pp,vecPoints)==true) {
                            physDeghost->set(i,j, imageFFT.point(i,j).mod()/(dx*dy));
//doesntwork                            physDeghost->set(i,j, imageFFT.point(i,j).real()/(dx*dy));
                        }
                    }
                }
                progress.setValue(progress.value()+1);
                physDeghost->TscanBrightness();
                imgs[1]=physDeghost;
            }

            parseSweep(visars[k]->settingsUi->physScale, visars[k]->settingsUi->physScaleType->currentText() == "Normal", visars[k]->sweepCoeff);

            std::array<physC,2> physfft={{imgs[0]->ft2(PHYS_FORWARD),imgs[1]->ft2(PHYS_FORWARD)}};
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            std::array<physC,2> zz_morlet;
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            std::array<physD,2> phase({{nPhysD(),nPhysD()}});

            for (int m=0;m<2;m++) {
                phase[m].resize(dx, dy);
                visars[k]->contrast[m].resize(dx, dy);
                visars[k]->intensity[m]= imgs[m]->copy();
                progress.setValue(progress.value()+1);
                qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
                physMath::phys_fast_gaussian_blur(visars[k]->intensity[m], visars[k]->settingsUi->resolution->value());
                progress.setValue(progress.value()+1);
                qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
                zz_morlet[m].resize(dx,dy);
                progress.setValue(progress.value()+1);
                qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            }

            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            double thick_norm=visars[k]->settingsUi->resolution->value()*M_PI/sqrt(pow(sr*dx,2)+pow(cr*dy,2));
            double lambda_norm=visars[k]->settingsUi->interfringe->value()/sqrt(pow(cr*dx,2)+pow(sr*dy,2));
            for (unsigned int m=0;m<2;m++) {
#pragma omp parallel for collapse(2)
                for (size_t x=0;x<(size_t)dx;x++) {
                    for (size_t y=0;y<(size_t)dy;y++) {
                        double xr = xx[x]*cr - yy[y]*sr; //rotate
                        double yr = xx[x]*sr + yy[y]*cr;

                        double e_x = -pow(M_PI*(xr*lambda_norm-1.0), 2);
                        double e_y = -pow(yr*thick_norm, 2);

                        double gauss = exp(e_x)*exp(e_y);

                        zz_morlet[m].Timg_matrix[y][x]=physfft[m].Timg_matrix[y][x]*gauss;
                    }
                }
                progress.setValue(progress.value()+1);
            }

            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            for (unsigned int m=0;m<2;m++) {
                physfft[m] = zz_morlet[m].ft2(PHYS_BACKWARD);
                progress.setValue(progress.value()+1);
            }
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            for (unsigned int m=0;m<2;m++) {
                #pragma omp parallel for
                for (size_t kk=0; kk<(size_t)(dx*dy); kk++) {
                    phase[m].Timg_buffer[kk] = -physfft[m].Timg_buffer[kk].arg()/(2*M_PI);
                    visars[k]->contrast[m].Timg_buffer[kk] = 2.0*physfft[m].Timg_buffer[kk].mod()/(dx*dy);
                    visars[k]->intensity[m].Timg_buffer[kk] -= visars[k]->contrast[m].point(kk)*cos(2*M_PI*phase[m].point(kk));
                }
                // removing left and right border
                int border=visars[k]->settingsUi->border->value();
                if ( border > 0) {
                    for (size_t y=0;y<(size_t)dy;y++) {
                        int b=0;
                        for (int x=0;x<border;x++) {
                            b+=visars[k]->intensity[m].point(x,y);
                            b+=visars[k]->intensity[m].point(dx-x,y);
                        }
                        b=b/(2*border);
                        for (size_t x=0;x<(size_t)dx;x++) {
                            visars[k]->intensity[m].set(x,y,visars[k]->intensity[m].point(x,y)-b);
                        }
                    }
                }
            }

            if (direction(k)!=0) {
                for (unsigned int m=0;m<2;m++) {
                    physMath::phys_transpose(phase[m]);
                    physMath::phys_transpose(visars[k]->contrast[m]);
                    physMath::phys_transpose(visars[k]->intensity[m]);
                }
            }
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
    //unwrap

            physD diff = phase[1]-phase[0];
            physD qual = visars[k]->contrast[1]*visars[k]->contrast[0];

            // QString methodName=visars[k]->settingsUi->method->currentText();
            physWave::unwrap_strategy strategy=physWave::QUALITY;
            // if (methodName=="H+V") {
            //     strategy=physWave::SIMPLE_HV;
            // } else if (methodName=="V+H") {
            //     strategy=physWave::SIMPLE_VH;
            // } else if (methodName=="Miguel") {
            //     strategy=physWave::MIGUEL;
            // } else if (methodName=="Mig+Qual") {
            //     strategy=physWave::MIGUEL_QUALITY;
            // } else if (methodName=="Quality") {
            //     strategy=physWave::QUALITY;
            // }

            physWave::phys_phase_unwrap(phase[0], qual, strategy, visars[k]->phaseUnwrap[0]);
            physWave::phys_phase_unwrap(diff,     qual, strategy, visars[k]->phaseUnwrap[1]);

            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            getPhase(k);

            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            updatePlot();
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            visars[k]->settingsUi->doWaveButton->setIcon(QIcon(":icons/refresh.png"));
            visars[k]->settingsUi->doWaveButton->setProperty("needWave",false);

            tabVelocity->setTabIcon(k,QIcon());
            tabPhase->setTabIcon(k,QIcon());



            if (physDeghost){
                qDebug() << visars[k]->settingsUi->DeghostCheck->checkState();
                qDebug() << physDeghost->getSize().x() << physDeghost->getSize().y();
                if (visars[k]->settingsUi->DeghostCheck->checkState()==1) {
                    visars[k]->ghostPhys=nparent->replacePhys(physDeghost,visars[k]->ghostPhys,false);
                } else if (visars[k]->settingsUi->DeghostCheck->checkState()==2) {
                    qDebug() << nPhysExists(physDeghost) << visars[k]->ghostPhys;
                    delete physDeghost;
                    if (nPhysExists(visars[k]->ghostPhys)) {
                        nparent->removePhys(visars[k]->ghostPhys);
                    }
                    visars[k]->ghostPhys=nullptr;
                }
            }

        } else {
            if (imgs[0] && imgs[1]) {
                DEBUG(imgs[0]->getH() << "," << imgs[0]->getW() << " " << imgs[1]->getH() << "," << imgs[1]->getW());
            } else {
                DEBUG("BIG ERROR! BIG ERROR! BIG ERROR! BIG ERROR! BIG ERROR! BIG ERROR! BIG ERROR! BIG ERROR! ");
            }
            statusBar()->showMessage("Size mismatch",5000);
        }
    }
}

void Visar::getPhase() {
    for (unsigned int k=0;k<visars.size();k++){
        getPhase(k);
    }
    updatePlot();
}

void Visar::getPhase(unsigned int k) {
    if (k < visars.size() && visars[k]->fringeRect && visars[k]->fringeLine) {
        visars[k]->settingsUi->plotPhaseIntensity->clearGraphs();
        std::array<nPhysD*,2> imgs={{getPhysFromCombo(visars[k]->settingsUi->refImage),getPhysFromCombo(visars[k]->settingsUi->shotImage)}};

        if (imgs[0] && imgs[1] && imgs[0]->getSize() == imgs[1]->getSize()) {


            QRect geom2=visars[k]->fringeRect->getRect(imgs[0]);
            if (direction(k)!=0) {
                geom2=QRect(QPoint(geom2.top(),geom2.left()),QSize(geom2.height(),geom2.width()));
            }
            for (unsigned int m=0;m<2;m++) {
                visars[k]->cPhase[m].clear();
                visars[k]->cIntensity[m].clear();
                visars[k]->cContrast[m].clear();
            }
            visars[k]->time_phase.clear();
            visars[k]->cPhaseErr.clear();
            visars[k]->cReflErr.clear();

            int refIntShift= visars[k]->settingsUi->intensityShift->value();

            for (int j=geom2.top(); j<geom2.bottom(); j++) {
                visars[k]->time_phase  << j;
                visars[k]->cPhase[0]  << visars[k]->phaseUnwrap[0].point(geom2.center().x(),j,0)-visars[k]->phaseUnwrap[0].point(geom2.center().x(),geom2.top(),0);
                visars[k]->cPhase[1]  << visars[k]->phaseUnwrap[1].point(geom2.center().x(),j,0);

                double meanIntRef=0.0;
                double meanIntShot=0.0;
                double contrastTmpRef=0.0;
                double contrastTmpShot=0.0;
                double meanPhaseTmp=0.0;
                double meanPhaseTmpSqr=0.0;
                double meanRefle=0.0;
                double meanRefleSqr=0.0;
                for (int i=geom2.left(); i<geom2.right();i++) {
                    double intRef=(visars[k]->intensity[0].point(i,j-refIntShift,0)-visars[k]->settingsUi->offRef->value())*visars[k]->settingsUi->multRef->value();
                    double intShot=visars[k]->intensity[1].point(i,j,0)-visars[k]->settingsUi->offRef->value();

                    meanIntRef+=intRef;
                    meanIntShot+=intShot;

                    meanRefle+= intShot/intRef;
                    meanRefleSqr+= pow(intShot/intRef,2);

                    contrastTmpRef+=visars[k]->contrast[0].point(i,j-refIntShift,0);
                    contrastTmpShot+=visars[k]->contrast[1].point(i,j,0);
                    meanPhaseTmp += visars[k]->phaseUnwrap[1].point(i,j);
                    meanPhaseTmpSqr += pow(visars[k]->phaseUnwrap[1].point(i,j),2);
                }


                meanIntRef/=geom2.width();
                meanIntShot/=geom2.width();
                meanRefle/=geom2.width();
                meanRefleSqr/=geom2.width();

                contrastTmpRef/=geom2.width();
                contrastTmpShot/=geom2.width();
                meanPhaseTmp /= geom2.width();
                meanPhaseTmpSqr /= geom2.width();

                visars[k]->cIntensity[0] << meanIntRef;
                visars[k]->cIntensity[1] << meanIntShot;
                visars[k]->cContrast[0]  << contrastTmpRef*visars[k]->settingsUi->multRef->value();
                visars[k]->cContrast[1]  << contrastTmpShot;
                visars[k]->cPhaseErr << sqrt(meanPhaseTmpSqr -pow(meanPhaseTmp,2));
                visars[k]->cReflErr << sqrt(meanRefleSqr -pow(meanRefle,2));


            }

            double buffer,bufferold,dummy=0.0;
            double offsetShift=0;
            if(visars[k]->cPhase[1].size()) {
                if (visars[k]->sweepCoeff.size() && visars[k]->sweepCoeff.front()>0) {
                    bufferold=visars[k]->cPhase[1].first();
                    for (int j=1;j<visars[k]->cPhase[1].size();j++){
                        buffer=visars[k]->cPhase[1][j];
                        if (fabs(buffer-bufferold)>0.5) dummy+=SIGN(bufferold-buffer);
                        bufferold=buffer;
                        visars[k]->cPhase[1][j]+=dummy;
                    }
                    offsetShift=visars[k]->cPhase[1].first();
                } else {
                    bufferold=visars[k]->cPhase[1].last();
                    for (int j=visars[k]->cPhase[1].size()-2;j>=0;j--){
                        buffer=visars[k]->cPhase[1][j];
                        if (fabs(buffer-bufferold)>0.5) dummy+=SIGN(bufferold-buffer);
                        bufferold=buffer;
                        visars[k]->cPhase[1][j]+=dummy;
                    }
                    offsetShift=visars[k]->cPhase[1].last();
                }
            }

            QPolygonF myLine;
            for (int i=0;i<visars[k]->cPhase[1].size();i++){
                double posx=geom2.x()+geom2.width()/2.0+(visars[k]->cPhase[1][i]+visars[k]->cPhase[0][i])*visars[k]->settingsUi->interfringe->value();
                double posy=visars[k]->time_phase[i];
                if (direction(k)==0) {		//fringes are vertical
                    myLine << QPointF(posx,posy);
                } else {
                    myLine << QPointF(posy,posx);
                }
            }
            visars[k]->fringeLine->setPoints(myLine);

            QCPGraph* graph;
            QPen pen;
            if (visars[k]->settingsUi->interfringe->value() != 0.0) {
                graph = visars[k]->settingsUi->plotPhaseIntensity->addGraph(visars[k]->settingsUi->plotPhaseIntensity->xAxis, visars[k]->settingsUi->plotPhaseIntensity->yAxis);
                graph->setName("Phase Visar "+QLocale().toString(k+1));
                pen.setColor(visars[k]->settingsUi->plotPhaseIntensity->yAxis->labelColor());
                graph->setPen(pen);
                graph->setData(visars[k]->time_phase,visars[k]->cPhase[1]);
            }

            visars[k]->velocityUi->offset->setTitle("Offset "+QLocale().toString(offsetShift));
            for (int j=0;j<visars[k]->cPhase[1].size();j++){
                if (direction(k)!=0) {		//fringes are vertical
                    visars[k]->cPhase[1][j] = offsetShift-visars[k]->cPhase[1][j];
                } else {
                    visars[k]->cPhase[1][j] = visars[k]->cPhase[1][j]-offsetShift;
                }
            }

            for (unsigned int m=0;m<2;m++) {

                pen.setStyle((m==1?Qt::SolidLine : Qt::DashLine));


                graph = visars[k]->settingsUi->plotPhaseIntensity->addGraph(visars[k]->settingsUi->plotPhaseIntensity->xAxis, visars[k]->settingsUi->plotPhaseIntensity->yAxis2);
                graph->setName("Intensity Visar "+QLocale().toString(k+1) + " " + (m==0?"ref":"shot"));
                pen.setColor(visars[k]->settingsUi->plotPhaseIntensity->yAxis2->labelColor());
                graph->setPen(pen);
                graph->setData(visars[k]->time_phase,visars[k]->cIntensity[m]);

                graph = visars[k]->settingsUi->plotPhaseIntensity->addGraph(visars[k]->settingsUi->plotPhaseIntensity->xAxis, visars[k]->settingsUi->plotPhaseIntensity->yAxis3);
                graph->setName("Contrast Visar "+QLocale().toString(k+1) + " " + (m==0?"ref":"shot"));
                pen.setColor(visars[k]->settingsUi->plotPhaseIntensity->yAxis3->labelColor());
                graph->setPen(pen);
                graph->setData(visars[k]->time_phase,visars[k]->cContrast[m]);

            }
            visars[k]->settingsUi->plotPhaseIntensity->rescaleAxes();
            visars[k]->settingsUi->plotPhaseIntensity->replot();
        }
    }
}

void
Visar::export_txt_multiple() {
    qDebug() << property("NeuSave-fileTxt");
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save VISARs and SOP"),property("NeuSave-fileTxt").toString(),tr("Text files (*.txt *.csv);;Any file (*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileTxt", fnametmp);
        QFile t(fnametmp);
        t.open(QIODevice::WriteOnly| QIODevice::Text);
        QTextStream out(&t);
        for (unsigned int k=0;k<visars.size();k++){
            out << export_one(k);
            out << Qt::endl << Qt::endl;
        }
        out << export_sop();
        t.close();
        statusBar()->showMessage(tr("Export in file:")+fnametmp,5000);
    }
}

void
Visar::export_txt() {
    QString title=tr("Export ");
    switch (tabs->currentIndex()) {
        case 0:
            title=tr("VISAR ")+QLocale().toString(tabPhase->currentIndex()+1);
            break;
        case 1:
            title=tr("VISAR ")+QLocale().toString(tabVelocity->currentIndex()+1);
            break;
        case 2:
            title=tr("SOP");
            break;
    }
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save ")+title,property("NeuSave-fileTxt").toString(),tr("Text files (*.txt *.csv);;Any file (*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileTxt", fnametmp);
        QFile t(fnametmp);
        t.open(QIODevice::WriteOnly| QIODevice::Text);
        QTextStream out(&t);
        switch (tabs->currentIndex()) {
            case 0:
                out << export_one(tabPhase->currentIndex());
                break;
            case 1:
                out << export_one(tabVelocity->currentIndex());
                break;
            case 2:
                out << export_sop();
                break;
            default:
                break;
        }
        t.close();
        statusBar()->showMessage(tr("Export in file:")+fnametmp,5000);
    }
}

void
Visar::export_clipboard() {
    QClipboard *clipboard = QApplication::clipboard();
    switch (tabs->currentIndex()) {
        case 0:
            clipboard->setText(export_one(tabPhase->currentIndex()));
            statusbar->showMessage(tr("Points copied to clipboard ")+tabPhase->tabText(tabPhase->currentIndex()));
            break;
        case 1: {
                QString txt("");
                for (unsigned int k=0;k<visars.size();k++){
                    txt += export_one(k)+"\n\n";
                }
                clipboard->setText(txt);
                statusbar->showMessage(tr("Points copied to clipboard ")+QLocale().toString(visars.size())+" visars");
                break;
            }
        case 2:
            clipboard->setText(export_sop());
            statusbar->showMessage(tr("Points copied to clipboard SOP"));
            break;
        default:
            break;
    }
}

QString Visar::export_sop() {
    QString out;
    if (enableSOP->isChecked()) {
        out += QString("#SOP Origin         : %L1\n").arg(sopOrigin->value());
        out += QString("#SOP Offset         : %L1\n").arg(sopTimeOffset->value());
        out += QString("#Center & magnif.   : %L1 %L2\n").arg(physOriginSpace->value()).arg(magnification->value());
        out += QString("#SOP Time scale     : %L1\n").arg(sopScale->text());
        out += QString("#SOP Direction      : %L1\n").arg(sopDirection->currentIndex()==0 ? "Vertical" : "Horizontal");
        out += QString("#Reflectivity       : %L1\n").arg(whichRefl->currentText());
        out += QString("#Calib              : %L1 %L2\n").arg(sopCalibT0->value()).arg(sopCalibA->value());
        QRectF geom2=sopRect->getRectF();
        out += QString("#Region (px)        : ") + QString::number(geom2.left())+" "+QString::number(geom2.right())+" "+QString::number(geom2.top())+" "+QString::number(geom2.bottom())+"\n";
        out += QString("#Time\tCounts\tTblackbody\tTgrayIn\tTgrayOut\n");

        for (int i=0;i<time_sop.size();i++) {
            out += QLocale().toString(time_sop[i])+ " ";
            for (unsigned int j=0;j<sopCurve.size();j++) {
                double val=sopCurve[j][i];
                out+=(val>=0?"+":"-")+QLocale().toString(fabs(val),'E',4)+ " ";
            }
            out += "\n";
        }
    }
    return out;
}

QString Visar::export_one(unsigned int k) {
    QString out;
    if (k<visars.size()) {
        if (visars[k]->velocityUi->enableVisar->isChecked()) {
            out += "#VISAR " + QLocale().toString(k+1) + "\n";
            out += QString("#Offset shift       : %L1 (%L2)\n").arg(visars[k]->velocityUi->offsetShift->value()).arg(visars[k]->velocityUi->offset->title());
            out += QString("#Sensitivity        : %L1\n").arg(visars[k]->settingsUi->sensitivity->value());
            out += QString("#Slit               : %L1\n").arg(visars[k]->settingsUi->resolution->value());
            out += QString("#Interfr. & angle   : %L1 %L2\n").arg(visars[k]->settingsUi->interfringe->value()).arg(visars[k]->settingsUi->angle->value());
            out += QString("#Intensity params   : %L1 %L2 %L3\n").arg(visars[k]->settingsUi->offRef->value()).arg(visars[k]->settingsUi->intensityShift->value()).arg(visars[k]->settingsUi->multRef->value());
            out += QString("#Reflectivity       : %L1 %L2\n").arg(visars[k]->velocityUi->reflOffset->value()).arg(visars[k]->velocityUi->reflRef->value());
            out += QString("#Sweep Time         : %L1 %L2\n").arg(visars[k]->settingsUi->physScaleType->currentText()).arg(visars[k]->settingsUi->physScale->text());
            out += QString("#Time zero & delay  : %L1 %L2\n").arg(visars[k]->velocityUi->physOrigin->value()).arg(visars[k]->velocityUi->offsetTime->value());
            out += QString("#Center & magnif.   : %L1 %L2\n").arg(visars[k]->settingsUi->physOriginSpace->value()).arg(visars[k]->settingsUi->magnification->value());
            out += QString("#Jumps              : %L1\n").arg(visars[k]->velocityUi->jumpst->text());
            QRectF geom2=visars[k]->fringeRect->getRectF();
            out += QString("#Region (px)        : ") + QString::number(geom2.left())+" "+QString::number(geom2.right())+" "+QString::number(geom2.top())+" "+QString::number(geom2.bottom())+"\n";
            out += QString("# Time       Velocity    ErrVel      Reflect     ErrRefl     Quality     Pixel       Shift       ErrShift    RefInt      ShotInt     RefContr    ShotContr.\n");
            for (int j=0;j<visars[k]->time_phase.size();j++) {
                QVector<double> values {visars[k]->time_vel[j],visars[k]->velocity[j], visars[k]->velError[j],
                            visars[k]->reflectivity[j], visars[k]->reflError[j], visars[k]->quality[j],
                            visars[k]->time_phase[j],
                            visars[k]->cPhase[1][j],visars[k]->cPhaseErr[j],
                            visars[k]->cIntensity[0][j], visars[k]->cIntensity[1][j],
                            visars[k]->cContrast[0][j],visars[k]->cContrast[1][j]};
                foreach (double val, values) {
                    out+=(val>=0?"+":"-")+QLocale().toString(fabs(val),'E',4)+ " ";
                }
                out += '\n';
            }
        }
    }
    return out;
}

void Visar::copy_image() {
    switch (tabs->currentIndex()) {
        case 0:
            if (tabPhase->currentIndex()>=0 && tabPhase->currentIndex()<(int)visars.size())
                visars[tabPhase->currentIndex()]->settingsUi->plotPhaseIntensity->copy_image();
            statusbar->showMessage("Phase image copied to clipboard",3000);
            break;
        case 1:
            plotVelocity->copy_image();
            statusbar->showMessage("Velocity image copied to clipboard",3000);
            break;
        case 2:
            sopPlot->copy_image();
            statusbar->showMessage("SOP image copied to clipboard",3000);
            break;
    }
}
