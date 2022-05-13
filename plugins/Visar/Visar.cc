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
#include "neutrino.h"

#include "ui_VisarSettings.h"
#include "ui_VisarVelocity.h"


VisarPhasePlot::VisarPhasePlot(QWidget* parent):
    nCustomPlotMouseX3Y(parent)
{

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
    nCustomPlotMouseX3Y(parent)
{
    xAxis->setLabel(tr("Time"));
    yAxis->setLabel(tr("Velocity"));
    yAxis2->setLabel(tr("Reflectivity"));
    yAxis3->setLabel(tr("Quality"));
}


nSOPPlot::nSOPPlot(QWidget* parent):
    nCustomPlotMouseX2Y(parent)
{
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

};


Visar::Visar(neutrino *parent) : nGenericPan(parent),
      numVisars(0)
{
    setupUi(this);

    //!START SOP stuff
    sopRect =  new nRect(this,1);
    sopRect->changeToolTip("SOP region");
    sopRect->setRect(QRectF(0,0,100,100));
    connect(actionRect3, SIGNAL(triggered()), sopRect, SLOT(togglePadella()));

    sopPlot->xAxis->setLabel(tr("Time"));
    sopPlot->yAxis->setLabel(tr("Counts"));
    sopPlot->yAxis2->setLabel(tr("Temperature"));


    show();


    setProperty("NeuSave-alphagraph",30);


    for (int l=2+numVisars; l<whichRefl->count();l++) {
        whichRefl->removeItem(l);
    }

    unsigned int kMax=1;
    if (property("NeuSave-numVisars").isValid()) {
        kMax=property("NeuSave-numVisars").toUInt();
    }
    for (unsigned int k=0; k<kMax; k++) {
        addVisar();
    }

    connect(actionAddVisar, SIGNAL(triggered()), this, SLOT(addVisar()));
    connect(actionDelVisar, SIGNAL(triggered()), this, SLOT(delVisar()));

    connect(actionSaveTxt, SIGNAL(triggered()), this, SLOT(export_txt()));
    connect(actionSaveTxtMultiple, SIGNAL(triggered()), this, SLOT(export_txt_multiple()));
    connect(actionCopy, SIGNAL(triggered()), this, SLOT(export_clipboard()));
    connect(actionCopyImage, SIGNAL(triggered()), this, SLOT(copy_image()));

//    connect(actionRefresh, SIGNAL(triggered()), this, SLOT(doWave()));

    connect(etalon_thickness, SIGNAL(valueChanged(double)), this, SLOT(calculate_etalon()));
    connect(etalon_dn_over_dlambda, SIGNAL(valueChanged(double)), this, SLOT(calculate_etalon()));
    connect(etalon_n0, SIGNAL(valueChanged(double)), this, SLOT(calculate_etalon()));
    connect(etalon_lambda, SIGNAL(valueChanged(double)), this, SLOT(calculate_etalon()));

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

    connect(plotVelocity,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    connect(sopPlot,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    connect(sopScale,SIGNAL(editingFinished()), this, SLOT(sweepChanged()));

    loadDefaults();
    sweepChanged();
    calculate_etalon();
}

void Visar::calculate_etalon() {
    double lam=etalon_lambda->value();
    double e=etalon_thickness->value();
    double n0=etalon_n0->value();
    double dn_dl = etalon_dn_over_dlambda->value();
    double c=_phys_cspeed;

    double delta= -1e-3*lam*n0/(n0*n0-1)*dn_dl;
    double tau=2*e/c*(n0-1.0/n0);

    double sens=lam*1e-9/(2*tau*(1+delta)); // km/s

    qDebug() << "delta tau sens"  << delta << tau << sens;
    etalon_sensitivity->setValue(sens);

}

void Visar::addVisar() {
    qDebug() << numVisars << tabPhase->count();

    QWidget *tab1 = new QWidget();
    QGridLayout *gridLayout1 = new QGridLayout(tab1);
    gridLayout1->setContentsMargins(0, 0, 0, 0);
    QWidget*wVisarSettings = new QWidget(tab1);
    wVisarSettings->setObjectName(QStringLiteral("wVisarVelocity"));
    gridLayout1->addWidget(wVisarSettings, 0, 0, 1, 1);

    tabPhase->addTab(tab1, "Visar"+QLocale().toString(numVisars+1));

    Ui::VisarSettings* ui_settings=new Ui::VisarSettings();
    ui_settings->setupUi(wVisarSettings);
    settingsUi.push_back(ui_settings);

    QWidget *tab2 = new QWidget();
    QGridLayout *gridLayout2 = new QGridLayout(tab2);
    gridLayout2->setContentsMargins(0, 0, 0, 0);
    QWidget*wVisarVelocity = new QWidget(tab2);
    wVisarVelocity->setObjectName(QStringLiteral("wvisarVelocity"));
    gridLayout2->addWidget(wVisarVelocity, 0, 0, 1, 1);

    tabVelocity->addTab(tab2, "Visar"+QLocale().toString(numVisars+1));

    Ui::VisarVelocity* ui_velocity=new Ui::VisarVelocity();
    ui_velocity->setupUi(wVisarVelocity);
    velocityUi.push_back(ui_velocity);

    //hack to save diffrent uis!!!
    foreach (QWidget *obj, wVisarSettings->findChildren<QWidget*>()+wVisarVelocity->findChildren<QWidget*>()) {
        obj->setObjectName(obj->objectName()+"-VISAR"+QLocale().toString(numVisars+1));
        obj->setProperty("id", numVisars);
    }

    decorate(tab1);
    decorate(tab2);


    QString name="Visar "+QLocale().toString(numVisars+1);
    phaseUnwrap.push_back(nPhysD());

    phaseUnwrap[numVisars].setName(name.toUtf8().constData());
    phaseUnwrap[numVisars].setShortName("phase");

    contrast.push_back({{nPhysD(),nPhysD()}});
    intensity.push_back({{nPhysD(),nPhysD()}});
    for (uint m=0;m<2;m++){
        QString name2=name+" "+QLocale().toString(m);
        contrast[numVisars][m].setName(name2.toUtf8().constData());
        contrast[numVisars][m].setShortName("contrast");
        intensity[numVisars][m].setName(name2.toUtf8().constData());
        intensity[numVisars][m].setShortName("intensity");
    }

    QApplication::processEvents();

    sweepCoeff.push_back(std::vector<double>());

    QAction *actionRect = new QAction(QIcon(":icons/rect.png"), "Region Visar"+QLocale().toString(numVisars+1),this);
    actionRect->setProperty("id",numVisars);

    toolBar->insertAction(actionDelVisar,actionRect);

    whichRefl->addItem("Visar"+QLocale().toString(numVisars+1));

    nLine* my_nline=new nLine(this,3);
    my_nline->changeToolTip("Fringeshift Visar "+QLocale().toString(numVisars+1));
    fringeLine.push_back(my_nline);

    nRect *my_rect=new nRect(this,1);
    my_rect->setRect(QRectF(0,0,100,100));
    my_rect->setProperty("id", numVisars);
    my_rect->changeToolTip("Visar region "+QLocale().toString(numVisars+1));
    connect(actionRect, SIGNAL(triggered()),my_rect, SLOT(togglePadella()));
    fringeRect.push_back(my_rect);


    velocity.push_back(QVector<double>());
    velError.push_back(QVector<double>());
    reflectivity.push_back(QVector<double>());
    reflError.push_back(QVector<double>());
    quality.push_back(QVector<double>());
    time_vel.push_back(QVector<double>());
    time_phase.push_back(QVector<double>());

    cPhase.push_back(QVector<double>());
    cPhaseErr.push_back(QVector<double>());
    cReflErr.push_back(QVector<double>());
    for (unsigned int m=0; m<2;m++) {
        cIntensity[m].push_back(QVector<double>());
        cContrast[m].push_back(QVector<double>());
    }

    QCPGraph* graph;
    QPen pen;
    QCPErrorBars *errorBars;
    QColor my_color;

    graph = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis3);
    graph->setName("Quality Visar "+QLocale().toString(numVisars+1));
    graph->setProperty("id",numVisars);
    pen.setColor(plotVelocity->yAxis3->labelColor());
    graph->setPen(pen);

    graph = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis2);
    graph->setName("Reflectivity Visar "+QLocale().toString(numVisars+1));
    graph->setProperty("id",numVisars);
    pen.setColor(plotVelocity->yAxis2->labelColor());
    graph->setPen(pen);

    errorBars = new QCPErrorBars(plotVelocity->xAxis, plotVelocity->yAxis2);
    errorBars->setName("Reflectivity error Visar "+QLocale().toString(numVisars+1));
    errorBars->setDataPlottable(graph);
    errorBars->setProperty("id",numVisars);
    my_color=plotVelocity->yAxis2->labelColor();
    my_color.setAlpha(property("NeuSave-alphagraph").toInt());
    pen.setColor(my_color);
    errorBars->setPen(pen);
    errorBars->setWhiskerWidth(0);
    errorBars->setSymbolGap(1);

    graph = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis);
    graph->setName("Velocity Visar "+QLocale().toString(numVisars+1));
    graph->setProperty("id",numVisars);
    pen.setColor(plotVelocity->yAxis->labelColor());
    graph->setPen(pen);


    errorBars = new QCPErrorBars(plotVelocity->xAxis, plotVelocity->yAxis);
    errorBars->setName("Velocity error Visar "+QLocale().toString(numVisars+1));
    errorBars->setDataPlottable(graph);
    errorBars->setProperty("id",numVisars);
    my_color=plotVelocity->yAxis->labelColor();
    my_color.setAlpha(property("NeuSave-alphagraph").toInt());
    pen.setColor(my_color);
    errorBars->setPen(pen);
    errorBars->setWhiskerWidth(0);
    errorBars->setSymbolGap(1);

    unsigned int k=numVisars;
    connect(fringeRect[k], SIGNAL(sceneChanged()), this, SLOT(getPhase()));
    connect(velocityUi[k]->offsetShift, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
    connect(settingsUi[k]->sensitivity, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
    connect(velocityUi[k]->jumpst, SIGNAL(editingFinished()), this, SLOT(updatePlot()));
    connect(velocityUi[k]->reflRef, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
    connect(velocityUi[k]->reflOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
    connect(velocityUi[k]->jump, SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));

    connect(velocityUi[k]->physOrigin, SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));
    connect(velocityUi[k]->offsetTime, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));

    connect(settingsUi[k]->guess, SIGNAL(released()), this, SLOT(getCarrier()));
    connect(settingsUi[k]->doWaveButton, SIGNAL(released()), this, SLOT(doWave()));

    connect(settingsUi[k]->multRef, SIGNAL(valueChanged(double)), this, SLOT(getPhase()));
    connect(settingsUi[k]->offRef, SIGNAL(editingFinished()), this, SLOT(getPhase()));
    connect(settingsUi[k]->intensityShift, SIGNAL(valueChanged(int)), this, SLOT(getPhase()));

    connect(velocityUi[k]->enableVisar, SIGNAL(released()), this, SLOT(updatePlot()));
    connect(velocityUi[k]->plotR, SIGNAL(released()), this, SLOT(updatePlot()));
    connect(velocityUi[k]->plotQ, SIGNAL(released()), this, SLOT(updatePlot()));

    connect(settingsUi[k]->physScale,SIGNAL(editingFinished()), this, SLOT(sweepChanged()));

    connect(settingsUi[k]->plotPhaseIntensity,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));

    connect(settingsUi[k]->interfringe, SIGNAL(valueChanged(double)), this, SLOT(needWave()));
    connect(settingsUi[k]->angle, SIGNAL(valueChanged(double)), this, SLOT(needWave()));
    connect(settingsUi[k]->resolution, SIGNAL(valueChanged(double)), this, SLOT(needWave()));
    connect(settingsUi[k]->refImage, SIGNAL(currentIndexChanged(int)), this, SLOT(needWave()));
    connect(settingsUi[k]->shotImage, SIGNAL(currentIndexChanged(int)), this, SLOT(needWave()));

    numVisars++;

    setProperty("NeuSave-numVisars",numVisars);
}

void Visar::delVisar() {
    if (numVisars>0) {
        QWidget* my_widget=tabPhase->widget(numVisars-1);
        tabPhase->removeTab(numVisars-1);
        my_widget->deleteLater();
        my_widget=tabVelocity->widget(numVisars-1);
        tabVelocity->removeTab(numVisars-1);
        my_widget->deleteLater();
        foreach (QAction *action, toolBar->actions()) {
            if (action->property("id").isValid() && action->property("id").toUInt()==numVisars-1) {
                toolBar->removeAction(action);
            }
        }
        QApplication::processEvents();

//        fringeLine.back()->deleteLater();
        delete fringeLine.back();
        QApplication::processEvents();
        fringeLine.pop_back();
//        fringeRect.back()->deleteLater();
        delete fringeRect.back();
        QApplication::processEvents();
        fringeRect.pop_back();

        settingsUi.pop_back();
        velocityUi.pop_back();

        phaseUnwrap.pop_back();

        contrast.pop_back();
        intensity.pop_back();

        sweepCoeff.pop_back();

        whichRefl->removeItem(numVisars+1);

        velocity.pop_back();
        velError.pop_back();
        reflectivity.pop_back();
        reflError.pop_back();
        quality.pop_back();
        time_vel.pop_back();
        time_phase.pop_back();

        cPhase.pop_back();
        cPhaseErr.pop_back();
        cReflErr.pop_back();
        for (unsigned int m=0; m<2;m++) {
            cIntensity[m].pop_back();
            cContrast[m].pop_back();
        }

        QList<QCPAbstractPlottable *> listplottable;
        for (int kk=0; kk< plotVelocity->plottableCount() ; kk++) {
            QVariant id=plotVelocity->plottable(kk)->property("id");
            if (id.isValid() && id.toUInt() == numVisars-1 ){
                listplottable << plotVelocity->plottable(kk);
            }
        }
        for (auto &plot: listplottable) {
            plotVelocity->removePlottable(plot);
        }

        QApplication::processEvents();
        numVisars--;
        setProperty("NeuSave-numVisars",numVisars);
    } else {
        statusbar->showMessage("Cannot remove last Visar");
    }
    updatePlot();
}

void Visar::loadSettings(QString my_settings) {
    if (my_settings.isEmpty()) {
        QString fname = QFileDialog::getOpenFileName(this, tr("Open INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf);; Any files (*.*)"));
        if (!fname.isNull()) {
            setProperty("NeuSave-fileIni",fname);
            loadSettings(fname);
        }
    } else {
        QSettings settings(my_settings,QSettings::IniFormat);
        settings.beginGroup("Properties");
        int kMax=settings.value("NeuSave-numVisars",2).toUInt();
        unsigned int numVisars_save=numVisars;
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
        setObjectVisibility(currentBuffer);
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

void Visar::sweepChanged(QLineEdit *line) {
    if(line==nullptr) {
        if (sender()) {
            QLineEdit *line=qobject_cast<QLineEdit *>(sender());
            if (line) {
                sweepChanged(line);
            }
        } else {
            for (unsigned int n=0; n< numVisars; n++) {
                sweepChanged(settingsUi[n]->physScale);
            }
            sweepChanged(sopScale);
        }
    } else {
        int k=0;
        std::vector<double> *vecsweep=nullptr;
        if (line==sopScale) {
            vecsweep = &sweepCoeffSOP;
        } else {
            k=line->property("id").toUInt();
            vecsweep = &sweepCoeff[k];
        }

        if (vecsweep) {
            vecsweep->clear();
            line->setPalette(QApplication::palette());
            QRegExp separator("[(;| |\t)]");
            QStringList strSweep=line->text().split(separator, QString::SkipEmptyParts);
            qDebug() << strSweep;
            foreach(QString str, strSweep) {
                bool ok;
                double coeff=locale().toDouble(str,&ok);
                if(ok) {
                    vecsweep->push_back(coeff);
                } else {
                    QPalette my_palette=line->palette();
                    my_palette.setColor(QPalette::Base,Qt::red);
                    line->setPalette(my_palette);
                    statusbar->showMessage("Cannot understant sweep coefficent "+str);
                    break;
                }
            }

            if (line==sopScale) {
                updatePlotSOP();
            } else {
                getPhase(k);
                updatePlot();
            }
        }
    }
}

double Visar::getTime(std::vector<double> &vecsweep, double p) {
    double time=0;
    for (unsigned int i=0;i<vecsweep.size();i++) {
        time+=vecsweep.at(i)/(i+1.0)*pow(p,i+1);
    }
    return time;
}

QPointF Visar::getTimeSpaceFromPixel(QPointF p) {
    int k=0;
    double posTime=0.0;
    double posSpace=0.0;
    if (tabs->currentIndex()==0) {
        k=tabPhase->currentIndex();
        if (k >= 0 && k<(int)numVisars) {
            posTime=(direction(k)==0 ? p.y() : p.x());
            posSpace=(direction(k)==0 ? p.x() : p.y())-settingsUi[k]->physOriginSpace->value();
            settingsUi[k]->plotPhaseIntensity->setMousePosition(posTime);
        }
    } else if (tabs->currentIndex()==1) {
        k=tabVelocity->currentIndex();
        if (k >= 0 && k<(int)numVisars) {
            double pos=direction(k)==0 ? p.y() : p.x();
            posTime=getTime(sweepCoeff[k],pos) - getTime(sweepCoeff[k],velocityUi[k]->physOrigin->value()) + velocityUi[k]->offsetTime->value();
            posSpace=((direction(k)==0 ? p.x() : p.y())-settingsUi[k]->physOriginSpace->value())*settingsUi[k]->magnification->value();
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
    int dir=((int) ((settingsUi[k]->angle->value()+360+45)/90.0) )%2;
    return dir;
}

void Visar::setObjectVisibility(nPhysD*phys) {
    if (phys) {
        for (unsigned int k=0;k<numVisars;k++){
            fringeLine[k]->setVisible(phys == getPhysFromCombo(settingsUi[k]->shotImage) || phys == getPhysFromCombo(settingsUi[k]->refImage));
            fringeRect[k]->setVisible(phys == getPhysFromCombo(settingsUi[k]->shotImage) || phys == getPhysFromCombo(settingsUi[k]->refImage));
        }
        sopRect->setVisible(enableSOP->isChecked() && (phys == getPhysFromCombo(sopRef) || phys == getPhysFromCombo(sopShot)));
    }
}

void Visar::tabChanged(int k) {
    QTabWidget *tabWidget=nullptr;

    if (sender()) tabWidget=qobject_cast<QTabWidget *>(sender());

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
            if (k<(int)numVisars) {
                k=tabWidget->currentIndex();
                if (k>=0 && velocityUi[k]->enableVisar->isChecked()) {
                    nparent->showPhys(getPhysFromCombo(settingsUi[k]->shotImage));
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
                for (unsigned int i=0; i<numVisars; i++) {
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
            double my_velocity=0;

            int numrefl=0;
            for (int numk=0;numk<reflList.size();numk++) {
                int k=reflList[numk];

                if ( k>=0 && k<(int)numVisars ) {
                    for (int j=1;j<time_vel[k].size();j++) {
                        double t_j1=time_vel[k][j-1];
                        double t_j=time_vel[k][j];

                        if (time_sop[i]>t_j1 && time_sop[i]<=t_j ) {
                            double valRj_1=reflectivity[k][j-1];
                            double valRj=reflectivity[k][j];

                            double valVj_1=velocity[k][j-1];
                            double valVj=velocity[k][j];

                            my_reflectivity+=valRj_1+(time_sop[i]-t_j1)*(valRj-valRj_1)/(t_j-t_j1);
                            my_velocity+=valVj_1+(time_sop[i]-t_j1)*(valVj-valVj_1)/(t_j-t_j1);

                            numrefl++;
                        }
                    }
                }
            }
            if (numrefl) {
                my_reflectivity/=numrefl;
                my_velocity/=numrefl;
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
    for (unsigned int k=0;k<numVisars;k++){
        if (velocityUi[k]->enableVisar->isChecked()) {
            if (velocityUi[k]->plotR->isChecked()) needReflectivityAxe=true;
            if (velocityUi[k]->plotQ->isChecked()) needQualityAxe=true;
        }
    }
    plotVelocity->yAxis2->setVisible(needReflectivityAxe);
    plotVelocity->yAxis3->setVisible(needQualityAxe);

    plotVelocity->clearItems();

    for (int g=0; g<plotVelocity->plottableCount(); g++) {
        if (plotVelocity->plottable(g)->property("JumpGraph").isValid()) {
            plotVelocity->removePlottable(plotVelocity->plottable(g));
        }
    }

    std::vector<double> minmax;
    for (unsigned int k=0;k<numVisars;k++){
        if (cPhase[k].size()==time_phase[k].size()){

            Qt::PenStyle pstyle=((int)k==tabVelocity->currentIndex()?Qt::SolidLine : Qt::DashLine);

            double sensitivity=settingsUi[k]->sensitivity->value();
            double deltat=velocityUi[k]->offsetTime->value()-getTime(sweepCoeff[k],velocityUi[k]->physOrigin->value());

            QVector<double> tjump,njump,rjump;
            QStringList jumpt=velocityUi[k]->jumpst->text().split(";", QString::SkipEmptyParts);
            velocityUi[k]->jumpst->setPalette(QApplication::palette());
            QPalette my_palette=velocityUi[k]->jumpst->palette();
            my_palette.setColor(QPalette::Base,Qt::red);

            foreach (QString piece, jumpt) {
                QString err_msg=" "+piece+QString("' VISAR ")+QLocale().toString(k+1)+tr(" Decimal separator is: ")+locale().decimalPoint();
                QStringList my_jumps=piece.split(QRegExp("\\s+"), QString::SkipEmptyParts);
                if (my_jumps.size()>1 && my_jumps.size()<=3) {
                    if (my_jumps.size()>1 && my_jumps.size()<=3) {
                        bool ok1, ok2, ok3=true;
                        double valdt=locale().toDouble(my_jumps.at(0),&ok1);
                        double valdn=locale().toDouble(my_jumps.at(1),&ok2);
                        double valdrefr_index=1.0;
                        if (my_jumps.size()==3) {
                            valdrefr_index=locale().toDouble(my_jumps.at(2),&ok3);
                            if (!ok3) {
                                velocityUi[k]->jumpst->setPalette(my_palette);
                                statusbar->showMessage(tr("Skipped unreadable refraction index '")+err_msg,5000);
                            }
                        }
                        if (sensitivity<0) valdn*=-1.0;
                        if (ok1 && ok2) {
                            tjump << valdt;
                            njump << valdn;
                        } else {
                            velocityUi[k]->jumpst->setPalette(my_palette);
                            statusbar->showMessage(tr("Skipped unreadable jump '")+err_msg,5000);
                        }
                        if (ok3) {
                            rjump << valdrefr_index;
                        } else {
                            rjump << 1.0;
                        }
                    }
                } else {
                    velocityUi[k]->jumpst->setPalette(my_palette);
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

            double offset=velocityUi[k]->offsetShift->value();

            QVector<QVector< double > > velJump_array(abs(velocityUi[k]->jump->value()));

            for (int i=0;i<abs(velocityUi[k]->jump->value());i++) {
                velJump_array[i].resize(time_phase[k].size());
            }

            time_vel[k].resize(time_phase[k].size());
            velocity[k].resize(time_phase[k].size());
            velError[k].resize(time_phase[k].size());
            reflectivity[k].resize(time_phase[k].size());
            reflError[k].resize(time_phase[k].size());
            quality[k].resize(time_phase[k].size());

            for (int j=0;j<time_phase[k].size();j++) {
                time_vel[k][j] = getTime(sweepCoeff[k],time_phase[k][j])+deltat;

                double fShot=cPhase[k][j];
                double iRef=cIntensity[0][k][j];
                double iShot=cIntensity[1][k][j];
                if (getPhysFromCombo(settingsUi[k]->shotImage)==getPhysFromCombo(settingsUi[k]->refImage)) {
                    iRef=1.0;
                }

                int njumps=0;
                double refr_index=1.0;
                for (int i=0;i<tjump.size();i++) {
                    if (time_vel[k][j]>tjump.at(i)) {
                        njumps+=njump.at(i);
                        refr_index=rjump.at(i);
                    }
                }

                double speed=(offset+fShot+njumps)*sensitivity/refr_index;
                double Rg=velocityUi[k]->reflOffset->value();
                double Rmat=velocityUi[k]->reflRef->value();
                double beta=-Rg/pow(1.0-Rg,2);
                double refle=iShot/iRef * (Rmat-beta) + beta;

                velocity[k][j] = speed;
                reflectivity[k][j] = refle;
                quality[k][j] = cContrast[1][k][j]/cContrast[0][k][j];
                velError[k][j] = 2.0*abs(cPhaseErr[k][j]*sensitivity/refr_index);
                reflError[k][j] = cReflErr[k][j]* (Rmat-beta) + beta;

                for (int i=0;i<abs(velocityUi[k]->jump->value());i++) {
                    int jloc=i+1;
                    if (sensitivity<0) jloc*=-1;
                    if (velocityUi[k]->jump->value()<0) jloc*=-1;
                    velJump_array[i][j] = (offset+fShot+jloc)*sensitivity/refr_index;
                }
            }

            QPen pen;
            pen.setStyle(pstyle);

            for (int kk=0; kk< plotVelocity->graphCount() ; kk++) {
                QCPGraph *my_graph=plotVelocity->graph(kk);
                if (my_graph->property("id").toUInt() == k ){
                    pen=my_graph->pen();
                    pen.setStyle(pstyle);
                    my_graph->setPen(pen);
                    if (my_graph->valueAxis()==plotVelocity->yAxis) {
                        if (settingsUi[k]->interfringe->value() != 0.0) {
                            my_graph->setData(time_vel[k],velocity[k]);
                        } else {
                            my_graph->data()->clear();
                        }
                    } else if (my_graph->valueAxis()==plotVelocity->yAxis2) {
                        if (velocityUi[k]->plotR->isChecked()) {
                            my_graph->setData(time_vel[k],reflectivity[k]);
                        } else {
                            my_graph->data()->clear();
                        }
                    } else if (my_graph->valueAxis()==plotVelocity->yAxis3) {
                        if (settingsUi[k]->interfringe->value() != 0.0 && velocityUi[k]->plotQ->isChecked() ) {
                            my_graph->setData(time_vel[k],quality[k]);
                        } else {
                            my_graph->data()->clear();
                        }
                    }
                    my_graph->setVisible(velocityUi[k]->enableVisar->isChecked());
                }
            }
            for (int kk=0; kk< plotVelocity->plottableCount() ; kk++) {
                QCPErrorBars *my_err = qobject_cast<QCPErrorBars*>(plotVelocity->plottable(kk));
                if (my_err && my_err->property("id").toUInt() == k ){
                    pen=my_err->pen();
                    pen.setStyle(pstyle);
                    my_err->setPen(pen);
                    if (my_err->valueAxis() == plotVelocity->yAxis) {
                        my_err->setData(velError[k]);
                    } else if (my_err->valueAxis() == plotVelocity->yAxis2) {
                        my_err->setData(reflError[k]);
                    }
                    my_err->setVisible(velocityUi[k]->enableVisar->isChecked());
                }
            }

            if (velocityUi[k]->jump->value()!=0) {
                for (int i=0;i<abs(velocityUi[k]->jump->value());i++) {
                    QCPGraph* graph = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis);
                    graph->setProperty("JumpGraph",k);
                    graph->setName("VelJump Visar"+QLocale().toString(k+1) + " #" +QLocale().toString(i));
                    QColor color(plotVelocity->yAxis->labelColor());
                    color.setAlpha(property("NeuSave-alphagraph").toInt());
                    pen.setColor(color);
                    graph->setPen(pen);
                    graph->setData(time_vel[k],velJump_array[i]);
                    graph->setVisible(velocityUi[k]->enableVisar->isChecked());
                }

            }
            if (time_vel[k].size()) {
                double mmin = *std::min_element(time_vel[k].constBegin(), time_vel[k].constEnd());
                double mmax = *std::max_element(time_vel[k].constBegin(), time_vel[k].constEnd());

                minmax.push_back(mmin);
                minmax.push_back(mmax);
                qDebug() << k << mmin << mmax;
            }
        }
    }

    plotVelocity->rescaleAxes(true);
    if (minmax.size()) {
        double mmin = *std::min_element(minmax.begin(),minmax.end());
        double mmax = *std::max_element(minmax.begin(),minmax.end());
        qDebug() << mmin << mmax;

        plotVelocity->xAxis->setRange(mmin,mmax);
    }

    plotVelocity->replot();
    updatePlotSOP();
}

void Visar::getCarrier() {
    if (sender() && sender()->property("id").isValid()) {
        unsigned int k=sender()->property("id").toUInt();
        getCarrier(k);
    }
}

void Visar::getCarrier(unsigned int k) {
    QComboBox *combo=nullptr;
    if (settingsUi[k]->carrierPhys->currentIndex()==0) {
        combo=settingsUi[k]->refImage;
    } else {
        combo=settingsUi[k]->shotImage;
    }

    nPhysD *phys=getPhysFromCombo(combo);
    if (phys && fringeRect[k]) {
        QRect geom2=fringeRect[k]->getRect(phys);
        nPhysD datamatrix = phys->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());

        vec2f vecCarr=physWave::phys_guess_carrier(datamatrix, settingsUi[k]->guessWeight->value());

        if (vecCarr.first()==0) {
            statusbar->showMessage("ERROR: Problem finding the carrier try to change the weight", 5000);
        } else {
            settingsUi[k]->interfringe->setValue(vecCarr.first());
            settingsUi[k]->angle->setValue(vecCarr.second());
            if (tabPhase->currentIndex() == static_cast<int>(k)) {
                statusbar->showMessage(tr("Carrier :")+QLocale().toString(vecCarr.first())+tr("px, ")+QLocale().toString(vecCarr.second())+tr("deg"));
            }
        }
    }
}

void Visar::doWave() {
    if (sender() && sender()->property("id").isValid()) {
        unsigned int k=sender()->property("id").toUInt();
        doWave(k);
    } else {
        for (unsigned int k=0;k<numVisars;k++){
            doWave(k);
        }
    }
}

void Visar::needWave() {
    qDebug() << ">>>>>>>>>>>>>>>>>>>>> CALLING IN THE NAME OF" << sender() << sender()->property("id");
    if (sender() && sender()->property("id").isValid()) {
        unsigned int k=sender()->property("id").toUInt();
        if (k< numVisars) {
            settingsUi[k]->doWaveButton->setIcon(QIcon(":icons/refreshRed.png"));
        }
    }
}

void Visar::doWave(unsigned int k) {
    if (k< numVisars) {
        std::array<nPhysD*,2> imgs={{getPhysFromCombo(settingsUi[k]->refImage),getPhysFromCombo(settingsUi[k]->shotImage)}};
        if (imgs[0] && imgs[1]  && imgs[0]->getSize() == imgs[1]->getSize()) {
            QProgressDialog progress("Filter visar "+QLocale().toString(k+1), "Cancel", 0, 19, this);
            progress.setCancelButton(nullptr);
            progress.setWindowModality(Qt::WindowModal);
            progress.setValue(0);
            progress.show();
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            sweepChanged(settingsUi[k]->physScale);

            std::array<physC,2> physfft={{imgs[0]->ft2(PHYS_FORWARD),imgs[1]->ft2(PHYS_FORWARD)}};
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            vec2i dim(imgs[0]->getSize());

            std::array<physC,2> zz_morlet;
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            std::array<physD,2> phase({{nPhysD(),nPhysD()}});

            for (int m=0;m<2;m++) {
                phase[m].resize(dim.x(), dim.y());
                contrast[k][m].resize(dim.x(), dim.y());
                intensity[k][m]= imgs[m]->copy();
                progress.setValue(progress.value()+1);
                qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
                physMath::phys_fast_gaussian_blur(intensity[k][m], settingsUi[k]->resolution->value());
                progress.setValue(progress.value()+1);
                qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
                zz_morlet[m].resize(dim.x(),dim.y());
                progress.setValue(progress.value()+1);
                qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            }

            std::vector<int> xx(dim.x()), yy(dim.y());
#pragma omp parallel for
            for (size_t i=0;i<(size_t)dim.x();i++) xx[i]=(i+(dim.x()+1)/2)%dim.x()-(dim.x()+1)/2; // swap and center
#pragma omp parallel for
            for (size_t i=0;i<(size_t)dim.y();i++) yy[i]=(i+(dim.y()+1)/2)%dim.y()-(dim.y()+1)/2;

            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            double cr = cos((settingsUi[k]->angle->value()) * _phys_deg);
            double sr = sin((settingsUi[k]->angle->value()) * _phys_deg);
            double thick_norm=settingsUi[k]->resolution->value()*M_PI/sqrt(pow(sr*dim.x(),2)+pow(cr*dim.y(),2));
            const double damp_norm=M_PI;

            double lambda_norm=settingsUi[k]->interfringe->value()/sqrt(pow(cr*dim.x(),2)+pow(sr*dim.y(),2));
            for (unsigned int m=0;m<2;m++) {
#pragma omp parallel for collapse(2)
                for (size_t x=0;x<(size_t)dim.x();x++) {
                    for (size_t y=0;y<(size_t)dim.y();y++) {
                        double xr = xx[x]*cr - yy[y]*sr; //rotate
                        double yr = xx[x]*sr + yy[y]*cr;

                        double e_x = -pow(damp_norm*(xr*lambda_norm-1.0), 2);
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
                for (size_t kk=0; kk<(size_t)(dim.x()*dim.y()); kk++) {
                    phase[m].Timg_buffer[kk] = -physfft[m].Timg_buffer[kk].arg()/(2*M_PI);
                    contrast[k][m].Timg_buffer[kk] = 2.0*physfft[m].Timg_buffer[kk].mod()/(dim.x()*dim.y());
                    intensity[k][m].Timg_buffer[kk] -= contrast[k][m].point(kk)*cos(2*M_PI*phase[m].point(kk));
                }
            }

            if (direction(k)!=0) {
                for (unsigned int m=0;m<2;m++) {
                    physMath::phys_transpose(phase[m]);
                    physMath::phys_transpose(contrast[k][m]);
                    physMath::phys_transpose(intensity[k][m]);
                }
            }
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
    //unwrap

            physD diff = phase[1]-phase[0];
            physD qual = contrast[k][1]*contrast[k][0];

    //        physD diff(dim.x(),dim.y(),0,"diff");
    //        physD qual(dim.x(),dim.y(),0,"qual");
    //        for (size_t kk=0; kk<(size_t)(dim.x()*dim.y()); kk++) {
    //            diff.set(kk,phase[1].point(kk)-phase[0].point(kk));
    //            qual.set(kk,contrast[k][1].point(kk)*contrast[k][0].point(kk));
    //        }

            physWave::phys_phase_unwrap(diff, qual, physWave::QUALITY, phaseUnwrap[k]);

            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            getPhase(k);

            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            updatePlot();
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            settingsUi[k]->doWaveButton->setIcon(QIcon(":icons/refresh.png"));
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
    if (sender() && sender()->property("id").isValid()) {
        unsigned int k=sender()->property("id").toUInt();
        getPhase(k);
    } else {
        for (unsigned int k=0;k<numVisars;k++){
            getPhase(k);
        }
    }
    updatePlot();
}

void Visar::getPhase(unsigned int k) {
    if (k < numVisars) {
        settingsUi[k]->plotPhaseIntensity->clearGraphs();
        std::array<nPhysD*,2> imgs={{getPhysFromCombo(settingsUi[k]->refImage),getPhysFromCombo(settingsUi[k]->shotImage)}};

        if (imgs[0] && imgs[1] && imgs[0]->getSize() == imgs[1]->getSize()) {


            QRect geom2=fringeRect[k]->getRect(imgs[0]);
            if (direction(k)!=0) {
                geom2=QRect(QPoint(geom2.top(),geom2.left()),QSize(geom2.height(),geom2.width()));
            }
            cPhase[k].clear();
            cIntensity[0][k].clear();
            cIntensity[1][k].clear();
            cContrast[0][k].clear();
            cContrast[1][k].clear();
            time_phase[k].clear();
            cPhaseErr[k].clear();
            cReflErr[k].clear();

            int refIntShift= settingsUi[k]->intensityShift->value();

            for (int j=geom2.top(); j<geom2.bottom(); j++) {
                time_phase[k]  << j;
                cPhase[k]  << phaseUnwrap[k].point(geom2.center().x(),j,0);

                double meanIntRef=0.0;
                double meanIntShot=0.0;
                double contrastTmpRef=0.0;
                double contrastTmpShot=0.0;
                double meanPhaseTmp=0.0;
                double meanRefle=0.0;
                for (int i=geom2.left(); i<geom2.right();i++) {
                    double intRef=(intensity[k][0].point(i,j-refIntShift,0)-settingsUi[k]->offRef->value())*settingsUi[k]->multRef->value();
                    double intShot=intensity[k][1].point(i,j,0)-settingsUi[k]->offRef->value();

                    meanIntRef+=intRef;
                    meanIntShot+=intShot;

                    meanRefle+= intShot/intRef;

                    contrastTmpRef+=contrast[k][0].point(i,j-refIntShift,0);
                    contrastTmpShot+=contrast[k][1].point(i,j,0);
                    meanPhaseTmp += phaseUnwrap[k].point(i,j);
                }


                meanIntRef/=geom2.width();
                meanIntShot/=geom2.width();
                meanRefle/=geom2.width();

                contrastTmpRef/=geom2.width();
                contrastTmpShot/=geom2.width();
                meanPhaseTmp /= geom2.width();
                cIntensity[0][k] << meanIntRef;
                cIntensity[1][k] << meanIntShot;
                cContrast[0][k]  << contrastTmpRef*settingsUi[k]->multRef->value();
                cContrast[1][k]  << contrastTmpShot;
                double sqrTmpPhase=0.0;
                double stdRefle=0.0;
                for (int i=geom2.left(); i<geom2.right();i++) {
                    sqrTmpPhase += pow(phaseUnwrap[k].point(i,j)-meanPhaseTmp,2);

                    double intRef=(intensity[k][0].point(i,j-refIntShift,0)-settingsUi[k]->offRef->value())*settingsUi[k]->multRef->value();
                    double intShot=intensity[k][1].point(i,j,0)-settingsUi[k]->offRef->value();

                    stdRefle+=pow(intShot/intRef - meanRefle,2);
                }
                cPhaseErr[k] << sqrt(sqrTmpPhase/geom2.width());
                cReflErr[k] << sqrt(stdRefle / geom2.width());

            }

            double buffer,bufferold,dummy=0.0;
            double offsetShift=0;
            if(cPhase[k].size()) {
                if (sweepCoeff[k].size() && sweepCoeff[k].front()>0) {
                    bufferold=cPhase[k].first();
                    for (int j=1;j<cPhase[k].size();j++){
                        buffer=cPhase[k][j];
                        if (fabs(buffer-bufferold)>0.5) dummy+=SIGN(bufferold-buffer);
                        bufferold=buffer;
                        cPhase[k][j]+=dummy;
                    }
                    offsetShift=cPhase[k].first();
                } else {
                    bufferold=cPhase[k].last();
                    for (int j=cPhase[k].size()-2;j>=0;j--){
                        buffer=cPhase[k][j];
                        if (fabs(buffer-bufferold)>0.5) dummy+=SIGN(bufferold-buffer);
                        bufferold=buffer;
                        cPhase[k][j]+=dummy;
                    }
                    offsetShift=cPhase[k].last();
                }
            }

            QPolygonF myLine;
            for (int i=0;i<cPhase[k].size();i++){
                double posx=geom2.x()+geom2.width()/2.0+cPhase[k][i]*settingsUi[k]->interfringe->value();
                double posy=time_phase[k][i];
                if (direction(k)==0) {		//fringes are vertical
                    myLine << QPointF(posx,posy);
                } else {
                    myLine << QPointF(posy,posx);
                }
            }
            fringeLine[k]->setPoints(myLine);

            QCPGraph* graph;
            QPen pen;
            if (settingsUi[k]->interfringe->value() != 0.0) {
                graph = settingsUi[k]->plotPhaseIntensity->addGraph(settingsUi[k]->plotPhaseIntensity->xAxis, settingsUi[k]->plotPhaseIntensity->yAxis);
                graph->setName("Phase Visar "+QLocale().toString(k+1));
                pen.setColor(settingsUi[k]->plotPhaseIntensity->yAxis->labelColor());
                graph->setPen(pen);
                graph->setData(time_phase[k],cPhase[k]);
            }

            velocityUi[k]->offset->setTitle("Offset "+QLocale().toString(offsetShift));
            for (int j=0;j<cPhase[k].size();j++){
                if (direction(k)!=0) {		//fringes are vertical
                    cPhase[k][j] = offsetShift-cPhase[k][j];
                } else {
                    cPhase[k][j] = cPhase[k][j]-offsetShift;
                }
            }

            for (unsigned int m=0;m<2;m++) {

                pen.setStyle((m==1?Qt::SolidLine : Qt::DashLine));


                graph = settingsUi[k]->plotPhaseIntensity->addGraph(settingsUi[k]->plotPhaseIntensity->xAxis, settingsUi[k]->plotPhaseIntensity->yAxis2);
                graph->setName("Intensity Visar "+QLocale().toString(k+1) + " " + (m==0?"ref":"shot"));
                pen.setColor(settingsUi[k]->plotPhaseIntensity->yAxis2->labelColor());
                graph->setPen(pen);
                graph->setData(time_phase[k],cIntensity[m][k]);

                graph = settingsUi[k]->plotPhaseIntensity->addGraph(settingsUi[k]->plotPhaseIntensity->xAxis, settingsUi[k]->plotPhaseIntensity->yAxis3);
                graph->setName("Contrast Visar "+QLocale().toString(k+1) + " " + (m==0?"ref":"shot"));
                pen.setColor(settingsUi[k]->plotPhaseIntensity->yAxis3->labelColor());
                graph->setPen(pen);
                graph->setData(time_phase[k],cContrast[m][k]);

            }
            settingsUi[k]->plotPhaseIntensity->rescaleAxes();
            settingsUi[k]->plotPhaseIntensity->replot();
        }
    }
}

void
Visar::export_txt_multiple() {
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save VISARs and SOP"),property("NeuSave-fileTxt").toString(),tr("Text files (*.txt *.csv);;Any files (*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileTxt", fnametmp);
        QFile t(fnametmp);
        t.open(QIODevice::WriteOnly| QIODevice::Text);
        QTextStream out(&t);
        for (unsigned int k=0;k<numVisars;k++){
            out << export_one(k);
            out << "\n\n";
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
            title=tr("VISAR")+QString(" ")+QLocale().toString(tabPhase->currentIndex()+1);
            break;
        case 1:
            title=tr("VISAR")+QString(" ")+QLocale().toString(tabVelocity->currentIndex()+1);
            break;
        case 2:
            title=tr("SOP");
            break;
    }
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save ")+title,property("NeuSave-fileTxt").toString(),tr("Text files (*.txt *.csv);;Any files (*)"));
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
                for (unsigned int k=0;k<numVisars;k++){
                    txt += export_one(k)+"\n\n";
                }
                clipboard->setText(txt);
                statusbar->showMessage(tr("Points copied to clipboard ")+QLocale().toString(numVisars)+" visars");
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
    out += QString("#SOP Origin       : %L1\n").arg(sopOrigin->value());
    out += QString("#SOP Offset       : %L1\n").arg(sopTimeOffset->value());
    out += QString("#Center & magnif. : %L1 %L2\n").arg(physOriginSpace->value()).arg(magnification->value());
    out += QString("#SOP Time scale   : %L1\n").arg(sopScale->text());
    out += QString("#SOP Direction    : %L1\n").arg(sopDirection->currentIndex()==0 ? "Vertical" : "Horizontal");
    out += QString("#Reflectivity     : %L1\n").arg(whichRefl->currentText());
    out += QString("#Calib            : %L1 %L2\n").arg(sopCalibT0->value()).arg(sopCalibA->value());
    out += QString("#Time\tCounts\tTblackbody\tTgrayIn\tTgrayOut\n");

    for (int i=0;i<time_sop.size();i++) {
        out += QLocale().toString(time_sop[i])+ " ";
        for (int j=0;j<4;j++) {
            double val=sopCurve[j][i];
            out+=(val>=0?"+":"-")+QLocale().toString(fabs(val),'E',4)+ " ";
        }
        out += "\n";
    }
    return out;
}

QString Visar::export_one(unsigned int k) {
    QString out;
    if (k<numVisars) {
        if (velocityUi[k]->enableVisar->isChecked()) {
            out += "#VISAR " + QLocale().toString(k+1) + "\n";
            out += QString("#Offset shift       : %L1\n").arg(velocityUi[k]->offsetShift->value());
            out += QString("#Sensitivity        : %L1\n").arg(settingsUi[k]->sensitivity->value());
            out += QString("#Reflectivity       : %L1 %L2\n").arg(velocityUi[k]->reflOffset->value()).arg(velocityUi[k]->reflRef->value());
            out += QString("#Sweep Time         : %L1\n").arg(settingsUi[k]->physScale->text());
            out += QString("#Time zero & delay  : %L1 %L2\n").arg(velocityUi[k]->physOrigin->value()).arg(velocityUi[k]->offsetTime->value());
            out += QString("#Center & magnif.   : %L1 %L2\n").arg(settingsUi[k]->physOriginSpace->value()).arg(settingsUi[k]->magnification->value());
            out += QString("#Jumps              : %L1\n").arg(velocityUi[k]->jumpst->text());
            out += QString("# Time       Velocity    ErrVel      Reflect     ErrRefl     Quality     Pixel       Shift       ErrShift    RefInt      ShotInt     RefContr    ShotContr.\n");
            for (int j=0;j<time_phase[k].size();j++) {
                QVector<double> values {time_vel[k][j],velocity[k][j], velError[k][j],
                            reflectivity[k][j], reflError[k][j], quality[k][j],
                            time_phase[k][j],
                            cPhase[k][j],cPhaseErr[k][j],
                            cIntensity[0][k][j], cIntensity[1][k][j],
                            cContrast[0][k][j],cContrast[1][k][j]};
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
            settingsUi[tabPhase->currentIndex()]->plotPhaseIntensity->copy_image();
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
