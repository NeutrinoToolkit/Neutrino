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
#include "neutrino.h"

#include "ui_Visar2.h"
#include "ui_Visar3.h"


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
    xAxis->setLabel(tr("Position [time]"));
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

    graph = addGraph(xAxis, yAxis2);
    graph->setName("SOP 3");
    graph->setPen(QPen(yAxis2->labelColor(),0.7));

};


Visar::Visar(neutrino *nparent)
    : nGenericPan(nparent),
      numVisars(0)
{
    setupUi(this);

    //!START SOP stuff
    sopRect =  new nRect(this,1);
    sopRect->changeToolTip(tr("SOP region"));
    sopRect->setRect(QRectF(0,0,100,100));
    connect(actionRect3, SIGNAL(triggered()), sopRect, SLOT(togglePadella()));

    sopPlot->xAxis->setLabel(tr("Time"));
    sopPlot->yAxis->setLabel(tr("Counts"));
    sopPlot->yAxis2->setLabel(tr("Temperature"));


    QApplication::processEvents();

    show();
    QApplication::processEvents();

    for (int l=2+numVisars; l<whichRefl->count();l++) {
        whichRefl->removeItem(l);
    }
    qDebug() << property("NeuSave-numVisars");
    if (property("NeuSave-numVisars").isValid()) {
        int kMax=property("NeuSave-numVisars").toInt();
        for (int k=0; k<kMax; k++) {
            addVisar();
        }
    } else{
        addVisar();
        addVisar();
    }
    connect(actionAddVisar, SIGNAL(triggered()), this, SLOT(addVisar()));
    connect(actionDelVisar, SIGNAL(triggered()), this, SLOT(delVisar()));

    connect(actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
    connect(actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));

    connect(actionSaveTxt, SIGNAL(triggered()), this, SLOT(export_txt()));
    connect(actionSaveTxtMultiple, SIGNAL(triggered()), this, SLOT(export_txt_multiple()));
    connect(actionCopy, SIGNAL(triggered()), this, SLOT(export_clipboard()));

    connect(actionGuess, SIGNAL(triggered()), this, SLOT(getCarrier()));
    connect(actionRefresh, SIGNAL(triggered()), this, SLOT(doWave()));

    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(bufferChanged(nPhysD*)));

    //!END SOP stuff
    QApplication::processEvents();

    loadDefaults();
    sweepChanged();

}

void Visar::addVisar() {
    disconnections();
    QWidget *tab1 = new QWidget();
    QGridLayout *gridLayout1 = new QGridLayout(tab1);
    gridLayout1->setContentsMargins(0, 0, 0, 0);
    QWidget*wvisar1 = new QWidget(tab1);
    wvisar1->setObjectName(QStringLiteral("wvisar1"));
    gridLayout1->addWidget(wvisar1, 0, 0, 1, 1);
    tabPhase->addTab(tab1, "Visar"+QString::number(numVisars+1));

    Ui::Visar2* ui_vel=new Ui::Visar2();
    ui_vel->setupUi(wvisar1);

    decorate(tab1);

    velocityUi.push_back(ui_vel);

    QWidget *tab2 = new QWidget();
    QGridLayout *gridLayout2 = new QGridLayout(tab2);
    gridLayout2->setContentsMargins(0, 0, 0, 0);
    QWidget*wvisar2 = new QWidget(tab2);
    wvisar2->setObjectName(QStringLiteral("wvisar1"));
    gridLayout2->addWidget(wvisar2, 0, 0, 1, 1);
    tabVelocity->addTab(tab2, "Visar"+QString::number(numVisars+1));

    Ui::Visar3* ui_set=new Ui::Visar3();
    ui_set->setupUi(wvisar2);

    decorate(tab2);

    phaseUi.push_back(ui_set);

    //hack to save diffrent uis!!!
    foreach (QWidget *obj, wvisar1->findChildren<QWidget*>()+wvisar2->findChildren<QWidget*>()) {
        obj->setObjectName(obj->objectName()+"-VISAR"+QString::number(numVisars+1));
        obj->setProperty("id", numVisars);
    }

    phase.push_back({nPhysD(),nPhysD()});
    contrast.push_back({nPhysD(),nPhysD()});
    intensity.push_back({nPhysD(),nPhysD()});

    for (int m=0;m<2;m++){
        QString name="Visar "+QString::number(numVisars+1)+" "+QString::number(m);
        phase[numVisars][m].setName(name.toUtf8().constData());
        phase[numVisars][m].setShortName("phase");
        contrast[numVisars][m].setName(name.toUtf8().constData());
        contrast[numVisars][m].setShortName("contrast");
        intensity[numVisars][m].setName(name.toUtf8().constData());
        intensity[numVisars][m].setShortName("intensity");
    }

    QApplication::processEvents();

    sweepCoeff.push_back(std::vector<double>());

    QAction *actionRect = new QAction(QIcon(":icons/rect.png"), "Region Visar"+QString::number(numVisars+1),this);
    actionRect->setProperty("id",numVisars);

    toolBar->insertAction(actionDelVisar,actionRect);

    whichRefl->addItem("Visar"+QString::number(numVisars+1));

    nLine* my_nline=new nLine(this,3);
    my_nline->changeToolTip(tr("Fringeshift Visar ")+QString::number(numVisars+1));
    fringeLine.push_back(my_nline);

    nRect *my_rect=new nRect(this,1);
    my_rect->setRect(QRectF(0,0,100,100));
    my_rect->setProperty("id", numVisars);
    my_rect->changeToolTip(tr("Visar region ")+QString::number(numVisars+1));
    connect(actionRect, SIGNAL(triggered()),my_rect, SLOT(togglePadella()));
    fringeRect.push_back(my_rect);


    velocity.push_back(QVector<double>());
    velError.push_back(QVector<double>());
    reflectivity.push_back(QVector<double>());
    quality.push_back(QVector<double>());
    time_vel.push_back(QVector<double>());
    time_phase.push_back(QVector<double>());

    for (unsigned int m=0; m<2;m++) {
        cPhase[m].push_back(QVector<double>());
        cIntensity[m].push_back(QVector<double>());
        cContrast[m].push_back(QVector<double>());
    }

    QCPGraph* graph;
    QPen pen;

    graph = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis3);
    graph->setName("Quality Visar "+QString::number(numVisars+1));
    graph->setProperty("id",numVisars);
    pen.setColor(plotVelocity->yAxis3->labelColor());
    graph->setPen(pen);

    graph = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis2);
    graph->setName("Reflectivity Visar "+QString::number(numVisars+1));
    graph->setProperty("id",numVisars);
    pen.setColor(plotVelocity->yAxis2->labelColor());
    graph->setPen(pen);

    graph = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis);
    graph->setName("Velocity Visar "+QString::number(numVisars+1));
    graph->setProperty("id",numVisars);
    pen.setColor(plotVelocity->yAxis->labelColor());
    graph->setPen(pen);


    QCPErrorBars *errorBars = new QCPErrorBars(plotVelocity->xAxis, plotVelocity->yAxis);
    errorBars->setDataPlottable(graph);
    errorBars->setProperty("id",numVisars);
    QColor my_color=plotVelocity->yAxis->labelColor();
    my_color.setAlpha(128);
    pen.setColor(my_color);
    errorBars->setPen(pen);

    numVisars++;

    setProperty("NeuSave-numVisars",numVisars);
    connections();

}

void Visar::delVisar() {
    saveDefaults();
    disconnections();
    if (numVisars>0) {
        tabPhase->removeTab(numVisars-1);
        tabVelocity->removeTab(numVisars-1);
        foreach (QAction *action, toolBar->actions()) {
            if (action->property("id").isValid() && action->property("id").toInt()==(int)numVisars-1) {
                action->deleteLater();
            }
        }
        fringeLine.back()->deleteLater();
        fringeLine.pop_back();
        fringeRect.back()->deleteLater();
        fringeRect.pop_back();

        velocityUi.pop_back();
        phaseUi.pop_back();

        phase.pop_back();
        contrast.pop_back();
        intensity.pop_back();

        sweepCoeff.pop_back();

        whichRefl->removeItem(numVisars+1);

        velocity.pop_back();
        velError.pop_back();
        reflectivity.pop_back();
        quality.pop_back();
        time_vel.pop_back();
        time_phase.pop_back();

        for (unsigned int m=0; m<2;m++) {
            cPhase[m].pop_back();
            cIntensity[m].pop_back();
            cContrast[m].pop_back();
        }
        for (int k=0; k< plotVelocity->graphCount() ; k++) {
            if (plotVelocity->graph(k)->property("id").toInt() == k ){
                plotVelocity->removeGraph(k);
            }
        }

        QApplication::processEvents();
        numVisars--;
        setProperty("NeuSave-numVisars",numVisars);
    }
    connections();
}

void Visar::loadSettings(QString my_settings) {
    disconnections();
    QSettings settings(my_settings,QSettings::IniFormat);
    for (unsigned int k=0;k<numVisars;k++) {
        delVisar();
    }
    while (settings.contains("interfringe-VISAR"+QString::number(numVisars+1))) {
        addVisar();
    }
    loadSettings(&settings);
    if (property("NeuSave-whichRefl").isValid()) {
        whichRefl->setCurrentIndex(property("NeuSave-whichRefl").toInt());
    }
    connections();
    sweepChanged();
    doWave();
}

void Visar::mouseAtPlot(QMouseEvent* e) {
    if (sender()) {
        nCustomPlotMouseX3Y *plot=qobject_cast<nCustomPlotMouseX3Y *>(sender());
        if(plot) {
            QString msg;
            QTextStream(&msg) << plot->xAxis->pixelToCoord(e->pos().x()) << ","
                              << plot->yAxis->pixelToCoord(e->pos().y()) << " "
                              << plot->yAxis2->pixelToCoord(e->pos().y()) << ":"
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
    int k=0;
    if(line==nullptr) {
        if (sender()) {
            QLineEdit *line=qobject_cast<QLineEdit *>(sender());
            sweepChanged(line);
        } else {
            for (unsigned int n=0; n< numVisars; n++) {
                sweepChanged(phaseUi[n]->physScale);
            }
            sweepChanged(sopScale);
        }
    } else {
        std::vector<double> *vecsweep=nullptr;
        if (line==sopScale) {
            vecsweep = &sweepCoeffSOP;
        } else {
            k=line->property("id").toInt();
            vecsweep = &sweepCoeff[k];
        }

        if (vecsweep) {
            vecsweep->clear();
            foreach(QString str, line->text().split(" ", QString::SkipEmptyParts)) {
                bool ok;
                double coeff=QLocale().toDouble(str,&ok);
                if(ok) {
                    vecsweep->push_back(coeff);
                } else {
                    statusbar->showMessage("Cannot understant sweep coefficint "+str);
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

void Visar::mouseAtMatrix(QPointF p) {
    int k=0;
    double position=0.0;
    if (tabs->currentIndex()==0) {
        k=tabPhase->currentIndex();
        if (k > 0 && k<(int)numVisars) {
            position=(direction(k)==0 ? p.y() : p.x());
            velocityUi[k]->plotPhaseIntensity->setMousePosition(position);
        }
    } else if (tabs->currentIndex()==1) {
        k=tabVelocity->currentIndex();
        if (k > 0 && k<(int)numVisars) {
            double pos=direction(k)==0 ? p.y() : p.x();
            position=getTime(sweepCoeff[k],pos) - getTime(sweepCoeff[k],phaseUi[k]->physOrigin->value()) + phaseUi[k]->offsetTime->value();
            plotVelocity->setMousePosition(position);
        }
    } else {
        double pos=sopDirection->currentIndex()==0 ? p.y() : p.x();
        position=getTime(sweepCoeffSOP,pos) - getTime(sweepCoeffSOP,sopOrigin->value()) + sopTimeOffset->value();
        sopPlot->setMousePosition(position);
    }
    statusbar->showMessage("Postion : "+QString::number(position));
}

int Visar::direction(int k) {
    int dir=((int) ((velocityUi[k]->angle->value()+360+45)/90.0) )%2;
    return dir;
}

void Visar::bufferChanged(nPhysD*phys) {
    for (unsigned int k=0;k<numVisars;k++){
        fringeRect[k]->hide();
        fringeLine[k]->hide();
        if (phys==getPhysFromCombo(velocityUi[k]->shotImage) || phys==getPhysFromCombo(velocityUi[k]->refImage) ) {
            fringeRect[k]->show();
            fringeLine[k]->show();
        }
        sopRect->hide();
        if (phys==getPhysFromCombo(sopShot) || phys==getPhysFromCombo(sopRef) ) {
            sopRect->show();
        }
    }
}

void Visar::tabChanged(int k) {
    DEBUG("here")
    QTabWidget *tabWidget=nullptr;

    if (sender()) tabWidget=qobject_cast<QTabWidget *>(sender());
    DEBUG("here")

    disconnections();
    DEBUG("here")
    if (tabWidget) {
        DEBUG("here")

        if (tabWidget==tabs) {
            DEBUG("here")
            if (k==0) {
                DEBUG("here")
                tabWidget=tabPhase;
            } else if (k==1) {
                DEBUG("here")
                tabWidget=tabVelocity;
            }
        } else if (tabWidget == tabVelocity) {
            DEBUG("here")
            updatePlot();
        } else if (tabWidget == tabPhase) {
            DEBUG("here")
            getPhase(k);
        }

        DEBUG("here")
        if (tabWidget==tabs) {
            DEBUG("here")
            nparent->showPhys(getPhysFromCombo(sopShot));
        } else {
            DEBUG("here")
            if (k<(int)numVisars) {
                DEBUG("here")
                unsigned int visnum=tabWidget->currentIndex();
                if (visnum<numVisars) {
                    DEBUG("here")
                    nparent->showPhys(getPhysFromCombo(velocityUi[visnum]->shotImage));
                }
            }
        }
    }
    DEBUG("here")
    connections();
    DEBUG("here")
}

void Visar::connections() {
    for (unsigned int k=0;k<numVisars;k++){
        connect(fringeRect[k], SIGNAL(sceneChanged()), this, SLOT(getPhase()));
        connect(phaseUi[k]->offsetShift, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        connect(phaseUi[k]->sensitivity, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        connect(phaseUi[k]->jumpst, SIGNAL(editingFinished()), this, SLOT(updatePlot()));
        connect(phaseUi[k]->reflRef, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        connect(phaseUi[k]->reflOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        connect(phaseUi[k]->jump, SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));

        connect(phaseUi[k]->physOrigin, SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));
        connect(phaseUi[k]->offsetTime, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));

        connect(velocityUi[k]->guess, SIGNAL(released()), this, SLOT(getCarrier()));
        connect(velocityUi[k]->doWaveButton, SIGNAL(released()), this, SLOT(doWave()));

        connect(velocityUi[k]->multRef, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        connect(velocityUi[k]->offRef, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        connect(velocityUi[k]->offShot, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        connect(velocityUi[k]->intensityShift, SIGNAL(editingFinished()), this, SLOT(getPhase()));

        connect(velocityUi[k]->enableVisar, SIGNAL(released()), this, SLOT(updatePlot()));

        connect(phaseUi[k]->physScale,SIGNAL(editingFinished()), this, SLOT(sweepChanged()));

        connect(velocityUi[k]->plotPhaseIntensity,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    }
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
    connect(enableSOP, SIGNAL(released()), this, SLOT(updatePlotSOP()));

    connect(plotVelocity,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    connect(sopPlot,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    connect(sopScale,SIGNAL(editingFinished()), this, SLOT(sweepChanged()));

}

void Visar::disconnections() {
    for (unsigned int k=0;k<numVisars;k++){
        disconnect(fringeRect[k], SIGNAL(sceneChanged()), this, SLOT(getPhase()));
        disconnect(phaseUi[k]->offsetShift, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        disconnect(phaseUi[k]->sensitivity, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        disconnect(phaseUi[k]->jumpst, SIGNAL(editingFinished()), this, SLOT(updatePlot()));
        disconnect(phaseUi[k]->reflRef, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        disconnect(phaseUi[k]->reflOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        disconnect(phaseUi[k]->jump, SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));

        disconnect(phaseUi[k]->physOrigin, SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));
        disconnect(phaseUi[k]->offsetTime, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));

        disconnect(velocityUi[k]->guess, SIGNAL(released()), this, SLOT(getCarrier()));
        disconnect(velocityUi[k]->doWaveButton, SIGNAL(released()), this, SLOT(doWave()));

        disconnect(velocityUi[k]->multRef, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        disconnect(velocityUi[k]->offRef, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        disconnect(velocityUi[k]->offShot, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        disconnect(velocityUi[k]->intensityShift, SIGNAL(editingFinished()), this, SLOT(getPhase()));

        disconnect(velocityUi[k]->enableVisar, SIGNAL(released()), this, SLOT(updatePlot()));

        disconnect(phaseUi[k]->physScale,SIGNAL(editingFinished()), this, SLOT(sweepChanged()));

        disconnect(velocityUi[k]->plotPhaseIntensity,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    }
    disconnect(tabs, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));
    disconnect(tabPhase, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));
    disconnect(tabVelocity, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));

    disconnect(sopRect, SIGNAL(sceneChanged()), this, SLOT(updatePlotSOP()));
    disconnect(sopRef, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    disconnect(sopShot, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    disconnect(sopTimeOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    disconnect(sopOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    disconnect(sopOrigin, SIGNAL(valueChanged(int)), this, SLOT(updatePlotSOP()));
    disconnect(sopDirection, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    disconnect(sopCalibT0, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    disconnect(sopCalibA, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    disconnect(whichRefl, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    disconnect(enableSOP, SIGNAL(released()), this, SLOT(updatePlotSOP()));

    disconnect(plotVelocity,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    disconnect(sopPlot,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    disconnect(sopScale,SIGNAL(editingFinished()), this, SLOT(sweepChanged()));
}

void Visar::updatePlotSOP() {
    if (!enableSOP->isChecked()) return;
    disconnections();
    nPhysD *shot=getPhysFromCombo(sopShot);
    nPhysD *ref=getPhysFromCombo(sopRef);
    int dir=sopDirection->currentIndex();
    if (shot) {
        QRect geom2=sopRect->getRect(shot);
        if (ref) geom2=geom2.intersected(QRect(0,0,ref->getW(),ref->getH()));
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
            case 0:
                break;
            case 1:
                for (unsigned int i=0; i<numVisars; i++) {
                    reflList << i;
                }
                break;
            default:
                reflList << whichRefl->currentIndex()-2;
                break;
        }

        sopCurve[1].resize(time_sop.size());
        sopCurve[2].resize(time_sop.size());
        sopCurve[3].resize(time_sop.size());

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
                sopCurve[3][i]=my_velocity;
            } else {
                sopCurve[1][i]=0.0;
                sopCurve[2][i]=my_temp;
                sopCurve[3][i]=0.0;
            }
        }
        sopPlot->graph(1)->setData(time_sop,sopCurve[1]);

        sopPlot->graph(2)->setData(time_sop,sopCurve[2]);

        sopPlot->graph(3)->setData(time_sop,sopCurve[3]);

        sopPlot->rescaleAxes();
        sopPlot->replot();
    }
    connections();
}

void Visar::updatePlot() {
    statusbar->showMessage("Updating");
    disconnections();

    plotVelocity->clearItems();

    for (int g=0; g<plotVelocity->plottableCount(); g++) {
        if (plotVelocity->plottable(g)->property("JumpGraph").toBool()) {
            plotVelocity->removePlottable(plotVelocity->plottable(g));
        }
    }

    for (unsigned int k=0;k<numVisars;k++){
        if (cPhase[0][k].size()==cPhase[1][k].size() && cPhase[0][k].size()==time_phase[k].size()){

            if (velocityUi[k]->enableVisar->isChecked()) {

                Qt::PenStyle pstyle=((int)k==tabVelocity->currentIndex()?Qt::SolidLine : Qt::DashLine);

                double sensitivity=phaseUi[k]->sensitivity->value();
                double deltat=phaseUi[k]->offsetTime->value()-getTime(sweepCoeff[k],phaseUi[k]->physOrigin->value());

                QVector<double> tjump,njump,rjump;
                QStringList jumpt=phaseUi[k]->jumpst->text().split(";", QString::SkipEmptyParts);
                foreach (QString piece, jumpt) {
                    QStringList my_jumps=piece.split(QRegExp("\\s+"), QString::SkipEmptyParts);
                    if (my_jumps.size()>1 && my_jumps.size()<=3) {
                        if (my_jumps.size()>1 && my_jumps.size()<=3) {
                            bool ok1, ok2, ok3=true;
                            double valdt=QLocale().toDouble(my_jumps.at(0),&ok1);
                            double valdn=QLocale().toDouble(my_jumps.at(1),&ok2);
                            double valdrefr_index=1.0;
                            if (my_jumps.size()==3) {
                                valdrefr_index=QLocale().toDouble(my_jumps.at(2),&ok3);
                            }
                            if (sensitivity<0) valdn*=-1.0;
                            if (ok1 && ok2) {
                                tjump << valdt;
                                njump << valdn;
                            } else {
                                statusbar->showMessage(tr("Skipped unreadable jump '")+piece+QString("' VISAR ")+QString::number(k+1),5000);
                            }
                            if (ok3) {
                                rjump << valdrefr_index;
                            } else {
                                rjump << 1.0;
                            }
                        }
                    } else {
                        statusbar->showMessage(tr("Skipped unreadable jump '")+piece+QString("' VISAR ")+QString::number(k+1),5000);
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

                double offset=phaseUi[k]->offsetShift->value();

                QVector<QVector< double > > velJump_array(abs(phaseUi[k]->jump->value()));
                for (int i=0;i<abs(phaseUi[k]->jump->value());i++) {
                    velJump_array[i].resize(time_phase[k].size());
                }

                time_vel[k].resize(time_phase[k].size());
                velocity[k].resize(time_phase[k].size());
                velError[k].resize(time_phase[k].size());
                reflectivity[k].resize(time_phase[k].size());
                quality[k].resize(time_phase[k].size());

                for (int j=0;j<time_phase[k].size();j++) {
                    time_vel[k][j] = getTime(sweepCoeff[k],time_phase[k][j])+deltat;

                    double fRef=cPhase[0][k][j];
                    double fShot=cPhase[1][k][j];
                    double iRef=cIntensity[0][k][j];
                    double iShot=cIntensity[1][k][j];
                    if (getPhysFromCombo(velocityUi[k]->shotImage)==getPhysFromCombo(velocityUi[k]->refImage)) {
                        fRef=0.0;
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

                    double speed=(offset+fShot-fRef+njumps)*sensitivity/refr_index;
                    //double refle=setvisar[k].reflRef->value()*iShot/iRef+setvisar[k].reflOffset->value();
                    double Rg=phaseUi[k]->reflOffset->value();
                    double Rmat=phaseUi[k]->reflRef->value();
                    double beta=-Rg/pow(1.0-Rg,2);
                    double refle=iShot/iRef * (Rmat-beta) + beta;

                    velocity[k][j] = speed;
                    reflectivity[k][j] = refle;
                    quality[k][j]= refle/(cContrast[1][k][j]/cContrast[0][k][j]);

                    for (int i=0;i<abs(phaseUi[k]->jump->value());i++) {
                        int jloc=i+1;
                        if (sensitivity<0) jloc*=-1;
                        if (phaseUi[k]->jump->value()<0) jloc*=-1;
                        velJump_array[i][j] = (offset+fShot-fRef+jloc)*sensitivity/refr_index;
                    }
                }

                QPen pen;
                pen.setStyle(pstyle);

                if (phaseUi[k]->jump->value()!=0) {
                    for (int i=0;i<abs(phaseUi[k]->jump->value());i++) {
                        QCPGraph* graph = plotVelocity->addGraph(plotVelocity->xAxis, plotVelocity->yAxis);
                        graph->setProperty("JumpGraph",true);
                        graph->setName("VelJump Visar"+QString::number(k+1) + " #" +QString::number(i));
                        QColor color(plotVelocity->yAxis->labelColor());
                        color.setAlpha(100);
                        pen.setColor(color);
                        graph->setPen(pen);
                        graph->setData(time_vel[k],velJump_array[i]);
                    }
                }


                for (int kk=0; kk< plotVelocity->graphCount() ; kk++) {
                    QCPGraph *my_graph=plotVelocity->graph(kk);
                    if (my_graph->property("id").toInt() == (int)k ){
                        pen=my_graph->pen();
                        pen.setStyle(pstyle);
                        my_graph->setPen(pen);
                        if (my_graph->valueAxis()==plotVelocity->yAxis) {
                            my_graph->setData(time_vel[k],velocity[k]);
                        } else if (my_graph->valueAxis()==plotVelocity->yAxis2) {
                            my_graph->setData(time_vel[k],reflectivity[k]);
                        } else if (my_graph->valueAxis()==plotVelocity->yAxis3) {
                            my_graph->setData(time_vel[k],quality[k]);
                        }
                    }
                }
                for (int kk=0; kk< plotVelocity->plottableCount() ; kk++) {
                    QCPErrorBars *my_err = qobject_cast<QCPErrorBars*>(plotVelocity->plottable(kk));
                    if (my_err && my_err->property("id").toInt() == (int)k ){
                            pen=my_err->pen();
                            pen.setStyle(pstyle);
                            my_err->setPen(pen);
                            my_err->setData(time_vel[k],velError[k]);
                    }
                }

            }
        }
    }
    plotVelocity->rescaleAxes();
    plotVelocity->replot();


    statusbar->showMessage("");

    updatePlotSOP();

    connections();
}

void Visar::getCarrier() {
    if (sender() && sender()->property("id").isValid()) {
        int k=sender()->property("id").toInt();
        getCarrier(k);
    } else {
        for (unsigned int k=0;k<numVisars;k++){
            getCarrier(k);
        }
    }
    if (tabs->currentIndex()==1) {
        statusbar->showMessage(tr("Carrier (")+QString::number(velocityUi[0]->interfringe->value())+tr("px, ")+velocityUi[0]->angle->value()+tr("deg) - (")+QString::number(velocityUi[1]->interfringe->value())+tr("px, ")+velocityUi[1]->angle->value()+tr("deg)"));
    }
}

void Visar::getCarrier(int k) {
    disconnections();
    QComboBox *combo=NULL;
    if (velocityUi[k]->carrierPhys->currentIndex()==0) {
        combo=velocityUi[k]->refImage;
    } else {
        combo=velocityUi[k]->shotImage;
    }

    nPhysD *phys=getPhysFromCombo(combo);
    if (phys && fringeRect[k]) {
        QRect geom2=fringeRect[k]->getRect(phys);
        nPhysD datamatrix = phys->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());

        vec2f vecCarr=phys_guess_carrier(datamatrix, velocityUi[k]->guessWeight->value());

        if (vecCarr.first()==0) {
            statusbar->showMessage("ERROR: Problem finding the carrier try to change the weight", 5000);
        } else {
            velocityUi[k]->interfringe->setValue(vecCarr.first());
            velocityUi[k]->angle->setValue(vecCarr.second());
            if (tabPhase->currentIndex()==k) {
                statusbar->showMessage(tr("Carrier :")+QString::number(vecCarr.first())+tr("px, ")+QString::number(vecCarr.second())+tr("deg"));
            }
        }
    }
    connections();
}

void Visar::doWave() {
    if (sender() && sender()->property("id").isValid()) {
        int k=sender()->property("id").toInt();
        doWave(k);
    } else {
        for (unsigned int k=0;k<numVisars;k++){
            doWave(k);
        }
    }
}


void Visar::doWave(int k) {
    disconnections();
    if (velocityUi[k]->enableVisar->isChecked()){
        std::array<nPhysD*,2> imgs={{getPhysFromCombo(velocityUi[k]->refImage),getPhysFromCombo(velocityUi[k]->shotImage)}};
        if (imgs[0] && imgs[1]  && imgs[0]->getSize() == imgs[1]->getSize()) {


            QProgressDialog progress("Filter visar "+QString::number(k+1), "Cancel", 0, property("NeuSave-VisarCounter").toInt(), this);
            progress.setCancelButton(0);
            progress.setWindowModality(Qt::WindowModal);
            progress.setValue(0);
            progress.show();
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            sweepChanged(phaseUi[k]->physScale);

            std::array<nPhysC,2> physfft={{imgs[0]->ft2(PHYS_FORWARD),imgs[1]->ft2(PHYS_FORWARD)}};
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            vec2 dim(imgs[0]->getSize());

            std::array<nPhysC,2> zz_morlet;
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            for (int m=0;m<2;m++) {
                phase[k][m].resize(dim.x(), dim.y());
                contrast[k][m].resize(dim.x(), dim.y());
                intensity[k][m]= imgs[m]->copy();
                progress.setValue(progress.value()+1);
                qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
                phys_fast_gaussian_blur(intensity[k][m], velocityUi[k]->resolution->value());
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
            double cr = cos((velocityUi[k]->angle->value()) * _phys_deg);
            double sr = sin((velocityUi[k]->angle->value()) * _phys_deg);
            double thick_norm=velocityUi[k]->resolution->value()*M_PI/sqrt(pow(sr*dim.x(),2)+pow(cr*dim.y(),2));
            double damp_norm=M_PI;

            double lambda_norm=velocityUi[k]->interfringe->value()/sqrt(pow(cr*dim.x(),2)+pow(sr*dim.y(),2));
#pragma omp parallel for collapse(2)
            for (size_t x=0;x<(size_t)dim.x();x++) {
                for (size_t y=0;y<(size_t)dim.y();y++) {
                    double xr = xx[x]*cr - yy[y]*sr; //rotate
                    double yr = xx[x]*sr + yy[y]*cr;

                    double e_x = -pow(damp_norm*(xr*lambda_norm-1.0), 2);
                    double e_y = -pow(yr*thick_norm, 2);

                    double gauss = exp(e_x)*exp(e_y)-exp(-pow(damp_norm, 2));

                    for (unsigned int m=0;m<2;m++) {
                        zz_morlet[m].Timg_matrix[y][x]=physfft[m].Timg_matrix[y][x]*gauss;
                    }
                }
            }

            progress.setValue(progress.value()+1);
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
                    phase[k][m].Timg_buffer[kk] = -physfft[m].Timg_buffer[kk].arg()/(2*M_PI);
                    contrast[k][m].Timg_buffer[kk] = 2.0*physfft[m].Timg_buffer[kk].mod()/(dim.x()*dim.y());
                    intensity[k][m].Timg_buffer[kk] -= contrast[k][m].point(kk)*cos(2*M_PI*phase[k][m].point(kk));
                }
            }
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            getPhase(k);

            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            updatePlot();
            progress.setValue(progress.value()+1);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            setProperty("NeuSave-VisarCounter",progress.value()+1);
        } else {
            statusBar()->showMessage("size mismatch",5000);
        }
    }
    connections();
}

void Visar::getPhase() {
    if (sender() && sender()->property("id").isValid()) {
        int k=sender()->property("id").toInt();
        getPhase(k);
    } else {
        for (unsigned int k=0;k<numVisars;k++){
            getPhase(k);
        }
    }
    updatePlot();
}

void Visar::getPhase(int k) {
    disconnections();

    if (k< (int) numVisars && velocityUi[k]->enableVisar->isChecked()) {
        velocityUi[k]->plotPhaseIntensity->clearGraphs();
        std::array<nPhysD*,2> imgs={{getPhysFromCombo(velocityUi[k]->refImage),getPhysFromCombo(velocityUi[k]->shotImage)}};

        if (imgs[0] && imgs[1] && imgs[0]->getSize() == imgs[1]->getSize()) {
            for (unsigned int m=0;m<cPhase.size();m++) {
                QRect geom2=fringeRect[k]->getRect(imgs[m]);
                time_phase[k].clear();
                cPhase[m][k].clear();
                cIntensity[m][k].clear();
                cContrast[m][k].clear();
                int intensityShift=velocityUi[k]->intensityShift->value();
                double offsetIntensity=(m==0?velocityUi[k]->offRef->value():velocityUi[k]->offShot->value());
                if (direction(k)==0) { //fringes are vertical
                    for (int j=geom2.top(); j<geom2.bottom();j++) {
                        time_phase[k]  << j;
                        cPhase[m][k]  << phase[k][m].point(geom2.center().x(),j,0);

                        double intensityTmp=0.0;
                        double contrastTmp=0.0;
                        for (int i=geom2.left(); i<geom2.right();i++) {
                            if (m==0) { //reference
                                intensityTmp+=intensity[k][m].point(i,j-intensityShift,0);
                                contrastTmp+=contrast[k][m].point(i,j-intensityShift,0);
                            } else { //shot
                                intensityTmp+=intensity[k][m].point(i,j,0);
                                contrastTmp+=contrast[k][m].point(i,j,0);
                            }

                        }
                        cIntensity[m][k] << (intensityTmp/geom2.width()-offsetIntensity)*((m==0)?velocityUi[k]->multRef->value():1.0);
                        cContrast[m][k]  << contrastTmp/geom2.height()*((m==0)?velocityUi[k]->multRef->value():1.0);
                    }
                } else { //fringes are horizontal
                    for (int j=geom2.left(); j<geom2.right();j++) {
                        time_phase[k]  << j;
                        cPhase[m][k]  << phase[k][m].point(j,geom2.center().y(),0);

                        double intensityTmp=0.0;
                        double contrastTmp=0.0;
                        for (int i=geom2.top(); i<geom2.bottom();i++) {
                            if (m==0) { //reference
                                intensityTmp+=intensity[k][m].point(j-intensityShift,i,0);
                                contrastTmp+=contrast[k][m].point(j-intensityShift,i,0);
                            } else { //shot
                                intensityTmp+=intensity[k][m].point(j,i,0);
                                contrastTmp+=contrast[k][m].point(j,i,0);
                            }
                        }
                        cIntensity[m][k] << (intensityTmp/geom2.height()-offsetIntensity)*((m==0)?velocityUi[k]->multRef->value():1.0);
                        cContrast[m][k]  << contrastTmp/geom2.height()*((m==0)?velocityUi[k]->multRef->value():1.0);
                    }
                }

                double buffer,bufferold,dummy=0.0;
                double offsetShift=0;
                if(cPhase[m][k].size()) {
                    if (sweepCoeff[k].size() && sweepCoeff[k].front()>0) {
                        bufferold=cPhase[m][k].first();
                        for (int j=1;j<cPhase[m][k].size();j++){
                            buffer=cPhase[m][k][j];
                            if (fabs(buffer-bufferold)>0.5) dummy+=SIGN(bufferold-buffer);
                            bufferold=buffer;
                            cPhase[m][k][j]+=dummy;
                        }
                        offsetShift=cPhase[m][k].first();
                    } else {
                        bufferold=cPhase[m][k].last();
                        for (int j=cPhase[m][k].size()-2;j>=0;j--){
                            buffer=cPhase[m][k][j];
                            if (fabs(buffer-bufferold)>0.5) dummy+=SIGN(bufferold-buffer);
                            bufferold=buffer;
                            cPhase[m][k][j]+=dummy;
                        }
                        offsetShift=cPhase[m][k].last();
                    }
                }
                phaseUi[k]->offset->setTitle("Offset "+QString::number(offsetShift));
                for (int j=0;j<cPhase[m][k].size();j++){
                    cPhase[m][k][j] -= offsetShift;
                }

                QPolygonF myLine;
                for (int i=0;i<cPhase[m][k].size();i++){
                    if (direction(k)==0) {		//fringes are vertical
                        myLine << QPointF(geom2.x()+geom2.width()/2.0+cPhase[m][k][i]*velocityUi[k]->interfringe->value(),time_phase[k][i]);
                    } else {
                        myLine << QPointF(time_phase[k][i],geom2.y()+geom2.height()/2.0-cPhase[m][k][i]*velocityUi[k]->interfringe->value());
                    }
                }
                fringeLine[k]->setPoints(myLine);

                QCPGraph* graph;
                QPen pen;
                pen.setStyle((m==1?Qt::SolidLine : Qt::DashLine));

                graph = velocityUi[k]->plotPhaseIntensity->addGraph(velocityUi[k]->plotPhaseIntensity->xAxis, velocityUi[k]->plotPhaseIntensity->yAxis);
                graph->setName("Phase Visar "+QString::number(k+1) + " " + (m==0?"ref":"shot"));
                pen.setColor(velocityUi[k]->plotPhaseIntensity->yAxis->labelColor());
                graph->setPen(pen);
                graph->setData(time_phase[k],cPhase[m][k]);

                graph = velocityUi[k]->plotPhaseIntensity->addGraph(velocityUi[k]->plotPhaseIntensity->xAxis, velocityUi[k]->plotPhaseIntensity->yAxis2);
                graph->setName("Intensity Visar "+QString::number(k+1) + " " + (m==0?"ref":"shot"));
                pen.setColor(velocityUi[k]->plotPhaseIntensity->yAxis2->labelColor());
                graph->setPen(pen);
                graph->setData(time_phase[k],cIntensity[m][k]);

                graph = velocityUi[k]->plotPhaseIntensity->addGraph(velocityUi[k]->plotPhaseIntensity->xAxis, velocityUi[k]->plotPhaseIntensity->yAxis3);
                graph->setName("Contrast Visar "+QString::number(k+1) + " " + (m==0?"ref":"shot"));
                pen.setColor(velocityUi[k]->plotPhaseIntensity->yAxis3->labelColor());
                graph->setPen(pen);
                graph->setData(time_phase[k],cContrast[m][k]);
            }
            velocityUi[k]->plotPhaseIntensity->rescaleAxes();
            velocityUi[k]->plotPhaseIntensity->replot();
        }
    }
    connections();
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
            out << endl << endl;
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
            title=tr("VISAR")+QString(" ")+QString::number(tabPhase->currentIndex()+1);
            break;
        case 1:
            title=tr("VISAR")+QString(" ")+QString::number(tabVelocity->currentIndex()+1);
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
                statusbar->showMessage(tr("Points copied to clipboard ")+QString::number(numVisars)+" visars");
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
            out += "#VISAR " + QString::number(k+1) + "\n";
            out += QString("#Offset shift       : %L1\n").arg(phaseUi[k]->offsetShift->value());
            out += QString("#Sensitivity        : %L1\n").arg(phaseUi[k]->sensitivity->value());
            out += QString("#Reflectivity       : %L1 %L2\n").arg(phaseUi[k]->reflOffset->value()).arg(phaseUi[k]->reflRef->value());
            out += QString("#Sweep Time         : %L1\n").arg(phaseUi[k]->physScale->text());
            out += QString("#Time zero & delay  : %L1 %L2\n").arg(phaseUi[k]->physOrigin->value()).arg(phaseUi[k]->offsetTime->value());
            out += QString("#Jumps              : %L1\n").arg(phaseUi[k]->jumpst->text());
            out += QString("# Time       Velocity    Reflect.    Quality     Pixel       RefShift    ShotShift   RefInt      ShotInt     RefContr.           ShotContr.\n");
            for (int j=0;j<time_phase[k].size();j++) {
                QVector<double> values {time_vel[k][j],velocity[k][j],
                            reflectivity[k][j],quality[k][j],
                            time_phase[k][j],
                            cPhase[0][k][j],cPhase[1][k][j],
                            cIntensity[0][k][j],cIntensity[1][k][j],
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
