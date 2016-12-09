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

    QPen pen;
    for (unsigned int k=0; k<2; k++) {
        QCPGraph* graph;
        graph = addGraph(xAxis, yAxis3);
        graph->setName("Quality Visar "+QString::number(k+1));
        pen.setColor(yAxis3->labelColor());
        graph->setPen(pen);

        graph = addGraph(xAxis, yAxis2);
        graph->setName("Reflectivity Visar "+QString::number(k+1));
        pen.setColor(yAxis2->labelColor());
        graph->setPen(pen);

        graph = addGraph(xAxis, yAxis);
        graph->setName("Velocity Visar "+QString::number(k+1));
        pen.setColor(yAxis->labelColor());
        graph->setPen(pen);
    }
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



Visar::~Visar() {
}

Visar::Visar(neutrino *nparent, QString winname)
    : nGenericPan(nparent, winname),
      sweepCoeff{{std::vector<double>(),std::vector<double>(),std::vector<double>()}}
{
    my_w.setupUi(this);

    connect(my_w.tabs, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));
    connect(my_w.tabPhase, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));
    connect(my_w.tabVelocity, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));

    connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
    connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));

    connect(my_w.actionSaveTxt, SIGNAL(triggered()), this, SLOT(export_txt()));
    connect(my_w.actionSaveTxtMultiple, SIGNAL(triggered()), this, SLOT(export_txt_multiple()));
    connect(my_w.actionCopy, SIGNAL(triggered()), this, SLOT(export_clipboard()));

    connect(my_w.actionGuess, SIGNAL(triggered()), this, SLOT(getCarrier()));
    connect(my_w.actionRefresh, SIGNAL(triggered()), this, SLOT(doWave()));

    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(bufferChanged(nPhysD*)));
    
    QList<QAction*> actionRects;
    actionRects << my_w.actionRect1 << my_w.actionRect2;
    for (int k=0;k<2;k++){
        fringeLine[k] = new nLine(nparent);
        fringeLine[k]->setParentPan(panName,3);
        fringeLine[k]->changeToolTip(tr("Fringeshift Visar ")+QString::number(k+1));

        fringeRect[k] =  new nRect(nparent);
        fringeRect[k]->setProperty("id", k);

        fringeRect[k]->setParentPan(panName,1);
        fringeRect[k]->changeToolTip(tr("Visar region ")+QString::number(k+1));
        fringeRect[k]->setRect(QRectF(0,0,100,100));
        connect(actionRects[k], SIGNAL(triggered()), fringeRect[k], SLOT(togglePadella()));
    }

    //!START SOP stuff
    sopRect =  new nRect(nparent);
    sopRect->setParentPan(panName,1);
    sopRect->changeToolTip(tr("SOP region"));
    sopRect->setRect(QRectF(0,0,100,100));
    connect(my_w.actionRect3, SIGNAL(triggered()), sopRect, SLOT(togglePadella()));



    my_w.sopPlot->xAxis->setLabel(tr("Time"));
    my_w.sopPlot->yAxis->setLabel(tr("Counts"));
    my_w.sopPlot->yAxis2->setLabel(tr("Temperature"));

    //!END SOP stuff
    
    QList<QWidget*> father1{my_w.wvisar1, my_w.wvisar2};
    QList<QWidget*> father2{my_w.setVisar1, my_w.setVisar2};
    for (int k=0;k<2;k++){
        visar[k].setupUi(father1.at(k));
        father1.at(k)->show();
        setvisar[k].setupUi(father2.at(k));
        father2.at(k)->show();

        //hack to save diffrent uis!!!
        foreach (QWidget *obj, father1.at(k)->findChildren<QWidget*>()+father2.at(k)->findChildren<QWidget*>()) {
            obj->setObjectName(obj->objectName()+"-VISAR"+QString::number(k+1));
            obj->setProperty("id", k);
        }

    }
    my_w.sopScale->setProperty("id", 2);

    for (int k=0;k<2;k++){
        for (int m=0;m<2;m++){
            QString name="Visar "+QString::number(k+1)+" "+QString::number(m);
            phase[k][m].setName(name.toUtf8().constData());
            phase[k][m].setShortName("phase");
            contrast[k][m].setName(name.toUtf8().constData());
            contrast[k][m].setShortName("contrast");
            intensity[k][m].setName(name.toUtf8().constData());
            intensity[k][m].setShortName("intensity");
        }
    }
    QApplication::processEvents();

    connections();
    tabChanged();
    sweepChanged();
    show();
}

void Visar::loadSettings(QString my_settings) {
    disconnections();
    nGenericPan::loadSettings(my_settings);
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

            my_w.statusbar->showMessage(msg);
        }
    }
}

void Visar::sweepChanged(QLineEdit *line) {
    DEBUG("here");
    if(line==nullptr) {
        if (sender()) {
            DEBUG("here");
            QLineEdit *line=qobject_cast<QLineEdit *>(sender());
            sweepChanged(line);
        } else {
            DEBUG("here");
            sweepChanged(setvisar[0].physScale);
            sweepChanged(setvisar[1].physScale);
            sweepChanged(my_w.sopScale);
        }
    } else {
        if (line->property("id").isValid()) {
            int k=line->property("id").toInt();
            DEBUG("here " << k);
            bool ok;
            sweepCoeff[k].clear();
            foreach(QString str, line->text().split(" ", QString::SkipEmptyParts)) {
                double coeff=QLocale().toDouble(str,&ok);
                if(ok) {
                    sweepCoeff[k].push_back(coeff);
                } else {
                    my_w.statusbar->showMessage("Cannot understant sweep coefficint "+str);
                    break;
                }
            }
            if (k<2) {
                getPhase(k);
                updatePlot();
            } else {
                updatePlotSOP();
            }
        }
    }
}

double Visar::getTime(int k, double p) {
    double time=0;
    for (unsigned int i=0;i<sweepCoeff[k].size();i++) {
        time+=sweepCoeff[k][i]/(i+1.0)*pow(p,i+1);
    }
    return time;
}

void Visar::mouseAtMatrix(QPointF p) {
    int k=0;
    double position=0.0;
    if (my_w.tabs->currentIndex()==0) {
        k=my_w.tabPhase->currentIndex();
        position=(direction(k)==0 ? p.y() : p.x());
        visar[k].plotPhaseIntensity->setMousePosition(position);
    } else if (my_w.tabs->currentIndex()==1) {
        k=my_w.tabVelocity->currentIndex();
        double pos=direction(k)==0 ? p.y() : p.x();
        position=getTime(k,pos) - getTime(k,setvisar[k].physOrigin->value()) + setvisar[k].offsetTime->value();
        my_w.plotVelocity->setMousePosition(position);
    } else {
        double pos=my_w.sopDirection->currentIndex()==0 ? p.y() : p.x();
        position=getTime(2,pos) - getTime(2,my_w.sopOrigin->value()) + my_w.sopTimeOffset->value();
        my_w.sopPlot->setMousePosition(position);
    }
    my_w.statusbar->showMessage("Postion : "+QString::number(position));
}

int Visar::direction(int k) {
    int dir=((int) ((visar[k].angle->value()+360+45)/90.0) )%2;
    return dir;
}

void Visar::bufferChanged(nPhysD*phys) {
    for (int k=0;k<2;k++){
        fringeRect[k]->hide();
        fringeLine[k]->hide();
        if (phys==getPhysFromCombo(visar[k].shotImage) ||
                phys==getPhysFromCombo(visar[k].refImage) ) {
            fringeRect[k]->show();
            fringeLine[k]->show();
        }
        sopRect->hide();
        if (phys==getPhysFromCombo(my_w.sopShot) ||
                phys==getPhysFromCombo(my_w.sopRef) ) {
            sopRect->show();
        }
    }
}

void Visar::tabChanged(int k) {
    QTabWidget *tabWidget=nullptr;

    if (sender()) tabWidget=qobject_cast<QTabWidget *>(sender());

    if (!tabWidget) tabWidget=my_w.tabs;

    if (tabWidget==my_w.tabs) {
        if (k==0) {
            tabWidget=my_w.tabPhase;
        } else if (k==1) {
            tabWidget=my_w.tabVelocity;
        }
    }
    
    if (k<2) {
        int visnum=tabWidget->currentIndex();
        nparent->showPhys(getPhysFromCombo(visar[visnum].shotImage));
        if (tabWidget==my_w.tabVelocity) {
            updatePlot();
        }
    } else if (k==2){
        nparent->showPhys(getPhysFromCombo(my_w.sopShot));
        updatePlotSOP();
    }
}

void Visar::connections() {
    for (int k=0;k<2;k++){
        connect(fringeRect[k], SIGNAL(sceneChanged()), this, SLOT(getPhase()));
        connect(setvisar[k].offsetShift, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        connect(setvisar[k].sensitivity, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        connect(setvisar[k].jumpst, SIGNAL(editingFinished()), this, SLOT(updatePlot()));
        connect(setvisar[k].reflRef, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        connect(setvisar[k].reflOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        connect(setvisar[k].jump, SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));

        connect(setvisar[k].physOrigin, SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));
        connect(setvisar[k].offsetTime, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));

        connect(visar[k].guess, SIGNAL(released()), this, SLOT(getCarrier()));
        connect(visar[k].doWaveButton, SIGNAL(released()), this, SLOT(doWave()));

        connect(visar[k].multRef, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        connect(visar[k].offRef, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        connect(visar[k].offShot, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        connect(visar[k].intensityShift, SIGNAL(editingFinished()), this, SLOT(getPhase()));

        connect(visar[k].enableVisar, SIGNAL(released()), this, SLOT(updatePlot()));

        connect(setvisar[k].physScale,SIGNAL(editingFinished()), this, SLOT(sweepChanged()));

        connect(visar[k].plotPhaseIntensity,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    }
    connect(sopRect, SIGNAL(sceneChanged()), this, SLOT(updatePlotSOP()));
    connect(my_w.sopRef, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    connect(my_w.sopShot, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    connect(my_w.sopTimeOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    connect(my_w.sopOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    connect(my_w.sopOrigin, SIGNAL(valueChanged(int)), this, SLOT(updatePlotSOP()));
    connect(my_w.sopDirection, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    connect(my_w.sopCalibT0, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    connect(my_w.sopCalibA, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    connect(my_w.whichRefl, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    connect(my_w.enableSOP, SIGNAL(released()), this, SLOT(updatePlotSOP()));

    connect(my_w.plotVelocity,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    connect(my_w.sopPlot,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    connect(my_w.sopScale,SIGNAL(editingFinished()), this, SLOT(sweepChanged()));

}

void Visar::disconnections() {
    for (int k=0;k<2;k++){
        disconnect(fringeRect[k], SIGNAL(sceneChanged()), this, SLOT(getPhase()));
        disconnect(setvisar[k].offsetShift, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        disconnect(setvisar[k].sensitivity, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        disconnect(setvisar[k].jumpst, SIGNAL(editingFinished()), this, SLOT(updatePlot()));
        disconnect(setvisar[k].reflRef, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        disconnect(setvisar[k].reflOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));
        disconnect(setvisar[k].jump, SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));

        disconnect(setvisar[k].physOrigin, SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));
        disconnect(setvisar[k].offsetTime, SIGNAL(valueChanged(double)), this, SLOT(updatePlot()));

        disconnect(visar[k].guess, SIGNAL(released()), this, SLOT(getCarrier()));
        disconnect(visar[k].doWaveButton, SIGNAL(released()), this, SLOT(doWave()));

        disconnect(visar[k].multRef, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        disconnect(visar[k].offRef, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        disconnect(visar[k].offShot, SIGNAL(editingFinished()), this, SLOT(getPhase()));
        disconnect(visar[k].intensityShift, SIGNAL(editingFinished()), this, SLOT(getPhase()));

        disconnect(visar[k].enableVisar, SIGNAL(released()), this, SLOT(updatePlot()));

        disconnect(setvisar[k].physScale,SIGNAL(editingFinished()), this, SLOT(sweepChanged()));

        disconnect(visar[k].plotPhaseIntensity,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    }
    disconnect(sopRect, SIGNAL(sceneChanged()), this, SLOT(updatePlotSOP()));
    disconnect(my_w.sopRef, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    disconnect(my_w.sopShot, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    disconnect(my_w.sopTimeOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    disconnect(my_w.sopOffset, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    disconnect(my_w.sopOrigin, SIGNAL(valueChanged(int)), this, SLOT(updatePlotSOP()));
    disconnect(my_w.sopDirection, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    disconnect(my_w.sopCalibT0, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    disconnect(my_w.sopCalibA, SIGNAL(valueChanged(double)), this, SLOT(updatePlotSOP()));
    disconnect(my_w.whichRefl, SIGNAL(currentIndexChanged(int)), this, SLOT(updatePlotSOP()));
    disconnect(my_w.enableSOP, SIGNAL(released()), this, SLOT(updatePlotSOP()));

    disconnect(my_w.plotVelocity,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    disconnect(my_w.sopPlot,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    disconnect(my_w.sopScale,SIGNAL(editingFinished()), this, SLOT(sweepChanged()));
}

void Visar::updatePlotSOP() {
    if (!my_w.enableSOP->isChecked()) return;
    disconnections();
    nPhysD *shot=getPhysFromCombo(my_w.sopShot);
    nPhysD *ref=getPhysFromCombo(my_w.sopRef);
    int dir=my_w.sopDirection->currentIndex();
    if (shot) {
        QRect geom2=sopRect->getRect(shot);
        if (ref) geom2=geom2.intersected(QRect(0,0,ref->getW(),ref->getH()));
        if (geom2.isEmpty()) {
            my_w.statusbar->showMessage(tr("Attention: the region is outside the image!"),2000);
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
                time_sop[j]=getTime(2,geom2.y()+j)-getTime(2,my_w.sopOrigin->value()) + my_w.sopTimeOffset->value();
                sopCurve[0][j]=sopData[j]/dx-my_w.sopOffset->value();
            }
            break;
        case 1:
            for (int i=0;i<dx;i++) {
                time_sop[i]=getTime(2,geom2.x()+i)-getTime(2,my_w.sopOrigin->value()) + my_w.sopTimeOffset->value();
                sopCurve[0][i]=sopData[i]/dy-my_w.sopOffset->value();
            }
            break;
        default:
            break;
        }
        my_w.sopPlot->graph(0)->setData(time_sop,sopCurve[0]);

        // TEMPERATURE FROM REFLECTIVITY
        QVector<int> reflList;
        switch (my_w.whichRefl->currentIndex()) {
        case 0:
            reflList << 0;
            break;
        case 1:
            reflList << 1;
            break;
        case 2:
            reflList << 0 << 1;
            break;
        default:
            break;
        }
        
        sopCurve[1].resize(time_sop.size());
        sopCurve[2].resize(time_sop.size());
        sopCurve[3].resize(time_sop.size());

        double my_T0=my_w.sopCalibT0->value();
        double my_A=my_w.sopCalibA->value();
        
        for (int i=0; i<time_sop.size(); i++) {
            double my_reflectivity=0;
            double my_velocity=0;
            
            int numrefl=0;
            for (int numk=0;numk<reflList.size();numk++) {
                int k=reflList[numk];

                if (reflectivity[k].size()>0) {
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
        my_w.sopPlot->graph(1)->setData(time_sop,sopCurve[1]);

        my_w.sopPlot->graph(2)->setData(time_sop,sopCurve[2]);

        my_w.sopPlot->graph(3)->setData(time_sop,sopCurve[3]);

        my_w.sopPlot->rescaleAxes();
        my_w.sopPlot->replot();
    }
    connections();
}

void Visar::updatePlot() {
    my_w.statusbar->showMessage("Updating");
    disconnections();

    my_w.plotVelocity->clearItems();

    for (int g=0; g<my_w.plotVelocity->plottableCount(); g++) {
        if (my_w.plotVelocity->plottable(g)->property("JumpGraph").toBool()) {
            my_w.plotVelocity->removePlottable(my_w.plotVelocity->plottable(g));
        }
    }

    for (int k=0;k<2;k++){
        if (cPhase[0][k].size()){

            if (visar[k].enableVisar->isChecked()) {

                double sensitivity=setvisar[k].sensitivity->value();
                double deltat=setvisar[k].offsetTime->value()-getTime(k,setvisar[k].physOrigin->value());

                QVector<double> tjump,njump,rjump;
                QStringList jumpt=setvisar[k].jumpst->text().split(";", QString::SkipEmptyParts);
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
                                my_w.statusbar->showMessage(tr("Skipped unreadable jump '")+piece+QString("' VISAR ")+QString::number(k+1),5000);
                            }
                            if (ok3) {
                                rjump << valdrefr_index;
                            } else {
                                rjump << 1.0;
                            }
                        }
                    } else {
                        my_w.statusbar->showMessage(tr("Skipped unreadable jump '")+piece+QString("' VISAR ")+QString::number(k+1),5000);
                    }
                }

                foreach (double a, tjump) {
                    QCPItemStraightLine* my_jumpLine=new QCPItemStraightLine(my_w.plotVelocity);
                    QPen pen(Qt::gray);
                    pen.setStyle((k==my_w.tabVelocity->currentIndex()?Qt::SolidLine : Qt::DashLine));
                    my_jumpLine->setPen(pen);
                    my_jumpLine->point1->setCoords(a,0);
                    my_jumpLine->point2->setCoords(a,1);
                }

                double offset=setvisar[k].offsetShift->value();

                QVector<QVector< double > > velJump_array(abs(setvisar[k].jump->value()));
                for (int i=0;i<abs(setvisar[k].jump->value());i++) {
                    velJump_array[i].resize(time_phase[k].size());
                }

                time_vel[k].resize(time_phase[k].size());
                velocity[k].resize(time_phase[k].size());
                reflectivity[k].resize(time_phase[k].size());
                quality[k].resize(time_phase[k].size());

                for (int j=0;j<time_phase[k].size();j++) {
                    time_vel[k][j] = getTime(k,time_phase[k][j])+deltat;

                    double fRef=cPhase[0][k][j];
                    double fShot=cPhase[1][k][j];
                    double iRef=cIntensity[0][k][j];
                    double iShot=cIntensity[1][k][j];
                    if (getPhysFromCombo(visar[k].shotImage)==getPhysFromCombo(visar[k].refImage)) {
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
                    double Rg=setvisar[k].reflOffset->value();
                    double Rmat=setvisar[k].reflRef->value();
                    double beta=-Rg/pow(1.0-Rg,2);
                    double refle=iShot/iRef * (Rmat-beta) + beta;

                    velocity[k][j] = speed;
                    reflectivity[k][j] = refle;
                    quality[k][j]= cContrast[1][k][j]/cContrast[0][k][j];

                    for (int i=0;i<abs(setvisar[k].jump->value());i++) {
                        int jloc=i+1;
                        if (sensitivity<0) jloc*=-1;
                        if (setvisar[k].jump->value()<0) jloc*=-1;
                        velJump_array[i][j] = (offset+fShot-fRef+jloc)*sensitivity/refr_index;
                    }
                }

                QPen pen;
                pen.setStyle((k==my_w.tabVelocity->currentIndex()?Qt::SolidLine : Qt::DashLine));

                if (setvisar[k].jump->value()!=0) {
                    for (int i=0;i<abs(setvisar[k].jump->value());i++) {
                        QCPGraph* graph = my_w.plotVelocity->addGraph(my_w.plotVelocity->xAxis, my_w.plotVelocity->yAxis);
                        graph->setProperty("JumpGraph",true);
                        graph->setName("VelJump Visar"+QString::number(k+1) + " #" +QString::number(i));
                        QColor color(my_w.plotVelocity->yAxis->labelColor());
                        color.setAlpha(100);
                        pen.setColor(color);
                        graph->setPen(pen);

                        graph->setData(time_vel[k],velJump_array[i]);
                    }
                }

                pen=my_w.plotVelocity->graph(3*k+0)->pen();
                pen.setStyle((k==my_w.tabVelocity->currentIndex()?Qt::SolidLine : Qt::DashLine));
                my_w.plotVelocity->graph(3*k+0)->setPen(pen);
                my_w.plotVelocity->graph(3*k+0)->setData(time_vel[k],quality[k]);

                pen=my_w.plotVelocity->graph(3*k+1)->pen();
                pen.setStyle((k==my_w.tabVelocity->currentIndex()?Qt::SolidLine : Qt::DashLine));
                my_w.plotVelocity->graph(3*k+1)->setPen(pen);
                my_w.plotVelocity->graph(3*k+1)->setData(time_vel[k],reflectivity[k]);

                pen=my_w.plotVelocity->graph(3*k+2)->pen();
                pen.setStyle((k==my_w.tabVelocity->currentIndex()?Qt::SolidLine : Qt::DashLine));
                my_w.plotVelocity->graph(3*k+2)->setPen(pen);
                my_w.plotVelocity->graph(3*k+2)->setData(time_vel[k],velocity[k]);

            }
        }
    }
    my_w.plotVelocity->rescaleAxes();
    my_w.plotVelocity->replot();


    my_w.statusbar->showMessage("");

    updatePlotSOP();

    connections();
}

void Visar::getCarrier() {
    if (sender() && sender()->property("id").isValid()) {
        int k=sender()->property("id").toInt();
        getCarrier(k);
    } else {
        for (int k=0;k<2;k++){
            getCarrier(k);
        }
    }
    if (my_w.tabs->currentIndex()==1) {
        my_w.statusbar->showMessage(tr("Carrier (")+QString::number(visar[0].interfringe->value())+tr("px, ")+visar[0].angle->value()+tr("deg) - (")+QString::number(visar[1].interfringe->value())+tr("px, ")+visar[1].angle->value()+tr("deg)"));
    }
}

void Visar::getCarrier(int k) {
    disconnections();
    QComboBox *combo=NULL;
    if (visar[k].carrierPhys->currentIndex()==0) {
        combo=visar[k].refImage;
    } else {
        combo=visar[k].shotImage;
    }

    nPhysD *phys=getPhysFromCombo(combo);
    if (phys && fringeRect[k]) {
        QRect geom2=fringeRect[k]->getRect(phys);
        nPhysD datamatrix = phys->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());

        vec2f vecCarr=phys_guess_carrier(datamatrix, visar[k].guessWeight->value());

        if (vecCarr.first()==0) {
            my_w.statusbar->showMessage("ERROR: Problem finding the carrier try to change the weight", 5000);
        } else {
            visar[k].interfringe->setValue(vecCarr.first());
            visar[k].angle->setValue(vecCarr.second());
            if (my_w.tabPhase->currentIndex()==k) {
                my_w.statusbar->showMessage(tr("Carrier :")+QString::number(vecCarr.first())+tr("px, ")+QString::number(vecCarr.second())+tr("deg"));
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
        for (int k=0;k<2;k++){
            doWave(k);
        }
    }
}


void Visar::doWave(int k) {
    disconnections();
    if (visar[k].enableVisar->isChecked()){
        if (getPhysFromCombo(visar[k].refImage) && getPhysFromCombo(visar[k].shotImage )  &&
                getPhysFromCombo(visar[k].refImage)->getW() == getPhysFromCombo(visar[k].shotImage)->getW() &&
                getPhysFromCombo(visar[k].refImage)->getH() == getPhysFromCombo(visar[k].shotImage)->getH()) {
            
            int counter=0;
            QProgressDialog progress("Filter visar "+QString::number(k+1), "Cancel", 0, 12, this);
            progress.setCancelButton(0);
            progress.setWindowModality(Qt::WindowModal);
            progress.show();
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            sweepChanged(setvisar[k].physScale);
            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            nPhysC physfftRef=getPhysFromCombo(visar[k].refImage)->ft2(PHYS_FORWARD);
            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            nPhysC physfftShot=getPhysFromCombo(visar[k].shotImage)->ft2(PHYS_FORWARD);
            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            size_t dx=physfftRef.getW();
            size_t dy=physfftRef.getH();
            
            
            nPhysImageF<mcomplex> zz_morletRef("Ref"), zz_morletShot("Shot");
            zz_morletRef.resize(dx,dy);
            zz_morletShot.resize(dx,dy);
            
            std::vector<int> xx(dx), yy(dy);
#pragma omp parallel for
            for (size_t i=0;i<dx;i++) xx[i]=(i+(dx+1)/2)%dx-(dx+1)/2; // swap and center
#pragma omp parallel for
            for (size_t i=0;i<dy;i++) yy[i]=(i+(dy+1)/2)%dy-(dy+1)/2;
            
            for (int m=0;m<2;m++) {
                phase[k][m].resize(dx, dy);
                contrast[k][m].resize(dx, dy);
                intensity[k][m].resize(dx, dy);
            }
            
            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
#pragma omp parallel for
            for (size_t kk=0; kk<dx*dy; kk++) {
                intensity[k][0].set(kk,getPhysFromCombo(visar[k].refImage)->point(kk));
                intensity[k][1].set(kk,getPhysFromCombo(visar[k].shotImage)->point(kk));
            }
            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            phys_fast_gaussian_blur(intensity[k][0], visar[k].resolution->value());
            phys_fast_gaussian_blur(intensity[k][1], visar[k].resolution->value());
            
            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            double cr = cos((visar[k].angle->value()) * _phys_deg);
            double sr = sin((visar[k].angle->value()) * _phys_deg);
            double thick_norm=visar[k].resolution->value()*M_PI/sqrt(pow(sr*dx,2)+pow(cr*dy,2));
            double damp_norm=M_PI;
            
            double lambda_norm=visar[k].interfringe->value()/sqrt(pow(cr*dx,2)+pow(sr*dy,2));
#pragma omp parallel for collapse(2)
            for (size_t x=0;x<dx;x++) {
                for (size_t y=0;y<dy;y++) {
                    double xr = xx[x]*cr - yy[y]*sr; //rotate
                    double yr = xx[x]*sr + yy[y]*cr;
                    
                    double e_x = -pow(damp_norm*(xr*lambda_norm-1.0), 2);
                    double e_y = -pow(yr*thick_norm, 2);
                    
                    double gauss = exp(e_x)*exp(e_y)-exp(-pow(damp_norm, 2));
                    
                    zz_morletRef.Timg_matrix[y][x]=physfftRef.Timg_matrix[y][x]*gauss;
                    zz_morletShot.Timg_matrix[y][x]=physfftShot.Timg_matrix[y][x]*gauss;
                    
                }
            }
            
            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            physfftRef = zz_morletRef.ft2(PHYS_BACKWARD);
            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
            physfftShot = zz_morletShot.ft2(PHYS_BACKWARD);
            
            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

#pragma omp parallel for
            for (size_t kk=0; kk<dx*dy; kk++) {
                phase[k][0].Timg_buffer[kk] = -physfftRef.Timg_buffer[kk].arg()/(2*M_PI);
                contrast[k][0].Timg_buffer[kk] = 2.0*physfftRef.Timg_buffer[kk].mod()/(dx*dy);
                intensity[k][0].Timg_buffer[kk] -= contrast[k][0].point(kk)*cos(2*M_PI*phase[k][0].point(kk));
                
                phase[k][1].Timg_buffer[kk] = -physfftShot.Timg_buffer[kk].arg()/(2*M_PI);
                contrast[k][1].Timg_buffer[kk] = 2.0*physfftShot.Timg_buffer[kk].mod()/(dx*dy);
                intensity[k][1].Timg_buffer[kk] -= contrast[k][1].point(kk)*cos(2*M_PI*phase[k][1].point(kk));
            }
            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            getPhase(k);

            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);

            updatePlot();
            progress.setValue(++counter);
            qApp->processEvents(QEventLoop::ExcludeUserInputEvents);
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
        for (int k=0;k<2;k++){
            getPhase(k);
        }
    }
    updatePlot();
}

void Visar::getPhase(int k) {
    disconnections();
    if (visar[k].enableVisar->isChecked()) {
        visar[k].plotPhaseIntensity->clearGraphs();
        QList<nPhysD*> imageList{getPhysFromCombo(visar[k].refImage),getPhysFromCombo(visar[k].shotImage)};
        if (!imageList.contains(nullptr)) {
            for (int m=0;m<2;m++) {
                QRect geom2=fringeRect[k]->getRect(imageList[m]);
                time_phase[k].clear();
                cPhase[m][k].clear();
                cIntensity[m][k].clear();
                cContrast[m][k].clear();
                int intensityShift=visar[k].intensityShift->value();
                double offsetIntensity=(m==0?visar[k].offRef->value():visar[k].offShot->value());
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
                        cIntensity[m][k] << (intensityTmp/geom2.width()-offsetIntensity)*((m==0)?visar[k].multRef->value():1.0);
                        cContrast[m][k]  << contrastTmp/geom2.height()*((m==0)?visar[k].multRef->value():1.0);
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
                        cIntensity[m][k] << (intensityTmp/geom2.height()-offsetIntensity)*((m==0)?visar[k].multRef->value():1.0);
                        cContrast[m][k]  << contrastTmp/geom2.height()*((m==0)?visar[k].multRef->value():1.0);
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
                setvisar[k].offset->setTitle("Offset "+QString::number(offsetShift));
                for (int j=0;j<cPhase[m][k].size();j++){
                    cPhase[m][k][j] -= offsetShift;
                }

                QPolygonF myLine;
                for (int i=0;i<cPhase[m][k].size();i++){
                    if (direction(k)==0) {		//fringes are vertical
                        myLine << QPointF(geom2.x()+geom2.width()/2.0+cPhase[m][k][i]*visar[k].interfringe->value(),time_phase[k][i]);
                    } else {
                        myLine << QPointF(time_phase[k][i],geom2.y()+geom2.height()/2.0-cPhase[m][k][i]*visar[k].interfringe->value());
                    }
                }
                fringeLine[k]->setPoints(myLine);

                QCPGraph* graph;
                QPen pen;
                pen.setStyle((m==my_w.tabPhase->currentIndex()?Qt::SolidLine : Qt::DashLine));

                graph = visar[k].plotPhaseIntensity->addGraph(visar[k].plotPhaseIntensity->xAxis, visar[k].plotPhaseIntensity->yAxis);
                graph->setName("Phase Visar "+QString::number(k+1) + " " + (m==0?"ref":"shot"));
                pen.setColor(visar[k].plotPhaseIntensity->yAxis->labelColor());
                graph->setPen(pen);
                graph->setData(time_phase[k],cPhase[m][k]);

                graph = visar[k].plotPhaseIntensity->addGraph(visar[k].plotPhaseIntensity->xAxis, visar[k].plotPhaseIntensity->yAxis2);
                graph->setName("Intensity Visar "+QString::number(k+1) + " " + (m==0?"ref":"shot"));
                pen.setColor(visar[k].plotPhaseIntensity->yAxis2->labelColor());
                graph->setPen(pen);
                graph->setData(time_phase[k],cIntensity[m][k]);

                graph = visar[k].plotPhaseIntensity->addGraph(visar[k].plotPhaseIntensity->xAxis, visar[k].plotPhaseIntensity->yAxis3);
                graph->setName("Contrast Visar "+QString::number(k+1) + " " + (m==0?"ref":"shot"));
                pen.setColor(visar[k].plotPhaseIntensity->yAxis3->labelColor());
                graph->setPen(pen);
                graph->setData(time_phase[k],cContrast[m][k]);
            }
            visar[k].plotPhaseIntensity->rescaleAxes();
            visar[k].plotPhaseIntensity->replot();
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
        for (int k=0;k<2;k++){
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
    switch (my_w.tabs->currentIndex()) {
    case 0:
        title=tr("VISAR")+QString(" ")+QString::number(my_w.tabPhase->currentIndex()+1);
        break;
    case 1:
        title=tr("VISAR")+QString(" ")+QString::number(my_w.tabVelocity->currentIndex()+1);
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
        switch (my_w.tabs->currentIndex()) {
        case 0:
            out << export_one(my_w.tabPhase->currentIndex());
            break;
        case 1:
            out << export_one(my_w.tabVelocity->currentIndex());
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
    switch (my_w.tabs->currentIndex()) {
    case 0:
        clipboard->setText(export_one(my_w.tabPhase->currentIndex()));
        my_w.statusbar->showMessage(tr("Points copied to clipboard ")+my_w.tabPhase->tabText(my_w.tabPhase->currentIndex()));
        break;
    case 1:
        clipboard->setText(export_one(0)+"\n\n"+export_one(1));
        my_w.statusbar->showMessage(tr("Points copied to clipboard both visars"));
        break;
    case 2:
        clipboard->setText(export_sop());
        my_w.statusbar->showMessage(tr("Points copied to clipboard SOP"));
        break;
    default:
        break;
    }
}

QString Visar::export_sop() {
    QString out;
    out += QString("#SOP Origin       : %L1\n").arg(my_w.sopOrigin->value());
    out += QString("#SOP Offset       : %L1\n").arg(my_w.sopTimeOffset->value());
    out += QString("#SOP Time scale   : %L1\n").arg(my_w.sopScale->text());
    out += QString("#SOP Direction    : %L1\n").arg(my_w.sopDirection->currentIndex()==0 ? "Vertical" : "Horizontal");
    out += QString("#Reflectivity     : %L1\n").arg(my_w.whichRefl->currentText());
    out += QString("#Calib            : %L1 %L2\n").arg(my_w.sopCalibT0->value()).arg(my_w.sopCalibA->value());
    out += QString("#Time\tCounts\tTblackbody\tTgrayIn\tTgrayOut\n");

    for (int i=0;i<time_sop.size();i++) {
        out += QLocale().toString(time_sop[i]);
        for (int j=0;j<4;j++) {
            double val=sopCurve[j][i];
            out+=(val>=0?"+":"-")+QLocale().toString(fabs(val),'E',4)+ " ";
        }
        out += "\n";
    }
    return out;
}

QString Visar::export_one(int k) {
    QString out;
    if (k<2) {
        if (visar[k].enableVisar->isChecked()) {
            out += "#VISAR " + QString::number(k+1) + "\n";
            out += QString("#Offset shift       : %L1\n").arg(setvisar[k].offsetShift->value());
            out += QString("#Sensitivity        : %L1\n").arg(setvisar[k].sensitivity->value());
            out += QString("#Reflectivity       : %L1 %L2\n").arg(setvisar[k].reflOffset->value()).arg(setvisar[k].reflRef->value());
            out += QString("#Sweep Time         : %L1\n").arg(setvisar[k].physScale->text());
            out += QString("#Time zero & delay  : %L1 %L2\n").arg(setvisar[k].physOrigin->value()).arg(setvisar[k].offsetTime->value());
            out += QString("#Jumps              : %L1\n").arg(setvisar[k].jumpst->text());
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