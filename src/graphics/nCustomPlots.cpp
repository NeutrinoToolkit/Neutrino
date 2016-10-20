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

#include "nCustomPlots.h"
#include "ui_nCustomPlot.h"
#include <QtSvg>


nCustomRangeLineEdit::nCustomRangeLineEdit(QCPAxis *axis):
    QWidget(qobject_cast<QWidget *>(axis)),
    my_min(new QLineEdit(qobject_cast<QWidget *>(axis))),
    my_max(new QLineEdit(qobject_cast<QWidget *>(axis))),
    my_axis(axis)
{

    QFont f=my_min->font();
    f.setPointSize(10);
    my_min->setFont(f);
    my_max->setFont(f);
    my_min->setAlignment(Qt::AlignRight);

    QToolButton *my_lock = new QToolButton(this);
    my_lock->setToolTip("Lock axis range");
    my_lock->setCheckable(true);
    my_lock->setChecked(my_axis->rangeLocked());
    my_lock->setIcon(QIcon(my_axis->rangeLocked()?":icons/lockClose":":icons/lockOpen"));

    setLock(my_axis->rangeLocked());
    rangeChanged(my_axis->range());

    connect(my_axis,SIGNAL(rangeChanged(QCPRange)),this,SLOT(rangeChanged(QCPRange)));
    connect(my_min,SIGNAL(textEdited(QString)),this,SLOT(setRange(QString)));
    connect(my_max,SIGNAL(textEdited(QString)),this,SLOT(setRange(QString)));
    connect(my_lock,SIGNAL(toggled(bool)),this,SLOT(setLock(bool)));

    QHBoxLayout* gridLayout = new QHBoxLayout(this);
    gridLayout->setMargin(0);
    gridLayout->addWidget(my_min);
    gridLayout->addWidget(new QLabel(":",this));
    gridLayout->addWidget(my_max);
    gridLayout->addWidget(my_lock);
}

void nCustomRangeLineEdit::setLock(bool check) {
    if (my_axis) {
        my_axis->setRangeLocked(check);
        if (!check) {
            my_axis->rescale(true);
            if (my_axis->parentPlot()) my_axis->parentPlot()->replot();
        }
        my_min->setEnabled(check);
        my_max->setEnabled(check);
        if(sender()) {
            QToolButton *my_lock = qobject_cast<QToolButton *>(sender());
            if (my_lock) {
                my_lock->setIcon(QIcon(check?":icons/lockClose":":icons/lockOpen"));
            }
        }
    }
}

void nCustomRangeLineEdit::rangeChanged(const QCPRange& newrange) {
    my_min->setText(QLocale().toString(newrange.lower));
    my_max->setText(QLocale().toString(newrange.upper));
}

void nCustomRangeLineEdit::setRange(QString minmax_str){
    if (my_axis && sender()) {
        QLineEdit *line = qobject_cast<QLineEdit *>(sender());
        if (line) {
            bool ok;
            double my_min_dbl=QLocale().toDouble(minmax_str,&ok);
            if (ok) {
                if (line==my_min) my_axis->setRangeLower(my_min_dbl);
                if (line==my_max) my_axis->setRangeUpper(my_min_dbl);
                if (my_axis->parentPlot()) my_axis->parentPlot()->replot();
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
nCustomPlot::nCustomPlot(QWidget* parent):
    QCustomPlot(parent),
     title(new QCPTextElement(this))
{

    setProperty("NeuSave-fileIni",objectName()+".ini");
    setProperty("NeuSave-fileTxt",objectName()+".txt");
    setProperty("NeuSave-fileExport",objectName()+".pdf");

    setContentsMargins(0,0,0,0);

    connect(this, SIGNAL(axisDoubleClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)), this, SLOT(myAxisDoubleClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)));

    setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

    axisRect()->setRangeDrag(0);
    axisRect()->setRangeZoom(0);
    axisRect()->setMargins(QMargins(0,0,0,0));
    setCursor(QCursor(Qt::CrossCursor));
    repaint();
}

void nCustomPlot::contextMenuEvent (QContextMenuEvent *ev) {
    QMainWindow *my_pad=nullptr;
    foreach (my_pad, findChildren<QMainWindow *>()) {
        if(my_pad->objectName() == "Preferences") break;
    }
    if (!my_pad) {
        my_pad=new QMainWindow(this,Qt::Tool);
        my_pad->setObjectName("Preferences");
        my_pad->setAttribute(Qt::WA_DeleteOnClose);
        Ui::nCustomPlot my_w;
        my_w.setupUi(my_pad);
        my_pad->setProperty("Preferences",true);
        my_pad->setWindowTitle("Plot "+objectName());
        connect(my_w.actionLoadPref,SIGNAL(triggered()),this,SLOT(loadSettings()));
        connect(my_w.actionSavePref,SIGNAL(triggered()),this,SLOT(saveSettings()));
        connect(my_w.actionChangeFonts, SIGNAL(triggered()), this, SLOT(changeAllFonts()));
        connect(my_w.actionRefresh, SIGNAL(triggered()), this, SLOT(replot()));
        connect(my_w.actionCopy, SIGNAL(triggered()), this, SLOT(copy_data()));
        connect(my_w.actionSave, SIGNAL(triggered()), this, SLOT(save_data()));
        connect(my_w.actionExport, SIGNAL(triggered()), this, SLOT(export_image()));
        if (!title.isNull()) {
            my_w.plotTitle->setText(title->text());
        }
        connect(my_w.plotTitle, SIGNAL(textEdited(QString)), this, SLOT(setTitle(QString)));
        connect(my_w.fontTitle,SIGNAL(released()),this,SLOT(changeTitleFont()));

        foreach (QCPAxis *axis, findChildren<QCPAxis *>()) {
            if (axis->visible()) {
                int row=my_w.labels_layout->rowCount();
                int col=0;

                QCheckBox *cb_axis = new QCheckBox("", this);
                QFont f = cb_axis->font();
                cb_axis->setChecked(true);
                cb_axis->setFont(f);
                cb_axis->setProperty("axis",qVariantFromValue((void *) axis));
                connect(cb_axis,SIGNAL(toggled(bool)),this,SLOT(showAxis(bool)));
                my_w.labels_layout->addWidget(cb_axis,row,col++,Qt::AlignCenter);


                QLineEdit *le = new QLineEdit(this);
                f.setPointSize(10);
                le->setFont(f);
                le->setText(axis->label());
                le->setProperty("axis",qVariantFromValue((void *) axis));
                connect(le,SIGNAL(textChanged(QString)),this,SLOT(setLabel(QString)));
                my_w.labels_layout->addWidget(le,row,col++);


                QCheckBox *cb_grid = new QCheckBox("", this);
                cb_grid->setTristate(true);
                cb_grid->setCheckState(axis->grid()->visible() && axis->grid()->subGridVisible()?Qt::Checked:(axis->grid()->visible()?Qt::PartiallyChecked:Qt::Unchecked));
                cb_grid->setFont(f);
                cb_grid->setProperty("grid",qVariantFromValue((void *) axis->grid()));
                connect(cb_grid,SIGNAL(stateChanged(int)),this,SLOT(showGrid(int)));
                my_w.labels_layout->addWidget(cb_grid,row,col++,Qt::AlignCenter);

                QCheckBox *cb_log = new QCheckBox("", this);
                cb_log->setChecked(axis->scaleType()==QCPAxis::stLogarithmic);
                cb_log->setFont(f);
                cb_log->setProperty("axis",qVariantFromValue((void *) axis));
                connect(cb_log,SIGNAL(toggled(bool)),this,SLOT(setLog(bool)));
                my_w.labels_layout->addWidget(cb_log,row,col++,Qt::AlignCenter);

                QToolButton *tb_color = new QToolButton(this);
                QPixmap px(20, 20);
                px.fill(axis->labelColor());
                tb_color->setIcon(px);
                tb_color->setProperty("axis",qVariantFromValue((void *) axis));
                connect(tb_color,SIGNAL(released()),this,SLOT(setColor()));
                my_w.labels_layout->addWidget(tb_color,row,col++,Qt::AlignCenter);

                QToolButton *tb_font = new QToolButton(this);
                tb_font->setIcon(QIcon(":icons/font"));
                tb_font->setProperty("axis",qVariantFromValue((void *) axis));
                connect(tb_font,SIGNAL(released()),this,SLOT(changeAxisFont()));
                my_w.labels_layout->addWidget(tb_font,row,col++,Qt::AlignCenter);

                nCustomRangeLineEdit *minmax = new nCustomRangeLineEdit(axis);
                minmax->setFont(f);
                my_w.labels_layout->addWidget(minmax,row,col++);
            }
        }
        for (int g=0; g<plottableCount(); g++) {
            QCPGraph *graph = qobject_cast<QCPGraph *>(plottable(g));
            if(graph) {
                int row=my_w.graphs_layout->rowCount();
                int col=0;
                QCheckBox *cb_graph = new QCheckBox("", this)   ;
                QFont f = cb_graph->font();

                cb_graph->setChecked(true);
                cb_graph->setFont(f);
                cb_graph->setProperty("graph",qVariantFromValue((void *) graph));
                connect(cb_graph,SIGNAL(toggled(bool)),this,SLOT(showGraph(bool)));
                my_w.graphs_layout->addWidget(cb_graph,row,col++,Qt::AlignCenter);

                QLabel *le = new QLabel(graph->name(),this);
                f.setPointSize(10);
                le->setFont(f);
                my_w.graphs_layout->addWidget(le,row,col++);


                QDoubleSpinBox *sb_thick= new QDoubleSpinBox(this);
                sb_thick->setFont(f);
                sb_thick->setRange(0,99);
                sb_thick->setDecimals(1);
                sb_thick->setSingleStep(0.1);
                sb_thick->setValue(graph->pen().widthF());
                sb_thick->setProperty("graph",qVariantFromValue((void *) graph));
                connect(sb_thick,SIGNAL(valueChanged(double)),this,SLOT(changeGraphThickness(double)));
                my_w.graphs_layout->addWidget(sb_thick,row,col++,Qt::AlignCenter);

                QToolButton *tb_copy = new QToolButton(this);
                tb_copy->setIcon(QIcon(":icons/saveClipboard"));
                tb_copy->setProperty("graph",qVariantFromValue((void *) graph));
                connect(tb_copy,SIGNAL(released()),this,SLOT(copy_data()));
                my_w.graphs_layout->addWidget(tb_copy,row,col++,Qt::AlignCenter);

                QToolButton *tb_save = new QToolButton(this);
                tb_save->setIcon(QIcon(":icons/saveTxt"));
                tb_save->setProperty("graph",qVariantFromValue((void *) graph));
                connect(tb_save,SIGNAL(released()),this,SLOT(save_data()));
                my_w.graphs_layout->addWidget(tb_save,row,col++,Qt::AlignCenter);

            }
        }

    }
    if (my_pad) {
        my_pad->show();
        my_pad->raise();
    }
}


void nCustomPlot::showGraph(bool val) {
    if (sender() && sender()->property("graph").isValid()) {
        QCPGraph *graph = (QCPGraph *) sender()->property("graph").value<void *>();
        if(hasPlottable(graph)) {
             graph->setVisible(val);
             replot();
        }
    }
}

void nCustomPlot::changeGraphThickness(double val) {
    if (sender() && sender()->property("graph").isValid()) {
        QCPGraph *graph = (QCPGraph *) sender()->property("graph").value<void *>();
        if(hasPlottable(graph)) {
            QPen p=graph->pen();
            p.setWidthF(val);
            graph->setPen(p);
            replot();
        }
    }
}


void nCustomPlot::showAxis(bool val) {
    if (sender() && sender()->property("axis").isValid()) {
        QCPAxis *axis = (QCPAxis *) sender()->property("axis").value<void *>();
        if (axis) {
            axis->setVisible(val);
            replot();
        }
    }
}


void nCustomPlot::get_data_graph(QTextStream &out, QCPGraph *graph) {
    out << "# " << graph->name() << endl;
    for (QCPGraphDataContainer::const_iterator it=graph->data()->begin(); it!=graph->data()->end(); ++it) {
        out << QLocale().toString(it->key) << " " << QLocale().toString(it->value) << endl;
    }
}

void nCustomPlot::get_data(QTextStream &out, QObject *obj) {
    if (obj) {
        QCPGraph *graph = qobject_cast<QCPGraph *>(obj);
        if (graph) {
            get_data_graph(out,graph);
        } else {
            if (obj->property("graph").isValid()) {
                graph = (QCPGraph *) sender()->property("graph").value<void *>();
            }
            if (graph) {
                get_data_graph(out, graph);
            } else {
                get_data(out);
            }
        }
    } else {
        out << "# " <<  objectName() << " (" << graphCount() << " graphs)" << endl;
        for (int g=0; g<graphCount(); g++) {
            if(graph(g)->visible()) {
                out << "# " << g <<" ";
                get_data_graph(out,graph(g));
                out << endl << endl;
            }
        }
    }
}

void nCustomPlot::save_data(){
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save data in text"),property("NeuSave-fileTxt").toString(),tr("Text files (*.txt *.csv);;Any files (*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileTxt", fnametmp);
        QFile t(fnametmp);
        t.open(QIODevice::WriteOnly| QIODevice::Text);
        QTextStream out(&t);
        get_data(out, sender());
        t.close();
    }
}

void nCustomPlot::copy_data(){
    QString t;
    QTextStream out(&t);
    get_data(out,sender());
    QApplication::clipboard()->setText(t);
}

void nCustomPlot::export_image(){
    QString fnametmp = QFileDialog::getSaveFileName(this,tr("Save Drawing"),property("NeuSave-fileExport").toString(),"Vector files (*.pdf,*.svg)");
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileExport", fnametmp);
        if (QFileInfo(fnametmp).suffix().toLower()==QString("pdf")) {
            savePdf(fnametmp, 0, 0, QCP::epAllowCosmetic, "Neutrino", objectName());
        } else if (QFileInfo(fnametmp).suffix().toLower()==QString("svg")) {
            QSvgGenerator printer;
            printer.setFileName(fnametmp);
            QCPPainter qcpPainter;
            qcpPainter.begin(&printer);
            toPainter(&qcpPainter, viewport().width(),viewport().height());
            qcpPainter.end();
        }
    }
}

void nCustomPlot::setTitle (QString my_title) {
    if (my_title.isEmpty()) {
        if (!title->text().isEmpty()) {
            plotLayout()->take(title);
        }
    } else {
        if (title->text().isEmpty()) {
            plotLayout()->insertRow(0);
            plotLayout()->addElement(0, 0, title);
        }
    }
    title->setText(my_title);
    plotLayout()->simplify();
    replot();
}

void nCustomPlot::changeAllFonts() {
    bool ok;
    QFont myfont = QFontDialog::getFont(&ok, title->font(), this, "Title Font");
    if (ok) {
        setTitleFont(myfont);
        foreach (QCPAxis *axis, findChildren<QCPAxis *>()) {
            axis->setLabelFont(myfont);
            axis->setTickLabelFont(myfont);
        }
        replot();
    }
}

void nCustomPlot::setTitleFont (QFont myfont) {
    title->setFont(myfont);
    replot();
}

void nCustomPlot::changeTitleFont() {
    bool ok;
    QFont myfont = QFontDialog::getFont(&ok, title->font(), this, "Title Font");
    if (ok) {
        setTitleFont(myfont);
    }
}

void nCustomPlot::changeAxisFont() {
    if (sender() && sender()->property("axis").isValid()) {
        QCPAxis *axis = (QCPAxis *) sender()->property("axis").value<void *>();
        if (axis) {
            bool ok;
            QFont myfont = QFontDialog::getFont(&ok, axis->labelFont(), this, axis->label()+" Font");
            if (ok) {
                axis->setLabelFont(myfont);
                axis->setTickLabelFont(myfont);
                replot();
            }
        }
    }
}

void nCustomPlot::showGrid(int val) {
    if (sender() && sender()->property("grid").isValid()) {
        QCPGrid *grid = (QCPGrid *) sender()->property("grid").value<void *>();
        if (grid) {
            switch (val) {
            case Qt::Unchecked:
                grid->setVisible(false);
                grid->setSubGridVisible(false);
                break;
            case Qt::PartiallyChecked:
                grid->setVisible(true);
                grid->setSubGridVisible(false);
                break;
            case Qt::Checked:
                grid->setVisible(true);
                grid->setSubGridVisible(true);
                break;
            default:
                break;
            }
            replot();
        }
    }
}

void nCustomPlot::setLog(bool val) {
    if (sender() && sender()->property("axis").isValid()) {
        QCPAxis *axis = (QCPAxis *) sender()->property("axis").value<void *>();
        if (axis) {
            axis->setScaleType(val?QCPAxis::stLogarithmic:QCPAxis::stLinear);
            replot();
        }
    }
}

void nCustomPlot::setLabel(QString name) {
    if (sender() && sender()->property("axis").isValid()) {
        QCPAxis *axis = (QCPAxis *) sender()->property("axis").value<void *>();
        if (axis) {
            axis->setLabel(name);
            replot();
        }
    }
}

void nCustomPlot::setColor() {
    if (sender() && sender()->property("axis").isValid()) {
        QAbstractButton *tb = qobject_cast<QAbstractButton *>(sender());
        if(tb) {
            QCPAxis *axis = (QCPAxis *) tb->property("axis").value<void *>();
            if (axis) {
                QColorDialog colordial(axis->labelColor(),this);
                colordial.setOption(QColorDialog::ShowAlphaChannel);
                colordial.exec();
                if (colordial.result() && colordial.currentColor().isValid()) {
                    for (int g=0; g<plottableCount(); g++) {
                        QCPGraph *graph = qobject_cast<QCPGraph *>(plottable(g));
                        if(hasPlottable(graph) && graph->valueAxis()==axis) {
                            QPen p=graph->pen();
                            p.setColor(colordial.currentColor());
                            graph->setPen(p);
                        }
                    }
                    axis->setLabelColor(colordial.currentColor());
                    axis->setTickLabelColor(colordial.currentColor());
                    QPixmap px(20, 20);
                    px.fill(colordial.currentColor());
                    tb->setIcon(px);
                    replot();
                }
            }
        }
    }
}


void nCustomPlot::myAxisDoubleClick(QCPAxis*ax,QCPAxis::SelectablePart,QMouseEvent*) {
    axisRect()->setRangeDragAxes(ax,ax);
    axisRect()->setRangeDrag(ax->orientation());
    axisRect()->setRangeZoomAxes(ax,ax);
    axisRect()->setRangeZoom(ax->orientation());
}

// SETTINGS
void
nCustomPlot::loadSettings(QSettings *my_set) {
    qDebug() << __FUNCTION__ << objectName() << metaObject()->className();

    if (my_set==nullptr) {
        QString fnametmp = QFileDialog::getOpenFileName(this, tr("Open INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf);; Any files (*.*)"));
        if (!fnametmp.isEmpty()) {
            setProperty("NeuSave-fileIni",fnametmp);
            loadSettings(new QSettings(fnametmp,QSettings::IniFormat));
        }
    } else {
        my_set->beginGroup(objectName());
        my_set->beginGroup("axes");
        QList<QVariant> labels ,grids, logs, ticks, colors, labelfonts, lock, range;
        labels = my_set->value("labels").toList();
        grids = my_set->value("grids").toList();
        logs = my_set->value("logs").toList();
        colors = my_set->value("colors").toList();
        labelfonts = my_set->value("labelfonts").toList();
        lock = my_set->value("lock").toList();
        range = my_set->value("range").toList();
        QList<QCPAxis *> axis=findChildren<QCPAxis *>();
        if (    axis.size() == labels.size() &&
                axis.size() == grids.size() &&
                axis.size() == logs.size() &&
                axis.size() == colors.size()  &&
                axis.size() == labelfonts.size() &&
                axis.size() == lock.size() &&
                axis.size() == range.size())
        {
            for (int i=0; i< axis.size(); i++) {
                if(axis.at(i)->visible()) {
                    axis.at(i)->setLabel(labels.at(i).toString());
                    axis.at(i)->setLabelColor(colors.at(i).value<QColor>());
                    axis.at(i)->setLabelFont(labelfonts.at(i).value<QFont>());
                    axis.at(i)->setTickLabelFont(labelfonts.at(i).value<QFont>());
                    axis.at(i)->setTickLabelColor(colors.at(i).value<QColor>());
                    axis.at(i)->grid()->setVisible(grids.at(i).toInt()>0);
                    axis.at(i)->grid()->setSubGridVisible(grids.at(i).toInt()>1);
                    axis.at(i)->setScaleType(logs.at(i).toBool()?QCPAxis::stLogarithmic:QCPAxis::stLinear);
                    axis.at(i)->setRangeLocked(lock.at(i).toBool());
                    QPointF prange=range.at(i).toPointF();
                    axis.at(i)->setRange(prange.x(),prange.y());
                    for (int g=0; g<plottableCount(); g++) {
                        QCPGraph *graph = qobject_cast<QCPGraph *>(plottable(g));
                        if(hasPlottable(graph) && graph->valueAxis()==axis.at(i)) {
                            QPen p=graph->pen();
                            p.setColor(colors.at(i).value<QColor>());
                            graph->setPen(p);
                        }
                    }

                }
            }
        }
        my_set->endGroup();
        title->setFont(qvariant_cast<QFont>(my_set->value("titleFont",title->font())));
        setTitle(my_set->value("title",title->text()).toString());

        if (my_set->childGroups().contains("Properties")) {
            my_set->beginGroup("Properties");
            foreach(QString my_key, my_set->allKeys()) {
                qDebug() << "load" <<  my_key << " : " << my_set->value(my_key);
                setProperty(my_key.toStdString().c_str(), my_set->value(my_key));
            }
            my_set->endGroup();
        }


        my_set->endGroup();
    }
}

void
nCustomPlot::saveSettings(QSettings *my_set) {
    qDebug() << "save settings" << objectName();
    if (my_set==nullptr) {
        QString fnametmp = QFileDialog::getSaveFileName(this, tr("Save INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf)"));
        if (!fnametmp.isEmpty()) {
            setProperty("NeuSave-fileIni",fnametmp);
            saveSettings(new QSettings(fnametmp,QSettings::IniFormat));
        }
    } else {
        my_set->beginGroup(objectName());
        my_set->beginGroup("axes");
        QList<QVariant> labels ,grids, logs, ticks, colors, labelfonts, lock, range;
        foreach (QCPAxis *axis, findChildren<QCPAxis *>()) {
            labels << axis->label();
            grids << (axis->grid()->visible() && axis->grid()->subGridVisible()?2:(axis->grid()->visible()?1:0));
            logs << QVariant::fromValue(axis->scaleType()==QCPAxis::stLogarithmic);
            colors << axis->labelColor();
            labelfonts << axis->labelFont();
            lock << axis->rangeLocked();
            range << QPointF(axis->range().lower,axis->range().upper);
        }
        my_set->setValue("labels",labels);
        my_set->setValue("grids",grids);
        my_set->setValue("logs",logs);
        my_set->setValue("colors",colors);
        my_set->setValue("labelfonts",labelfonts);
        my_set->setValue("lock",lock);
        my_set->setValue("range",range);
        my_set->endGroup();
        my_set->setValue("titleFont",title->font());
        my_set->setValue("title",title->text());

        my_set->beginGroup("Properties");
        foreach(QByteArray ba, dynamicPropertyNames()) {
            qDebug() << "save" << ba << " : " << property(ba);
            if(ba.startsWith("NeuSave")) {
                qDebug() << "write" << ba << " : " << property(ba);
                my_set->setValue(ba, property(ba));
            }
        }
        my_set->endGroup();

        my_set->endGroup();
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// plot as nCustomPlot but with x mouse line
nCustomPlotMouseX::nCustomPlotMouseX(QWidget* parent): nCustomPlot(parent) {
}

void nCustomPlotMouseX::setMousePosition(double position) {
    if (!mouseMarker) mouseMarker = new QCPItemStraightLine(this);
    if (mouseMarker) {
        mouseMarker->point1->setCoords(position,0);
        mouseMarker->point2->setCoords(position,1);
    }
    replot();
}

// plot as nCustomPlot but with x and y mouse lines
nCustomPlotMouseXY::nCustomPlotMouseXY(QWidget* parent): nCustomPlot(parent) {
    setMousePosition(0,0);
}

void nCustomPlotMouseXY::setMousePosition(double positionX, double positionY) {
    if (!mouseMarkerX) mouseMarkerX = new QCPItemStraightLine(this);
    if (!mouseMarkerY) mouseMarkerY = new QCPItemStraightLine(this);

    if (mouseMarkerX && mouseMarkerY) {
        mouseMarkerX->point1->setCoords(positionX,0);
        mouseMarkerX->point2->setCoords(positionX,1);
        mouseMarkerY->point1->setCoords(0,positionY);
        mouseMarkerY->point2->setCoords(1,positionY);
    }
    replot();
}


// plot as nCustomPlot but with x and y mouse lines
nCustomDoublePlot::nCustomDoublePlot(QWidget* parent):
    nCustomPlotMouseXY(parent){

    xAxis2->setVisible(true);
    yAxis2->setVisible(true);

    yAxis->setRangeReversed(true);

    xAxis->setLabel(tr("X"));
    yAxis2->setLabel(tr("X value"));
    yAxis2->setLabelColor(Qt::red);
    yAxis2->setTickLabelColor(Qt::red);
    yAxis->setLabel(tr("Y"));
    xAxis2->setLabel(tr("Y value"));
    xAxis2->setLabelColor(Qt::blue);
    xAxis2->setTickLabelColor(Qt::blue);

    addGraph(xAxis, yAxis2);
    graph(0)->setPen(QPen(yAxis2->labelColor()));
    addGraph(yAxis, xAxis2);
    graph(1)->setPen(QPen(xAxis2->labelColor()));

}


nCustomPlotMouseX2Y::nCustomPlotMouseX2Y(QWidget* parent):
    nCustomPlotMouseX(parent)
{
    yAxis2->setVisible(true);

    yAxis->setLabelColor(Qt::red);
    yAxis->setTickLabelColor(Qt::red);

    yAxis2->setLabelColor(Qt::blue);
    yAxis2->setTickLabelColor(Qt::blue);

};

nCustomPlotMouseX3Y::nCustomPlotMouseX3Y(QWidget* parent):
    nCustomPlotMouseX2Y(parent),
    yAxis3(axisRect(0)->addAxis(QCPAxis::atRight,0))
{
    yAxis3->setLabelColor(Qt::darkCyan);
    yAxis3->setTickLabelColor(Qt::darkCyan);

};


