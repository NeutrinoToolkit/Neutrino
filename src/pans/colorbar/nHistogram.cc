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
#include "nColorBarWin.h"
#include "nHistogram.h"
#include "neutrino.h"

nHistogram::nHistogram (QWidget *parent) : QWidget(parent)
{
    setMouseTracking(true);
    offsety=8;
    dyColorBar=offsety*3/2;
    offsetx=6;
    setMinimumHeight(200);
    parentPan=NULL;
}


void nHistogram::colorValue(double val) {
    colorvalue=val;
    repaint();
}

void nHistogram::mouseMoveEvent (QMouseEvent *e)
{
    if (parentPan->currentBuffer) {
        double  frac_value=(e->pos().x()-offsetx)/((double) width()-2*offsetx);
        frac_value=std::max(0.0,std::min(1.0, frac_value));
        //        frac_value=pow(frac_value, parentPan->currentBuffer->gamma());

        if (e->pos().y()<dyColorBar + 4*offsety) {
            colorvalue=parentPan->my_w.lineMin->text().toDouble()+frac_value*(parentPan->my_w.lineMax->text().toDouble()-parentPan->my_w.lineMin->text().toDouble());
        } else {
            colorvalue=parentPan->currentBuffer->get_min()+frac_value*(parentPan->currentBuffer->get_max()-parentPan->currentBuffer->get_min());
        }
        parentPan->my_w.statusbar->showMessage(tr("Value : ")+QString::number(colorvalue),2000);
    }
}

void nHistogram::paintEvent(QPaintEvent *)
{
    QPainter p(this);

    int dy= height()-2*offsety;
    int dx= width()-2*offsetx;
    int dyHisto=2*offsety+2*dyColorBar;

    drawPicture(p);

    if (parentPan->currentBuffer) {
        if (parentPan->my_w.lineMax->text().toDouble() != parentPan->my_w.lineMin->text().toDouble()) {
            int position=offsetx+dx*(colorvalue-parentPan->my_w.lineMin->text().toDouble())/(parentPan->my_w.lineMax->text().toDouble()-parentPan->my_w.lineMin->text().toDouble());
            position=std::min(std::max(position,offsetx),width()-offsetx);
            p.drawLine(position,2*offsety,position,2*offsety+dyColorBar);
        }
        if (parentPan->currentBuffer->get_max()!=parentPan->currentBuffer->get_min()) {
            int position=offsetx+dx*(colorvalue-parentPan->currentBuffer->get_min())/(parentPan->currentBuffer->get_max()-parentPan->currentBuffer->get_min());
            position=std::min(std::max(position,offsetx),width()-offsetx);
            p.drawLine(position,dyHisto,position,dy);
        }
    }
}

void nHistogram::drawPicture (QPainter &p) {
    int dy= height()-2*offsety;
    int dx= width()-2*offsetx;
    int dyHisto=2*offsety+2*dyColorBar;

    if (parentPan->nparent->nPalettes[parentPan->nparent->colorTable].size()) {
        for (int i=0; i<256; i++) {
            QColor colore=QColor((int)parentPan->nparent->nPalettes[parentPan->nparent->colorTable][3*i+0],(int)parentPan->nparent->nPalettes[parentPan->nparent->colorTable][3*i+1],(int)parentPan->nparent->nPalettes[parentPan->nparent->colorTable][3*i+2]);
            p.setBrush(colore);
            p.setPen(QPen(colore));
            p.drawRect(offsetx+i*dx/256,2*offsety,1+dx/256,dyColorBar);
        }
    }

    p.setPen(QColor(Qt::black));
    p.setBrush(QColor(0,0,0,0));
    p.drawRect(offsetx,2*offsety,dx,dyColorBar);

    if (parentPan->currentBuffer) {

        QFont font;
        font.setPointSize(offsety);
        p.setFont(font);

        p.setPen(QColor(Qt::black));
        p.setBrush(QColor(0,0,0,127));
        p.setPen(QColor(0,0,0,127));
        const std::vector<double> vettore=parentPan->currentBuffer->get_histogram();

        if (vettore.size()>0) {
            double dx2=((double) dx)/(vettore.size()-1);
            QPolygonF polygon;
            double minivec=vettore.at(0);
            double maxivec=vettore.at(0);
            double frac=0;
            for (unsigned int i=0; i<vettore.size(); i++) {
                if (vettore.at(i) > maxivec) maxivec=vettore.at(i);
                if (vettore.at(i) < minivec) minivec=vettore.at(i);
            }

            if (maxivec!=minivec) {
                polygon << QPointF(offsetx,height()-2*offsety);
                for (unsigned int i=0; i<vettore.size(); i++) {
                    if (parentPan->my_w.actionLog->isChecked()) {
                        frac=log10(vettore.at(i)-minivec+1.0)/log10(maxivec-minivec+1.0);
                    } else {
                        frac=(vettore.at(i)-minivec)/(maxivec-minivec);
                    }
                    frac=std::max(0.0,std::min(1.0,frac));
                    polygon << QPointF(offsetx+i*dx2,dy-(dy-dyHisto)*frac);
                }
                polygon << QPointF(width()-offsetx,height()-2*offsety);
                p.drawPolygon(polygon);
            }
        }


        p.setBrush(QColor(0,0,0,127));
        p.setPen(QColor(0,0,0,127));

        double mini=dx*(parentPan->my_w.lineMin->text().toDouble()-parentPan->currentBuffer->get_min())/(parentPan->currentBuffer->get_max()-parentPan->currentBuffer->get_min());
        double maxi=dx*(parentPan->my_w.lineMax->text().toDouble()-parentPan->currentBuffer->get_min())/(parentPan->currentBuffer->get_max()-parentPan->currentBuffer->get_min());
        p.drawLine(offsetx,2*offsety+dyColorBar,offsetx+mini,dyHisto);
        p.drawLine(dx+offsetx,2*offsety+dyColorBar,offsetx+maxi,dyHisto);

        double mini_screen=std::min(mini,maxi);
        double maxi_screen=std::max(mini,maxi);

        p.drawRect(offsetx,dyHisto,mini_screen,dy-dyHisto);
        p.drawRect(offsetx+maxi_screen,dyHisto,dx-maxi_screen,dy-dyHisto);

        p.setPen(QColor(Qt::black));
        p.setBrush(QColor(0,0,0,0));
        p.drawRect(offsetx,dyHisto,dx,dy-dyHisto);

        int num_labels=0;
        bool too_big=false;
        vec2f minmax=parentPan->currentBuffer->property.have("display_range") ? parentPan->currentBuffer->property["display_range"] : parentPan->currentBuffer->get_min_max();
        while (!too_big) {
            num_labels++;
            too_big=false;
            for (int i=0; i<=num_labels; i++) {
                double val=minmax.x()+pow(double(i)/num_labels,parentPan->currentBuffer->gamma())*(minmax.y()-minmax.x());
                if (p.fontMetrics().width(QString::number(val))>double(dx)/(1.5*num_labels)) {
                    too_big=true;
                }
            }
            if (too_big) {
                num_labels--;
                for (int i=0; i<=num_labels; i++) {
                    p.drawLine(offsetx+i*dx/num_labels,offsety+4,offsetx+i*dx/num_labels,2*offsety);
                    double val=minmax.x()+pow(double(i)/num_labels,parentPan->currentBuffer->gamma())*(minmax.y()-minmax.x());
                    QString str1=QString::number(val);
                    QRectF rect1(0,2,p.fontMetrics().width(str1),offsety);

                    int align= Qt::AlignVCenter;
                    if (i==0) {
                        align|=Qt::AlignLeft;
                        rect1.moveLeft(0);
                    } else if (i==num_labels){
                        align|=Qt::AlignRight;
                        rect1.moveRight(width());
                    } else {
                        align|=Qt::AlignHCenter;
                        rect1.moveLeft(offsetx+i*dx/num_labels-rect1.width()/2);
                    }
                    p.drawText(rect1,align,str1);
                }
            }
        }

        num_labels=0;
        too_big=false;
        minmax=parentPan->currentBuffer->get_min_max();
        while (!too_big) {
            num_labels++;
            too_big=false;
            for (int i=0; i<=num_labels; i++) {
                double val=minmax.x()+pow(double(i)/num_labels,parentPan->currentBuffer->gamma())*(minmax.y()-minmax.x());
                if (p.fontMetrics().width(QString::number(val))>double(dx)/(1.5*num_labels)) {
                    too_big=true;
                }
            }
            if (too_big) {
                num_labels--;
                for (int i=0; i<=num_labels; i++) {
                    p.drawLine(offsetx+i*dx/num_labels,height()-2*offsety,offsetx+i*dx/num_labels,height()-offsety-4);
                    double val=minmax.x()+pow(double(i)/num_labels,parentPan->currentBuffer->gamma())*(minmax.y()-minmax.x());
                    QString str1=QString::number(val);
                    QRectF rect1(0,height()-(offsety+2),p.fontMetrics().width(str1),offsety);

                    int align= Qt::AlignVCenter;
                    if (i==0) {
                        align|=Qt::AlignLeft;
                        rect1.moveLeft(0);
                    } else if (i==num_labels){
                        align|=Qt::AlignRight;
                        rect1.moveRight(width());
                    } else {
                        align|=Qt::AlignHCenter;
                        rect1.moveLeft(offsetx+i*dx/num_labels-rect1.width()/2);
                    }
                    p.drawText(rect1,align,str1);
                }
            }
        }


    } else {
        QFont font;
        font.setPointSize(3*offsety);
        p.setFont(font);
        p.drawText(geometry(),Qt::AlignHCenter|Qt::AlignVCenter,"No image present");
    }

}



