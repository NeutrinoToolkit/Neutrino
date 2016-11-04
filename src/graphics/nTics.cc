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
#include "nTics.h"
#include "neutrino.h"

nTics::nTics(neutrino *neu) : QGraphicsItem(),
    nparent(neu),
    color(QColor(Qt::black)),
    rulerVisible(false),
    gridVisible(false)
{	
}

// reimplementation
QRectF nTics::boundingRect() const {
    return nparent->my_w.my_view->sceneRect();
}

void nTics::changeColor() {
	QColorDialog colordial(color);
	colordial.setOption(QColorDialog::ShowAlphaChannel);
	colordial.exec();
	if (colordial.result() && colordial.currentColor().isValid()) {
		color=colordial.currentColor();
		update();
	}
}

void
nTics::paint(QPainter* p, const QStyleOptionGraphicsItem* option, QWidget* ) {
	if (nparent->currentBuffer) {
// enable this for testing 
        //p->drawRect(boundingRect());
        p->setClipRect( option->exposedRect );
		p->setFont(nparent->my_w.my_view->scaledFont);
		vec2f my_or=nparent->currentBuffer->get_origin();
		vec2f my_sc=nparent->currentBuffer->get_scale();
		QPen pen;
		pen.setColor(color);
		double factor=1.0/nparent->my_w.my_view->transform().m11();
		pen.setWidthF(factor);
		p->setPen(pen);
				
		int ticsPerDecade[5]={1,2,4,5,10};
		typedef QPair <QString,QRectF > my_pair;		
		QList<my_pair> rects;
		QPainterPath allTics;
		QPainterPath allGrid;
		
		QSizeF size(nparent->currentBuffer->getW(),nparent->currentBuffer->getH());

        int exponentX=log10(std::abs(my_sc.x()*size.width()));
		for (int k=0;k<5;k++) {
			allTics=QPainterPath();
			allGrid=QPainterPath();
			rects.clear();
			double ticsTmp=ticsPerDecade[k]*pow(10.0,exponentX-1);
			if (my_sc.x()>0){
				for (int i=-5.0*my_or.x()*my_sc.x()/ticsTmp-1;i<=5.0*(nparent->currentBuffer->getW()-my_or.x())*my_sc.x()/ticsTmp+1;i+=1) {
					double position=(i*ticsTmp/5.0/my_sc.x()+my_or.x());
					if (position>=0&&position<=nparent->currentBuffer->getW()) {
						allTics.moveTo(position,0);
						if (i%5) {
							allTics.lineTo(position,-0.15*p->fontMetrics().height());
						} else {
							if(gridVisible) {
								allGrid.moveTo(position,0);
								allGrid.lineTo(position,size.height());
							}
							allTics.lineTo(position,-0.3*p->fontMetrics().height());
							double numLabel=i*ticsTmp/5.0;
                            if (std::abs(exponentX)>2) numLabel/=pow(10.0,exponentX);
                            QString label=" "+QLocale().toString(numLabel)+" ";
							QSizeF labelSize=QSizeF(p->fontMetrics().width(label), p->fontMetrics().height());
							rects << qMakePair(label, QRectF(position-labelSize.width()/2,-1.3*labelSize.height(),labelSize.width(),labelSize.height()));
						}
					}
				}			
			} else {
				for (int i=-5.0*my_or.x()*my_sc.x()/ticsTmp;i>5.0*(nparent->currentBuffer->getW()-my_or.x())*my_sc.x()/ticsTmp+1;i-=1) {
					double position=(i*ticsTmp/5.0/my_sc.x()+my_or.x());
					if (position>=0&&position<=nparent->currentBuffer->getW()) {
						allTics.moveTo(position,0);
						if (i%5) {
							allTics.lineTo(position,-0.15*p->fontMetrics().height());
						} else {
							if(gridVisible) {
								allGrid.moveTo(position,0);
								allGrid.lineTo(position,size.height());
							}
							allTics.lineTo(position,-0.3*p->fontMetrics().height());
							double numLabel=i*ticsTmp/5.0;
                            if (std::abs(exponentX)>2) numLabel/=pow(10.0,exponentX);
                            QString label=" "+QLocale().toString(numLabel)+" ";
							QSizeF labelSize=QSizeF(p->fontMetrics().width(label), p->fontMetrics().height());
							rects << qMakePair(label, QRectF(position-labelSize.width()/2,-1.3*labelSize.height(),labelSize.width(),labelSize.height()));
						}
					}
				}			
			}
			bool intersected=false;
			for (int i=1; i<rects.size(); i++) {
				if(rects.at(i).second.intersects(rects.at(i-1).second)) {
					intersected=true;
				}
			}
			if (!intersected) break;			
		}
		foreach (my_pair pair, rects) {
			p->drawText(pair.second,Qt::AlignBottom|Qt::AlignHCenter,pair.first);
		}
		
        QString label;
        if (std::abs(exponentX)>2) {
            label+="x 1e"+QLocale().toString(exponentX)+" ";
		}
        if (!nparent->currentBuffer->property["unitsX"].is_none())
            label+=QString::fromStdString(nparent->currentBuffer->property["unitsX"]);
        QSizeF labelSize=QSizeF(p->fontMetrics().width(label), p->fontMetrics().height());
        if (label.trimmed().size()) p->drawText(QRectF(size.width()-labelSize.width(),-2.3*labelSize.height(),labelSize.width(),labelSize.height()),Qt::AlignTop|Qt::AlignHCenter,label);

//		allTics.moveTo(0,0);
//        allTics.lineTo(nparent->currentBuffer->getW(),0);
//        allTics.moveTo(0,0);
//		allTics.lineTo(0,nparent->currentBuffer->getH());
        allTics.addRect(0,0,nparent->currentBuffer->getW(),nparent->currentBuffer->getH());
        p->drawPath(allTics);

		p->setPen(QColor(rulerColor));
		p->drawPath(allGrid);
		p->setPen(QColor(color));
		

		
        int exponentY=log10(std::abs(my_sc.y()*size.height()));
		for (int k=0;k<5;k++) {
			allTics=QPainterPath();
			allGrid=QPainterPath();
			rects.clear();
			double ticsTmp=ticsPerDecade[k]*pow(10.0,exponentY-1);
			if (my_sc.y()>0){
				for (int i=-5.0*my_or.y()*my_sc.y()/ticsTmp-1;i<=5.0*(nparent->currentBuffer->getH()-my_or.y())*my_sc.y()/ticsTmp+1;i+=1) {
					double position=(i*ticsTmp/5.0/my_sc.y()+my_or.y());
					if (position>=0&&position<=nparent->currentBuffer->getH()) {
						allTics.moveTo(0,position);
						if (i%5) {
							allTics.lineTo(-0.15*p->fontMetrics().height(),position);
						} else {
							if(gridVisible) {
								allGrid.moveTo(0,position);
								allGrid.lineTo(size.width(),position);
							}
							allTics.lineTo(-0.3*p->fontMetrics().height(),position);
							double numLabel=i*ticsTmp/5.0;
                            if (std::abs(exponentY)>2) numLabel/=pow(10.0,exponentY);
                            QString label=" "+QLocale().toString(numLabel)+" ";
							QSizeF labelSize=QSizeF(p->fontMetrics().width(label), p->fontMetrics().height());
							rects << qMakePair(label, QRectF(position-labelSize.width()/2,0.3*p->fontMetrics().height(),labelSize.width(),labelSize.height()));
						}
					}
				}			
			} else {
				for (int i=-5.0*my_or.y()*my_sc.y()/ticsTmp;i>5.0*(nparent->currentBuffer->getH()-my_or.y())*my_sc.y()/ticsTmp+1;i-=1) {
					double position=(i*ticsTmp/5.0/my_sc.y()+my_or.y());
					if (position>=0&&position<=nparent->currentBuffer->getH()) {
						allTics.moveTo(0,position);
						if (i%5) {
							allTics.lineTo(-0.15*p->fontMetrics().height(),position);
						} else {
							if(gridVisible) {
								allGrid.moveTo(0,position);
								allGrid.lineTo(size.width(),position);
							}
							allTics.lineTo(-0.3*p->fontMetrics().height(),position);
							double numLabel=i*ticsTmp/5.0;
                            if (std::abs(exponentY)>2) numLabel/=pow(10.0,exponentY);
                            QString label=" "+QLocale().toString(numLabel)+" ";
							QSizeF labelSize=QSizeF(p->fontMetrics().width(label), p->fontMetrics().height());
							rects << qMakePair(label, QRectF(position-labelSize.width()/2,0.3*p->fontMetrics().height(),labelSize.width(),labelSize.height()));
						}
					}
				}			
			}

			bool intersected=false;
			for (int i=1; i<rects.size(); i++) {
				if(rects.at(i).second.intersects(rects.at(i-1).second)) {
					intersected=true;
				}
			}
			if (!intersected) break;			
		}
		p->rotate(90);
		foreach (my_pair pair, rects) {
			p->drawText(pair.second,Qt::AlignBottom|Qt::AlignHCenter,pair.first);
		}
		
        QString labelDim=QLocale().toString((int)nparent->currentBuffer->getW())+" x "+QLocale().toString((int)nparent->currentBuffer->getH());
		QSizeF labelDimSize=QSizeF(p->fontMetrics().width(labelDim), p->fontMetrics().height());
		if (nparent->my_w.my_view->showDimPixel) {
			p->drawText(QRectF((size.height()-labelDimSize.width())/2.0,-size.width()-labelDimSize.height(),labelDimSize.width(),labelDimSize.height()),Qt::AlignTop|Qt::AlignHCenter,labelDim);
		}
		
        label.clear();
        if (std::abs(exponentY)>2) {
            label+="x 1e"+QLocale().toString(exponentY)+" ";
		}
        if (!nparent->currentBuffer->property["unitsY"].is_none())
            label+=QString::fromStdString(nparent->currentBuffer->property["unitsY"]);
        labelSize=QSizeF(p->fontMetrics().width(label), p->fontMetrics().height());
        if (label.trimmed().size()) p->drawText(QRectF(size.height()-labelSize.width(),1.3*labelSize.height(),labelSize.width(),labelSize.height()),Qt::AlignTop|Qt::AlignHCenter,label);
        
		p->rotate(-90);

		p->drawPath(allTics);
		
		p->setPen(QColor(rulerColor));
		p->drawPath(allGrid);
		p->setPen(QColor(color));

        if (nparent->nPalettes[nparent->colorTable].size()) {
			QPen emptyPen=pen;
			emptyPen.setColor(QColor(0,0,0,0));
			emptyPen.setWidth(0);
			p->setPen(emptyPen);
			for (int i=0; i<256; i++) {
                QColor colore=QColor((int)nparent->nPalettes[nparent->colorTable][3*i+0],(int)nparent->nPalettes[nparent->colorTable][3*i+1],(int)nparent->nPalettes[nparent->colorTable][3*i+2]);
				p->setBrush(colore);
				//			p.setPen(QPen(colore));
				double dx=((double) size.width())/256.0;
				QRectF colorRect=QRectF(i*dx,size.height()+p->fontMetrics().height()/4.0,dx, p->fontMetrics().height()/2.0);
				p->drawRect(colorRect);
			}
		} else {
//			qDebug() << nparent->colorTable << (void*)&nparent->nPalettes;
//			qDebug() << nparent->nPalettes;
//			qDebug() << (void*) nparent->nPalettes[nparent->colorTable];
			WARNING("problem!!!! exetern nPalettes not found");
		}
		
		p->setPen(pen);
		p->setBrush(QColor(0,0,0,0));
		
		vec2f minmax=nparent->currentBuffer->property["display_range"];
		double mini=minmax.first();
		double maxi=minmax.second();
		
		if (maxi != mini ) {
			// this is bad should be simplified...
			//		int exponent=log10(abs(maxi-mini));
            int exponentCB=log10(std::abs(maxi-mini))-fmod(log10(std::abs(maxi-mini)),3.0);
			QList<QRectF> colorRects;
			int colorTics;
			for (colorTics=10;colorTics>=0;colorTics--) {
				colorRects.clear();
				for (int i=0; i<=colorTics; i++) {
					double number=mini+i*(maxi-mini)/((double)colorTics);
					if (exponentCB!=0) number/=pow(10.0,exponentCB);
                    QString label=" "+QLocale().toString(number)+" ";
					QSizeF labelSize=QSizeF(p->fontMetrics().width(" "+label+" "), p->fontMetrics().height());
					colorRects << QRectF(i*size.width()/((double)colorTics)-labelSize.width()/2,size.height()+55,labelSize.width(),labelSize.height());
				}
				bool intersected=false;
				for (int i=1; i<colorRects.size(); i++) {
					if(colorRects.at(i).intersects(colorRects.at(i-1))) {
						intersected=true;
						break;
					}
				}
				if (!intersected) break;
			}

			for (int i=0; i<=colorTics; i++) {
				allTics.moveTo(i*((double)size.width())/((double)colorTics),size.height()+3.0*p->fontMetrics().height()/4.0);
				allTics.lineTo(i*((double)size.width())/((double)colorTics),size.height()+p->fontMetrics().height());

                double number=mini+pow(double(i)/colorTics,1.0/nparent->currentBuffer->gamma())*(maxi-mini);

				if (exponentCB!=0) number/=pow(10.0,exponentCB);
                QString label=QLocale().toString(number,'f',2);
				QSize labelSize=QSize(p->fontMetrics().width(label), p->fontMetrics().height());
				p->drawText(QRectF(i*size.width()/((double)colorTics)-labelSize.width()/2,size.height()+p->fontMetrics().height(),labelSize.width(),labelSize.height()),Qt::AlignTop|Qt::AlignHCenter,label);
			}
            
            label.clear();
            if (std::abs(exponentCB)>2) {
                label+="x 1e"+QLocale().toString(exponentCB)+" ";
            }
            if (!nparent->currentBuffer->property["unitsCB"].is_none())
                label+=QString::fromStdString(nparent->currentBuffer->property["unitsCB"]);
            QSizeF labelSize=QSizeF(p->fontMetrics().width(label), p->fontMetrics().height());
            if (label.trimmed().size()) p->drawText(QRectF(size.width()-labelSize.width(),size.height()+2.0*p->fontMetrics().height(),labelSize.width(),labelSize.height()),Qt::AlignTop|Qt::AlignHCenter,label);

		} else {
            QString label;
            vec2f range=nparent->currentBuffer->get_min_max();
            if (range.first()==range.second()) {
                label="All image is "+QLocale().toString(range.first());
            } else {
                label="Colorbar is "+QLocale().toString(mini)+":"+QLocale().toString(maxi)+ " (" +QLocale().toString(range.first())+":"+QLocale().toString(range.second())+")";
            }

			QSize labelSize=QSize(p->fontMetrics().width(label), p->fontMetrics().height());
			p->drawText(QRectF((size.width()-labelSize.width())/2,size.height()+p->fontMetrics().height(),labelSize.width(),labelSize.height()),Qt::AlignTop|Qt::AlignHCenter,label);			
		}

		allTics.addRect(0,size.height()+p->fontMetrics().height()/4.0,size.width(), p->fontMetrics().height()/2.0);
		p->drawPath(allTics);
		
		//now draw the ruler
		if (rulerVisible && !gridVisible) {
			p->setPen(QColor(rulerColor));
			QPainterPath ruler;
			ruler.moveTo(0,nparent->currentBuffer->get_origin().y());
			ruler.lineTo(nparent->currentBuffer->getW(),nparent->currentBuffer->get_origin().y());
			ruler.moveTo(nparent->currentBuffer->get_origin().x(),0);
			ruler.lineTo(nparent->currentBuffer->get_origin().x(),nparent->currentBuffer->getH());
			p->drawPath(ruler);
		}
	}
}








