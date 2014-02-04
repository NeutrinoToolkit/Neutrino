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
	setMouseTracking(TRUE);
	offsety=9;
	dyColorBar=offsety*3/2;
	offsetx=6;
	setProperty("fileExport","Colorbar.pdf");
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
		double frac_value;
		frac_value=(e->pos().x()-offsetx)/((double) width()-2*offsetx);
		frac_value=max(0.0,min(1.0, frac_value));
		if (e->pos().y()<dyColorBar + 4*offsety) {
			colorvalue=parentPan->my_w.lineMin->text().toDouble()+frac_value*(parentPan->my_w.lineMax->text().toDouble()-parentPan->my_w.lineMin->text().toDouble());
		} else {
			colorvalue=parentPan->currentBuffer->Tminimum_value+frac_value*(parentPan->currentBuffer->Tmaximum_value-parentPan->currentBuffer->Tminimum_value);
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
			position=min(max(position,offsetx),width()-offsetx);
			p.drawLine(position,2*offsety,position,2*offsety+dyColorBar);
		}
		if (parentPan->currentBuffer->Tmaximum_value!=parentPan->currentBuffer->Tminimum_value) {
			int position=offsetx+dx*(colorvalue-parentPan->currentBuffer->Tminimum_value)/(parentPan->currentBuffer->Tmaximum_value-parentPan->currentBuffer->Tminimum_value);
			position=min(max(position,offsetx),width()-offsetx);
			p.drawLine(position,dyHisto,position,dy);
		}
	}
}

void nHistogram::drawPicture (QPainter &p) {
	int dy= height()-2*offsety;
	int dx= width()-2*offsetx;
	int dyHisto=2*offsety+2*dyColorBar;
	unsigned char *listacolori=parentPan->nparent->nPalettes[parentPan->nparent->colorTable];

	if (listacolori) {
		for (int i=0; i<256; i++) {
			QColor colore=QColor((int)listacolori[3*i+0],(int)listacolori[3*i+1],(int)listacolori[3*i+2]);
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
		vector<double> vettore=parentPan->currentBuffer->get_histogram();

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
					if (parentPan->my_w.checkBox->isChecked()) {
						frac=log10(vettore.at(i)-minivec+1.0)/log10(maxivec-minivec+1.0);
					} else {
						frac=(vettore.at(i)-minivec)/(maxivec-minivec);
					}
					frac=max(0.0,min(1.0,frac));
					polygon << QPointF(offsetx+i*dx2,dy-(dy-dyHisto)*frac);
				}
				polygon << QPointF(width()-offsetx,height()-2*offsety);
				p.drawPolygon(polygon);
			}
		}
		
		
		p.setBrush(QColor(0,0,0,127));
		p.setPen(QColor(0,0,0,127));
				
		double mini=dx*(parentPan->my_w.lineMin->text().toDouble()-parentPan->currentBuffer->Tminimum_value)/(parentPan->currentBuffer->Tmaximum_value-parentPan->currentBuffer->Tminimum_value);
		double maxi=dx*(parentPan->my_w.lineMax->text().toDouble()-parentPan->currentBuffer->Tminimum_value)/(parentPan->currentBuffer->Tmaximum_value-parentPan->currentBuffer->Tminimum_value);
		p.drawLine(offsetx,2*offsety+dyColorBar,offsetx+mini,dyHisto);
		p.drawLine(dx+offsetx,2*offsety+dyColorBar,offsetx+maxi,dyHisto);
		
		double mini_screen=min(mini,maxi);
		double maxi_screen=max(mini,maxi);
		
		p.drawRect(offsetx,dyHisto,mini_screen,dy-dyHisto);
		p.drawRect(offsetx+maxi_screen,dyHisto,dx-maxi_screen,dy-dyHisto);
		
		p.setPen(QColor(Qt::black));
		p.setBrush(QColor(0,0,0,0));
		p.drawRect(offsetx,dyHisto,dx,dy-dyHisto);
		
		double deltacolor;
		double deltaimage;
		QString pippo;
		
		int tics1=0;
		while (1) {
			tics1++;
			deltacolor=(parentPan->my_w.lineMax->text().toDouble()-parentPan->my_w.lineMin->text().toDouble())/((double)tics1);
			bool finish=false;
			for (int i=0; i<=tics1; i++) {
				int wid=p.fontMetrics().width(QString::number(parentPan->my_w.lineMin->text().toDouble()+i*deltacolor));
				if (wid>dx/tics1/2) finish=true;
			}
			if (finish) break;
		}
		
		int tics2=0;
		while (1) {
			tics2++;
			bool finish=false;
			deltaimage=(parentPan->currentBuffer->Tmaximum_value-parentPan->currentBuffer->Tminimum_value)/((double)tics2);
			for (int i=0; i<=tics2; i++) {
				int wid=p.fontMetrics().width(QString::number(parentPan->currentBuffer->Tminimum_value+i*deltaimage));
				if (wid>dx/tics2/2) finish=true;
			}
			if (finish) break;
		}
		
		for (int i=0; i<=tics1; i++) {
			p.drawLine(offsetx+i*dx/tics1,offsety+4,offsetx+i*dx/tics1,2*offsety);
			QString str1=QString::number(parentPan->my_w.lineMin->text().toDouble()+i*deltacolor);
			QRectF rect1(0,2,p.fontMetrics().width(str1),offsety);
			
			int align= Qt::AlignVCenter;
			if (i==0) {
				align|=Qt::AlignLeft;
				rect1.moveLeft(0);
			} else if (i==tics1){
				align|=Qt::AlignRight;
				rect1.moveRight(width());
			} else {
				align|=Qt::AlignHCenter;
				rect1.moveLeft(offsetx+i*dx/tics1-rect1.width()/2);
			}
			p.drawText(rect1,align,QString::number(parentPan->my_w.lineMin->text().toDouble()+i*deltacolor));
		}
		for (int i=0; i<=tics2; i++) {
			p.drawLine(offsetx+i*dx/tics2,height()-2*offsety,offsetx+i*dx/tics2,height()-offsety-4);
			
			QString str2=QString::number(parentPan->currentBuffer->Tminimum_value+i*deltaimage);
			QRectF rect2(0,height()-(offsety+2),p.fontMetrics().width(str2),offsety);
			
			int align= Qt::AlignVCenter;
			if (i==0) {
				align|=Qt::AlignLeft;
				rect2.moveLeft(0);
			} else if (i==tics2){
				align|=Qt::AlignRight;
				rect2.moveRight(width());
			} else {
				align|=Qt::AlignHCenter;
				rect2.moveLeft(offsetx+i*dx/tics2-rect2.width()/2);
			}
			p.drawText(rect2,align,QString::number(parentPan->currentBuffer->Tminimum_value+i*deltaimage));
		}
		
	} else {
		QFont font;
		font.setPointSize(2*offsety);
		p.setFont(font);		
		p.drawText(geometry(),Qt::AlignHCenter|Qt::AlignVCenter,"No image present");
	}
	
}

void nHistogram::export_PDF_slot () {
	QString filter;
	if (QFileInfo(property("fileExport").toString()).suffix().toLower()==QString("svg")) {
		filter="Scable Vector Graphics (*.svg);; PDF files (*.pdf);; Any files (*)";
	} else if (QFileInfo(property("fileExport").toString()).suffix().toLower()==QString("pdf")) {
		filter="PDF files (*.pdf);; Scable Vector Graphics (*.svg);; Any files (*)";
	}
	QString fout = QFileDialog::getSaveFileName(this,tr("Save Drawing"),property("fileExport").toString(),filter);
	if (!fout.isEmpty()) {
		setProperty("fileExport",fout);
		QPainter p;
		QPrinter *printer;
		QSvgGenerator *svgGen;
		if (QFileInfo(fout).suffix().toLower()==QString("svg")) {
			svgGen=new QSvgGenerator();
			svgGen->setFileName(fout);
			svgGen->setSize(size());
			svgGen->setTitle("Neutrino");
			svgGen->setDescription("Colorbar");
			p.begin( svgGen );
		} else if (QFileInfo(fout).suffix().toLower()==QString("pdf")) {
			printer=new QPrinter();
			printer->setOutputFormat(QPrinter::PdfFormat);
			printer->setOutputFileName(fout);
			printer->setColorMode(QPrinter::Color);
			printer->setPaperSize(size()*1.2,QPrinter::Point);
			printer->setOrientation(QPrinter::Portrait);
			p.begin( printer );
		}
		drawPicture(p);
	}
}



