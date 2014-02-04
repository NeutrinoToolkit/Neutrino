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
#include "nOpenRAW.h"
#include "neutrino.h"
#include "nPhysFormats.h"

// physWavelets

nOpenRAW::nOpenRAW(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);
	connect(my_w.okButton, SIGNAL(pressed()), this, SLOT(doOpen()));
	decorate();
}

void nOpenRAW::checkStringList() {
	fileList.removeDuplicates();
	if (fileList.size()>0) {
		my_w.okButton->setText(tr("Open ")+QString::number(fileList.size())+tr(" image")+(fileList.size()>1?tr("s"):""));
	} else {
		my_w.okButton->setText(tr("No image"));
	}
}

void nOpenRAW::add(QString fname) {
	fileList << fname;
	checkStringList();
}

void nOpenRAW::add(QStringList fnames) {
	fileList << fnames;
	checkStringList();
}

void nOpenRAW::doOpen () {
	foreach (QString fname, fileList) {
		nPhysD *datamatrix = new nPhysD(my_w.width->value(), my_w.height->value(), 0.0,fname.toUtf8().constData());
		phys_open_RAW(datamatrix,my_w.kind->currentIndex(), my_w.skip->value(), my_w.endian->isChecked());		
		if (datamatrix && datamatrix->getSurf()>0) {
			datamatrix->setShortName(QFileInfo(fname).fileName().toStdString());
			datamatrix->setFromName(fname.toStdString());
			datamatrix->TscanBrightness(); //FIXME: this should be removed when fixed deep copy
			datamatrix->setType(PHYS_FILE);
			nparent->addShowPhys(datamatrix);
			
			QSettings settings("neutrino","");
			QStringList listarecentfiles=settings.value("recentFiles").toStringList();
			foreach (QString str, listarecentfiles) {
				if (!QFile(str).exists()) {
					listarecentfiles.removeAll(str);
				}
			}
			listarecentfiles.prepend(fname);
			listarecentfiles.removeDuplicates();
			while (listarecentfiles.size()>20) {
				listarecentfiles.removeLast();
			}
			settings.setValue("recentFiles",listarecentfiles);
			
			nparent->updateRecentFileActions();
		}
	}
}

