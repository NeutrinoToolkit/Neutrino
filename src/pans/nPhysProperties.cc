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
#include "neutrino.h"
#include "nPhysProperties.h"
nPhysProperties::nPhysProperties(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname) {
	my_w.setupUi(this);

	setWindowFlags(Qt::Tool);
	// CHECK: this should be ok...

	my_w.splitter->setStretchFactor(0, 1);
	my_w.splitter->setStretchFactor(1, 2);
	connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(bufferChanged(nPhysD*)));
	
	connect(my_w.propertyList, SIGNAL(itemSelectionChanged()), this, SLOT(showProperty()));

	bufferChanged(nparent->currentBuffer);

	show();
	decorate();
}

void
nPhysProperties::bufferChanged(nPhysD *my_phys) {
	string currentProperty("");
	if (my_w.propertyList->selectedItems().size() >0) {
		currentProperty=my_w.propertyList->selectedItems().first()->text().toStdString();
	}
	my_w.propertyList->clear();
	my_w.propertyValue->clear();
	DEBUG(currentProperty);
	if (my_phys) {
		setWindowTitle(QString::fromStdString(my_phys->getName()));
		for(anymap::iterator iter=my_phys->property.begin();iter!=my_phys->property.end(); iter++ ) {
			QListWidgetItem *item=new QListWidgetItem(QString::fromStdString(iter->first));
			my_w.propertyList->addItem(item);			
			if (iter->first==currentProperty) {
				my_w.propertyValue->setPlainText(QString::fromStdString(iter->second));
				item->setSelected(true);
				DEBUG("here\n");
			}
		}
	}
}

void
nPhysProperties::showProperty() {
	DEBUG("Here");
	if (my_w.propertyList->currentItem()) {
		string currentKey=my_w.propertyList->currentItem()->text().toStdString();
		DEBUG(currentKey);
		if (currentBuffer) {
			my_w.propertyValue->setPlainText(QString::fromStdString(currentBuffer->property[currentKey]));
		}
	}
}








