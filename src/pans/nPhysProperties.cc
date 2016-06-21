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
    nGenericPan::bufferChanged(my_phys);
    if (my_phys) {
        std::string currentProperty("");
        if (my_w.propertyList->selectedItems().size() >0) {
            currentProperty=my_w.propertyList->selectedItems().first()->text().toStdString();
        }
        my_w.propertyList->clear();
        my_w.propertyValue->clear();
        DEBUG(currentProperty);
        setWindowTitle(QString::fromUtf8(my_phys->getName().c_str()));
        for(anymap::iterator iter=my_phys->property.begin();iter!=my_phys->property.end(); iter++ ) {
            QListWidgetItem *item=new QListWidgetItem(QString::fromUtf8(iter->first.c_str()));
            my_w.propertyList->addItem(item);
            if (iter->first==currentProperty) {
                std::string myval=iter->second.get_str();
                my_w.propertyValue->setPlainText(QString::fromUtf8(myval.c_str()));
                item->setSelected(true);
            }
        }
    }
}

void
nPhysProperties::showProperty() {
	if (my_w.propertyList->currentItem()) {
        std::string currentKey=my_w.propertyList->currentItem()->text().toStdString();
        DEBUG(currentKey);
        DEBUG(currentBuffer);
        if (currentBuffer) {
            QVariant my_variant=toVariant(currentBuffer->property[currentKey]);
            DEBUG(my_variant.toString().toStdString());
            my_w.propertyValue->setPlainText(my_variant.toString());
		}
	}
}

void nPhysProperties::on_changePhysProperty_pressed() {
    DEBUG("Do something");
    QVariant pippo(my_w.propertyValue->toPlainText());
    DEBUG(pippo.toString().toStdString());
    std::string item=  my_w.propertyList->currentItem()->text().toStdString();
    if (currentBuffer) {
        anydata my_val=toAnydata(pippo);
        currentBuffer->property[item]=my_val;
        nparent->showPhys(currentBuffer);
    }
}








