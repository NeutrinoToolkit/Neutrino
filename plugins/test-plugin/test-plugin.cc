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

#include "test-plugin.h"

#include "neutrino.h"

// virtuals
test_plugin::test_plugin()
	: nparent(NULL)
{ }

bool
test_plugin::instantiate(neutrino *neu)
{
	if (neu) {
		nparent = neu;
	} else
		return false;

	my_GP = new mySkelGUI(nparent, QString("This test_plugin is a test_plugin"));

	connect(my_GP, SIGNAL(destroyed(QObject *)), this, SLOT(pan_closed(QObject *)));
}

void
test_plugin::pan_closed(QObject *qobj)
{
	std::cerr<<"[test_plugin] pan closed"<<std::endl;
	emit(plugin_died(this));
}

// ------------------------------------------------------------------------------

mySkelGUI::mySkelGUI(neutrino *nparent, QString winname)
	: nGenericPan(nparent, winname)
{
	// here my pan creator
	
	// you probably want to instantiate the widget from Ui::
	my_w.setupUi(this);

	decorate();

}

Q_EXPORT_PLUGIN2(test_plugin, test_plugin)
