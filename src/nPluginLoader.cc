#include "nPluginLoader.h"

nPluginLoader::nPluginLoader(QString pname, neutrino *neu)
	: QPluginLoader(pname), iface(NULL), nParent(neu)
{

	  DEBUG(10, "Parsing lib "<<pname.toStdString());

	  QObject *p_obj = instance();

	  if (p_obj) {
		  iface = qobject_cast<nPlug *>(p_obj);
			if (iface) {
				DEBUG("plugin \""<<iface->name().toStdString()<<"\" cast success");
	
				//if (plug_iface->instantiate(this))
				//	DEBUG("plugin \""<<plug_iface->name().toStdString()<<"\" instantiate success");
				//QAction *action=new QAction(this);

			} else {
				DEBUG("plugin load fail");
			}
		} else {
			DEBUG("plugin cannot be loaded (linking problems?)");
		}


}

void
nPluginLoader::launch() {
	if (iface && nParent)
		iface->instantiate(nParent);

}
