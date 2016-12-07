#include <QtGui>

#include "nPlug.h"

#ifndef __nPluginLoader
#define __nPluginLoader

// holds plugin initialization (basic problem: nPlug is not a QObject

class nPluginLoader : public QPluginLoader {

	Q_OBJECT

public:
	nPluginLoader(QString, neutrino *);

	QString name()
	{ if (iface) return iface->name();
	else return QString(""); }

    bool ok()
    { return iface!=nullptr; }

    bool unload() {
        qDebug() << "killing me soflty" << iface;
        if (iface) {
            delete iface;
            iface=nullptr;
            nParent=nullptr;
        }
        return QPluginLoader::unload();
    }

public slots:
	
	void launch(void);


private:
    nPlug *iface;
    neutrino *nParent;
	
};

#endif
