#include <QtGui>
#include <QMenu>

#include "nPlug.h"

#ifndef nPluginLoader_H_
#define nPluginLoader_H_

// holds plugin initialization (basic problem: nPlug is not a QObject

class nPluginLoader : public QPluginLoader {

	Q_OBJECT

public:
	nPluginLoader(QString, neutrino *);

    QString name() {return (iface ? iface->name() : QString("")); }

    bool ok() { return iface!=nullptr; }

    bool unload() {
        qDebug() << "killing me soflty" << iface;
        if (iface) {
            delete iface;
            iface=nullptr;
            nParent=nullptr;
        }
        return QPluginLoader::unload();
    }

    static QPointer<QMenu> getMenu(QString entryPoint, neutrino* neu);

public slots:
	
    void run(void);

private:
    nPlug *iface;
    neutrino *nParent;
};

#endif
