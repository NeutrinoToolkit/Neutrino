#include "nPluginLoader.h"
#include "neutrino.h"
#include <QMenu>

nPluginLoader::nPluginLoader(QString pname, neutrino *neu)
    : QPluginLoader(pname), iface(nullptr), nParent(neu)
{

      qDebug() << "Parsing lib " << pname;

	  QObject *p_obj = instance();

      qDebug() << "here";

      if (p_obj) {
          qDebug() << "here";
          iface = qobject_cast<nPlug *>(p_obj);
			if (iface) {

                qDebug() << "here";

                QString name_plugin=iface->name();
                QPointer<QMenu> my_menu=nParent->my_w.menuPlugins;

                // in case the interface returns an empty name (default if method not overridden), pick up the name of the file
                if (name_plugin.isEmpty()) {
                    name_plugin=QFileInfo(pname).baseName();

                    #if defined(Q_OS_MAC) || defined(Q_OS_LINUX)
                    if (name_plugin.startsWith("lib")) {
                        name_plugin.remove(0,3);
                    }
                    #endif
                } else {
                    QStringList my_list=name_plugin.split(";");

                    if (my_list.size()>1) {
                        name_plugin=my_list.takeLast();
                        QWidget *parentMenu=nParent->my_w.menubar;
                        int i=0;
                        while (i<my_list.size()) {
                            bool found=false;
                            foreach (QMenu *menu, parentMenu->findChildren<QMenu*>()) {
                                if (menu->title()==my_list.at(i)) {
                                    found=true;
                                    if (i<my_list.size()) {
                                        parentMenu=menu;
                                        my_menu=menu;
                                        break;
                                    }
                                }
                            }
                            if (!found) {
                                if (qobject_cast<QMenuBar*>(parentMenu)) {
                                    my_menu=(qobject_cast<QMenuBar*>(parentMenu))->addMenu(my_list.at(i));
                                } else if(qobject_cast<QMenu*>(parentMenu)) {
                                    my_menu=(qobject_cast<QMenu*>(parentMenu))->addMenu(my_list.at(i));
                                }
                                my_menu->setTitle(my_list.at(i));
                                parentMenu=my_menu;
                            }
                            i++;
                        }
                    }
                }

                foreach (QAction *action, my_menu->actions()) {
                    if (!(action->isSeparator() || action->menu()) && action->text()==name_plugin && action->isEnabled()) {
                        qDebug() << "here" << action->data();
                        if (!action->data().isNull()) {
                            QPluginLoader *my_qplugin=action->data().value<QPluginLoader*>();
                            qDebug() << action->data() << my_qplugin;
                            if (my_qplugin!=nullptr) {
                                if(my_qplugin->instance()){
                                    delete my_qplugin;
                                    qDebug() << "instance removed";
                                }
                                my_menu->removeAction(action);
                                my_qplugin=new QPluginLoader(pname);
                                p_obj = my_qplugin->instance();
                                if (p_obj) {
                                    iface = qobject_cast<nPlug *>(p_obj);
                                    if (iface) {
                                        qDebug() << "reloaded";
                                    }
                                }
                            }
                        } else {
                            QMessageBox dlg(QMessageBox::Critical, tr("Plugin error"),pname+tr(" already loaded and can't be unloaded"));
                            dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
                            dlg.exec();
                        }
                    }
                }

                QApplication::processEvents();

                QPointer<QAction>  my_action = new QAction(nParent);
                my_action->setText(name_plugin);
                my_action->setProperty("neuPlugin",true);
                QVariant v;
                v.setValue(this);
                my_action->setData(v);
                connect (my_action, SIGNAL(triggered()), this, SLOT(launch()));
                my_menu->addAction(my_action);

            } else {
                QMessageBox dlg(QMessageBox::Critical, tr("Plugin error"),pname+tr(" does not look like a Neutrino plugin"));
                dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
                dlg.exec();
            }
      } else {
          QMessageBox dlg(QMessageBox::Critical, tr("Plugin error"),pname+tr(" does not look like a plugin"));
          dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
          dlg.exec();
      }
}

void
nPluginLoader::launch() {
    qDebug() << "here";
    qDebug() << iface->name();

    if (iface && nParent) {
        qDebug() << "here";
        iface->instantiate(nParent);
        qDebug() << "here";
    }
    qDebug() << "here";
}
