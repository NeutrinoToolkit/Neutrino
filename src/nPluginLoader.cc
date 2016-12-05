#include "nPluginLoader.h"
#include "neutrino.h"
#include <QMenu>

nPluginLoader::nPluginLoader(QString pname, neutrino *neu)
	: QPluginLoader(pname), iface(NULL), nParent(neu)
{

      qDebug() << "Parsing lib " << pname;

	  QObject *p_obj = instance();

      if (p_obj) {
		  iface = qobject_cast<nPlug *>(p_obj);
			if (iface) {


                QString name_plugin=iface->name();

                // in case the interface returns an empty name (default if methon not overridden), pick up the name of the file
                if (name_plugin.isEmpty()) {
                    name_plugin=QFileInfo(pname).baseName();

                    #if defined(Q_OS_MAC) || defined(Q_OS_LINUX)
                    if (name_plugin.startsWith("lib")) {
                        name_plugin.remove(0,3);
                    }
                    #endif
                }

                QPointer<QMenu> my_menu=nParent->my_w.menuPlugins;
                QStringList my_list=name_plugin.split(";");

                if (my_list.size()>1) {
                    pname=my_list.takeLast();
                    QWidget *parentMenu=nParent->my_w.menubar;
                    unsigned int i=0;
                    while (i<my_list.size()) {
                        bool found=false;
                        foreach (QMenu *menu, parentMenu->findChildren<QMenu*>()) {
                            if (menu->title()==my_list.at(i)) {
                                found=true;
                                if (i<my_list.size()) {
                                    i++;
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
                            i++;
                        }
                    }
                }

                QPointer<QAction>  my_action;
                foreach (QAction *action, my_menu->actions()) {
                    if (!(action->isSeparator() || action->menu())) {
                        if (action->text()==name_plugin) {
                            my_action=action;
                            nPluginLoader *my_npl=qvariant_cast<nPluginLoader*>(action->data());
                            if (my_npl) {
                                if (!my_npl->unload()) {
                                    QMessageBox dlg(QMessageBox::Critical, tr("Plugin error"),pname+tr(" already loaded and can't be unloaded"));
                                    dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
                                    dlg.exec();
                                }
                            }
                        }
                    }
                }

                if (my_action.isNull()) {
                    my_action = new QAction(nParent);
                }
                my_action->setText(name_plugin);
                my_action->setProperty("neuPlugin",true);
                my_action->setData(qVariantFromValue((void *) this));
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
    if (iface && nParent) {
		iface->instantiate(nParent);
    }
}
