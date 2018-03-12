#include "nPluginLoader.h"
#include "neutrino.h"
#include <QMenu>

nPluginLoader::nPluginLoader(QString pname, neutrino *neu) :
    QPluginLoader(pname),
    iface(nullptr),
    nParent(neu)
{
    setLoadHints(QLibrary::ResolveAllSymbolsHint | QLibrary::ExportExternalSymbolsHint);
    qDebug() << "Plugin" << pname ;
    QObject *p_obj = instance();

    if (p_obj) {
        iface = qobject_cast<nPlug *>(p_obj);
        if (iface) {
            QString name_plugin(iface->name());
            // in case the interface returns an empty name (default if method not overridden), pick up the name of the file
            if (name_plugin.isEmpty()) {
                name_plugin=QFileInfo(pname).baseName();

#if defined(Q_OS_MAC) || defined(Q_OS_LINUX)
                if (name_plugin.startsWith("lib")) {
                    name_plugin.remove(0,3);
                }
#endif
            }

            QIcon icon_plugin;
            QKeySequence shortcut_key;

            nPanPlug *my_panPlug = qobject_cast<nPanPlug *>(p_obj);

            if (my_panPlug) {
                icon_plugin=my_panPlug->icon();
                shortcut_key = my_panPlug->shortcut();

                if (!icon_plugin.isNull()) {

                    QList<QAction*> my_actions = neu->my_w->toolBar->actions();
                    foreach (QAction *my_action_tmp, my_actions) {
                        if (!(my_action_tmp->isSeparator() || my_action_tmp->menu()) &&
                                my_action_tmp->text()==name_plugin &&
                                my_action_tmp->isEnabled() &&
                                !my_action_tmp->data().isNull()) {
                            neu->my_w->toolBar->removeAction(my_action_tmp);
                        }
                    }
                    QPointer<QAction>  my_action;
                    my_action = new QAction(icon_plugin,name_plugin,nParent);

                    QVariant v;
                    v.setValue(this);
                    my_action->setData(v);
                    connect (my_action, SIGNAL(triggered()), this, SLOT(run()));

                    neu->my_w->toolBar->addAction(my_action);
                }

                QPointer<QMenu> my_menu=getMenu(my_panPlug->menuEntryPoint(),nParent);
                QList<QAction*> my_actions=my_menu->actions();
                foreach (QAction *my_action,my_actions) {
                    if (!(my_action->isSeparator() || my_action->menu()) && my_action->text()==name_plugin && my_action->isEnabled() && !my_action->data().isNull()) {
                        QPluginLoader *my_qplugin=my_action->data().value<QPluginLoader*>();
                        if (my_qplugin!=nullptr) {
                            if(my_qplugin->instance()){
                                delete my_qplugin;
                            }
                            my_qplugin=new QPluginLoader(pname);
                            p_obj = my_qplugin->instance();
                            if (p_obj) {
                                iface = qobject_cast<nPlug *>(p_obj);
                            }
                        }
                        my_menu->removeAction(my_action);
                    }
                }

                QApplication::processEvents();

                QPointer<QAction>  my_action = new QAction(icon_plugin, name_plugin.replace("_"," "), nParent);
                my_action->setShortcut(shortcut_key);
                QVariant v;
                v.setValue(this);
                my_action->setData(v);
                connect (my_action, SIGNAL(triggered()), this, SLOT(run()));
                my_menu->addAction(my_action);
                qDebug() << "found menu:" << my_menu;
            } else {
                run();
            }

        } else {
            QMessageBox dlg(QMessageBox::Critical, tr("Plugin error"),pname+tr(" does not look like a Neutrino plugin"));
            dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
            dlg.exec();
        }
    } else {
        QMessageBox dlg(QMessageBox::Warning, tr("Plugin error"),tr("Error loading plugin ")+QFileInfo(fileName()).fileName());
        dlg.setDetailedText(errorString());
        dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
        dlg.exec();
    }
}

QPointer<QMenu> nPluginLoader::getMenu(QString menuEntry, neutrino* neu) {
    QPointer<QMenu> my_menu;

    if (menuEntry.isEmpty()) {
        my_menu=neu->my_w->menuAnalysis;
    } else {
        QStringList my_list=menuEntry.split(";");
        // need a QWidget because it might be a QToolBar or QMenu
        QWidget *parentMenu=neu->my_w->menubar;
        for (int i=0; i<my_list.size(); i++) {
            bool found=false;
            foreach (QMenu *menu, parentMenu->findChildren<QMenu*>()) {
                if (menu->title()==my_list.at(i) || menu->objectName()==QString("menu"+my_list.at(i))) {
                    found=true;
                    if (i<my_list.size()) {
                        parentMenu = my_menu = menu;
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
                parentMenu = my_menu;
            }
        }
    }
    return my_menu;
}

void
nPluginLoader::run() {
    qDebug() << "here";
    if (iface && nParent) {
        qDebug() << iface->name();
        bool retval = iface->instantiate(nParent);
        qDebug() << retval;
        if (!retval) {
            QMessageBox dlg(QMessageBox::Critical, tr("Plugin error"),iface->name()+tr(" cannot be instantiated"));
            dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
            dlg.exec();
        }
    }
}
