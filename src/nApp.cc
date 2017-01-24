#include "nApp.h"
#include "neutrino.h"

#ifdef __neutrino_key
#include "nHash.h"
#endif


NApplication::NApplication( int &argc, char **argv ) : QApplication(argc, argv) {
#ifdef USE_QT5
    setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

    setOrganizationName("ParisTech");
    setOrganizationDomain("edu");
    setApplicationName("Neutrino");
    setApplicationVersion(__VER);

#ifdef __neutrino_key
    std::string hh = getNHash();
    qDebug() << "got nHash: "<< hh << std::endl;
    setProperty("nHash", hh.c_str());
#endif
}


#ifdef HAVE_PYTHONQT
QList<neutrino*> NApplication::neus() {
    QList<neutrino*> retList;
    foreach (QWidget *widget, QApplication::topLevelWidgets()) {
        neutrino *my_neu=qobject_cast<neutrino *>(widget);
        if (my_neu) retList<< my_neu;
    }
    return retList;
}
#endif

bool NApplication::notify(QObject *rec, QEvent *ev)
{
    try {
        return QApplication::notify(rec, ev);
    } catch (std::exception &e) {
        QMessageBox dlg(QMessageBox::Critical, tr("Exception"), e.what());
        dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
        dlg.exec();
    }

    return false;
}


bool NApplication::event(QEvent *ev) {
    qDebug() << ev;
    if (ev->type() == QEvent::FileOpen) {
        QWidget *widget = QApplication::activeWindow();
        neutrino *neu=qobject_cast<neutrino *>(widget);
        if (neu == NULL) {
            nGenericPan *pan=qobject_cast<nGenericPan *>(widget);
            if (pan) neu = pan->nparent;
        }
        if (neu == NULL) neu = new neutrino();
        neu->fileOpen(static_cast<QFileOpenEvent *>(ev)->file());
    } else {
        return QApplication::event(ev);
    }
    return true;
}

