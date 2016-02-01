#include "nApp.h"


QList<neutrino*> NApplication::neus() {
    QList<neutrino*> retList;
    foreach (QWidget *widget, QApplication::topLevelWidgets()) {
        neutrino *my_neu=qobject_cast<neutrino *>(widget);
        if (my_neu) retList<< my_neu;
    }
    return retList;
}

QStringList NApplication::neuNames() {
    QStringList retList;
    foreach (neutrino* my_neu, neus()) {
        retList<< my_neu->objectName();
    }
    return retList;
}

neutrino* NApplication::neu(QString neu_name) {
    QList<neutrino*> retList;
    foreach (neutrino* my_neu, neus()) {
        if (my_neu->objectName()==neu_name) return my_neu;
    }
    return new neutrino();
}

bool NApplication::event(QEvent *ev) {
    DEBUG("MAC APPLICATION EVENT " << ev->type());
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

