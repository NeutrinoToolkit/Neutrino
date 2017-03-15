#ifndef __HolderGUI
#define __HolderGUI

#include "nHolder.h"
#include "nGenericPan.h"
#include "ui_HolderGUI.h"


class HolderGUI: public QMainWindow, private Ui::HolderGUIs {
    Q_OBJECT

public:
	Q_INVOKABLE HolderGUI();
	

public slots:
    
//    void addPan(nGenericPan* pan) {
//        panlist.push_back(pan);
//    }
//    
//    void delPan(nGenericPan* pan) {
//        panlist.removeAll(pan);
//    }

private:
//    QList<nGenericPan*> panlist;

};


#endif
