#ifndef __nHolderGui
#define __nHolderGui

#include "nPhysImageF.h"
#include "nPhysD.h"
#include <list>
#include "ui_nHolderGui.h"


class nHolderGui: public nGenericPan, private Ui::nHolderGuis { 
    Q_OBJECT

public:
	Q_INVOKABLE nHolderGui();
	

};


#endif
