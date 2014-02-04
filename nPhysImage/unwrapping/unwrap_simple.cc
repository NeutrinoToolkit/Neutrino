#include "unwrap_simple.h"

void unwrap_simple_h (nPhysD *phase, nPhysD *uphase) {
	double buffer,bufferold,dummy,diff;
	unsigned int i,j;
	
	for (j=0;j<phase->getH();j++) {
		dummy=0.0;
		bufferold=phase->point(0,j);
		uphase->set(0,j,bufferold);
		for (i=1;i<phase->getW();i++) {
			buffer=phase->point(i,j);
			diff=buffer-bufferold;
			if (diff>0.5) {
				dummy-=1.0;
			} else if (diff<-0.5) {
				dummy+=1.0;
			}
			bufferold=buffer;
			uphase->set(i,j,buffer+dummy);
		}
	}
}

void unwrap_simple_v (nPhysD *phase, nPhysD *uphase) {
	double buffer,bufferold,dummy,diff;
	unsigned int i,j;

	for (i=0;i<phase->getW();i++) {
		dummy=0.0;
		bufferold=phase->point(i,0);
		uphase->set(i,0,bufferold);
		for (j=1;j<phase->getH();j++) {
			buffer=phase->point(i,j);
			diff=buffer-bufferold;
			if (diff>0.5) {
				dummy-=1.0;
			} else if (diff<-0.5) {
				dummy+=1.0;
			}
			bufferold=buffer;
			uphase->set(i,j,buffer+dummy);
		}
	}
}
