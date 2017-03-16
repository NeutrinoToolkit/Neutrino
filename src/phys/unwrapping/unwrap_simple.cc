#include "unwrap_simple.h"

void unwrap_simple_h (physD *phase, physD *uphase) {
	double buffer,bufferold,dummy,diff;
	unsigned int i,j;
	
	for (j=0;j<phase->getH();j++) {
		dummy=0.0;
		bufferold=phase->point(0,j,0);
		uphase->set(0,j,bufferold);
		for (i=1;i<phase->getW();i++) {
			buffer=phase->point(i,j,0);
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

void unwrap_simple_v (physD *phase, physD *uphase) {
	double buffer,bufferold,dummy,diff;
	unsigned int i,j;

	for (i=0;i<phase->getW();i++) {
		dummy=0.0;
		bufferold=phase->point(i,0,0);
		uphase->set(i,0,bufferold);
		for (j=1;j<phase->getH();j++) {
			buffer=phase->point(i,j,0);
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
