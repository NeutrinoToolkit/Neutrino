#include "unwrap.h"
#include <cmath>
#include <limits>

// unwrap a line of n pixels: get(k) wrapped phase, start(k) value already in the output (if from_uphase), set(k,v) output
// NaN pixels stay NaN and are skipped; the line is anchored on its first finite pixel
template <class G, class S, class P>
static void simple_line(unsigned int n, G get, S start, P set, bool from_uphase) {
    double dummy=0.0;
    double bufferold=std::numeric_limits<double>::quiet_NaN();
    bool anchored=false;
    for (unsigned int k=0; k<n; k++) {
        double buffer=get(k);
        if (!std::isfinite(buffer)) {
            set(k,buffer);
            continue;
        }
        if (!anchored) {
            if (from_uphase && std::isfinite(start(k))) {
                dummy=std::round(start(k)-buffer); // continue from the first pass
            }
            anchored=true;
        } else {
            double diff=buffer-bufferold;
            if (diff>0.5) {
                dummy-=1.0;
            } else if (diff<-0.5) {
                dummy+=1.0;
            }
        }
        bufferold=buffer;
        set(k,buffer+dummy);
    }
}

void unwrap::simple_h (physD &phase, physD &uphase, bool from_uphase) {
    for (unsigned int j=0;j<phase.getH();j++) {
        simple_line(phase.getW(),
                    [&](unsigned int i) { return phase.point(i,j); },
                    [&](unsigned int i) { return uphase.point(i,j); },
                    [&](unsigned int i, double v) { uphase.set(i,j,v); },
                    from_uphase);
    }
}

void unwrap::simple_v (physD &phase, physD &uphase, bool from_uphase) {
    for (unsigned int i=0;i<phase.getW();i++) {
        simple_line(phase.getH(),
                    [&](unsigned int j) { return phase.point(i,j); },
                    [&](unsigned int j) { return uphase.point(i,j); },
                    [&](unsigned int j, double v) { uphase.set(i,j,v); },
                    from_uphase);
    }
}

// old implementation: the second pass restarted every line from the wrapped phase,
// so H+V was a pure V unwrap (each column with its own offset)
// void unwrap::simple_h (physD &phase, physD &uphase) {
// 	double buffer,bufferold,dummy,diff;
// 	unsigned int i,j;
//
//     for (j=0;j<phase.getH();j++) {
// 		dummy=0.0;
//         bufferold=phase.point(0,j);
//         uphase.set(0,j,bufferold);
//         for (i=1;i<phase.getW();i++) {
//             buffer=phase.point(i,j);
// 			diff=buffer-bufferold;
// 			if (diff>0.5) {
// 				dummy-=1.0;
// 			} else if (diff<-0.5) {
// 				dummy+=1.0;
// 			}
// 			bufferold=buffer;
//             uphase.set(i,j,buffer+dummy);
// 		}
// 	}
// }
//
// void unwrap::simple_v (physD &phase, physD &uphase) {
// 	double buffer,bufferold,dummy,diff;
// 	unsigned int i,j;
//
//     for (i=0;i<phase.getW();i++) {
// 		dummy=0.0;
//         bufferold=phase.point(i,0);
//         uphase.set(i,0,bufferold);
//         for (j=1;j<phase.getH();j++) {
//             buffer=phase.point(i,j);
// 			diff=buffer-bufferold;
// 			if (diff>0.5) {
// 				dummy-=1.0;
// 			} else if (diff<-0.5) {
// 				dummy+=1.0;
// 			}
// 			bufferold=buffer;
//             uphase.set(i,j,buffer+dummy);
// 		}
// 	}
// }
