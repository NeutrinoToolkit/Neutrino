#include <QtCore>

#ifndef __panThread
#define __panThread

class panThread : public QThread {
    Q_OBJECT
public:
	panThread();
	~panThread(){};
	
    void setThread(void *iparams, void (*ifunc)(void *, int &));
    void run();
	void stop();
    
    int n_iter;
    QString err_message;

private:
    void *params;
    void (*calculation_function)(void *, int &);

};
#endif
