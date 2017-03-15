#include <QtCore>

#ifndef __panThread
#define __panThread

class sleeper_thread : public QThread
{
public:
	static void msleep(unsigned long msecs)
	{
		QThread::msleep(msecs);
	}
};


class panThread : public QThread {
    Q_OBJECT
public:
	panThread();
	~panThread(){};
	
	void setThread(void *iparams, void (*ifunc)(void *, int &));
	void run();
	void stop();
    
	void *params;
	void (*calculation_function)(void *, int &);

    int n_iter;
    QString err_message;

};
#endif
