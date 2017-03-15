#include "nApp.h"


nApp::nApp( int &argc, char **argv ) : QApplication(argc, argv) {
    setAttribute(Qt::AA_UseHighDpiPixmaps);

    setOrganizationName("ParisTech");
    setOrganizationDomain("edu");
    setApplicationName("Neutrino");
    setApplicationVersion(__VER);

    QCoreApplication::addLibraryPath(QCoreApplication::applicationDirPath()+QString("/plugins"));

    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    nApp::changeLocale(my_set.value("locale",QLocale()).toLocale());
    nApp::changeThreads(my_set.value("threads",1).toInt());
    my_set.endGroup();


}


bool nApp::notify(QObject *rec, QEvent *ev)
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


bool nApp::event(QEvent *ev) {
    qDebug() << ev;
//    if (ev->type() == QEvent::FileOpen) {
//        QWidget *widget = QApplication::activeWindow();
//        neutrino *neu=qobject_cast<neutrino *>(widget);
//        if (neu == NULL) {
//            nGenericPan *pan=qobject_cast<nGenericPan *>(widget);
//            if (pan) neu = pan->nparent;
//        }
//        if (neu == NULL) neu = new neutrino();
//        neu->fileOpen(static_cast<QFileOpenEvent *>(ev)->file());
//    } else {
        return QApplication::event(ev);
//    }
//    return true;
}

QString nApp::localeToString(const QLocale &locale) {
    return QLocale::languageToString(locale.language())+ " " +QLocale::countryToString(locale.country())+ " " +QLocale::scriptToString(locale.script())+ " " +QString(locale.decimalPoint());
}

bool nApp::localeLessThan(const QLocale &loc1, const QLocale &loc2)
{
    return localeToString(loc1) < localeToString(loc2);
}


void nApp::changeLocale(QLocale locale) {
    if (locale!=QLocale()) {

        qDebug() << QLocale::languageToString(locale.language()) <<
                    QLocale::scriptToString(locale.script()) <<
                    QLocale::countryToString(locale.country()) <<
                    locale.bcp47Name() <<
                    locale.country() <<
                    locale.name() <<
                    locale.decimalPoint();

        QLocale().setDefault(locale);
        QSettings settings("neutrino","");
        settings.beginGroup("nPreferences");
        settings.setValue("locale",locale);
        settings.endGroup();

        QString fileTransl(":translations/neutrino_"+locale.name()+".qm");
        if(QFileInfo(fileTransl).exists()) {
            QPointer<QTranslator> translator(new QTranslator(qApp));
            if (translator->load(fileTransl)) {
                qApp->installTranslator(translator);
                qDebug() << "installing translator" << fileTransl;
            } else {
                delete translator;
            }
        }
    }
}


void nApp::changeThreads(int num) {
    if (num<=1) {
        fftw_cleanup_threads();
    } else {
        fftw_init_threads();
        fftw_plan_with_nthreads(num);
    }
#ifdef HAVE_OPENMP
    omp_set_num_threads(num);
#endif
    DEBUG("\n\nTHREADS THREADS THREADS THREADS THREADS THREADS " << num << "\n\n");
}


