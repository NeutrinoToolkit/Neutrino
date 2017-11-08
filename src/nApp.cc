#include "nApp.h"
#include "neutrino.h"
#include <QtNetwork>

#ifdef __neutrino_key
#include "nHash.h"
#endif



nApp::nApp( int &argc, char **argv ) : QApplication(argc, argv) {
    QCoreApplication::setOrganizationName("ParisTech");
    QCoreApplication::setOrganizationDomain("edu");
    QCoreApplication::setApplicationName("Neutrino");
    QCoreApplication::setApplicationVersion(__VER);

#if defined(Q_OS_MAC)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

#ifdef __neutrino_key
    std::string hh = getNHash();
    qDebug() << "got nHash: "<< hh << std::endl;
    setProperty("nHash", hh.c_str());
#endif

#if QT_VERSION >= QT_VERSION_CHECK(5, 4, 0)
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
#endif
    QCoreApplication::addLibraryPath(QCoreApplication::applicationDirPath()+QString("/plugins"));

    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    changeLocale(my_set.value("locale",QLocale()).toLocale());
    changeThreads(my_set.value("threads",1).toInt());
    if (!my_set.contains("checkUpdates")) {
        my_set.setValue("checkUpdates",QMessageBox::Yes == QMessageBox::question(nullptr,tr("Attention"),tr("Do you want to check for updates?"), QMessageBox::Yes | QMessageBox::No));
    }
    if (my_set.value("checkUpdates",true).toBool()) {
        checkUpdates();
    }
    my_set.endGroup();

}


void nApp::checkUpdates() {
    QNetworkAccessManager manager;
    QNetworkReply *response = manager.get(QNetworkRequest(QUrl("https://api.github.com/repos/NeutrinoToolkit/Neutrino/git/refs/tags/latest")));
    QEventLoop event;
    connect(response,SIGNAL(finished()),&event,SLOT(quit()));
    event.exec();
    QString html = response->readAll(); // Source should be stored here

    QJsonDocument json = QJsonDocument::fromJson(html.toUtf8());

    QJsonObject responseObject = json.object();

    if (responseObject.contains("object") && responseObject.value("object").isObject()) {
        QJsonObject objreponse = responseObject.value("object").toObject();
        qDebug() << objreponse;

        if (objreponse.contains("sha") && objreponse.value("sha").isString()) {
            QString ver = QString(__VER_LATEST);
            QString thisVer=objreponse.value("sha").toString();

            if (ver != thisVer) {
                QMessageBox::information(nullptr,tr("Attention"),tr("New version available <a href=\"https://github.com/NeutrinoToolkit/Neutrino/releases/tag/latest\">here</a> "), QMessageBox::Ok);
            }

        }
    }

}

QList<neutrino*> nApp::neus() {
    QList<neutrino*> retList;
    foreach (QWidget *widget, QApplication::topLevelWidgets()) {
        neutrino *my_neu=qobject_cast<neutrino *>(widget);
        if (my_neu) retList<< my_neu;
    }
    return retList;
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
//    qDebug() << ev;
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


