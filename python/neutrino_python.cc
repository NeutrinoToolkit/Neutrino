/*
 *
 *    Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 *
 *    This file is part of neutrino.
 *
 *    Neutrino is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Neutrino is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public License
 *    along with neutrino.  If not, see <http://www.gnu.org/licenses/>.
 *
 *    Contact Information:
 *	Alessandro Flacco <alessandro.flacco@polytechnique.edu>
 *	Tommaso Vinci <tommaso.vinci@polytechnique.edu>
 *
 */

// The neutrino python module (nanobind): Neutrino windows in a python program, e.g. with PySide6
//
//     from PySide6.QtWidgets import QApplication
//     import neutrino, numpy as np
//     app = QApplication([])
//     w = neutrino.Window()
//     w.addShowPhys(np.random.rand(512, 512))
//     w.bufferChanged.connect(lambda: print(w.currentBuffer().mean()))
//     app.exec()
//
// The QApplication must use the same Qt library as libNeutrino (e.g. Homebrew pyside with Homebrew qt).
// Without PySide, neutrino.app() creates the QApplication and neutrino.exec() runs it.
//
// Window: a Neutrino main window. Image: a buffer of a window (numpy arrays are copied in and out).
// Pan: a pan (plugin) of a window: set/get/button by widget name, and any slot, Q_INVOKABLE or Q_PROPERTY
// of the plugin by name (Qt meta-object). widget() gives the PySide object (shiboken6).

#include <map>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include <QApplication>
#include <QPointer>
#include <QMetaMethod>
#include <QMetaProperty>
#include <QPointF>
#include <QRectF>

#include <cstring>
#include <limits>
#include <map>
#include <memory>

#include "neutrino.h"
#include "nCore.h"
#include "nGenericPan.h"
#include "nPanPlug.h"
#include "nPhysD.h"
#include "nVersion.h"

namespace nb = nanobind;
using namespace nb::literals;

using Array2D = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using ArrayIn = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// ------------------------------------------------------------------ application

static QApplication* checkApplication() {
    QApplication *qapp=qobject_cast<QApplication*>(QCoreApplication::instance());
    if (qapp) return qapp;
    // a PySide application that this Qt library doesn't see: PySide6 has its own copy of Qt
    nb::object modules=nb::module_::import_("sys").attr("modules");
    if (nb::cast<bool>(modules.attr("__contains__")("PySide6.QtCore"))) {
        nb::object qtcore=nb::module_::import_("PySide6.QtCore");
        if (!qtcore.attr("QCoreApplication").attr("instance")().is_none()) {
            throw std::runtime_error("PySide6 uses another Qt library (" + nb::cast<std::string>(qtcore.attr("qVersion")()) + ", "
                                     + nb::cast<std::string>(nb::str(qtcore.attr("__file__"))) + ") than Neutrino (" + qVersion()
                                     + "): PySide6 must be built on the same Qt (e.g. Homebrew pyside with Homebrew qt)");
        }
    }
    throw std::runtime_error("no QApplication: create it first (PySide6: QApplication([])) or call neutrino.app()");
}

static int app_argc=1;
static char app_name[]="neutrino";
static char *app_argv[]={app_name, nullptr};
// the QApplication made by neutrino.app() (deleted at the exit of python)
static QApplication *own_app=nullptr;

// ------------------------------------------------------------------ conversions

static Array2D toArray(const physD &phys) {
    size_t h=phys.getH(), w=phys.getW();
    double *data=new double[std::max<size_t>(h*w,1)];
    if (h*w) std::memcpy(data, phys.Timg_buffer, h*w*sizeof(double));
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double*>(p); });
    return Array2D(data, {h, w}, owner);
}

// the rows of the array are the y of the image
static void fromArray(physD &phys, const ArrayIn &arr) {
    unsigned int h=arr.shape(0), w=arr.shape(1);
    if (phys.getW()!=w || phys.getH()!=h) phys.resize(w, h);
    if (h*w) std::memcpy(phys.Timg_buffer, arr.data(), size_t(h)*w*sizeof(double));
}

static nb::object fromVariant(const QVariant &val);
static QVariant toVariant(nb::handle obj);
static bool convertTo(QVariant &val, QMetaType type);

// ------------------------------------------------------------------ Image

struct PyImage {
    QPointer<neutrino> neu;
    nPhysD *phys;

    nPhysD* get() const {
        if (!neu || !neu->nPhysExists(phys)) throw std::runtime_error("the image was closed");
        return phys;
    }
    void changed() const {
        nPhysD *p=get();
        p->TscanBrightness(); // min and max, and the display cache
        if (neu->getCurrentBuffer()==p) neu->showPhys(p);
        neu->emitPhysChanged(p);
    }
};

// the window that has this image (nullptr if none)
static neutrino* windowOf(nPhysD *phys) {
    for (auto &neu : nCore::self()->neus()) {
        if (neu->nPhysExists(phys)) return neu;
    }
    return nullptr;
}

// ------------------------------------------------------------------ Window and Pan

struct PyWindow {
    QPointer<neutrino> neu;
    neutrino* get() const {
        if (!neu) throw std::runtime_error("the window was closed");
        return neu;
    }
};

struct PyPan {
    QPointer<nGenericPan> pan;
    nGenericPan* get() const {
        if (!pan) throw std::runtime_error("the pan was closed");
        return pan;
    }
};

static nb::object toPython(nPhysD *phys, neutrino *neu=nullptr) {
    if (!phys) return nb::none();
    if (!neu) neu=windowOf(phys);
    return nb::cast(PyImage{neu, phys});
}

static nb::object toPython(nGenericPan *pan) {
    if (!pan) return nb::none();
    return nb::cast(PyPan{pan});
}

// the PySide objects of the widgets, one each until the widget is destroyed: when the last python reference to a
// wrapInstance object goes away, shiboken invalidates the objects found through it (e.g. findChild on a temporary
// pan.widget() gave "Internal C++ object already deleted")
static std::map<QWidget*, nb::object> pyside_widgets;

// the PySide object of a widget (shiboken6 must be the one of a PySide6 on the same Qt)
static nb::object pysideWidget(QWidget *wdg, const char *pysideClass) {
    // nb::module_ shiboken=nb::module_::import_("shiboken6");
    // nb::object cls=nb::module_::import_("PySide6.QtWidgets").attr(pysideClass);
    // return shiboken.attr("wrapInstance")(nb::int_(reinterpret_cast<uintptr_t>(wdg)), cls);
    auto it=pyside_widgets.find(wdg);
    if (it!=pyside_widgets.end()) return it->second;
    nb::module_ shiboken=nb::module_::import_("shiboken6");
    nb::object cls=nb::module_::import_("PySide6.QtWidgets").attr(pysideClass);
    nb::object obj=shiboken.attr("wrapInstance")(nb::int_(reinterpret_cast<uintptr_t>(wdg)), cls);
    pyside_widgets[wdg]=obj;
    QObject::connect(wdg, &QObject::destroyed, nCore::self(), [wdg]() {
        if (pyside_widgets.count(wdg)==0 || !Py_IsInitialized()) return; // (cleared by _cleanup)
        nb::gil_scoped_acquire gil;
        auto found=pyside_widgets.find(wdg);
        if (found==pyside_widgets.end()) return;
        // the other python references raise instead of using a deleted widget
        try {
            nb::module_::import_("shiboken6").attr("invalidate")(found->second);
        } catch (nb::python_error &e) {
            e.discard_as_unraisable("neutrino: invalidating the PySide widget");
        }
        pyside_widgets.erase(found);
    });
    return obj;
}

// ------------------------------------------------------------------ QVariant <-> python

static nb::object fromVariant(const QVariant &val) {
    if (!val.isValid()) return nb::none();
    const QMetaType type=val.metaType();
    switch (type.id()) {
        case QMetaType::Bool: return nb::bool_(val.toBool());
        case QMetaType::Int: case QMetaType::Long: case QMetaType::LongLong: case QMetaType::Short:
            return nb::int_(val.toLongLong());
        case QMetaType::UInt: case QMetaType::ULong: case QMetaType::ULongLong: case QMetaType::UShort:
            return nb::int_(val.toULongLong());
        case QMetaType::Double: case QMetaType::Float: return nb::float_(val.toDouble());
        case QMetaType::QString: return nb::str(val.toString().toUtf8().constData());
        case QMetaType::QByteArray: return nb::bytes(val.toByteArray().constData(), val.toByteArray().size());
        case QMetaType::QStringList: {
            nb::list ret;
            for (auto &s : val.toStringList()) ret.append(nb::str(s.toUtf8().constData()));
            return ret;
        }
        case QMetaType::QVariantList: {
            nb::list ret;
            for (auto &v : val.toList()) ret.append(fromVariant(v));
            return ret;
        }
        case QMetaType::QPointF: case QMetaType::QPoint: {
            QPointF p=val.toPointF();
            return nb::make_tuple(p.x(), p.y());
        }
        case QMetaType::QSizeF: case QMetaType::QSize: {
            QSizeF s=val.toSizeF();
            return nb::make_tuple(s.width(), s.height());
        }
        case QMetaType::QRectF: case QMetaType::QRect: {
            QRectF r=val.toRectF();
            return nb::make_tuple(r.x(), r.y(), r.width(), r.height());
        }
        default: break;
    }
    if (type==QMetaType::fromType<nPhysD*>()) return toPython(val.value<nPhysD*>());
    if (type==QMetaType::fromType<QList<QList<qreal>>>()) {
        nb::list ret;
        for (auto &row : val.value<QList<QList<qreal>>>()) {
            nb::list r;
            for (auto x : row) r.append(nb::float_(x));
            ret.append(r);
        }
        return ret;
    }
    if (type.flags() & QMetaType::PointerToQObject) {
        QObject *obj=val.value<QObject*>();
        if (nGenericPan *pan=qobject_cast<nGenericPan*>(obj)) return toPython(pan);
        if (neutrino *neu=qobject_cast<neutrino*>(obj)) return nb::cast(PyWindow{neu});
        if (obj==nullptr) return nb::none();
    }
    if (val.canConvert<QString>()) return nb::str(val.toString().toUtf8().constData());
    throw std::runtime_error(std::string("cannot convert a ")+type.name()+" to python");
}

static QVariant toVariant(nb::handle obj) {
    if (obj.is_none()) return QVariant();
    if (nb::isinstance<nb::bool_>(obj)) return QVariant(nb::cast<bool>(obj));
    if (nb::isinstance<nb::int_>(obj)) return QVariant(nb::cast<long long>(obj));
    if (nb::isinstance<nb::float_>(obj)) return QVariant(nb::cast<double>(obj));
    if (nb::isinstance<nb::str>(obj)) return QVariant(QString::fromStdString(nb::cast<std::string>(obj)));
    if (nb::isinstance<PyImage>(obj)) return QVariant::fromValue(nb::cast<PyImage&>(obj).get());
    if (nb::isinstance<PyPan>(obj)) return QVariant::fromValue(static_cast<QObject*>(nb::cast<PyPan&>(obj).get()));
    if (nb::isinstance<PyWindow>(obj)) return QVariant::fromValue(static_cast<QObject*>(nb::cast<PyWindow&>(obj).get()));
    if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        QVariantList ret;
        for (nb::handle item : obj) ret.append(toVariant(item));
        return ret;
    }
    // numpy scalars and other numbers
    if (PyIndex_Check(obj.ptr())) return QVariant(nb::cast<long long>(nb::int_(obj)));
    if (PyFloat_Check(obj.ptr()) || nb::hasattr(obj, "__float__")) return QVariant(nb::cast<double>(nb::float_(obj)));
    throw nb::type_error(("cannot convert " + nb::cast<std::string>(nb::repr(obj)) + " for Qt").c_str());
}

// ------------------------------------------------------------------ meta-object calls

// calls the slot / Q_INVOKABLE method "name" of obj whose parameters accept the python arguments
// (the most derived class first; the methods with default arguments have one entry per number of arguments)
static nb::object invokeMethod(QObject *obj, const std::string &name, nb::args args) {
    const QMetaObject *mo=obj->metaObject();
    std::string errors;
    for (int i=mo->methodCount()-1; i>=0; i--) {
        QMetaMethod method=mo->method(i);
        if (method.name()!=name.c_str() || method.parameterCount()!=int(args.size())) continue;
        if (method.methodType()!=QMetaMethod::Slot && method.methodType()!=QMetaMethod::Method) continue;
        if (method.parameterCount()>10) continue;

        std::vector<QVariant> values;
        bool ok=true;
        for (int j=0; j<method.parameterCount(); j++) {
            QVariant val;
            try {
                val=toVariant(args[j]);
            } catch (const std::exception &) {
                ok=false;
                break;
            }
            QMetaType ptype=method.parameterMetaType(j);
            if (ptype==QMetaType::fromType<QVariant>()) {
                val=QVariant::fromValue(val); // the method takes the QVariant itself
            // } else if (val.metaType()!=ptype && !val.convert(ptype)) {
            } else if (!convertTo(val, ptype)) {
                ok=false;
                break;
            }
            values.push_back(val);
        }
        if (!ok) {
            errors += "\n  " + std::string(method.methodSignature().constData());
            continue;
        }

        QMetaType rtype=method.returnMetaType();
        bool hasReturn=rtype.isValid() && rtype.id()!=QMetaType::Void;
        QVariant ret= hasReturn ? QVariant(rtype, nullptr) : QVariant();
        void *argv[11];
        argv[0]= hasReturn ? ret.data() : nullptr;
        for (size_t j=0; j<values.size(); j++) argv[j+1]=values[j].data();
        // the method may run a Qt event loop (e.g. a calculation thread): python code of other slots can run
        QMetaObject::metacall(obj, QMetaObject::InvokeMetaMethod, i, argv);
        if (rtype==QMetaType::fromType<QVariant>()) return fromVariant(*static_cast<QVariant*>(ret.data()));
        return hasReturn ? fromVariant(ret) : nb::none();
    }
    if (errors.empty()) {
        throw nb::attribute_error((std::string(mo->className())+" has no method "+name+" with "+std::to_string(args.size())+" arguments").c_str());
    }
    throw nb::type_error((std::string("wrong argument types for ")+mo->className()+"::"+name+", candidates:"+errors).c_str());
}

static bool hasMethod(QObject *obj, const std::string &name) {
    const QMetaObject *mo=obj->metaObject();
    for (int i=0; i<mo->methodCount(); i++) {
        QMetaMethod method=mo->method(i);
        if (method.name()==name.c_str() && (method.methodType()==QMetaMethod::Slot || method.methodType()==QMetaMethod::Method)) return true;
    }
    return false;
}

// attribute lookup of Window and Pan when the name is not a bound method: Qt methods and properties
static nb::object metaGetattr(QPointer<QObject> obj, const std::string &name) {
    if (!obj) throw std::runtime_error("the object was deleted");
    if (hasMethod(obj, name)) {
        return nb::cpp_function([obj, name](nb::args args) -> nb::object {
            if (!obj) throw std::runtime_error("the object was deleted");
            return invokeMethod(obj, name, args);
        }, nb::name(name.c_str()));
    }
    int idx=obj->metaObject()->indexOfProperty(name.c_str());
    if (idx>=0) return fromVariant(obj->metaObject()->property(idx).read(obj));
    throw nb::attribute_error((std::string(obj->metaObject()->className())+" has no attribute "+name).c_str());
}

// QVariant::convert, and (x, y) to the Qt points and sizes (Qt doesn't convert a list): the signals give them as tuples
static bool convertTo(QVariant &val, QMetaType type) {
    if (val.metaType()==type) return true;
    const int id=type.id();
    if ((id==QMetaType::QPointF || id==QMetaType::QPoint || id==QMetaType::QSizeF || id==QMetaType::QSize) &&
        val.metaType().id()==QMetaType::QVariantList) {
        QVariantList list=val.toList();
        bool okx=false, oky=false;
        double x=list.size()==2 ? list[0].toDouble(&okx) : 0, y=list.size()==2 ? list[1].toDouble(&oky) : 0;
        if (!okx || !oky) return false;
        switch (id) {
            case QMetaType::QPointF: val=QPointF(x, y); break;
            case QMetaType::QPoint: val=QPoint(qRound(x), qRound(y)); break;
            case QMetaType::QSizeF: val=QSizeF(x, y); break;
            default: val=QSize(qRound(x), qRound(y)); break;
        }
        return true;
    }
    return val.convert(type);
}

static void setQtProperty(QObject *obj, const std::string &name, nb::handle value) {
    int idx=obj->metaObject()->indexOfProperty(name.c_str());
    QVariant val=toVariant(value);
    if (idx>=0) {
        QMetaProperty prop=obj->metaObject()->property(idx);
        // if (!val.convert(prop.metaType()) || !prop.write(obj, val)) {
        if (!convertTo(val, prop.metaType()) || !prop.write(obj, val)) {
            throw nb::type_error(("cannot set the property "+name).c_str());
        }
    } else {
        obj->setProperty(name.c_str(), val); // dynamic property (e.g. the NeuSave- ones)
    }
}

// ------------------------------------------------------------------ signals

// a python callable connected to a Qt signal: called with the arguments it accepts (like PySide)
struct PyCallback {
    nb::object fn;
    size_t nargs;
    PyCallback(nb::object f) : fn(std::move(f)), nargs(std::numeric_limits<size_t>::max()) {
        try {
            nb::object params=nb::module_::import_("inspect").attr("signature")(fn).attr("parameters").attr("values")();
            size_t n=0;
            for (nb::handle p : params) {
                int kind=nb::cast<int>(p.attr("kind"));
                if (kind==2) return;          // *args: everything
                if (kind==0 || kind==1) n++;  // positional
            }
            nargs=n;
        } catch (nb::python_error &e) {
            e.restore();
            PyErr_Clear(); // no signature (builtins): everything
        }
    }
    ~PyCallback() {
        if (Py_IsInitialized()) {
            nb::gil_scoped_acquire gil;
            fn.reset();
        } else {
            fn.release();
        }
    }
    void operator()(std::vector<nb::object> args) {
        if (args.size()>nargs) args.resize(nargs);
        nb::tuple targs=nb::steal<nb::tuple>(PyTuple_New(args.size()));
        // for (size_t i=0; i<args.size(); i++) PyTuple_SET_ITEM(targs.ptr(), i, args[i].release().ptr());
        // (PyTuple_SetItem: the macro is not in the stable ABI of the wheel)
        for (size_t i=0; i<args.size(); i++) PyTuple_SetItem(targs.ptr(), i, args[i].release().ptr());
        try {
            fn(*targs);
        } catch (nb::python_error &e) {
            // don't throw into Qt: printed as unraisable, like an exception in a PySide slot
            e.discard_as_unraisable(fn);
        }
    }
};

// connections made from python, to disconnect them by callable
struct PyConnection {
    QPointer<QObject> obj;
    std::string signal;
    std::shared_ptr<PyCallback> callback;
    QMetaObject::Connection connection;
};
static std::vector<PyConnection> py_connections;

template <typename Sender, typename... Args, typename Conv>
static QMetaObject::Connection connectSignal(Sender *sender, void (Sender::*signal)(Args...), std::shared_ptr<PyCallback> cb, Conv conv) {
    return QObject::connect(sender, signal, sender, [cb, conv, sender](Args... args) {
        nb::gil_scoped_acquire gil;
        (*cb)(conv(sender, args...));
    });
}

using SignalConnector = std::function<QMetaObject::Connection(neutrino*, std::shared_ptr<PyCallback>)>;

// the signals of neutrino available from python, with the conversion of their arguments
static const std::map<std::string, SignalConnector>& windowSignals() {
    static const std::map<std::string, SignalConnector> window_signals = {
        {"bufferChanged", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::bufferChanged, cb,
            [](neutrino *n, nPhysD *p) { return std::vector<nb::object>{toPython(p, n)}; }); }},
        {"physAdd", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::physAdd, cb,
            [](neutrino *n, nPhysD *p) { return std::vector<nb::object>{toPython(p, n)}; }); }},
        {"physDel", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::physDel, cb,
            [](neutrino *n, nPhysD *p) { return std::vector<nb::object>{toPython(p, n)}; }); }},
        {"physChanged", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::physChanged, cb,
            [](neutrino *n, nPhysD *p) { return std::vector<nb::object>{toPython(p, n)}; }); }},
        {"physReordered", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::physReordered, cb,
            [](neutrino *) { return std::vector<nb::object>{}; }); }},
        {"panAdd", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::panAdd, cb,
            [](neutrino *, nGenericPan *p) { return std::vector<nb::object>{toPython(p)}; }); }},
        {"panDel", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::panDel, cb,
            [](neutrino *, nGenericPan *p) { return std::vector<nb::object>{toPython(p)}; }); }},
        {"mouseAtMatrix", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::mouseAtMatrix, cb,
            [](neutrino *, QPointF p) { return std::vector<nb::object>{nb::make_tuple(p.x(), p.y())}; }); }},
        {"mouseAtWorld", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::mouseAtWorld, cb,
            [](neutrino *, QPointF p) { return std::vector<nb::object>{nb::make_tuple(p.x(), p.y())}; }); }},
        {"nZoom", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::nZoom, cb,
            [](neutrino *, double z) { return std::vector<nb::object>{nb::float_(z)}; }); }},
        {"colorValue", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::colorValue, cb,
            [](neutrino *, double v) { return std::vector<nb::object>{nb::float_(v)}; }); }},
        {"closeAll", [](neutrino *n, std::shared_ptr<PyCallback> cb) { return connectSignal(n, &neutrino::closeAll, cb,
            [](neutrino *) { return std::vector<nb::object>{}; }); }},
        {"destroyed", [](neutrino *n, std::shared_ptr<PyCallback> cb) {
            // no python object of the window any more: no argument
            return QObject::connect(n, &QObject::destroyed, [cb]() { nb::gil_scoped_acquire gil; (*cb)({}); }); }},
    };
    return window_signals;
}

struct PySignal {
    QPointer<neutrino> neu;
    std::string name;

    void connect(nb::object fn) {
        if (!neu) throw std::runtime_error("the window was closed");
        if (!PyCallable_Check(fn.ptr())) throw nb::type_error("connect needs a callable");
        auto cb=std::make_shared<PyCallback>(fn);
        QMetaObject::Connection conn=windowSignals().at(name)(neu, cb);
        py_connections.push_back({neu.data(), name, cb, conn});
    }
    // without argument: all the python connections of this signal
    bool disconnect(nb::object fn) {
        bool found=false;
        for (auto it=py_connections.begin(); it!=py_connections.end();) {
            if (!it->obj) {
                it=py_connections.erase(it);
                continue;
            }
            if (it->obj==neu.data() && it->signal==name && (fn.is_none() || it->callback->fn.equal(fn))) {
                QObject::disconnect(it->connection);
                it=py_connections.erase(it);
                found=true;
            } else {
                it++;
            }
        }
        return found;
    }
};

// ------------------------------------------------------------------ module

// NB_MODULE(neutrino, m) {
// neutrino._neutrino: neutrino/__init__.py imports PySide6 first (its Qt is the one of the module)
NB_MODULE(_neutrino, m) {
    m.doc()="Neutrino image analysis in python: windows, images (numpy arrays) and pans (plugins)";
#ifndef __phys_debug
    // PySide keeps the python callables of its pending timers and connections (and what they refer to, e.g. a
    // neutrino.Window) after the end of python: harmless, but nanobind would print "leaked instances" at the exit
    nb::set_leak_warnings(false);
#endif

    m.def("app", []() {
        if (!QCoreApplication::instance()) {
            own_app=new QApplication(app_argc, app_argv);
        }
        checkApplication();
    }, "Creates the QApplication if there is none (to use Neutrino without PySide6)");

    m.def("exec", []() {
        QApplication *qapp=checkApplication();
        nb::gil_scoped_release release; // the slots acquire it
        return qapp->exec();
    }, "Runs the Qt event loop until the last window is closed (without PySide6: with PySide6 use app.exec())");

    m.def("windows", []() {
        checkApplication();
        std::vector<PyWindow> ret;
        for (auto &neu : nCore::self()->neus()) ret.push_back(PyWindow{neu});
        return ret;
    }, "The Neutrino windows");

    m.def("plugins", []() {
        std::vector<std::string> ret;
        for (auto &plug : nPanPlug::all()) ret.push_back(plug.name.toStdString());
        return ret;
    }, "The names of the plugins (Window.openPan)");

    // at the exit of python, while Qt is still there: otherwise the windows (and the objects waiting for deleteLater)
    // are deleted by the static destructors of Qt, after its GUI data
    m.def("_cleanup", []() {
        py_connections.clear();
        pyside_widgets.clear();
        if (QCoreApplication::instance()==nullptr) return;
        // like neutrino::fileClose, without the questions: the pans first (they have items in the scene of the window)
        for (auto &neu : nCore::self()->neus()) {
            for (auto &pan : neu->getPanList()) pan->close();
        }
        QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
        for (auto &neu : nCore::self()->neus()) {
            neu->saveDefaults();
            delete neu;
        }
        QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
        nCore::self()->deleteLogWin();
        if (own_app) {
            delete own_app;
            own_app=nullptr;
        }
    });
    nb::module_::import_("atexit").attr("register")(m.attr("_cleanup"));

    m.def("version", []() { return std::string(SHAVERSION__); }, "The git sha of Neutrino");
    m.def("qtVersion", []() { return std::string(qVersion()); }, "The version of the Qt library of Neutrino");

    // ------------------------------------------------------------ Signal
    nb::class_<PySignal>(m, "Signal", "A signal of a Neutrino window: connect(callable), disconnect(callable=None)")
        .def("connect", &PySignal::connect, "slot"_a)
        .def("disconnect", &PySignal::disconnect, "slot"_a=nb::none())
        .def("__repr__", [](const PySignal &s) { return "<neutrino.Signal " + s.name + ">"; });

    // ------------------------------------------------------------ Image
    nb::class_<PyImage>(m, "Image", "An image (buffer) of a Neutrino window; numpy arrays are copied in and out")
        .def_prop_ro("valid", [](const PyImage &img) { return img.neu && img.neu->nPhysExists(img.phys); })
        .def_prop_ro("shape", [](const PyImage &img) {
            nPhysD *p=img.get();
            return std::make_pair(p->getH(), p->getW());
        }, "(height, width): the shape of the array")
        .def_prop_rw("name", [](const PyImage &img) { return img.get()->getName(); },
            [](PyImage &img, const std::string &name) { img.get()->setName(name); img.neu->emitPhysChanged(img.phys); })
        .def_prop_rw("shortName", [](const PyImage &img) { return img.get()->getShortName(); },
            [](PyImage &img, const std::string &name) { img.get()->setShortName(name); img.neu->emitPhysChanged(img.phys); })
        .def_prop_rw("origin", [](const PyImage &img) {
                vec2f o=img.get()->get_origin();
                return std::make_pair(o.x(), o.y());
            }, [](PyImage &img, std::pair<double,double> o) {
                img.get()->set_origin(vec2f(o.first, o.second));
                img.neu->emitPhysChanged(img.phys);
            }, "(x, y) of the origin, in pixels")
        .def_prop_rw("scale", [](const PyImage &img) {
                vec2f s=img.get()->get_scale();
                return std::make_pair(s.x(), s.y());
            }, [](PyImage &img, std::pair<double,double> s) {
                img.get()->set_scale(vec2f(s.first, s.second));
                img.neu->emitPhysChanged(img.phys);
            }, "(x, y) size of a pixel")
        .def("array", [](const PyImage &img) { return toArray(*img.get()); }, "A copy of the pixels: array[y, x]")
        .def("__array__", [](const PyImage &img, nb::object dtype, nb::object) {
                nb::object arr=nb::cast(toArray(*img.get()));
                return dtype.is_none() ? arr : arr.attr("astype")(dtype);
            }, "dtype"_a=nb::none(), "copy"_a=nb::none())
        .def("update", [](PyImage &img, ArrayIn arr) {
                fromArray(*img.get(), arr);
                img.changed();
            }, "array"_a, "Replaces the pixels (the shape can change)")
        .def("show", [](PyImage &img) { img.neu->showPhys(img.get()); }, "Makes it the visible image of its window")
        // the properties of the image (anymap, e.g. from the file; the PyShell properties/getProperty/setProperty)
        .def("properties", [](const PyImage &img) {
                std::vector<std::string> ret;
                for (auto &it : img.get()->prop) ret.push_back(it.first);
                return ret;
            }, "The names of the properties of the image")
        // (-> nb::object: in nanobind 3 nb::none() is a different type)
        .def("property", [](const PyImage &img, const std::string &name) -> nb::object {
                nPhysD *phys=img.get();
                auto it=phys->prop.find(name);
                if (it==phys->prop.end()) return nb::none();
                return fromVariant(::toVariant(it->second));
            }, "name"_a, "A property of the image (None if it has none of that name)")
        .def("setProperty", [](PyImage &img, const std::string &name, nb::handle value) {
                QVariant val=toVariant(value);
                img.get()->prop[name]=::toAnydata(val);
                img.changed();
            }, "name"_a, "value"_a, "Sets a property of the image")
        // .def("close", [](PyImage &img) { img.neu->closeBuffer(img.get()); })
        .def("close", [](PyImage &img, bool ask) {
                nPhysD *phys=img.get();
                QPointer<neutrino> neu=img.neu;
                bool old=neu->askCloseUnsaved;
                neu->askCloseUnsaved=ask;
                neu->closeBuffer(phys);
                if (neu) neu->askCloseUnsaved=old;
            }, nb::arg("ask")=false,
            "Closes the image; ask=True: asks to save it if it was not saved (like the GUI)")
        .def("window", [](const PyImage &img) { img.get(); return PyWindow{img.neu}; })
        .def("__eq__", [](const PyImage &a, const PyImage &b) { return a.phys==b.phys; })
        .def("__hash__", [](const PyImage &img) { return std::hash<void*>()(img.phys); })
        .def("__repr__", [](const PyImage &img) {
            if (!img.neu || !img.neu->nPhysExists(img.phys)) return std::string("<neutrino.Image closed>");
            return "<neutrino.Image '" + img.phys->getShortName() + "' " + std::to_string(img.phys->getH()) + "x" + std::to_string(img.phys->getW()) + ">";
        });

    // ------------------------------------------------------------ Pan
    nb::class_<PyPan>(m, "Pan", "A pan (plugin window). Besides these methods, the slots, Q_INVOKABLE methods "
                                "and Q_PROPERTY of the plugin are available by name")
        .def_prop_ro("valid", [](const PyPan &p) { return !p.pan.isNull(); })
        .def_prop_ro("name", [](const PyPan &p) { return p.get()->panName().toStdString(); })
        .def("set", [](PyPan &p, const std::string &widget, nb::handle value, int occurrence) {
                p.get()->set(QString::fromStdString(widget), toVariant(value), occurrence);
            }, "widget"_a, "value"_a, "occurrence"_a=1, "Sets the value of the widget with that object name")
        .def("get", [](PyPan &p, const std::string &widget, int occurrence) {
                return fromVariant(p.get()->get(QString::fromStdString(widget), occurrence));
            }, "widget"_a, "occurrence"_a=1, "The value of the widget with that object name")
        .def("button", [](PyPan &p, const std::string &widget, int occurrence) {
                p.get()->button(QString::fromStdString(widget), occurrence);
            }, "widget"_a, "occurrence"_a=1, "Clicks the button with that object name (returns when its calculation is done)")
        .def("getData", [](PyPan &p, const std::string &widget, int occurrence) {
                return fromVariant(QVariant::fromValue(p.get()->getData(QString::fromStdString(widget), occurrence)));
            }, "widget"_a, "occurrence"_a=1, "The data of a plot")
        .def("widgets", [](PyPan &p) {
                std::vector<std::string> ret;
                for (auto &w : p.get()->widgets()) ret.push_back(w.toStdString());
                return ret;
            }, "The object names of the widgets (for set, get, button)")
        .def("window", [](PyPan &p) { return PyWindow{p.get()->nparent}; })
        .def("close", [](PyPan &p) { if (p.pan) p.pan->close(); })
        .def("raise_", [](PyPan &p) { p.get()->raiseIt(); })
        .def("widget", [](PyPan &p) { return pysideWidget(p.get(), "QMainWindow"); }, "The PySide6 QMainWindow of the pan")
        .def("setProperty", [](PyPan &p, const std::string &name, nb::handle value) { setQtProperty(p.get(), name, value); })
        .def("property", [](PyPan &p, const std::string &name) { return fromVariant(p.get()->property(name.c_str())); })
        .def("__getattr__", [](PyPan &p, const std::string &name) {
                if (name.rfind("__", 0)==0) throw nb::attribute_error(name.c_str());
                return metaGetattr(QPointer<QObject>(p.get()), name);
            })
        .def("__eq__", [](const PyPan &a, const PyPan &b) { return a.pan==b.pan; })
        .def("__hash__", [](const PyPan &p) { return std::hash<void*>()(p.pan.data()); })
        .def("__repr__", [](const PyPan &p) {
            return p.pan ? "<neutrino.Pan '" + p.pan->panName().toStdString() + "'>" : std::string("<neutrino.Pan closed>");
        });

    // ------------------------------------------------------------ Window
    auto window = nb::class_<PyWindow>(m, "Window", "A Neutrino window. Besides these methods, its slots are available by name")
        .def("__init__", [](PyWindow *self) {
                checkApplication();
                // new (self) PyWindow{new neutrino()};
                // a script must not wait for a dialog, also when app.quit() closes the window (askCloseUnsaved)
                neutrino *neu=new neutrino();
                neu->askCloseUnsaved=false;
                new (self) PyWindow{neu};
            }, "A new Neutrino window (needs a QApplication)")
        .def_prop_ro("valid", [](const PyWindow &w) { return !w.neu.isNull(); })
        .def_prop_rw("askCloseUnsaved", [](const PyWindow &w) { return w.get()->askCloseUnsaved; },
            [](PyWindow &w, bool ask) { w.get()->askCloseUnsaved=ask; },
            "Closing an image or the window asks to save the images not saved (like the GUI). False for the windows "
            "made in python (also when app.quit() closes them), True for the others")
        .def("addPhys", [](PyWindow &w, ArrayIn arr, const std::string &name) {
                nPhysD *phys=new nPhysD(physD(0, 0, 0.0, name.empty() ? "python" : name));
                fromArray(*phys, arr);
                phys->setShortName(name.empty() ? "python" : name);
                phys->TscanBrightness();
                w.get()->addPhys(phys);
                return PyImage{w.neu, phys};
            }, "array"_a, "name"_a="", "Adds a copy of the 2D array (array[y, x]) as an image")
        .def("addShowPhys", [](PyWindow &w, ArrayIn arr, const std::string &name) {
                nPhysD *phys=new nPhysD(physD(0, 0, 0.0, name.empty() ? "python" : name));
                fromArray(*phys, arr);
                phys->setShortName(name.empty() ? "python" : name);
                phys->TscanBrightness();
                w.get()->addShowPhys(phys);
                return PyImage{w.neu, phys};
            }, "array"_a, "name"_a="", "Adds a copy of the 2D array (array[y, x]) as an image and shows it")
        .def("currentBuffer", [](PyWindow &w) -> nb::object {
                nPhysD *p=w.get()->getCurrentBuffer();
                return p ? nb::cast(toArray(*p)) : nb::none();
            }, "A copy of the pixels of the visible image (None if no image)")
        .def("currentImage", [](PyWindow &w) { return toPython(w.get()->getCurrentBuffer(), w.neu); }, "The visible image")
        .def("images", [](PyWindow &w) {
                std::vector<PyImage> ret;
                for (auto &p : w.get()->getBufferList()) ret.push_back(PyImage{w.neu, p});
                return ret;
            }, "The images of the window")
        .def("fileOpen", [](PyWindow &w, const std::string &fname) {
                std::vector<PyImage> ret;
                for (auto &p : w.get()->fileOpen(QString::fromStdString(fname))) ret.push_back(PyImage{w.neu, p});
                return ret;
            }, "filename"_a, "Opens a file (all the formats of Neutrino), returns its images")
        .def("fileSave", [](PyWindow &w, PyImage &img, const std::string &fname) {
                return w.get()->fileSave(img.get(), QString::fromStdString(fname));
            }, "image"_a, "filename"_a)
        .def("openPan", [](PyWindow &w, const std::string &name) {
                return toPython(w.get()->openPan(QString::fromStdString(name), false));
            }, "name"_a, "Opens a pan (neutrino.plugins() for the names), None if not found")
        .def("getPan", [](PyWindow &w, const std::string &name, int occurrence) {
                return toPython(w.get()->getPan(QString::fromStdString(name), occurrence));
            }, "name"_a, "occurrence"_a=1, "An open pan")
        .def("pans", [](PyWindow &w) {
                std::vector<PyPan> ret;
                for (auto &p : w.get()->getPanList()) ret.push_back(PyPan{p});
                return ret;
            }, "The open pans")
        // .def("close", [](PyWindow &w) { if (w.neu) w.neu->close(); })
        .def("close", [](PyWindow &w, bool ask) {
                if (!w.neu) return;
                QPointer<neutrino> neu=w.neu;
                bool old=neu->askCloseUnsaved;
                neu->askCloseUnsaved=ask;
                neu->close();
                if (neu) neu->askCloseUnsaved=old;
            }, nb::arg("ask")=false,
            "Closes the window; ask=True: asks to save the images not saved (like the GUI)")
        .def("raise_", [](PyWindow &w) { w.get()->raise(); w.get()->activateWindow(); })
        .def("widget", [](PyWindow &w) { return pysideWidget(w.get(), "QMainWindow"); }, "The PySide6 QMainWindow of the window")
        .def("setProperty", [](PyWindow &w, const std::string &name, nb::handle value) { setQtProperty(w.get(), name, value); })
        .def("property", [](PyWindow &w, const std::string &name) { return fromVariant(w.get()->property(name.c_str())); })
        .def("__getattr__", [](PyWindow &w, const std::string &name) {
                if (name.rfind("__", 0)==0) throw nb::attribute_error(name.c_str());
                return metaGetattr(QPointer<QObject>(w.get()), name);
            })
        .def("__eq__", [](const PyWindow &a, const PyWindow &b) { return a.neu==b.neu; })
        .def("__hash__", [](const PyWindow &w) { return std::hash<void*>()(w.neu.data()); })
        .def("__repr__", [](const PyWindow &w) {
            return w.neu ? "<neutrino.Window " + w.neu->property("winId").toString().toStdString() + ">" : std::string("<neutrino.Window closed>");
        });

    for (auto &sig : windowSignals()) {
        const std::string &name=sig.first;
        window.def_prop_ro(name.c_str(), [name](PyWindow &w) { w.get(); return PySignal{w.neu, name}; });
    }
}
