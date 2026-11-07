/*
 *
 *    Copyright (C) 2013 Alessand Flacco, Tommaso Vinci All Rights Reserved
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
#include "neutrino.h"
#include "Image_list.h"
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>

Image_list::Image_list(neutrino *nparent) : nGenericPan(nparent) {
    setupUi(this);

    images->header()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
    images->header()->setStretchLastSection (true);
    images->setupSorting();

    // all connections die with this pan: no need to disconnect in the destructor
    connect(nparent, &neutrino::bufferChanged, this, &Image_list::updatePad);
    connect(nparent, &neutrino::physAdd, this, &Image_list::physAdd);
    connect(nparent, &neutrino::physDel, this, &Image_list::physDel);
    connect(nparent, &neutrino::physChanged, this, &Image_list::physChanged);
    connect(nparent, &neutrino::physReordered, this, &Image_list::renumber);

    connect(images, &QTreeWidget::itemSelectionChanged, this, &Image_list::selectionChanged);

    foreach (nPhysD *phys, nparent->getBufferList()) {
        updateItem(phys);
    }
    renumber();
    images->sortByColumn(0,Qt::AscendingOrder);

    connect(actionShort, &QAction::triggered, this, &Image_list::changeProperties);
    connect(actionName, &QAction::triggered, this, &Image_list::changeProperties);
    connect(actionOrigin, &QAction::triggered, this, &Image_list::changeProperties);
    connect(actionScale, &QAction::triggered, this, &Image_list::changeProperties);
    connect(actionRemove, &QAction::triggered, this, &Image_list::buttonRemovePhys);
    connect(actionCopy, &QAction::triggered, this, &Image_list::buttonCopyPhys);
    connect(actionRescale, &QAction::triggered, this, &Image_list::changeProperties);
    connect(actionScalePoly, &QAction::triggered, this, &Image_list::changeScalePoly);

    show(true);
    updatePad(currentBuffer);
}

void Image_list::on_horizontalSlider_valueChanged(int val) {
    nPhysD *to_display=nparent->getBuffer(val);
    if (nPhysExists(to_display)) {
        nparent->showPhys(to_display);
    }
}

void Image_list::selectionChanged() {
    if (updating) return;
    QList<QTreeWidgetItem *> sel=images->selectedItems();
    if (sel.size()) {
        QTreeWidgetItem *item=sel.last();
        nPhysD *my_phys=getPhys(item);
        if (nPhysExists(my_phys)) {
            lineEdit->setText(QString::fromUtf8(my_phys->getFromName().c_str()));
            lineEdit->setCursorPosition(0);
            updating=true; // keep the user's (possibly multiple) selection
            nparent->showPhys(my_phys);
            updating=false;
        } else {
            // stale row: should not happen, but never keep an item without its image
            itemsMap.erase(my_phys);
            delete item;
        }
    }
}

nPhysD* Image_list::getPhys(QTreeWidgetItem* item) {
    return item ? item->data(0,Qt::UserRole).value<nPhysD*>() : nullptr;
}

void Image_list::buttonCopyPhys() {
    foreach (QTreeWidgetItem* item, images->selectedItems()) {
        nPhysD *phys=getPhys(item);
        if (nPhysExists(phys)) {
            nparent->addShowPhys(new nPhysD(*phys));
        }
    }
}

void Image_list::buttonRemovePhys() {
    QList<nPhysD*> my_list;
    for (auto& item: images->selectedItems()) {
        my_list << getPhys(item);
    }
    for (auto& my_phys: my_list) {
        nparent->removePhys(my_phys); // ignores images already gone
    }
}

void Image_list::changeProperties() {
    QList<nPhysD*> physSelected;
    foreach (QTreeWidgetItem* item, images->selectedItems()) {
        nPhysD *phys=getPhys(item);
        if (nPhysExists(phys)) {
            physSelected << phys;
        }
    }
    if (physSelected.isEmpty() && nPhysExists(currentBuffer)) {
        physSelected << currentBuffer;
    }
    if (physSelected.isEmpty()) return;

    // the tree rows are refreshed through neutrino::physChanged
    QTreeWidgetItem *lastItem=updateItem(physSelected.last());
    bool ok;
    QString text;

    if (sender()==actionShort) {
        text = QInputDialog::getText(this, tr("Change Short Name"),tr("Short name:"), QLineEdit::Normal, lastItem->data(1,Qt::DisplayRole).toString(), &ok);
        if (ok && !text.isEmpty()) {
            foreach (nPhysD* phys, physSelected) {
                phys->setShortName(text.toStdString());
                nparent->emitPhysChanged(phys);
            }
        }
    } else if (sender()==actionScale) {
        text = QInputDialog::getText(this, tr("Change Scale"),tr("Scale:"), QLineEdit::Normal, lastItem->data(2,Qt::DisplayRole).toString(), &ok);
        if (ok && !text.isEmpty()) {
            QStringList lista=text.split(' ', Qt::SkipEmptyParts);
            bool ok1=false, ok2=false;
            double xVal=0, yVal=0;
            if (lista.size()==1) {
                xVal=yVal=QLocale().toDouble(lista.at(0),&ok1);
                ok2=ok1;
            } else if (lista.size()==2) {
                xVal=QLocale().toDouble(lista.at(0),&ok1);
                yVal=QLocale().toDouble(lista.at(1),&ok2);
            }
            if (ok1 && ok2) {
                foreach (nPhysD* phys, physSelected) {
                    if (!nPhysExists(phys)) continue;
                    // a linear scale replaces the non-linear axis whose value changes
                    if (xVal!=phys->get_scale().x()) phys->set_scale_poly(PHYS_X, {});
                    if (yVal!=phys->get_scale().y()) phys->set_scale_poly(PHYS_Y, {});
                    phys->set_scale(xVal,yVal);
                    nparent->emitPhysChanged(phys);
                }
            }
        }
    } else if (sender()==actionOrigin) {
        text = QInputDialog::getText(this, tr("Change Origin"),tr("Origin:"), QLineEdit::Normal, lastItem->data(3,Qt::DisplayRole).toString(), &ok);
        if (ok && !text.isEmpty()) {
            QStringList lista=text.split(' ', Qt::SkipEmptyParts);
            if (lista.size()==2) {
                bool ok1,ok2;
                double xOrigin=QLocale().toDouble(lista.at(0),&ok1);
                double yOrigin=QLocale().toDouble(lista.at(1),&ok2);
                if (ok1 && ok2) {
                    foreach (nPhysD* phys, physSelected) {
                        if (!nPhysExists(phys)) continue;
                        phys->set_origin(xOrigin,yOrigin);
                        nparent->emitPhysChanged(phys);
                    }
                }
            }
        }
    } else if (sender()==actionName) {
        text = QInputDialog::getText(this, tr("Change Name"),tr("Name:"), QLineEdit::Normal, lastItem->data(4,Qt::DisplayRole).toString(), &ok);
        if (ok && !text.isEmpty()) {
            foreach (nPhysD* phys, physSelected) {
                if (!nPhysExists(phys)) continue;
                phys->setName(text.toStdString());
                nparent->emitPhysChanged(phys);
            }
        }
    } else if (sender()==actionRescale) {
        vec2i lastsize=physSelected.last()->getSize();
        std::stringstream ss;
        ss << lastsize.x() << " " << lastsize.y();
        text = QInputDialog::getText(this, tr("Change Size"),tr("Size:"), QLineEdit::Normal, QString::fromStdString(ss.str()), &ok);
        if (ok && !text.isEmpty()) {
            QStringList lista=text.split(' ', Qt::SkipEmptyParts);
            if (lista.size()==1) {
                bool ok1;
                double scale=QLocale().toDouble(lista.at(0),&ok1);
                if (ok1 && scale>0) {
                    foreach (nPhysD* phys, physSelected) {
                        if (!nPhysExists(phys)) continue;
                        vec2i newsize= phys->getSize()*scale;
                        nparent->addShowPhys(new nPhysD(physMath::phys_resample(*phys, newsize)));
                    }
                }
            } else if (lista.size()==2) {
                bool ok1,ok2;
                int sizex=QLocale().toInt(lista.at(0),&ok1);
                int sizey=QLocale().toInt(lista.at(1),&ok2);
                if (ok1 && ok2 && sizex>0 && sizey>0) {
                    foreach (nPhysD* phys, physSelected) {
                        if (!nPhysExists(phys)) continue;
                        nparent->addShowPhys(new nPhysD(physMath::phys_resample(*phys, vec2i(sizex, sizey))));
                    }
                }
            }
        }
    }
}

// the non-linear axes: world = P(p) - P(origin), P a polynomial of the absolute pixel along each axis
void Image_list::changeScalePoly() {
    QList<nPhysD*> physSelected;
    foreach (QTreeWidgetItem* item, images->selectedItems()) {
        nPhysD *phys=getPhys(item);
        if (nPhysExists(phys)) physSelected << phys;
    }
    if (physSelected.isEmpty() && nPhysExists(currentBuffer)) physSelected << currentBuffer;
    if (physSelected.isEmpty()) return;

    auto toText=[](const std::vector<double> &c) {
        QStringList l;
        for (double v : c) l << QLocale().toString(v,'g',12);
        return l.join("  ");
    };
    nPhysD *last=physSelected.last();
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Non-linear scale"));
    QFormLayout *form=new QFormLayout(&dialog);
    QLabel *help=new QLabel(tr("<b>P(p) = c0 + c1·p + c2·p² + ...</b> of the pixel p, counted from the left/top "
                               "border.<br>World coordinate: P(p) with the origin at 0, otherwise "
                               "P(p) − P(origin) + c0.<br>Empty: linear axis. Degree 1: sets the scale and the origin."
                               "<br><i>The rulers, the mouse and the lineouts use the polynomial, the other tools "
                               "its mean slope (≈ in the list).</i>"), &dialog);
    help->setWordWrap(true);
    form->addRow(help);
    QLineEdit *ex=new QLineEdit(toText(last->get_scale_poly(PHYS_X)), &dialog);
    QLineEdit *ey=new QLineEdit(toText(last->get_scale_poly(PHYS_Y)), &dialog);
    ex->setPlaceholderText(tr("linear: %1").arg(QLocale().toString(last->get_scale().x())));
    ey->setPlaceholderText(tr("linear: %1").arg(QLocale().toString(last->get_scale().y())));
    ex->setToolTip(tr("Separated by spaces or ;"));
    ey->setToolTip(ex->toolTip());
    form->addRow(tr("x(px)"), ex);
    form->addRow(tr("y(px)"), ey);
    QDialogButtonBox *buttons=new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    form->addRow(buttons);
    dialog.resize(420, dialog.sizeHint().height());
    if (dialog.exec()!=QDialog::Accepted) return;

    auto parse=[](QString t, bool &ok) {
        std::vector<double> c;
        ok=true;
        for (const QString &s : t.split(QRegularExpression("[\\s;]+"), Qt::SkipEmptyParts)) {
            bool okv;
            double v=QLocale().toDouble(s,&okv);
            if (!okv) v=QLocale::c().toDouble(s,&okv);
            if (!okv) { ok=false; break; }
            c.push_back(v);
        }
        return c;
    };
    bool okx, oky;
    std::vector<double> cx=parse(ex->text(),okx), cy=parse(ey->text(),oky);
    if (!okx || !oky) {
        Ui::Image_list::statusBar->showMessage(tr("Not a list of numbers"),5000);
        return;
    }
    foreach (nPhysD* phys, physSelected) {
        if (!nPhysExists(phys)) continue;
        bool okSet=phys->set_scale_poly(PHYS_X, cx);
        okSet = phys->set_scale_poly(PHYS_Y, cy) && okSet;
        if (!okSet) Ui::Image_list::statusBar->showMessage(tr("Not monotonic over the image: axis not changed"),5000);
        nparent->emitPhysChanged(phys);
    }
}

void Image_list::keyPressEvent(QKeyEvent *e){
    switch (e->key()) {
        case Qt::Key_Return:
        case Qt::Key_Enter: {
            QList<QTreeWidgetItem *> sel=images->selectedItems();
            if (sel.size()) {
                nparent->showPhys(getPhys(sel.first())); // showPhys ignores images not in the list
            }
            break;
        }
        case Qt::Key_Backspace:
        case Qt::Key_Delete:
            buttonRemovePhys();
            break;
    }
    e->accept();
}

// the row of an image (created if missing), filled from the image
QTreeWidgetItem* Image_list::updateItem(nPhysD *my_phys) {
    if (!nPhysExists(my_phys)) return nullptr;
    QTreeWidgetItem* it=nullptr;
    auto found=itemsMap.find(my_phys);
    if (found!=itemsMap.end() && found->second) {
        it=found->second;
    } else {
        const bool wasUpdating=updating;
        updating=true;
        it=new QTreeWidgetItem(images, QTreeWidgetItem::UserType);
        it->setData(0,Qt::UserRole,QVariant::fromValue(my_phys));
        itemsMap[my_phys]=it;
        updating=wasUpdating;
    }
    it->setData(1,Qt::DisplayRole,QString::fromUtf8(my_phys->getShortName().c_str()));
    if (my_phys->get_scale().x()==my_phys->get_scale().y()) {
        it->setData(2,Qt::DisplayRole,QLocale().toString(my_phys->get_scale().x()));
    } else {
        it->setData(2,Qt::DisplayRole,QLocale().toString(my_phys->get_scale().x())+" "+QLocale().toString(my_phys->get_scale().y()));
    }
    // non-linear axes: the scale is their mean slope (≈), the polynomials in the tooltip
    const bool px=my_phys->has_scale_poly(PHYS_X), py=my_phys->has_scale_poly(PHYS_Y);
    if (px || py) {
        it->setData(2,Qt::DisplayRole,"≈ "+QLocale().toString(my_phys->get_scale().x())+" "+QLocale().toString(my_phys->get_scale().y()));
        QString tip=tr("Non-linear scale (mean slope shown)");
        if (px) tip+="\nx: "+QString::fromStdString(physPoly::str(my_phys->get_scale_poly(PHYS_X)));
        if (py) tip+="\ny: "+QString::fromStdString(physPoly::str(my_phys->get_scale_poly(PHYS_Y)));
        it->setToolTip(2,tip);
    } else {
        it->setToolTip(2,QString());
    }
    it->setData(3,Qt::DisplayRole,QLocale().toString(my_phys->get_origin().x())+" "+QLocale().toString(my_phys->get_origin().y()));
    it->setData(4,Qt::DisplayRole,QString::fromUtf8(my_phys->getName().c_str()));
    return it;
}

// first column: position in neutrino's list
void Image_list::renumber() {
    const QList<nPhysD*> list=nparent->getBufferList();
    const int width=list.size() ? 1+int(log10(list.size())) : 1;
    for (int k=0;k<list.size();k++){
        auto found=itemsMap.find(list[k]);
        if (found!=itemsMap.end() && found->second) {
            QString mynum=QString::number(k).rightJustified(width, ' ');
            if (mynum!=found->second->data(0,Qt::DisplayRole).toString()) {
                found->second->setData(0,Qt::DisplayRole,mynum);
            }
        }
    }
    horizontalSlider->setMaximum(std::max(0,int(list.size())-1));
    horizontalSlider->setVisible(list.size()>1);
}

// visible image changed
void Image_list::updatePad(nPhysD *my_phys) {
    if (!nPhysExists(my_phys)) return;
    QTreeWidgetItem* it=updateItem(my_phys);
    lineEdit->setText(QString::fromUtf8(my_phys->getFromName().c_str()));
    lineEdit->setCursorPosition(0);
    if (!updating) {
        // selection follows the image shown (unless the change comes from our own selection)
        updating=true;
        images->clearSelection();
        if (it) {
            it->setSelected(true);
            images->scrollToItem(it);
        }
        horizontalSlider->blockSignals(true);
        horizontalSlider->setValue(nparent->indexOf(my_phys));
        horizontalSlider->blockSignals(false);
        updating=false;
    }
    renumber();
}

void Image_list::physChanged(nPhysD *my_phys) {
    updateItem(my_phys);
}

// emitted by neutrino while my_phys is still valid but already out of the list
void Image_list::physDel(nPhysD *my_phys) {
    auto found=itemsMap.find(my_phys);
    if (found!=itemsMap.end()) {
        const bool wasUpdating=updating;
        updating=true;
        delete found->second;
        itemsMap.erase(found);
        updating=wasUpdating;
    }
    renumber();
    if (itemsMap.empty()) {
        lineEdit->setText(tr("No image"));
    }
}

/// new image entry point
void Image_list::physAdd(nPhysD *my_phys) {
    updateItem(my_phys);
    renumber();
}
