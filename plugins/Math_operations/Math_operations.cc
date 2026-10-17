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
#include "Math_operations.h"
#include "neutrino.h"
#include <QStandardItemModel>

// physWavelets

Math_operations::Math_operations(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);
    // connect(calculate, SIGNAL(pressed()), this, SLOT(doOperation()));
    // clicked: pressed() runs at mouse-down and not with the keyboard
    connect(calculate, SIGNAL(clicked()), this, SLOT(doOperation()));
    connect(operation, SIGNAL(currentIndexChanged(int)), this, SLOT(enableGroups(int)));
    operatorResult=nullptr;

    for(int i=0;i<operation->count();i++){
        // operation->itemText(i);
        if (operation->itemText(i).startsWith("-")) {
            // qDebug() << i;
            separator.push_back(i);
            // the separator lines cannot be selected
            QStandardItemModel *model = qobject_cast<QStandardItemModel *>(operation->model());
            if (model && model->item(i)) model->item(i)->setEnabled(false);
        }
    }
    show();
    // the widgets for the restored operation (enableGroups is called only when the operation changes)
    enableGroups(operation->currentIndex());

}

void Math_operations::enableGroups (int num) {
    // qDebug() << ">>>>>>>>>>>>>" << num;

    first->setEnabled(true);
    second->setEnabled(true);
    radioImage1->setEnabled(true);
    radioImage2->setEnabled(true);
    image1->setEnabled(true);
    image2->setEnabled(true);
    radioNumber1->setEnabled(true);
    radioNumber2->setEnabled(true);
    num1->setEnabled(true);
    num2->setEnabled(true);

    if(num > separator[0] ) {
        if (num < separator[1]) {
            radioImage1->setChecked(true);
            radioNumber1->setEnabled(false);
            num1->setEnabled(false);
            radioNumber2->setChecked(true);
            radioImage2->setEnabled(false);
            image2->setEnabled(false);
        } else {
            radioImage1->setChecked(true);
            radioNumber1->setEnabled(false);
            num1->setEnabled(false);
            second->setEnabled(false);
        }
    }
}

void Math_operations::doOperation () {
    saveDefaults();
    if(currentBuffer) {
        nPhysD *my_phys1=getPhysFromCombo(image1);
        nPhysD *my_phys2=getPhysFromCombo(image2);

        for (int k=0;k<operation->count();k++) {
            DEBUG("Operator " << k << " " << operation->currentIndex() << " " << separator[0] << " " << separator[1] << " " << operation->itemText(k).toStdString());
        }
        DEBUG("Operator " << operation->currentIndex() << " " << separator[0] << " " << separator[1] << " " << operation->currentText().toStdString());
        nPhysD *myresult=nullptr;
        if (std::find(separator.begin(), separator.end(), operation->currentIndex()) != separator.end()) {
            statusbar->showMessage(tr("Select an operation"),5000);
            return;
        }
        // parameters separated by any whitespace (split(" ") gave empty items with double spaces)
        QStringList scalarList=num2->text().split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (operation->currentIndex() < separator[0]) { // two images neeeded
            nPhysD *operand1=nullptr;
            nPhysD *operand2=nullptr;

            if (radioNumber1->isChecked() && radioNumber2->isChecked()) {
                statusbar->showMessage("One operand must be an image",5000);
                return;
            }

            if ((radioImage1->isChecked() && !my_phys1) || (radioImage2->isChecked() && !my_phys2) ||
                (radioNumber1->isChecked() && !my_phys2) || (radioNumber2->isChecked() && !my_phys1)) {
                statusbar->showMessage(tr("Missing image"),5000);
                return;
            }

            if (radioNumber1->isChecked() && my_phys2) {
                double val=QLocale().toDouble(num1->text());
                // qDebug() <<"radio 1" << val;
                operand1=new nPhysD(my_phys2->getW(),my_phys2->getH(),val,num1->text().toStdString());
                operand1->set_origin(my_phys2->get_origin());
                operand1->set_scale(my_phys2->get_scale());
            }

            if (radioNumber2->isChecked() && my_phys1) {
                double val=QLocale().toDouble(num2->text());
                // qDebug() <<"radio 2" << val;
                operand2=new nPhysD(my_phys1->getW(),my_phys1->getH(),val,num2->text().toStdString());
                operand2->set_origin(my_phys1->get_origin());
                operand2->set_scale(my_phys1->get_scale());
            }

            if (radioImage1->isChecked()) {
                operand1=new nPhysD(*my_phys1);
            }
            if (radioImage2->isChecked()) {
                operand2=new nPhysD(*my_phys2);
            }

            if (operand1->get_scale() != operand2->get_scale())  {
                QMessageBox::warning(this,tr("Scale problem"),tr("The images do not have the same scale"),QMessageBox::Ok);
            }

            QRectF r1=QRectF(-operand1->get_origin().x(),-operand1->get_origin().y(),operand1->getW(),operand1->getH());
            QRectF r2=QRectF(-operand2->get_origin().x(),-operand2->get_origin().y(),operand2->getW(),operand2->getH());

            // obs
            QRectF rTot=r1.intersected(r2);;
            // qDebug() << operation->currentText();
            if (rTot.isEmpty()) {
                statusbar->showMessage(tr("The images do not overlap"),5000);
                delete operand1;
                delete operand2;
                return;
            }

            myresult=new nPhysD(rTot.width(),rTot.height(), 1.0);
            myresult->setName(operation->currentText().toStdString()+" "+operand1->getName()+" "+operand2->getName());
            myresult->setFromName(operand1->getFromName()+" "+operand2->getFromName());
            myresult->set_origin(-rTot.left(),-rTot.top());
            vec2f shift1=-(myresult->get_origin() - operand1->get_origin());
            vec2f shift2=-(myresult->get_origin() - operand2->get_origin());
            // integer pixel shifts: point(unsigned,unsigned) got negative doubles (undefined) and truncated them
            vec2i ishift1((int)std::lround(shift1.x()),(int)std::lround(shift1.y()));
            vec2i ishift2((int)std::lround(shift2.x()),(int)std::lround(shift2.y()));
            for (size_t j=0; j<myresult->getH(); j++) {
                for (size_t i=0; i<myresult->getW(); i++) {
                    // double val1=operand1->point(i+shift1.x(),j+shift1.y());
                    // double val2=operand2->point(i+shift2.x(),j+shift2.y());
                    double val1=operand1->point(vec2i(i,j)+ishift1);
                    double val2=operand2->point(vec2i(i,j)+ishift2);
                    if (std::isfinite(val1) && std::isfinite(val2)) {
                        switch (operation->currentIndex()) {
                            case 0:
                                myresult->set(i,j,val1+val2);
                                break;
                            case 1:
                                myresult->set(i,j,val1-val2);
                                break;
                            case 2:
                                myresult->set(i,j,val1*val2);
                                break;
                            case 3:
                                myresult->set(i,j,val1/val2);
                                break;
                            case 4:
                                myresult->set(i,j,0.5*(val1+val2));
                                break;
                            case 5:
                                myresult->set(i,j,std::min(val1,val2));
                                break;
                            case 6:
                                myresult->set(i,j,std::max(val1,val2));
                                break;
                            case 7:
                                myresult->set(i,j,fmod(val1,val2));
                                break;
                        }
                    } else {
                        myresult->set(i,j,std::isfinite(val1)?val1:val2);
                    }

                }
            }
            delete operand1;
            delete operand2;
        } else 	if (operation->currentIndex() < separator[1]) { // 1 image and 1(or more) scalars
            if (my_phys1) {
                if (operation->currentIndex()==separator[0]+1) {
                    bool ok;
                    double scalar=QLocale().toDouble(num2->text(),&ok);
                    if (ok) {
                        myresult=new nPhysD(*my_phys1);
                        physMath::phys_pow(*myresult, scalar);
                    } else {
                        // statusbar->showMessage(tr("ERROR: Expected 2 values"));
                        statusbar->showMessage(tr("ERROR: Value should be a float"));
                    }
                } else if (operation->currentIndex()==separator[0]+2) {
                    // QStringList scalars=num2->text().split(" ");
                    QStringList scalars=scalarList;
                    if  (scalars.size()==1){
                        bool ok;
                        double scalar=QLocale().toDouble(num2->text(),&ok);
                        if (ok) {
                            myresult=new nPhysD(*my_phys1);
                            physMath::phys_fast_gaussian_blur(*myresult, scalar);
                        } else {
                            statusbar->showMessage(tr("ERROR: Expected a float radius"));
                        }
                    } else if (scalars.size()==2) {
                        bool ok1, ok2;
                        double scalar1=QLocale().toDouble(scalars.at(0),&ok1);
                        double scalar2=QLocale().toDouble(scalars.at(1),&ok2);
                        if(ok1 && ok2) {
                            myresult=new nPhysD(*my_phys1);
                            physMath::phys_fast_gaussian_blur(*myresult, scalar1, scalar2);
                        } else {
                            statusbar->showMessage(tr("ERROR: Expected 2 floats"));
                        }
                    } else {
                        statusbar->showMessage(tr("ERROR: Expected 1 or 2 floats"));
                    }
                } else if (operation->currentIndex()==separator[0]+3) {
                    // QStringList scalars=num2->text().split(" ");
                    QStringList scalars=scalarList;
                    if  (scalars.size()==1){
                        bool ok;
                        // int scalar=num2->text().toInt(&ok);
                        int scalar=QLocale().toInt(scalars.at(0),&ok);
                        if (ok) {
                            myresult=new nPhysD(*my_phys1);
                            physMath::phys_median_filter(*myresult, scalar);
                        } else {
                            statusbar->showMessage(tr("ERROR: Scalar should be an integer"));
                        }
                    } else {
                        statusbar->showMessage(tr("ERROR: Expected 1 integer"));
                    }
                } else if (operation->currentIndex()==separator[0]+4) {
                    bool ok;
                    double scalar=QLocale().toDouble(num2->text(),&ok);
                    if (ok) {
                        myresult=new nPhysD(my_phys1->rotated(scalar));
                    } else {
                        statusbar->showMessage(tr("ERROR: Value should be a float"));
                    }
                } else if (operation->currentIndex()==separator[0]+5) {
                    bool ok;
                    double scalar=QLocale().toDouble(num2->text(),&ok);
                    if (ok) {
                        myresult=new nPhysD(*my_phys1);
                        physMath::phys_gauss_laplace(*myresult,scalar);
                    } else {
                        statusbar->showMessage(tr("ERROR: Value should be a float"));
                    }
                } else if (operation->currentIndex()==separator[0]+6) {
                    bool ok;
                    double scalar=QLocale().toDouble(num2->text(),&ok);
                    if (ok) {
                        myresult=new nPhysD(*my_phys1);
                        physMath::phys_gauss_sobel(*myresult,scalar);
                    } else {
                        statusbar->showMessage(tr("ERROR: Value should be a float"));
                    }
                } else if (operation->currentIndex()==separator[0]+7) {
                    bool ok;
                    double scalar=QLocale().toDouble(num2->text(),&ok);
                    if (ok) {
                        myresult=new nPhysD(*my_phys1);
                        physMath::phys_integratedNe(*myresult,scalar);
                    } else {
                        statusbar->showMessage(tr("ERROR: Value should be a float"));
                    }
                } else if (operation->currentIndex()==separator[0]+8) {
                    bool ok;
                    double scalar=QLocale().toDouble(num2->text(),&ok);
                    if (ok) {
                        myresult=new nPhysD(*my_phys1);
                        physMath::phys_remainder(*myresult,scalar);
                    } else {
                        statusbar->showMessage(tr("ERROR: Value should be a float"));
                    }
                } else if (operation->currentIndex()==separator[0]+9) { //Resize
                    // QStringList scalars=num2->text().split(" ");
                    QStringList scalars=scalarList;
                    if (scalars.size()==1) {
                        bool ok1;
                        double scalar1=QLocale().toDouble(scalars.at(0),&ok1);
                        if (ok1) {
                            vec2i newsize= my_phys1->getSize()*scalar1;
                            myresult=new nPhysD(physMath::phys_resample(*my_phys1, newsize));
                        } else {
                            statusbar->showMessage(tr("ERROR: Expected 1 float"));
                        }
                    } else if (scalars.size()==2) {
                        bool ok1, ok2;
                        int scalar1=QLocale().toInt(scalars.at(0),&ok1);
                        int scalar2=QLocale().toInt(scalars.at(1),&ok2);
                        if (ok1 && ok2) {
                            myresult=new nPhysD(physMath::phys_resample(*my_phys1, vec2i(scalar1, scalar2)));
                        } else {
                            statusbar->showMessage(tr("ERROR: Expected 2 integers separated by space"));
                        }
                    } else {
                        statusbar->showMessage(tr("ERROR: Expected 1 float (factor) or 2 integers (size)"));
                    }
                } else if (operation->currentIndex()==separator[0]+10) { //add noise
                    bool ok;
                    double scalar=QLocale().toDouble(num2->text(),&ok);
                    if (ok) {
                        myresult=new nPhysD(*my_phys1);
                        physMath::add_noise(*myresult,scalar);
                    } else {
                        statusbar->showMessage(tr("ERROR: Value should be a float"));
                    }
                } else if (operation->currentIndex()==separator[0]+11) { //crop
                    // qDebug() << "here";
                    // QRegularExpression myexp("^\\s*(\\d+)\\s*x\\s*(\\d+)\\s*\\+(\\d+)\\s*\\+(\\d+)\\s*$|^\\s*(\\d+)\\s*x\\s*(\\d+)\\s*$");
                    // (WxH matched groups 5 and 6: lastCapturedIndex was 6 and WxH was always refused)
                    QRegularExpression myexp("^\\s*(\\d+)\\s*x\\s*(\\d+)\\s*(?:\\+\\s*(\\d+)\\s*\\+\\s*(\\d+)\\s*)?$");

                    // qDebug() << myexp;
                    QString my_vals=num2->text();
                    // qDebug() << my_vals;
                    QRegularExpressionMatch my_match=myexp.match(my_vals);
                    // qDebug() << my_match.lastCapturedIndex();
                    // if (my_match.lastCapturedIndex()==2 || my_match.lastCapturedIndex()==4) {
                    if (my_match.hasMatch()) {
                        QVector<int> vals;
                        // for (int i=1;i<=my_match.lastCapturedIndex(); i++) {
                        for (int i=1;i<=4; i++) {
                             vals.push_back(my_match.captured(i).toInt()); // X and Y not given: empty -> 0
                        }
                        // while (vals.size()<4) {
                        //     vals.push_back(0);
                        // }
                        // qDebug() << vals;
                        myresult=new nPhysD(*my_phys1);
                        physMath::phys_crop(*myresult,vals[0],vals[1],vals[2],vals[3]);
                    } else {
                        statusbar->showMessage(tr("ERROR: need 2 or 4 numbers of the form WxH or WxH+X+Y"));
                    }
                }
            }

        } else { // 1 image and 1(or more) scalars
            if (!my_phys1) {
                statusbar->showMessage(tr("Missing image"),5000);
                return;
            }
            myresult=new nPhysD(*my_phys1);
            myresult->setName(my_phys1->getName());
            if (operation->currentIndex()==separator[1]+1) {
                physMath::phys_transpose(*myresult);
            } else if (operation->currentIndex()==separator[1]+2) {
                physMath::phys_log10(*myresult);
            } else if (operation->currentIndex()==separator[1]+3) {
                physMath::phys_log(*myresult);
            } else if (operation->currentIndex()==separator[1]+4) {
                physMath::phys_abs(*myresult);
            } else if (operation->currentIndex()==separator[1]+5) {
                physMath::phys_square(*myresult);
            } else if (operation->currentIndex()==separator[1]+6) {
                physMath::phys_sqrt(*myresult);
            } else if (operation->currentIndex()==separator[1]+7) {
                physMath::phys_sin(*myresult);
            } else if (operation->currentIndex()==separator[1]+8) {
                physMath::phys_cos(*myresult);
            } else if (operation->currentIndex()==separator[1]+9) {
                physMath::phys_tan(*myresult);
            } else if (operation->currentIndex()==separator[1]+10) {
                physMath::phys_laplace(*myresult);
            } else if (operation->currentIndex()==separator[1]+11) {
                physMath::phys_sobel(*myresult);
            } else if (operation->currentIndex()==separator[1]+12) {
                physMath::phys_sobel_dir(*myresult);
            } else if (operation->currentIndex()==separator[1]+13) {
                physMath::phys_scharr(*myresult);
            } else if (operation->currentIndex()==separator[1]+14) {//TODO: write it better
                // for (unsigned int i=0; i<myresult->getSurf();i++) {
                //     myresult->set(i,0);
                // }
                // for (int p=0; p<nparent->getBufferList().size();p++) {
                //     for (unsigned int i=0; i<myresult->getSurf();i++) {
                //         myresult->set(i,myresult->point(i)+nparent->getBufferList()[p]->point(i));
                //     }
                // }
                // for (unsigned int i=0; i<myresult->getSurf();i++) {
                //     myresult->set(i,myresult->point(i)/nparent->getBufferList().size());
                // }
                // physMath::phys_scharr(*myresult);
                // (the Scharr filter made it an edge map; images of other sizes were mixed by flat index and NaN spread)
                // average of the images with the size of image1, NaN skipped, the previous result excluded
                std::vector<double> sum(myresult->getSurf(),0.0);
                std::vector<int> count(myresult->getSurf(),0);
                int used=0;
                for (nPhysD *img : nparent->getBufferList()) {
                    if (img==operatorResult || img->getW()!=my_phys1->getW() || img->getH()!=my_phys1->getH()) continue;
                    used++;
                    for (unsigned int i=0; i<myresult->getSurf();i++) {
                        double val=img->point(i);
                        if (std::isfinite(val)) {
                            sum[i]+=val;
                            count[i]++;
                        }
                    }
                }
                for (unsigned int i=0; i<myresult->getSurf();i++) {
                    myresult->set(i,count[i] ? sum[i]/count[i] : std::numeric_limits<double>::quiet_NaN());
                }
                statusbar->showMessage(tr("Average of %1 images of size %2x%3").arg(used).arg(my_phys1->getW()).arg(my_phys1->getH()),5000);
            }
        }
        // qDebug() << "here";
        if (myresult) {
            myresult->reset_display();
            myresult->TscanBrightness();
            myresult->setShortName(operation->currentText().toStdString());
            if (myresult->getSurf()>0) {
                erasePrevious->setEnabled(true);
                if (erasePrevious->isChecked()) {
                    operatorResult=nparent->replacePhys(myresult,operatorResult, true);
                } else {
                    nparent->addShowPhys(myresult);
                    operatorResult=myresult;
                }

            }
        }
    }
   // qDebug() << "here";
}

