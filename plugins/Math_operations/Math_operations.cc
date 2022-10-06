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

// physWavelets

Math_operations::Math_operations(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);
    connect(calculate, SIGNAL(pressed()), this, SLOT(doOperation()));
    connect(operation, SIGNAL(currentIndexChanged(int)), this, SLOT(enableGroups(int)));
    operatorResult=nullptr;

    for(int i=0;i<operation->count();i++){
        operation->itemText(i);
        if (operation->itemText(i).startsWith("-")) {
            qDebug() << i;
            separator.push_back(i);
        }
    }
    show();

}

void Math_operations::enableGroups (int num) {
    qDebug() << ">>>>>>>>>>>>>" << num;

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
        if (operation->currentIndex() < separator[0]) { // two images neeeded
            nPhysD *operand1=nullptr;
            nPhysD *operand2=nullptr;

            if (radioNumber1->isChecked() && radioNumber2->isChecked()) {
                statusbar->showMessage("One operand must be an image",5000);
                return;
            }

            if (radioNumber1->isChecked() && my_phys2) {
                double val=QLocale().toDouble(num1->text());
                qDebug() <<"radio 1" << val;
                operand1=new nPhysD(my_phys2->getW(),my_phys2->getH(),val,num1->text().toStdString());
                operand1->set_origin(my_phys2->get_origin());
                operand1->set_scale(my_phys2->get_scale());
            }

            if (radioNumber2->isChecked() && my_phys1) {
                double val=QLocale().toDouble(num2->text());
                qDebug() <<"radio 2" << val;
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
            qDebug() << operation->currentText();

            myresult=new nPhysD(rTot.width(),rTot.height(), 1.0);
            myresult->setName(operation->currentText().toStdString()+" "+operand1->getName()+" "+operand2->getName());
            myresult->setFromName(operand1->getFromName()+" "+operand2->getFromName());
            myresult->set_origin(-rTot.left(),-rTot.top());
            vec2f shift1=-(myresult->get_origin() - operand1->get_origin());
            vec2f shift2=-(myresult->get_origin() - operand2->get_origin());
            for (size_t j=0; j<myresult->getH(); j++) {
                for (size_t i=0; i<myresult->getW(); i++) {
                    double val1=operand1->point(i+shift1.x(),j+shift1.y());
                    double val2=operand2->point(i+shift2.x(),j+shift2.y());
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
                        statusbar->showMessage(tr("ERROR: Expected 2 values"));
                    }
                } else if (operation->currentIndex()==separator[0]+2) {
                    QStringList scalars=num2->text().split(" ");
                    if  (scalars.size()==1){
                        bool ok;
                        double scalar=QLocale().toDouble(num2->text(),&ok);
                        if (ok) {
                            myresult=new nPhysD(*my_phys1);
                            physMath::phys_fast_gaussian_blur(*myresult, scalar);
                        } else {
                            statusbar->showMessage(tr("ERROR: Exepcted a float radius"));
                        }
                    } else if (scalars.size()==2) {
                        bool ok1, ok2;
                        double scalar1=QLocale().toDouble(scalars.at(0),&ok1);
                        double scalar2=QLocale().toDouble(scalars.at(1),&ok2);
                        if(ok1 && ok2) {
                            myresult=new nPhysD(*my_phys1);
                            physMath::phys_fast_gaussian_blur(*myresult, scalar1, scalar2);
                        } else {
                            statusbar->showMessage(tr("ERROR: Exepcted 2 values"));
                        }
                    }
                } else if (operation->currentIndex()==separator[0]+3) {
                    QStringList scalars=num2->text().split(" ");
                    if  (scalars.size()==1){
                        bool ok;
                        int scalar=num2->text().toInt(&ok);
                        if (ok) {
                            myresult=new nPhysD(*my_phys1);
                            physMath::phys_median_filter(*myresult, scalar);
                        } else {
                            statusbar->showMessage(tr("ERROR: Scalar should be an integer"));
                        }
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
                    QStringList scalars=num2->text().split(" ");
                    if (scalars.size()==1) {
                        bool ok1;
                        double scalar1=QLocale().toDouble(scalars.at(0),&ok1);
                        if (ok1) {
                            vec2i newsize= my_phys1->getSize()*scalar1;
                            myresult=new nPhysD(physMath::phys_resample(*my_phys1, newsize));
                        } else {
                            statusbar->showMessage(tr("ERROR: Exepcted 1 double"));
                        }
                    } else if (scalars.size()==2) {
                        bool ok1, ok2;
                        int scalar1=QLocale().toInt(scalars.at(0),&ok1);
                        int scalar2=QLocale().toInt(scalars.at(1),&ok2);
                        if (ok1 && ok2) {
                            myresult=new nPhysD(physMath::phys_resample(*my_phys1, vec2i(scalar1, scalar2)));
                        } else {
                            statusbar->showMessage(tr("ERROR: Exepcted 2 int sperated by space"));
                        }
                    } else {
                        statusbar->showMessage(tr("ERROR: Exepcted 2 values"));
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
                } else if (operation->currentIndex()==separator[0]+11) { //add noise
                    qDebug() << "here";
                    QRegularExpression myexp("^\\s*(\\d+)\\s*x\\s*(\\d+)\\s*\\+(\\d+)\\s*\\+(\\d+)\\s*$|^\\s*(\\d+)\\s*x\\s*(\\d+)\\s*$");



                    qDebug() << myexp;
                    QString my_vals=num2->text();
                    qDebug() << my_vals;
                    QRegularExpressionMatch my_match=myexp.match(my_vals);
                    qDebug() << my_match.lastCapturedIndex();
                    if (my_match.lastCapturedIndex()==2 || my_match.lastCapturedIndex()==4) {
                        QVector<int> vals;
                        for (int i=1;i<=my_match.lastCapturedIndex(); i++) {
                             vals.push_back(my_match.captured(i).toInt());
                        }
                        while (vals.size()<4) {
                            vals.push_back(0);
                        }
                        qDebug() << vals;
                        myresult=new nPhysD(*my_phys1);
                        physMath::phys_crop(*myresult,vals[0],vals[1],vals[2],vals[3]);
                    } else {
                        statusbar->showMessage(tr("ERROR: need 2 or 4 numbers of the form WxH or WxH+X+Y"));
                    }
                }
            }

        } else { // 1 image and 1(or more) scalars
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
            }
        }
        qDebug() << "here";
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
   qDebug() << "here";
}

