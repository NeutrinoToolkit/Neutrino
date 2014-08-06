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
#include "nHDF5.h"
#include "neutrino.h"
#include "nPhysFormats.h"

#include "hdf5.h"
#include "hdf5_hl.h"


#define HDF5_MAX_NAME 2048

nHDF5::nHDF5(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);
	my_w.treeWidget->setColumnHidden((my_w.treeWidget->columnCount()-1),true);
	
	neutrinoProperties << "fileHDF";
	decorate();
	
	connect(my_w.actionOpen, SIGNAL(triggered()), this, SLOT(showFile()));
	connect(my_w.actionClose, SIGNAL(triggered()), this, SLOT(removeFile()));
	connect(my_w.actionCopy, SIGNAL(triggered()), this, SLOT(copyPath()));
	connect(my_w.treeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(openData(QTreeWidgetItem*, int)));
	connect(my_w.treeWidget, SIGNAL(itemPressed(QTreeWidgetItem*, int)), this, SLOT(itemEntered(QTreeWidgetItem*, int)));
	
}

void nHDF5::copyPath(){
	QString clipText;
	foreach(QTreeWidgetItem *item, my_w.treeWidget->selectedItems()) {
		clipText+="openFile(\""+getFilename(item)+"\",\""+item->data(3,0).toString()+"\") ";
	}
	QApplication::clipboard()->setText(clipText);
}

void nHDF5::removeFile(){
	foreach(QTreeWidgetItem *item, my_w.treeWidget->selectedItems()) {
		while (item->parent()) {
			item=item->parent();
		}
		delete item;
	}
}

void nHDF5::itemEntered(QTreeWidgetItem *item, int) {
	my_w.statusBar->showMessage(item->data(3,0).toString(),5000);
}

QString nHDF5::getFilename(QTreeWidgetItem *item) {
	while (item->parent()) {
		item=item->parent();
	}
	return item->data(3,0).toString();
}

void nHDF5::openData(QTreeWidgetItem *item, int) {
	QString dataName=item->data(3,0).toString();
	nparent->showPhys(phys_open_HDF5(getFilename(item).toStdString(),dataName.toStdString()));
}

void nHDF5::showFile(QString fname) {
	my_w.statusBar->showMessage("Parsing "+fname);
	QApplication::processEvents();
	for (int i=0;i<my_w.treeWidget->topLevelItemCount();i++) {
		QTreeWidgetItem *item=my_w.treeWidget->topLevelItem(i);
		if (item->data(3,0).toString()==fname) {
			delete item;
		}
	}
	hid_t faplist_id=H5Pcreate(H5P_FILE_ACCESS);
	hid_t file_id = H5Fopen (fname.toUtf8(), H5F_ACC_RDONLY, faplist_id);
	if (file_id>0) {
		nparent->updateRecentFileActions(fname);
		hid_t grp = H5Gopen(file_id,"/", H5P_DEFAULT);
		QTreeWidgetItem *item=new QTreeWidgetItem(my_w.treeWidget,QStringList(QFileInfo(fname).fileName()));
		item->setData(1,0,"File");
		item->setData(3,0,fname);
		scanGroup(grp,item);
		my_w.treeWidget->header()->setResizeMode(QHeaderView::ResizeToContents);
		H5Gclose(grp);
		H5Fclose(file_id);
		my_w.statusBar->showMessage("Done",2000);
	} else {
		statusBar()->showMessage("Problem opening file",5000);
	}
}

void nHDF5::showFile() {
	QStringList fnames=QFileDialog::getOpenFileNames(this,tr("Open HDF file source"),property("fileHDF").toString(),tr("HDF5")+QString(" (*.h5);;")+tr("Any files")+QString(" (*)"));
	foreach (QString fname, fnames) {
		showFile(fname);
	}
	if (!fnames.isEmpty()) setProperty("fileHDF", fnames);
}

void nHDF5::scanDataset(hid_t did, QTreeWidgetItem *item2) {
    for (int i = 0; i < H5Aget_num_attrs(did); i++) {
		hid_t aid =	H5Aopen_idx(did, (unsigned int)i );
		scanAttribute(aid, item2);
		H5Aclose(aid);
    }
	DEBUG(11,"BEGIN DATASET");
	ssize_t sizeName=1+H5Iget_name(did, NULL,0);
	char *ds_name=new char[sizeName];
	H5Iget_name(did, ds_name,sizeName);	
	hid_t sid = H5Dget_space(did);
	hid_t tid = H5Dget_type(did);
	
	H5T_class_t t_class = H5Tget_class(tid);
	
	hid_t nativeType;
	int ndims=0;
	hsize_t *dims=NULL;
	
	ndims=H5Sget_simple_extent_ndims(sid);
	DEBUG(11,"NAME" << ds_name << " " << ndims);
	if (t_class == H5T_FLOAT) {
		item2->setData(1,0,"DS Float");
		dims=new hsize_t[ndims];
		H5Sget_simple_extent_dims(sid,dims,NULL);
		nativeType=H5Tget_native_type(tid,H5T_DIR_DEFAULT);
	} if (t_class == H5T_INTEGER) {
		QString title="DS Integer "+QString::number(ndims)+"D ";
		dims=new hsize_t[ndims];
		H5Sget_simple_extent_dims(sid,dims,NULL);
		for (int d=ndims-1;d>=0;d--) title += QString::number(dims[d])+"x";
		title.chop(1);
		item2->setData(1,0,title);
		nativeType=H5Tget_native_type(tid,H5T_DIR_DEFAULT);		
	} else if(t_class == H5T_ARRAY) {
		//! TODO: check this
		QString title="DS Array "+QString::number(ndims)+"D ";
		dims=new hsize_t[ndims];
		H5Sget_simple_extent_dims(sid,dims,NULL);
		for (int d=ndims-1;d>=0;d--) title += QString::number(dims[d])+"x";
		title.chop(1);
		item2->setData(1,0,title);
		delete[] dims; 
		dims=NULL;
		
		item2->setData(1,0,title);
		ndims = H5Tget_array_ndims(tid);
		dims=new hsize_t[ndims];
		H5Tget_array_dims2(tid, dims);
		nativeType=H5Tget_native_type(H5Tget_super(tid),H5T_DIR_DEFAULT);
	} else if(t_class == H5T_COMPOUND) {
		
		DEBUG(11,"*** BEGIN COMPOUND ***");
		int size=H5Tget_size(tid);
		
		ndims=H5Sget_simple_extent_ndims(sid);
		DEBUG(11,"ndims: " << ndims);
		dims=new hsize_t[ndims];
		H5Sget_simple_extent_dims(sid,dims,NULL);
		for (int pippo=0;pippo<ndims;pippo++) {
			size*=dims[pippo];
			DEBUG(11,"dims[" << pippo << "]=" << dims[pippo]);
		}
		
		delete[] dims;
		dims=NULL;
		
		
		item2->setData(1,0,"DS Compound");
		int nCompund=H5Tget_nmembers(tid);
		DEBUG(11,"nCompound " << nCompund << " size: " << size );
		item2->setData(2,0,QString::number(nCompund)+" objs");
		char *buffer=new char[size*nCompund];
		hid_t dataread=H5Dread(did, tid, sid, H5S_ALL, H5P_DEFAULT, buffer);
		DEBUG(11,"did " << did << " tid "<< tid << " sid "<< sid << " dataread " << dataread);
		if (dataread>=0) {
			int position=0;
			for (int i=0; i< nCompund ; i++) {
				char *compoundName=H5Tget_member_name (tid,i);
				QTreeWidgetItem *item3=new QTreeWidgetItem(item2,QStringList(QString(compoundName)));
				item3->setFlags(item3->flags()|Qt::ItemIsEditable);
				hid_t compoundType=H5Tget_member_type(tid,i);
				size_t compundOffset=H5Tget_member_offset(tid,i);
				hid_t classCompoundType=H5Tget_class(compoundType);
				hid_t nativeType = H5Tget_native_type(compoundType,H5T_DIR_DEFAULT);
				
				if (H5Tequal(nativeType,H5T_NATIVE_DOUBLE)) {
					double *val=(double*)(&(buffer[compundOffset]));
					item3->setData(1,0,"double");
					item3->setData(2,0,QString::number(*val));
				} else if (H5Tequal(nativeType,H5T_NATIVE_FLOAT)) {
					float *val=(float*)(&(buffer[compundOffset]));
					item3->setData(1,0,"float");
					item3->setData(2,0,QString::number(*val));
				} else if (H5Tequal(nativeType,H5T_NATIVE_INT)) {
					int *val=(int*)(&(buffer[compundOffset]));
					item3->setData(1,0,"int");
					item3->setData(2,0,QString::number(*val));
				} else if (H5Tequal(nativeType,H5T_NATIVE_USHORT)) {
					unsigned short *val=(unsigned short*)(&(buffer[compundOffset]));
					item3->setData(1,0,"ushort");
					item3->setData(2,0,QString::number(*val));
				} else if (H5Tequal(nativeType,H5T_NATIVE_SHORT)) {
					short *val=(short*)(&(buffer[compundOffset]));
					item3->setData(1,0,"short");
					item3->setData(2,0,QString::number(*val));
				} else if (H5Tequal(nativeType,H5T_NATIVE_UINT)) {
					unsigned int *val=(unsigned int*)(&(buffer[compundOffset]));
					item3->setData(1,0,"uint");
					item3->setData(2,0,QString::number(*val));
				} else if (H5Tequal(nativeType,H5T_NATIVE_LONG)) {
					long int *val=(long int*)(&(buffer[compundOffset]));
					item3->setData(1,0,"long int");
					item3->setData(2,0,QString::number(*val));
				} else if (H5Tequal(nativeType,H5T_NATIVE_UCHAR)) {
					unsigned char *val=(unsigned char*)(&(buffer[compundOffset]));
					item3->setData(1,0,"uchar");
					item3->setData(2,0,QString::number((int)*val));
				} else if (H5Tequal(nativeType,H5T_NATIVE_CHAR)) {
					char *val=(char*)(&(buffer[compundOffset]));
					item3->setData(1,0,"char");
					item3->setData(2,0,QString::number((int)*val));
				} else if (classCompoundType==H5T_STRING) {
					item3->setData(1,0,"String");
					item3->setData(2,0,QString::number(H5Tget_size(nativeType)));
				} else if (classCompoundType==H5T_COMPOUND) {
					item3->setData(1,0,"Compound");
					item3->setData(2,0,QString::number(H5Tget_size(nativeType)));
				} else if (classCompoundType==H5T_ENUM) {
					item3->setData(1,0,"Enum");
					item3->setData(2,0,QString::number(H5Tget_size(nativeType)));
				} else if (classCompoundType==H5T_VLEN) {
					item3->setData(1,0,"Vlen");
					item3->setData(2,0,QString::number(H5Tget_size(nativeType)));
				} else {
					item3->setData(1,0,"unknown "+QString::number(classCompoundType));
					item3->setData(2,0,QString::number(H5Tget_size(nativeType)));
				}
				
				position+=H5Tget_size(nativeType);
				delete[] compoundName;
			}
		}
		delete[] buffer;
		DEBUG(11,"***  END COMPOUND  ***");
	} else {
		item2->setData(1,0,"DS "+QString::number(t_class));					
		item2->setData(2,0,"Ask for support!");					
	}
	
	QString strKind=QString::number(ndims)+"D ";
	if (ndims==2) {
		strKind+=QString::number(dims[1])+"x"+QString::number(dims[0]);
		QTreeWidgetItem *dataParent=item2->parent();
		while (dataParent) {
			dataParent->setExpanded(true);
			dataParent->setForeground(0,QBrush(QColor(0,0,200)));
			dataParent=dataParent->parent();
		}
		item2->setExpanded(true);
		item2->setData(2,0,strKind);
		item2->setData(3,0,ds_name);
		item2->setForeground(0,QBrush(QColor(0,0,200)));
		item2->setForeground(1,QBrush(QColor(0,0,200)));
		item2->setForeground(2,QBrush(QColor(0,0,200)));
	}	
	
	delete[] ds_name;
	if (dims) delete[] dims;
	H5Tclose(tid);
	H5Sclose(sid);
	DEBUG(11,"  END DATASET");
}

void nHDF5::scanAttribute(hid_t aid, QTreeWidgetItem *parentItem, nPhysD *my_data) {
	ssize_t len = 1+H5Aget_name(aid, 0, NULL );
	vector<char> attrName(len);
	H5Aget_name(aid, len, &attrName[0] );
	
	QTreeWidgetItem *item3=new QTreeWidgetItem(parentItem,QStringList(QString(&attrName[0])));
	item3->setFlags(item3->flags()|Qt::ItemIsEditable);
	hid_t aspace = H5Aget_space(aid); /* the dimensions of the attribute data */
	hid_t atype  = H5Aget_type(aid);
	H5A_info_t aInfo;
	H5Aget_info(aid, &aInfo);
	hid_t nativeType = H5Tget_native_type(atype,H5T_DIR_DEFAULT);
	hid_t classAType=H5Tget_class(atype);
	if (classAType ==  H5T_FLOAT) {
		if (H5Tequal(nativeType,H5T_NATIVE_DOUBLE)) {
			int nelem=aInfo.data_size/sizeof(double);
			item3->setData(1,0,"Attr double");
			vector<double> val(nelem);
			if (H5Aread(aid, nativeType, (void*)(&val[0])) >= 0) {
				if (my_data && nelem==2) {
					if (strcmp(&attrName[0],"physOrigin")==0) {
						my_data->set_origin(val[0],val[1]);
					} else if (strcmp(&attrName[0],"physScale")==0) {
						my_data->set_scale(val[0],val[1]);
					}
				}
				QString strData=QString::number(val[0]);
				for (int i=1;i<nelem;i++) strData+=" "+QString::number(val[i]);
				item3->setData(2,0,strData);
			}
		}
	} else if (classAType ==  H5T_INTEGER) {
		int nelem=aInfo.data_size/sizeof(int);
		item3->setData(1,0,"Attr int");
		vector<int> val(nelem);
		if (H5Aread(aid, nativeType, (void*)(&val[0])) >= 0) {
			QString strData=QString::number(val[0]);
			for (int i=1;i<nelem;i++) strData+=" "+QString::number(val[i]);
			item3->setData(2,0,strData);
		}
	} else if (classAType == H5T_STRING) {
		vector<char> val;
		item3->setData(1,0,"Attr string");
		if (my_data) {
			int sizeStr=1+aInfo.data_size;
			val.resize(sizeStr);
			if (H5Aread(aid, nativeType, &val[0]) >= 0) {
				if (my_data) {
					if (strcmp(&attrName[0],"physShortName")==0) my_data->setShortName(string(&val[0]));
					if (strcmp(&attrName[0],"physName")==0) my_data->setName(string(&val[0]));
				}
				item3->setData(2,0,QString(&val[0]));
			}
		} else {
			hssize_t ssiz=H5Sget_simple_extent_npoints(aspace);
			size_t tsiz=H5Tget_size(atype);
			int sizeStr=ssiz*tsiz;
			//			hsize_t dims=0;
			if( sizeStr >= 0) {
				val.resize(sizeStr);		
				if (H5Aread(aid, nativeType, (&val[0])) >= 0) {
					if (H5Tis_variable_str(atype)) {
						item3->setData(2,0,QString(*((char**) (&val[0]))));					
					} else {
						item3->setData(2,0,QString((&val[0])));
					}
				}
			}
			
		}
	}
	H5Tclose(atype);
	H5Sclose(aspace);
}

void nHDF5::scanGroup(hid_t gid, QTreeWidgetItem *parentItem) {
	ssize_t sizeName=1+H5Iget_name(gid, NULL,0);
	char *group_name=new char[sizeName];
	H5Iget_name(gid, group_name,sizeName);	
	
	for (int i = 0; i < H5Aget_num_attrs(gid); i++) {
		hid_t aid =	H5Aopen_idx(gid, (unsigned int)i );
		scanAttribute(aid,parentItem);
		H5Aclose(aid);
	}
	//	DEBUG(11,group_name);
	
	H5G_info_t infoGroup;
	H5Gget_info(gid,&infoGroup);
	for (hsize_t i = 0; i < infoGroup.nlinks; i++) {
		int size = 1+H5Lget_name_by_idx (gid, ".", H5_INDEX_NAME, H5_ITER_INC,i, NULL, 0, H5P_DEFAULT);
		char *memb_name=new char[size];
		H5Lget_name_by_idx (gid, ".", H5_INDEX_NAME, H5_ITER_INC,i, memb_name, size, H5P_DEFAULT);
		
		//		DEBUG(11,group_name << "/" << memb_name << " " << i);
		//		QApplication::processEvents();
		
		QTreeWidgetItem *item2=new QTreeWidgetItem(parentItem,QStringList(QString(memb_name)));
		switch(H5Gget_objtype_by_idx(gid, i)) {
			case H5G_LINK:
				item2->setData(1,0,"Link");
				herr_t status;
				char target[HDF5_MAX_NAME];
				status = H5Gget_linkval(gid, memb_name, HDF5_MAX_NAME, target  ) ;
				item2->setData(2,0,QString(target));
				break;
			case H5G_GROUP: {
				hid_t grpId = H5Gopen(gid,memb_name, H5P_DEFAULT);
				scanGroup(grpId,item2);
				H5Gclose(grpId);
				break;
			}
			case H5G_TYPE: {
				hid_t typeId = H5Topen(gid,memb_name, H5P_DEFAULT);
				item2->setData(1,0,H5Tget_class(typeId) < 0?"Type invalid":"Type valid");
				item2->setData(2,0,"Don't know what to do with this!");
				H5Tclose(typeId);
				break;
			}
			case H5G_DATASET: {				
				hid_t did = H5Dopen(gid, memb_name, H5P_DEFAULT);
				scanDataset(did, item2);
				H5Dclose(did);
				break;
			}
			default:
				item2->setData(1,0,"Unknown");
				break;
		}
		delete[] memb_name;
	}
	delete[] group_name;
	QApplication::processEvents();
}


nPhysD* nHDF5::phys_open_HDF5(string fileName, string dataName) {
	nPhysD *my_data=NULL;
	DEBUG(fileName << " " << dataName);
	hid_t fid = H5Fopen (fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	if (fid >= 0) {
		hid_t did = H5Dopen(fid,dataName.c_str(), H5P_DEFAULT);
		if (did>=0) {
			ssize_t sizeName=1+H5Iget_name(did, NULL,0);
			char *ds_name=new char[sizeName];
			H5Iget_name(did, ds_name,sizeName);	
			hid_t sid = H5Dget_space(did);
			hid_t tid = H5Dget_type(did);
			
			size_t tsiz = H5Tget_size(tid);
			H5T_class_t t_class = H5Tget_class(tid);
			
			hid_t nativeType;
			char *buffer = NULL;
			int ndims=0;
			hsize_t *dims=NULL;
			
			hid_t file_space_id=H5S_ALL;
			int narray=1;
			if (t_class == H5T_FLOAT) {
				ndims=H5Sget_simple_extent_ndims(sid);
				dims=new hsize_t[ndims];
				H5Sget_simple_extent_dims(sid,dims,NULL);
				int npoints=H5Sget_simple_extent_npoints(sid);
				buffer = new char[tsiz*npoints];
				nativeType=H5Tget_native_type(tid,H5T_DIR_DEFAULT);
			} else if (t_class == H5T_COMPOUND) {
				DEBUG(10, "compound");
			} else if(t_class == H5T_ARRAY) {
				ndims=H5Sget_simple_extent_ndims(sid);
				DEBUG("ndims: " << ndims);
				dims=new hsize_t[ndims];
				H5Sget_simple_extent_dims(sid,dims,NULL);
				for (int pippo=0;pippo<ndims;pippo++) {
					narray*=dims[pippo];
					DEBUG("dims[" << pippo << "]=" << dims[pippo]);
				}
				
				delete dims;
				
				
				ndims = H5Tget_array_ndims(tid);
				dims=new hsize_t[ndims];
				H5Tget_array_dims2(tid, dims);
				buffer = new char[tsiz*narray];
				DEBUG(5,"H5T_ARRAY " << dims[0] << " " << dims[1] << " tsiz " << tsiz << " narray " << narray);
				DEBUG(5,"ALLOCATED " << tsiz * narray);
				nativeType=H5Tget_native_type(H5Tget_super(tid),H5T_DIR_DEFAULT);
			}
			if (buffer && ndims==2) {
				my_data=new nPhysD(ds_name);
				my_data->setType(PHYS_FILE);
				string strName(ds_name);
				strName.erase(0,strName.find_last_of('/'));
				my_data->setShortName(strName);
				
				for (int i = 0; i < H5Aget_num_attrs(did); i++) {
					hid_t aid =	H5Aopen_idx(did, (unsigned int)i );
					scan_hdf5_attributes(aid, my_data);
					H5Aclose(aid);
				}
				
				DEBUG("did " << did << " tid "<< tid << " sid "<< sid );
				int status = H5Dread(did, tid, sid, file_space_id, H5P_DEFAULT, buffer);
				DEBUG("status" << status);
				my_data->resize(dims[1],dims[0]);
				for (int na=0;na<narray;na++) {
					DEBUG("na "<< na);
					if (H5Tequal(nativeType,H5T_NATIVE_USHORT)) {
						for (size_t k=0;k<my_data->getSurf();k++) {
							my_data->set(k,my_data->point(k)+((unsigned short*) buffer)[k]);
						}
					} else if (H5Tequal(nativeType,H5T_NATIVE_INT)) {
						for (size_t k=0;k<my_data->getSurf();k++) {
							my_data->set(k,my_data->point(k)+((int*) buffer)[k+na*my_data->getSurf()]);
						}
					} else if (H5Tequal(nativeType,H5T_NATIVE_UINT)) {
						for (size_t k=0;k<my_data->getSurf();k++) {
							my_data->set(k,my_data->point(k)+((unsigned int*) buffer)[k+na*my_data->getSurf()]);
						}
					} else if (H5Tequal(nativeType,H5T_NATIVE_FLOAT)) {
						for (size_t k=0;k<my_data->getSurf();k++) {
							my_data->set(k,my_data->point(k)+((float*) buffer)[k+na*my_data->getSurf()]);
						}
					} else if (H5Tequal(nativeType,H5T_NATIVE_DOUBLE)) {
						for (size_t k=0;k<my_data->getSurf();k++) {
							my_data->set(k,my_data->point(k)+((double*) buffer)[k+na*my_data->getSurf()]);
						}
					}
				}
				for (size_t k=0;k<my_data->getSurf();k++) {
					my_data->set(k,my_data->point(k)/narray);
				}
				
			}
			
			delete [] ds_name;
			delete [] dims;
			delete [] buffer;
			H5Tclose(tid);
			H5Sclose(sid);				
			H5Dclose(did);
			H5Fclose(fid);
		}
	}
	if (my_data) my_data->TscanBrightness();
	return my_data;
}


int nHDF5::phys_write_HDF5(nPhysImageF<double> *phys, string fname) {
	if (phys) {
		if (H5Zfilter_avail(H5Z_FILTER_DEFLATE)) {
			unsigned int	filter_info;
			herr_t status = H5Zget_filter_info (H5Z_FILTER_DEFLATE, &filter_info);
			if ( (filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED) && (filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED) ) {
				hid_t file_id = H5Fcreate (fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
				
				hsize_t dims[2]={phys->getH(),phys->getW()};
				hid_t space = H5Screate_simple (2, dims, NULL);
				
				hid_t dcpl = H5Pcreate (H5P_DATASET_CREATE);
				status = H5Pset_deflate (dcpl, 9);
				hsize_t chunk[2] = {4, 8};
				status = H5Pset_chunk (dcpl, 2, chunk);
				
				hid_t dset = H5Dcreate (file_id, "/neutrino", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
				status = H5Dwrite (dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, phys->Timg_buffer);
				
				status= H5LTset_attribute_string(file_id,"/","version", __VER);
				double data[2];
				data[0]=phys->get_origin().x();
				data[1]=phys->get_origin().y();
				status= H5LTset_attribute_double(file_id,"neutrino","physOrigin", data, 2);
				data[0]=phys->get_scale().x();
				data[1]=phys->get_scale().y();
				status= H5LTset_attribute_double(file_id,"neutrino","physScale", data, 2);
				status= H5LTset_attribute_string(file_id,"neutrino","physName", phys->getName().c_str());
				status= H5LTset_attribute_string(file_id,"neutrino","physShortName", phys->getShortName().c_str());
				
				
				status = H5Pclose (dcpl);
				status = H5Dclose (dset);
				status = H5Sclose (space);
				status = H5Fclose (file_id);
			}
		}
		return 0;
	}
	return -1;
}

void nHDF5::scan_hdf5_attributes(hid_t aid, nPhysImageF<double> *my_data){
	ssize_t len = 1+H5Aget_name(aid, 0, NULL );
	char *attrName=new char[len];
	H5Aget_name(aid, len, attrName );
	
	hid_t aspace = H5Aget_space(aid); /* the dimensions of the attribute data */
	hid_t atype  = H5Aget_type(aid);
	H5A_info_t aInfo;
	H5Aget_info(aid, &aInfo);
	hid_t nativeType = H5Tget_native_type(atype,H5T_DIR_DEFAULT);
	hid_t classAType=H5Tget_class(atype);
	if (classAType ==  H5T_FLOAT) {
		if (H5Tequal(nativeType,H5T_NATIVE_DOUBLE)) {
			int nelem=aInfo.data_size/sizeof(double);
			vector<double> val(nelem);
			if (H5Aread(aid, nativeType, (void*)(&val[0])) >= 0) {
				if (my_data && nelem==2) {
					if (strcmp(attrName,"physOrigin")==0) {
						my_data->set_origin(val[0],val[1]);
					} else if (strcmp(attrName,"physScale")==0) {
						my_data->set_scale(val[0],val[1]);
					}
				}
			}
		}
	} else if (classAType ==  H5T_INTEGER) {
		int nelem=aInfo.data_size/sizeof(int);
		vector<int> val(nelem);
		if (H5Aread(aid, nativeType, (void*)(&val[0])) >= 0) {
            my_data->property[string(attrName)]=val[0]; 
		}
	} else if (classAType == H5T_STRING) {
		char *val =NULL;
		if (my_data) {
			int sizeStr=1+aInfo.data_size;
			val=new char[sizeStr];
			if (H5Aread(aid, nativeType, val) >= 0) {
				if (my_data) {
					if (strcmp(attrName,"physShortName")==0) my_data->setShortName(string(val));
					if (strcmp(attrName,"physName")==0) my_data->setName(string(val));
				}
			}
		} else {
			hssize_t ssiz=H5Sget_simple_extent_npoints(aspace);
			size_t tsiz=H5Tget_size(atype);
			int sizeStr=ssiz*tsiz;
			//			hsize_t dims=0;
			if( sizeStr >= 0) {
				val = new char[sizeStr];		
				if (H5Aread(aid, nativeType, val) >= 0) {
					if (H5Tis_variable_str(atype)) {
						// use this *((char**) val);					
					} else {
						// use this val);
					}
				}
			}
			
		}
		delete [] val;
	}
	delete [] attrName;
	H5Tclose(atype);
	H5Sclose(aspace);
}
