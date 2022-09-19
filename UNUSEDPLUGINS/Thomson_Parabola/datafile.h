/*
 * Class for reading matrix-shaped datafiles
 *
 * (C) Alessandro Flacco 2008
 */

#include <iostream>
#include <fstream>
#include <vector>

#ifndef __datafile_h
#define __datafile_h

using namespace std;

class datafile {

public:
	datafile()
	{ }

	datafile(const char *filename, int n)
		: myfilename(filename), ncols(n)
	{
		dataspace = new double *[ncols];
//		for (int i=0; i<ncols; i++)
//			dataspace[i] = new vector<double>;
	}

	~datafile()
	{ }

	void readfile()
	{

		int readline = 0, rcol, readvalue = 0;
		double readval;
		bool bailout = false;
	
		ifile.open(myfilename.c_str());
	
		// count file lines
		int linecount = 0;
		string iline;
	
		while (getline(ifile, iline)) {
			linecount++;
		}
		ifile.clear();
		ifile.seekg(0,ios::beg);
	
	
		// resize vectors
		for (int i=0; i<ncols; i++) 
			dataspace[i] = new double [linecount];
			//((vector<double> *)dataspace[i])->resize(linecount);
	
		// ORRIBILE E OSCENO... (per questa volta non ho voglia di far meglio...)
		while (true) {
			for (rcol = 0; rcol<ncols; rcol++) {
				ifile>>readval;
				if (ifile.eof()) {
					bailout = true;
					break;
				}
				
				readvalue++;
				dataspace[rcol][readline] = readval;
				//cout<<"[datafile] "<<(*((vector<double> *)(dataspace[rcol])))[readline]<<endl;
	
			}
			
			if (bailout)
				break;
			readline++;
		}
	
		if ( (readline*ncols) != readvalue)
			cerr<<"[datafile] Field count mismatch, datafile is invalid!"<<endl;
		
		// resize vectors
		//for (int i=0; i<ncols; i++) 
		//	((vector<double> *)dataspace[i])->resize(readline);
		//
		
		nrows = readline;
	
		cerr<<"[datafile] Read ["<<readline<<","<<ncols<<"] vector"<<endl;
	//	cerr<<"vecsize: "<<((vector<double> *)dataspace[0])->size()<<endl;
	
	}

	double **dataspace;


	// questo lo implemento quando mi verra' voglia..
	void parseFile()
	{ }

	void readFile();

	ifstream ifile;
	string myfilename;
	
	int ncols;
	int nrows;

};

//void
//datafile::readfile()
//{
//	int readline = 0, rcol, readvalue = 0;
//	double readval;
//	bool bailout = false;
//
//	ifile.open(myfilename.c_str());
//
//	// count file lines
//	int linecount = 0;
//	string iline;
//
//	while (getline(ifile, iline)) {
//		linecount++;
//	}
//	ifile.clear();
//	ifile.seekg(0,ios::beg);
//
//
//	// resize vectors
//	for (int i=0; i<ncols; i++) 
//		dataspace[i] = new double [linecount];
//		//((vector<double> *)dataspace[i])->resize(linecount);
//
//	// ORRIBILE E OSCENO... (per questa volta non ho voglia di far meglio...)
//	while (true) {
//		for (rcol = 0; rcol<ncols; rcol++) {
//			ifile>>readval;
//			if (ifile.eof()) {
//				bailout = true;
//				break;
//			}
//			
//			readvalue++;
//			dataspace[rcol][readline] = readval;
//			//cout<<"[datafile] "<<(*((vector<double> *)(dataspace[rcol])))[readline]<<endl;
//
//		}
//		
//		if (bailout)
//			break;
//		readline++;
//	}
//
//	if ( (readline*ncols) != readvalue)
//		cerr<<"[datafile] Field count mismatch, datafile is invalid!"<<endl;
//	
//	// resize vectors
//	//for (int i=0; i<ncols; i++) 
//	//	((vector<double> *)dataspace[i])->resize(readline);
//	//
//	
//	nrows = readline;
//
//	cerr<<"[datafile] Read ["<<readline<<","<<ncols<<"] vector"<<endl;
////	cerr<<"vecsize: "<<((vector<double> *)dataspace[0])->size()<<endl;
//}


#endif
