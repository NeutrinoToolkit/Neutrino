/*
 *
 *	Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 * 
 *	This file is part of nPhysImage library.
 *
 *	nPhysImage is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU Lesser General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	(at your option) any later version.
 *
 *	nPhysImage is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public License
 *	along with neutrino.  If not, see <http://www.gnu.org/licenses/>.
 *
 *	Contact Information: 
 *	Alessandro Flacco <alessandro.flacco@polytechnique.edu>
 *	Tommaso Vinci <tommaso.vinci@polytechnique.edu>
 *
 */
#include "nPhysFormats.h"
#include "NaNStream.h"
#include "bidimvec.h"
#include "nPhysMaths.h"
#include <time.h>
#include <zlib.h>

#ifdef HAVE_LIBTIFF
#define int32 tiff_int32
#define uint32 tiff_uint32
extern "C" {
#include <tiffio.h>
}
#undef int32
#undef uint32
#endif

#ifdef HAVE_LIBCFITSIO
#include "fitsio.h"
#endif

#define CHUNK 0x4000
#define windowBits 15
#define ENABLE_ZLIB_GZIP 32
#define GZIP_ENCODING 16

using namespace std;


physDouble_txt::physDouble_txt(const char *ifilename)
: nPhysImageF<double>(string(ifilename), PHYS_FILE)
{
	clock_t time1, time2, time3;
	time1=clock();
	
	ifstream ifile(ifilename);
	// 1. get image statistics
	string tline;
	int nlines = 0, ncols = 0;
	
	while (getline (ifile, tline)) {
		if ((tline.find('#') == string::npos) && (tline.find_first_not_of(' ') != tline.find_last_not_of(' ')))
			nlines++;
	}
	
	ifile.clear();
	ifile.seekg(0,ios::beg);
	string comment("");
	do {
		getline (ifile, tline);
		comment.append(tline);
	} while ((tline.find('#') != string::npos) || (tline.find_first_not_of(' ') == tline.find_last_not_of(' ')));
	
	stringstream ss(tline);
	string word;
   	while( ss >> word ) ++ncols;
	
	DEBUG(5,"file has "<<nlines<<" lines and "<<ncols<<" cols");
	
	//physImage mycc(ncols, nlines);
	this->resize(ncols, nlines);
	
	// 2. read and save five
	ifile.clear();
	ifile.seekg(0, ios::beg);
	
	time2=clock();
	int row = 0, col = 0;
	int w = getW();
	while (getline(ifile, tline) && row<nlines) {
		col=0;
		stringstream sline(tline);
   		while( sline >> word && col<ncols) {
	   		set(row*w+col, strtod(word.c_str(),NULL));
   			col++;
   		}
		row++;
	}
	
	TscanBrightness();
	time3=clock();
	DEBUG(5,"times: " << time2-time1 << " " << time3-time1);
}

/* This below is the older version, kept just in case */
/* This below is the older version, kept just in case */
/* This below is the older version, kept just in case */
/*
 physDouble_txt::physDouble_txt(const char *ifilename)
 : nPhysImageF<double>(string(ifilename), PHYS_FILE)
 {
 clock_t time1, time2, time3;
 time1=clock();
 
 ifstream ifile(ifilename);
 // 1. get image statistics
 string tline;
 int nlines = 0, ncols = 0;
 
 while (getline (ifile, tline)) {
 
 if ((tline.find('#') == string::npos) && (tline.find_first_not_of(' ') != tline.find_last_not_of(' ')))
 nlines++;
 }
 time2=clock();
 
 ifile.clear();
 ifile.seekg(0,ios::beg);
 
 do {
 getline (ifile, tline);
 } while ((tline.find('#') != string::npos) || (tline.find_first_not_of(' ') == tline.find_last_not_of(' ')));
 
 stringstream ss(tline);
 NaNStream nis(ss, ss);
 
 while ((ss.tellg() < ss.str().length()) && (ss.tellg() > -1)) {
 double readv;
 char ch;
 //ss>>readv>>ch;
 //tom
 //		if (ss>>readv) ncols++;
 
 // .alex. -> zappo via a piene mani, secondo me non serve tutta questa roba
 //	string ostr = ss.str();
 //	for (register size_t ii=0; ii<ostr.length(); ii++) {
 //		if (isprint(ostr[ii])) {
 
 
 nis>>readv;
 ncols++;
 //			break;
 //		}
 //	}
 }
 
 cerr<<"file has "<<nlines<<" lines and "<<ncols<<" cols"<<endl;
 
 //physImage myimg(ncols, nlines);
 this->resize(ncols, nlines);
 
 // 2. read and save five
 ifile.clear();
 ifile.seekg(0, ios::beg);
 
 int row = 0, col = 0;
 char ch;
 int w = getW(), h = getH();
 while (getline(ifile, tline)) {
 if ((tline.find('#') == string::npos) && (tline.find_first_not_of(' ') != tline.find_last_not_of(' '))) {
 stringstream sline(tline);
 NaNStream nis(sline, sline);
 //while ((sline.tellg() < sline.str().length()) && (sline.tellg() > -1)) {
 for (register size_t col=0; col<ncols; col++) {
 nis>>Timg_buffer[row*w+col];
 //sline>>Timg_buffer[row*w+col];
 //sline>>Timg_buffer[row*width+col]>>ch;	// forse questo non ci va
 //col++;
 }
 //col = 0;
 row++;
 }
 }
 
 TscanBrightness();
 time3=clock();
 cerr << "times: " << time1 << " " << time2 << " " << time3 << endl;
 }
 */
physDouble_asc::physDouble_asc(const char *ifilename)
: nPhysImageF<double>(string(ifilename), PHYS_FILE)
{
	ifstream ifile(ifilename);
	// 1. get image statistics
	string tline;
	int nlines = 0, ncols = 0;
	while (getline (ifile, tline))
		nlines++;
	ifile.clear();
	ifile.seekg(0,ios::beg);
	
	getline (ifile, tline);
	stringstream ss(tline);
	
	while ((ss.tellg() < (int) ss.str().length()) && (ss.tellg() > -1)) {
		double readv;
		char ch;
		ss>>readv>>ch;
		ncols++;
	}
	
	DEBUG(5,"file has "<<nlines<<" lines and "<<ncols<<" cols");
	
	this->resize(ncols, nlines);
	
	// 2. read and save five
	ifile.clear();
	ifile.seekg(0, ios::beg);
	int row = 0, col = 0;
	char ch;
	while (getline(ifile, tline)) {
		stringstream sline(tline);
		while ((sline.tellg() < (int) sline.str().length()) && (sline.tellg() > -1)) {
			sline >> Timg_buffer[row*getW()+col] >> ch;
			col++;
		}
		col = 0;
		row++;
	}
	TscanBrightness();
}


physInt_pgm::physInt_pgm(const char *ifilename)
: nPhysImageF<int>(string(ifilename), PHYS_FILE)
{
#ifdef HAVE_LIBNETPBM
	int grays;
	int **readbuf;
	int w, h;
	
	FILE *ifd;
	ifd = fopen(ifilename,"rb");
	readbuf = (int **)pgm_readpgm(ifd, &w, &h, (gray *)&grays);
	
	this->resize(w, h);
	
	//if (grays<256) bpp = 1;
	//else bpp = 2;
	
	DEBUG(5,"width: "<<getW());
	DEBUG(5,"height: "<<getH());
	DEBUG(5,"grays: "<<grays);
	
	for (size_t i=0; i<getH(); i++) {
		for (size_t j=0; j<getW(); j++) {
			set(i,j,(int)((readbuf[i])[j]));
		}
	}
	
	TscanBrightness();
#else
	WARNING("nPhysImage was compile without netpbm library");
#endif
}


#ifdef HAVE_LIBNETPBM
physGray_pgm::physGray_pgm(const char *ifilename)
: nPhysImageF<gray>(string(ifilename), PHYS_FILE)
{
	gray grays;
	gray **readbuf;
	int w, h;
	
	FILE *ifd;
	ifd = fopen(ifilename,"rb");
	readbuf = pgm_readpgm(ifd, &w, &h, &grays);
	
	this->resize(w, h);
	
	//	if (grays<256) bpp = 1;
	//	else bpp = 2;
	
	DEBUG(5,"width: "<<getW());
	DEBUG(5,"height: "<<getH());
	DEBUG(5,"grays: "<<grays);
	
	for (size_t i=0; i<getH(); i++) {
		memcpy(Timg_matrix[i], readbuf[i], w*sizeof(gray));
	}
	
	TscanBrightness();
}
#endif

physInt_sif::physInt_sif(string ifilename)
: nPhysImageF<int>(ifilename, PHYS_FILE)
{
	// Andor camera .sif file
	
	string temp_string;
	stringstream ss;
	int skiplines=0;
	
	ifstream ifile(ifilename.c_str(), ios::in | ios::binary);
	getline(ifile, temp_string);
	if ( temp_string.substr(0,5)!=string("Andor")) {
		WARNING("Does not start with Andor "<<ifilename);
		return;
	}
	
	// matrix informations on line 5
	for (size_t i=0; i<3; i++) {
		getline(ifile, temp_string);
		ss.str(""); ss.clear(); ss << setw(2) << setfill('0') << skiplines++;
		property["sif-a-"+ss.str()]=temp_string;
	}
	getline(ifile, temp_string);
	ss.str(""); ss.clear(); ss << setw(2) << setfill('0') << skiplines++;
	property["sif-b-"+ss.str()]=temp_string;

	int w, h;
	ss.str(temp_string);
	ss >> w;
	ss >> h;
	this->resize(w, h);
	
	getline(ifile, temp_string);
	ss.str(""); ss.clear(); ss << setw(2) << setfill('0') << skiplines++;
	property["sif-c-"+ss.str()]=temp_string;
	
	getline(ifile, temp_string);
	ss.str(temp_string);

	int binary_header=0,useless=0;
	ss >> useless >> binary_header;
	DEBUG("unused value " << useless);
	vector<char> buf(binary_header);
	ifile.read(&buf[0], buf.size());
	

	/* 
	 * brought to you by some braindead @Andor's!
	 *
	 * 1. look for "Pixel number" (the first occurrence)
	 * 2. look for a line with *a single* number on it (no indent)
	 * 3. read the value and jump by the amount of lines!
	 *
	 * (thank you, Andor, thank you, I love this!)
	 */
	
	temp_string.clear();
	string control_string="Pixel number"; 
	while (!ifile.eof()) {
		getline(ifile, temp_string);
		ss.str(""); ss.clear(); ss << setw(2) << setfill('0') << skiplines++;
		property["sif-d-"+ss.str()]=temp_string;
		if (temp_string.substr(0,12) == control_string) {
			break;
		}		
	}	

	temp_string.clear();
	int magic_number = 0; // usually 3 (lol)
	while (!ifile.eof()) {
		getline(ifile, temp_string);
		istringstream iss(temp_string);
		
		ss.str(""); ss.clear(); ss << setw(2) << setfill('0') << skiplines++;
		property["sif-e-"+ss.str()]=temp_string;

		// most readable ever
                if ( !(iss >> std::noskipws >> magic_number).fail() && iss.eof() ) {
                    break;
		}
	}

	// jump magic lines
	DEBUG(5, "jump "<<magic_number<<" lines for the glory of Ra");
	for (size_t i=0; i<magic_number; i++) {
		getline(ifile, temp_string);
	}

	// consistency check
	
	int init_matrix = ifile.tellg();
	ifile.seekg(0, ifile.end);
	long int datacheck = ifile.tellg()-init_matrix-getSurf()*sizeof(float);

	if (ifile.eof() || ifile.fail()) {
		throw phys_fileerror("SIF: header parsing reached end of file");
	}

	if (datacheck < 0) {
		stringstream oss;
		oss<<"Failed consistency check before SIF matrix read\n";
		oss<<"init_matrix: "<<init_matrix<<"\n";
		oss<<"end_file: "<<ifile.tellg()<<"\n";
		oss<<"matrix surface: "<<getSurf()<<"\n";
		oss<<"matrix size: "<<getSurf()*sizeof(float)<<"\n";

		throw phys_fileerror(oss.str());

	} else {
		// get data
		ifile.seekg(init_matrix);
		DEBUG(5,"size : "<<getW()<< " x " <<getH() << " + " << ifile.tellg() );
		ss.str(""); ss.clear(); ss << init_matrix << " bytes";
		property["sif-header"]=ss.str();
		vector<float> readb(getSurf());
	
		ifile.read((char*)(&readb[0]),getSurf()*sizeof(float));
		DEBUG(ifile.gcount());
		ifile.close();
		for (size_t i=0; i<getSurf(); i++) set(i,(int) readb[i]);
	
		TscanBrightness();
		DEBUG(get_min() << " " << get_max());

	}

}

physShort_b16::physShort_b16(const char *ifilename)
: nPhysImageF<short>(string(ifilename), PHYS_FILE)
{
	
	char *ptr = new char[3], tempch;
	char *readb;
	char revision;
	
	int header_size=0; // cambia con la revision del file
	
	
	ifstream ifile(ifilename, ios::in | ios::binary);
	assert(ifile);
	
	ifile.read(ptr,3);
	
	//if ( string(ptr) != string("PCO") ) {
	if ( strncmp(ptr,"PCO",3) != 0 ) {
		WARNING("not a PCO file ["<<ptr<<"]");
		return;
	}
	
	// alla camWare succhiano i cazzi
	ifile.seekg(0x05,ios::beg);
	ifile.read(&revision,1);
	if (revision=='a') header_size=384;
	else if (revision=='b') header_size=512;
	
	
	// bpp (idiozia del formato: e' un carattere a 0x09)
	// altra idiozia: non capisco la specificazione di bpp in revision a: fisso a 2
	int bpp = 0;
	if (revision=='b') {
		ifile.seekg(0x09,ios::beg);
		ifile.read(&tempch,1);
		bpp = (int)tempch;
	} else if (revision=='a') bpp=2;
	
	int w, h;
	
	// width
	ifile.seekg(0x0c,ios::beg);
	ifile.read((char *)&w,4);
	
	
	// height
	ifile.seekg(0x10,ios::beg);
	ifile.read((char *)&h,4);
	
	this->resize(w, h);
	
	
	DEBUG(5,"width: "<<getW());
	DEBUG(5,"height: "<<getH());
	
	
	ifile.seekg(header_size,ios::beg);
	assert (readb = new char [w*bpp]);
	
	for (int i=0; i<h; i++) {
		
		memset(readb,0,w*bpp);
		ifile.read(readb,w*bpp);
		
		//		for (j=0; j<w; j++) {
		//			Timg_matrix[i][j] = (short)( *((unsigned short *)&readb[bpp*j]) );
		//		}
		
		memcpy(Timg_matrix[i], readb, w*sizeof(short));
	}
	
	
	ifile.close();
	delete readb;
	delete [] ptr;
}


physDouble_img::physDouble_img(string ifilename)
: nPhysImageF<double>(ifilename, PHYS_FILE) {
    
	unsigned short buffer;
	ifstream ifile(ifilename.c_str(), ios::in | ios::binary);
    
	int w=0;
	int h=0;
	int skipbyte=0;
    int kind=-1;
    
    bool endian=false;
    
	ifile.read((char *)&buffer,sizeof(unsigned short));
	
	if (buffer == 19785) { // Hamamatsu
		ifile.read((char *)&buffer,sizeof(unsigned short));
		skipbyte=buffer;
		ifile.read((char *)&buffer,sizeof(unsigned short));
		w=buffer;
		ifile.read((char *)&buffer,sizeof(unsigned short));
		h=buffer;
        
		ifile.seekg (4, ios_base::cur);
        
		ifile.read((char *)&buffer,sizeof(unsigned short));
		kind=buffer;
        
        
		ifile.seekg (50,ios_base::cur);
		
		string buffer2;
		buffer2.resize(skipbyte);
		ifile.read((char *)&buffer2[0],skipbyte);		
        
        property["info"]=buffer2;
        
        switch (kind) {
            case 2: // unsigned short int
                kind=2;
                break;
            case 3: // unsigned int
                kind=4;
                break;
            default:
                break;
        }		
        
	} else if (buffer == 512) { // ARP blue ccd camera w optic fibers...
	   	ifile.read((char *)&buffer,sizeof(unsigned short));
	   	if (buffer==7) {
			ifile.read((char *)&buffer,sizeof(unsigned short));
			skipbyte=buffer;
			ifile.seekg(skipbyte,ios_base::beg);
			ifile.read((char *)&buffer,sizeof(unsigned short));
			w=buffer;
			ifile.read((char *)&buffer,sizeof(unsigned short));
			h=buffer;
            kind=2;
            ifile.read((char *)&buffer,sizeof(unsigned short));
        }
	} else { // LIL images
		ifile.seekg(ios_base::beg);
		vector<unsigned int>lil_header (4);
	   	ifile.read((char *)&lil_header[0],lil_header.size()*sizeof(unsigned int));	   	
		if (lil_header[0]==2 && lil_header[3]==1) {
			// lil_header[0] = dimension of the matrix
			// lil_header[3] = kind of data (1=unisgned short, 2=long, 3= float, 4=double)
			w=lil_header[1];
			h=lil_header[2];
            kind=2;
		}
	}
    
    if (kind!=-1) {
        resize(w, h);
        skipbyte=ifile.tellg();
        ifile.close();
        property["kind"]=kind;
        property["skip bytes"]=skipbyte;
        int retVal=phys_open_RAW(this,kind,skipbyte,endian);
        if (retVal!=0) resize(0,0);
	}
}

physUint_imd::physUint_imd(string ifilename)
: nPhysImageF<unsigned int>(ifilename, PHYS_FILE)
{
	// Optronics luli
	// we should also check if a .imi text file exists and read it?
	
	unsigned short buffer_header;
	ifstream ifile(ifilename.c_str(), ios::in | ios::binary);
	unsigned short w=0;
	unsigned short h=0;
	
	ifile.read((char *)&buffer_header,sizeof(unsigned short));
	property["imd-version"]=buffer_header;
	ifile.read((char *)&buffer_header,sizeof(unsigned short));
	w=buffer_header;
	ifile.read((char *)&buffer_header,sizeof(unsigned short));
	h=buffer_header;
	
	this->resize(w, h);
   	ifile.read((char *)Timg_buffer,sizeof(unsigned int)*w*h);
	
	ifile.close();
	
	string ifilenameimg=ifilename;
	ifilenameimg.resize(ifilenameimg.size()-3);
	ifilenameimg = ifilenameimg+"imi";
	ifstream ifileimg(ifilenameimg.c_str(), ios::in);
	if (ifileimg) {
		string comment(""),temp_line;		
		while (!ifileimg.eof()) {
			getline(ifileimg, temp_line);
			comment.append(temp_line);
		}
		ifileimg.close();
		property["imi-info"]=comment;			
	}
	
	TscanBrightness();	
}


// --------- write out ------------
//

//std::ostream &
//operator<< (std::ostream &lhs, phys_properties &pp)
//{
//	lhs<<pp.phys_name<<"\n";
//	lhs<<pp.phys_orig<<"\n";
//	lhs<<pp.phys_short_name<<"\n";
//	lhs<<pp.phys_from_name<<"\n";
//	lhs<<pp.origin<<"\n";
//	lhs<<pp.scale<<"\n";
//	lhs<<bool(true);
//	return lhs;
//}
//
//// --------- read properties ------------
////
//std::istream &
//operator>> (std::istream &lhs, phys_properties &pp)
//{
//	string line;
//	getline(lhs,pp.phys_name);
//	// TODO enum!!!!!!
//	// 	int pippo;
//	// 	lhs>>pippo;
//	// 	pp.phys_orig=pippo;
//	getline(lhs,line);
//	
//	getline(lhs,pp.phys_short_name);
//	getline(lhs,pp.phys_from_name);
//	getline(lhs,line);
//	pp.origin=bidimvec<double>(line);
//	getline(lhs,line);
//	pp.scale=bidimvec<double>(line);
//	getline(lhs,line);
//	//pp.derived=(line=="0");
//	return lhs;
//}

// dump out for state save
int
phys_dump_binary(nPhysImageF<double> *my_phys, const char *fname) {
	ofstream ofile(fname, ios::out | ios::binary);
	return phys_dump_binary(my_phys,ofile);
}

int
phys_dump_binary(nPhysImageF<double> *my_phys, std::ofstream &ofile) {
	
	if (ofile.fail()) {
		WARNING("ostream error");
		return -1;
	}
	
	if (my_phys == NULL) {
		WARNING("phys error");
		return -1;
	}
	
	//int pos = ofile.tellg();
	//int written_data = 0;
	
	//ofile<<my_phys->property<<"\n";
	my_phys->property.dumper(ofile);
	
	DEBUG("Starting binary dump...");
	
	
	//ofile<<my_phys->getW()<<"\n";
	//ofile<<my_phys->getH()<<"\n";
	
	// Compress data using zlib
	int buffer_size=my_phys->getSurf()*sizeof(double);
	unsigned char *out= new unsigned char [buffer_size];
	z_stream strm;
	
	strm.zalloc = Z_NULL;
	strm.zfree  = Z_NULL;
	strm.opaque = Z_NULL;
	int status;
	status=deflateInit2 (&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, windowBits | GZIP_ENCODING, 8, Z_DEFAULT_STRATEGY);
	if (status < 0) {
		// .alex. -- questa non compila (e non sono sicuro sia a causa di anymap)
		WARNING("Zlib a bad status of " << status);
		exit (EXIT_FAILURE);
	}
	
	strm.next_in = (unsigned char *) my_phys->Timg_buffer;
	strm.avail_in = my_phys->getSurf()*sizeof(double);
	strm.avail_out = buffer_size;
	strm.next_out = out;
	status= deflate (& strm, Z_FINISH);
	
	if (status < 0) {
		WARNING("Zlib a bad status of " << status);
		exit (EXIT_FAILURE);
	}
	int writtendata = buffer_size - strm.avail_out;
	
	deflateEnd (& strm);
	DEBUG(5,"writtendata " << writtendata);
	
	int oo = my_phys->getW();
	ofile.write((const char *)&oo, sizeof(int));
	oo = my_phys->getH();
	ofile.write((const char *)&oo, sizeof(int));
	
	//ofile << writtendata << "\n"; // TODO: to be written in binary
	ofile.write((const char *)&writtendata, sizeof (int));
	ofile.write((const char *)out, sizeof (char)*writtendata);
	DEBUG("Binary dump finished");
	
	delete [] out;
	//	ofile.write((const char *)my_phys->Timg_buffer, my_phys->getSurf()*sizeof(double));
	ofile<<"\n"<<std::flush;
	
	//written_data = ofile.tellg()-pos;
	
	return 0;
}

physDouble_tiff::physDouble_tiff(const char *ifilename)
: nPhysImageF<double>(string(ifilename), PHYS_FILE) {
#ifdef HAVE_LIBTIFF
	TIFF* tif = TIFFOpen(ifilename, "r");
	if (tif) {
		DEBUG("Opened");
		unsigned short samples=1, compression, config;
		TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
		TIFFGetField(tif, TIFFTAG_COMPRESSION, &compression);
		TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &config);
		
		
		DEBUG("COMPRESSION_NONE " << compression << " " << COMPRESSION_NONE);
		DEBUG("PLANARCONFIG_CONTIG " << config << " " << PLANARCONFIG_CONTIG);
		DEBUG("SAMPLES" << samples);
		//		vector<unsigned short> extra(samples);
		// 		TIFFGetField(tif, TIFFTAG_EXTRASAMPLES, &extra[0]);
		// 		for (int k=0;k<samples;k++) {
		// 			DEBUG("extra " << k << "  " << extra[k]);
		// 		}
		
		if (compression==COMPRESSION_NONE && config==PLANARCONFIG_CONTIG ) {
			float resx=1.0, resy=1.0;
			TIFFGetField(tif, TIFFTAG_XRESOLUTION, &resx);
			TIFFGetField(tif, TIFFTAG_YRESOLUTION, &resy);
			if (resx!=0.0 && resy!=0.0) {
				set_scale(resx,resy);
			}
			float posx=0.0, posy=0.0;
			TIFFGetField(tif, TIFFTAG_XPOSITION, &posx);
			TIFFGetField(tif, TIFFTAG_YPOSITION, &posy);
			set_origin(posx,posy);
			
			char *desc=NULL;
			if (TIFFGetField(tif, TIFFTAG_DOCUMENTNAME, &desc)) {
				setName(desc);
				DEBUG(desc);
			}
			if (TIFFGetField(tif, TIFFTAG_IMAGEDESCRIPTION, &desc)) {
 				property["info"]=string(desc);
				DEBUG(desc);
			}
			
			setFromName(ifilename);
			unsigned short bytesperpixel=0;
			TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bytesperpixel);
			bytesperpixel/=8;
			unsigned int w=0, h=0;
			TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
			TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
			DEBUG("Width " << w << " Height " << h << " BYTESperPIXEL " << bytesperpixel);
			if (w*h>0 && bytesperpixel>0 ) {
				tsize_t scanlineSize=TIFFScanlineSize(tif);
				tdata_t buf = _TIFFmalloc(scanlineSize);
				DEBUG("Buf allocated " << scanlineSize);
				resize(w,h);
				for (tiff_uint32 j = 0; j < h; j++) {
					TIFFReadScanline(tif, buf, j);
					for (tiff_uint32 i=0; i<w; i++) {
						double val=0;
						// 						samples=1;
						for (int k=0;k<samples;k++) {
							if (bytesperpixel == sizeof(char)) {
								val+=((unsigned char*)buf)[i*samples];
							} else if (bytesperpixel == sizeof(short)) {
								val+=((unsigned short*)buf)[i*samples];
							} else if (bytesperpixel == sizeof(float)) {
								val+=((float*)buf)[i*samples];
							} else if (bytesperpixel == sizeof(double)) {
								val+=((double*)buf)[i*samples];
							}
						}
						set(i,j,val/samples);
					}
				}
				_TIFFfree(buf);
				TscanBrightness();
			}
		}
	}
	TIFFClose(tif);
#else
	WARNING("nPhysImage was not compiled with tiff support!");
#endif
}

int
phys_write_tiff(nPhysImageF<double> *my_phys, const char * ofilename) {
#ifdef HAVE_LIBTIFF
	TIFF* tif = TIFFOpen(ofilename, "w");
	if (tif && my_phys) {
		TIFFSetWarningHandler(NULL);
		TIFFSetField(tif, TIFFTAG_SUBFILETYPE, 0);
		TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1);
		TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
		TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
		TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
		TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
 		TIFFSetField(tif, TIFFTAG_DOCUMENTNAME, my_phys->getName().c_str());
 		TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, my_phys->getShortName().c_str());
 		TIFFSetField(tif, TIFFTAG_SOFTWARE, "neutrino");
 		TIFFSetField(tif, TIFFTAG_COPYRIGHT, "http::/web.luli.polytchnique.fr/neutrino");
		
		TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, my_phys->getW());
		TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, my_phys->getH());
		TIFFSetField(tif, TIFFTAG_IMAGELENGTH, my_phys->getH());
		float scalex=my_phys->get_scale().x();
 		TIFFSetField(tif, TIFFTAG_XRESOLUTION, scalex);
		float scaley=my_phys->get_scale().y();
 		TIFFSetField(tif, TIFFTAG_YRESOLUTION, scaley);
		float origx=my_phys->get_origin().x();
		TIFFSetField(tif, TIFFTAG_XPOSITION, origx);
		float origy=my_phys->get_origin().y();
		TIFFSetField(tif, TIFFTAG_XPOSITION, origy);
		TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8*sizeof(float));
		DEBUG("tiff " << TIFFScanlineSize(tif) << " " << my_phys->getW() <<  " " << my_phys->getH());
		unsigned char *buf = (unsigned char *) _TIFFmalloc(TIFFScanlineSize(tif));
		for (size_t j = 0; j < my_phys->getH(); j++) {
 			for (size_t i=0; i<my_phys->getW(); i++) {
 				((float*)buf)[i]=(float)my_phys->point(i,j) ;
 			}
 			TIFFWriteScanline(tif, buf, j, 0);
 		}
		_TIFFfree(buf);
		TIFFClose(tif);
		return 0;
	} else {
		return -1;
	}
#else
	WARNING("nPhysImage was not compiled with tiff support!");
#endif
}

std::vector <nPhysImageF<double> *> phys_open_spe(std::string ifilename) {
	std::vector <nPhysImageF<double> *> vecReturn;
	
	ifstream ifile(ifilename.c_str(), ios::in);
	string header;
	header.resize(4100);
	ifile.read((char *)&header[0],header.size());
	unsigned short type= *((unsigned short *) &header[108]);
	
	unsigned short height= *((unsigned short *) &header[656]);
	unsigned short width= *((unsigned short *) &header[42]);
	unsigned short noscan= *((unsigned short *) &header[34]);
	unsigned int NumFrames= *((unsigned int *) &header[1446]);
	
	if (noscan == 65535) {
		int lnoscan = *((int *) &header[664]);
		if (lnoscan != -1) NumFrames = lnoscan / height;
	} else {
		NumFrames = noscan / height;
	}
	
	DEBUG(type << " " << height  << " " << width << " " << noscan << " " << NumFrames);
	
	for (unsigned int nf=0;nf<NumFrames;nf++) {
		nPhysD *phys=new nPhysD(width,height,0.0);
		phys->property["spe-frame"]=vec2f(nf,NumFrames);
		switch (type) {
			case 0: {
				phys->property["spe-type"]="float";
				vector<float> buffer(width*height);
				ifile.read((char*) &buffer[0],width*height*sizeof(float));
				for (unsigned int i=0; i<phys->getSurf(); i++) {
					phys->set(i,buffer[i]);
				}
				break;
			}
			case 1: {
				phys->property["spe-type"]="int";
				vector<int> buffer(width*height);
				DEBUG(sizeof(long));
				ifile.read((char*) &buffer[0],width*height*sizeof(int));
				for (unsigned int i=0; i<phys->getSurf(); i++) {
					phys->set(i,buffer[i]);
				}
				break;
			}
			case 2: {
				phys->property["spe-type"]="short";
				vector<short> buffer(width*height);
				ifile.read((char*) &buffer[0],width*height*sizeof(short));
				for (unsigned int i=0; i<phys->getSurf(); i++) {
					phys->set(i,buffer[i]);
				}
				break;
			}
			case 3: {
				phys->property["spe-type"]="unsigned short";
				vector<unsigned short> buffer(width*height);
				ifile.read((char*) &buffer[0],width*height*sizeof(unsigned short));
				for (unsigned int i=0; i<phys->getSurf(); i++) {
					phys->set(i,buffer[i]);
				}
				break;
			}
			default:
				break;
		}
		
		phys->TscanBrightness();
		vecReturn.push_back(phys);
	}
	
	ifile.close();
	return vecReturn;
}

std::vector <nPhysImageF<double> *> phys_open_inf(std::string ifilename) {
	std::vector <nPhysImageF<double> *> imagelist;
	ifstream ifile(ifilename.c_str(), ios::in);
	if (ifile) {
		string ifilenameimg=ifilename;
		ifilenameimg.resize(ifilenameimg.size()-3);
		ifilenameimg = ifilenameimg+"img";
		ifstream ifileimg(ifilenameimg.c_str(), ios::in);
		if (ifileimg) {
			DEBUG("start");
			string line;
			getline(ifile,line);
			if (line.compare(string("BAS_IMAGE_FILE"))!=0) {
				WARNING("does not start with BAS_IMAGE_FILE");
				return imagelist;
			}
			getline(ifile,line); //this is the basename
			getline(ifile,line); //this we don't know what it is
			getline(ifile,line);
			double resx=atoi(line.c_str());
			getline(ifile,line);
			double resy=atoi(line.c_str());
			getline(ifile,line);
			int bit=atoi(line.c_str());
			getline(ifile,line);
			int w=atoi(line.c_str());
			getline(ifile,line);
			int h=atoi(line.c_str());
			nPhysImageF<double> *linearized = new nPhysImageF<double>(w,h,0.0,(string("(linearized)")+ifilename).c_str());
			linearized->setType(PHYS_FILE);
			linearized->setFromName(ifilename);
			linearized->setShortName("(linearized)");
			imagelist.push_back(linearized);
			nPhysImageF<double> *original = new nPhysImageF<double>(w,h,0.0,ifilename.c_str());
			original->setType(PHYS_FILE);
			original->setFromName(ifilename);
			original->setShortName("(original)");
			imagelist.push_back(original);
			getline(ifile,line);
			double sensitivity=atoi(line.c_str());
			getline(ifile,line);
			double latitude=atoi(line.c_str());
			// FIXME: this is awful
			switch (bit) {
				case 8: {
					vector<char> buf(original->getSurf());
					ifileimg.read(&buf[0], original->getSurf());
					for (size_t i=0;i<original->getSurf();i++) {
						original->set(i,swap_endian<char>(buf[i]));
						linearized->set(i,((resx*resy)/10000.0) * (4000.0/sensitivity) * pow(10,latitude*(original->point(i)/pow(2.0,bit)-0.5)));
					}
					break;
				}
				case 16: {
					vector<unsigned short> buf(original->getSurf());
					ifileimg.read((char*)(&buf[0]), sizeof(unsigned short)*original->getSurf());
					for (size_t i=0;i<original->getSurf();i++) {
						original->set(i,swap_endian<unsigned short>(buf[i]));
						linearized->set(i,((resx*resy)/10000.0) * (4000.0/sensitivity) * pow(10,latitude*(original->point(i)/pow(2.0,bit)-0.5)));
					}
					break;
				}
			}
			original->property["resx"] = linearized->property["resx"] = resx;
			original->property["resy"] = linearized->property["resy"] = resy;
			original->property["bits"] = linearized->property["bits"] = bit;
			original->property["sens"] = linearized->property["sens"] = sensitivity;
			original->property["lati"] = linearized->property["lati"] = latitude;
			linearized->set_scale(resx,resy);
			linearized->TscanBrightness();
			original->TscanBrightness();
		}
		ifileimg.close();
	}
	ifile.close();
	return imagelist;
}

int phys_write_fits(nPhysImageF<double> *phys, const char * fname, float compression) {
#ifdef HAVE_LIBCFITSIO
	fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
	int status=0;
	if (fits_create_file(&fptr, fname, &status)) {
		fits_report_error(stderr, status);
		return status;
	}
	if (fits_set_compression_type(fptr, GZIP_1, &status)) {
		fits_report_error(stderr, status);
		return status;
	}
	int ndim=2;
	long ndimLong[2]={phys->getW(),phys->getH()};
	if (fits_set_tile_dim(fptr,ndim,ndimLong,&status)) {
		fits_report_error(stderr, status);
		return status;
	}
	if (fits_set_quantize_level(fptr, 0.0, &status)) {
		fits_report_error(stderr, status);
		return status;
	}
	long naxes[2]; naxes[0]= phys->getW(); naxes[1] =phys->getH();
	
	if (fits_create_img(fptr,  DOUBLE_IMG, 2, naxes, &status)) {
		fits_report_error(stderr, status);
		return status;
	}
	if (fits_write_img(fptr, TDOUBLE, 1, phys->getSurf(), phys->Timg_buffer, &status)) {
		fits_report_error(stderr, status);
		return status;
	}
	double orig_x=phys->get_origin().x();
	if (fits_update_key(fptr, TDOUBLE, "ORIGIN_X", &orig_x, "nPhysImage origin x", &status)) {
		fits_report_error(stderr, status);
		return status;
	}
	double orig_y=phys->get_origin().y();
	if (fits_update_key(fptr, TDOUBLE, "ORIGIN_Y", &orig_y, "nPhysImage origin y", &status)) {
		fits_report_error(stderr, status);
		return status;
	}
    
	double scale_x=phys->get_scale().x();
	if (fits_update_key(fptr, TDOUBLE, "SCALE_X", &scale_x, "nPhysImage scale x", &status)) {
		fits_report_error(stderr, status);
		return status;
	}
	double scale_y=phys->get_scale().y();
	if (fits_update_key(fptr, TDOUBLE, "SCALE_Y", &scale_y, "nPhysImage scale y", &status)) {
		fits_report_error(stderr, status);
		return status;
	}
	string name=phys->getName();
	if (fits_update_key_longstr(fptr, "NAME", &name[0], "nPhysImage name", &status)) {
		fits_report_error(stderr, status);
		return status;
	}
    
	if (fits_close_file(fptr, &status)) {
		fits_report_error(stderr, status);
		return status;
	}
	return 1;	
#endif
}

std::vector <nPhysImageF<double> *> phys_open_fits(std::string ifilename) {
	vector<nPhysD *> retVec;
#ifdef HAVE_LIBCFITSIO
	fitsfile *fptr;
	char card[FLEN_CARD];
	int status = 0,  nkeys, ii;
	
	fits_open_file(&fptr, ifilename.c_str(), READONLY, &status);
	int bitpix;
	int anaxis;
	
	if (fits_is_compressed_image(fptr, &status)) {
		fits_report_error(stderr, status);
	}
	DEBUG("fits compressed " << status);
    
	int hdupos=0;
	if (fits_get_hdu_num(fptr, &hdupos)) {
		fits_report_error(stderr, status);
	}
	DEBUG("fits_get_hdu_num " << hdupos);
	int single = (hdupos != 1) ? 1 : 0;
	DEBUG("single " << single);
	
	for (; !status; hdupos++)  {
 		DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		
		nPhysD *myphys=new nPhysD(ifilename,PHYS_FILE);
		
		DEBUG(myphys->getShortName());
		int hdutype;
		if (fits_get_hdu_type(fptr, &hdutype, &status)) {
			fits_report_error(stderr, status);
		}
        // 		DEBUG("fits_get_hdu_type " << hdutype);
        
        
        // 		if (hdutype == IMAGE_HDU) {
        // 			long naxes[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        // 			for (ii = 0; ii < 9; ii++)
        // 				naxes[ii] = 1;
        // 			  int naxis = 0;
        // 			fits_get_img_param(fptr, 9, &bitpix, &naxis, naxes, &status);
        // 
        // 			long totpix = naxes[0] * naxes[1] * naxes[2] * naxes[3] * naxes[4] * naxes[5] * naxes[6] * naxes[7] * naxes[8];
        // // 			DEBUG("totpix " << totpix);
        // 		}
        
		fits_get_img_type(fptr,&bitpix,&status);
        // 		DEBUG(5,"fits_get_img_type " << bitpix);
        
		fits_get_img_dim(fptr,&anaxis,&status);
        // 		DEBUG(5,"fits_get_img_dim " << anaxis);
		vec2f orig=myphys->get_origin();
		vec2f scale=myphys->get_scale();
        
		fits_get_hdrspace(fptr, &nkeys, NULL, &status);
		for (ii = 1; ii <= nkeys; ii++)  {
			fits_read_record(fptr, ii, card, &status); /* read keyword */
			if (status)
				fits_report_error(stderr, status);
			string cardStr=string(card);
            // 			transform(cardStr.begin(), cardStr.end(), cardStr.begin(), ::tolower);
			if (fits_get_keyclass(card)==TYP_USER_KEY) {
                // 				DEBUG(cardStr);
				string ctrl="ORIGIN_X";
				if (cardStr.compare(0,ctrl.length(),ctrl)==0) {
					char dtype;
					fits_get_keytype(card, &dtype, &status);
					if (dtype=='F') {
						double val;
						fits_read_key_dbl(fptr, ctrl.c_str(), &val, NULL, &status );
						orig.set_first(val);
					}
				}
				ctrl="ORIGIN_Y";
				if (cardStr.compare(0,ctrl.length(),ctrl)==0) {
					char dtype;
					fits_get_keytype(card, &dtype, &status);
					if (dtype=='F') {
						double val;
						fits_read_key_dbl(fptr, ctrl.c_str(), &val, NULL, &status );
						orig.set_second(val);
					}
				}
				ctrl="SCALE_X";
				if (cardStr.compare(0,ctrl.length(),ctrl)==0) {
					char dtype;
					fits_get_keytype(card, &dtype, &status);
					if (dtype=='F') {
						double val;
						fits_read_key_dbl(fptr, ctrl.c_str(), &val, NULL, &status );
						scale.set_first(val);
					}
				}
				ctrl="SCALE_Y";
				if (cardStr.compare(0,ctrl.length(),ctrl)==0) {
					char dtype;
					fits_get_keytype(card, &dtype, &status);
					if (dtype=='F') {
						double val;
						fits_read_key_dbl(fptr, ctrl.c_str(), &val, NULL, &status );
						scale.set_second(val);
					}
				}
			} else {
				stringstream ss; ss << setw(log10(nkeys)+1) << setfill('0') << ii;
				myphys->property["fits-"+ss.str()]=card;
			}
		}
		myphys->set_origin(orig);
		myphys->set_scale(scale);
        // 		property.dumper(std::cerr);
        
        
		if (anaxis==2) {
			long axissize[2], fpixel[2];
            
			fits_get_img_size(fptr,anaxis,axissize,&status);
			if (status)
				fits_report_error(stderr, status);
            
			long totalsize=axissize[0]*axissize[1];
			fpixel[0]=fpixel[1]=1;
            
			myphys->resize(axissize[0],axissize[1]);
            
			fits_read_pix(fptr, TDOUBLE, fpixel, totalsize, NULL, (void *)myphys->Timg_buffer, NULL, &status);
			if (status)
				fits_report_error(stderr, status);
            
			myphys->TscanBrightness();
            
		}
		if (myphys->getSurf()) {
			retVec.push_back(myphys);
		} else {
			delete myphys;
		}
		if (single) break;
		fits_movrel_hdu(fptr, 1, NULL, &status);  /* try to move to next HDU */
	}
	
	fits_close_file(fptr, &status);
#endif
	return retVec;
}

vector <nPhysImageF<double> *> phys_resurrect_binary(std::string fname) { 
	
	vector <nPhysImageF<double> *> imagelist;
	ifstream ifile(fname.c_str(), ios::in | ios::binary);
	int ret;
	while(ifile.peek()!=-1) {
		
		nPhysD *datamatrix = new nPhysD();
		DEBUG("here");
		
		// crisstho della madonna...
		// i morti con i morti, i vivi con i vivi
		//	int ret=phys_resurrect_old_binary(datamatrix,ifile);
		//	if (ret>=0 && datamatrix->getSurf()>0) {
		//		imagelist.push_back(datamatrix);
		//	} else {
		
		ret=phys_resurrect_binary(datamatrix,ifile);
 		if (ret>=0 && datamatrix->getSurf()>0) {
			imagelist.push_back(datamatrix);
		} else {
			delete datamatrix;
			//		}
			
		}
		
	}
	ifile.close();
	return imagelist;
}

int
phys_resurrect_old_binary(nPhysImageF<double> * my_phys, std::ifstream &ifile) {
	
	streampos posfile=ifile.tellg();
	
 	if (ifile.fail() || ifile.eof()) {
 		WARNING("istream error");
 		return -1;
 	}
	
	string line;
	getline(ifile,line);
	my_phys->setName(line);
	getline(ifile,line);
	getline(ifile,line);
	my_phys->setShortName(line);
	getline(ifile,line);
	my_phys->setFromName(line);
	
	getline(ifile,line);
	vec2f orig(line);
	my_phys->set_origin(orig);
	
	getline(ifile,line);
	vec2f scale(line);
	my_phys->set_scale(scale);
	
	getline(ifile,line);
	if (line!="1") {
		DEBUG("OLD neu file, rewind to pos " << posfile);
		ifile.seekg(posfile);
		return -1;
	}
	
	// 	getline(ifile,pp.phys_short_name);
	// 	getline(ifile,pp.phys_from_name);
	// 	getline(ifile,line);
	// 	pp.origin=bidimvec<double>(line);
	// 	getline(ifile,line);
	// 	pp.scale=bidimvec<double>(line);
	// 	getline(ifile,line);
	// 
	
	
	getline(ifile,line);
	int my_w=atoi(line.c_str());
 	getline(ifile,line);
	int my_h=atoi(line.c_str());
	
 	getline(ifile,line);
	int buffer_size=atoi(line.c_str());
	if (buffer_size>(int) (my_w*my_h*sizeof(double))) return -1; // this should not happend
	
 	my_phys->resize(my_w,my_h);
	unsigned char *in= new  unsigned char [buffer_size];
 	ifile.read((char *)in, buffer_size*sizeof(char));
	
	z_stream strm;
	
	strm.zalloc = Z_NULL;
	strm.zfree  = Z_NULL;
	strm.opaque = Z_NULL;
	int status;
	status = inflateInit2 (&strm,windowBits | GZIP_ENCODING);
	if (status < 0) {
		WARNING("Zlib a bad status of " << status);
	  	delete in;
		delete my_phys;
	 	my_phys=NULL;
		return (EXIT_FAILURE);
	}
	
	strm.next_in = in;
	strm.avail_in = buffer_size;
	strm.next_out = (unsigned char *)my_phys->Timg_buffer;
	strm.avail_out = my_phys->getSurf()*sizeof(double);
	status = inflate (&strm, Z_SYNC_FLUSH);
	if (status < 0) {
		WARNING("Zlib a bad status of " << status);
	 	delete in;
	 	delete my_phys;
	 	my_phys=NULL;
		return (EXIT_FAILURE);
	}
	inflateEnd (& strm);
 	delete in;
 	
 	my_phys->setType(PHYS_FILE);
 	
	//	ifile.read((char *)my_phys->Timg_buffer, my_phys->getSurf()*sizeof(double));
	my_phys->TscanBrightness();
	getline(ifile,line);
	
	
	return 0;
}

int
phys_resurrect_binary(nPhysImageF<double> * my_phys, std::ifstream &ifile) {
	
 	if (ifile.fail() || ifile.eof()) {
 		WARNING("istream error");
 		return -1;
 	}
	
	my_phys->property.loader(ifile);
	
	// w/h/size binary read
	int my_w, my_h, buffer_size;
	ifile.read((char *)&my_w, sizeof(int));
	ifile.read((char *)&my_h, sizeof(int));
	ifile.read((char *)&buffer_size, sizeof(int));
	
	string line;
 	/*getline(ifile,line);
	 int my_w=atoi(line.c_str());
	 getline(ifile,line);
	 int my_h=atoi(line.c_str());
	 
	 getline(ifile,line);
	 int buffer_size=atoi(line.c_str());*/
	
	DEBUG("w: "<<my_w);
	DEBUG("h: "<<my_h);
	DEBUG("s: "<<buffer_size);
	
	if (buffer_size>(int) (my_w*my_h*sizeof(double))) return -1; // this should not happend
	
 	my_phys->resize(my_w,my_h);
	unsigned char *in= new  unsigned char [buffer_size];
 	ifile.read((char *)in, buffer_size*sizeof(char));
	
	z_stream strm;
	
	strm.zalloc = Z_NULL;
	strm.zfree  = Z_NULL;
	strm.opaque = Z_NULL;
	int status;
	status = inflateInit2 (&strm,windowBits | GZIP_ENCODING);
	if (status < 0) {
		WARNING("Zlib a bad status of " << status);
	  	delete in;
		// my_phys is passed as an argument: function shoud not mess around with it!
		//delete my_phys;
	 	my_phys=NULL;
		return (EXIT_FAILURE);
	}
	
	strm.next_in = in;
	strm.avail_in = buffer_size;
	strm.next_out = (unsigned char *)my_phys->Timg_buffer;
	strm.avail_out = my_phys->getSurf()*sizeof(double);
	status = inflate (&strm, Z_SYNC_FLUSH);
	if (status < 0) {
		WARNING("Zlib a bad status of " << status);
	 	delete in;
		// my_phys is passed as an argument: function shoud not mess around with it!
	 	//delete my_phys;
	 	my_phys=NULL;
		return (EXIT_FAILURE);
	}
	inflateEnd (& strm);
 	delete [] in;
 	
 	my_phys->setType(PHYS_FILE);
 	
	//	ifile.read((char *)my_phys->Timg_buffer, my_phys->getSurf()*sizeof(double));
	my_phys->TscanBrightness();
	getline(ifile,line);
	return 0;	
}


/*int
 phys_resurrect_old_binary(nPhysImageF<double> * my_phys, std::ifstream &ifile) {
 
 if (ifile.fail() || ifile.eof()) {
 WARNING("istream error");
 return -1;
 }
 
 
 // TODO: re-write property read for oldstyle file format
 //ifile >> my_phys->property;
 //my_phys->property.loader(ifile);
 
 string line;
 getline(ifile,line);
 int my_w=atoi(line.c_str());
 getline(ifile,line);
 int my_h=atoi(line.c_str());
 
 getline(ifile,line);
 int buffer_size=atoi(line.c_str());
 if (buffer_size>(int) (my_w*my_h*sizeof(double))) return -1; // this should not happend
 
 my_phys->resize(my_w,my_h);
 unsigned char *in= new  unsigned char [buffer_size];
 ifile.read((char *)in, buffer_size*sizeof(char));
 
 z_stream strm;
 
 strm.zalloc = Z_NULL;
 strm.zfree  = Z_NULL;
 strm.opaque = Z_NULL;
 int status;
 status = inflateInit2 (&strm,windowBits | GZIP_ENCODING);
 if (status < 0) {
 WARNING("Zlib a bad status of " << status);
 delete in;
 delete my_phys;
 my_phys=NULL;
 return (EXIT_FAILURE);
 }
 
 strm.next_in = in;
 strm.avail_in = buffer_size;
 strm.next_out = (unsigned char *)my_phys->Timg_buffer;
 strm.avail_out = my_phys->getSurf()*sizeof(double);
 status = inflate (&strm, Z_SYNC_FLUSH);
 if (status < 0) {
 WARNING("Zlib a bad status of " << status);
 delete in;
 delete my_phys;
 my_phys=NULL;
 return (EXIT_FAILURE);
 }
 inflateEnd (& strm);
 delete in;
 
 my_phys->setType(PHYS_FILE);
 
 //	ifile.read((char *)my_phys->Timg_buffer, my_phys->getSurf()*sizeof(double));
 my_phys->TscanBrightness();
 getline(ifile,line);
 return 0;
 
 
 }*/

int
phys_open_RAW(nPhysImageF<double> * my_phys, int kind, int skipbyte, bool endian){
	DEBUG("HERE________________________________");
	if (my_phys==NULL || my_phys->getSurf()==0) return -1;
	
	ifstream ifile(my_phys->getName().c_str(), ios::in | ios::binary);
	if (ifile.fail()) return -1;
	ifile.seekg(skipbyte);
	if (ifile.fail() || ifile.eof()) return -1;
	int bpp=0;
	
	DEBUG (my_phys->getSurf());
	switch (kind) {
		case 0: bpp=sizeof(unsigned char); break;
		case 1: bpp=sizeof(signed char); break;
		case 2: bpp=sizeof(unsigned short); break;
		case 3: bpp=sizeof(signed short); break;
		case 4: bpp=sizeof(unsigned int); break;
		case 5: bpp=sizeof(signed int); break;
		case 6: bpp=sizeof(float); break;
		case 7: bpp=sizeof(double); break;
		default: kind=-1;
	}
	
	if (kind<0) return -1;
	
	
	vector<char> buffer(bpp*my_phys->getSurf());
	ifile.read(&buffer[0], buffer.size());
	ifile.close();
	for (size_t i=0;i<my_phys->getSurf();i++) {
		switch (kind) {
			case 0: {
				unsigned char buf = *((unsigned char*)(&buffer[i*bpp]));
				my_phys->set(i,endian ? swap_endian<unsigned char>(buf) : buf);
				break;
			}
			case 1: {
				signed char buf = *((signed char*)(&buffer[i*bpp]));
				my_phys->set(i,endian ? swap_endian<signed char>(buf) : buf);
				break;
			}
			case 2: {
				unsigned short buf = *((unsigned short*)(&buffer[i*bpp]));
				my_phys->set(i,endian ? swap_endian<unsigned short>(buf) : buf);
				break;
			}
			case 3: {
				signed short buf = *((signed short*)(&buffer[i*bpp]));
				my_phys->set(i,endian ? swap_endian<signed short>(buf) : buf);
				break;
			}
			case 4: {
				unsigned int buf = *((unsigned int*)(&buffer[i*bpp]));
				my_phys->set(i,endian ? swap_endian<unsigned int>(buf) : buf);
				break;
			}
			case 5: {
				signed int buf = *((signed int*)(&buffer[i*bpp]));
				my_phys->set(i,endian ? swap_endian<signed int>(buf) : buf);
				break;
			}
			case 6: {
				float buf = *((float*)(&buffer[i*bpp]));
				my_phys->set(i,endian ? swap_endian<float>(buf) : buf);
				break;
			}
			case 7: {
				double buf = *((double*)(&buffer[i*bpp]));
				my_phys->set(i,endian ? swap_endian<double>(buf) : buf);
				break;
			}
		}
	}
	
	my_phys->TscanBrightness();
	return 0;
}

//! open HDF4 file (works for: Omega LLR  visar images)
vector <nPhysImageF<double> *> phys_open_HDF4(string fname) {
	vector <nPhysImageF<double> *> imagelist;
	DEBUG("HERE");
#if defined(HAVE_LIBMFHDF) || defined(HAVE_LIBMFHDFDLL)
	DEBUG("HERE2");
	int32 sd_id, sds_id, n_datasets, n_file_attrs, index,status ;
	int32 dim_sizes[3];
	int32 rank, num_type, attributes;
	int32 i;
	
	char name[64];
	
	nPhysImageF<double> *background=NULL;
	/* Open the file and initiate the SD interface. */
	sd_id = SDstart(fname.c_str(), DFACC_READ);
	if (sd_id != FAIL) {
		/* Determine the contents of the file. */
		SDfileinfo(sd_id, &n_datasets, &n_file_attrs);
		DEBUG(5,"datasets " << n_datasets);
		for (index = 0; index < n_datasets; index++) {
			DEBUG(5,"Image number " << index);
			sds_id = SDselect(sd_id, index);
			SDgetinfo(sds_id, name, &rank, dim_sizes, &num_type, &attributes);
			DEBUG(5,"	  rank " << rank << " attributes " << attributes << " : " << num_type);
			for (i=0;i<rank;i++) {
				DEBUG(5,"	  " << i << " = " << dim_sizes[i]);
			}
			
			int surf=1;
			int32 start[rank], edges[rank];
			for (i=0;i<rank;i++) {
				surf*=dim_sizes[i];
				start[i]=0;
				edges[i]=dim_sizes[i];
			}
			vector<char> data;
			switch (num_type) {
				case 3:
				case 4:
				case 20:
				case 21:
					data.resize(surf*sizeof(char));
					break;
				case 5:
					data.resize(surf*sizeof(float));
					break;
				case 6:
					data.resize(surf*sizeof(double));
					break;
				case 22:
				case 23:
					data.resize(surf*sizeof(short));
					break;
				case 24:
				case 25:
					data.resize(surf*sizeof(int));
					break;
			}
			
			if (data.size()>0) {
				status=SDreaddata(sds_id,start,NULL,edges,(VOIDP)&data[0]);
				if (status!=FAIL && rank>1) {
					int numMat=(rank==2?1:dim_sizes[0]);
					if (dim_sizes[rank-1]*dim_sizes[rank-2]>0) {
						DEBUG(5,"	  num mat " << numMat << " " << dim_sizes[rank-1]<< " x " << dim_sizes[rank-2]);
						for (i=0;i<numMat;i++) {
							DEBUG(5,"	  mat " << i);
							nPhysImageF<double> *my_data = new nPhysImageF<double>(dim_sizes[rank-1],dim_sizes[rank-2],0.0,"hdf4");
							for (size_t k=0;k<my_data->getSurf();k++) {
								switch (num_type) {
									case 3:
										my_data->set(k,((char *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
										break;
									case 4:
										my_data->set(k,((unsigned char *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
										break;
									case 20:
										my_data->set(k,(int)((char *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
										break;
									case 21:
										my_data->set(k,(int)((unsigned char *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
										break;
									case 5:
										my_data->set(k,(float) ((float *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
										break;
									case 6:
										my_data->set(k,((double *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
										break;
									case 22:
										my_data->set(k,((short *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
										break;
									case 23:
										my_data->set(k,((unsigned short *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
										break;
									case 24:
										my_data->set(k,((int *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
										break;
									case 25:
										my_data->set(k,((unsigned int *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
										break;
								}
							}
							double origin[2];
							int attr_origin=SDfindattr(sds_id, "physOrigin");
							if (attr_origin!=-1) {
								if (SDreadattr(sds_id,attr_origin,origin)==0) my_data->set_origin(origin[0],origin[1]);
							}
							double scale[2];
							int attr_scale=SDfindattr(sds_id, "physScale");
							if (attr_scale!=-1) {
								if (SDreadattr(sds_id,attr_scale,scale)==0) my_data->set_scale(scale[0],scale[1]);
							}
							if (i==1 && !background) {
								background=my_data;
							} else {
								imagelist.push_back(my_data);
							}
						}
						
					}
				}
			}
			
			SDendaccess(sds_id);
		}
		
		if (background) {
			for (vector<nPhysImageF<double> *>::iterator it=imagelist.begin() ; it < imagelist.end(); it++ ) {
				*(*it)=(*(*it))-(*background);
			}
			delete background;
		}
		
		/* Terminate access to the SD interface and close the file. */
		SDend(sd_id);
	}
#endif
	return imagelist;
}


int phys_write_HDF4(nPhysImageF<double> *phys, const char* fname) {
#if defined(HAVE_LIBMFHDF) || defined(HAVE_LIBMFHDFDLL)	
	if (phys) {
		intn istat=0;
		int32 sd_id = SDstart(fname, DFACC_CREATE);
		if (sd_id != FAIL) {
			int32 start[2], dimsizes[2];
			dimsizes[0]=phys->getH();
			dimsizes[1]=phys->getW();
			start[0]=0;
			start[1]=0;
			int32 sds_id=SDcreate(sd_id, phys->getName().c_str(), DFNT_FLOAT64, 2, dimsizes);
			comp_info c_info;
			c_info.deflate.level=6;
			istat+=SDsetcompress(sds_id, COMP_CODE_DEFLATE, &c_info);
			istat+=SDwritedata(sds_id, start, NULL, dimsizes, (VOIDP)phys->Timg_buffer);
			double data[2];
			data[0]=phys->get_origin().x();
			data[1]=phys->get_origin().y();
			istat+=SDsetattr(sds_id, "physOrigin", DFNT_FLOAT64, 2, data);
			data[0]=phys->get_scale().x();
			data[1]=phys->get_scale().y();
			istat+=SDsetattr(sds_id, "physScale", DFNT_FLOAT64, 2, data);
			istat+=SDendaccess(sds_id);
			istat += SDend(sd_id);
		}
		return istat;
	}
#endif
	return -1;
}


int inflate(FILE *source, FILE *dest)
{
    int ret;
    unsigned have;
    z_stream strm;
    vector<unsigned char> in(CHUNK), out(CHUNK);
    
    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
	ret = inflateInit2 (&strm,windowBits | GZIP_ENCODING);
    if (ret != Z_OK)
        return ret;
    
    /* decompress until deflate stream ends or end of file */
    do {
        strm.avail_in = fread(&in[0], 1, CHUNK, source);
        if (ferror(source)) {
            (void)inflateEnd(&strm);
            return Z_ERRNO;
        }
        if (strm.avail_in == 0)
            break;
        strm.next_in = &in[0];
        
        /* run inflate() on input until output buffer not full */
        do {
            strm.avail_out = CHUNK;
            strm.next_out = &out[0];
            ret = inflate(&strm, Z_NO_FLUSH);
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            switch (ret) {
				case Z_NEED_DICT:
					ret = Z_DATA_ERROR;     /* and fall through */
					break;
				case Z_DATA_ERROR:
				case Z_MEM_ERROR:
					(void)inflateEnd(&strm);
					return ret;
            }
            have = CHUNK - strm.avail_out;
            if (fwrite(&out[0], 1, have, dest) != have || ferror(dest)) {
                (void)inflateEnd(&strm);
                return Z_ERRNO;
            }
        } while (strm.avail_out == 0);
        
        /* done when inflate() says it's done */
    } while (ret != Z_STREAM_END);
    
    /* clean up and return */
    (void)inflateEnd(&strm);
    return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}

string gunzip (string filezipped) {
    
    // 	string fileunzipped(tmpnam(NULL));
    // 	if (!fileunzipped.empty()) {
	
    // 		string fileunzipped_ext=filezipped;
    // 		size_t last_idx = fileunzipped_ext.find_last_of(".");
    // 		if (string::npos != last_idx) {
    // 			fileunzipped_ext.erase(last_idx,filezipped.size());
    // 		} else {
    // 			return string();
    // 		}
    // 		last_idx = fileunzipped_ext.find_last_of(".");
    // 		if (string::npos != last_idx) {
    // 			fileunzipped_ext.erase(0,last_idx);
    // 		} else {
    // 			return string();
    // 		}
    // 		fileunzipped.append(fileunzipped_ext);
    
    string fileunzipped=filezipped;
    size_t last_idx = filezipped.find_last_of(".");
    
    if (string::npos == last_idx) return string();
    
    fileunzipped.erase(last_idx,filezipped.size());
    
    DEBUG(fileunzipped);
    
    FILE *filein;
    filein = fopen(filezipped.c_str(),"rb");
    if (filein == NULL) return string();
    
    FILE *fileout;
    fileout = fopen(fileunzipped.c_str(),"wb");
    if (fileout == NULL) return string();
    
    if (inflate(filein, fileout) != Z_OK ) {
        unlink(fileunzipped.c_str());
        return string();
    }
    fclose(filein);
    fclose(fileout);
    // 	}	
	return fileunzipped;
}

std::vector <nPhysImageF<double> *> phys_open(std::string fname) {
	std::vector <nPhysImageF<double> *> retPhys;
	size_t last_idx=0;
	
	string ext=fname;
	last_idx = fname.find_last_of(".");
	if (string::npos != last_idx) {
		ext.erase(0,last_idx + 1);
	}
	
	transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	
	string name=fname;
	last_idx = fname.find_last_of("\\/");
	if (std::string::npos != last_idx) {
		name.erase(0,last_idx + 1);
	}
	
	DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " << name << " " << ext);
	
	nPhysD *datamatrix=NULL;
	if (ext=="pgm") {
#ifdef HAVE_LIBNETPBM
		datamatrix = new nPhysD;
		*datamatrix = physGray_pgm(fname.c_str());
#else
		WARNING("nPhysImage was compiled without netpbm library");
#endif
	} else if (ext=="txt") {
		// FIXME: questo e' un baco bastardo: ATTENZIONE! no deep copy when passing reference from
		// matrices of the same type!
		//datamatrix = new nPhysD;
		//*datamatrix = physDouble_txt(fname.toStdString());
		//
		//anzi, forse una soluzione semplice esiste: castare Double ad altro nome nei
		//costruttori specializzati. a revoir.
		datamatrix = new physDouble_txt(fname.c_str());
	} else if (ext.substr(0,3)=="tif") {
		datamatrix = new physDouble_tiff(fname.c_str());
	} else if (ext=="spe") {
		vector <nPhysImageF<double> *> imagelist=phys_open_spe(fname);
		retPhys.insert(retPhys.end(), imagelist.begin(), imagelist.end());
	} else if (ext=="inf") {
		vector <nPhysImageF<double> *> imagelist=phys_open_inf(fname);
		retPhys.insert(retPhys.end(), imagelist.begin(), imagelist.end());
	} else if (ext=="sif") {
		datamatrix = new nPhysD;
		*datamatrix = physInt_sif(fname.c_str());
	} else if (ext=="b16") {
		datamatrix = new nPhysD;
		*datamatrix = physShort_b16(fname.c_str());
	} else if (ext=="img") {
		datamatrix = new physDouble_img(fname);
	} else if (ext=="imd") {
		datamatrix = new nPhysD;
		*datamatrix = physUint_imd(fname.c_str());
		phys_divide(*datamatrix,1000.);
	} else if (ext.substr(0,3)=="fit") {
		vector <nPhysImageF<double> *> imagelist=phys_open_fits(fname);
		retPhys.insert(retPhys.end(), imagelist.begin(), imagelist.end());
	} else if (ext=="hdf") {
		vector <nPhysImageF<double> *> imagelist=phys_open_HDF4(fname);
		retPhys.insert(retPhys.end(), imagelist.begin(), imagelist.end());
	} else if (ext=="neu") {
		vector <nPhysImageF<double> *> imagelist=phys_resurrect_binary(fname);
		retPhys.insert(retPhys.end(), imagelist.begin(), imagelist.end());
	} else if (ext=="gz") {
		string filenameunzipped = gunzip(fname);
		if ((!filenameunzipped.empty()) && (filenameunzipped!=fname)) {
			vector <nPhysImageF<double> *> imagelist=phys_open(filenameunzipped);
			unlink(filenameunzipped.c_str());
			for (vector<nPhysImageF<double> *>::iterator it=imagelist.begin() ; it < imagelist.end(); it++ ) {
				(*it)->setName("");
				(*it)->setShortName("");
			}
			retPhys.insert(retPhys.end(), imagelist.begin(), imagelist.end());
		}
		DEBUG(filenameunzipped);
	}
    
	if (datamatrix) retPhys.push_back(datamatrix);
	for (size_t i=0;i<retPhys.size();i++) {
		
		std::ostringstream ss;
		if (retPhys.size()>1) {
			ss << i+1 << " ";
		}
		
		DEBUG( "<" << retPhys[i]->getName() << "> <" <<  retPhys[i]->getShortName() << ">");
		
		// if Name and ShortName are set, don't change them
		if (retPhys[i]->getName().empty())
			retPhys[i]->setName(ss.str()+fname);
		
 		if (retPhys[i]->getShortName().empty())
			retPhys[i]->setShortName(ss.str()+name);
		
		DEBUG( "<" << retPhys[i]->getName() << "> <" <<  retPhys[i]->getShortName() << ">");
		
		retPhys[i]->setFromName(fname);
 		retPhys[i]->setType(PHYS_FILE);
	}
	return retPhys;
}
