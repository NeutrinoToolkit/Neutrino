/*
 *
 *    Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 * 
 *    This file is part of nPhysImage library.
 *
 *    nPhysImage is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    nPhysImage is distributed in the hope that it will be useful,
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
#include "nPhysFormats.h"
#include "NaNStream.h"
#include "bidimvec.h"
#include "nPhysMaths.h"
#include <time.h>
#include <zlib.h>

#ifdef HAVE_LIBTIFF
#define int32 tiff_int32
#define uint32 tiff_uint32
#include <tiff.h>
#include <tiffio.h>
#undef int32
#undef uint32
#endif

#ifdef HAVE_LIBCFITSIO
#include "fitsio.h"
#endif

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
	
	do {
		getline (ifile, tline);
	} while ((tline.find('#') != string::npos) || (tline.find_first_not_of(' ') == tline.find_last_not_of(' ')));
	
	stringstream ss(tline);
	string word;
   	while( ss >> word ) ++ncols;
	
	DEBUG(5,"file has "<<nlines<<" lines and "<<ncols<<" cols");
	
	//physImage myimg(ncols, nlines);
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
	   		Timg_buffer[row*w+col]= strtod(word.c_str(),NULL);
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
			sline>>Timg_buffer[row*getW()+col]>>ch;
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
			(Timg_matrix[i])[j] = (int)((readbuf[i])[j]);
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
	

physInt_sif::physInt_sif(const char *ifilename)
: nPhysImageF<int>(string(ifilename), PHYS_FILE)
{
	// Andor camera .sif file
	//
	
	char ptr[5];
	int temp;
	char *readb;
	char *rowptr;
	int w, h;
	
	string temp_string;

	ifstream ifile(ifilename, ios::in | ios::binary);
	
	ifile.read(ptr,5);
	
	if ( strncmp(ptr,"Andor",5) != 0 ) {
		WARNING("Does not start with Andor "<<ifilename);
		return;
	}
	
	int bpp = 4;
	
	WARNING("fixed bpp: "<<bpp);
	
	// matrix informations on line 5
	for (size_t i=0; i<5; i++)
		getline(ifile, temp_string);
	
	stringstream ss(temp_string);
	ss>>w;
	ss>>h;
	
	this->resize(w, h);
	
	DEBUG(5,"width: "<<getW());
	DEBUG(5,"height: "<<getH());
	
	// At least two versions of .sif have been observed, with 22 or 25 lines before the raw part
	// Last line is usually a single 0 on a terminated line
	ifile.seekg(0, ios::beg);
	for (size_t i=0; i<22; i++)
		getline(ifile, temp_string);
	
	DEBUG(5,"raw init @ "<<ifile.tellg());
	
	if (strcmp(temp_string.c_str(), "0") != 0) {
		DEBUG(5,"Old Andor format, workaround!");
		while (strcmp(temp_string.c_str(), "0") != 0)
			getline(ifile, temp_string);
	}
	
	// get data
	
	readb = new char [getW()*bpp];
	for (size_t i=0; i<getH(); i++) {
		
		memset(readb,0,getW()*bpp);
		rowptr = readb;
		ifile.read(readb,getW()*bpp);
		
		for (size_t j=0; j<getW(); j++) {
			temp = (int) ( *((float *)&readb[bpp*j]) );
			(Timg_matrix[i])[j] = temp;
		}
	}
	
	delete [] readb;
	TscanBrightness();
	
	ifile.close();
	
}

physShort_b16::physShort_b16(const char *ifilename)
: nPhysImageF<short>(string(ifilename), PHYS_FILE)
{
	
	char *ptr = new char[3], tempch;
	char *readb;
	char *rowptr;
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
		rowptr = readb;
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

physShort_img::physShort_img(string ifilename)
: nPhysImageF<unsigned short>(ifilename, PHYS_FILE)
{
	// Hamamatsu Streak Camera
	//
	unsigned short buffer;
	int bytes=sizeof(unsigned short);
	
	FILE *fin;
	int w=0;
	int h=0;
	int skipbyte=0;
	
	if ((fin=fopen(ifilename.c_str(),"rb")) == NULL) return;
	
	if (fread (&buffer,bytes,1,fin)==1) {
		if (buffer == 19785) { //hamamatsu TODO: check all the stuff inside
			fread (&buffer,bytes,1,fin);
			skipbyte=buffer;
			fread (&buffer,bytes,1,fin);
			w=buffer;
			fread (&buffer,bytes,1,fin);
			h=buffer;
			fseek(fin, 56, SEEK_CUR);
			char *buffer2=new char[skipbyte];
			int i = fread (buffer2,1,skipbyte,fin);
			buffer2[i] ='\0';
            DEBUG(5,string(buffer2));
			delete [] buffer2;
//			fseek(fin, skipbyte, SEEK_CUR);
		} else if (buffer == 512) { // ARP blue ccd camera w optic fibers...
			fread (&buffer,bytes,1,fin);
			if (buffer==7) {
				fread (&buffer,bytes,1,fin);
				skipbyte=buffer+4;
				fread (&buffer,bytes,1,fin);
				w=buffer;
				fread (&buffer,bytes,1,fin);
				h=buffer;
			}
			fread (&buffer,bytes,1,fin);
		} else { // LIL images
			rewind(fin);
			unsigned int lil_header[4];
			fread (lil_header,sizeof(unsigned int),4,fin);
			if (lil_header[0]==2 && lil_header[3]==1) {
				// lil_header[0] = dimension of the matrix
				// lil_header[3] = kind of data (1=unisgned short, 2=long, 3= float, 4=double)
				w=lil_header[1];
				h=lil_header[2];
			}
		}
	}
	DEBUG(5, "w "<< w << " h "<<h);
	if (w*h>0) {
		this->resize(w, h);
		fread(Timg_buffer,bytes,w*h,fin);
	}	
	fclose(fin);
}

physShort_imd::physShort_imd(string ifilename)
: nPhysImageF<unsigned int>(ifilename, PHYS_FILE)
{
	// Optronics luli
	//
	unsigned short buffer_header;
	unsigned int buffer;
	FILE *fin;
	unsigned short w=0;
	unsigned short h=0;
	
	if ((fin=fopen(ifilename.c_str(),"rb")) == NULL) return;
	
	if (fread (&buffer_header,sizeof(buffer_header),1,fin)==1) {
		DEBUG(5,"header " << buffer_header);
	}
	if (fread (&buffer_header,sizeof(buffer_header),1,fin)==1) {
		DEBUG(5,"width " << buffer_header);
		w=buffer_header;
	}
	if (fread (&buffer_header,sizeof(buffer_header),1,fin)==1) {
		DEBUG(5,"height " << buffer_header);
		h=buffer_header;
	}
	
	this->resize(w, h);
    fread(Timg_buffer,sizeof(unsigned int),w*h,fin);
	
	
	fclose(fin);
	TscanBrightness();
	DEBUG(5,"object_name: "<<getName());
	DEBUG(5,"filename: "<<getName());
	
	
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


#define CHUNK 0x4000
#define windowBits 15
#define GZIP_ENCODING 16

// dump out for state save
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
		//WARNING("Zlib a bad status of " << status);
		exit (EXIT_FAILURE);
	}

    	strm.next_in = (unsigned char *) my_phys->Timg_buffer;
    	strm.avail_in = my_phys->getSurf()*sizeof(double);
	strm.avail_out = buffer_size;
	strm.next_out = out;
	status= deflate (& strm, Z_FINISH);
	
	if (status < 0) {
		WARNING("Zlib a bad status of ");
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
		unsigned short samples, compression, config;
// 		unsigned short *extra;
		TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
		TIFFGetField(tif, TIFFTAG_COMPRESSION, &compression);
		TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &config);
		
		
		DEBUG("COMPRESSION_NONE " << compression << " " << COMPRESSION_NONE);
		DEBUG("PLANARCONFIG_CONTIG " << config << " " << PLANARCONFIG_CONTIG);
		DEBUG("SAMPLES" << samples);
// 		TIFFGetField(tif, TIFFTAG_EXTRASAMPLES, extra);
// 
// 		for (int k=0;k<samples;k++) {
// 			DEBUG("extra " << k << "  " << extra[k]);
// 		}
// 		delete extra;
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
				setShortName(desc);
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
phys_write_tiff(nPhysImageF<double> *my_phys, const char *ofilename, int bytes) {
#ifdef HAVE_LIBTIFF
	TIFF* tif = TIFFOpen(ofilename, "w");
	if (tif && my_phys) {
		TIFFSetField(tif, TIFFTAG_SUBFILETYPE, 0);
		TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1);
		TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
		TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
		TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
		TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
		TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
 		TIFFSetField(tif, TIFFTAG_DOCUMENTNAME, my_phys->getName().c_str());
 		TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, my_phys->getShortName().c_str());
 		TIFFSetField(tif, TIFFTAG_MAKE,  my_phys->getFromName().c_str());
 		TIFFSetField(tif, TIFFTAG_SOFTWARE, "neutrino");
 		TIFFSetField(tif, TIFFTAG_COPYRIGHT, "http::/web.luli.polytchnique.fr/neutrino");
		
		TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, my_phys->getW());
		TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, my_phys->getH());
		TIFFSetField(tif, TIFFTAG_IMAGELENGTH, my_phys->getH());
		TIFFSetField(tif, TIFFTAG_XRESOLUTION, my_phys->get_scale().x());
	    TIFFSetField(tif, TIFFTAG_YRESOLUTION, my_phys->get_scale().y());
		
		TIFFSetField(tif, TIFFTAG_XPOSITION, my_phys->get_origin().x());
		TIFFSetField(tif, TIFFTAG_YPOSITION, my_phys->get_origin().y());
		TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8*bytes);
		unsigned char *buf = (unsigned char *) _TIFFmalloc(TIFFScanlineSize(tif));
		for (size_t j = 0; j < my_phys->getH(); j++) {
 			for (size_t i=0; i<my_phys->getW(); i++) {
 			    if (bytes==sizeof(float)) {
     				((float*)buf)[i]=(float)my_phys->point(i,j) ;
     			} else if (bytes==sizeof(int)){
     				((int*)buf)[i]=(float)my_phys->point(i,j) ;
     			}
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

std::vector <nPhysImageF<double> *> phys_open_inf(std::string ifilename) {
	std::vector <nPhysImageF<double> *> imagelist;
	ifstream ifile(ifilename.c_str(), ios::in);
	if (ifile) {
		string ifilenameimg=ifilename;
		ifilenameimg.resize(ifilenameimg.size()-3);
		ifilenameimg = ifilenameimg+"img";
		ifstream ifileimg(ifilenameimg.c_str(), ios::in);
		if (ifileimg) {
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
			nPhysImageF<double> *linearized = new nPhysImageF<double>(w,h,0.0,(string("(linearized) ")+ifilename).c_str());
			linearized->setType(PHYS_FILE);
			linearized->setFromName(ifilename);
			imagelist.push_back(linearized);
			nPhysImageF<double> *original = new nPhysImageF<double>(w,h,0.0,ifilename.c_str());
			original->setType(PHYS_FILE);
			original->setFromName(ifilename);
			imagelist.push_back(original);
			
			getline(ifile,line);
			double sensitivity=atoi(line.c_str());
			getline(ifile,line);
			double latitude=atoi(line.c_str());
			DEBUG(5,"resx = "<< resx);
			DEBUG(5,"resy = "<< resy);
			DEBUG(5,"bit  = "<< bit);
			DEBUG(5,"w    = "<< w);
			DEBUG(5,"h    = "<< h);
			DEBUG(5,"sens = "<< sensitivity);
			DEBUG(5,"lat  = "<< latitude);
			// FIXME: this is awful
			switch (bit) {
				case 8: {
					char *buf=new char[original->getSurf()];
					ifileimg.read(buf, sizeof(char)*original->getSurf());
					for (size_t i=0;i<original->getSurf();i++) {
						original->Timg_buffer[i]=swap_endian<char>(buf[i]);
						linearized->Timg_buffer[i] = ((resx*resy)/10000.0) * (4000.0/sensitivity) * pow(10,latitude*(original->Timg_buffer[i]/pow(2.0,bit)-0.5));
					}
					delete [] buf;
					break;
				}
				case 16: {
					unsigned short *buf=new unsigned short[original->getSurf()];
					ifileimg.read((char*)buf, sizeof(unsigned short)*original->getSurf());
					for (size_t i=0;i<original->getSurf();i++) {
						original->Timg_buffer[i]=swap_endian<unsigned short>(buf[i]);
						linearized->Timg_buffer[i] = ((resx*resy)/10000.0) * (4000.0/sensitivity) * pow(10,latitude*(original->Timg_buffer[i]/pow(2.0,bit)-0.5));
					}
					delete [] buf;
					break;
				}
			}
		}
		ifileimg.close();
	}
	ifile.close();
    if (imagelist.size()==2) {
        string name=ifilename;
        size_t last_idx = ifilename.find_last_of("\\/");
        if (std::string::npos != last_idx) {
            name.erase(0,last_idx + 1);
        }
        imagelist.at(0)->setShortName("(linearized)");
        imagelist.at(0)->setName(name+"(linearized)");
    }
	return imagelist;
}

physDouble_fits::physDouble_fits(string ifilename)
: nPhysImageF<double>(ifilename, PHYS_FILE) {
#ifdef HAVE_LIBCFITSIO
    fitsfile *fptr;
    char card[FLEN_CARD];
    int status = 0,  nkeys, ii;
	
    fits_open_file(&fptr, ifilename.c_str(), READONLY, &status);
	int bitpix;
	int anaxis;
	
	fits_get_img_type(fptr,&bitpix,&status);
	if (status)
        fits_report_error(stderr, status);
	
	DEBUG(5,"fits_get_img_type " << bitpix);
	fits_get_img_dim(fptr,&anaxis,&status);
	if (status)
		fits_report_error(stderr, status);
	if (anaxis==2) {
		
		long axissize[2], fpixel[2];
		
		fits_get_img_size(fptr,anaxis,axissize,&status);
		if (status)
			fits_report_error(stderr, status);
		
		long totalsize=axissize[0]*axissize[1];
		fpixel[0]=fpixel[1]=1;
		
		resize(axissize[0],axissize[1]);
		
		fits_get_hdrspace(fptr, &nkeys, NULL, &status);
		for (ii = 1; ii <= nkeys; ii++)  {
			fits_read_record(fptr, ii, card, &status); /* read keyword */
			if (status)
				fits_report_error(stderr, status);
			WARNING(ii << " : " << card);
		}
		
		fits_read_pix(fptr, TDOUBLE, fpixel, totalsize, NULL, (void *)Timg_buffer, NULL, &status);
		if (status)
			fits_report_error(stderr, status);
		
		TscanBrightness();
		
	}
	
    fits_close_file(fptr, &status);
#endif
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
        //    int ret=phys_resurrect_old_binary(datamatrix,ifile);
        //    if (ret>=0 && datamatrix->getSurf()>0) {
        //        imagelist.push_back(datamatrix);
        //    } else {

		ret=phys_resurrect_binary(datamatrix,ifile);
 		if (ret>=0 && datamatrix->getSurf()>0) {
     			imagelist.push_back(datamatrix);
		} else {
    			delete datamatrix;
	//        }
    
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

	std::cerr<<"w: "<<my_w<<std::endl;
	std::cerr<<"h: "<<my_h<<std::endl;
	std::cerr<<"s: "<<buffer_size<<std::endl;

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
	if (my_phys==NULL || my_phys->getSurf()==0) return -1;
	
	ifstream ifile(my_phys->getName().c_str(), ios::in | ios::binary);
	if (ifile.fail()) return -1;
	ifile.seekg(skipbyte);
	if (ifile.fail() || ifile.eof()) return -1;
	for (size_t i=0;i<my_phys->getSurf();i++) {
		switch (kind) {
			case 0: {
				char buf;
				ifile.read(&buf, sizeof(buf));
				my_phys->Timg_buffer[i] = endian ? swap_endian<char>(buf) : buf;
				break;
			}
			case 1: {
				unsigned short buf;
				ifile.read((char*)&buf, sizeof(buf));
				my_phys->Timg_buffer[i] = endian ? swap_endian<unsigned short>(buf) : buf;
				break;
			}
			case 2: {
				short buf;
				ifile.read((char*)&buf, sizeof(buf));
				my_phys->Timg_buffer[i] = endian ? swap_endian<short>(buf) : buf;
				break;
			}
			case 3: {
				unsigned int buf;
				ifile.read((char*)&buf, sizeof(buf));
				my_phys->Timg_buffer[i] = endian ? swap_endian<unsigned int>(buf) : buf;
				break;
			}
			case 4: {
				int buf;
				ifile.read((char*)&buf, sizeof(buf));
				my_phys->Timg_buffer[i] = endian ? swap_endian<int>(buf) : buf;
				break;
			}
			case 5: {
				float buf;
				ifile.read((char*)&buf, sizeof(buf));
				my_phys->Timg_buffer[i] = endian ? swap_endian<float>(buf) : buf;
				break;
			}
			case 6: {
				double buf;
				ifile.read((char*)&buf, sizeof(buf));
				my_phys->Timg_buffer[i] = endian ? swap_endian<double>(buf) : buf;
				break;
			}
			default:
				return -1;
				break;
		}
		// 		cerr << kind << "\t" <<  endian << "\t" <<  i << "\t" <<  my_phys->Timg_buffer[i] << endl;
	}
	ifile.close();
	my_phys->TscanBrightness();
	return 0;
}

//! open HDF4 file (works for: Omega LLR  visar images)
vector <nPhysImageF<double> *> phys_open_HDF4(string fname) {
	vector <nPhysImageF<double> *> imagelist;
	WARNING("HERE");
#ifdef HAVE_LIBMFHDF
	WARNING("HERE2");
	int32 sd_id, sds_id, n_datasets, n_file_attrs, index,status ;
	int32 dim_sizes[3];
	int32 rank, num_type, attributes, istat;
	int32 i;
	
	char name[64];
	
	nPhysImageF<double> *background=NULL;
	/* Open the file and initiate the SD interface. */
	sd_id = SDstart(fname.c_str(), DFACC_READ);
	if (sd_id != FAIL) {
		/* Determine the contents of the file. */
		istat = SDfileinfo(sd_id, &n_datasets, &n_file_attrs);
		DEBUG(5,"datasets " << n_datasets);
		for (index = 0; index < n_datasets; index++) {
			DEBUG(5,"Image number " << index);
			sds_id = SDselect(sd_id, index);
			istat = SDgetinfo(sds_id, name, &rank, dim_sizes, &num_type, &attributes);
			DEBUG(5,"      rank " << rank << " attributes " << attributes << " : " << num_type);
			for (i=0;i<rank;i++) {
				DEBUG(5,"      " << i << " = " << dim_sizes[i]);
			}
			
			int surf=1;
			int32 start[rank], edges[rank];
			for (i=0;i<rank;i++) {
				surf*=dim_sizes[i];
				start[i]=0;
				edges[i]=dim_sizes[i];
			}
			char *data=NULL;
			switch (num_type) {
				case 3:
				case 4:
				case 20:
				case 21:
					data =new char [surf*sizeof(char)];
					break;
				case 5:
					data =new char [surf*sizeof(float)];
					break;
				case 6:
					data =new char [surf*sizeof(double)];
					break;
				case 22:
				case 23:
					data =new char [surf*sizeof(short)];
					break;
				case 24:
				case 25:
					data =new char [surf*sizeof(int)];
					break;
			}
			
			if (data) {
				status=SDreaddata(sds_id,start,NULL,edges,(VOIDP)data);
				if (status!=FAIL && rank>1) {
					int numMat=(rank==2?1:dim_sizes[0]);
					if (dim_sizes[rank-1]*dim_sizes[rank-2]>0) {
						DEBUG(5,"      num mat " << numMat << " " << dim_sizes[rank-1]<< " x " << dim_sizes[rank-2]);
						for (i=0;i<numMat;i++) {
							DEBUG(5,"      mat " << i);
							nPhysImageF<double> *my_data = new nPhysImageF<double>(dim_sizes[rank-1],dim_sizes[rank-2],0.0,"hdf4");
							for (size_t k=0;k<my_data->getSurf();k++) {
								switch (num_type) {
									case 3:
										my_data->Timg_buffer[k]=((char *)data)[k+i*dim_sizes[1]*dim_sizes[2]];
										break;
									case 4:
										my_data->Timg_buffer[k]=((unsigned char *)data)[k+i*dim_sizes[1]*dim_sizes[2]];
										break;
									case 20:
										my_data->Timg_buffer[k]=(int)((char *)data)[k+i*dim_sizes[1]*dim_sizes[2]];
										break;
									case 21:
										my_data->Timg_buffer[k]=(int)((unsigned char *)data)[k+i*dim_sizes[1]*dim_sizes[2]];
										break;
									case 5:
										my_data->Timg_buffer[k]=(float) ((float *)data)[k+i*dim_sizes[1]*dim_sizes[2]];
										break;
									case 6:
										my_data->Timg_buffer[k]=((double *)data)[k+i*dim_sizes[1]*dim_sizes[2]];
										break;
									case 22:
										my_data->Timg_buffer[k]=((short *)data)[k+i*dim_sizes[1]*dim_sizes[2]];
										break;
									case 23:
										my_data->Timg_buffer[k]=((unsigned short *)data)[k+i*dim_sizes[1]*dim_sizes[2]];
										break;
									case 24:
										my_data->Timg_buffer[k]=((int *)data)[k+i*dim_sizes[1]*dim_sizes[2]];
										break;
									case 25:
										my_data->Timg_buffer[k]=((unsigned int *)data)[k+i*dim_sizes[1]*dim_sizes[2]];
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
				delete [] data;
			}
			
			istat = SDendaccess(sds_id);
		}
		
		if (background) {
			for (vector<nPhysImageF<double> *>::iterator it=imagelist.begin() ; it < imagelist.end(); it++ ) {
				*(*it)=(*(*it))-(*background);
			}
			delete background;
		}
		
		/* Terminate access to the SD interface and close the file. */
		istat = SDend(sd_id);
	}
#endif
	return imagelist;
}



nPhysImageF<double> * phys_open_HDF5(std::string fileName, std::string dataName) {
    nPhysD *my_data=NULL;
	DEBUG(fileName << " " << dataName);
#ifdef HAVE_LIBHDF5
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
				strName.erase(0,strName.find_last_of("/"));
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
#endif
	return my_data;
}


#ifdef HAVE_LIBHDF5
void scan_hdf5_attributes(hid_t aid, nPhysImageF<double> *my_data){
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
			double *val=new double[nelem];
			if (H5Aread(aid, nativeType, (void*)val) >= 0) {
				if (my_data && nelem==2) {
					if (strcmp(attrName,"physOrigin")==0) {
						my_data->set_origin(val[0],val[1]);
					} else if (strcmp(attrName,"physScale")==0) {
						my_data->set_scale(val[0],val[1]);
					}
				}
			}
			delete [] val;
		}
	} else if (classAType ==  H5T_INTEGER) {
		int nelem=aInfo.data_size/sizeof(int);
		int *val=new int[nelem];
		if (H5Aread(aid, nativeType, (void*)val) >= 0) {
		}
		delete [] val;
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
#endif

int phys_write_HDF4(nPhysImageF<double> *phys, const char* fname) {
#ifdef HAVE_LIBMFHDF
	if (phys) {
		intn istat=0;
		int32 sd_id = SDstart(fname, DFACC_CREATE);
		if (sd_id != FAIL) {
			istat += phys_write_HDF4_SD(phys,sd_id);
			istat += SDend(sd_id);
		}
		return istat;
	}
#endif
	return -1;
}

#ifdef HAVE_LIBMFHDF
int phys_write_HDF4_SD(nPhysImageF<double> * phys, int sd_id) {
	intn istat=0;
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
	return sds_id;
}
#endif

// int phys_write_HDF5(nPhysImageF<double> *phys, string fname) {
// #ifdef __phys_HDF
// 	hid_t       file_id;
// 	hsize_t     dims[2]={phys->getH(),phys->getW()};
// 	herr_t      status;
// 
// 	file_id = H5Fcreate (fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
//    	status = H5LTmake_dataset(file_id,"neutrino",2,dims,H5T_NATIVE_DOUBLE,phys->Timg_buffer);
// 	status= H5LTset_attribute_string(file_id,"/","version", __VER);
// 	double data[2];
// 	data[0]=phys->get_origin().x();
// 	data[1]=phys->get_origin().y();
// 	status= H5LTset_attribute_double(file_id,"neutrino","physOrigin", data, 2);
// 	data[0]=phys->get_scale().x();
// 	data[1]=phys->get_scale().y();
// 	status= H5LTset_attribute_double(file_id,"neutrino","physScale", data, 2);
// 	status= H5LTset_attribute_string(file_id,"neutrino","physName", phys->getName().c_str());
// 	status= H5LTset_attribute_string(file_id,"neutrino","physShortName", phys->getShortName().c_str());
// 
//    	cerr << "[hdf5] status " << status << endl;
//  	status = H5Fclose (file_id);
// #endif
// }

bool phys_is_HDF5(std::string fileName) {
#ifdef HAVE_LIBHDF5
	hid_t fid = H5Fopen (fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	if (fid >= 0) {
		H5Fclose(fid);
		return true;
	}
#endif
	return false;
}

int phys_write_HDF5(nPhysImageF<double> *phys, string fname) {
#ifdef HAVE_LIBHDF5
	if (phys) {
		if (H5Zfilter_avail(H5Z_FILTER_DEFLATE)) {
			unsigned int    filter_info;
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
#endif
	return -1;
}

// -------------------------------


std::vector <nPhysImageF<double> *> phys_open(std::string fname, std::string optarg) {
    std::vector <nPhysImageF<double> *> retPhys;
    size_t last_idx=0;

    WARNING("DEBUG in warning for release!");
	
	string ext=fname;
    last_idx = fname.find_last_of(".");
	if (string::npos != last_idx) {
		ext.erase(0,last_idx + 1);
	}

	string name=fname;
    last_idx = fname.find_last_of("\\/");
	if (std::string::npos != last_idx) {
		name.erase(0,last_idx + 1);
	}

    WARNING(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
	
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
		datamatrix = new nPhysD;
		*datamatrix = physShort_img(fname);
	} else if (ext=="imd") {
		datamatrix = new nPhysD;
		*datamatrix = physShort_imd(fname.c_str());
		phys_divide(*datamatrix,1000.);
	} else if (ext.substr(0,3)=="fit") {
		datamatrix = new physDouble_fits(fname);
	} else if (ext=="hdf") {
		vector <nPhysImageF<double> *> imagelist=phys_open_HDF4(fname);
        retPhys.insert(retPhys.end(), imagelist.begin(), imagelist.end());
	} else if (ext=="h5") {
		if (!optarg.empty()) datamatrix = phys_open_HDF5(fname,optarg);
	} else if (ext=="neu") {
		vector <nPhysImageF<double> *> imagelist=phys_resurrect_binary(fname);
        retPhys.insert(retPhys.end(), imagelist.begin(), imagelist.end());
	}
	
    WARNING("DEBUG in warning for release 2!");

	if (datamatrix) retPhys.push_back(datamatrix);
    for (size_t i=0;i<retPhys.size();i++) {
        WARNING("DEBUG in warning for release! -> "<< i );

		std::ostringstream ss;
		ss << name;
		if (retPhys.size()>1) {
			ss << " " << i+1;
		}

		// if Name and ShortName are set, don't change them
		if (retPhys[i]->getName().empty())
			retPhys[i]->setName(ss.str());
		
		if (retPhys[i]->getShortName().empty())
			retPhys[i]->setShortName(ss.str());

		retPhys[i]->setFromName(fname);
 		retPhys[i]->setType(PHYS_FILE);
	}
	return retPhys;
}
