
#include <vector>
#include <memory>
#include <cmath>
#include "bidimvec.h"
#include "nPhysExceptions.h"

#ifndef __physData
#define __physData

template <class T>
class physData : protected std::vector<T> {

public:

	physData()
	{
		width = 0;
		height = 0;


		DEBUG(10, "constructor")	;
	}

	void resize(int w, int h, T val=0)
	{
		std::vector<T>::resize(w*h, val);
		width = w; height = h;
		data_ptr = std::vector<T>::data();
	}

	inline T point(size_t x, size_t y, T nan_value=std::numeric_limits<T>::signaling_NaN()) const {
		if (x>width || y>height) return nan_value;
		return data_ptr[y*width+x];
	}

	inline T point(vec2 p, T nan_value=std::numeric_limits<T>::signaling_NaN()) const
	{ return point(p.x(), p.y(), nan_value); }

	inline T point(size_t xy, T nan_value=std::numeric_limits<T>::signaling_NaN()) const
	{
		if (xy<size()) {
			return data_ptr[xy];
		} else {
			return nan_value;
		}
	}


	T sum()
	{ T sumTot=0; for (size_t i=0; i<getSurf(); i++) sumTot+=point(i); return sumTot; }

	inline void set(unsigned int x, unsigned int y, T val) {
		if (x>width || y>height) {
			throw physData_ooAccess();
			return;
		}
		data_ptr[y*width+x] = val;
	}

	inline void set(vec2 p, T val) {
		set(p.x(), p.y(), val);
	}

	inline void set(size_t xy, T val) {
		if (xy>size()) {
			throw physData_ooAccess();
			return;
		}
		data_ptr[xy] = val;
	}

	size_t size() const
	{ return std::vector<T>::size(); }

	vec2 vsize() const
	{ return vec2(width, height); }


	inline size_t getW() const
	{ return width; }

	inline size_t getH() const
	{ return height; }

	inline size_t getSurf() const
	{ return size(); }

	// get row/col
	void get_Trow(size_t, size_t, std::vector<T> &);
	void set_Trow(size_t, size_t, std::vector<T> &);

	const T *data_pointer()
	{ return data_ptr; }

	typename std::vector<T>::iterator buf_itr()
	{ return std::vector<T>::begin(); }

	void swap_vector(size_t w, size_t h, std::vector<T> &vec)
	{
		if (width*height != vec.size()) {
			DEBUG("WARNING: size mismatch. w:"<<width<<", h:"<<height<<", size: "<<vec.size());
			return;
		}
		width = w;
		height = h;
		std::vector<T>::swap(vec);
	}


protected:
	unsigned int width, height;

private:
	T *data_ptr;

};


//! get_Trow specialization. Row copy/move can be faster for the use of bulk copy methods
//! WARNING: uses % on index and offset (for more interesting solutions)
template<class T> void
physData<T>::get_Trow(size_t index, size_t offset, std::vector<T> &vec) {

	vec.resize(getW());

	typename std::vector<T>::iterator vitr;
	typename std::vector<T>::iterator begitr;

	begitr = this->begin();

	offset = offset%getW();

	//vitr = std::copy(Timg_matrix[index%getH()] + offset, Timg_matrix[index%getH()] + getW(), vec.begin());
	vitr = std::copy(begitr+(index%getH()) + offset, begitr + (index%getH()) + getW(), vec.begin());
	if (offset > 0)
		std::copy(begitr+(index%getH()), begitr+(index%getH()) + offset, vitr);
	//std::copy(Timg_matrix[index%getH()], Timg_matrix[index%getH()] + offset, vitr);

}

template<class T> void
physData<T>::set_Trow(size_t index, size_t offset, std::vector<T> &vec) {

	typename std::vector<T>::iterator optr;
	offset = offset%getW();

	optr = std::copy(vec.end()-offset, vec.end(), std::vector<T>::begin()+(index%getH()));
	optr = std::copy(vec.begin(), vec.end()-offset, optr);

}



#endif
