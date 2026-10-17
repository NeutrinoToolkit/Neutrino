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
#include "nHdf5io.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>

namespace nHdf5 {

namespace {

// no HDF5 error stack printed on stderr (missing objects, not HDF5 files...): errors are returned
class QuietErrors {
public:
    QuietErrors() { H5Eget_auto2(H5E_DEFAULT, &old_func, &old_data); H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); }
    ~QuietErrors() { H5Eset_auto2(H5E_DEFAULT, old_func, old_data); }
private:
    H5E_auto2_t old_func = nullptr;
    void *old_data = nullptr;
};

std::string number(double val) {
    std::ostringstream os;
    os.precision(10);
    os << val;
    return os.str();
}

std::string joinDims(const std::vector<hsize_t> &dims) {
    // fastest dimension first (W x H x N), as shown before
    std::string ret;
    for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
        if (!ret.empty()) ret += "x";
        ret += std::to_string(*it);
    }
    return ret.empty() ? "scalar" : ret;
}

std::vector<hsize_t> getDims(hid_t space) {
    int rank = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(rank > 0 ? rank : 0);
    if (rank > 0) H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    return dims;
}

bool isNumeric(hid_t type) {
    H5T_class_t cls = H5Tget_class(type);
    return cls == H5T_INTEGER || cls == H5T_FLOAT;
}

// 2D arrays of numbers (a dataset of them is read as their mean)
bool isImageArray(hid_t type) {
    if (H5Tget_class(type) != H5T_ARRAY || H5Tget_array_ndims(type) != 2) return false;
    Handle super(H5Tget_super(type), H5Tclose);
    return super.valid() && isNumeric(super);
}

// identity of an object (to stop on loops of hard links)
std::string objectKey(hid_t obj) {
#if H5_VERSION_GE(1,12,0)
    H5O_info2_t info;
    if (H5Oget_info3(obj, &info, H5O_INFO_BASIC) < 0) return std::string();
    return std::to_string(info.fileno) + ":" + std::string(reinterpret_cast<const char *>(&info.token), sizeof(info.token));
#else
    H5O_info_t info;
    if (H5Oget_info2(obj, &info, H5O_INFO_BASIC) < 0) return std::string();
    return std::to_string(info.fileno) + ":" + std::to_string(info.addr);
#endif
}

void readStrings(hid_t aid, hid_t type, hid_t space, hssize_t n, std::vector<std::string> &strings) {
    if (n <= 0) return;
    if (H5Tis_variable_str(type) > 0) {
        Handle mem(H5Tcopy(H5T_C_S1), H5Tclose);
        H5Tset_size(mem, H5T_VARIABLE);
        H5Tset_cset(mem, H5Tget_cset(type));
        std::vector<char *> buf(n, nullptr);
        if (H5Aread(aid, mem, buf.data()) >= 0) {
            for (auto p : buf) strings.push_back(p ? std::string(p) : std::string());
#if H5_VERSION_GE(1,12,0)
            H5Treclaim(mem, space, H5P_DEFAULT, buf.data());
#else
            H5Dvlen_reclaim(mem, space, H5P_DEFAULT, buf.data());
#endif
        }
    } else {
        size_t size = H5Tget_size(type);
        if (size == 0) return;
        std::vector<char> buf(n * size + 1, 0);
        Handle mem(H5Tcopy(type), H5Tclose);
        if (H5Aread(aid, mem, buf.data()) >= 0) {
            bool spacePad = (H5Tget_strpad(type) == H5T_STR_SPACEPAD);
            for (hssize_t i = 0; i < n; i++) {
                std::string str(buf.data() + i * size, size);
                str = str.substr(0, str.find('\0'));
                if (spacePad) str.erase(str.find_last_not_of(' ') + 1);
                strings.push_back(str);
            }
        }
    }
}

Attribute readAttribute(hid_t aid, const std::string &name) {
    Attribute attr;
    attr.name = name;
    Handle type(H5Aget_type(aid), H5Tclose);
    Handle space(H5Aget_space(aid), H5Sclose);
    if (!type.valid() || !space.valid()) return attr;
    attr.type = typeName(type);
    hssize_t n = H5Sget_simple_extent_npoints(space);
    if (n <= 0) return attr;
    if (isNumeric(type)) {
        // read as double whatever the size: int8/int16 overflowed an int buffer before
        attr.numbers.resize(n);
        if (H5Aread(aid, H5T_NATIVE_DOUBLE, attr.numbers.data()) < 0) attr.numbers.clear();
    } else if (H5Tget_class(type) == H5T_STRING) {
        readStrings(aid, type, space, n, attr.strings);
    }
    return attr;
}

herr_t attributeCallback(hid_t loc, const char *name, const H5A_info_t *, void *data) {
    Handle aid(H5Aopen(loc, name, H5P_DEFAULT), H5Aclose);
    if (aid.valid()) {
        static_cast<std::vector<Attribute> *>(data)->push_back(readAttribute(aid, name));
    }
    return 0;
}

std::vector<Attribute> readAttributes(hid_t loc) {
    std::vector<Attribute> attrs;
    H5Aiterate2(loc, H5_INDEX_NAME, H5_ITER_INC, nullptr, attributeCallback, &attrs);
    return attrs;
}

void scanDataset(hid_t did, Node &node) {
    node.type = Node::dataset;
    node.attributes = readAttributes(did);
    Handle type(H5Dget_type(did), H5Tclose);
    Handle space(H5Dget_space(did), H5Sclose);
    if (!type.valid() || !space.valid()) return;
    node.dims = getDims(space);
    node.typeText = typeName(type);
    node.dataText = joinDims(node.dims);
    size_t rank = node.dims.size();
    if (isNumeric(type)) {
        if (rank <= 1) node.kind = Kind::data1D;
        else if (rank == 2) node.kind = Kind::image;
        else if (rank == 3) {
            node.kind = Kind::frames;
            node.dataText += " (" + std::to_string(node.dims[0]) + " frames)";
        }
    } else if (isImageArray(type)) {
        node.kind = Kind::image;
        node.dataText += " (mean of the arrays)";
    } else if (H5Tget_class(type) == H5T_COMPOUND) {
        // only the members: the data was read here before (all of it, to show the first record)
        int nmembers = H5Tget_nmembers(type);
        node.dataText += ", " + std::to_string(nmembers) + " members";
        for (int i = 0; i < nmembers; i++) {
            Node member;
            char *mname = H5Tget_member_name(type, i);
            member.name = mname ? mname : "";
            if (mname) H5free_memory(mname);
            Handle mtype(H5Tget_member_type(type, i), H5Tclose);
            member.typeText = mtype.valid() ? typeName(mtype) : "unknown";
            member.dataText = "member";
            node.children.push_back(member);
        }
    }
}

void scanGroup(hid_t gid, Node &node, std::set<std::string> &visited, int depth) {
    node.type = Node::group;
    node.attributes = readAttributes(gid);
    H5G_info_t ginfo;
    if (H5Gget_info(gid, &ginfo) < 0) return;
    for (hsize_t i = 0; i < ginfo.nlinks; i++) {
        ssize_t len = H5Lget_name_by_idx(gid, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        if (len < 0) continue;
        std::vector<char> nameBuf(len + 1, 0);
        H5Lget_name_by_idx(gid, ".", H5_INDEX_NAME, H5_ITER_INC, i, nameBuf.data(), len + 1, H5P_DEFAULT);
        Node child;
        child.name = std::string(nameBuf.data());
        child.path = (node.path == "/" ? "" : node.path) + "/" + child.name;

        H5L_info_t linfo;
        if (H5Lget_info(gid, child.name.c_str(), &linfo, H5P_DEFAULT) < 0) {
            child.typeText = "unreadable link";
        } else if (linfo.type == H5L_TYPE_SOFT || linfo.type == H5L_TYPE_EXTERNAL) {
            std::vector<char> val(linfo.u.val_size + 1, 0);
            H5Lget_val(gid, child.name.c_str(), val.data(), val.size(), H5P_DEFAULT);
            if (linfo.type == H5L_TYPE_SOFT) {
                child.type = Node::softLink;
                child.typeText = "Link";
                child.dataText = std::string(val.data());
            } else {
                child.type = Node::externalLink;
                child.typeText = "External link";
                const char *efile = nullptr, *eobj = nullptr;
                if (H5Lunpack_elink_val(val.data(), linfo.u.val_size, nullptr, &efile, &eobj) >= 0 && efile && eobj) {
                    child.dataText = std::string(efile) + "::" + eobj;
                }
            }
        } else if (linfo.type == H5L_TYPE_HARD) {
            Handle obj(H5Oopen(gid, child.name.c_str(), H5P_DEFAULT), H5Oclose);
            if (!obj.valid()) {
                child.typeText = "cannot open";
            } else {
                switch (H5Iget_type(obj)) {
                case H5I_GROUP: {
                    std::string key = objectKey(obj);
                    if (depth > 64 || (!key.empty() && visited.count(key))) {
                        // hard link loop (or a group already shown): not scanned again
                        child.type = Node::group;
                        child.typeText = "Group (already shown)";
                    } else {
                        visited.insert(key);
                        scanGroup(obj, child, visited, depth + 1);
                        child.typeText = "Group";
                    }
                    break;
                }
                case H5I_DATASET:
                    scanDataset(obj, child);
                    break;
                case H5I_DATATYPE:
                    child.type = Node::datatype;
                    child.typeText = "Type " + typeName(obj);
                    break;
                default:
                    child.typeText = "Unknown";
                    break;
                }
            }
        } else {
            child.typeText = "User defined link";
        }
        node.children.push_back(std::move(child));
    }
}

Handle openFile(const std::string &fname, std::string &error) {
    Handle file(H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (!file.valid()) error = "cannot open " + fname + " as HDF5";
    return file;
}

// an open numeric dataset with its dimensions
struct Dataset {
    Handle file, did, type, space;
    std::vector<hsize_t> dims;
};

bool openDataset(const std::string &fname, const std::string &path, Dataset &ds, std::string &error) {
    ds.file = openFile(fname, error);
    if (!ds.file.valid()) return false;
    ds.did = Handle(H5Dopen2(ds.file, path.c_str(), H5P_DEFAULT), H5Dclose);
    if (!ds.did.valid()) {
        error = "cannot open dataset " + path;
        return false;
    }
    ds.type = Handle(H5Dget_type(ds.did), H5Tclose);
    ds.space = Handle(H5Dget_space(ds.did), H5Sclose);
    if (!ds.type.valid() || !ds.space.valid()) {
        error = "cannot read dataset " + path;
        return false;
    }
    ds.dims = getDims(ds.space);
    return true;
}

std::string lastName(const std::string &path) {
    size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

// names, origin, scale (and integer attributes as properties) saved with the image
void applyAttributes(hid_t did, physD &image) {
    for (auto &attr : readAttributes(did)) {
        if (attr.name == "physOrigin" && attr.numbers.size() == 2) {
            image.set_origin(attr.numbers[0], attr.numbers[1]);
        } else if (attr.name == "physScale" && attr.numbers.size() == 2) {
            image.set_scale(attr.numbers[0], attr.numbers[1]);
        } else if (attr.name == "physName" && attr.strings.size() == 1) {
            image.setName(attr.strings[0]);
        } else if (attr.name == "physShortName" && attr.strings.size() == 1) {
            image.setShortName(attr.strings[0]);
        } else if (attr.numbers.size() == 1 && attr.type.find("int") != std::string::npos) {
            image.prop[attr.name] = (int)attr.numbers[0];
        }
    }
}

void setNames(physD &image, const std::string &fname, const std::string &name) {
    image.setName(name);
    image.setShortName(lastName(name));
    image.setFromName(fname);
    image.setType(PHYS_FILE);
}

bool readFrameOpen(Dataset &ds, const std::string &fname, const std::string &path, hsize_t frame, physD &image, std::string &error) {
    if (frame >= ds.dims[0]) {
        error = "frame " + std::to_string(frame) + " out of range";
        return false;
    }
    Handle fspace(H5Dget_space(ds.did), H5Sclose);
    hsize_t start[3] = {frame, 0, 0};
    hsize_t count[3] = {1, ds.dims[1], ds.dims[2]};
    Handle mspace(H5Screate_simple(2, &ds.dims[1], nullptr), H5Sclose);
    if (H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, nullptr, count, nullptr) < 0) {
        error = "cannot select frame " + std::to_string(frame);
        return false;
    }
    image.resize(ds.dims[2], ds.dims[1]);
    if (H5Dread(ds.did, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, image.Timg_buffer) < 0) {
        error = "cannot read frame " + std::to_string(frame) + " of " + path;
        return false;
    }
    setNames(image, fname, path + "[" + std::to_string(frame) + "]");
    applyAttributes(ds.did, image);
    // the saved names are the ones of the dataset: keep the frame
    image.setShortName(lastName(path) + "[" + std::to_string(frame) + "]");
    return true;
}

bool openFrames(const std::string &fname, const std::string &path, Dataset &ds, std::string &error) {
    if (!openDataset(fname, path, ds, error)) return false;
    if (!isNumeric(ds.type) || ds.dims.size() != 3) {
        error = path + " is not a 3D dataset of numbers";
        return false;
    }
    return true;
}

}

std::string Attribute::text() const {
    std::string ret;
    for (auto &val : numbers) ret += (ret.empty() ? "" : " ") + number(val);
    for (auto &str : strings) ret += (ret.empty() ? "" : " ") + str;
    return ret;
}

std::string typeName(hid_t type) {
    size_t size = H5Tget_size(type);
    switch (H5Tget_class(type)) {
    case H5T_INTEGER:
        return std::string(H5Tget_sign(type) == H5T_SGN_NONE ? "uint" : "int") + std::to_string(8 * size);
    case H5T_FLOAT:
        return "float" + std::to_string(8 * size);
    case H5T_STRING:
        return H5Tis_variable_str(type) > 0 ? "string" : "string[" + std::to_string(size) + "]";
    case H5T_COMPOUND:
        return "compound";
    case H5T_ARRAY: {
        Handle super(H5Tget_super(type), H5Tclose);
        int rank = H5Tget_array_ndims(type);
        std::vector<hsize_t> adims(rank > 0 ? rank : 0);
        if (rank > 0) H5Tget_array_dims2(type, adims.data());
        return "array " + (super.valid() ? typeName(super) : std::string("?")) + "[" + joinDims(adims) + "]";
    }
    case H5T_ENUM:
        return "enum";
    case H5T_BITFIELD:
        return "bitfield";
    case H5T_OPAQUE:
        return "opaque";
    case H5T_REFERENCE:
        return "reference";
    case H5T_VLEN: {
        Handle super(H5Tget_super(type), H5Tclose);
        return "vlen " + (super.valid() ? typeName(super) : std::string("?"));
    }
    default:
        return "unknown";
    }
}

bool scanFile(const std::string &fname, Node &root, std::string &error) {
    QuietErrors quiet;
    Handle file = openFile(fname, error);
    if (!file.valid()) return false;
    Handle grp(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose);
    if (!grp.valid()) {
        error = "cannot open the root group of " + fname;
        return false;
    }
    root = Node();
    root.path = "/";
    root.name = fname;
    std::set<std::string> visited;
    visited.insert(objectKey(grp));
    scanGroup(grp, root, visited, 0);
    root.typeText = "File";
    return true;
}

bool readImage(const std::string &fname, const std::string &path, physD &image, std::string &error) {
    QuietErrors quiet;
    Dataset ds;
    if (!openDataset(fname, path, ds, error)) return false;
    if (isNumeric(ds.type) && ds.dims.size() == 2) {
        // HDF5 converts any integer/float type and byte order to double
        image.resize(ds.dims[1], ds.dims[0]);
        if (H5Dread(ds.did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, image.Timg_buffer) < 0) {
            error = "cannot read " + path;
            return false;
        }
    } else if (isImageArray(ds.type)) {
        hsize_t adims[2];
        H5Tget_array_dims2(ds.type, adims);
        Handle mem(H5Tarray_create2(H5T_NATIVE_DOUBLE, 2, adims), H5Tclose);
        hssize_t narray = H5Sget_simple_extent_npoints(ds.space);
        size_t surf = adims[0] * adims[1];
        if (narray <= 0 || surf == 0) {
            error = path + " is empty";
            return false;
        }
        std::vector<double> buffer(narray * surf);
        if (H5Dread(ds.did, mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data()) < 0) {
            error = "cannot read " + path;
            return false;
        }
        image.resize(adims[1], adims[0]);
        for (size_t k = 0; k < surf; k++) {
            double sum = 0;
            for (hssize_t na = 0; na < narray; na++) sum += buffer[k + na * surf];
            image.set(k, sum / narray);
        }
    } else {
        error = path + " is not a 2D dataset of numbers";
        return false;
    }
    setNames(image, fname, path);
    applyAttributes(ds.did, image);
    return true;
}

bool readFrame(const std::string &fname, const std::string &path, hsize_t frame, physD &image, std::string &error) {
    QuietErrors quiet;
    Dataset ds;
    if (!openFrames(fname, path, ds, error)) return false;
    return readFrameOpen(ds, fname, path, frame, image, error);
}

bool readFrames(const std::string &fname, const std::string &path, std::vector<physD> &images, std::string &error) {
    QuietErrors quiet;
    Dataset ds;
    if (!openFrames(fname, path, ds, error)) return false;
    images.clear();
    for (hsize_t frame = 0; frame < ds.dims[0]; frame++) {
        physD image;
        if (!readFrameOpen(ds, fname, path, frame, image, error)) return false;
        images.push_back(std::move(image));
    }
    return true;
}

bool readFramesMean(const std::string &fname, const std::string &path, physD &image, std::string &error) {
    QuietErrors quiet;
    Dataset ds;
    if (!openFrames(fname, path, ds, error)) return false;
    size_t surf = ds.dims[1] * ds.dims[2];
    std::vector<double> sum(surf, 0.0);
    std::vector<unsigned int> count(surf, 0);
    physD frameImage;
    for (hsize_t frame = 0; frame < ds.dims[0]; frame++) {
        if (!readFrameOpen(ds, fname, path, frame, frameImage, error)) return false;
        for (size_t k = 0; k < surf; k++) {
            double val = frameImage.point(k);
            if (std::isfinite(val)) {
                sum[k] += val;
                count[k]++;
            }
        }
    }
    image.resize(ds.dims[2], ds.dims[1]);
    for (size_t k = 0; k < surf; k++) {
        image.set(k, count[k] ? sum[k] / count[k] : std::numeric_limits<double>::quiet_NaN());
    }
    setNames(image, fname, path + "[mean]");
    applyAttributes(ds.did, image);
    image.setShortName(lastName(path) + "[mean]");
    return true;
}

bool readData1D(const std::string &fname, const std::string &path, std::vector<double> &data, std::string &error) {
    QuietErrors quiet;
    Dataset ds;
    if (!openDataset(fname, path, ds, error)) return false;
    if (!isNumeric(ds.type) || ds.dims.size() > 1) {
        error = path + " is not a 1D dataset of numbers";
        return false;
    }
    hssize_t n = H5Sget_simple_extent_npoints(ds.space);
    data.resize(n > 0 ? n : 0);
    if (n > 0 && H5Dread(ds.did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0) {
        error = "cannot read " + path;
        return false;
    }
    return true;
}

namespace {

bool writeDoubles(hid_t loc, const char *name, const std::vector<double> &vals) {
    hsize_t n = vals.size();
    Handle space(H5Screate_simple(1, &n, nullptr), H5Sclose);
    Handle attr(H5Acreate2(loc, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    return attr.valid() && H5Awrite(attr, H5T_NATIVE_DOUBLE, vals.data()) >= 0;
}

bool writeString(hid_t loc, const char *name, const std::string &str) {
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    H5Tset_size(type, std::max<size_t>(1, str.size()));
    H5Tset_strpad(type, H5T_STR_NULLPAD);
    H5Tset_cset(type, H5T_CSET_UTF8);
    Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    Handle attr(H5Acreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    std::string buf = str.empty() ? std::string(1, '\0') : str;
    return attr.valid() && H5Awrite(attr, type, buf.data()) >= 0;
}

}

bool writeImage(const physD &cimage, const std::string &fname, const std::string &version, std::string &error) {
    QuietErrors quiet;
    physD &image = const_cast<physD &>(cimage); // the accessors are not const
    if (image.getSurf() == 0 || image.Timg_buffer == nullptr) {
        error = "empty image";
        return false;
    }
    Handle file(H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    if (!file.valid()) {
        error = "cannot create " + fname;
        return false;
    }
    hsize_t dims[2] = {image.getH(), image.getW()};
    Handle space(H5Screate_simple(2, dims, nullptr), H5Sclose);
    Handle dcpl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
    // 256x256 chunks (were 4x8: very slow and bigger), deflate only if this HDF5 can write it
    hsize_t chunk[2] = {std::min<hsize_t>(dims[0], 256), std::min<hsize_t>(dims[1], 256)};
    H5Pset_chunk(dcpl, 2, chunk);
    unsigned int filter_info = 0;
    if (H5Zfilter_avail(H5Z_FILTER_DEFLATE) > 0 && H5Zget_filter_info(H5Z_FILTER_DEFLATE, &filter_info) >= 0 &&
        (filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED)) {
        H5Pset_shuffle(dcpl);
        H5Pset_deflate(dcpl, 4);
    }
    Handle dset(H5Dcreate2(file, "neutrino", H5T_IEEE_F64LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT), H5Dclose);
    if (!dset.valid() || H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, image.Timg_buffer) < 0) {
        error = "cannot write the image in " + fname;
        return false;
    }
    bool ok = writeDoubles(dset, "physOrigin", {image.get_origin().x(), image.get_origin().y()}) &&
              writeDoubles(dset, "physScale", {image.get_scale().x(), image.get_scale().y()}) &&
              writeString(dset, "physName", image.getName()) &&
              writeString(dset, "physShortName", image.getShortName()) &&
              writeString(file, "version", version);
    if (!ok) {
        error = "cannot write the attributes in " + fname;
        return false;
    }
    dset.reset();
    if (H5Fflush(file, H5F_SCOPE_GLOBAL) < 0) {
        error = "cannot write " + fname;
        return false;
    }
    return true;
}

}
