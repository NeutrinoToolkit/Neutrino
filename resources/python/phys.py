import gzip
import numpy as np
import struct 
import sys
import re

class phys(np.ndarray):
    """numpy acess to neu and neus files"""
    
    def __repr__(self):
        ret='Properties:\n'
        pad=30
        suf='[...]'
        for k,v in self.prop.items() :
            if len(k)>pad-len(suf): k=k[0:pad-len(suf)]+suf
            sv=str(v)
            if len(sv)>pad-len(suf): sv=sv[0:pad-len(suf)]+suf
            ret += k.ljust(pad, ' ') + " : " + sv.rjust(pad, ' ') +"\n"

        return ret+'\nData:\n'+str(self)+'\n'
        
    def __new__(cls, input_array, prop=None):
        obj = np.asarray(input_array).view(cls)
        obj.prop = prop
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.prop = getattr(obj, 'prop', {})
        
    def plot(self, **kwargs):
        try:
            from matplotlib import pyplot as plt                      
        except ImportError:
            print("Need matplotlib to plot!")
        else:        
            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(111)
            plt.imshow(self)
            ax.set_aspect('equal')
            plt.colorbar()
            plt.show()
    
    def dumpToStream(self, fp):
        fp.write(b"@@ phys_properties\n")
        for k,v in self.prop.items():
            mystr= (k + " = " + str(v) + "\n").encode()
            fp.write(mystr)
        fp.write(b"@@ end\n")            
        fp.write(struct.pack('I', self.shape[1]))
        fp.write(struct.pack('I', self.shape[0]))
        data=self.astype('float64')
        data_comp=gzip.compress(data,compresslevel=8)
        fp.write(struct.pack('I', int(len(data_comp))))
        fp.write(data_comp)
        fp.write(b"\n")            

    def dump(self, fname):
        with open(fname,"wb") as fp:
            self.dumpToStream(fp)

    @staticmethod
    def writeNeus(mylist,fname):
        with open(fname,"wb") as fp:
            head="Neutrino pyWrapper "+str(len(mylist))+" 0"
            fp.write(head.encode())
            fp.write(b'\n')
            for myphys in mylist:
                fp.write(b'NeutrinoImage\n')
                myphys.dumpToStream(fp)

    @staticmethod
    def parseAnyData(val: str):
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                try:
                    myprop[key] = int(val)
                except ValueError:
                    if val.count(":")==1 and val[0]=='(' and val[-1]==")":
                        m = re.match(r'\((.*):(.*)\)', val) 
                        if m and len(m.groups()) == 2 :
                            try:
                                 return tuple(map(int, [m[1],m[2]]))
                            except ValueError:
                                try:
                                     return tuple(map(double, [m[1],m[2]]))
                                except ValueError:
                                     return tuple(map(str, [m[1],m[2]]))

        return val

        
    @staticmethod
    def readNeus(fname, byteorder=sys.byteorder):
        ret=[]
        myprop={}
        inside_prop = False
        with open(fname,"rb") as fp:
            for line in fp:
                if line == b"@@ phys_properties\n" :
                    inside_prop = True
                elif line == b"@@ end\n" :
                    w,h,s =[int.from_bytes(fp.read(4), byteorder=byteorder) for _ in range(3)]
                    e = np.frombuffer(gzip.decompress(fp.read(s)), dtype=np.float, count=w*h).reshape((h,w))
                    ret.append(phys(e, prop=myprop))
                    inside_prop = False
                    myprop={}
                else:
                    if inside_prop :
                        pos = line.find(b" = ")
                        key=line[0:pos].decode()
                        val=line[pos+3:-1].decode()
                        myprop[key] = phys.parseAnyData(val)
                
        return ret
      