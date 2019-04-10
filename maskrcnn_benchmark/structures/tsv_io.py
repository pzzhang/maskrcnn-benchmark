import logging
import os
import os.path as op


def generate_lineidx(filein, idxout):
    if op.isfile(idxout):
        print("overwrite lineidx file: {}".format(idxout))
    with open(filein,'r') as tsvin, open(idxout,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()


class TSVFile(object):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx' 
        self._fp = None
        self._lineidx = None
        self._ensure_lineidx_loaded()
    
    def num_rows(self):
        return len(self._lineidx) 

    def seek(self, idx):
        self._ensure_tsv_opened()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
                generate_lineidx(self.tsv_file, self.lineidx)
            # guard the case when lineidx file is outdated
            # in tsv_writer, lineidx file is saved after tsv file.
            # so if the timestamp is not larger than tsv file, 
            # it is likely that the lineidx is outdated, so update it.
            if op.getmtime(self.lineidx) < op.getmtime(self.tsv_file):
                generate_lineidx(self.tsv_file, self.lineidx)
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')

