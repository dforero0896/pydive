import os

from helpers import c_ascii_writer_single, c_ascii_writer_double
import numpy as np

def c_void_npy_to_text(npy_filename, text_filename):

    voids = np.load(npy_filename)
    print(f"==> Writing contents of {npy_filename}  to {text_filename}", flush=True)
    if voids.dtype == np.double:
        c_ascii_writer_double(voids, voids.shape[0], text_filename+".TMP")
    elif voids.dtype == np.float32:
        c_ascii_writer_single(voids, voids.shape[0], text_filename+".TMP")
    os.rename(text_filename+".TMP", text_filename)
    print(f"\t Done.", flush=True)

def npy_to_text(npy_filename, text_filename):
    voids = np.load(npy_filename)
    print(f"==> Writing contents of {npy_filename}  to {text_filename}", flush=True)
    np.savetxt(text_filename+".TMP", voids, fmt='%.4f')
    os.rename(text_filename+".TMP", text_filename)
    print(f"\t Done.", flush=True)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-npy_file", type=str)
    parser.add_argument("-text_file", type=str)
    parser.add_argument("-ncols", type=int, default=4)
    args = parser.parse_args()
    if args.ncols == 4:
        c_void_npy_to_text(args.npy_file, args.text_file)
    else:
        npy_to_text(args.npy_file, args.text_file)



    
