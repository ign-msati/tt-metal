import sys
def dump_np_array(arr, filename=None, isHex=False):
    from itertools import product

    if isinstance(filename, str):
        f = open(filename, "wt")

    else:
        f = sys.stdout

    print("shape ", arr.shape, file=f)

    idx_ranges = [range(dim) for dim in arr.shape]

    for idx_tup in product(*idx_ranges):
        if (idx_tup[-1] % 8) == 0:
            len_idx_tup = len(idx_tup)

            for i in range(len_idx_tup):
                if i == 0:
                    print("\n", file=f, end="")

                if i == (len_idx_tup - 1):
                    print(" %3d|" % (idx_tup[i]), file=f, end="")

                else:
                    print(" %2d" % (idx_tup[i]), file=f, end="")

        if isHex:
            print(" %s" % (hex(arr[idx_tup])), file=f, end="")

        else:
            print(" %3f" % (arr[idx_tup].item()), file=f, end="")

    if isinstance(filename, str):
        f.close()