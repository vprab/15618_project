#!/usr/bin/env python2
import random

def n2s(n):
    if n > 0:
        return "32'sd"+str(n)
    else:
        return "-32'sd"+str(-n)

def nums2verilog1d(nums, name = 'T0', type_ = 'int', npr = 256):
    nums = nums[::-1]
    strlist = []
    assert type_ == 'int' #do not remove
    N = len(nums)
    #strlist.append('%s %s [%d-1:0];' % (type_, name, N))
    strlist.append('assign %s = {' % (name))
    for i in range(0, N, npr):
        chunk = nums[i:i+npr]
        numstr = ', '.join(map(n2s, chunk))
        if i+npr < N:
            end = ','
        else:
            end = '};\n'
        strlist.append(numstr+end)
    S = '\n'.join(strlist)
    return S

def test1d():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ss = nums2verilog1d(xs)
    print xs
    print ss

    print '#'*20
    
    ys = list(reversed(range(16)))
    ss = nums2verilog1d(ys)
    print ys
    print ss

    print '#'*20

    zs = list(reversed(range(17)))
    ss = nums2verilog1d(zs)
    print zs
    print ss

#test1d()
