# -*- coding: utf-8 -*-
"""
Created on %(date)s


@author: %Ilian
"""

def somme_liste_4 (L1,L2,L3,L4)    :
    L=[]
    for i in range (len(L4)) : 
        Lt=[]
        
        for k in range (len(L4[i])):
            A=L1[i]
            B=L2[i]
            C=L3[i]
            D=L4[i]
            Lt.append( A[k]+B[k]+C[k]+D[k] )
        L.append(Lt)
    return (L)

def somme_liste_3 (L1,L2,L3)    :
    L=[]
    for i in range (len(L1)) : 
        L.append( L1[i]+L2[i]+L3[i] )
    return (L)