import numpy as np


def my_unfold(tensor, mode):
    '''
    Função que executa a operação de unfolding de um tensor de 3ª Ordem.

    :param tensor: matriz i x k
    :param mode: modo desejado (1, 2 ou 3)
    :return : matriz do modo {(i x jk), (j, ik), (k, ij)}
    '''
    t = tensor.shape
    if mode is 1: # i, jk
        return np.transpose(tensor, (1,0,2)).reshape(t[mode], int(np.prod(t)/t[mode]))
    elif mode is 2: # j, ik
        return np.transpose(tensor, (2,0,1)).reshape(t[mode], int(np.prod(t)/t[mode]))
    elif mode is 3:# k, ij
        # shape 3 ordem = (i0, i1, i2)
        return np.transpose(tensor, (0,2,1)).reshape(t[0], int(np.prod(t)/t[0]))
    else:
        print("Modo não suportado")
        return None

def my_kr(mt_a, mt_b):
    '''
    Função que executa o produto de Khatri-Rao entre A e b.

    :param mt_a: matriz i x k
    :param mt_b: matriz j x k
    :return : matriz ij x k
    '''
    if mt_a.shape[1] is mt_b.shape[1]:
        ncol = mt_a.shape[1]
        return np.einsum('az,bz->abz', mt_a, mt_b).reshape((-1, ncol))
    else:
        print('Erro: As matrizes precisam ter o mesmo número de colunas!\n')
        return None
