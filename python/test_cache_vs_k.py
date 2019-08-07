import lsgdual_impl as lsgdual
import time
# import gdual_impl as gdual
import matplotlib.pyplot as plt
import numpy as np

N = 5;
# q_vals = [10,20,40,60,80,100,125,150,175,200,250,300,400,600,800,1000]
q = 400;
k_ratios = np.concatenate((np.arange(0.01, 0.1, 0.01), np.linspace(0.1, 1.0, 19)))
PLT_TITLE = 'compose vs k (caching q = %d)' % q

rt_A          = np.zeros(len(k_ratios))
rt_lsgdcacheA = np.zeros(len(k_ratios))
# rt_lsgdcacheB = np.zeros(len(q_vals))

for i_k in range(len(k_ratios)):
    k = int(np.ceil(k_ratios[i_k] * q))
    # s = lsgdual.lsgdual_xdx(0.5, 10)
    F = lsgdual.exp(lsgdual.lsgdual_xdx(7, q))
    G = lsgdual.exp(lsgdual.lsgdual_xdx(5, q))

    start = time.time()
    for i in range(N):
        A = lsgdual._compose_brent_kung_A(F, k)
    end = time.time()
    rt_A[i_k] = (end - start) / N
    # B = lsgdual._compose_brent_kung_B(G)

    start = time.time()
    for i in range(N):
        H = lsgdual.compose_brent_kung(G, F, A = A, k = k)
    end = time.time()
    # print("lsgd:std %f" % (end - start))
    # rt_lsgdstd.append(end - start)
    rt_lsgdcacheA[i_k] = (end - start) / N

    # start = time.time()
    # for i in range(N):
    #     H = lsgdual.compose_brent_kung(G, F, B = B)
    # end = time.time()
    # # print("lsgd:std %f" % (end - start))
    # # rt_lsgdstd.append(end - start)
    # rt_lsgdcacheB[i_q] = (end - start) / N

    # F = lsgdual.lsgd2gd(F)
    # G = lsgdual.lsgd2gd(G)
    #
    # start = time.time()
    # for i in range(1,N):
    #     H = gdual.compose_brent_kung(F, G)
    # end = time.time()
    # # print("gd:bk    %f" % (end - start))
    # # rt_gdbk.append(end - start)
    # rt_gdbk[i_q] = end - start

    # start = time.time()
    # for i in range(1,N):
    #     H = gdual.compose(F, G)
    # end = time.time()
    # print("gd:std   %f" % (end - start))
    # rt_gdstd.append(end - start)
    # rt_gdstd[i_q] = end - start

# plt.plot(q_vals, rt_lsgd, q_vals, rt_lsgdcacheA, q_vals, rt_lsgdcacheB)
# plt.legend(('regular', 'cache A', 'cache B'))
plt.plot(k_ratios * q, rt_A, k_ratios * q, rt_lsgdcacheA, k_ratios * q, rt_A + rt_lsgdcacheA)
plt.axvline(x=int(np.ceil(np.sqrt(q))))
plt.legend(('precompute', 'compose', 'total'))
plt.ylabel('rt (s)')
plt.xlabel('k')
plt.title(PLT_TITLE)
plt.savefig(PLT_TITLE + '.png')
plt.show()